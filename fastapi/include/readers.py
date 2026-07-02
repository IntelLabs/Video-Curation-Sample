import ctypes
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import Process, Value
from multiprocessing.shared_memory import SharedMemory

import av
import cv2
import numpy as np
import torch
from ultralytics.utils.checks import check_imgsz

# ----- SETUP LOGGING -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Suppress low-delay reference block warnings from OpenCV/PyAV/FFmpeg
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
logging.getLogger("libav").setLevel(logging.CRITICAL)
logging.getLogger("libav.hevc").setLevel(logging.CRITICAL)
av.logging.set_level(av.logging.PANIC)

main_app_logger = logging.getLogger(__name__)

# ----- PIPELINE CONFIGURATION -----
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
cv2.setNumThreads(0)

from include.default_configs import ENABLE_QUERYING_DEFAULT
from include.utils import PipelineConfig, manual_fps_calculation, str2bool

BASE_PIPELINE_CONFIG = PipelineConfig(
    CODE_DIR=os.getenv("CODE_DIR", "/home"),
    CUSTOM_MODEL_FLAG=str2bool(os.getenv("CUSTOM_MODEL_FLAG", False)),
    DBHOST=os.getenv("DBHOST", "vdms-service"),
    DEBUG=os.getenv("DEBUG", "0"),
    DEVICE=os.getenv("DEVICE", "CPU"),
    ENABLE_QUERYING=os.getenv("ENABLE_QUERYING", ENABLE_QUERYING_DEFAULT),
    INGESTION=os.getenv("INGESTION", "object"),
    MODEL_NAME=os.getenv("MODEL_NAME", "yolo11n"),
    OMIT_DETECTIONS_FLAG=str2bool(os.getenv("OMIT_DETECTIONS_FLAG", False)),
    SHARED_MODEL=os.getenv("SHARED_MODEL", False),
    SHARED_OUTPUT=os.getenv("SHARED_OUTPUT", "/var/www/mp4"),
    TEST_MODE=str2bool(os.getenv("TEST_MODE", False)),
    TMP_LOCATION=os.getenv("TMP_LOCATION", "/var/www/cache"),
    UDF_HOST=os.getenv("UDF_HOST", "udf-service"),
    UDF_PORT=5011,
)

# def capture_shared_memory_worker(source_input, shm_name, frame_shape, running_flag):
#     """Isolated background process bypassing the GIL to update raw uncompressed RAM."""
#     if str(source_input).lower().startswith("rtsp://"):
#         os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;buffer_size;15728640;threads;16"
#     elif "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
#         del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

#     cap = cv2.VideoCapture(source_input, cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         running_flag.value = False
#         return

#     existing_shm = SharedMemory(name=shm_name)
#     shared_array = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_shm.buf)

#     try:
#         while running_flag.value:
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 running_flag.value = False
#                 break
#             shared_array[:] = frame[:]
#     except Exception:
#         running_flag.value = False
#     finally:
#         cap.release()
#         existing_shm.close()

import sys

# Self-contained worker script executed in an isolated process shell to prevent global CUDA import pollution
# WORKER_SCRIPT = """
# import os
# import sys
# import cv2
# import numpy as np
# from multiprocessing.shared_memory import SharedMemory
# source_input = sys.argv[1]
# shm_name = sys.argv[2]
# h, w, c = map(int, sys.argv[3].split(','))
# frame_shape = (h, w, c)
# retry_cnt = 0
# max_retries = 5
# cap = None
# if str(source_input).lower().startswith("rtsp://"):
#     os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;buffer_size;15728640;threads;16"
# while retry_cnt < max_retries:
#     cap = cv2.VideoCapture(source_input, cv2.CAP_FFMPEG)
#     if cap.isOpened():
#         break
#     retry_cnt += 1
#     # Exit immediately if it's a local file path layout to save CPU cycles
#     if not str(source_input).lower().startswith("rtsp://"):
#         break
#     wait_time = retry_cnt * 2
#     # Standard string stdout message to pass back to the tracking shell logs
#     print(f"RTSP background channel pending... Retry ({retry_cnt}/{max_retries}) in {wait_time}s.")
#     import time
#     time.sleep(wait_time)
# if cap is None or not cap.isOpened():
#     import sys
#     sys.exit(1)
# try:
#     shm = SharedMemory(name=shm_name)
#     shared_array = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
#     # Read continuously until stdin is closed by the parent process
#     while sys.stdin.read(1) == '1':
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             break
#         shared_array[:] = frame[:]
# except Exception:
#     pass
# finally:
#     cap.release()
# """
from multiprocessing import Condition


def capture_shared_memory_worker(
    source_input,
    shm_names,
    frame_shape,
    running_flag,
    raw_frame_counter,
    latest_idx,
    reader_idx,
    frame_condition,
    buffer_occupancy,
    target_fps,
):
    """Isolated background process using triple-buffering pointer rotation to eliminate memcpy."""
    if str(source_input).lower().startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp;buffer_size;33554432;threads;32;max_delay;500000"
        )
    # "rtsp_transport;tcp;buffer_size;52428800;fifo_size;500000;max_delay;100000;stimeout;2000000"
    # "rtsp_transport;tcp;buffer_size;15728640;threads;16"
    elif "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
        del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

    retry_cnt = 0
    max_retries = 5
    cap = None
    is_rtsp_stream = str(source_input).lower().startswith("rtsp://")

    while retry_cnt < max_retries:
        cap = cv2.VideoCapture(
            source_input,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
        )
        if cap.isOpened():
            break

        retry_cnt += 1
        if not is_rtsp_stream or retry_cnt >= max_retries:
            main_app_logger.error(
                f"Critical: Could not open/connect to video resource: {source_input}"
            )
            running_flag.value = False
            return

        wait_time = retry_cnt * 2
        main_app_logger.warning(
            f"RTSP process connection pending... Retry ({retry_cnt}/{max_retries}) in {wait_time} seconds."
        )
        time.sleep(wait_time)

    # existing_shm = SharedMemory(name=shm_name)
    # shared_array = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_shm.buf)
    # Map all 3 shared memory regions simultaneously
    shm_blocks = [SharedMemory(name=name) for name in shm_names]
    # from multiprocessing import resource_tracker
    # for shm in shm_blocks:
    #     resource_tracker.unregister(shm._name, "shared_memory")
    arrays = [
        np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf) for shm in shm_blocks
    ]

    write_idx = 0
    worker_frame_num = 0.0
    worker_next_process_idx = 0.0
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    step_size = float(native_fps) / float(target_fps)

    try:
        while running_flag.value:
            # 1. Check if this frame is a targeted frame before running heavy decode operations
            is_target_frame = worker_frame_num >= worker_next_process_idx
            worker_frame_num += 1.0

            if not is_target_frame:
                # Fast metadata-only skip: Zero heavy decode compute or CPU/GPU overhead
                if not cap.grab():
                    running_flag.value = False
                    break
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                running_flag.value = False
                break

            worker_next_process_idx += step_size

            # Prevent local files from overwriting unread ring buffer slots ---
            if not str(source_input).lower().startswith("rtsp://"):
                while buffer_occupancy.value >= 2 and running_flag.value:
                    pass

            # Dynamic Ring Buffer Selection: Identify free block
            curr_latest = latest_idx.value
            curr_reader = reader_idx.value
            for idx in (0, 1, 2):
                if idx != curr_latest and idx != curr_reader:
                    write_idx = idx
                    break

            arrays[write_idx][:] = frame[:]

            # Notify waiting processing loops without any polling latency
            with frame_condition:
                latest_idx.value = write_idx
                raw_frame_counter.value += 1
                buffer_occupancy.value += 1
                frame_condition.notify_all()  # Wake up all waiting reader threads instantly!

    except Exception:
        running_flag.value = False
    finally:
        cap.release()
        for shm in shm_blocks:
            shm.close()


class BaseReader:
    def __init__(
        self,
        source,
        target_fps=BASE_PIPELINE_CONFIG.TARGET_FPS,
        clip_duration=BASE_PIPELINE_CONFIG.CLIP_DURATION,
        MODEL_W=BASE_PIPELINE_CONFIG.MODEL_W,
        MODEL_H=BASE_PIPELINE_CONFIG.MODEL_H,
        queue_size=2,
    ):
        self.source = source
        self.is_rtsp = str(self.source).startswith("rtsp://")
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.total_input_frames = 0  # Increments by step size (Physical frames)
        self.target_frames_passed = 0  # Increments sequentially (Target space frames)
        # self.frame_idx = 0.0
        self.total_shm_copy_time = 0.0
        self.total_h2d_time = 0.0
        self.total_gpu_resize_time = 0.0
        self.total_d2h_time = 0.0
        self.total_queue_wait_time = 0.0
        self.MODEL_H = MODEL_H
        self.MODEL_W = MODEL_W

        self.target_fps = (
            float(target_fps) if target_fps not in [None, 0] else target_fps
        )
        self.clip_duration = (
            float(clip_duration) if clip_duration not in [None, 0] else clip_duration
        )

        probe_cap = None
        max_retries = 5
        retry_cnt = 0
        connected = False

        # while probe_retry < max_retries:
        #     probe_cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        #     if probe_cap.isOpened():
        #         break
        #     probe_retry += 1
        #     if not self.is_rtsp:
        #         break
        #     wait_time = probe_retry * 2
        #     main_app_logger.warning(
        #         f"Connection pending... Retry ({probe_retry}/{max_retries}) in {wait_time} seconds."
        #     )
        #     time.sleep(wait_time)

        while not connected and not self.stopped:
            try:
                probe_cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if probe_cap.isOpened():
                    self.get_fps_and_framecnt(
                        probe_cap, self.target_fps, self.clip_duration
                    )
                    self.frame_width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.frame_height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.numFrames = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    if self.input_fps <= 0 or self.frame_width <= 0:
                        raise RuntimeError(
                            "VideoCapture opened but returned invalid stream properties."
                        )
                    self.get_frameWH()
                    connected = True
                    probe_cap.release()
                else:
                    raise RuntimeError(
                        "OpenCV VideoCapture failed to open target URI resource context."
                    )
            except Exception as e:
                retry_cnt += 1
                self.init_error = str(e)
                # Exit if local file resource or retry count exceeded
                if not self.is_rtsp or retry_cnt >= max_retries:
                    main_app_logger.error(
                        f"Critical: Could not open/connect to {self.source}"
                    )
                    self.reconnect_failed = True
                    self.stopped = True
                    # Halt and exit the server immediately
                    raise RuntimeError(
                        f"Critical stream reader initialization failure: {self.init_error}"
                    )
                    return

                wait_time = retry_cnt * 2
                main_app_logger.warning(
                    f"Connection pending... Retry ({retry_cnt}/{max_retries}) in {wait_time} seconds."
                )
                time.sleep(wait_time)

        # if probe_cap is None or not probe_cap.isOpened():
        #     self.reconnect_failed = True
        #     self.stopped = True
        #     raise RuntimeError(f"OpenCV failed to open target resource: {self.source}")

        # self.get_fps_and_framecnt(probe_cap, self.target_fps, self.clip_duration)
        # self.frame_width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.frame_height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.numFrames = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.get_frameWH()
        # probe_cap.release()

        self.frame_shape = (self.frame_height, self.frame_width, 3)
        self.frame_bytes = self.frame_width * self.frame_height * 3

        num_shm_slots = 3
        self.shms = [
            SharedMemory(create=True, size=self.frame_bytes)
            for _ in range(num_shm_slots)
        ]

        # Unregister slots in the main process to eliminate leak warnings and KeyErrors cleanly
        # from multiprocessing import resource_tracker
        # for shm in self.shms:
        #     resource_tracker.unregister(shm._name, "shared_memory")

        self.running_flag = Value("b", True)
        self.raw_frame_counter = Value("i", 0)
        self.latest_idx = Value("i", 0)  # Tracks the newest complete frame index
        self.reader_idx = Value("i", -1)  # Locks the frame currently being processed
        shm_names = [shm.name for shm in self.shms]
        # atomic counter to track unread slot density ---
        self.buffer_occupancy = Value("i", 0)

        # Create a shared cross-process condition lock variable
        self.frame_condition = Condition()

        self.worker = Process(
            target=capture_shared_memory_worker,
            args=(
                self.source,
                shm_names,
                self.frame_shape,
                self.running_flag,
                self.raw_frame_counter,
                self.latest_idx,
                self.reader_idx,
                self.frame_condition,
                self.buffer_occupancy,
                self.target_fps,
            ),
            daemon=True,
        )
        self.worker.start()
        time.sleep(2.5 if self.is_rtsp else 0.05)

        if not self.running_flag.value:
            for shm in self.shms:
                shm.close()
                shm.unlink()
            raise RuntimeError(
                "Background worker process failed to map the stream link."
            )

        # self.shared_array_view = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.shms.buf)
        # self.static_buffer_numpy = np.empty(self.frame_shape, dtype=np.uint8)

        # self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        # self.thread.start()

    def read(self):
        try:
            wait_time = 0.1 if self.stopped else 2.0
            return self.frame_queue.get(timeout=wait_time)
        except Exception:
            return None, None, None

    def start(self):
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stopped = True
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        for shm in self.shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

        if hasattr(self, "bridge_thread"):
            self.bridge_thread.join(timeout=1.0)

        if hasattr(self, "worker"):
            self.worker.join(timeout=1.0)

    # Gets video details
    def get_fps_and_framecnt(self, cap, target_fps, clip_duration):
        self.input_fps = cap.get(cv2.CAP_PROP_FPS)  # hardware fps
        # print(f"in fps: {self.input_fps} target fps: {target_fps}")
        if self.input_fps == 0:  # Case when FPS isn't available
            self.input_fps = manual_fps_calculation(cap, num_frames=10)
            print(f"new in fps: {self.input_fps}")
        self.numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.is_rtsp:
            self.total_input_frames = self.numFrames
        # If the stream can't connect, stop immediately instead of calculating.
        if self.input_fps <= 0:
            raise RuntimeError(
                f"Failed to initialize stream reader endpoint: {self.source}"
            )

        self.target_fps = (
            target_fps
            if target_fps not in [None, 0] and self.input_fps > target_fps
            else self.input_fps
        )

        if self.input_fps > 0 and self.target_fps > 0:
            self.frame_skip = max(1, int(self.input_fps / self.target_fps))
        else:
            self.frame_skip = 1
        # self.skip_count = self.frame_skip - 1

        if clip_duration is None:
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            clip_duration = frame_count / self.input_fps
        self.max_frames_per_clip = int(self.target_fps * float(clip_duration))
        self.frame_interval = 1.0 / self.target_fps  # 0.0666s
        # print(
        #     f"in fps: {self.input_fps} self.target fps: {self.target_fps} self.frame_skip: {self.frame_skip}"
        # )

    # Gets frame W and H details
    def get_frameWH(self):
        if (self.frame_height * self.frame_width) < (self.MODEL_H * self.MODEL_W):
            new_sizeHW = check_imgsz([self.MODEL_H, self.MODEL_W])  # expects hxw
        else:
            new_sizeHW = check_imgsz(
                [self.frame_height, self.frame_width]
            )  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]
        self.height = new_sizeWH[1]

        # Configure scaling for 8K-to-Model coordinate mapping
        self.resize_h, self.resize_w = [self.MODEL_H, self.MODEL_W]
        self.scale_x = self.frame_width / self.MODEL_W
        self.scale_y = self.frame_height / self.MODEL_H


class CPUReader(BaseReader):
    """Asynchronous CPU frame reader and processor utilizing AVX2 optimized OpenCV routines."""

    def __init__(
        self,
        source,
        target_fps=BASE_PIPELINE_CONFIG.TARGET_FPS,
        clip_duration=BASE_PIPELINE_CONFIG.CLIP_DURATION,
        MODEL_W=BASE_PIPELINE_CONFIG.MODEL_W,
        MODEL_H=BASE_PIPELINE_CONFIG.MODEL_H,
        queue_size=2,
    ):
        # self.source_input = source_input
        # self.frame_queue = queue.Queue(maxsize=30)
        # self.stopped = False
        # self.total_shm_copy_time = 0.0
        # self.total_h2d_time = 0.0
        # self.total_gpu_resize_time = 0.0
        # self.total_d2h_time = 0.0
        super().__init__(
            source,
            target_fps=target_fps,
            clip_duration=clip_duration,
            MODEL_W=MODEL_W,
            MODEL_H=MODEL_H,
            queue_size=queue_size,
        )

        # self.MODEL_H = MODEL_H
        # self.MODEL_W = MODEL_W
        self.device = "CPU"

        # self.shm = SharedMemory(create=True, size=self.frame_bytes)
        # self.running_flag = Value('b', True)

        # self.worker = Process(
        #     target=capture_shared_memory_worker,
        #     args=(self.source_input, self.shm.name, self.frame_shape, self.running_flag),
        #     daemon=True
        # )
        # # self.worker.start()
        # time.sleep(2.5)  # Warm up wait window

        # if not self.running_flag.value:
        #     self.shm.close()
        #     self.shm.unlink()
        #     raise RuntimeError("Background worker process failed to map the stream link.")

        # self.shared_array_view = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.static_buffer_numpy = np.empty(self.frame_shape, dtype=np.uint8)

        # self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        # self.thread.start()

    def _processing_loop(self):
        # target_fps = 15.0
        # frame_num = 0.0
        # next_process_idx = 0.0
        # step_size = float(self.input_fps) / float(self.target_fps)
        # # self.total_queue_wait_time = 0.0

        # # Base loop pacing on the native camera interval (e.g., 33.3ms for 30 FPS)
        # inbound_frame_interval = 1.0 / float(self.input_fps)
        last_processed_shm_idx = -1
        while not self.stopped:  # and self.running_flag.value:
            # loop_start = time.perf_counter()
            # is_target_frame = (frame_num >= next_process_idx)

            # frame_num += 1.0

            # if not is_target_frame:
            #     continue

            with self.frame_condition:
                # Thread sleeps instantly until the background worker calls notify_all()
                # self.frame_condition.wait(timeout=1.0)
                # Only wait if the background worker hasn't delivered a new index yet
                while (
                    self.latest_idx.value == last_processed_shm_idx and not self.stopped
                ):
                    if not self.frame_condition.wait(timeout=0.05):
                        if (
                            not self.running_flag.value
                            and self.latest_idx.value == last_processed_shm_idx
                        ):
                            self.stopped = True
                            break
                        continue
                if self.stopped:
                    break

                active_idx = self.latest_idx.value
                self.reader_idx.value = active_idx
                last_processed_shm_idx = active_idx

                # Signal the worker that a slot has cleared up losslessly ---
                self.buffer_occupancy.value = max(0, self.buffer_occupancy.value - 1)

                # is_target_frame = (frame_num >= next_process_idx)
                # Reconstruct source frame position using step spacing ratio
                # if self.numFrames != 0:
                # self.total_input_frames = int(
                #     self.target_frames_passed * getattr(self, "frame_skip", 1)
                # )
                if self.is_rtsp:
                    self.total_input_frames = int(self.raw_frame_counter.value)

                # Update counters inside target_fps space loop execution
                self.target_frames_passed += 1

            # if not is_target_frame:
            #     continue

            # # Check if frame is a target frame before heavy operations
            # # if frame_num >= next_process_idx:
            # next_process_idx += step_size

            t_copy = time.perf_counter()
            # np.copyto(self.static_buffer_numpy, self.shared_array_view)
            # Zero-Copy Pointer Swap: lock the latest completed buffer index
            # active_idx = self.latest_idx.value
            # self.reader_idx.value = active_idx
            # last_processed_shm_idx = active_idx

            # # Signal the worker that a slot has cleared up losslessly ---
            # self.buffer_occupancy.value = max(0, self.buffer_occupancy.value - 1)

            # Wrap an array view directly around the locked active SHM region
            active_shm = self.shms[active_idx]
            frame_view = np.ndarray(
                self.frame_shape, dtype=np.uint8, buffer=active_shm.buf
            )
            self.total_shm_copy_time += time.perf_counter() - t_copy

            if not self.is_rtsp or not self.frame_queue.full():
                t_queue_block = time.perf_counter()
                # self.frame_queue.put((True, cpu_360p_frame))
                self.frame_queue.put(
                    (True, frame_view, self.target_frames_passed)
                )  # self.static_buffer_numpy.copy()))
                self.total_queue_wait_time += time.perf_counter() - t_queue_block

            # frame_num += 1.0

            # Keep the reader thread aligned with target ingestion speeds
            # elapsed = time.perf_counter() - loop_start
            # time_to_wait = inbound_frame_interval - elapsed
            # if time_to_wait > 0:
            #     time.sleep(time_to_wait)

        # self.frame_idx = frame_num
        # self.running_flag.value = False


class GPUReader(BaseReader):
    """Asynchronous GPU frame reader leveraging Pinned Host Memory and PyTorch CUDA tensors."""

    def __init__(
        self,
        source,
        gpu_id=1,  # 0,
        target_fps=BASE_PIPELINE_CONFIG.TARGET_FPS,
        clip_duration=BASE_PIPELINE_CONFIG.CLIP_DURATION,
        MODEL_W=BASE_PIPELINE_CONFIG.MODEL_W,
        MODEL_H=BASE_PIPELINE_CONFIG.MODEL_H,
        queue_size=2,
    ):
        # self.source_input = source_input
        # self.frame_queue = queue.Queue(maxsize=30)
        # self.stopped = False
        # self.total_shm_copy_time = 0.0
        # self.total_h2d_time = 0.0
        # self.total_gpu_resize_time = 0.0
        # self.total_d2h_time = 0.0
        super().__init__(
            source,
            target_fps=target_fps,
            clip_duration=clip_duration,
            MODEL_W=MODEL_W,
            MODEL_H=MODEL_H,
            queue_size=queue_size,
        )

        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")

        # self.shm = SharedMemory(create=True, size=self.frame_bytes)
        # self.running_flag = Value('b', True)

        # self.worker = Process(
        #     target=capture_shared_memory_worker,
        #     args=(self.source_input, self.shm.name, self.frame_shape, self.running_flag),
        #     daemon=True
        # )
        # self.worker.start()
        # time.sleep(2.5)  # Warm up wait window

        # if not self.running_flag.value:
        #     self.shm.close()
        #     self.shm.unlink()
        #     raise RuntimeError("Background worker process failed to map the stream link.")

        # self.shared_array_view = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.shm.buf)

        # Reusable hardware space optimizations
        # self.static_buffer_tensor = torch.empty(self.frame_shape, dtype=torch.uint8, pin_memory=True)
        # self.static_buffer_numpy = self.static_buffer_tensor.numpy()
        # Reusable double-buffered hardware space optimizations
        # self.static_buffer_tensors = [
        #     torch.empty(self.frame_shape, dtype=torch.uint8, pin_memory=True),
        #     torch.empty(self.frame_shape, dtype=torch.uint8, pin_memory=True)
        # ]
        # self.static_buffer_numpys = [t.numpy() for t in self.static_buffer_tensors]
        # self.buffer_selector = 0  # Alternates between 0 and 1
        self.upload_stream = torch.cuda.Stream(device=self.device)

        self.download_stream = torch.cuda.Stream(device=self.device)
        self.d2h_buffers = [
            torch.empty((360, 640, 3), dtype=torch.uint8, pin_memory=True),
            torch.empty((360, 640, 3), dtype=torch.uint8, pin_memory=True),
        ]
        self.d2h_numpys = [b.numpy() for b in self.d2h_buffers]
        self.d2h_selector = 0

        # Reusable hardware space optimizations using OS-level Page-Locking
        try:
            self.cudart = ctypes.CDLL("libcudart.so")  # Linux
        except OSError:
            self.cudart = ctypes.CDLL("cudart64_120.dll")  # Windows

        self.pinned_views = []
        cudaHostRegisterPortable = 0x01  # Visible to all CUDA contexts

        # Permanently register and create zero-copy tensor views over all 3 SHM allocations
        for shm in self.shms:
            # 1. Create a ctypes character array mapping directly onto the memoryview buffer
            ctypes_array = (ctypes.c_char * self.frame_bytes).from_buffer(shm.buf)

            # 2. Extract the true virtual memory address pointer from the ctypes overlay
            shm_ptr = ctypes.c_void_p(ctypes.addressof(ctypes_array))

            # 3. Pin the raw memory chunk inside the OS page tracking tables
            res = self.cudart.cudaHostRegister(
                shm_ptr, self.frame_bytes, cudaHostRegisterPortable
            )
            if res != 0:
                raise RuntimeError(f"cudaHostRegister failed with status code: {res}")

            # 4. Create a high-speed zero-copy NumPy -> Torch wrapper view over that allocation
            shm_numpy = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=shm.buf)
            shm_tensor = torch.from_numpy(shm_numpy)
            self.pinned_views.append(shm_tensor)

        # self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        # self.thread.start()

    def _processing_loop(self):
        # target_fps = 15.0
        # frame_num = 0.0
        # next_process_idx = 0.0
        # step_size = float(self.input_fps) / float(self.target_fps)
        # # self.total_queue_wait_time = 0.0

        # # Base loop pacing on the native camera interval (e.g., 33.3ms for 30 FPS)
        # inbound_frame_interval = 1.0 / float(self.input_fps)
        last_processed_shm_idx = -1
        torch.cuda.synchronize(self.device)

        while not self.stopped:  # and self.running_flag.value:
            # loop_start = time.perf_counter()
            # is_target_frame = (frame_num >= next_process_idx)

            # # Central timeline step tracking
            # frame_num += 1.0

            # if not is_target_frame:
            #     continue

            with self.frame_condition:
                # Thread sleeps instantly until the background worker calls notify_all()
                # self.frame_condition.wait(timeout=1.0)
                # Only wait if the background worker hasn't delivered a new index yet
                while (
                    self.latest_idx.value == last_processed_shm_idx and not self.stopped
                ):
                    if not self.frame_condition.wait(timeout=0.05):
                        if (
                            not self.running_flag.value
                            and self.latest_idx.value == last_processed_shm_idx
                        ):
                            self.stopped = True
                            break
                        continue
                if self.stopped:
                    break

                active_idx = self.latest_idx.value
                self.reader_idx.value = active_idx
                last_processed_shm_idx = active_idx

                # Update counters inside target_fps space loop execution
                self.target_frames_passed += 1

                # Reconstruct source frame position using step spacing ratio
                # if self.numFrames != 0:
                # self.total_input_frames = int(
                #     self.target_frames_passed * getattr(self, "frame_skip", 1)
                # )
                if self.is_rtsp:
                    self.total_input_frames = int(self.raw_frame_counter.value)

                # Signal the worker that a slot has cleared up losslessly ---
                self.buffer_occupancy.value = max(0, self.buffer_occupancy.value - 1)

                # is_target_frame = (frame_num >= next_process_idx)

            # if not is_target_frame:
            #     continue

            # # Check if frame is a target frame before heavy operations
            # # if frame_num >= next_process_idx:
            # next_process_idx += step_size

            t_copy = time.perf_counter()
            # np.copyto(self.static_buffer_numpy, self.shared_array_view)
            # Zero-Copy Pointer Swap: lock the latest completed buffer index
            # active_idx = self.latest_idx.value
            # self.reader_idx.value = active_idx
            # last_processed_shm_idx = active_idx

            # # Signal the worker that a slot has cleared up losslessly ---
            # self.buffer_occupancy.value = max(0, self.buffer_occupancy.value - 1)

            # Wrap an array view directly around the locked active SHM region
            # active_shm = self.shms[active_idx]
            # frame_view = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=active_shm.buf)

            # Select the current active buffer slot
            # buf_idx = self.buffer_selector
            # current_numpy_buf = self.static_buffer_numpys[buf_idx]
            # current_tensor_buf = self.static_buffer_tensors[buf_idx]

            # np.copyto(current_numpy_buf, frame_view)
            current_tensor_view = self.pinned_views[active_idx]
            self.total_shm_copy_time += time.perf_counter() - t_copy

            # if self.frame_queue.full() and self.is_rtsp:
            #     # If the consumer loop is dragging, discard this frame instantly
            #     # without allocating any memory or touching the PCIe bus.
            #     return

            # ASYNCHRONOUS PCIe UPLOAD (Only runs for target frames)
            t_h2d = time.perf_counter()

            # Isolate the PCIe upload onto its own dedicated side hardware stream
            # with torch.cuda.stream(self.upload_stream):
            # compute_tensor = self.static_buffer_tensor.to(self.device, non_blocking=True)
            # Upload directly from the selected zero-copy host view area
            # compute_tensor = torch.from_numpy(frame_view).to(self.device, non_blocking=True)
            # compute_tensor = current_tensor_buf.to(self.device, non_blocking=True)

            # Transfer the raw [H, W, 3] uint8 tensor to VRAM via DMA copy
            compute_tensor = current_tensor_view.to(self.device, non_blocking=True)
            bgr_tensor = compute_tensor[:, :, [2, 1, 0]].contiguous()
            # torch.cuda.synchronize(self.device)
            self.total_h2d_time += time.perf_counter() - t_h2d

            # self.buffer_selector = 1 - self.buffer_selector

            # if not self.is_rtsp or not self.frame_queue.full():
            t_queue_block = time.perf_counter()
            # Record an event on the stream and make the main thread wait for it asynchronously
            current_event = torch.cuda.Event()
            current_event.record(self.upload_stream)
            # current_event.wait()
            torch.cuda.current_stream().wait_event(current_event)

            # self.frame_queue.put((True, cpu_360p_frame))
            self.frame_queue.put(
                (True, bgr_tensor.detach(), self.target_frames_passed)
            )  # .clone()))
            self.total_queue_wait_time += time.perf_counter() - t_queue_block

            # frame_num += 1.0
            # self.frame_idx += 1

            # Keep the reader thread aligned with target ingestion speeds
            # elapsed = time.perf_counter() - loop_start
            # time_to_wait = inbound_frame_interval - elapsed
            # if time_to_wait > 0:
            #     time.sleep(time_to_wait)
            # Explicitly delete frame variables to free their references
            if "compute_tensor" in locals():
                del compute_tensor

            # Force PyTorch's internal allocator to release cached segments back to the OS
            # torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()

        # self.frame_idx = frame_num
        # self.running_flag.value = False

    def stop(self):
        """Cleanly unregisters the locked memory mapping blocks from the OS page table."""
        if hasattr(self, "cudart") and hasattr(self, "shms"):
            import ctypes

            for shm in self.shms:
                # 1. Safely bind ctypes to the underlying memory view structure
                ctypes_array = (ctypes.c_char * self.frame_bytes).from_buffer(shm.buf)

                # 2. Extract the address pointer
                shm_ptr = ctypes.c_void_p(ctypes.addressof(ctypes_array))

                # 3. Unregister the address safely from the OS tables
                self.cudart.cudaHostUnregister(shm_ptr)

                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        super().stop()
