# ==============================================================================
# IMPORTS
import ctypes
import gc
import inspect
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import types
from collections import deque
from multiprocessing import Condition, Process, Value
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from ultralytics.utils.checks import check_imgsz

sys.path.insert(1, str(Path(__file__).parent.parent))
from include.default_configs import ENABLE_QUERYING_DEFAULT
from include.utils import (
    PipelineConfig,
    ResourceTrackerFilter,
    manual_fps_calculation,
    str2bool,
)

# ==============================================================================
# LOGGING

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("libav").setLevel(logging.CRITICAL)
logging.getLogger("libav.hevc").setLevel(logging.CRITICAL)
av.logging.set_level(av.logging.PANIC)

main_app_logger = logging.getLogger(__name__)


# ==============================================================================
# PIPELINE CONFIGURATION

cv2.setNumThreads(0)

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


def capture_shared_memory_worker(
    startup_event,
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
    slot_available_event,
    num_shm_slots,
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

    if startup_event:
        startup_event.wait()  # Worker will pause here until the main app is ready

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
            raw_frame_counter.value = int(worker_frame_num)
            worker_frame_num += 1.0

            if not is_target_frame:
                # Fast metadata-only skip: Zero heavy decode compute or CPU/GPU overhead
                if not cap.grab():
                    running_flag.value = False
                    with frame_condition:
                        frame_condition.notify_all()
                    break
                # raw_frame_counter.value += 1
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                running_flag.value = False
                with frame_condition:
                    frame_condition.notify_all()
                break

            worker_next_process_idx += step_size

            # Prevent local files from overwriting unread ring buffer slots ---
            # if not str(source_input).lower().startswith("rtsp://"):
            #     # while buffer_occupancy.value >= 2 and running_flag.value:
            #     #     pass
            #     with frame_condition:
            #         while buffer_occupancy.value >= 2 and running_flag.value:
            #             # Drop the thread into a low-power kernel sleep state
            #             frame_condition.wait(timeout=0.01)

            # Prevent local files from overwriting unread ring buffer slots losslessly
            # if not str(source_input).lower().startswith("rtsp://"):
            #     if buffer_occupancy.value >= 2:
            #         slot_available_event.clear()  # Lock the gate
            #         while buffer_occupancy.value >= 2 and running_flag.value:
            #             # Block instantly via kernel context without time-stepping drifts
            #             slot_available_event.wait(timeout=1.0)

            if not str(source_input).lower().startswith("rtsp://"):
                max_allowed_occupancy = num_shm_slots - 1
                while (
                    buffer_occupancy.value >= max_allowed_occupancy
                    and running_flag.value
                ):
                    time.sleep(0.001)

            # Dynamic Ring Buffer Selection: Identify free block
            curr_latest = latest_idx.value
            curr_reader = reader_idx.value

            write_idx = None
            for idx in range(num_shm_slots):
                if idx != curr_latest and idx != curr_reader:
                    write_idx = idx
                    break

            # Fallback if somehow all slots are locked (should theoretically never happen)
            if write_idx is None:
                write_idx = (curr_latest + 1) % num_shm_slots

            arrays[write_idx][:] = frame[:]

            # Notify waiting processing loops without any polling latency
            with frame_condition:
                latest_idx.value = write_idx
                # raw_frame_counter.value += 1
                buffer_occupancy.value += 1
                frame_condition.notify_all()  # Wake up all waiting reader threads instantly!

    except Exception:
        running_flag.value = False
    finally:
        cap.release()

        # Delete the memoryview-holding arrays first
        del arrays

        for shm in shm_blocks:
            try:
                shm.close()
            except Exception:
                pass
        del shm_blocks
        gc.collect()


class BaseReader:
    def __init__(
        self,
        source,
        startup_event=None,
        target_fps=BASE_PIPELINE_CONFIG.TARGET_FPS,
        clip_duration=BASE_PIPELINE_CONFIG.CLIP_DURATION,
        MODEL_W=BASE_PIPELINE_CONFIG.MODEL_W,
        MODEL_H=BASE_PIPELINE_CONFIG.MODEL_H,
        queue_size=2,
    ):
        self.source = source
        self.is_rtsp = str(self.source).startswith("rtsp://")
        # self.frame_queue = queue.Queue(maxsize=queue_size)
        self.frame_queue = deque(maxlen=queue_size)
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

        # Need exactly 3 slots to prevent a lock collision
        num_shm_slots = max(3, queue_size)
        self.shms = [
            SharedMemory(create=True, size=self.frame_bytes)
            for _ in range(num_shm_slots)
        ]

        self.running_flag = Value("b", True)
        self.raw_frame_counter = Value("i", 0)
        self.latest_idx = Value("i", 0)  # Tracks the newest complete frame index
        self.reader_idx = Value("i", -1)  # Locks the frame currently being processed
        shm_names = [shm.name for shm in self.shms]
        # atomic counter to track unread slot density ---
        self.buffer_occupancy = Value("i", 0)

        # Create a shared cross-process condition lock variable
        self.frame_condition = Condition()
        self.slot_available_event = mp.Event()
        self.slot_available_event.set()  # Default to unblocked state

        self.worker = Process(
            target=capture_shared_memory_worker,
            args=(
                startup_event,
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
                self.slot_available_event,
                num_shm_slots,
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

    def read(self):
        """
        Optimized Kernel-Native Condition Event queue fetcher.
        Parks the consumer thread instantly without CPU thrashing or GIL locks,
        waking up on hardware-driven signals from the producer process.
        """
        # Escape check: Stream explicitly halted and queue empty
        if self.stopped and len(self.frame_queue) == 0:
            return False, None, None, None, None

        # # 2. Native Hardware Wait Boundary
        # # If the queue is starved, drop the thread into an un-polled kernel block
        # if len(self.frame_queue) == 0 and not self.stopped:
        #     with self.frame_condition:
        #         # Re-verify inside the lock to safely block multi-thread race patterns
        #         while len(self.frame_queue) == 0 and not self.stopped:
        #             # Increase watchdog timeout from 2.0s to 10.0s for local file testing
        #             timeout_sec = 2.0 if self.is_rtsp else 10.0

        #             # Thread blocks via OS futex without time-stepping drift or context jitter.
        #             # Wakes up immediately when the processing loop fires notify_all().
        #             if not self.frame_condition.wait(timeout=timeout_sec):
        #                 # Watchdog trigger: 2 seconds of zero frame activity indicates a dead stream
        #                 return (
        #                     False,
        #                     None,
        #                     None,
        #                     self.target_frames_passed,
        #                     self.total_input_frames,
        #                 )

        # Lock-Free Pop Extraction Pass
        if len(self.frame_queue) > 0:
            try:
                return self.frame_queue.popleft()
            except IndexError:
                pass  # Multi-thread race fallback guard rail

        # Brief non-blocking yield if queue is momentarily empty
        if not self.stopped:
            time.sleep(0.001)
            if len(self.frame_queue) > 0:
                try:
                    return self.frame_queue.popleft()
                except IndexError:
                    pass

        return False, None, None, self.target_frames_passed, self.total_input_frames

    def start(self):
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        """Cleanly unregisters the locked memory mapping blocks from the OS page table."""
        if self.stopped:
            return

        self.stopped = True

        # Signal the worker process to exit its loop
        if hasattr(self, "running_flag"):
            self.running_flag.value = False

        if hasattr(self, "worker") and self.worker is not None:
            try:
                if self.worker.is_alive():
                    self.worker.terminate()
                    self.worker.join(timeout=0.5)
                    # Execute an OS-level SIGKILL fallback if process flags hang
                    if self.worker.is_alive() and getattr(self.worker, "pid", None):
                        os.kill(self.worker.pid, signal.SIGKILL)
                        self.worker.join()
            except Exception:
                pass
            self.worker = None

        # Join active processing pools and thread handles safely
        if hasattr(self, "frame_condition") and self.frame_condition is not None:
            try:
                with self.frame_condition:
                    self.frame_condition.notify_all()
            except Exception:
                pass
            self.frame_condition = None

        if hasattr(self, "thread") and self.thread is not None:
            try:
                if self.thread.is_alive():
                    self.thread.join(timeout=0.5)
            except Exception:
                pass
            self.thread = None

        # 6. Set and unbind OS-level Event notification primitive file descriptor pipelines
        if (
            hasattr(self, "slot_available_event")
            and self.slot_available_event is not None
        ):
            try:
                self.slot_available_event.set()
                if (
                    hasattr(self.slot_available_event, "_handle")
                    and self.slot_available_event._handle
                ):
                    self.slot_available_event._handle.close()
            except Exception:
                pass
            self.slot_available_event = None

        # 7. Drain and empty the frame buffer queue completely
        if hasattr(self, "frame_queue") and self.frame_queue is not None:
            try:
                while len(self.frame_queue) > 0:
                    self.frame_queue.popleft()
            except Exception:
                pass
            self.frame_queue = None

        if hasattr(self, "shms") and self.shms:
            for shm_block in list(self.shms):
                try:
                    if hasattr(shm_block, "buf") and shm_block.buf is not None:
                        shm_block.buf.release()
                except Exception:
                    pass
                try:
                    shm_block.close()
                except Exception:
                    pass
                try:
                    shm_block.unlink()
                except Exception:
                    pass
            self.shms.clear()

        if hasattr(self, "bridge_thread"):
            self.bridge_thread.join(timeout=1.0)

        for primitive_attr in [
            "running_flag",
            "buffer_occupancy",
            "latest_idx",
            "reader_idx",
            "raw_frame_counter",
        ]:
            if hasattr(self, primitive_attr):
                setattr(self, primitive_attr, None)

        self.clean_up_tensors_and_arrays()

    def clean_up_tensors_and_arrays(self):
        main_app_logger.info(
            "[READER CLEANUP] Safely stripping active runtime arrays ..."
        )
        # Collect references safely using strict type checking
        # This completely avoids pulling unmanaged proxy objects from the heap
        all_live_objects = gc.get_objects()

        target_tensors = []
        target_arrays = []

        for obj in all_live_objects:
            try:
                obj_type = type(obj)
                if isinstance(obj, types.FrameType):
                    frame_info = inspect.getframeinfo(obj)
                    if (
                        "openvino" in frame_info.filename
                        or "openvino.py" in frame_info.filename
                    ):
                        # Clear the frame's local variable dictionary to break circular references
                        obj.f_locals.clear()

                # Check for concrete types to prevent triggering proxy __getattr__ hooks
                if obj_type is torch.Tensor:
                    target_tensors.append(obj)
                elif obj_type is np.ndarray:
                    # Explicitly guard size checks to keep it completely stable
                    if obj.base is None and obj.ndim > 0:
                        target_arrays.append(obj)
            except Exception:
                pass

        # 3. Truncate discovered references in-place without triggering deletions
        reclaimed_tensors = 0
        for tensor in target_tensors:
            try:
                # Truncate raw storage footprint safely
                tensor.data = torch.empty(0, device=self.device_input)
                reclaimed_tensors += 1
            except Exception:
                pass

        reclaimed_arrays = 0
        for arr in target_arrays:
            try:
                # Shrink writeable numpy arrays down to 0 bytes safely
                if arr.flags.writeable:
                    arr.resize((0,), refcheck=False)
                    reclaimed_arrays += 1
            except (ValueError, SystemError):
                pass

        main_app_logger.info(
            f"[READER CLEANUP] Reclaimed {reclaimed_tensors} tensors and {reclaimed_arrays} arrays safely."
        )

        # Clean local registers immediately
        all_live_objects = None
        target_tensors = None
        target_arrays = None
        gc.collect()

    def unlink_shared_memory(self, all_shm_keys):
        for key in all_shm_keys:
            val = getattr(self, key, None)
            if val is None:
                continue
            for shm in list(val):
                if shm is not None:
                    try:
                        # Drop direct memoryview trackers instantly
                        if hasattr(shm, "buf") and shm.buf is not None:
                            try:
                                shm.buf.release()
                            except Exception:
                                pass

                        # Invalidate private mmap structures to drop pytest stack frame caches
                        if hasattr(shm, "_mmap") and shm._mmap is not None:
                            try:
                                shm._mmap = None
                            except Exception:
                                pass

                        # try:
                        #     shm.__class__.__del__ = lambda self: None
                        # except Exception:
                        #     pass

                        # # Unlink text descriptors out of the core tracker process
                        # shm_descriptor = shm._name if shm._name.startswith("/") else f"/{shm._name}"
                        # try:
                        #     unregister(shm_descriptor, "shared_memory")
                        # except Exception:
                        #     pass

                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass
            try:
                val.clear()
            except Exception:
                pass

    # Gets video details
    def get_fps_and_framecnt(self, cap, target_fps, clip_duration):
        self.input_fps = cap.get(cv2.CAP_PROP_FPS)  # hardware fps
        # print(f"in fps: {self.input_fps} target fps: {target_fps}")
        if self.input_fps == 0:  # Case when FPS isn't available
            self.input_fps = manual_fps_calculation(cap, num_frames=10)
            main_app_logger.debug(f"new in fps: {self.input_fps}")
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

        self.step_size = self.input_fps / self.target_fps
        if self.input_fps > 0 and self.target_fps > 0:
            self.frame_skip = max(1, int(self.step_size))
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

        self.width = new_sizeHW[1]
        self.height = new_sizeHW[0]

        # Configure scaling for 8K-to-Model coordinate mapping
        self.resize_h, self.resize_w = [self.MODEL_H, self.MODEL_W]
        self.scale_x = self.frame_width / self.MODEL_W
        self.scale_y = self.frame_height / self.MODEL_H


class CPUReader(BaseReader):
    """Asynchronous CPU frame reader and processor utilizing AVX2 optimized OpenCV routines."""

    def __init__(
        self,
        source,
        startup_event=None,
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
            startup_event=startup_event,
            target_fps=target_fps,
            clip_duration=clip_duration,
            MODEL_W=MODEL_W,
            MODEL_H=MODEL_H,
            queue_size=queue_size,
        )

        # self.MODEL_H = MODEL_H
        # self.MODEL_W = MODEL_W
        self.device_index = "cpu"

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

    @torch.inference_mode()
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
                    if not self.frame_condition.wait(timeout=0.5):
                        if (
                            not self.running_flag.value
                            # and self.latest_idx.value == last_processed_shm_idx
                        ):
                            self.stopped = True
                            break
                        # continue
                    # self.frame_condition.wait()
                if self.stopped:
                    break

                active_idx = self.latest_idx.value
                self.reader_idx.value = active_idx
                last_processed_shm_idx = active_idx

                # Signal the worker that a slot has cleared up losslessly ---
                self.buffer_occupancy.value = max(0, self.buffer_occupancy.value - 1)
                # self.slot_available_event.set()

                # is_target_frame = (frame_num >= next_process_idx)
                # Reconstruct source frame position using step spacing ratio
                # if self.numFrames != 0:
                # self.total_input_frames = int(
                #     self.target_frames_passed * getattr(self, "frame_skip", 1)
                # )
                # if self.is_rtsp:

                target_frames_passed = self.target_frames_passed
                # total_input_frames = int(
                #     target_frames_passed * getattr(self, "frame_skip", 1)
                # )  # int(self.raw_frame_counter.value)
                total_input_frames = round(target_frames_passed * self.step_size)

                # Update counters inside target_fps space loop execution
                self.total_input_frames = total_input_frames

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

            t_queue_block = time.perf_counter()
            # if not self.is_rtsp or not self.frame_queue.full():
            #     t_queue_block = time.perf_counter()
            #     # self.frame_queue.put((True, cpu_360p_frame))
            #     self.frame_queue.put(
            #         (True, frame_view, target_frames_passed, total_input_frames),
            #         # timeout=1.0,
            #     )  # self.static_buffer_numpy.copy()))
            #     self.total_queue_wait_time += time.perf_counter() - t_queue_block

            if self.is_rtsp:
                # If the consumer loop drops frames, instantly evict the oldest
                # matrix view reference to keep frame processing real-time.
                if len(self.frame_queue) >= self.frame_queue.maxlen:
                    try:
                        self.frame_queue.popleft()
                    except IndexError:
                        pass
                self.frame_queue.append(
                    (True, frame_view, None, target_frames_passed, total_input_frames)
                )
            else:
                # Lossless tracking sequence for local testing files
                # while len(self.frame_queue) >= self.frame_queue.maxlen and not self.stopped:
                #     time.sleep(0.001)
                safe_cpu_copy = frame_view.copy()
                self.frame_queue.append(
                    (
                        True,
                        safe_cpu_copy,
                        None,
                        target_frames_passed,
                        total_input_frames,
                    )
                )

            self.total_queue_wait_time += time.perf_counter() - t_queue_block
            self.target_frames_passed += 1

            # frame_num += 1.0

            # Keep the reader thread aligned with target ingestion speeds
            # elapsed = time.perf_counter() - loop_start
            # time_to_wait = inbound_frame_interval - elapsed
            # if time_to_wait > 0:
            #     time.sleep(time_to_wait)

        # self.frame_queue.put((False, None, target_frames_passed, total_input_frames))
        # self.frame_idx = frame_num
        # self.running_flag.value = False


class GPUReader(BaseReader):
    """Asynchronous GPU frame reader leveraging Pinned Host Memory and PyTorch CUDA tensors."""

    def __init__(
        self,
        source,
        startup_event=None,
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
            startup_event=startup_event,
            target_fps=target_fps,
            clip_duration=clip_duration,
            MODEL_W=MODEL_W,
            MODEL_H=MODEL_H,
            queue_size=queue_size,
        )

        self.gpu_id = gpu_id
        self.device_index = torch.device(f"cuda:{gpu_id}")

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

        self._static_gpu_frame_buffer = torch.empty(
            (self.frame_height, self.frame_width, 3),
            dtype=torch.uint8,
            device=self.device_index,
        )
        self._flipped_static_gpu_frame_buffer = torch.empty_like(
            self._static_gpu_frame_buffer
        )
        # Pre-allocate a second buffer for the BGR conversion.
        # This is our destination tensor.
        self._bgr_gpu_frame_buffer = torch.empty(
            (self.frame_height, self.frame_width, 3),
            dtype=torch.uint8,
            device=self.device_index,
        )

        # self.static_buffer_tensor = torch.empty(self.frame_shape, dtype=torch.uint8, pin_memory=True)
        # self.static_buffer_numpy = self.static_buffer_tensor.numpy()
        # Reusable double-buffered hardware space optimizations
        # self.static_buffer_tensors = [
        #     torch.empty(self.frame_shape, dtype=torch.uint8, pin_memory=True),
        #     torch.empty(self.frame_shape, dtype=torch.uint8, pin_memory=True)
        # ]
        # self.static_buffer_numpys = [t.numpy() for t in self.static_buffer_tensors]
        # self.buffer_selector = 0  # Alternates between 0 and 1
        self.upload_stream = torch.cuda.Stream(device=self.device_index)

        self.download_stream = torch.cuda.Stream(device=self.device_index)
        self.d2h_buffers = [
            torch.empty((360, 640, 3), dtype=torch.uint8, pin_memory=True),
            torch.empty((360, 640, 3), dtype=torch.uint8, pin_memory=True),
        ]
        self.d2h_numpys = [b.numpy() for b in self.d2h_buffers]
        self.d2h_selector = 0

        self.pinned_views = []
        for shm in self.shms:
            shm_numpy = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=shm.buf)
            shm_tensor = torch.from_numpy(shm_numpy)
            self.pinned_views.append(shm_tensor)
        # # Reusable hardware space optimizations using OS-level Page-Locking
        # try:
        #     self.cudart = ctypes.CDLL("libcudart.so")  # Linux
        # except OSError:
        #     try:
        #         self.cudart = ctypes.CDLL("libcudart.so.12")
        #     except OSError:
        #         self.cudart = ctypes.CDLL("cudart64_120.dll")

        # self.cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        # self.cudart.cudaHostRegister.restype = ctypes.c_int

        # self.cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        # self.cudart.cudaHostUnregister.restype = ctypes.c_int

        # self.pinned_views = []
        # cudaHostRegisterPortable = 0x01  # Visible to all CUDA contexts

        # # Permanently register and create zero-copy tensor views over all 3 SHM allocations
        # for shm in self.shms:
        #     # 1. Create a ctypes character array mapping directly onto the memoryview buffer
        #     ctypes_array = (ctypes.c_char * self.frame_bytes).from_buffer(shm.buf)

        #     # 2. Extract the true virtual memory address pointer from the ctypes overlay
        #     shm_ptr = ctypes.c_void_p(ctypes.addressof(ctypes_array))

        #     # 3. Pin the raw memory chunk inside the OS page tracking tables
        #     res = self.cudart.cudaHostRegister(
        #         shm_ptr, self.frame_bytes, cudaHostRegisterPortable
        #     )
        #     # if res != 0:
        #     if res == 712:
        #         # main_app_logger.warning(
        #         #     f"[CUDA SHM] Address {hex(ctypes.addressof(ctypes_array))} is already page-locked. Reusing active pin safely."
        #         # )
        #         main_app_logger.info(
        #             f"[CUDA RESTORE] Found stale page-lock at {hex(shm_ptr.value)}. Forcing reset..."
        #         )
        #         # try:
        #         #     # Load low-level driver to break the lazy process hold
        #         #     # import ctypes
        #         #     # cuda_driver = ctypes.CDLL("libcuda.so")  #.6")
        #         #     # cuda_driver.cuMemHostUnregister(shm_ptr)
        #         #     # force_clear_driver_cuda_pin(shm_ptr.value)
        #         #     try:
        #         #         cuda_driver = ctypes.CDLL("libcuda.so.6")
        #         #     except OSError:
        #         #         cuda_driver = ctypes.CDLL("libcuda.so")

        #         #     # Break the registration using the low-level driver layout hook
        #         #     cuda_driver.cuMemHostUnregister(shm_ptr)
        #         self.cudart.cudaHostUnregister(shm_ptr)
        #         # Re-register immediately within the new PyTorch runtime context
        #         res = self.cudart.cudaHostRegister(
        #             shm_ptr, self.frame_bytes, cudaHostRegisterPortable
        #         )
        #         # except Exception as e:
        #         #     main_app_logger.debug(
        #         #         f"Driver pin restoration fallback skipped: {e}"
        #         #     )
        #     if res != 0 and res != 712:
        #         raise RuntimeError(f"cudaHostRegister failed with status code: {res}")

        #     # 4. Create a high-speed zero-copy NumPy -> Torch wrapper view over that allocation
        #     # shm_numpy = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=shm.buf)
        #     shm_numpy = np.frombuffer(shm._mmap, dtype=np.uint8).reshape(
        #         self.frame_shape
        #     )
        #     shm_tensor = torch.from_numpy(shm_numpy)
        #     self.pinned_views.append(shm_tensor)

        # self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        # self.thread.start()

    @torch.inference_mode()
    def _processing_loop_v1(self):
        last_processed_shm_idx = -1
        # torch.cuda.synchronize(self.device_index)
        torch.cuda.synchronize()

        while not self.stopped:  # and self.running_flag.value:
            with self.frame_condition:
                # Thread sleeps instantly until the background worker calls notify_all()
                # self.frame_condition.wait(timeout=1.0)
                # Only wait if the background worker hasn't delivered a new index yet
                while (
                    self.latest_idx.value == last_processed_shm_idx and not self.stopped
                ):
                    if not self.frame_condition.wait(timeout=0.5):
                        if (
                            not self.running_flag.value
                            # and self.latest_idx.value == last_processed_shm_idx
                        ):
                            self.stopped = True
                            break
                    #     continue
                    # self.frame_condition.wait()
                if self.stopped:
                    break

                active_idx = self.latest_idx.value
                self.reader_idx.value = active_idx
                last_processed_shm_idx = active_idx

                # Signal the worker that a slot has cleared up losslessly ---
                self.buffer_occupancy.value = max(0, self.buffer_occupancy.value - 1)
                # self.slot_available_event.set()

                target_frames_passed = self.target_frames_passed
                # total_input_frames = int(
                #     target_frames_passed * getattr(self, "frame_skip", 1)
                # )  # int(self.raw_frame_counter.value)
                total_input_frames = round(target_frames_passed * self.step_size)

                # Update counters inside target_fps space loop execution
                self.total_input_frames = total_input_frames

            t_copy = time.perf_counter()
            # np.copyto(current_numpy_buf, frame_view)
            current_tensor_view = self.pinned_views[active_idx]
            self.total_shm_copy_time += time.perf_counter() - t_copy

            # ASYNCHRONOUS PCIe UPLOAD (Only runs for target frames)
            t_h2d = time.perf_counter()

            self._static_gpu_frame_buffer.copy_(current_tensor_view, non_blocking=False)

            # 2. Native Channel Inversion View (RGB -> BGR)
            # Instead of calling OpenCV, use standard tensor index slicing.
            # This keeps the execution entirely in PyTorch's native C++ backend.
            # bgr_view = self._static_gpu_frame_buffer[:, :, [2, 1, 0]]

            # 3. Fast Contiguity Restoration Block
            # Rather than letting downstream handlers detect a non-contiguous stride
            # layout (which causes major latency spikes), enforce memory linearity
            # asynchronously inside the upload stream context here:
            # bgr_tensor = bgr_view.contiguous()
            # bgr_tensor = torch.flip(self._static_gpu_frame_buffer, dims=[2]).contiguous()
            # torch.cuda.synchronize(self.device_index)
            self._bgr_gpu_frame_buffer[:, :, 0] = self._static_gpu_frame_buffer[:, :, 2]
            self._bgr_gpu_frame_buffer[:, :, 1] = self._static_gpu_frame_buffer[:, :, 1]
            self._bgr_gpu_frame_buffer[:, :, 2] = self._static_gpu_frame_buffer[:, :, 0]

            # The final tensor to be sent is now the pre-allocated BGR buffer.
            bgr_tensor = self._bgr_gpu_frame_buffer
            self.total_h2d_time += time.perf_counter() - t_h2d

            # self.buffer_selector = 1 - self.buffer_selector

            # if not self.is_rtsp or not self.frame_queue.full():
            t_queue_block = time.perf_counter()
            # Record an event on the stream and make the main thread wait for it asynchronously
            current_event = torch.cuda.Event()
            current_event.record(self.upload_stream)
            # current_event.wait()
            # ---
            if self.is_rtsp:
                if len(self.frame_queue) >= self.frame_queue.maxlen:
                    try:
                        # Pop the oldest item from the queue
                        stale_item = self.frame_queue.popleft()
                        # Extract its tracking event reference
                        stale_event = stale_item[2]
                        if stale_event:
                            # Clear the event reference without blocking the hot loop
                            del stale_event
                    except IndexError:
                        pass
                self.frame_queue.append(
                    (
                        True,
                        bgr_tensor,
                        current_event,
                        target_frames_passed,
                        total_input_frames,
                    )
                )

            else:
                # LOCAL Lossless Files: Check the previous frame's hardware event
                # instead of blocking on the current frame's upload status.
                if len(self.frame_queue) >= self.frame_queue.maxlen:
                    # Retrieve the oldest active item to check its status
                    oldest_frame_set = self.frame_queue[0]
                    oldest_event = oldest_frame_set[2]

                    if oldest_event and not oldest_event.query():
                        # Only wait if the hardware lane is completely saturated
                        # oldest_event.synchronize()

                        # Yield thread execution priority cooperatively for a microscopic window
                        # to let the PCIe DMA engine finish transferring bytes smoothly
                        time.sleep(0.0005)
                        # 2. Asynchronous Watchdog Check
                        if not oldest_event.query():
                            oldest_event.synchronize()  # Hard fence fallback only when fully saturated

                # self.frame_queue.append(
                #     (True, bgr_tensor, current_event, target_frames_passed, total_input_frames)
                # )
                with self.frame_condition:
                    self.frame_queue.append(
                        (
                            True,
                            bgr_tensor,
                            current_event,
                            target_frames_passed,
                            total_input_frames,
                        )
                    )
                    self.frame_condition.notify_all()  # Instantly wake up your main read lane!

            # ---

            self.total_queue_wait_time += time.perf_counter() - t_queue_block
            self.target_frames_passed += 1
            del bgr_tensor, current_tensor_view  # bgr_view

    # @torch.inference_mode()
    # def _processing_loop(self):
    #     last_processed_shm_idx = -1
    #     # torch.cuda.synchronize(self.device_index)
    #     torch.cuda.synchronize()

    #     while not self.stopped:  # and self.running_flag.value:
    #         with self.frame_condition:
    #             # Thread sleeps instantly until the background worker calls notify_all()
    #             # self.frame_condition.wait(timeout=1.0)
    #             # Only wait if the background worker hasn't delivered a new index yet
    #             while (
    #                 self.latest_idx.value == last_processed_shm_idx and not self.stopped
    #             ):
    #                 if not self.frame_condition.wait(timeout=0.5):
    #                     if (
    #                         not self.running_flag.value
    #                         # and self.latest_idx.value == last_processed_shm_idx
    #                     ):
    #                         self.stopped = True
    #                         break
    #                 #     continue
    #                 # self.frame_condition.wait()
    #             if self.stopped:
    #                 break

    #             active_idx = self.latest_idx.value
    #             self.reader_idx.value = active_idx
    #             last_processed_shm_idx = active_idx

    #             # Signal the worker that a slot has cleared up losslessly ---
    #             self.buffer_occupancy.value = max(0, self.buffer_occupancy.value - 1)
    #             # self.slot_available_event.set()

    #             target_frames_passed = self.target_frames_passed
    #             # total_input_frames = int(
    #             #     target_frames_passed * getattr(self, "frame_skip", 1)
    #             # )  # int(self.raw_frame_counter.value)
    #             total_input_frames = round(target_frames_passed * self.step_size)

    #             # Update counters inside target_fps space loop execution
    #             self.total_input_frames = total_input_frames

    #         t_copy = time.perf_counter()
    #         # np.copyto(current_numpy_buf, frame_view)
    #         current_tensor_view = self.pinned_views[active_idx]
    #         self.total_shm_copy_time += time.perf_counter() - t_copy

    #         # ASYNCHRONOUS PCIe UPLOAD (Only runs for target frames)
    #         t_h2d = time.perf_counter()

    #         with torch.cuda.stream(self.upload_stream):
    #             self._static_gpu_frame_buffer.copy_(current_tensor_view, non_blocking=True)

    #             # Record the hardware completion event directly
    #             current_event = torch.cuda.Event()
    #             current_event.record(self.upload_stream)
    #         self.total_h2d_time += time.perf_counter() - t_h2d

    #         # self.buffer_selector = 1 - self.buffer_selector

    #         # if not self.is_rtsp or not self.frame_queue.full():
    #         t_queue_block = time.perf_counter()
    #         # Record an event on the stream and make the main thread wait for it asynchronously
    #         # current_event = torch.cuda.Event()
    #         # current_event.record(self.upload_stream)
    #         # current_event.wait()
    #         # ---
    #         if len(self.frame_queue) >= self.frame_queue.maxlen:
    #             try:
    #                 self.frame_queue.popleft()
    #             except IndexError:
    #                 pass

    #         self.frame_queue.append(
    #             (
    #                 True,
    #                 self._static_gpu_frame_buffer,
    #                 current_event,
    #                 target_frames_passed,
    #                 total_input_frames,
    #             )
    #         )

    #         # ---

    #         self.total_queue_wait_time += time.perf_counter() - t_queue_block
    #         self.target_frames_passed += 1
    #         del bgr_tensor, current_tensor_view  # bgr_view

    @torch.inference_mode()
    def _processing_loop(self):
        """
        High-Throughput Asynchronous GPU Frame Ingestion Loop.

        Transfers decoded 8K BGR frames from shared memory into pre-allocated VRAM buffers
        over a dedicated CUDA upload stream without blocking the CPU thread.
        """
        last_processed_shm_idx = -1

        # Pre-synchronize the upload stream before entering the hot loop
        if hasattr(self, "upload_stream") and self.upload_stream is not None:
            self.upload_stream.synchronize()

        while not self.stopped:
            # PULL NEXT READY FRAME INDEX FROM BACKGROUND WORKER -----------------
            if hasattr(self, "frame_condition") and self.frame_condition is not None:
                with self.frame_condition:
                    # Wait for the background reader process to deliver a new frame slot
                    while (
                        self.latest_idx.value == last_processed_shm_idx
                        and not self.stopped
                    ):
                        if not self.frame_condition.wait(timeout=0.2):
                            if (
                                not getattr(self, "running_flag", None)
                                or not self.running_flag.value
                            ):
                                self.stopped = True
                                break
                    if self.stopped:
                        break

            active_idx = self.latest_idx.value
            self.reader_idx.value = active_idx
            last_processed_shm_idx = active_idx

            # Decrement buffer occupancy atomically
            if hasattr(self, "buffer_occupancy"):
                self.buffer_occupancy.value = max(0, self.buffer_occupancy.value - 1)

            # Calculate physical vs. target frame indices
            target_frames_passed = self.target_frames_passed
            total_input_frames = round(
                target_frames_passed * getattr(self, "step_size", 1.0)
            )
            self.total_input_frames = total_input_frames

            # 2. ZERO-COPY HOST TENSOR RETRIEVAL -----------------------------------
            current_tensor_view = self.pinned_views[active_idx]

            # 3. ASYNCHRONOUS PCIe UPLOAD TO GPU (VRAM) ---------------------------
            t_h2d = time.perf_counter()

            # Execute DMA copy and hardware event record on the dedicated upload stream
            with torch.cuda.stream(self.upload_stream):
                # Asynchronous PCIe Host-to-Device transfer (Takes < 0.1ms CPU time)
                self._static_gpu_frame_buffer.copy_(
                    current_tensor_view, non_blocking=True
                )

                # Fast Zero-Allocation Color Swap (BGR Device -> RGB Device)
                # .flip(dims=[2]) creates a fast virtual view.
                # .copy_() writes it contiguously into the pre-allocated RGB buffer.
                self._flipped_static_gpu_frame_buffer.copy_(
                    self._static_gpu_frame_buffer.flip(dims=[2]), non_blocking=True
                )

                # Record hardware completion barrier on the upload stream
                current_event = torch.cuda.Event()
                current_event.record(self.upload_stream)

            self.total_h2d_time += time.perf_counter() - t_h2d

            # 4. LOCK-FREE THREAD-SAFE QUEUE INGESTION -----------------------------
            t_queue_block = time.perf_counter()

            # Evict oldest unread frame if queue is full to preserve real-time latency
            # if len(self.frame_queue) >= self.frame_queue.maxlen:
            #     try:
            #         self.frame_queue.popleft()
            #     except IndexError:
            #         pass

            # # Push the VRAM buffer and hardware completion event to the consumer queue
            # self.frame_queue.append(
            #     (
            #         True,
            #         self._static_gpu_frame_buffer,
            #         current_event,
            #         target_frames_passed,
            #         total_input_frames,
            #     )
            # )

            if getattr(self, "is_rtsp", False):
                # Live RTSP: Evict oldest frame if consumer falls behind
                if len(self.frame_queue) >= self.frame_queue.maxlen:
                    try:
                        self.frame_queue.popleft()
                    except IndexError:
                        pass
                self.frame_queue.append(
                    (
                        True,
                        self._flipped_static_gpu_frame_buffer,
                        current_event,
                        target_frames_passed,
                        total_input_frames,
                    )
                )
            else:
                # Local Video File: Wait cooperatively so ZERO frames are lost
                while (
                    len(self.frame_queue) >= self.frame_queue.maxlen
                    and not self.stopped
                ):
                    time.sleep(0.001)

                self.frame_queue.append(
                    (
                        True,
                        self._flipped_static_gpu_frame_buffer,
                        current_event,
                        target_frames_passed,
                        total_input_frames,
                    )
                )

            self.total_queue_wait_time += time.perf_counter() - t_queue_block
            self.target_frames_passed += 1

            # Release local pointer references immediately
            del current_tensor_view

    def stop(self):
        """GPU specialized cleaner layer. Unpins hardware tracking tables, flushes VRAM

        and synchronizes active side streams BEFORE parsing to the BaseReader.
        """
        if self.stopped:
            return

        self.stopped = True

        # Signal the worker process to exit its loop
        if hasattr(self, "running_flag"):
            self.running_flag.value = False

        # 1. Drive standalone GPU hardware unpinning while shm buffer mappings are valid
        # if hasattr(self, "release_hardware_pinsv2"):
        #     self.release_hardware_pinsv2()

        # 2. Break active PyTorch tensor data reference vectors to drop allocator pin metrics
        for tensor_attr in [
            "pinned_views",
            "d2h_buffers",
            "d2h_numpys",
            "_static_gpu_frame_buffer",
            "_bgr_gpu_frame_buffer",
        ]:
            if hasattr(self, tensor_attr):
                setattr(self, tensor_attr, None)

        # 3. Synchronize and overwrite side hardware execution lanes
        if hasattr(self, "upload_stream") and self.upload_stream is not None:
            try:
                self.upload_stream.synchronize()
            except Exception:
                pass
            self.upload_stream = None

        if hasattr(self, "download_stream") and self.download_stream is not None:
            try:
                self.download_stream.synchronize()
            except Exception:
                pass
            self.download_stream = None

        # 4. Dismantle low-level ctypes CDLL pointers to completely drop unmanaged contexts
        # self.cudart = None

        # 5. Hand execution over to BaseReader to tear down shared primitives and worker structures
        super().stop()

    def release_hardware_pins_v1(self):
        if hasattr(self, "cudart") and getattr(self, "shms", None):
            # import ctypes

            for shm in self.shms:
                try:
                    # Re-map the raw structural array pointer identity block
                    ctypes_array = (ctypes.c_char * self.frame_bytes).from_buffer(
                        shm.buf
                    )
                    shm_ptr = ctypes.c_void_p(ctypes.addressof(ctypes_array))

                    # Force the CUDA device table driver to drop the OS memory pin
                    res = self.cudart.cudaHostUnregister(shm_ptr)
                    if res != 0:
                        pass  # Ignore if already unpinned natively
                except Exception:
                    pass
            main_app_logger.info(
                "\033[92m[CUDA UNPIN] Shared memory pages successfully released from hardware table.\033[0m"
            )

    def release_hardware_pinsv2(self):
        """
        Reverses the CUDA host registration by mapping the exact memory locations
        and force-unpins them before closing the underlying OS shared memory handles.
        """
        # import ctypes
        # import gc
        # import sys

        # Localized stream filter to swallow intermediate garbage collection signals

        original_stderr = sys.stderr
        sys.stderr = ResourceTrackerFilter(original_stderr)

        try:
            # 2. Drop active processing frame buffers and views to unlock reference paths
            # if (
            #     hasattr(self, "_static_gpu_frame_buffer")
            #     and self._static_gpu_frame_buffer is not None
            # ):
            #     self._static_gpu_frame_buffer = None

            # if hasattr(self, "pinned_views") and self.pinned_views:
            #     for idx in range(len(self.pinned_views)):
            #         self.pinned_views[idx] = None
            #     self.pinned_views.clear()

            # # 3. Synchronize background streaming channels
            # if hasattr(self, "upload_stream") and self.upload_stream is not None:
            #     try:
            #         self.upload_stream.synchronize()
            #     except Exception:
            #         pass
            #     self.upload_stream = None

            # 4. Iterate and forcefully unregister the exact memory allocation pointers
            if hasattr(self, "cudart") and getattr(self, "shms", None):
                for shm in list(self.shms):
                    try:
                        if hasattr(shm, "buf") and shm.buf is not None:
                            # Map the exact size layout matching line 969 to capture the real pointer
                            ctypes_array = (
                                ctypes.c_char * self.frame_bytes
                            ).from_buffer(shm.buf)
                            shm_ptr = ctypes.c_void_p(ctypes.addressof(ctypes_array))

                            # Direct low-level Driver API fallback execution to wipe lazy driver context maps
                            # if hasattr(self, "force_clear_driver_cuda_pin"):
                            #     self.force_clear_driver_cuda_pin(shm_ptr.value)

                            # High-level Runtime API eviction
                            self.cudart.cudaHostUnregister(shm_ptr)
                        # if res == 0:
                        #     main_app_logger.info(
                        #         f"[CUDA UNPIN] Successfully unpinned address context: {hex(shm_ptr.value)}"
                        #     )

                        # 5. Instantly close and unlink POSIX shared memory allocations to release locks
                        # shm.close()
                        # shm.unlink()
                    except Exception as e:
                        main_app_logger.debug(
                            f"Failed to cleanly unpin shared memory region: {e}"
                        )

                # self.shms.clear()

        finally:
            # 6. Flush native device pools completely
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            sys.stderr = original_stderr
            # print(
            #     "\033[92m\n[CUDA UNPIN] Hardware tables fully cleared.\033[0m",
            #     flush=True,
            # )

    def release_hardware_pins(self):
        """Forcefully purges child workers and cleanly unpins raw CUDA memory layers."""

        # 1. FORCE-KILL THE BACKGROUND MULTIPROCESSING WORKER FIRST
        # If left alive, it anchors the page lock inside the Linux kernel tables!
        if hasattr(self, "worker") and self.worker is not None:
            try:
                if self.worker.is_alive():
                    self.worker.terminate()
                    self.worker.join(timeout=0.1)

                # Hard kill fallback if the worker process ignores graceful signals
                if self.worker.is_alive() and getattr(self.worker, "pid", None):
                    if self.worker.pid > 1:
                        # main_app_logger.info(
                        #     f"[KILL] Evicting worker holding lock (PID: {self.worker.pid})"
                        # )
                        try:
                            os.kill(self.worker.pid, signal.SIGKILL)
                            self.worker.join()
                        except ProcessLookupError:
                            pass  # It died naturally right before the signal arrived
            except Exception:
                pass
            self.worker = None

        # 2. Drop active PyTorch frame views to unlock storage layout references
        if (
            hasattr(self, "_static_gpu_frame_buffer")
            and self._static_gpu_frame_buffer is not None
        ):
            self._static_gpu_frame_buffer = None

        if hasattr(self, "pinned_views") and self.pinned_views:
            for idx in range(len(self.pinned_views)):
                self.pinned_views[idx] = None
            self.pinned_views.clear()

        # Flush outstanding elements inside your streaming frame buffers queue
        if hasattr(self, "frame_queue"):
            try:
                while len(self.frame_queue) > 0:
                    self.frame_queue.popleft()
            except Exception:
                pass
        gc.collect()

        # 3. Safely unregister host pages via the exact allocation pointers
        if hasattr(self, "cudart") and getattr(self, "shms", None):
            for shm in list(self.shms):
                try:
                    # Reconstruct the exact pointer alignment mapping to clear the driver registration
                    ctypes_array = (ctypes.c_char * self.frame_bytes).from_buffer(
                        shm.buf
                    )
                    shm_ptr = ctypes.c_void_p(ctypes.addressof(ctypes_array))

                    # Call high-speed unregister on the exact pointer value
                    self.cudart.cudaHostUnregister(shm_ptr)

                    # Release internal mmap references safely before executing close()
                    if hasattr(shm, "buf") and shm.buf is not None:
                        try:
                            shm.buf.release()
                        except Exception:
                            pass

                    if hasattr(shm, "_mmap") and shm._mmap is not None:
                        try:
                            shm._mmap = None
                        except Exception:
                            pass

                    # Direct file descriptor removal and unlinking from /dev/shm
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            self.shms.clear()

        # 4. Flush native memory pools completely back to Host OS
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def force_clear_driver_cuda_pin(raw_address_int):
    """
    Forcefully drops an unmanaged page-lock registration directly out of the
    NVIDIA Driver API context using a raw virtual address integer pointer.
    """
    if not raw_address_int:
        return

    main_app_logger.info(
        f"[HARDWARE FORCE-CLEAR] Evicting address {hex(raw_address_int)} from driver context..."
    )

    try:
        # 1. Load the core low-level CUDA Driver API library (Linux fallback)
        try:
            cuda_driver = ctypes.CDLL("libcuda.so.6")
        except OSError:
            cuda_driver = ctypes.CDLL("libcuda.so")  # Alternate system path fallback

        # 2. Reconstruct the raw void pointer from the integer key
        void_ptr = ctypes.c_void_p(raw_address_int)

        # 3. Invoke cuMemHostUnregister directly to break the lazy driver lock
        # Status code 0 indicates absolute success
        status = cuda_driver.cuMemHostUnregister(void_ptr)

        if status == 0:
            main_app_logger.info(
                f" [SUCCESS] Address {hex(raw_address_int)} successfully purged from driver context."
            )
        else:
            # Code 101/713 typically means it was already lazily swept by an OS page update
            main_app_logger.info(
                f" [INFO] Driver returned status code {status} for address {hex(raw_address_int)}. Cleared."
            )

    except Exception as error:
        main_app_logger.info(
            f" [FAILED] Driver force-clear encountered an error: {error}"
        )
