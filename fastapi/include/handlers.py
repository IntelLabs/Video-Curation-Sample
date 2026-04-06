import asyncio
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import numpy as np

# Force OpenCV to use a single thread for its operations.
# This prevents internal OpenCV threads from "racing" against your AI logic.
cv2.setNumThreads(1)
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from fastapi import FastAPI

# Global lock for thread-safe access to the shared YOLO model
model_lock = threading.Lock()

# Create a global lock for stream management
stream_lock = asyncio.Lock()

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#     # "rtsp_transport;tcp|hwaccel;cuda|threads;2|probesize;32|analyzeduration;0"
#     # "rtsp_transport;tcp|hwaccel;cuda|threads;4|probesize;5000000|analyzeduration;5000000"
#      "rtsp_transport;udp|hwaccel;cuda|threads;8"
#     "|stimeout;5000000|listen_timeout;5000" # Add timeouts to prevent hanging
# )
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|hwaccel;cuda|threads;auto|low_delay;1|probesize;5000000"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;1|probesize;32|analyzeduration;0"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;2|probesize;32|analyzeduration;0"
)


# from process_stream import extract_metadata_from_results, release_clip_and_reencode, retry_query
from include.utils import (
    CODE_DIR,
    CUSTOM_MODEL_FLAG,
    DEBUG,
    DEBUG_FLAG,
    DETECTION_THRESHOLD,
    DEVICE,
    # MODEL_H,
    MODEL_NAME,
    # MODEL_PRECISION,
    # MODEL_W,
    OMIT_DETECTIONS_FLAG,
    TARGET_FPS,
    YOLO_CLASS_NAMES,
    PipelineMapping,
    draw_label,
    filter_contained_boxes,
    get_detection_color,
    get_display_frame_in_bytes,
    manual_fps_calculation,
    merge_boxes_limit,
    metadata2vdms,
)

# ----- SETUP LOGGING -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
main_app_logger = logging.getLogger()


# ----- SPECIAL VARIABLES -----
MODEL_MAX_BATCH_SIZE = 64
MODEL_PRECISION = "FP16"
MODEL_W, MODEL_H = (640, 640)
# MODEL_W, MODEL_H = (1280, 1280)
CLIP_DURATION = 10  # seconds
KERNEL_RATIO = 0.05  # 0.03 # .05  # .025
MASK_MAX_VALUE = 255
MASK_THRESHOLD_VALUE = 127
MAX_DETECTIONS = 100
MAX_WORKERS = 4
# DISPLAY_FRAME_SIZE = (640, 360)
# DISPLAY_FRAME_QUALITY = 80
DISPLAY_FRAME_SIZE = (960, 540)
DISPLAY_FRAME_QUALITY = 50
ENABLE_QUERYING = False
return_bytes = True  # True, False

if CUSTOM_MODEL_FLAG:
    model_path = f"{CODE_DIR}/resources/models/ultralytics/custom_models/{MODEL_NAME}"
else:
    model_path = f"{CODE_DIR}/resources/models/ultralytics/{MODEL_NAME}/{MODEL_PRECISION}/{MODEL_NAME}"
    # model_path = f"{CODE_DIR}/{MODEL_NAME}"

if DEVICE == "GPU":
    model_path += ".engine"
    # 1. Force PyTorch to initialize the CUDA context
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    model_path += "_openvino_model/"

# ----- GLOBAL VARIABLES -----
manager = None  # Manager()
local_processes = {}
all_metadata = {}  # manager.dict()
send_metadata_queue = queue.Queue()  # manager.Queue()


# ----- INGESTION FUNCTIONS -----
# method to create clips (read frame write to file; add name to list)
def send_metadata():
    global all_metadata
    clip_filename = ""
    clip_key = ""
    width = 0
    height = 0
    while True:
        try:
            queue_details = send_metadata_queue.get()
            if queue_details is None:
                break

            (clip_key, clip_filename, width, height) = queue_details

            metadata2vdms(
                clip_key,
                clip_filename,
                all_metadata[clip_key],
                width,
                height,
            )
            del all_metadata[clip_key]

        except queue.Empty:
            pass


def handle_done(future):
    try:
        future.result()
    except Exception as e:
        print(f"Task error: {e}")


def save_and_finalize_clip(
    clip_key,
    _out_vid,
    clip_filename,
    tmp_file,
    target_fps,
    frame_width,
    frame_height,
):
    if DEBUG == "1":
        print(
            f"[TIMING],start_release_clip,{clip_key},{time.time()}",
            flush=True,
        )
    _out_vid.release()
    if DEBUG == "1":
        print(
            f"[TIMING],end_release_clip,{clip_key},{time.time()}",
            flush=True,
        )

    # Re-encode video in order to seek via ffmpeg later
    GENERAL_OPTS = "-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -flush_packets 0"  #  -filter:v fps={target_fps}
    CONVERSION = f"-c:v libx264 -preset ultrafast -filter:v fps=fps={target_fps}"  # "-c:v libx264 -preset medium"
    reencode_cmd = f"ffmpeg -y -i {tmp_file} {GENERAL_OPTS} {CONVERSION} -crf 23 -c:a copy {clip_filename}"
    cmd_list = shlex.split(reencode_cmd)
    if DEBUG == "1":
        print(
            f"[TIMING],start_reencode,{clip_key},{time.time()}",
            flush=True,
        )
    subprocess.run(cmd_list, check=True)
    end_time = time.time()
    # filename = str(Path(clip_filename).name)
    if DEBUG == "1":
        print(
            f"[TIMING],end_reencode,{clip_key},{end_time}",
            flush=True,
        )
        print(f"[TIMING],Save clip,{clip_key},{end_time}", flush=True)
    os.remove(tmp_file)

    send_metadata_queue.put(
        (
            clip_key,
            clip_filename,
            frame_width,
            frame_height,
        )
    )


async def auto_cleanup_janitor(app):
    while True:
        await asyncio.sleep(10)
        now = time.time()

        async with stream_lock:
            # Iterating over a list of keys to avoid "dictionary changed size" error
            for name in list(app.state.active_streams.keys()):
                streamer = app.state.active_streams.get(name)
                if not streamer:
                    continue

                backlog = streamer.get_executor_backlog()

                # Check if the stream is marked inactive OR timed out
                # streamer.active should be False when the video source ends
                is_stale = now - streamer.last_heartbeat > 30

                should_remove = False

                if not streamer.active and backlog == 0:
                    should_remove = True  # Video ended naturally
                elif is_stale and backlog == 0:
                    should_remove = True  # Browser tab closed/Network lost
                elif now - streamer.last_heartbeat > 90:
                    should_remove = True  # Hard timeout for hung processes

                if should_remove:
                    async with stream_lock:
                        if DEBUG == "1":
                            print(f"CLEANUP: Removing {name} from active_streams")
                        streamer.stop()
                        app.state.active_streams.pop(name, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    if not hasattr(app.state, "active_streams"):
        app.state.active_streams = {}
        app.state.status = "Ready"
        app.state.model = YOLO(model_path, verbose=False, task="detect")
        # app.state.model_lock = threading.Lock()
    device_input = "cuda" if DEVICE == "GPU" else "cpu"
    print("Starting shared model warmup...")
    dummy_input = torch.zeros((1, 3, MODEL_H, MODEL_W)).to(device_input)
    for _ in range(20):
        _ = app.state.model(dummy_input, verbose=False)

    janitor_task = asyncio.create_task(auto_cleanup_janitor(app))

    if DEBUG == "1":
        print(f"--- APP STARTUP | PID: {os.getpid()} | STATE READY ---")

    yield

    # --- CLEANUP ---
    janitor_task.cancel()

    # for s in app.state.active_streams.values():
    #     s.stop()

    async with stream_lock:
        for name, streamer in list(app.state.active_streams.items()):
            print(f"Shutting down stream: {name}")
            streamer.stop()  # Custom stop method defined below
            app.state.active_streams.pop(name, None)
    app.state.status = "Stopped"


class HybridReader:
    """
    Decouples frame acquisition from processing.
    Uses a background thread to ingest frames into a small deque,
    preventing OpenCV buffer lag.
    """

    def __init__(self, source, target_fps=TARGET_FPS):
        self.source = str(source)
        self.cap = self._create_capture()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Force low latency

        self.frame_queue = deque(maxlen=5)  # Keep queue small to stay "real-time"
        self.stopped = False
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.device = DEVICE  # Global from include.utils

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def stop(self):
        """Cleanly stop the reader and release resources."""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
        self.frame_queue.clear()
        # Optionally join if want to ensure the thread is dead
        # self.thread.join(timeout=1.0)

    def _create_capture(self):
        """Creates a VideoCapture with stable RTSP options."""
        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

    def update(self):
        """
        Continuously grabs frames. Throttles local files to maintain
        the target FPS and manages RTSP reconnections.
        """
        retry_attempt = 0
        max_retries = 10
        is_network_stream = "://" in self.source  # Detect if it's RTSP
        last_frame_time = time.perf_counter()

        while not self.stopped:
            # Grab frame from buffer
            if not self.cap.grab():
                if not is_network_stream:
                    self.stopped = True
                    break

                # --- RECONNECTION LOGIC ---
                retry_attempt += 1
                if retry_attempt > max_retries:
                    print(f"❌ [RTSP] Max retries reached for {self.source}. Stopping.")
                    self.stopped = True
                    break

                # Exponential Backoff: Wait 2s, 4s, 8s... up to 30s
                wait_time = min(2**retry_attempt, 30)
                # print(f"⚠️ [RTSP] Connection lost. Retry {retry_attempt}/{max_retries} in {wait_time}s...")

                self.cap.release()
                time.sleep(wait_time)
                self.cap = self._create_capture()
                continue

            # Throttle ingestion for local files to match real-time cadence
            elapsed = time.perf_counter() - last_frame_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

            last_frame_time = time.perf_counter()
            success, frame = self.cap.retrieve()

            if success:
                retry_attempt = 0  # Reset retries on successful frame
                self.frame_queue.append(frame)

                # CPU/GPU Specific Handling
                # if self.device == "GPU":
                #     # Keep as-is for DMA upload
                #     self.frame_queue.append(frame)
                # else:
                #     # For CPU: Downscale immediately to save AI thread work
                #     # This is the BIGGEST FPS gain for CPU mode
                #     # small_frame = cv2.resize(frame, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)
                #     # self.frame_queue.append(small_frame)
                #     self.frame_queue.append(frame)

                # last_frame_time = time.time()

    def read(self):
        return self.frame_queue.popleft() if self.frame_queue else None


class BaseHandler:
    """
    Core handler for camera metadata, hardware resource allocation,
    and the common AI processing pipeline (BGS and YOLO).
    """

    def __init__(self, source, name, active_streams, **kwargs):
        target_fps = kwargs.get("target_fps", TARGET_FPS)
        self.model = kwargs.get("model")

        if not self.model:
            self.model = YOLO(model_path, verbose=False, task="detect")
            self.model_warmup()

        if hasattr(self.model, "names"):
            self.label_source = []
            for k, v in self.model.names.items():
                self.label_source.append(v)
        else:
            self.label_source = YOLO_CLASS_NAMES

        self.name = name
        self.source = source
        self.active = True
        self.active_streams = active_streams
        self.frame_ready_event = asyncio.Event()
        self.loop = asyncio.get_event_loop()

        # Initialize hardware capture and determine stream properties
        self.get_valid_video_capture()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        self.get_fps_and_framecnt(target_fps)
        self.get_frameWH()

        # Configure scaling for 8K-to-Model coordinate mapping
        self.resize_h, self.resize_w = [MODEL_H, MODEL_W]
        self.scale_x = self.frame_width / MODEL_W
        self.scale_y = self.frame_height / MODEL_H
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.numFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine minimum contour size relative to frame resolution
        self.min_contour_area = int(
            # (0.005 * self.frame_width) * (0.005 * self.frame_height)
            # (0.005 * self.resize_w) * (0.005 * self.resize_h)
            (0.01 * self.resize_w) * (0.01 * self.resize_h)
        )  # 207

        # Performance Tracking
        self.frame_count = 0  # Frame count for videos
        self.stat_frame_count = 0
        self.stat_fps = 0
        self.latest_processed_frame = None
        self.last_frame_id = 0
        self.last_delivered_frame_id = -1  # Track what was actually sent

        # Video Clipping
        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1, mp4v
        self.clip_id = 0
        self.clip_filename = ""
        self.clip_key = ""
        self.tmp_file = ""

        # Default Kernels
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dilate_kernel_for_enhanced_mask = np.ones((21, 21), np.uint8)

        # Device based setup
        if DEVICE == "GPU":
            self.prepare_gpu_pipeline()
            if len(self.active_streams) == 0:
                self.warmup()
        else:
            self.operation_device_map = PipelineMapping(
                detection_device="cpu"
            )  # No CUDA HERE
            self.prepare_cpu_pipeline()

        # Start dedicated inference thread and timers
        self.stat_start_time = time.perf_counter()
        self.last_heartbeat = time.time()
        self.setup_threads()

    def get_valid_video_capture(self, connection_timeout=180):
        # Robust Capture Setup with 3-minute Retry Logic
        # connection_timeout = 180  # 3 minutes in seconds
        start_connect_time = time.time()
        retry_interval = 5  # Wait 5s between attempts

        self.cap = None
        print(f"📡 [CONNECTING] {self.name} | Source: {self.source}")

        while time.time() - start_connect_time < connection_timeout:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                # Quick check: can we actually grab a frame?
                ret, _ = self.cap.read()
                if ret:
                    print(f"✅ [CONNECTED] {self.name} established successfully.")
                    break

            # If we get here, connection failed (e.g., 404 Not Found)
            self.cap.release()
            print(
                f"⚠️ [RETRYING] {self.name} | Stream not ready, retrying in {retry_interval}s..."
            )
            time.sleep(retry_interval)

        # Final Connection Check
        if not self.cap or not self.cap.isOpened():
            print(
                f"❌ [FAILED] {self.name} could not connect after 3 minutes. Aborting."
            )
            self.active = False
            return  # Exit early to prevent downstream FPS 0.0 crashes

    def setup_threads(self):
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.process_thread = threading.Thread(
            target=self.run_realtime_inference, daemon=True
        )

    def start(self):
        self.process_thread.start()

    def allocate_cpu(self, bkgd_mask_queue_size=3):
        # pass
        self.resized_frame = np.zeros((3, self.resize_h, self.resize_w), dtype="uint8")
        # cv2.cuda.createContinuous(
        #     self.resize_h, self.resize_w, cv2.CV_8UC3
        # )

        self.fgMask = np.zeros(
            (self.resize_h, self.resize_w), dtype="uint8"
        )  # For resize
        self.prev_bkgd = np.ones((self.resize_h, self.resize_w), dtype="uint8")

        # bkgd_mask_queue_size = 3
        self.mask_history = deque(maxlen=bkgd_mask_queue_size)
        self.mask_history.append(self.prev_bkgd)

    def prepare_cpu_pipeline(self, method="knn"):
        self.operation_device_map = PipelineMapping()  # "full_cpu"
        self.device_input = self.operation_device_map.detection_device

        self.allocate_cpu()

        # Subtraction
        if method == "knn":
            history = 300  # int(5 * self.target_fps)
            background_thresh = 350
            NSamples = 10
            kNNSamples = 2
            self.lr = -1  # .01  #-1  # 0.001  #1 / (5 * self.target_fps)  # -1  # 0.01  # 1 / history

            self.backSub = cv2.createBackgroundSubtractorKNN(
                history=history,  # default 500
                dist2Threshold=background_thresh,  # default 400
                detectShadows=False,  # default True
            )
            self.backSub.setkNNSamples(kNNSamples)
            self.backSub.setNSamples(NSamples)
        elif method == "mog2":
            history = int(2 * self.target_fps)
            background_thresh = 10
            self.lr = 0.001

            self.backSub = cv2.createBackgroundSubtractorMOG2(
                history=history,  # default 500
                varThreshold=background_thresh,  # default 16
                detectShadows=False,  # default True
            )
        else:
            raise ValueError(f"Provided method ({method}) is not available.")

    def cleanup_cpu(self):
        """
        Purges large 8K NumPy buffers and CPU-based AI resources.
        """
        # Nullify specific class references to allow Garbage Collection
        self.executor = None
        self.reader = None
        self.latest_processed_frame = None

        # 1. Clear the Ping-Pong buffers (up to 200MB of RAM)
        if hasattr(self, "encode_buffers"):
            self.encode_buffers.clear()

        # 2. Clear the 10s video clip buffer
        if hasattr(self, "frame_buffer"):
            self.frame_buffer.clear()

        # 3. Explicitly nullify large arrays to trigger Garbage Collection
        self.resized_frame = None
        self.fgMask = None
        self.prev_bkgd = None

        # 4. Clear the BGS history
        if hasattr(self, "mask_history"):
            self.mask_history.clear()

    def allocate_gpu(self, bkgd_mask_queue_size=3):
        """
        Allocates persistent GpuMat buffers and CUDA streams to
        enable zero-copy GPU processing.
        """
        self.stream = cv2.cuda.Stream()
        self.gpu_fullres_frame = cv2.cuda.GpuMat(
            self.frame_height, self.frame_width, cv2.CV_8UC3
        )
        self.resized_frame = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3)
        self.fgMask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        self.prev_bkgd = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        self.prev_bkgd.setTo((255,))
        self.mask_history = deque(maxlen=bkgd_mask_queue_size)
        self.mask_history.append(self.prev_bkgd)
        self.gpu_threshold_dst_frame = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )
        self.gpu_morphed_frame = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )

        # Create continuous buffers to prevent stride artifacts during 8K downloads
        self.pinned_downloaded_resizedframe_np = cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC3
        )
        self.pinned_downloaded_frame_np = cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )
        cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1, self.gpu_threshold_dst_frame
        )
        cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1, self.gpu_morphed_frame
        )

        # self.fgMask = cv2.cuda.GpuMat(
        #     self.resize_h, self.resize_w, cv2.CV_8UC1
        # )  # For resize
        # # self.fgMask = cv2.cuda.createContinuous(
        # #     self.resize_h, self.resize_w, cv2.CV_8UC1
        # # )

        # self.prev_bkgd = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        # self.prev_bkgd.setTo((255,))

        # self.mask_history = deque(maxlen=bkgd_mask_queue_size)
        # self.mask_history.append(self.prev_bkgd)

        # self.gpu_threshold_dst_frame = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        # cv2.cuda.createContinuous(self.resize_h, self.resize_w, cv2.CV_8UC1, self.gpu_threshold_dst_frame)
        # self.gpu_threshold_dst_frame = cv2.cuda.createContinuous(self.resize_h, self.resize_w, cv2.CV_8UC1)

        # self.gpu_morphed_frame = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        # cv2.cuda.createContinuous(self.resize_h, self.resize_w, cv2.CV_8UC1, self.gpu_morphed_frame)
        # self.gpu_morphed_frame = cv2.cuda.createContinuous(
        #     self.resize_h, self.resize_w, cv2.CV_8UC1
        # )

        # self.pinned_downloaded_frame_np = cv2.cuda.createContinuous(
        #     self.resize_h, self.resize_w, cv2.CV_8UC1
        # )

    def prepare_gpu_pipeline(self):
        self.operation_device_map = PipelineMapping(
            resize_device="gpu",
            bkgd_subtraction_device="gpu",
            threshold_device="gpu",
            erodeAndDilate_device="gpu",
            detection_device="gpu",
        )  # rbtd_detection_gpu

        self.device_input = "cuda"

        self.allocate_gpu()

        # Subtraction
        history = int(2 * self.target_fps)  # 300  # int(5 * self.target_fps)
        self.lr = 0.001
        background_thresh = 10  # 350
        # self.lr = (
        #     -1
        # )  # .01  #-1  # 0.001  #1 / (5 * self.target_fps)  # -1  # 0.01  # 1 / history
        # bkgd_mask_queue_size = 3
        self.backSub = cv2.cuda.createBackgroundSubtractorMOG2(
            history=history,  # Clear ghosts of fast drones in ~2 seconds (2*fps)
            varThreshold=background_thresh,  # High threshold to ignore "shimmer" and compression noise  # default 16
            detectShadows=False,  # default True
        )

        self.dilate_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8U, self.dilate_kernel
        )
        self.dilate_filter_for_enhanced_mask = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8UC1, self.dilate_kernel_for_enhanced_mask
        )

    def cleanup_gpu(self):
        """
        Explicitly releases all GPU-allocated memory to prevent
        VRAM leaks in 8K concurrent streams.
        """
        # Iterate through class attributes to explicitly release VRAM.
        for attr_name in list(self.__dict__.keys()):
            attr_value = getattr(self, attr_name)

            # 2. Check if the attribute is a GpuMat
            if isinstance(attr_value, cv2.cuda.GpuMat):
                # 🏎️ Force the NVIDIA driver to deallocate this specific memory segment
                attr_value.release()
                setattr(self, attr_name, None)
                print(f"✅ Released GpuMat: {attr_name}")

        if hasattr(self, "gpu_fullres_frame") and self.gpu_fullres_frame is not None:
            try:
                self.gpu_fullres_frame.release()
            except Exception:
                self.gpu_fullres_frame = None

        if hasattr(self, "gpu_encoder_8k_buf") and self.gpu_encoder_8k_buf is not None:
            try:
                self.gpu_encoder_8k_buf.release()
            except Exception:
                self.gpu_encoder_8k_buf = None

        if hasattr(self, "gpu_display_frame") and self.gpu_display_frame is not None:
            try:
                self.gpu_display_frame.release()
            except Exception:
                self.gpu_display_frame = None

        self.pinned_downloaded_resizedframe_np = None
        self.gpu_threshold_dst_frame = None
        self.gpu_morphed_frame = None
        self.pinned_downloaded_frame_np = None

        # 3. Handle specific buffers (like your Ping-Pong lists)
        if hasattr(self, "encode_buffers"):
            self.encode_buffers.clear()

        # 4. Clear the BGS history
        if hasattr(self, "mask_history"):
            self.mask_history.clear()

        # 4. Optional: Final flush of the CUDA caching allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warmup(self):
        # WARM UP (Crucial for first-run latency)
        # JIT kernels are compiled on the first call
        h, w = self.resize_h, self.resize_w

        self.gpu_warmup_input_frame_np = cv2.cuda.createContinuous(h, w, cv2.CV_8UC3)

        if self.gpu_warmup_input_frame_np is not None:
            self.gpu_warmup_input_frame_np[:] = [255, 0, 0]

            gpu_warmup_input_frame = cv2.cuda.GpuMat(h, w, cv2.CV_8U)
            gpu_warmup_input_frame.upload(self.gpu_warmup_input_frame_np)
            cv2.cuda.createContinuous(h, w, cv2.CV_8U, gpu_warmup_input_frame)

            # Trigger compiler
            gpu_warmup_frame = cv2.cuda.GpuMat(h, w, cv2.CV_8U)
            cv2.cuda.createContinuous(h, w, cv2.CV_8U, gpu_warmup_frame)

            # cv2.cuda.cvtColor(
            #     gpu_warmup_input_frame,
            #     cv2.COLOR_BGR2GRAY,
            #     stream=self.stream,
            #     dst=gpu_warmup_frame,
            # )
            cv2.cuda.resize(
                gpu_warmup_input_frame,
                (self.resize_w, self.resize_h),
                stream=self.stream,
                dst=gpu_warmup_frame,
                interpolation=cv2.INTER_NEAREST,
            )
            # Thresholding
            gpu_threshold_dst_frame = cv2.cuda.GpuMat(h, w, cv2.CV_8U)
            cv2.cuda.createContinuous(h, w, cv2.CV_8U, gpu_threshold_dst_frame)
            cv2.cuda.threshold(
                gpu_warmup_frame,
                MASK_THRESHOLD_VALUE,
                MASK_MAX_VALUE,
                cv2.THRESH_BINARY,
                gpu_threshold_dst_frame,
                self.stream,
            )

            gpu_morphed_frame = cv2.cuda.GpuMat(h, w, cv2.CV_8U)
            cv2.cuda.createContinuous(h, w, cv2.CV_8U, gpu_morphed_frame)
            self.dilate_filter.apply(
                gpu_threshold_dst_frame, gpu_morphed_frame, self.stream
            )
            self.stream.waitForCompletion()

    def model_warmup(self):
        """Run warmup in a separate thread to prevent FastAPI lockup."""

        def _warmup(iterations=5):
            with model_lock:  # Use the global lock
                print(f"Starting warmup for {self.name}...")
                dummy_input = torch.zeros((1, 3, self.resize_h, self.resize_w)).to(
                    self.device_input
                )
                for _ in range(iterations):
                    _ = self.model(dummy_input, verbose=False)
                print(f"Warmup complete for {self.name}")

        # Run in the background so the dashboard loads instantly
        threading.Thread(target=_warmup, daemon=True).start()

    def get_executor_backlog(self):
        """Returns the number of tasks currently waiting in the thread pool queue."""
        # access the internal queue of the executor
        return self.executor._work_queue.qsize()

    def stop(self):
        self.active = False  # Signals the while loops to exit

        # Release the VideoWriter if it exists
        # if ENABLE_QUERYING and self.video_writer:
        #     # self.release_clip_and_reencode()
        #     self.save_and_finalize_clip(
        #         self.clip_key,
        #         self.video_writer,
        #         self.clip_filename,
        #         self.tmp_file,
        #         self.target_fps,
        #         MODEL_W,
        #         MODEL_H,
        #     )

        # Close the OpenCV capture
        if self.cap:
            self.cap.release()

        # Join threads if you want to be 100% sure they are closed
        # self.update_thread.join(timeout=1.0)
        # self.process_thread.join(timeout=1.0)

    def apply_background_subtraction_cpu(
        self, include_history=True, method="and", stream=None
    ):
        self.fgMask = self.backSub.apply(
            self.cpu_resized_frame, learningRate=float(self.lr)
        )

        if include_history:
            # If this is the first run, clone the mask instead of ANDing with an empty/white buffer
            # if len(self.mask_history) < 1:
            #     self.prev_bkgd.setTo(0, stream)  # Clear the initial white buffer

            for m in list(self.mask_history):
                # Dilate the historical mask on CPU
                dilated = cv2.dilate(
                    m, self.dilate_kernel_for_enhanced_mask, iterations=1
                )

                if method == "or":
                    # Bitwise OR on CPU
                    cv2.bitwise_or(self.prev_bkgd, dilated, dst=self.prev_bkgd)
                else:
                    # Bitwise AND on CPU
                    cv2.bitwise_and(self.prev_bkgd, dilated, dst=self.prev_bkgd)

            self.mask_history.append(self.fgMask.copy())

            if (
                self.prev_bkgd.max() != self.prev_bkgd.min()
                and self.prev_bkgd.max() > 0
            ):
                combined_mask_bool = (self.fgMask > 0) | (self.prev_bkgd > 0)
                self.fgMask = combined_mask_bool.astype(np.uint8) * 255

    def apply_background_subtraction_gpu(
        self, include_history=True, method="and", stream=None
    ):
        self.fgMask = self.backSub.apply(
            self.resized_frame, float(self.lr), stream=stream
        )

        if include_history:
            # If this is the first run, clone the mask instead of ANDing with an empty/white buffer
            if len(self.mask_history) < 1:
                self.prev_bkgd.setTo(0, stream)  # Clear the initial white buffer

            for m in list(self.mask_history):
                # Dilate the historical mask on GPU
                dilated = self.dilate_filter_for_enhanced_mask.apply(m, stream=stream)

                if method == "or":
                    # Bitwise OR on GPU
                    cv2.cuda.bitwise_or(
                        self.prev_bkgd, dilated, self.prev_bkgd, stream=stream
                    )
                else:
                    # Bitwise AND on GPU
                    cv2.cuda.bitwise_and(
                        self.prev_bkgd, dilated, self.prev_bkgd, stream=stream
                    )

            self.mask_history.append(self.fgMask.clone())
            min_val, max_val, _, _ = cv2.cuda.minMaxLoc(self.prev_bkgd)

            if max_val != min_val and max_val > 0:
                self.fgMask = cv2.cuda.bitwise_or(
                    self.fgMask, self.prev_bkgd, stream=stream
                )

    def check_disk_usage(self, path, min_gb=0.5):
        """Returns True if there is at least min_gb available at path."""
        try:
            total, used, free = shutil.disk_usage(path)
            # Convert bytes to Gigabytes
            free_gb = free / (2**30)
            return free_gb > min_gb
        except Exception as e:
            print(f"Disk check error: {e}")
            return False

    # Gets video fps and framecount
    def get_fps_and_framecnt(self, target_fps):
        self.input_fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # hardware fps
        # print(f"in fps: {sself.input_fps} target fps: {target_fps}")
        if self.input_fps == 0:  # Case when FPS isn't available
            self.input_fps = manual_fps_calculation(self.source, num_frames=10)
            print(f"new in fps: {self.input_fps}")

        self.target_fps = (
            target_fps
            if target_fps not in [None, 0] and self.input_fps > target_fps
            else self.input_fps
        )
        print(f"in fps: {self.input_fps} self.target fps: {self.target_fps}")

        self.frame_skip = int(self.input_fps / self.target_fps)
        if self.frame_skip < 1:
            self.frame_skip = 1
        self.skip_count = self.frame_skip - 1

        self.MAX_FRAMES_PER_CLIP = int(self.target_fps * CLIP_DURATION)
        self.target_interval = 1.0 / self.target_fps  # 0.0666s

        if DEBUG == "1":
            print(f"FPS of {self.name} input stream: {self.input_fps}", flush=True)
            print(f"FPS of {self.name} output mp4: {self.target_fps}", flush=True)

    # Gets frame W and H details
    def get_frameWH(self):
        input_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (input_height * input_width) < (MODEL_H * MODEL_W):
            new_sizeHW = check_imgsz([MODEL_H, MODEL_W])  # expects hxw
        else:
            new_sizeHW = check_imgsz([input_height, input_width])  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]
        self.height = new_sizeWH[1]

    def update_frame(self):
        self.stat_frame_count += 1
        elapsed = time.perf_counter() - self.stat_start_time
        if elapsed > 0.5:
            self.stat_fps = round(self.stat_frame_count / elapsed, 1)

    def run_model(self, frame, batch=1, device_input="cuda", stream=True):
        results = self.model.predict(
            frame,
            imgsz=(self.resize_h, self.resize_w),
            batch=batch,
            device=device_input,
            verbose=False,
            stream=stream,
            max_det=MAX_DETECTIONS,
        )
        return results

    # Main inference loop
    def run_realtime_inference(self):
        pass


class VideoStreamHandler_WIP(BaseHandler):
    """
    Advanced handler optimized for 8K resolution at 15FPS.
    Implements Ping-Pong buffering and background JPEG encoding to bypass the GIL.
    """

    def __init__(self, source, name, active_streams, **kwargs):
        """
        Initializes the 8K pipeline and pre-allocates isolated memory buffers.

        Args:
            source (str): The RTSP URL or local file path.
            name (str): Unique identifier for the stream.
            active_streams (dict): Global dictionary tracking all running handlers.
        """
        # Initialize BaseHandler to set up cap, frame dimensions, and model
        super().__init__(source, name, active_streams, **kwargs)
        self.reader = HybridReader(source=self.source, target_fps=self.target_fps)

        # Isolated memory buffers to prevent the Producer loop from overwriting
        # frames currently being encoded by the background worker.
        self.encode_buffers = [
            np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8),
            np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8),
        ]
        self.buf_idx = 0

        # 3. Enhanced synchronization for 8K/15FPS
        self.frame_buffer = deque(
            maxlen=self.MAX_FRAMES_PER_CLIP
        )  # 10s buffer for video clips
        self.buffer_lock = threading.Lock()
        self.disp_w, self.disp_h = DISPLAY_FRAME_SIZE

        if DEVICE == "GPU":
            # This prevents the AI thread from overwriting the encoder's data.
            self.gpu_encoder_8k_buf = cv2.cuda.createContinuous(
                self.frame_height, self.frame_width, cv2.CV_8UC3
            )

            # Continuous allocation prevents stride/padding artifacts
            self.gpu_display_frame = cv2.cuda.createContinuous(
                self.disp_h, self.disp_w, cv2.CV_8UC3
            )

            # Create a dedicated background stream for encoding tasks
            self.encode_stream = cv2.cuda.Stream()

    def setup_threads(self):
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)  # 1)
        self.process_thread = threading.Thread(
            target=self.run_realtime_inference, daemon=True
        )

    def start(self):
        """
        Starts the decoupled ingestion and inference threads in the correct order.
        """
        # 1. Start the hardware-decoupled reader first
        self.reader.start()

        # 2. Small delay to allow the reader's deque to populate
        time.sleep(0.1)

        # 3. Start the main inference producer loop
        if not self.process_thread.is_alive():
            self.process_thread.start()

        return self

    def stop(self):
        """
        Comprehensive resource release. Stops threads, shuts down the pool,
        and purges VRAM to prevent leaks in concurrent 8K environments.
        """
        # Signal threads to stop
        self.active = False

        # Stop reader thread first
        if hasattr(self, "reader"):
            self.reader.stop()

        # if self.process_thread.is_alive():
        #     self.process_thread.join(timeout=1.0)

        # Shutdown the executor to stop background JPEG encoding
        # wait=True ensures no 'zombie' threads are left accessing GpuMats
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        if self.cap:
            self.cap.release()
            self.cap = None

        # Purge HW Buffers
        if DEVICE == "GPU":
            self.cleanup_gpu()
        else:
            self.cleanup_cpu()

        # Final Reset of the FastAPI event
        self.frame_ready_event.set()  # Unblock any generators waiting on this stream

    def async_yolo_task(self, data):
        """
        Orchestrates the AI pipeline for a single frame.

        Steps:
        1. (if GPU) Synchronizes CUDA stream and downloads the mask.
        2. Executes contour-based YOLO detection.
        3. Handoffs the frame to the background JPEG encoder.
        """
        # return_bytes=False
        try:
            frameNum = data["frameNum"]

            # Use pre-allocated pinned memory from BaseHandler for 8K mask download
            if self.device_input == "cuda":
                self.stream.waitForCompletion()
                # Ensure the download uses the instance-specific CUDA stream
                self.pinned_downloaded_frame_np = data["mask"].download(self.stream)
                # data["mask"].download(self.stream, self.pinned_downloaded_frame_np)

            # Run contour-based YOLO logic and draw overlays
            frame_bytes = self.contour2predictions(
                frameNum,
                self.pinned_downloaded_frame_np
                if self.device_input == "cuda"
                else data["mask"],
                data["full_frame"],
                device_input=self.device_input,
                repeat_count=data["repeat_count"],
                return_bytes=return_bytes,
            )

            # Thread-safe update of the latest frame for FastAPI StreamingResponse
            if return_bytes and frameNum > self.last_delivered_frame_id:
                if frame_bytes:
                    self.latest_processed_frame = frame_bytes
                    self.last_delivered_frame_id = frameNum
                self.last_frame_id = frameNum
                self.last_heartbeat = time.time()
                # Signal the FastAPI generator that a new frame is ready
                self.loop.call_soon_threadsafe(self.frame_ready_event.set)

            # Offload JPEG encoding to a background worker to release the GIL
            if not return_bytes:
                # Rotate between buffers so the encoder has a 'locked' memory space
                self.buf_idx = (self.buf_idx + 1) % 2
                target_buf = self.encode_buffers[self.buf_idx]

                # Perform a deep memory copy into the isolated buffer
                # np.copyto(target_buf, data["full_frame"])
                # np.copyto(target_buf, data["full_frame"], casting='unsafe')
                # target_buf[:] = data["full_frame"]
                target_buf[:] = np.ascontiguousarray(data["full_frame"])

                # Submit to background worker to bypass the GIL
                self.executor.submit(self._encode_and_signal, target_buf, frameNum)

        except Exception:
            logging.error(f"Async YOLO Error in {self.name}: {traceback.format_exc()}")

    def _encode_and_signal(self, pixels, frame_num):
        """Worker task for JPEG encoding to bypass GIL during stream delivery."""
        if not pixels.any():
            logging.warning(
                f"⚠️ [DEBUG] {self.name} | Frame {frame_num}: Buffer is ALL BLACK (0.0). Check memory copy."
            )

        # Downscale for display FIRST to make encoding faster
        # display_frame = cv2.resize(pixels, DISPLAY_FRAME_SIZE, interpolation=cv2.INTER_NEAREST)
        if DEVICE == "GPU":
            # 🚀 GPU OPTIMIZED PATH: Avoids CPU-RAM bus saturation
            self.gpu_encoder_8k_buf.upload(pixels, stream=self.encode_stream)

            # Downscale for dashboard BEFORE downloading to CPU
            cv2.cuda.resize(
                self.gpu_encoder_8k_buf,
                DISPLAY_FRAME_SIZE,
                stream=self.encode_stream,
                dst=self.gpu_display_frame,
            )

            # Ensure resize is complete before CPU download
            self.encode_stream.waitForCompletion()
            display_frame = self.gpu_display_frame.download(self.encode_stream)
        else:
            # CPU Fallback
            display_frame = cv2.resize(
                pixels, DISPLAY_FRAME_SIZE, interpolation=cv2.INTER_NEAREST
            )

        # Standard JPEG compression
        success, buffer = cv2.imencode(
            ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, DISPLAY_FRAME_QUALITY]
        )
        if not success:
            logging.error(
                f"❌ [DEBUG] {self.name} | Frame {frame_num}: JPEG Encoding Failed."
            )
            return

        # Update state and signal FastAPI
        self.latest_processed_frame = buffer.tobytes()
        self.last_delivered_frame_id = frame_num
        self.last_frame_id = frame_num
        self.last_heartbeat = time.time()

        # Thread-safe event set for the FastAPI loop
        self.loop.call_soon_threadsafe(self.frame_ready_event.set)

    def update_ui_fallback(self, frame, frame_num):
        # FALLBACK: If AI is busy, worker thread encodes raw frame for the UI.
        # This offloads the 40ms CPU cost from the main Producer loop.
        display_frame = cv2.resize(
            frame, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR
        )
        _, buffer = cv2.imencode(
            ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, DISPLAY_FRAME_QUALITY]
        )

        # print(f"DEFAULT DISP frameNum/last_frame_id {frame_num} self.last_delivered_frame_id: {self.last_delivered_frame_id}", flush=True)  #\n\tframe_bytes: {frame_bytes}", flush=True)
        self.latest_processed_frame = buffer.tobytes()
        self.last_delivered_frame_id = frame_num
        self.last_frame_id = frame_num
        self.last_heartbeat = time.time()
        # Signal the FastAPI generator that a new frame is ready
        self.loop.call_soon_threadsafe(self.frame_ready_event.set)

    def process_frame_async(self, frame, frame_num, repeat_count=1):
        """Worker task: Handles BGS, YOLO, or Raw Fallback."""
        try:
            # Check if we should run AI or just provide a raw preview to keep 15 FPS
            # Use a backlog of 2 to allow for slight GPU jitter
            if self.get_executor_backlog() < 5:  # 2:
                if self.device_input == "cuda":
                    inf_data = self.test_rbtdc_detection_gpu_optimized3(
                        frame, frame_num, repeat_count=repeat_count
                    )
                else:
                    inf_data = self.test_full_cpu_detection_gpu(
                        frame, frame_num, repeat_count=repeat_count
                    )

                if inf_data:
                    self.async_yolo_task(inf_data)
                    return  # Exit after successful AI update

            # else:
            #     # FAST FALLBACK PATH: Update UI with raw frame to keep 15 FPS
            #     self.update_ui_fallback(frame, frame_num)

        except Exception:
            logging.error(f"Pipeline failure for {self.name}: {traceback.format_exc()}")

    def run_realtime_inference(self):
        """
        Main producer loop. Fetches frames from the reader and
        dispatches them to the AI pipeline.
        """
        while self.active:
            # Get frame from the reader's deque
            frame = self.reader.read()

            if frame is not None:
                self.frame_count += 1

                # Re-allocate only if needed, otherwise pass pointer
                with self.buffer_lock:
                    self.frame_buffer.append(frame.copy())

                # Offload to AI or Fallback
                self.executor.submit(self.process_frame_async, frame, self.frame_count)
                self.update_frame()
                self.last_heartbeat = time.time()

                # Only submit to AI if the executor is not overwhelmed.
                # 8K frames are ~100MB each; a backlog of 5 = 500MB RAM usage.
                # if self.get_executor_backlog() < 5:
                #     self.executor.submit(
                #         self.process_frame_async,
                #         frame,
                #         self.frame_count
                #     )
                # else:
                #     # 🏎️ FALLBACK: AI is busy. Push raw frame to UI immediately.
                #     # This ensures the 'Live' view stays at 15 FPS.
                #     self.update_ui_fallback(frame, self.frame_count)

                # self.update_frame()
            else:
                # Prevent CPU pinning if camera is slow
                time.sleep(0.001)
        self.stop()

    def test_rbtdc_detection_gpu_optimized3(self, frame, frameNum, repeat_count=1):
        """
        GPU-Accelerated Motion Detection Pipeline (Producer).

        This function performs high-speed background subtraction (BGS) on a downscaled
        version of the 8K frame to identify regions of interest (ROIs).

        Args:
            frame (np.ndarray): The raw 8K input frame.
            frameNum (float): Chronological timestamp or ID.

        Returns:
            dict: Contains the frame ID, the GPU-resident motion mask, and the original 8K frame.
        """
        stream = self.stream
        # Resize directly into the pre-allocated Pinned Memory
        # This avoids a temporary CPU allocation
        # H, W = self.resize_h, self.resize_w
        # self.cpu_resized_frame = cv2.resize(frame, (W, H))
        # self.video_writer.write(self.cpu_resized_frame)

        # Upload 8K frame to GPU memory
        self.gpu_fullres_frame.upload(frame, stream=stream)

        # Downscale to MODEL_W/H (e.g., 640x640) for fast BGS analysis
        cv2.cuda.resize(
            self.gpu_fullres_frame,
            (self.resize_w, self.resize_h),
            stream=stream,
            dst=self.resized_frame,
            interpolation=cv2.INTER_NEAREST,
        )

        if ENABLE_QUERYING and self.video_writer:  # and not self.video_queue.full():
            self.pinned_downloaded_resizedframe_np = self.resized_frame.download(stream)
            # self.resized_frame.download(self.stream, self.pinned_downloaded_resizedframe_np)
            for _ in range(repeat_count):
                # self.video_queue.put((self.video_writer, self.pinned_downloaded_resizedframe_np.copy()))
                self.video_writer.write(self.pinned_downloaded_resizedframe_np)

        # Apply Background Subtraction on GPU
        self.apply_background_subtraction_gpu(
            include_history=True, method="and", stream=stream
        )

        # Clean up the motion mask using Thresholding and Morphology (Dilation)
        cv2.cuda.threshold(
            self.fgMask,
            MASK_THRESHOLD_VALUE,
            MASK_MAX_VALUE,
            cv2.THRESH_BINARY,
            dst=self.gpu_threshold_dst_frame,
            stream=stream,
        )
        self.dilate_filter.apply(
            self.gpu_threshold_dst_frame, dst=self.gpu_morphed_frame, stream=stream
        )

        return {
            "frameNum": frameNum,
            "mask": self.gpu_morphed_frame,  # GpuMat pointer to cleaned mask
            "full_frame": frame,  # Kept for high-res cropping
            "repeat_count": repeat_count,
        }

    def test_full_cpu_detection_gpu(self, frame, frameNum, repeat_count=1):
        """
        CPU-Based Motion Detection Pipeline (Producer).

        Performs background subtraction on the CPU to identify moving objects.
        Ideal for saving VRAM or for environments without high-end NVIDIA GPUs.

        Args:
            frame (np.ndarray): The raw 8K input frame.
            frameNum (float): Unique ID for the current frame.

        Returns:
            dict: Motion data containing the frame ID, CPU-based mask, and original 8K frame.
        """
        # Resize the 8K frame to a smaller 'model' size (e.g., 640x640)
        # Using INTER_NEAREST as it is the fastest CPU interpolation method.
        H, W = self.resize_h, self.resize_w
        self.cpu_resized_frame = cv2.resize(
            frame, (W, H), interpolation=cv2.INTER_NEAREST
        )
        if ENABLE_QUERYING:
            for _ in range(repeat_count):
                self.video_writer.write(self.cpu_resized_frame)

        # Apply Background Subtraction on CPU
        self.apply_background_subtraction_cpu(include_history=True, method="and")

        # Clean up the motion mask using Thresholding and Morphology (Dilation)
        _, mask = cv2.threshold(
            self.fgMask, MASK_THRESHOLD_VALUE, MASK_MAX_VALUE, cv2.THRESH_BINARY
        )
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)

        return {
            "frameNum": frameNum,
            "mask": mask,
            "full_frame": frame,  # Kept for high-res cropping
            "repeat_count": repeat_count,
        }

    def get_detections_for_contours_bbs(
        self,
        frameNum,
        foi,
        contours,
        thickness=2,
        device_input="cuda",
        return_bytes=True,
    ):
        """
        Motion-Triggered YOLO Inference Logic.

        Instead of running YOLO on a massive 8K frame, this function:
        1. Extracts bounding boxes from the motion contours.
        2. Merges nearby boxes into optimal 640x640 crops.
        3. Runs a single batch inference on those crops.
        4. Maps detection coordinates back to the original 8K space.

        Args:
            foi (np.ndarray): 'Frame of Interest' (the 8K raw frame).
            contours (list): Contours extracted from the motion mask.
        """
        # global active_streams
        # source = self.source
        stream_name = self.name
        num_objs = 0
        # predictions = []
        metadata = dict()
        # frame_bytes = 'b'
        cropped_imgs, cropped_coords = [], []
        H, W = foi.shape[:2]  # Unpack once
        bbs_full_res = []

        if not contours:
            # if return_bytes:
            frame_bytes = get_display_frame_in_bytes(
                foi,
                display_size=DISPLAY_FRAME_SIZE,
                quality=DISPLAY_FRAME_QUALITY,
                return_bytes=return_bytes,
            )
            return metadata, frame_bytes  # num_objs, predictions

        # Filter small noise and convert contours to 8K-space bounding boxes
        raw_bbs = []
        padding = 64
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.min_contour_area:
                x1, y1, w, h = cv2.boundingRect(c)

                # Scale coordinates from 640p BGS-space to 8K-space
                xx1 = max(0, int((x1 * self.scale_x)) - padding)
                yy1 = max(0, int((y1 * self.scale_y)) - padding)
                xx2 = min(W, int(((x1 + w) * self.scale_x)) + padding)
                yy2 = min(H, int(((y1 + h) * self.scale_y)) + padding)
                raw_bbs.append([area, [xx1, yy1, xx2, yy2]])

        # Merge overlapping boxes into batches (Capped at 64 for TensorRT stability)
        bbs_full_res = sorted(
            [pair[1] for pair in raw_bbs],
            key=lambda x: x[0],
            reverse=True,
        )  # [:MAX_DETECTIONS]

        dist_thresh = min(0.05 * W, 0.05 * H)
        merged = merge_boxes_limit(
            bbs_full_res, dist_threshold=dist_thresh, size_limit=MODEL_W
        )
        merged = filter_contained_boxes(merged, containment_thresh=0.9)

        # Extract crops at full-resolution
        for x1, y1, x2, y2 in merged:
            if (
                x2 > x1
                and y2 > y1
                and (x2 - x1) < self.frame_width
                and (y2 - y1) < self.frame_height
            ):
                crop = foi[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] > 31 and crop.shape[1] > 31:
                    cropped_imgs.append(crop)
                    cropped_coords.append((x1, y1))

            if len(cropped_imgs) == MODEL_MAX_BATCH_SIZE:
                # logging.warning(
                #     f"⚠️ [LIMIT] {self.name} found {len(cropped_imgs)} contours. Capping to 64 for TensorRT."
                # )
                # cropped_imgs = cropped_imgs[:MODEL_MAX_BATCH_SIZE]
                # cropped_coords = cropped_coords[:MODEL_MAX_BATCH_SIZE]
                break

        if not cropped_imgs:
            frame_bytes = get_display_frame_in_bytes(
                foi,
                display_size=DISPLAY_FRAME_SIZE,
                quality=DISPLAY_FRAME_QUALITY,
                return_bytes=return_bytes,
            )
            return metadata, frame_bytes  # num_objs, predictions

        # if len(cropped_imgs) > MODEL_MAX_BATCH_SIZE:
        #     logging.warning(
        #         f"⚠️ [LIMIT] {self.name} found {len(cropped_imgs)} contours. Capping to 64 for TensorRT."
        #     )
        #     cropped_imgs = cropped_imgs[:MODEL_MAX_BATCH_SIZE]
        #     cropped_coords = cropped_coords[:MODEL_MAX_BATCH_SIZE]

        # Run Inference (Keep stream=False as it is stable)
        with model_lock:  # Use the global lock
            results = self.model.predict(
                cropped_imgs,
                imgsz=MODEL_W,
                batch=len(cropped_imgs),
                device=device_input,
                verbose=False,
                stream=True,
                max_det=MAX_DETECTIONS,
                # classes=[0],  # only "person",
                # conf=0.45,
            )
            # Convert generator to list while still inside the lock
            # to ensure results aren't overwritten by another thread.
            results = list(results)

        # Process results and draw 8K-space overlays
        for ridx, r in enumerate(results):
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Move to CPU in one bulk operation per crop
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            off_x, off_y = cropped_coords[ridx]

            for j in range(len(boxes)):
                num_objs += 1
                bx1, by1, bx2, by2 = boxes[j]
                abs_x1, abs_y1 = off_x + bx1, off_y + by1
                abs_x2, abs_y2 = off_x + bx2, off_y + by2
                class_id = clss[j]
                class_name = self.label_source[class_id]
                confidence = confs[j]
                if confidence > DETECTION_THRESHOLD:
                    if not OMIT_DETECTIONS_FLAG:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        print(
                            # f"[OBJECT DETECTION] {class_name} detected in frame {frameNum} (Total detected: {current_cnt})",
                            f"[{timestamp}] {stream_name} DETECTION on Frame {frameNum}: {class_name} detected",
                            flush=True,
                        )

                    bb_color = get_detection_color(class_id, is_bgr=True)

                    foi = cv2.rectangle(
                        foi,
                        (abs_x1, abs_y1),
                        (abs_x2, abs_y2),
                        bb_color,
                        thickness,
                    )
                    label = f"{class_name} {confidence:.2f}"
                    draw_label(foi, label, (abs_x1, abs_y1), color=bb_color, padding=5)

                    height = min(abs_y2, H) - max(0, abs_y1)
                    width = min(abs_x2, W) - max(0, abs_x1)
                    # object_res = [
                    #     abs_x1,
                    #     abs_y1,
                    #     height,
                    #     width,
                    #     class_name,
                    #     confidence,
                    #     H,
                    #     W,
                    # ]

                    # Resized
                    scale_x = self.resize_w / W
                    scale_y = self.resize_h / H
                    object_res = [
                        int(abs_x1 * scale_x),
                        int(abs_y1 * scale_y),
                        int(height * scale_y),
                        int(width * scale_x),
                        class_name,
                        confidence,
                        int(self.resize_h),
                        int(self.resize_w),
                    ]

                    framenum_str = f"{frameNum:04d}_{j:04d}"
                    if DEBUG_FLAG:
                        meta_str = ",".join(
                            [str(o) for o in object_res + [framenum_str]]
                        )
                        print(f"[{stream_name} METADATA],{meta_str}", flush=True)

                    # Full Res
                    metadata[framenum_str] = {
                        "frameId": frameNum,
                        "bbId": framenum_str,
                        "bbox": {
                            "x": int(object_res[0]),
                            "y": int(object_res[1]),
                            "height": int(object_res[2]),
                            "width": int(object_res[3]),
                            "object": str(object_res[4]),
                            "object_det": {
                                "confidence": float(object_res[5]),
                                "frameH": int(object_res[6]),
                                "frameW": int(object_res[7]),
                            },
                        },
                    }

        # Queue frame for display (reduce quality for 8K bandwidth)
        frame_bytes = get_display_frame_in_bytes(
            foi,
            display_size=DISPLAY_FRAME_SIZE,
            quality=DISPLAY_FRAME_QUALITY,
            return_bytes=return_bytes,
        )

        return metadata, frame_bytes

    def contour2predictions(
        self,
        frameNum,
        mask,
        frame,
        device_input="cpu",
        repeat_count=1,
        return_bytes=True,
    ):
        """
        The 'Glue' function that connects Motion Detection to AI Inference.

        Args:
            mask (GpuMat/np.ndarray): The motion mask (from BGS).
            frame (np.ndarray): The original 8K frame.
        """
        # Extract contours from the motion mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Write frame
        # self.video_writer.write(frame)
        # if self.video_writer:
        #     for _ in range(repeat_count):
        # self.video_writer.write(self.cpu_resized_frame)

        # Pass contours to the YOLO detection logic
        metadata = dict()
        metadata, frame_bytes = self.get_detections_for_contours_bbs(
            frameNum,
            frame,
            contours,
            thickness=2,
            device_input=device_input,
            return_bytes=return_bytes,
        )

        # Update global metadata for storage (Database/JSON)
        if metadata:
            all_metadata.setdefault(
                self.clip_key,
                {
                    "object": {},
                    "face": {},
                },
            )
            all_metadata[self.clip_key]["object"].update(metadata)

        # frame_bytes returned even if no metadata available
        return frame_bytes
