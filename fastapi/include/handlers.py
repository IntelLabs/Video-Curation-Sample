import asyncio
import json
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
from pathlib import Path

import cv2
import numpy as np
import torch
from include.utils import (
    # MODEL_H,
    # MODEL_PRECISION,
    # MODEL_W,
    CODE_DIR,
    CUSTOM_MODEL_FLAG,
    DEBUG,
    DEBUG_FLAG,
    DETECTION_THRESHOLD,
    DEVICE,
    MODEL_NAME,
    OMIT_DETECTIONS_FLAG,
    SHARED_OUTPUT,
    YOLO_CLASS_NAMES,
    PipelineMapping,
    draw_label,
    filter_contained_boxes,
    get_detection_color,
    get_display_frame_in_bytes,
    manual_fps_calculation,
    merge_boxes_limit,
    metadata2vdms_with_retry,
)
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from fastapi import FastAPI

# ----- SETUP LOGGING -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
main_app_logger = logging.getLogger()


# ----- PIPELINE CONFIGURATION -----
# Force OpenCV to use a single thread for its operations.
# This prevents internal OpenCV threads from "racing" against AI logic.
cv2.setNumThreads(1)

# Model expected imgsz and size to resize images
MODEL_W, MODEL_H = (640, 640)  # (1280, 1280)
MODEL_PRECISION = "FP16"
SHARED_MODEL = False

# Maximum allowed batch sixe for inference ("GPU": tensorRT)
MODEL_MAX_BATCH_SIZE = 64

# Maximum detections returned from model
MAX_DETECTIONS = 100

# Framerate for metadata extraction and videos used for querying
TARGET_FPS = os.getenv("TARGET_FPS", 15)

# Duration of video clips in seconds which are used for querying
CLIP_DURATION = os.getenv("CLIP_DURATION", 10)

# Bounding boxes returned from pipeline
# object (includes yolo), motion (includes bbs no yolo)
DETECTION_TYPE = "object"

# Frame size and quality for displaying frames in browser
DISPLAY_FRAME_QUALITY = 50  # 80
DISPLAY_FRAME_SIZE = (960, 540)  # (640, 360)
RETURN_BYTES = True  # True, False
THICKNESS = 2

# Flag for enabling querying
# False: Detection only
# True: Include saving video clips, sending metadata to VDMS
ENABLE_QUERYING = False  # True, False

# Values used for OpenCV thresholding
MASK_MAX_VALUE = 255
MASK_THRESHOLD_VALUE = 127


# Pixels added to each dimension of bounding boxes in full resolution image
RAW_BB_FULL_RES_PADDING = 10  # 64

# Contour Cleaning: Maximum size of merged boxes
MERGE_SIZE_LIMIT = MODEL_W  # MODEL_W ,960

# Number workers used for ThreadPoolExecutor
MAX_WORKERS = 4

# Optimizes RTSP ingestion with hardware acceleration and low-delay flags
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|hwaccel;cuda|threads;auto|low_delay;1|probesize;5000000"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;1|probesize;32|analyzeduration;0"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;2|probesize;32|analyzeduration;0"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;2|probesize;32|analyzeduration;0"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;4|probesize;5000000|analyzeduration;5000000"
    #  "rtsp_transport;udp|hwaccel;cuda|threads;8|stimeout;5000000|listen_timeout;5000"
)


# ----- VARIABLE ADJUSTMENTS -----
CLIP_DURATION = None if CLIP_DURATION == "None" else CLIP_DURATION

if DETECTION_TYPE == "motion" and ENABLE_QUERYING:
    ENABLE_QUERYING = False
    DISPLAY_FRAME_QUALITY = 100
    THICKNESS = 10

if CUSTOM_MODEL_FLAG:
    model_path = f"{CODE_DIR}/resources/models/ultralytics/custom_models/{MODEL_NAME}"
else:
    model_path = f"{CODE_DIR}/resources/models/ultralytics/{MODEL_NAME}/{MODEL_PRECISION}/{MODEL_NAME}"
    # model_path = f"{CODE_DIR}/{MODEL_NAME}"

if DEVICE == "GPU":
    model_path += ".engine"
    # Force PyTorch to initialize the CUDA context
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    model_path += "_openvino_model/"


# ----- GLOBAL VARIABLES -----
if ENABLE_QUERYING:
    # Tracks all metadata
    all_metadata = {}

    # Tracks clip_filename once video re-encoded
    # video_ready_list = {}

    # Queue for metadata being sent to vdms
    send_metadata_queue = queue.Queue()

    # Tracks if both components are finished:
    #   {"clip_name": {"video": bool, "meta": bool}}
    clip_completion_tracker = {}


# ----- FASTAPI APPLICATION STARTUP/SHUTDOWN -----
# The lifespan parameter handles startup and shutdown
async def auto_cleanup_janitor(app):
    while True:
        await asyncio.sleep(10)
        now = time.time()

        # --- Stream Monitoring ---
        async with app.state.stream_lock:
            # Iterating over a list of keys to avoid "dictionary changed size" error
            for name, streamer in list(app.state.active_streams.items()):
                # streamer = app.state.active_streams.get(name)
                if not streamer:
                    continue

                ai_backlog = streamer.get_executor_backlog()
                video_backlog = streamer.write_queue.qsize() if ENABLE_QUERYING else 0
                io_backlog = (
                    streamer.io_executor._work_queue.qsize()
                    if hasattr(streamer, "io_executor")
                    else 0
                )

                # Check if the stream is marked inactive OR timed out
                # streamer.active should be False when the video source ends
                is_stale = now - streamer.last_heartbeat > 30

                should_remove = False

                if not streamer.active and (
                    ai_backlog == 0 and video_backlog == 0 and io_backlog == 0
                ):
                    should_remove = True  # Video ended naturally
                elif is_stale and (
                    ai_backlog == 0 and video_backlog == 0 and io_backlog == 0
                ):
                    should_remove = True  # Browser tab closed/Network lost
                elif now - streamer.last_heartbeat > 90:
                    should_remove = True  # Hard timeout for hung processes

                if should_remove:
                    async with app.state.stream_lock:
                        if DEBUG == "1":
                            print(f"CLEANUP: Removing {name} from active_streams")
                        streamer.stop()
                        app.state.active_streams.pop(name, None)

        # --- Synchronization Data Purge ---
        if ENABLE_QUERYING:
            # Remove trackers older than 5 minutes (300s)
            stale_keys = [
                k
                for k, v in clip_completion_tracker.items()
                if (now - v.get("start", now)) > 300
            ]
            for k in stale_keys:
                clip_completion_tracker.pop(k, None)
                all_metadata.pop(k, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    if not hasattr(app.state, "active_streams"):
        app.state.active_streams = {}

    app.state.status = "Ready"
    app.state.stream_lock = asyncio.Lock()
    if SHARED_MODEL:
        app.state.model = YOLO(model_path, verbose=False, task="detect")

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
    async with app.state.stream_lock:
        for name, streamer in list(app.state.active_streams.items()):
            print(f"Shutting down stream: {name}")
            streamer.stop()  # Custom stop method defined below
            app.state.active_streams.pop(name, None)
    app.state.status = "Stopped"


# ----- INGESTION FUNCTIONS -----
def save_and_finalize_clip(
    clip_key,
    _out_vid,
    clip_filename,
    tmp_file,
    target_fps,
    frame_width,
    frame_height,
    clip_metadata,
    frame_in_clip_count,
):
    # global video_ready_list
    if DEBUG == "1":
        print(
            f"[TIMING],start_release_clip,{clip_key},{time.time()}",
            flush=True,
        )

    if _out_vid is not None:
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

    # Re-encode command with background-friendly flags
    # 1. nice -n 19: Lowers CPU priority to avoid stutters in the main app
    # 2. -threads 2: Limits core usage
    # 3. -preset ultrafast: Fastest possible h264 encoding
    # 4. -crf 28: Slightly higher than default (23) for even less CPU work
    # reencode_cmd = (
    #     f"nice -n 19 ffmpeg -y -i {tmp_file} "
    #     f"-threads 2 -c:v libx264 -preset ultrafast -crf 28 "
    #     f"-filter:v fps=fps={target_fps} -c:a copy -tune zerolatency "
    #     f"-hide_banner -loglevel error {clip_filename}"
    # )

    try:
        cmd_list = shlex.split(reencode_cmd)

        if DEBUG == "1":
            print(
                f"[TIMING],start_reencode,{clip_key},{time.time()}",
                flush=True,
            )

        subprocess.run(cmd_list, check=True)
        end_time = time.time()

        if DEBUG == "1":
            print(
                f"[TIMING],end_reencode,{clip_key},{end_time}",
                flush=True,
            )
            print(f"[TIMING],Save clip,{clip_key},{end_time}", flush=True)

        # Mark video as ready
        # video_ready_list[clip_filename] = frame_in_clip_count

        # Signal tracker
        check_and_dispatch_to_vdms(
            clip_filename, frame_width, frame_height, component="video"
        )

        # Cleanup the temporary RAM-disk file immediately
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    except Exception as e:
        print(f" [ERROR] Clip finalization failed for {clip_key}: {e}")


def send_metadata():
    """
    Consumer thread that sends metadata to VDMS.
    If retries fail, it saves the data to a local JSON 'dead-letter' file.
    """
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

            # # (clip_key, clip_filename, width, height, clip_metadata) = queue_details
            (clip_filename, width, height) = queue_details
            clip_key = Path(clip_filename).name
            clip_metadata = all_metadata.pop(clip_key, None)

            if clip_metadata:
                success = metadata2vdms_with_retry(
                    clip_key,
                    clip_filename,
                    clip_metadata,
                    width,
                    height,
                )

                # CUSTOM ERROR HANDLER: Final Failure Fallback
                if not success:
                    error_path = f"{CODE_DIR}/failed_metadata/{clip_key}.json"
                    os.makedirs(os.path.dirname(error_path), exist_ok=True)

                    with open(error_path, "w") as f:
                        json.dump(
                            {
                                "clip_filename": clip_filename,
                                "width": width,
                                "height": height,
                                "metadata": clip_metadata,
                                "failed_at": datetime.now().isoformat(),
                            },
                            f,
                        )

                    main_app_logger.error(
                        f" [CRITICAL] Permanent VDMS failure. Data saved to: {error_path}"
                    )

                send_metadata_queue.task_done()
            else:
                main_app_logger.error(
                    f" [MISSING] Metadata for {clip_key} was lost before upload!"
                )

        except Exception as e:
            # pass
            print(f"Exception occurred in send_metadata: {e}")


def check_and_dispatch_to_vdms(clip_filename, width, height, component):
    """
    Synchronizes AI and Video threads. Logs which component finished first.
    """
    clip_key = Path(clip_filename).name

    # Initialize tracker if first time seeing this clip
    if clip_key not in clip_completion_tracker:
        clip_completion_tracker[clip_key] = {
            "video": False,
            "meta": False,
            "start": time.time(),
        }

    tracker = clip_completion_tracker[clip_key]
    tracker[component] = True

    # Identify the bottleneck
    if tracker["video"] and tracker["meta"]:
        total_wait = time.time() - tracker["start"]
        main_app_logger.info(
            f" [SYNC] {clip_key} Fully Ready. Total processing time: {total_wait:.2f}s"
        )
        send_metadata_queue.put((clip_filename, width, height))
        clip_completion_tracker.pop(clip_key, None)
    else:
        other_component = "meta" if component == "video" else "video"
        main_app_logger.info(
            f" [WAIT] {clip_key}: {component} finished. Waiting for {other_component}..."
        )


# ----- PIPELINE CLASSES -----
class HybridReader:
    """
    Decouples frame acquisition from processing.
    Uses a background thread to ingest frames into a small deque,
    preventing OpenCV buffer lag.
    """

    def __init__(self, source, target_fps=TARGET_FPS, clip_duration=CLIP_DURATION):
        self.source = str(source)
        self.cap = self._create_capture(target_fps, clip_duration)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Force low latency
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

        # self.frame_queue = deque(maxlen=5)  # Keep queue small to stay "real-time"
        self.frame_queue = queue.Queue(maxsize=5)
        self.stopped = False
        self.device = DEVICE  # Global from include.utils
        self.frame_idx = 0
        self.target_frame_idx = 0
        # self.frame_queue = queue.Queue(maxsize=30)

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def stop(self):
        """Cleanly stop the reader and release resources."""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

        # self.frame_queue.clear()
        # Optionally join if want to ensure the thread is dead
        # self.thread.join(timeout=1.0)

    def _create_capture(self, target_fps, clip_duration):
        """Creates a VideoCapture with stable RTSP options."""
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.get_fps_and_framecnt(cap, target_fps, clip_duration)
        return cap

    def get_fps_and_framecnt(self, cap, target_fps, clip_duration):
        self.input_fps = int(cap.get(cv2.CAP_PROP_FPS))  # hardware fps
        # print(f"in fps: {sself.input_fps} target fps: {target_fps}")
        if self.input_fps == 0:  # Case when FPS isn't available
            self.input_fps = manual_fps_calculation(self.source, num_frames=10)
            print(f"new in fps: {self.input_fps}")

        self.target_fps = (
            target_fps
            if target_fps not in [None, 0] and self.input_fps > target_fps
            else self.input_fps
        )

        self.frame_skip = int(self.input_fps / self.target_fps)
        if self.frame_skip < 1:
            self.frame_skip = 1
        # self.skip_count = self.frame_skip - 1

        if clip_duration is None:
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            clip_duration = frame_count / self.input_fps
        self.max_frames_per_clip = int(self.target_fps * float(clip_duration))
        self.frame_interval = 1.0 / self.target_fps  # 0.0666s
        print(
            f"in fps: {self.input_fps} self.target fps: {self.target_fps} self.frame_skip: {self.frame_skip}"
        )

    # def update(self):
    #     """
    #     Continuously grabs frames. Throttles local files to maintain
    #     the target FPS and manages RTSP reconnections.
    #     """
    #     retry_attempt = 0
    #     max_retries = 10
    #     is_network_stream = "://" in self.source  # Detect if it's RTSP
    #     last_frame_time = time.perf_counter()

    #     while not self.stopped:
    #         # Grab frame from buffer
    #         if not self.cap.grab():
    #             if not is_network_stream:
    #                 self.stopped = True
    #                 break

    #             # --- RECONNECTION LOGIC ---
    #             retry_attempt += 1
    #             if retry_attempt > max_retries:
    #                 print(f"❌ [RTSP] Max retries reached for {self.source}. Stopping.")
    #                 self.stopped = True
    #                 break

    #             # Exponential Backoff: Wait 2s, 4s, 8s... up to 30s
    #             wait_time = min(2**retry_attempt, 30)
    #             # print(f"⚠️ [RTSP] Connection lost. Retry {retry_attempt}/{max_retries} in {wait_time}s...")

    #             self.cap.release()
    #             time.sleep(wait_time)
    #             self.cap = self._create_capture()
    #             continue

    #         # Throttle ingestion for local files to match real-time cadence
    #         elapsed = time.perf_counter() - last_frame_time
    #         if elapsed < self.frame_interval:
    #             time.sleep(self.frame_interval - elapsed)

    #         last_frame_time = time.perf_counter()
    #         success, frame = self.cap.retrieve()

    #         if success:
    #             retry_attempt = 0  # Reset retries on successful frame
    #             # self.frame_queue.append(frame)
    #             self.frame_queue.put(frame)

    #             # CPU/GPU Specific Handling
    #             # if self.device == "GPU":
    #             #     # Keep as-is for DMA upload
    #             #     self.frame_queue.append(frame)
    #             # else:
    #             #     # For CPU: Downscale immediately to save AI thread work
    #             #     # This is the BIGGEST FPS gain for CPU mode
    #             #     # small_frame = cv2.resize(frame, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)
    #             #     # self.frame_queue.append(small_frame)
    #             #     self.frame_queue.append(frame)

    #             # last_frame_time = time.time()

    def update(self):
        """
        Continuously grabs frames. Throttles local files to maintain
        the target FPS and manages RTSP reconnections.
        """
        # retry_attempt = 0
        # max_retries = 10
        # is_network_stream = "://" in self.source  # Detect if it's RTSP
        # last_frame_time = time.perf_counter()

        while not self.stopped:
            # Determine if current frame_idx should be "KEPT" or "SKIPPED"
            # to match the target cadence
            should_keep = int(self.frame_idx * self.target_fps / self.input_fps) > int(
                (self.frame_idx - 1) * self.target_fps / self.input_fps
            )

            if should_keep:
                # Fully decode this frame
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame_queue.put(frame)
                # self.target_frame_idx += 1
                # print(f"Target Frame {self.target_frame_idx} in queue\n", flush=True)
            else:
                # Fast-forward the pointer without decoding (minimal CPU)
                self.cap.grab()

            self.frame_idx += 1

    def read(self):
        # return self.frame_queue.popleft() if self.frame_queue else None
        try:
            # If the reader is stopped, don't wait a full second;
            # check immediately to speed up the "Draining" phase.
            wait_time = 0.1 if self.stopped else 1.0
            return self.frame_queue.get(timeout=wait_time)
        except queue.Empty:
            return None


class BaseHandler:
    """
    Core handler for camera metadata, hardware resource allocation,
    and the common AI processing pipeline (BGS and YOLO).
    """

    def __init__(self, source, name, active_streams, **kwargs):
        self.name = name
        self.source = source
        self.active = True
        self.active_streams = active_streams
        self.device_input = "cuda" if DEVICE == "GPU" else "cpu"
        self.disp_w, self.disp_h = DISPLAY_FRAME_SIZE

        target_fps = int(kwargs.get("target_fps", TARGET_FPS))
        clip_duration = kwargs.get("clip_duration", CLIP_DURATION)
        provided_model = kwargs.get("model")
        self.resize_h, self.resize_w = [MODEL_H, MODEL_W]

        if isinstance(provided_model, str) or provided_model is None:
            self.model = YOLO(model_path, verbose=False, task="detect")
            self.model_warmup()
        else:
            self.model = provided_model

        try:
            if hasattr(self.model, "names"):
                self.label_source = []
                for k, v in self.model.names.items():
                    self.label_source.append(v)
            else:
                self.label_source = YOLO_CLASS_NAMES
        except Exception:
            self.label_source = YOLO_CLASS_NAMES

        self.frame_ready_event = asyncio.Event()
        self.loop = asyncio.get_event_loop()
        self._is_stopped = False  # 🛡️ Shutdown guard
        self._stop_lock = threading.Lock()  # 🔒 Local lock for this instance

        # Initialize hardware capture and determine stream properties
        self.get_valid_video_capture()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        self.get_fps_and_framecnt(target_fps, clip_duration)
        self.get_frameWH()

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
        self.frame_in_clip_count = 0

        # Initialize Reader
        self.reader = HybridReader(source=self.source, target_fps=self.target_fps)

        if ENABLE_QUERYING:
            # Thread-safe queue for the resized frames (640x640)
            # maxlen=300 allows for a 20-second buffer in case of extreme disk lag
            # Non-blocking queue for frames and control signals
            self.write_queue = queue.Queue(maxsize=300)
            self.writer_done = False

        # Default Kernels
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dilate_kernel_for_enhanced_mask = np.ones((5, 5), np.uint8)  # (21, 21)

        # Device based setup
        if not SHARED_MODEL:
            self.model_warmup()

        if DEVICE == "GPU":
            self.prepare_gpu_pipeline()
            if len(self.active_streams) == 0:
                self.warmup()
        else:
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
        # Executor for Async YOLO tasks and FFmpeg re-encoding
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # Producer: Handles acquisition and AI metadata logs
        self.process_thread = threading.Thread(
            target=self.run_realtime_inference, daemon=True
        )

        if ENABLE_QUERYING:
            # NEW: Dedicated I/O pool for Disk/GPU transfers (Higher worker count for 8K)
            self.io_executor = ThreadPoolExecutor(max_workers=8)

            # Dedicated FFmpeg pool so re-encoding doesn't slow down live AI
            self.ffmpeg_executor = ThreadPoolExecutor(max_workers=2)

            # Sends metadata to VDMS
            self.metadata_thread = threading.Thread(target=send_metadata, daemon=True)

            # Consumer: Handles GPU-to-CPU download and Disk I/O (Writing resized frames to RAM disk)
            self.writer_thread = threading.Thread(
                target=self._video_writer, daemon=True
            )

    def start(self):
        """
        Starts the decoupled ingestion and inference threads in the correct order.
        """
        # Start the hardware-decoupled reader first
        self.reader.start()

        if ENABLE_QUERYING:
            self._initialize_writer()

        # Small delay to allow the reader's deque to populate
        time.sleep(0.1)

        # Start the producer and consumer threads
        if not self.process_thread.is_alive():
            self.process_thread.start()

        if ENABLE_QUERYING and not self.metadata_thread.is_alive():
            self.metadata_thread.start()

        if ENABLE_QUERYING and not self.writer_thread.is_alive():
            self.writer_thread.start()

        return self

    def stop(self):
        """
        Comprehensive resource release. Stops threads, shuts down the pool,
        and purges VRAM to prevent leaks in concurrent 8K environments.
        """
        # global video_ready_list
        with self._stop_lock:
            if self._is_stopped:
                return  # Already stopped by another thread

        # Signal threads to stop
        self.active = False
        main_app_logger.info(f" [STOP] Initiating shutdown for {self.name}")

        # Force the final clip to rotate even if under 10 seconds
        if ENABLE_QUERYING and self.video_writer:
            main_app_logger.info(f" [STOP] Forcing final clip rotation for {self.name}")
            # self.finalize_clip(
            #     self.clip_key,
            #     self.tmp_file,
            #     self.clip_filename,
            #     self.video_writer
            # )

            # Manually trigger metadata completion for the final (forced) clip
            if self.clip_key in all_metadata:
                tracker = clip_completion_tracker.setdefault(
                    self.clip_key, {"video": False, "meta": False}
                )
                tracker["meta"] = True
                check_and_dispatch_to_vdms(
                    self.clip_filename, self.resize_w, self.resize_h, component="meta"
                )

            clip_metadata = all_metadata.get(self.clip_key, {"object": {}, "face": {}})

            self.ffmpeg_executor.submit(
                save_and_finalize_clip,
                self.clip_key,
                self.video_writer,
                self.clip_filename,
                self.tmp_file,
                self.target_fps,
                self.resize_w,
                self.resize_h,
                clip_metadata,
                self.frame_in_clip_count,
            )
            self.video_writer = None

        # Close the OpenCV capture
        if self.cap:
            self.cap.release()
            self.cap = None

        # Stop reader thread
        if hasattr(self, "reader"):
            self.reader.stop()

        if ENABLE_QUERYING:
            if hasattr(self, "write_queue"):
                # Signal the writer queue to unblock the worker thread
                self.write_queue.put(None)

                # Wait for the _video_writer thread to finish writing remaining frames
                timeout = 10
                start_wait = time.time()
                while not self.writer_done and (time.time() - start_wait < timeout):
                    time.sleep(0.1)

        # Shutdown the executor to stop background JPEG encoding
        # wait=True ensures no 'zombie' threads are left accessing GpuMats
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        # Make sure all frames are flushed to RAM
        if hasattr(self, "io_executor"):
            self.io_executor.shutdown(wait=True)

        # Shutdown ffmpeg once disc files are ready
        if hasattr(self, "ffmpeg_executor"):
            self.ffmpeg_executor.shutdown(wait=True)

        # This ensures the next /dashboard_stats call won't see this stream
        if self.name in self.active_streams:
            self.active_streams.pop(self.name, None)

        # Unblock any waiting FastAPI generators
        self.frame_ready_event.set()

        # Purge HW Buffers
        self._is_stopped = True
        if DEVICE == "GPU":
            self.cleanup_gpu()
        else:
            self.cleanup_cpu()

        self._check_shm_safety(threshold_percent=0)  # Forced cleanup of current tmp
        main_app_logger.info(f" [STOP] {self.name} resources fully released.")

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

    def prepare_cpu_pipeline(self, method="mog2"):
        self.operation_device_map = PipelineMapping()  # "full_cpu"
        self.device_input = self.operation_device_map.detection_device

        self.allocate_cpu()

        # Subtraction
        if method == "knn":
            history = 300  # int(5 * self.target_fps)
            background_thresh = 350
            NSamples = 10
            kNNSamples = 2
            self.lr = 1 / history

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
            self.lr = 1 / history

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

        # Clear the Ping-Pong buffers (up to 200MB of RAM)
        # if hasattr(self, "encode_buffers"):
        #     self.encode_buffers.clear()

        # Explicitly nullify large arrays to trigger Garbage Collection
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
        self.lr = 1 / history
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
        # if hasattr(self, "encode_buffers"):
        #     self.encode_buffers.clear()

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
            # with model_lock:  # Use the global lock
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

    def apply_background_subtraction_cpu(self, include_history=True, method="and"):
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
    def get_fps_and_framecnt(self, target_fps, clip_duration):
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
        # self.skip_count = self.frame_skip - 1

        if clip_duration is None:
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            clip_duration = frame_count / self.input_fps
        self.max_frames_per_clip = int(self.target_fps * clip_duration)
        self.target_interval = 1.0 / self.target_fps  # 0.0666s

        if DEBUG == "1":
            print(f"FPS of {self.name} input stream: {self.input_fps}", flush=True)
            print(f"FPS of {self.name} output mp4: {self.target_fps}", flush=True)

    # Gets frame W and H details
    def get_frameWH(self):
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.numFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # input_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # input_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (self.frame_height * self.frame_width) < (MODEL_H * MODEL_W):
            new_sizeHW = check_imgsz([MODEL_H, MODEL_W])  # expects hxw
        else:
            new_sizeHW = check_imgsz(
                [self.frame_height, self.frame_width]
            )  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]
        self.height = new_sizeWH[1]

        # Configure scaling for 8K-to-Model coordinate mapping
        self.resize_h, self.resize_w = [MODEL_H, MODEL_W]
        self.scale_x = self.frame_width / MODEL_W
        self.scale_y = self.frame_height / MODEL_H

    def update_frame(self):
        self.stat_frame_count += 1
        elapsed = time.perf_counter() - self.stat_start_time
        if elapsed > 0.5:
            self.stat_fps = round(self.stat_frame_count / elapsed, 1)

    def run_model(
        self, frame, imgsz=(MODEL_H, MODEL_W), batch=1, device_input="cuda", stream=True
    ):
        results = self.model.predict(
            frame,
            imgsz=imgsz,
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

    def _initialize_writer(self):
        """Sets up a new VideoWriter on a RAM disk for near-zero latency."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # clip_filename has a unique name indicating clip # and timestamp
        # if "://" not in str(self.source):
        self.clip_filename = (
            f"{SHARED_OUTPUT}/{self.name}_{self.clip_id}_{timestamp}_{os.getpid()}.mp4"
        )
        # else:
        # self.clip_filename = f"{SHARED_OUTPUT}/{self.name}_{timestamp}.mp4"

        # Key used by all_metadata
        self.clip_key = Path(self.clip_filename).name

        # Temporary filename before re-encoding via ffmpeg
        self.tmp_file = f"/dev/shm/{self.clip_key}"

        # Initialize Video Writer
        self.video_writer = cv2.VideoWriter(
            self.tmp_file, self.fourcc, self.target_fps, (self.resize_w, self.resize_h)
        )
        if not self.video_writer.isOpened():
            print(f" [CRITICAL] VideoWriter failed to open for {self.tmp_file}!")

    def finalize_clip(self, clip_key, tmp_file, final_filename, video_writer):
        """Signals the worker to finalize; does NOT release the writer here."""
        if video_writer:
            writer_to_finalize = video_writer
            self.video_writer = None  # Producer will create new writer on next frame

            # Send a rotation signal that includes the writer object to be closed
            self.write_queue.put(
                {
                    "control": "ROTATE",
                    "old_key": clip_key,
                    "old_tmp": tmp_file,
                    "old_final": final_filename,
                    "writer_to_finalize": writer_to_finalize,
                }
            )

    def _write_video_frame(self, active_writer, frame_payload):
        # 1. Download from GPU if necessary
        # frame = (
        #     frame_payload.download()
        #     if hasattr(frame_payload, "download")
        #     else frame_payload
        # )
        # 1. Non-blocking download if using GPU
        if hasattr(frame_payload, "download"):
            # Use the dedicated encode_stream to prevent blocking the AI stream
            frame = frame_payload.download(self.encode_stream)
            # self.encode_stream.waitForCompletion()
        else:
            frame = frame_payload
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        # 2. FORCE RESIZE to match (resize_w, resize_h) exactly
        # This prevents the silent 0KB failure if dimensions are off by 1px
        h, w = frame.shape[:2]
        if w != int(self.resize_w) or h != int(self.resize_h):
            frame = cv2.resize(
                frame,
                (int(self.resize_w), int(self.resize_h)),
                interpolation=cv2.INTER_NEAREST,
            )
        active_writer.write(frame)

    def _video_writer(self):
        """
        Consumer: Writes frames to RAM disk and rotates files.
        Ensures all frames are flushed before FFmpeg starts re-encoding.
        """
        global all_metadata
        # Use a small separate pool for Disk I/O to avoid blocking the GIL
        while self.active or not self.write_queue.empty():
            try:
                data = self.write_queue.get(timeout=1)
                if data is None:
                    break

                # if data.get("control") == "ROTATE":
                #     self.finalize_clip(data["old_key"], data["old_tmp"], data["old_final"], data.get("writer_to_finalize"))
                #     continue
                if data.get("control") == "ROTATE":
                    # 1. Release the writer only AFTER all frames in queue are written
                    writer_to_close = data.get("writer_to_finalize")
                    # if writer_to_close:
                    #     # writer_to_close.release()
                    #     # Finalize in background so we can immediately start the next clip
                    #     self.io_executor.submit(writer_to_close.release)

                    # clip_metadata = all_metadata.pop(data["old_key"], {"object": {}, "face": {}})

                    # # 2. Submit the background re-encode task
                    # self.ffmpeg_executor.submit(
                    #     save_and_finalize_clip,
                    #     data["old_key"],
                    #     None,
                    #     data["old_final"],
                    #     data["old_tmp"],
                    #     self.target_fps,
                    #     self.resize_w,
                    #     self.resize_h,
                    #     clip_metadata
                    # )
                    # if writer_to_close:
                    clip_metadata = all_metadata.pop(
                        data["old_key"], {"object": {}, "face": {}}
                    )

                    def finalize_and_then_reencode(writer, info, meta):
                        try:
                            # First, force the OpenCV writer to close and flush to /dev/shm
                            if writer:
                                writer.release()

                            if (
                                os.path.exists(info["old_tmp"])
                                and os.path.getsize(info["old_tmp"]) > 0
                            ):
                                self.ffmpeg_executor.submit(
                                    save_and_finalize_clip,
                                    info["old_key"],
                                    None,
                                    info["old_final"],
                                    info["old_tmp"],
                                    self.target_fps,
                                    self.resize_w,
                                    self.resize_h,
                                    meta,
                                    info["frame_in_clip_count"],
                                )
                            else:
                                print(
                                    f" [ERROR] {info['old_key']} is empty. Skipping re-encode."
                                )
                        except Exception as e:
                            print(f" [CRITICAL] Clip finalization failed: {e}")

                    self.io_executor.submit(
                        finalize_and_then_reencode,
                        writer_to_close,
                        data,
                        clip_metadata,
                    )
                    self.write_queue.task_done()
                    continue

                frame_payload = data.get("frame")
                active_writer = data.get("writer_to_finalize")
                if frame_payload is not None and active_writer:
                    # Use a sequential write to prevent frames from overlapping during rotation
                    self._write_video_frame(active_writer, frame_payload)

                self.write_queue.task_done()
            except queue.Empty:
                continue
        self.writer_done = True


class VideoStreamHandler_WIP(BaseHandler):
    """
    Advanced handler optimized for 8K resolution at 15FPS (Target FPS).
    Implements Ping-Pong buffering and background JPEG encoding to bypass the GIL.
    Decouples AI logs from Disk I/O to maintain 15FPS and accurate clip indexing.
    """

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
        # If backlog is very high, drop JPEG quality to 25 to clear the 'pause' faster
        backlog = self.get_executor_backlog()
        adaptive_quality = (
            20 if backlog > (self.dynamic_limit * 2) else DISPLAY_FRAME_QUALITY
        )

        # FALLBACK: If AI is busy, worker thread encodes raw frame for the UI.
        # This offloads the 40ms CPU cost from the main Producer loop.
        display_frame = cv2.resize(
            frame, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR
        )
        _, buffer = cv2.imencode(
            ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, adaptive_quality]
        )

        # print(f"DEFAULT DISP frameNum/last_frame_id {frame_num} self.last_delivered_frame_id: {self.last_delivered_frame_id}", flush=True)  #\n\tframe_bytes: {frame_bytes}", flush=True)
        self.latest_processed_frame = buffer.tobytes()
        self.last_delivered_frame_id = frame_num
        self.last_frame_id = frame_num
        self.last_heartbeat = time.time()
        # Signal the FastAPI generator that a new frame is ready
        self.loop.call_soon_threadsafe(self.frame_ready_event.set)

    def _check_shm_safety(self, threshold_percent=90):
        """
        Scans /dev/shm and deletes the oldest .mp4 files if usage exceeds threshold.
        This prevents the 8K stream from crashing the entire container.
        """
        import shutil
        from pathlib import Path

        # 1. Check current usage of the RAM disk
        usage = shutil.disk_usage("/dev/shm")
        percent_used = (usage.used / usage.total) * 100

        if percent_used > threshold_percent:
            print(
                f" [CRITICAL] /dev/shm usage at {percent_used:.1f}%. Purging old clips..."
            )

            # 2. Get all .mp4 files in /dev/shm sorted by oldest first
            shm_path = Path("/dev/shm")
            clips = sorted(shm_path.glob("*.mp4"), key=lambda x: x.stat().st_mtime)

            # 3. Delete files until we are under 70% usage or run out of files
            for clip in clips:
                try:
                    # Don't delete the file the current writer is actively using!
                    if str(clip) == self.tmp_file:
                        continue

                    clip.unlink()
                    print(f" [PURGE] Deleted {clip.name} to free RAM.")

                    # Re-check usage after each deletion
                    usage = shutil.disk_usage("/dev/shm")
                    if (usage.used / usage.total) * 100 < 70:
                        break
                except Exception as e:
                    print(f" [ERROR] Could not purge {clip}: {e}")

    # ----- FRAME PROCESSING -----
    def process_frame_async(self, clip_filename, frame, frame_num, skip_ai=False):
        try:
            # 1. FORCE ORDER: Metadata must always be processed if querying is ON.
            # If querying is OFF, we can skip the heavy YOLO call, but we MUST
            # stay in this thread to keep the timeline consistent.
            inf_data = None
            if ENABLE_QUERYING or not skip_ai:
                if self.device_input == "cuda":
                    inf_data = self.test_rbtd_detection_gpu(frame)  # , frame_num)
                else:
                    inf_data = self.test_full_cpu(frame)  # , frame_num)

            # 2. UNIFIED SIGNALING:
            # Every 3rd frame (for 5 FPS display) must be encoded and signaled here.
            # This ensures AI frames and Raw frames never "jump" over each other.
            if inf_data:
                if ENABLE_QUERYING:
                    frame_2_write = (
                        self.resized_frame.clone()
                        if DEVICE == "GPU"
                        else self.cpu_resized_frame.copy()
                    )
                    # Video writing logic (already in your code)
                    self.write_queue.put(
                        {
                            "frame": frame_2_write,
                            "writer_to_finalize": self.video_writer,
                        }
                    )

                # If we are skipping AI for display fluidity, tell the task
                # to skip drawing boxes, but still process the metadata.
                inf_data["is_display_frame"] = frame_num % 3 == 0
                inf_data["suppress_boxes"] = skip_ai
                inf_data["frameNum"] = frame_num
                self.async_yolo_task(clip_filename, inf_data, RETURN_BYTES)

            elif frame_num % 3 == 0:
                # NO MOTION DETECTED FALLBACK:
                # We call update_ui_fallback from INSIDE this worker thread.
                # This preserves the exact queue order.
                self.update_ui_fallback(frame, frame_num)

        except Exception:
            logging.error(f"Pipeline failure for {self.name}: {traceback.format_exc()}")

    def run_realtime_inference(self):
        """Producer: Maintains the target FPS and updates clip IDs."""
        last_frame_time = time.perf_counter()
        while self.active:
            frame = self.reader.read()
            if frame is not None:
                self.frame_count += 1

                # Calculate a dynamic limit: tolerate 0.5 seconds of lag.
                # If target_fps is 15, the limit is 7. If target_fps is 30, the limit is 15.
                self.dynamic_limit = max(2, int(0.5 * self.target_fps))

                # Determine if this frame should be AI or Raw based on backlog
                # But ALWAYS submit to the executor to maintain frame order.
                backlog = self.get_executor_backlog()

                # Use a "Skip AI" flag instead of a "continue" skip
                skip_ai = backlog > self.dynamic_limit

                self.frame_in_clip_count += 1

                # 1. Immediate Rotation (Ensures logs match the new key instantly)
                if (
                    ENABLE_QUERYING
                    and self.frame_in_clip_count > self.max_frames_per_clip
                ):
                    # Perform safety check BEFORE starting the next 185MB clip
                    # self._check_shm_safety(threshold_percent=90)

                    # Pass old state to consumer before updating
                    old_writer = self.video_writer
                    old_key, old_tmp, old_final, old_frame_in_clip_count = (
                        self.clip_key,
                        self.tmp_file,
                        self.clip_filename,
                        self.frame_in_clip_count,
                    )

                    self.write_queue.put(
                        {
                            "control": "ROTATE",
                            "old_key": old_key,
                            "old_tmp": old_tmp,
                            "old_final": old_final,
                            "frame_in_clip_count": old_frame_in_clip_count,
                            "writer_to_finalize": old_writer,
                        }
                    )

                    self.clip_id += 1
                    self.frame_in_clip_count = 1
                    # self.video_writer = None # Nullify on main thread
                    self._initialize_writer()
                    # self.clip_id += 1
                    # self.frame_in_clip_count = 1
                    # self.video_writer = None # Nullify on main thread
                    # self._initialize_writer()
                    self._check_shm_safety(threshold_percent=90)

                # 2. Handoff to AI and Writer
                self.executor.submit(
                    self.process_frame_async,
                    self.clip_filename,
                    frame,
                    self.frame_count,
                    skip_ai,
                )

                # --- PRECISE CLOCK SYNC ---
                # This prevents the producer from "lapping" the consumer
                # and building that jumpy backlog in the first place.
                elapsed = time.perf_counter() - last_frame_time
                if elapsed < self.target_interval:
                    time.sleep(self.target_interval - elapsed)
                last_frame_time = time.perf_counter()

                self.update_frame()
                self.last_heartbeat = time.time()

            elif self.reader.stopped:
                self.active = False
                break
            else:
                time.sleep(0.01)
        self.stop()

    # ----- DEVICE-SPECIFIC PIPELINES -----
    def test_rbtd_detection_gpu(self, frame):  # , frameNum):
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

        # if ENABLE_QUERYING and self.video_writer:  # and not self.video_queue.full():
        #     self.pinned_downloaded_resizedframe_np = self.resized_frame.download(stream)
        #     # self.resized_frame.download(self.stream, self.pinned_downloaded_resizedframe_np)
        #     #     self.video_writer.write(self.pinned_downloaded_resizedframe_np)
        #     self.write_queue.put(self.pinned_downloaded_resizedframe_np.copy())

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
            # "frameNum": frameNum,  # overall frame
            "mask": self.gpu_morphed_frame,  # GpuMat pointer to cleaned mask
            "full_frame": frame,  # Kept for high-res cropping
        }

    def test_full_cpu(self, frame):  # , frameNum):
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
        # H, W = self.resize_h, self.resize_w
        self.cpu_resized_frame = cv2.resize(
            frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST
        )
        # if ENABLE_QUERYING and self.video_writer:
        #     self.write_queue.put(self.cpu_resized_frame.copy())

        # Apply Background Subtraction on CPU
        self.apply_background_subtraction_cpu(include_history=True, method="and")

        # Clean up the motion mask using Thresholding and Morphology (Dilation)
        _, mask = cv2.threshold(
            self.fgMask, MASK_THRESHOLD_VALUE, MASK_MAX_VALUE, cv2.THRESH_BINARY
        )
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)

        return {
            # "frameNum": frameNum,  # overall frame
            "mask": mask,
            "full_frame": frame,  # Kept for high-res cropping
        }

    # ----- ROI RELATED -----
    def filter_rois(self, raw_bbs, dist_thresh=25, containment_thresh=0.9):
        bbs_full_res = sorted(
            [pair[1] for pair in raw_bbs],
            key=lambda x: x[0],
            # reverse=True,
        )

        merged = merge_boxes_limit(
            bbs_full_res,
            dist_threshold=dist_thresh,
            min_area=self.min_contour_area,
            max_size=MERGE_SIZE_LIMIT,
        )
        merged = filter_contained_boxes(merged, containment_thresh=containment_thresh)
        return merged

    def get_detections_for_contours_bbs(
        self,
        frameNum,
        foi,
        contours,
        thickness=THICKNESS,
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
        stream_name = self.name
        num_objs = 0
        metadata = dict()
        cropped_imgs, cropped_coords = [], []
        H, W = foi.shape[:2]  # Unpack once

        if not contours:
            adaptive_quality = 30 if ENABLE_QUERYING else DISPLAY_FRAME_QUALITY
            frame_bytes = get_display_frame_in_bytes(
                foi,
                display_size=DISPLAY_FRAME_SIZE,
                quality=adaptive_quality,
                return_bytes=return_bytes,
            )
            return metadata, frame_bytes  # num_objs, predictions

        # Filter small noise and convert contours to 8K-space bounding boxes
        raw_bbs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.min_contour_area:
                x1, y1, w, h = cv2.boundingRect(c)

                # Scale coordinates from 640p BGS-space to 8K-space
                xx1 = max(0, int((x1 * self.scale_x)) - RAW_BB_FULL_RES_PADDING)
                yy1 = max(0, int((y1 * self.scale_y)) - RAW_BB_FULL_RES_PADDING)
                xx2 = min(W, int(((x1 + w) * self.scale_x)) + RAW_BB_FULL_RES_PADDING)
                yy2 = min(H, int(((y1 + h) * self.scale_y)) + RAW_BB_FULL_RES_PADDING)
                raw_bbs.append([area, [xx1, yy1, xx2, yy2]])

        dist_thresh = min(0.05 * W, 0.05 * H)
        merged = self.filter_rois(raw_bbs, dist_thresh=dist_thresh)

        # Extract crops at full-resolution
        crop_cnt = 0
        for x1, y1, x2, y2 in merged:
            if (
                (x2 - x1) > 31
                and (y2 - y1) > 31
                and (x2 - x1) < self.frame_width
                and (y2 - y1) < self.frame_height
            ):
                crop_cnt += 1
                if DETECTION_TYPE == "motion":
                    foi = cv2.rectangle(
                        foi,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        thickness,
                    )
                else:
                    crop = foi[y1:y2, x1:x2]
                    cropped_imgs.append(crop)
                    cropped_coords.append((x1, y1))

                if crop_cnt == MODEL_MAX_BATCH_SIZE:
                    # logging.warning(
                    #     f"⚠️ [LIMIT] {self.name} found {len(cropped_imgs)} contours. Capping to 64 for TensorRT."
                    # )
                    # cropped_imgs = cropped_imgs[:MODEL_MAX_BATCH_SIZE]
                    # cropped_coords = cropped_coords[:MODEL_MAX_BATCH_SIZE]
                    break

        if not cropped_imgs or DETECTION_TYPE == "motion":
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
        results = list(
            self.run_model(
                cropped_imgs,
                imgsz=(self.resize_h, self.resize_w),
                batch=len(cropped_imgs),
                device_input=device_input,
                stream=True,
            )
        )

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

    def get_reduced_contour(self, mask, contours):
        foi = np.zeros_like(mask)
        for c in contours:
            area = cv2.contourArea(c)
            x1, y1, w, h = cv2.boundingRect(c)
            if (
                area > self.min_contour_area and w < self.resize_w and h < self.resize_h
            ):  # and area / (w*h) >=0.3:  # and 0.5 < (w / h) < 2.0: # w/ solidity & aspect
                x2 = x1 + w
                y2 = y1 + h
                foi = cv2.rectangle(foi, (x1, y1), (x2, y2), 255, -1)  # BGR- CYAN
        reduced_contours, _ = cv2.findContours(
            foi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return reduced_contours, foi

    def contour2predictions(
        self,
        clip_filename,
        frameNum,  # Frame used in metadata
        mask,
        frame,
        device_input="cpu",
        return_bytes=True,
    ):
        """
        The 'Glue' function that connects Motion Detection to AI Inference.

        Args:
            mask (GpuMat/np.ndarray): The motion mask (from BGS).
            frame (np.ndarray): The original 8K frame.
        """
        global all_metadata, video_ready_list
        clip_key = Path(self.clip_filename).name
        # Extract contours from the motion mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, mask = self.get_reduced_contour(mask, contours)

        # Write frame
        # self.video_writer.write(frame)
        # self.video_writer.write(self.cpu_resized_frame)

        # Pass contours to the YOLO detection logic
        metadata = dict()
        metadata, frame_bytes = self.get_detections_for_contours_bbs(
            frameNum,  # Frame used in metadata
            frame,
            contours,
            thickness=THICKNESS,
            device_input=device_input,
            return_bytes=return_bytes,
        )

        # Update global metadata for storage (Database/JSON)
        if metadata and ENABLE_QUERYING:
            all_metadata.setdefault(
                clip_key,
                {
                    "object": {},
                    "face": {},
                },
            )
            all_metadata[clip_key]["object"].update(metadata)
            # print(f"self.clip_key: {self.clip_key} objs: {list(metadata.keys())}")

        is_last_frame = (frameNum % self.max_frames_per_clip) == 0
        if is_last_frame and ENABLE_QUERYING:
            # Update the tracker that AI work for this clip is done
            check_and_dispatch_to_vdms(
                clip_filename, self.resize_w, self.resize_h, component="meta"
            )

        # frame_bytes returned even if no metadata available
        return frame_bytes

    def async_yolo_task(self, clip_filename, data, return_bytes=True):
        """
        Orchestrates the AI pipeline for a single frame.

        Steps:
        1. (if GPU) Synchronizes CUDA stream and downloads the mask.
        2. Executes contour-based YOLO detection.
        3. Handoffs the frame to the background JPEG encoder.
        """
        try:
            frameNum = data["frameNum"]  # overall frame

            # Use pre-allocated pinned memory from BaseHandler for 8K mask download
            if self.device_input == "cuda":
                self.stream.waitForCompletion()
                # Ensure the download uses the instance-specific CUDA stream
                self.pinned_downloaded_frame_np = data["mask"].download(self.stream)
                # data["mask"].download(self.stream, self.pinned_downloaded_frame_np)

            # Run contour-based YOLO logic and draw overlays
            frame_bytes = self.contour2predictions(
                clip_filename,
                ((frameNum - 1) % self.max_frames_per_clip)
                + 1,  # Frame used in metadata; index 1
                self.pinned_downloaded_frame_np
                if self.device_input == "cuda"
                else data["mask"],
                data["full_frame"],
                device_input=self.device_input,
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
            # if not return_bytes:
            #     # Rotate between buffers so the encoder has a 'locked' memory space
            #     self.buf_idx = (self.buf_idx + 1) % 2
            #     target_buf = self.encode_buffers[self.buf_idx]

            #     # Perform a deep memory copy into the isolated buffer
            #     # np.copyto(target_buf, data["full_frame"])
            #     # np.copyto(target_buf, data["full_frame"], casting='unsafe')
            #     # target_buf[:] = data["full_frame"]
            #     target_buf[:] = np.ascontiguousarray(data["full_frame"])

            #     # Submit to background worker to bypass the GIL
            #     self.executor.submit(self._encode_and_signal, target_buf, frameNum)

        except Exception:
            logging.error(f"Async YOLO Error in {self.name}: {traceback.format_exc()}")
