import asyncio
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
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
from multiprocessing import shared_memory
from pathlib import Path

import cupy
import cupyx.scipy
import cupyx.scipy.ndimage
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from fastapi import FastAPI

sys.path.insert(1, str(Path(__file__).parent.parent))
from include.default_configs import ENABLE_QUERYING_DEFAULT
from include.models import get_model
from include.utils import (
    BOUNDS_KERNEL,
    DETECTION_ACCEL_KERNEL,
    VDMSPool,
    find_contours_gpu_equivalent,
    merge_boxes_cpu,
    merge_boxes_gpu,
)

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

main_app_logger = logging.getLogger(__name__)
STREAM_ARG = False


def log_to_logger(message, level="info"):
    try:
        if level.lower() == "debug":
            main_app_logger.debug(message)
        elif level.lower() == "warning":
            main_app_logger.warning(message)
        else:
            main_app_logger.info(message)
    except Exception:
        pass


# ----- PIPELINE CONFIGURATION -----
# Force OpenCV to run sequentially to prevent context-switching overhead
cv2.setNumThreads(0)  # Forces OpenCV loops to run strictly sequentially
# cv2.setNumThreads(os.cpu_count() or 4)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())  # "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


from include.utils import (
    PipelineConfig,
    PipelineMapping,
    draw_label,
    get_detection_color,
    metadata2vdms_with_retry,
)


class AsyncVideoWriter:
    """Handles disk saving operations asynchronously on a background thread."""

    def __init__(self, path, fourcc, fps, size):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.05)
                if frame is None:
                    break
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def write_frame(self, frame):
        self.queue.put(frame)

    def release(self):
        self.running = False
        self.thread.join()
        self.writer.release()


class AsyncDisplayVideoWriter:
    """
    Lock-free single-element buffer optimized for 8K pipelines.
    Eliminates queue.Queue lock contention to preserve strict target frame rates.
    """

    def __init__(self, target_fps, display_size, quality=60):
        self.target_fps = float(target_fps)
        self.disp_w, self.disp_h = display_size
        self.quality = int(quality)

        # 1. 🔄 FIX: Use a lockless deque with maxlen=1 instead of queue.Queue
        self.buffer = deque(maxlen=1)
        self.running = True
        self.handler = None

        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def set_handler_context(self, handler_instance):
        self.handler = handler_instance

    def _write_loop(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]

        while self.running:
            try:
                # 2. 🔄 FIX: Non-blocking pop from the single-element buffer
                if not self.buffer:
                    time.sleep(0.005)
                    continue

                display_frame, frame_num = self.buffer.popleft()

                if self.handler is None or not self.handler.active:
                    continue

                if (
                    display_frame.shape[1] != self.disp_w
                    or display_frame.shape[0] != self.disp_h
                ):
                    display_frame = cv2.resize(
                        display_frame,
                        (self.disp_w, self.disp_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                ret, jpeg_buf = cv2.imencode(".jpg", display_frame, encode_param)
                if not ret:
                    continue

                frame_bytes = jpeg_buf.tobytes()
                frame_len = len(frame_bytes)
                num_shms = len(self.handler.shms)

                write_idx = (self.handler.ready_buffer_idx.value + 1) % num_shms

                shm_block = self.handler.shms[write_idx]
                shm_block.buf[:frame_len] = memoryview(frame_bytes)

                self.handler.shm_frame_lengths[write_idx] = frame_len
                self.handler.ready_buffer_idx.value = write_idx
                self.handler.shared_details["last_id"] = frame_num
                self.handler.mp_last_id.value = frame_num

                if hasattr(self.handler, "loop") and self.handler.loop is not None:
                    self.handler.loop.call_soon_threadsafe(
                        self.handler.frame_ready_event.set
                    )
                else:
                    self.handler.frame_ready_event.set()

            except Exception as e:
                print(
                    f"[STREAM-ERROR] Lock-free frame streaming dropped: {e}", flush=True
                )
                continue

    def write_frame(self, frame, frame_num):
        """Thread-safe lock-free atomic submission track."""
        if self.running:
            # 3. 🔄 FIX: Directly append to overwrite the old entry without thread locks
            self.buffer.append((frame, frame_num))

    def release(self):
        self.running = False
        self.buffer.clear()
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)


BASE_PIPELINE_CONFIG = PipelineConfig(
    SHARED_MODEL=os.getenv("SHARED_MODEL", False),
    ENABLE_QUERYING=os.getenv("ENABLE_QUERYING", ENABLE_QUERYING_DEFAULT),
)


# Optimizes RTSP ingestion with hardware acceleration and low-delay flags
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|hwaccel;cuda|threads;auto|low_delay;1|probesize;5000000"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;1|probesize;32|analyzeduration;0"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;2|probesize;32|analyzeduration;0"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;2|probesize;32|analyzeduration;0"
    # "rtsp_transport;tcp|hwaccel;cuda|threads;4|probesize;5000000|analyzeduration;5000000"
    #  "rtsp_transport;udp|hwaccel;cuda|threads;8|stimeout;5000000|listen_timeout;5000"
)


# ----- GLOBAL VARIABLES -----
# ENABLE_QUERYING = os.getenv("ENABLE_QUERYING", ENABLE_QUERYING_DEFAULT)
# if BASE_PIPELINE_CONFIG.ENABLE_QUERYING:
# Tracks all metadata
all_metadata = {}

# Tracks clip_filename once video re-encoded
# video_ready_list = {}

# Queue for metadata being sent to vdms
send_metadata_queue = queue.Queue()

# Tracks if both components are finished:
#   {"clip_name": {"video": bool, "meta": bool}}
clip_completion_tracker = {}

PADDING_PX = 10


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
                video_backlog = (
                    streamer.write_queue.qsize()
                    if streamer.config.ENABLE_QUERYING
                    else 0
                )
                # io_backlog = (
                #     streamer.io_executor._work_queue.qsize()
                #     if hasattr(streamer, "io_executor")
                #     else 0
                # )

                # Check if the stream is marked inactive OR timed out
                # streamer.active should be False when the video source ends
                is_stale = now - streamer.last_heartbeat > 30

                should_remove = False

                if not streamer.active and (
                    ai_backlog == 0 and video_backlog == 0  # and io_backlog == 0
                ):
                    should_remove = True  # Video ended naturally
                elif is_stale and (
                    ai_backlog == 0 and video_backlog == 0  # io_backlog == 0
                ):
                    should_remove = True  # Browser tab closed/Network lost
                elif now - streamer.last_heartbeat > 90:
                    should_remove = True  # Hard timeout for hung processes

                if should_remove:
                    async with app.state.stream_lock:
                        if BASE_PIPELINE_CONFIG.DEBUG == "1":
                            print(f"CLEANUP: Removing {name} from active_streams")
                        streamer.stop()
                        app.state.active_streams.pop(name, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    if not hasattr(app.state, "classes"):
        app.state.classes = None

    if not hasattr(app.state, "active_streams"):
        app.state.active_streams = {}

    app.state.status = "Ready"
    app.state.stream_lock = asyncio.Lock()
    if BASE_PIPELINE_CONFIG.SHARED_MODEL:
        app.state.model = YOLO(
            BASE_PIPELINE_CONFIG.model_path, verbose=False, task="detect"
        )

        device_input = "cuda" if BASE_PIPELINE_CONFIG.DEVICE == "GPU" else "cpu"
        print("Starting shared model warmup...")
        dummy_input = torch.zeros(
            (1, 3, BASE_PIPELINE_CONFIG.MODEL_H, BASE_PIPELINE_CONFIG.MODEL_W)
        ).to(device_input)
        for _ in range(20):
            _ = app.state.model(dummy_input, verbose=False)

        del dummy_input
        # torch.cuda.empty_cache()
        print("Shared model warmup and VRAM purge complete.")

    janitor_task = asyncio.create_task(auto_cleanup_janitor(app))

    if BASE_PIPELINE_CONFIG.DEBUG == "1":
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
def nv12_to_rgb_torch(
    nv12_tensor, h, w, is_h264_8k=False, out_buffer=None, is_bgr=True
):
    """
    Highly optimized NV12/YUV to RGB/BGR conversion.
    Fixes the 'Blue Frame' by applying proper YUV-to-RGB matrix math.
    """
    with torch.no_grad():
        if is_h264_8k:
            # 8K Planar Path: Map the channels directly
            y = nv12_tensor[0:1, :, :].half()
            u = nv12_tensor[1:2, :, :].half()
            v = nv12_tensor[2:3, :, :].half()  # Use the 3rd channel!

            # No column slicing (0::2) needed if it's already planar.
            # But we must ensure U and V match Y dimensions if they were subsampled.
            if u.shape[-1] != w or u.shape[-2] != h:
                u = F.interpolate(u.unsqueeze(0), size=(h, w), mode="bilinear").squeeze(
                    0
                )
                v = F.interpolate(v.unsqueeze(0), size=(h, w), mode="bilinear").squeeze(
                    0
                )
        else:
            # Standard NV12 Path: Image is [H*1.5, W]
            y = nv12_tensor[:h, :w].unsqueeze(0).half()
            uv = (
                nv12_tensor[h:, :w]
                .reshape(h // 2, w // 2, 2)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .half()
            )
            # Upsample Chroma (4:2:0 -> 4:4:4)
            uv_up = F.interpolate(uv, size=(h, w), mode="nearest")
            u = uv_up[0, 0:1, :, :]
            v = uv_up[0, 1:2, :, :]

        # --- YUV to RGB Conversion Math (BT.709) ---
        # 1. Normalize Luma and center Chroma
        y = (y - 16.0) * 1.164
        u = u - 128.0
        v = v - 128.0

        # 2. Matrix Multiplication (Coefficients for natural color)
        r = y + 1.793 * v
        g = y - 0.213 * u - 0.533 * v
        b = y + 2.112 * u

        # 3. Stack into final order
        if is_bgr:
            colored_img = torch.cat([b, g, r], dim=0)
        else:
            colored_img = torch.cat([r, g, b], dim=0)

        # 4. Final Clamp and Format
        colored_img.clamp_(0, 255)
        output = colored_img.to(torch.uint8)

        if out_buffer is not None:
            out_buffer.copy_(output)
            return out_buffer

        return output


def send_metadata(
    VDMS_POOL=None,
    DEBUG_FLAG=BASE_PIPELINE_CONFIG.DEBUG_FLAG,
    INGESTION=BASE_PIPELINE_CONFIG.INGESTION,
    TEST_MODE=BASE_PIPELINE_CONFIG.TEST_MODE,
    UDF_HOST=BASE_PIPELINE_CONFIG.UDF_HOST,
    UDF_PORT=BASE_PIPELINE_CONFIG.UDF_PORT,
    DBHOST=BASE_PIPELINE_CONFIG.DBHOST,
    DBPORT=BASE_PIPELINE_CONFIG.DBPORT,
):
    """
    Consumer thread that sends metadata to VDMS.
    If retries fail, it saves the data to a local JSON 'dead-letter' file.
    """
    if VDMS_POOL is None:
        # VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)
        VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)

    global all_metadata, send_metadata_queue
    clip_filename = ""
    clip_key = ""
    width = 0
    height = 0
    while True:
        try:
            queue_details = send_metadata_queue.get()
            if queue_details is None:
                print(
                    "[METADATA-DEBUG D-pill] Poison pill received. Terminating send_metadata thread loops cleanly.",
                    flush=True,
                )
                break

            # # (clip_key, clip_filename, width, height, clip_metadata) = queue_details
            (clip_filename, width, height) = queue_details
            clip_key = Path(clip_filename).name
            clip_metadata = all_metadata.pop(clip_key, None)

            if clip_metadata:
                main_app_logger.info(
                    f"sending 2 vdms: {clip_key} with {len(clip_metadata)} items."
                )
                success = metadata2vdms_with_retry(
                    clip_key,
                    clip_filename,
                    clip_metadata,
                    width,
                    height,
                    VDMS_POOL=VDMS_POOL,
                    DEBUG_FLAG=DEBUG_FLAG,
                    INGESTION=INGESTION,
                    TEST_MODE=TEST_MODE,
                    UDF_HOST=UDF_HOST,
                    UDF_PORT=UDF_PORT,
                    DBHOST=DBHOST,
                    DBPORT=DBPORT,
                )

                # CUSTOM ERROR HANDLER: Final Failure Fallback
                if not success:
                    error_path = f"{BASE_PIPELINE_CONFIG.CODE_DIR}/failed_metadata/{clip_key}.json"
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
            print(f"[EXCEPTION] Exception occurred in send_metadata: {e}")
            traceback.print_exc()


# ----- STREAM HANDLERS -----
def scale_clusters_to_8k(merged_640, frame_w=7680, frame_h=4320):
    # Ratios for 8K projection
    scale_x = frame_w / 640.0
    scale_y = frame_h / 640.0
    final_rois = []

    for box in merged_640:
        # Calculate centroid in 640p space
        cx_640 = (box[0] + box[2]) / 2.0
        cy_640 = (box[1] + box[3]) / 2.0

        # Map to 8K space with float precision to avoid offset drift
        cx_8k = cx_640 * scale_x
        cy_8k = cy_640 * scale_y

        # Center the 640x640 YOLO crop at the 8K centroid
        half = 320
        nx1 = max(0, int(cx_8k - half))
        ny1 = max(0, int(cy_8k - half))
        nx2 = min(frame_w, nx1 + 640)
        ny2 = min(frame_h, ny1 + 640)

        # Shift back if clamped at 8K boundaries
        if nx2 == frame_w:
            nx1 = max(0, frame_w - 640)
        if ny2 == frame_h:
            ny1 = max(0, frame_h - 640)

        final_rois.append([nx1, ny1, nx1 + 640, ny1 + 640])

    return final_rois


# def rendering_worker(
#     queue,
#     shared_details,
#     ready_idx,
#     reader_active_idx,
#     frame_lengths,
#     signal_queue,
#     display_size,
#     quality,
# ):
#     disp_w, disp_h = display_size
#     # Attach to both buffers
#     shm_names = shared_details["shm_names"]
#     worker_shms = [mp.shared_memory.SharedMemory(name=n) for n in shm_names]
#     num_shms = len(shm_names)

#     try:
#         while True:
#             print("[FASTAPI-TRACE D] rendering_worker child process waiting on queue.get()...", flush=True)
#             item = queue.get()
#             if item is None:  # Sentinel value to stop the worker
#                 break

#             # frame is display size
#             # metadata in resized res
#             display_frame, frameNum, metadata_or_bbs, class_list = item
#             print(f"[FASTAPI-TRACE E] rendering_worker child dequeued Frame #{frameNum}.", flush=True)
#             scale_display_x = disp_w / 640
#             scale_display_y = disp_h / 640

#             if isinstance(metadata_or_bbs, dict):
#                 # Case: Object Detection
#                 display_frame = get_metadata_overlay(
#                     display_frame,
#                     metadata_or_bbs,
#                     class_list,
#                     (scale_display_x, scale_display_y),
#                     (disp_w, disp_h),
#                 )

#             elif metadata_or_bbs is not None:
#                 # Case: Motion Detections Only (SF Path)
#                 display_frame = get_bb_overlay(
#                     display_frame,
#                     metadata_or_bbs,
#                     (scale_display_x, scale_display_y),
#                     (disp_w, disp_h),
#                 )

#             # writer.write(display_frame)
#             if frameNum > shared_details["last_id"]:  # self.last_delivered_frame_id:
#                 frame_bytes = get_display_frame_in_bytes(
#                     display_frame,
#                     display_size=display_size,
#                     quality=quality,
#                     return_bytes=True,
#                 )
#                 if frame_bytes:
#                     # THE HARD GUARD: If the reader is currently touching RAM, skip this write.
#                     # This prevents the '1-minute' scramble by ensuring zero memory overlap.
#                     # if signal_queue.full():
#                     #     continue
#                     frame_len = len(frame_bytes)

#                     forbidden_idx = [ready_idx.value, reader_active_idx.value]
#                     available_idx = [
#                         i for i in range(num_shms) if i not in forbidden_idx
#                     ]

#                     # if not available_idx:
#                     #     continue

#                     # # Write to the buffer that is NOT currently 'ready'
#                     # # write_idx = (shared_details["buffer_idx"] + 1) % 2
#                     # # write_idx = 1 if ready_idx.value == 0 else 0
#                     # # current_ready = ready_idx.value
#                     # # write_idx = (current_ready + 1) % 3
#                     # write_idx = available_idx[0]

#                     if not available_idx:
#                         # Instead of skipping completely and freezing, force a fallback
#                         # to the next standard ring channel
#                         write_idx = (ready_idx.value + 1) % num_shms
#                     else:
#                         write_idx = available_idx[0]

#                     shm = worker_shms[write_idx]

#                     # Zero-copy write to RAM
#                     shm.buf[:frame_len] = frame_bytes

#                     frame_lengths[write_idx] = frame_len
#                     # shared_details["buffer_idx"] = write_idx
#                     ready_idx.value = write_idx
#                     # shared_details["last_id"] = frameNum
#                     # self.last_frame_id = frameNum
#                     # self.last_heartbeat = time.time()
#                     # Signal the FastAPI generator that a new frame is ready
#                     # self.loop.call_soon_threadsafe(self.frame_ready_event.set)
#                     # self.mp_frame_ready_event.set()
#                     # try:
#                     #     signal_queue.put_nowait(True)
#                     # except Exception:
#                     #     pass
#                     # signal_queue.put(True)
#                     try:
#                         print(f"[FASTAPI-TRACE F] Frame #{frameNum} written to SHM slot {write_idx} ({frame_len} bytes). Signaling parent...", flush=True)
#                         signal_queue.put_nowait((True, frameNum))
#                     except queue.Full:
#                         print(f"[FASTAPI-TRACE WARN] signal_queue full on Frame #{frameNum}", flush=True)
#                         # pass  # If the web app is slightly behind, do not freeze the child process!

#         # END While

#     except Exception as e:
#         print(f"[EXCEPTION] Error while rendering display: {e}")
#     finally:
#         for s in worker_shms:
#             s.close()


def get_metadata_overlay(
    display_frame, metadata_or_bbs, class_list, scale_display, disp_size, is_bgr=True
):
    scale_display_x, scale_display_y = scale_display
    disp_w, disp_h = disp_size
    for _, obj in metadata_or_bbs.items():
        bbox = obj["bbox"]
        x = max(0, int(bbox["x"] * scale_display_x))
        y = max(0, int(bbox["y"] * scale_display_y))
        w = min(disp_w, int(bbox["width"] * scale_display_x))
        h = min(disp_h, int(bbox["height"] * scale_display_y))

        class_name = bbox["object"]
        class_id = class_list.index(class_name) if class_name in class_list else 0
        confidence = bbox.get("object_det", {}).get("confidence", 0.0)

        bb_color = get_detection_color(class_id, is_bgr=is_bgr)
        label = f"{class_name} {confidence:.2f}"

        display_frame = cv2.rectangle(
            display_frame, (x, y), (x + w, y + h), bb_color, 2
        )
        draw_label(display_frame, label, (x, y), color=bb_color, padding=5)
    return display_frame


def get_bb_overlay(display_frame, metadata_or_bbs, scale_display, disp_size):
    scale_display_x, scale_display_y = scale_display
    disp_w, disp_h = disp_size
    for box in metadata_or_bbs:
        if torch.is_tensor(box):
            x1, y1, x2, y2 = box.to(torch.int).cpu().tolist()
        else:
            x1, y1, x2, y2 = map(int, box)

        x1 = max(0, int(x1 * scale_display_x))
        y1 = max(0, int(y1 * scale_display_y))
        x2 = min(disp_w, int(x2 * scale_display_x))
        y2 = min(disp_h, int(y2 * scale_display_y))
        display_frame = cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return display_frame


# def test_rendering_worker(queue, display_size, out_path, target_fps):
#     """
#     Ultra-efficient video saver for TEST_MODE.
#     Pipes raw BGR frames directly into an internal FFmpeg engine subshell.
#     """
#     disp_w, disp_h = display_size

#     # Construct optimized MPEG-4 parameters to match main pipeline architecture
#     ffmpeg_cmd = [
#         "ffmpeg",
#         "-y",
#         "-probesize",
#         "32",
#         "-analyzeduration",
#         "0",
#         "-f",
#         "rawvideo",
#         "-pix_fmt",
#         "bgr24",
#         "-s",
#         f"{disp_w}x{disp_h}",
#         "-r",
#         str(int(target_fps)),
#         "-i",
#         "-",
#         "-c:v",
#         "libx264",
#         "-crf",
#         "23",
#         # "-c:v", "mpeg4",  # Or "libx264" if you prefer H.264
#         # "-qscale:v", "4",  # Quality scale (use -crf 23 if using libx264)
#         str(out_path),
#     ]

#     # Spawn background daemon process
#     proc = subprocess.Popen(
#         ffmpeg_cmd,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#         bufsize=10**7,
#     )

#     try:
#         while True:
#             item = queue.get()
#             if item is None:  # Sentinel value to drain and close the process
#                 break

#             display_frame, frameNum, metadata_or_bbs, class_list = item
#             # display_frame = np.ascontiguousarray(display_frame)
#             scale_display_x = disp_w / 640
#             scale_display_y = disp_h / 640

#             # --- Draw Detection Overlays ---
#             if isinstance(metadata_or_bbs, dict):
#                 # Object Mode (YOLO Structs)
#                 display_frame = get_metadata_overlay(
#                     display_frame,
#                     metadata_or_bbs,
#                     class_list,
#                     (scale_display_x, scale_display_y),
#                     (disp_w, disp_h),
#                 )

#             elif metadata_or_bbs is not None:
#                 # # Motion / Smart Filtering Overlay Path
#                 display_frame = get_bb_overlay(
#                     display_frame,
#                     metadata_or_bbs,
#                     (scale_display_x, scale_display_y),
#                     (disp_w, disp_h),
#                 )

#             # Pipe continuous raw contiguous memory block directly into kernel filesystem handles
#             proc.stdin.write(np.ascontiguousarray(display_frame).tobytes())
#             # queue.task_done()

#     except Exception as e:
#         print(f"[TEST-WORKER-EXCEPTION] Video compilation error: {e}")
#     finally:
#         if proc.stdin:
#             proc.stdin.close()
#         proc.wait()


class DeviceBaseHandler:
    def __init__(
        self, source, name, active_streams, config=BASE_PIPELINE_CONFIG, **kwargs
    ):
        self.name = name
        self.source = source
        self.is_rtsp = str(self.source).startswith("rtsp:/")
        self.active = True
        self.active_streams = active_streams

        self.config = config
        configstr = "\n".join(
            [f"\t{k}: {v}" for k, v in config.__dict__.items() if not k.startswith("_")]
        )
        log_to_logger(f"PipelineConfig: \n{configstr}\n", level="info")

        self.loop = asyncio.get_event_loop()
        self.frame_ready_event = asyncio.Event()
        # self.frame_ready_event = threading.Event()
        self._is_stopped = False
        self._stop_lock = threading.Lock()  # Local lock for this instance
        self.mp_frame_ready_event = mp.Event()

        # From global
        self.device = self.config.DEVICE
        self.device_input = self.config.device_input
        # self.disp_w, self.disp_h = self.config.DISPLAY_FRAME_SIZE
        self.resize_h, self.resize_w = [self.config.MODEL_H, self.config.MODEL_W]

        self.setup_reader(self.config.TARGET_FPS, self.config.CLIP_DURATION)

        # Kwargs
        # clip_duration = kwargs.get("clip_duration", CLIP_DURATION)
        self.initialize_variables()

        provided_model = kwargs.get("model")
        self.setup_model(provided_model)

        self.prepare_pipeline()

        # Start dedicated inference thread and timers
        # self.stat_start_time = time.perf_counter() # timing to display frame
        self.last_heartbeat = time.time()
        self.setup_threads()

    def setup_model(self, provided_model, force_export=False):
        if (
            self.frame_width * self.frame_height
        ) <= self.config.SMART_FILTERING_PIXEL_CONSTRAINT:
            if "_noSF" not in self.config.model_path:
                oldpath = Path(self.config.model_path)
                old_modelname = self.config.MODEL_NAME
                self.config.MODEL_NAME = f"{old_modelname}_noSF"
                new_model_name = oldpath.name.replace(
                    old_modelname, self.config.MODEL_NAME
                )
                self.config.model_path = str(oldpath.parent / new_model_name)

        if provided_model is not None and not isinstance(provided_model, str):
            self.model = provided_model
            self.label_source = [v for k, v in self.model.names.items()]
        else:
            # if isinstance(provided_model, str) or provided_model is None:
            # if Path(self.config.model_path).exists():
            #     self.model = YOLO(self.config.model_path, verbose=False, task="detect")
            #     self.label_source = []
            #     for k, v in self.model.names.items():
            #         self.label_source.append(v)
            # else:
            run_platform_name = "engine" if "cuda" in self.device_input else "openvino"
            self.model, _, self.label_source = get_model(
                Path(self.config.model_path).parent,
                self.config.MODEL_NAME.replace("_noSF", ""),
                run_platform_name,
                self.device_input,
                batch=self.config.MODEL_MAX_BATCH_SIZE,
                force_export=force_export,
                sf_enabled=self.config.sf_enabled,
                model_h=self.resize_h,
                model_w=self.resize_w,
            )

            if run_platform_name == "openvino":
                self.model.predictor.args.embed = False

            if not self.config.sf_enabled:
                self.model_warmup(self.frame_height, self.frame_width)
            else:
                self.model_warmup(self.resize_h, self.resize_w)
        # else:
        #     self.model = provided_model
        #     self.label_source = []
        #     for k, v in self.model.names.items():
        #         self.label_source.append(v)

    def initialize_variables(self):
        # 1. HARD GUARD: Capture reader values and ensure the connection is active
        self.input_fps = self.reader.input_fps
        self.target_fps = self.reader.target_fps
        self.frame_width = self.reader.frame_width
        self.frame_height = self.reader.frame_height
        self.numFrames = self.reader.numFrames

        if self.input_fps <= 0 or self.frame_width <= 0 or self.frame_height <= 0:
            main_app_logger.error(
                f"[{self.name}] Stream handler fast-fail triggered. Destination "
                f"unreachable or invalid ({self.source}). Terminating pipeline configuration."
            )
            # Instantly stop background threads to prevent zombie process leakage
            if hasattr(self, "reader") and self.reader is not None:
                self.reader.stop()

            raise RuntimeError(
                f"Failed to initialize stream reader endpoint: {self.source}"
            )

        # 2. Proceed with calculation mechanics safely only if values are healthy
        self.step_size = self.input_fps / self.target_fps
        self.frame_skip = self.reader.frame_skip
        self.max_frames_per_clip = self.reader.max_frames_per_clip
        self.frame_interval = self.reader.frame_interval

        self.duration_s = self.numFrames / self.input_fps
        self.expected_num_frames = int(self.duration_s * self.target_fps)
        self.get_frameWH()

        # Determine minimum contour size relative to frame resolution
        self.min_contour_area = int(
            (self.config.ROI_MIN_AREA_RATIO * self.resize_w)
            * (self.config.ROI_MIN_AREA_RATIO * self.resize_h)
        )  # 207

        self.dist_thresh_8k = max(
            self.config.ROI_DISTANCE_THRESH_RATIO * self.frame_width,
            self.config.ROI_DISTANCE_THRESH_RATIO * self.frame_height,
        )
        multiplier = 2.0 if self.device_input == "cpu" else 1.0
        self.dist_thresh_640 = (
            max(
                self.config.ROI_DISTANCE_THRESH_RATIO * self.resize_w,
                self.config.ROI_DISTANCE_THRESH_RATIO * self.resize_h,
            )
            * multiplier
        )  # 0.05 * self.resize_w
        self.scales_tensor = torch.tensor(
            [self.scale_x, self.scale_y, self.scale_x, self.scale_y],
            # device="cpu",
            device=self.device_input,
        )

        self.disp_w, self.disp_h = self.config.DISPLAY_FRAME_SIZE

        # Performance Tracking
        self.elapsed_display_time = 0.0
        self.frame_count = 0  # Frame count for videos
        self.frame_count_target = 0
        self.last_delivered_frame_id = -1  # Track what was actually sent
        self.last_frame_id = 0
        self.latest_processed_frame = None
        self.next_process_idx = 0.0
        self.stat_fps = 0
        self.stat_frame_count = 0
        self.total_objects_detected = 0

        self.writer_done = True

        # Video Clipping
        # self.video_writer = None
        self.ffmpeg_proc = None  # Replaces cv2.VideoWriter completely
        # self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1, mp4v
        # self.clip_id = 0
        # self.clip_filename = ""
        self.clip_filename_pattern = f"{self.config.SHARED_OUTPUT}/{self.name}_%03d.mp4"
        # self.clip_key = f"{self.name}_000.mp4"
        # self.tmp_file = ""
        self.frame_in_clip_count = 0

        if self.config.ENABLE_QUERYING:
            # Thread-safe queue for the resized frames (640x640)
            # maxlen=300 allows for a 20-second buffer in case of extreme disk lag
            # Non-blocking queue for frames and control signals
            self.write_queue = queue.Queue(maxsize=300)
            if not self.config.TEST_MODE:
                self.send_metadata_queue = queue.Queue()
            self.writer_done = False
        self.stop_writer = threading.Event() if self.config.ENABLE_QUERYING else None

        # --- PRE-ALLOCATED ZERO-COPY HARDWARE RING WORKSPACE ---
        self.ring_depth = 4  # 8
        self.gpu_ring_idx = 0
        self.cpu_ring_idx = 0

        self.pinned_matrices = []
        self.pinned_tensors = []

        self.ingest_ring_depth = 4
        self.ingest_ring_idx = 0
        self.ingest_ring = []

        for _ in range(self.ingest_ring_depth):
            # Match the exact dimensions of the 8K reader output
            mat = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            try:
                cv2.cuda.registerPageLocked(
                    mat
                )  # Page-lock for ultra-fast DMA transfers
            except Exception:
                pass
            self.ingest_ring.append(mat)

        # Pre-allocate a 4-slot ring buffer for raw 8K BGR frames
        self.ai_ring_depth = 4
        self.ai_ring_idx = 0
        self.frame_stride_bytes = (
            self.resize_w * self.resize_h * 3
        )  # ~99.5 MB per frame

        self.ai_shms = []
        self.ai_shm_names = []
        self.ai_pinned_tensors = []  # Explicit property initialization 🚀
        init_timestamp = int(time.time_ns())
        for i in range(self.ai_ring_depth):
            name = f"shm_ai_640_{self.name}_{i}_{os.getpid()}_{init_timestamp}"

            try:
                # Attempt to attach to a lingering zombie segment
                old_shm = shared_memory.SharedMemory(name=name)
                old_shm.close()
                old_shm.unlink()  # Permanently destroys the old OS block handle
                main_app_logger.warning(
                    f"Cleaned up residual zombie shared memory block: {name}"
                )
            except FileNotFoundError:
                pass  # Block doesn't exist, safe to proceed normal initialization

            shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.frame_stride_bytes
            )
            self.ai_shms.append(shm)
            self.ai_shm_names.append(name)

            # Map a zero-copy lockless numpy array view straight onto the memory block
            view = np.ndarray(
                (self.resize_h, self.resize_w, 3), dtype=np.uint8, buffer=shm.buf
            )

            # Page-lock the host buffer window to maximize PCIe bus transfer bandwidth
            try:
                cv2.cuda.registerPageLocked(view)
            except Exception:
                pass

            # Expose a direct matching PyTorch host tensor map to secure high-speed uploads
            self.ai_pinned_tensors.append(torch.from_numpy(view))

        # Pre-allocate a 4D FP16 GPU staging canvas to maximize Tensor Core performance
        # if self.device_input == "cuda":
        # self.ai_gpu_staging = torch.empty(
        #     (1, 3, self.frame_height, self.frame_width),
        #     dtype=torch.float16,
        #     device=f"cuda:{self.gpu_id}",
        # )
        # self.preview_gpu_staging = torch.empty(
        #     (1, 3, self.frame_height, self.frame_width),
        #     dtype=torch.float16,
        #     device=f"cuda:{self.gpu_id}",
        # )

        # Pre-allocate 640x640 workspace footprint across CPU and GPU spaces
        for _ in range(self.ring_depth):
            mat = np.zeros((self.resize_h, self.resize_w, 3), dtype=np.uint8)
            try:
                cv2.cuda.registerPageLocked(mat)
            except cv2.error:
                pass
            self.pinned_matrices.append(mat)
            self.pinned_tensors.append(torch.from_numpy(mat))
        # Isolate CUDA tasks using a dedicated stream and independent hardware completion barriers
        self.processing_stream = (
            torch.cuda.Stream() if self.device_input == "cuda" else None
        )
        # Pre-allocated hardware events guarantee completely non-blocking stream isolation
        self.slot_events = (
            [torch.cuda.Event() for _ in range(8)]
            if self.device_input == "cuda"
            else None
        )

        self.gpu_float_staging = None
        if self.device_input == "cuda":
            self.gpu_float_staging = torch.empty(
                (1, 3, self.frame_height, self.frame_width),
                dtype=torch.float16,
                device=f"cuda:{self.gpu_id}",
            )

        # Default Kernels
        self.dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            # cv2.MORPH_RECT,
            (self.config.DILATE_KERNEL_SIZE, self.config.DILATE_KERNEL_SIZE),
        )
        # self.dilate_kernel_for_enhanced_mask = np.ones((15,15), np.uint8)  # 5, 5) (21, 21)
        self.dilate_kernel_for_enhanced_mask = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            # cv2.MORPH_RECT,
            (
                self.config.BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE,
                self.config.BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE,
            ),
        )

    def setup_reader(self, target_fps, clip_duration):
        # if hasattr(self, "reader"):
        #     del self.reader

        # Add a tiny sleep or garbage collect to ensure the GPU handle is released
        gc.collect()
        torch.cuda.empty_cache()  # Clear any remaining context

        # TODO: Further investigate GPU path, bkgd subtraction sensitive to artifacts
        self.gpu_id = 0
        is_rtsp_stream = str(self.source).startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        determined_queue_size = (
            4 if is_rtsp_stream else 2
        )  # (0 if self.config.TEST_MODE else 2)
        if self.device_input == "cuda":  # and not self.is_rtsp:
            from include.readers import GPUReader

            self.reader = GPUReader(
                source=self.source,
                target_fps=target_fps,
                clip_duration=clip_duration,
                gpu_id=0,  # self.gpu_id,
                queue_size=determined_queue_size,
            )
        else:
            from include.readers import CPUReader

            self.reader = CPUReader(
                source=self.source,
                target_fps=target_fps,
                clip_duration=clip_duration,
                queue_size=determined_queue_size,
            )

    def prepare_pipeline(self):
        if self.device_input == "cuda":
            self.prepare_gpu_pipeline()
            if len(self.active_streams) == 0:
                self.gpu_warmup()
        else:
            self.prepare_cpu_pipeline()

        # AUTOMATED BACKSUB WARMUP GATE
        # Priming the Gaussian mixture model before the pipeline loop kicks off
        if getattr(self.config, "SMART_FILTERING_ENABLED", False):
            video_source = getattr(self, "source", None)
            if video_source and os.path.exists(video_source):
                print(f"[INIT] Warming up background subtractor using: {video_source}")
                warmup_cap = cv2.VideoCapture(video_source)

                # 20 to 30 frames are optimal to prime the baseline history grid
                for _ in range(30):
                    ret, raw_frame = warmup_cap.read()
                    if not ret:
                        break

                    # Downscale the warmup frames to match your active pipeline resolution rules
                    resized_warmup = cv2.resize(
                        raw_frame,
                        (self.resize_w, self.resize_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    # Force a highly aggressive learning rate (e.g., 0.1) during initialization
                    # to lock down the background model quickly, avoiding delayed frame processing.
                    if self.device_input == "cuda":
                        # If you use a GPU-bound background subtractor
                        self.bgs_cuda.apply(
                            resized_warmup, learningRate=self.lr, stream=self.bgs_stream
                        )
                    else:
                        # Standard CPU track fallback
                        self.bgs.apply(resized_warmup, learningRate=self.lr)

                warmup_cap.release()
                print("[INIT] Background subtractor warm up complete. GMM initialized.")

    def setup_threads(self):
        # Shared 10MB memory for display
        self.setup_shared_memory()

        # Executor for Async YOLO tasks and FFmpeg re-encoding
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        self.clip_executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        print(
            f"sf_enabled: {self.config.sf_enabled}\tTEST_MODE: {self.config.TEST_MODE}",
            flush=True,
        )

        self.signal_queue = mp.Queue(maxsize=32)  # 1)
        self.render_queue = mp.Queue(maxsize=5)

        # Producer: Handles acquisition and AI metadata logs
        # self.process_thread = threading.Thread(
        #     target=self.run_realtime_inference,
        #     args=(self.config.sf_enabled,),
        #     daemon=True,
        # )
        self.process_thread = threading.Thread(
            target=lambda: self.run_realtime_inference(
                sf_enabled=self.config.sf_enabled
            ),
            daemon=True,
        )

        if self.config.TEST_MODE:
            test_dir = os.getenv(
                "TEST_SUITE_RENDER_DIR", str(Path(self.config.SHARED_OUTPUT))
            )
            os.makedirs(test_dir, exist_ok=True)

            out_path = os.path.join(test_dir, f"{self.name}_detections_output.mp4")
            self.output_path = out_path
            log_to_logger(
                f"[TEST MODE] Detection results saved to: {out_path}", level="info"
            )

            self.async_writer = AsyncVideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*"avc1"),  # avc1, mp4v
                float(self.target_fps),
                (self.disp_w, self.disp_h),
            )

            # self.render_proc = threading.Thread(
            #     target=test_rendering_worker,
            #     args=(
            #         self.render_queue,
            #         (self.disp_w, self.disp_h),
            #         out_path,
            #         self.target_fps,
            #     ),
            #     daemon=True,
            # )

            # Dummy target alignment to prevent execution signature exceptions
            self.render_proc = threading.Thread(target=lambda: None, daemon=True)
            self.display_proc = threading.Thread(target=lambda: None, daemon=True)
        else:
            self.async_writer = AsyncDisplayVideoWriter(
                float(self.target_fps),
                (self.disp_w, self.disp_h),
                quality=int(self.config.DISPLAY_FRAME_QUALITY),
            )
            self.async_writer.set_handler_context(self)
            # self.async_writer.start_worker()
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    self.loop = asyncio.get_event_loop()
                except RuntimeError:
                    self.loop = None  # Headless fallback context

            self.render_proc = threading.Thread(target=lambda: None, daemon=True)
            self.display_proc = threading.Thread(target=lambda: None, daemon=True)

        if self.config.ENABLE_QUERYING:
            # NEW: Dedicated I/O pool for Disk/GPU transfers (Higher worker count for 8K)
            # self.io_executor = ThreadPoolExecutor(max_workers=8)

            # Dedicated FFmpeg pool so re-encoding doesn't slow down live AI
            # self.ffmpeg_executor = ThreadPoolExecutor(max_workers=2)

            if not self.config.TEST_MODE:
                # Sends metadata to VDMS
                self.metadata_thread = threading.Thread(
                    target=send_metadata,
                    args=(
                        VDMSPool(self.config.DBHOST, self.config.DBPORT, size=10),
                        self.config.DEBUG_FLAG,
                        self.config.INGESTION,
                        self.config.TEST_MODE,
                        self.config.UDF_HOST,
                        self.config.UDF_PORT,
                        self.config.DBHOST,
                        self.config.DBPORT,
                    ),
                    daemon=True,
                )

            # Consumer: Handles GPU-to-CPU download and Disk I/O (Writing resized frames to RAM disk)
            self.writer_thread = threading.Thread(
                target=self.video_writer_core_loop,
                args=(self.stop_writer,),
                daemon=True,
            )

    def setup_shared_memory(self):
        self.manager = mp.Manager()

        self.shms = []
        shm_names = []
        num_shms = 3

        # Shared Integer to track which buffer is "Ready" for the UI
        # 'i' for integer, initialized to 0
        self.ready_buffer_idx = mp.Value("i", 0)
        self.reader_active_idx = mp.Value("i", -1)
        self.shm_frame_lengths = mp.Array("i", [0 for _ in range(num_shms)])
        self.mp_last_id = mp.Value("i", -1)
        self.mp_frame_ready_flag = mp.Value("b", False)

        display_timestamp = int(time.time_ns())

        for idx in range(num_shms):
            # self.shm = mp.shared_memory.SharedMemory(create=True, size=10*1024*1024)
            shm_name = f"shm_{self.name}_{idx}_{os.getpid()}_{display_timestamp}"
            # print(f"[DEBUG]: Setting up SHM {shm_name}", flush=True)
            try:
                shm = mp.shared_memory.SharedMemory(
                    name=shm_name, create=True, size=10 * 1024 * 1024
                )
            except FileExistsError:
                # Attach to existing memory
                shm = mp.shared_memory.SharedMemory(name=shm_name)
            except Exception as e:
                main_app_logger.error(f"Failed to initialize shared memory: {e}")
                raise
            self.shms.append(shm)
            shm_names.append(shm_name)

        self.signal_queue = self.manager.Queue(maxsize=32)
        self.render_queue = self.manager.Queue(maxsize=5)

        self.shared_details = self.manager.dict()
        self.shared_details["shm_names"] = shm_names
        # self.shared_details["buffer_idx"] = 0
        # self.shared_details["frame_length"] = [0 for _ in range(num_shms)]
        self.shared_details["last_id"] = -1

    def start(self):
        """
        Starts the decoupled ingestion and inference threads in the correct order.
        """
        # PRE-SYNC: Ensure GPU is idle before timing starts
        if self.device_input == "cuda":
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Start the hardware-decoupled reader first
        self.reader.start()

        if self.config.ENABLE_QUERYING:
            self._initialize_writer()

        # Small delay to allow the reader's deque to populate
        time.sleep(0.1)

        # Start the producer and consumer threads
        if hasattr(self, "process_thread") and not self.process_thread.is_alive():
            self.process_thread.start()

        # If running on the CPU track, block the main test thread briefly until the CPU
        # has successfully populated the first few frames into the render queue.
        # This completely prevents the FFmpeg rendering worker from starving and skipping
        # the initial 3-4 seconds of video.
        # if getattr(self, "device_input", "cpu") == "cpu":
        #     print("[INIT] Waiting for CPU to prime the render queue...")
        #     timeout_counter = 0
        #     # Wait until at least 5 frames are buffered in the queue or 10 seconds pass
        #     while self.render_queue.qsize() < 5 and timeout_counter < 100:
        #         time.sleep(0.1)
        #         timeout_counter += 1
        #     print(
        #         f"[INIT] Render queue primed with {self.render_queue.qsize()} frames. Launching worker."
        #     )

        if not self.config.DISABLE_DETECTION:
            self.render_proc.start()

            self.display_proc.start()

        if (
            self.config.ENABLE_QUERYING
            and not self.config.TEST_MODE
            and not self.metadata_thread.is_alive()
        ):
            self.metadata_thread.start()

        if self.config.ENABLE_QUERYING and not self.writer_thread.is_alive():
            self.writer_thread.start()

        return self

    # def stopv1(self):
    #     """
    #     Comprehensive resource release. Safely drains the frame pipelines,
    #     forces a graceful FFmpeg flush to prevent 'moov atom' index corruption,
    #     and cleanly unlinks shared memory layers.
    #     """
    #     with self._stop_lock:
    #         if self._is_stopped:
    #             return  # Already stopped by another thread

    #         # 1. Instantly pull out of active dashboards to stop inbound traffic
    #         if self.name in self.active_streams:
    #             self.active_streams.pop(self.name, None)

    #         self.active = False
    #         print(
    #             f"[STOP] Initiating graceful flush shutdown for {self.name}",
    #             flush=True,
    #         )

    #         # 2. Trigger your event handler flags
    #         if hasattr(self, "stop_writer") and self.stop_writer is not None:
    #             try:
    #                 self.stop_writer.set()
    #             except Exception:
    #                 pass

    #         # 3. PHASE 1: UNBLOCK CONSUMER CORES (Poison Pill Deliveries First)
    #         if hasattr(self, "write_queue") and self.write_queue is not None:
    #             try:
    #                 # Clear out pending frame backlogs to speed up shutdown execution
    #                 while not self.write_queue.empty():
    #                     try:
    #                         self.write_queue.get_nowait()
    #                     except Exception:
    #                         break
    #                 # Dispatch clean poison pill token to release the consumer thread
    #                 self.write_queue.put(None)
    #             except Exception:
    #                 pass

    #         # if hasattr(self, "render_queue") and self.render_queue is not None:
    #         #     try:
    #         #         # while not self.render_queue.empty():
    #         #         #     try:
    #         #         #         self.render_queue.get_nowait()
    #         #         #     except Exception:
    #         #         #         break
    #         #         # self.render_queue.put_nowait(None)
    #         #         while not self.render_queue.empty():
    #         #             time.sleep(
    #         #                 0.01
    #         #             )  # Give the disk writer room to catch up cleanly

    #         #         # Deliver the poison pill token ONLY after the queue is empty
    #         #         self.render_queue.put(None, timeout=2.0)
    #         #     except Exception:
    #         #         pass

    #         if hasattr(self, "render_queue") and self.render_queue is not None:
    #             try:
    #                 # --- THE DEFINITIVE UNBLOCKING FIX ---
    #                 # Drop any lingering items aggressively instead of sleeping infinitely
    #                 # on an empty queue check that never clears.
    #                 while not self.render_queue.empty():
    #                     try:
    #                         self.render_queue.get_nowait()
    #                     except Exception:
    #                         break

    #                 # Immediately push the poison pill sentinel to terminate the worker safely
    #                 self.render_queue.put_nowait(None)
    #             except Exception:
    #                 pass

    #         # 4. PHASE 2: GRACEFUL FFmpeg DEFLATION GATE (Bypasses Moov Issues)
    #         if hasattr(self, "ffmpeg_proc") and self.ffmpeg_proc is not None:
    #             try:
    #                 print(
    #                     "[STOP] Closing video pipeline write handles to flush metadata...",
    #                     flush=True,
    #                 )
    #                 if self.ffmpeg_proc.stdin:
    #                     self.ffmpeg_proc.stdin.close()  # Safely alerts FFmpeg to finalize files

    #                 if self.ffmpeg_proc.stderr:
    #                     self.ffmpeg_proc.stderr.close()  # Instantly forces readline() to return None and exits thread safely

    #                 # Grant a soft 5-second window for storage layer disk synchronization
    #                 self.ffmpeg_proc.wait(timeout=5.0)
    #                 print(
    #                     " [STOP] FFmpeg closed cleanly with valid indexing atoms.",
    #                     flush=True,
    #                 )
    #             except subprocess.TimeoutExpired:
    #                 print(
    #                     " [STOP-WARN] Video flush timed out. Forcing hard termination.",
    #                     flush=True,
    #                 )
    #                 try:
    #                     self.ffmpeg_proc.kill()
    #                 except Exception:
    #                     pass
    #             except Exception as io_err:
    #                 print(
    #                     f" [STOP-WARN] Error during streaming flush: {io_err}",
    #                     flush=True,
    #                 )
    #             finally:
    #                 self.ffmpeg_proc = None
    #                 self.video_writer = None

    #         # 5. PHASE 3: SHUTDOWN MULTIPROCESSING DAEMONS
    #         for proc_attr in ["render_proc", "ai_proc"]:
    #             proc = getattr(self, proc_attr, None)
    #             if proc is not None:
    #                 try:
    #                     if proc.is_alive():
    #                         proc.terminate()
    #                     proc.join(timeout=0.5)
    #                     proc.close()
    #                 except Exception:
    #                     pass
    #                 setattr(self, proc_attr, None)

    #         # 6. PHASE 4: FINAL SEGMENT EVALUATION CONVERGENCE
    #         try:
    #             final_clip_key = f"{self.name}_{self.clip_id:03d}.mp4"
    #             final_clip_path = f"{self.config.SHARED_OUTPUT}/{final_clip_key}"

    #             # Check if the final truncated or partial segment actually exists before dispatching
    #             if (
    #                 os.path.exists(final_clip_path)
    #                 and os.path.getsize(final_clip_path) > 0
    #             ):
    #                 print(
    #                     f" [STOP-FLUSH] Registering finalized terminal clip: {final_clip_key}",
    #                     flush=True,
    #                 )
    #                 global clip_completion_tracker
    #                 if final_clip_key not in clip_completion_tracker:
    #                     clip_completion_tracker[final_clip_key] = {
    #                         "video": False,
    #                         "meta": False,
    #                         "start_time": time.time(),
    #                     }

    #                 clip_completion_tracker[final_clip_key]["video"] = True
    #                 clip_completion_tracker[final_clip_key]["meta"] = True
    #                 self._evaluate_barrier_and_dispatch(
    #                     final_clip_key, final_clip_path, self.resize_w, self.resize_h
    #                 )
    #         except Exception as final_flush_err:
    #             print(
    #                 f" [STOP-WARN] Final segment tracking layer bypass failed: {final_flush_err}",
    #                 flush=True,
    #             )

    #         # 8. PHASE 6: UNMAP HARDWARE MEMORY OBJECTS
    #         if hasattr(self, "shared_details"):
    #             try:
    #                 # Force-close the internal lock primitive hidden inside the Manager dict proxy
    #                 if hasattr(self.shared_details, "_ctx") and hasattr(
    #                     self.shared_details._ctx, "RLock"
    #                 ):
    #                     lock = self.shared_details._ctx.RLock()
    #                     if hasattr(lock, "_semlock"):
    #                         lock._semlock._close()
    #                 self.shared_details.clear()
    #             except Exception:
    #                 pass
    #             del self.shared_details

    #         if (
    #             hasattr(self, "mp_frame_ready_event")
    #             and self.mp_frame_ready_event is not None
    #         ):
    #             try:
    #                 # mp.Event allocates an internal ctx.Cond / ctx.Lock boundary pair
    #                 if hasattr(self.mp_frame_ready_event, "_cond"):
    #                     cond = self.mp_frame_ready_event._cond
    #                     if hasattr(cond, "_lock") and hasattr(cond._lock, "_semlock"):
    #                         cond._lock._semlock._close()
    #             except Exception:
    #                 pass
    #             self.mp_frame_ready_event = None

    #         for q_attr in ["signal_queue", "render_queue"]:
    #             if hasattr(self, q_attr):
    #                 q = getattr(self, q_attr)
    #                 if q is not None:
    #                     try:
    #                         # 1. Flush and break down standard background worker threads safely
    #                         q.close()
    #                         q.join_thread()

    #                         # 2. CRITICAL: Force-close the hidden POSIX lock primitives to clear resource_tracker limits
    #                         if hasattr(q, "_rlock") and q._rlock is not None:
    #                             if hasattr(q._rlock, "_semlock"):
    #                                 q._rlock._semlock._close()

    #                         if hasattr(q, "_writer") and q._writer is not None:
    #                             if hasattr(q._writer, "_semlock"):
    #                                 q._writer._semlock._close()
    #                     except Exception:
    #                         pass
    #                     setattr(self, q_attr, None)

    #         for primitive_attr in [
    #             "ready_buffer_idx",
    #             "reader_active_idx",
    #             "shm_frame_lengths",
    #         ]:
    #             if hasattr(self, primitive_attr):
    #                 obj = getattr(self, primitive_attr)
    #                 if obj is not None and hasattr(obj, "get_lock"):
    #                     try:
    #                         # Grab the hidden lower-level POSIX context lock map
    #                         lock = obj.get_lock()
    #                         # If the lock handle is bound to an active system descriptor, release it
    #                         if hasattr(lock, "_semlock"):
    #                             lock._semlock._close()  # Forces immediate unlinking at the OS level
    #                     except Exception:
    #                         pass
    #                 setattr(self, primitive_attr, None)

    #         if hasattr(self, "manager") and self.manager is not None:
    #             try:
    #                 self.manager.shutdown()
    #             except Exception:
    #                 pass
    #             self.manager = None

    #         if hasattr(self, "pinned_matrices") and self.pinned_matrices:
    #             for active_mat in self.pinned_matrices:
    #                 try:
    #                     cv2.cuda.unregisterPageLocked(active_mat)
    #                 except Exception:
    #                     pass
    #             self.pinned_matrices.clear()
    #             self.pinned_tensors.clear()

    #         # FIXED: Explicitly clear the high-speed AI tensor staging array allocations
    #         if hasattr(self, "ai_pinned_tensors") and self.ai_pinned_tensors:
    #             self.ai_pinned_tensors.clear()

    #         if hasattr(self, "ai_shms") and self.ai_shms:
    #             for shm in self.ai_shms:
    #                 try:
    #                     shm.close()
    #                     shm.unlink()
    #                 except Exception:
    #                     pass
    #             self.ai_shms.clear()
    #             if hasattr(self, "ai_shm_names") and self.ai_shm_names:
    #                 self.ai_shm_names.clear()

    #         if hasattr(self, "shms") and self.shms:
    #             for shm in self.shms:
    #                 shm.close()
    #                 try:
    #                     shm.unlink()
    #                 except FileNotFoundError:
    #                     pass
    #             self.shms.clear()
    #             if (
    #                 hasattr(self, "shared_details")
    #                 and "shm_names" in self.shared_details
    #             ):
    #                 try:
    #                     self.shared_details.pop("shm_names", None)
    #                 except Exception:
    #                     pass

    #         if hasattr(self, "cap") and self.cap is not None:
    #             self.cap.release()
    #             self.cap = None

    #         for pool_name in [
    #             "executor",
    #             # "io_executor",
    #             "clip_executor",
    #             # "ffmpeg_executor",
    #         ]:
    #             if hasattr(self, pool_name) and getattr(self, pool_name) is not None:
    #                 try:
    #                     getattr(self, pool_name).shutdown(wait=True)
    #                 except Exception:
    #                     pass
    #                 setattr(self, pool_name, None)

    #         # self.reader = None
    #         self._is_stopped = True
    #         print(f" [STOP] {self.name} pipeline resources fully released.", flush=True)

    def stop(self):
        """Overrides handlers.py to flush thread objects safely without deadlocking."""
        with self._stop_lock:
            if self._is_stopped:
                return

            print(
                f"\n[PIPELINE-TEARDOWN] Initiating graceful thread drainage for {self.name}",
                flush=True,
            )
            self.active = False
            time.sleep(0.05)

            # 1. Rapidly clear the async video writer thread queue first
            if hasattr(self, "async_writer") and self.async_writer is not None:
                try:
                    self.async_writer.release()
                except Exception:
                    traceback.print_exc()

            # Instantly pull out of active dashboards to stop inbound traffic
            if self.name in self.active_streams:
                self.active_streams.pop(self.name, None)

            if not self.config.TEST_MODE:
                if hasattr(self, "send_metadata_queue"):
                    local_send_metadata_queue = self.send_metadata_queue
                else:
                    global send_metadata_queue
                    local_send_metadata_queue = send_metadata_queue
                if local_send_metadata_queue is not None:
                    try:
                        # 1. Wait for any in-flight ThreadPoolExecutor inference workers
                        # to finish submitting final frame maps to all_metadata
                        if hasattr(self, "executor") and self.executor is not None:
                            while self.get_executor_backlog() > 0:
                                time.sleep(0.01)

                        backlog_count = local_send_metadata_queue.qsize()
                        print(
                            f"[TEARDOWN] send_metadata_queue has {backlog_count} pending uploads. Delivering poison pill sentinel...",
                            flush=True,
                        )

                        # 2. Push the explicit sentinel None token to tell the consumer loop to close
                        local_send_metadata_queue.put(None)

                        # 3. CRITICAL BRIDGING BLOCK: If the thread is active and alive,
                        # block the main process context to let it completely empty the queue
                        if (
                            hasattr(self, "metadata_thread")
                            and self.metadata_thread is not None
                        ):
                            if self.metadata_thread.is_alive():
                                # Force the main thread to join until the consumer hits the sentinel and exits
                                self.metadata_thread.join(timeout=2.0)

                            if self.metadata_thread.is_alive():
                                print(
                                    "[TEARDOWN-WARN] Metadata database thread hung on socket lock. Bypassing safely to prevent deadlock.",
                                    flush=True,
                                )
                            else:
                                print(
                                    "[TEARDOWN] Metadata database thread exited cleanly.",
                                    flush=True,
                                )

                    except Exception as teardown_err:
                        print(
                            f"[TEARDOWN WARN] Metadata drainage exception: {teardown_err}",
                            flush=True,
                        )

            # =====================================================================
            # DECUPLED MULTI-PROCESS TEARDOWN LIFECYCLE SEQUENCE
            # =====================================================================
            # 1. FIRST: Drain and synchronize your thread pool tasks so EVERY frame drops into the IPC queue
            if hasattr(self, "display_pool") and self.display_pool is not None:
                try:
                    if hasattr(self.display_pool, "_work_queue"):
                        backlog = self.display_pool._work_queue.qsize()
                        if backlog > 0:
                            print(
                                f"\n\033[94m[STAGE 1/4] Synchronizing display pool threads. Draining {backlog} background tasks...\033[0m",
                                flush=True,
                            )
                            backlog_s = time.perf_counter()
                            while (
                                self.display_pool._work_queue.qsize() > 0
                                or len(self.reorder_staging_buffer) > 0
                            ):
                                # Yield a brief slice to let worker threads complete downscaling
                                time.sleep(0.01)
                            backlog_e = time.perf_counter() - backlog_s
                            print(f"\t\033[94mTook {backlog_e} secs\033[0m", flush=True)
                    print(
                        "\033[92m[SUCCESS] All background image transformations completed cleanly.\033[0m",
                        flush=True,
                    )
                    self.display_pool.shutdown(wait=True)
                except Exception as e:
                    print(
                        f"[TEARDOWN ERROR] Thread pool synchronization failed: {e}",
                        flush=True,
                    )
                setattr(self, "display_pool", None)

            # 1. FIRST: Instantly dispatch the shutdown token so the worker process knows to flush entries
            if hasattr(self, "render_queue") and self.render_queue is not None:
                try:
                    print(
                        f"\n\033[94m[STAGE 1/3] Dispatching shutdown sentinel (None) to rendering process worker (Backlog: {self.render_queue.qsize()} frames)...\033[0m",
                        flush=True,
                    )
                    # Delivering the poison pill immediately tells FFmpeg to drain remaining blocks sequentially
                    self.render_queue.put(None, timeout=2.0)
                except Exception as queue_err:
                    print(
                        f"\033[91m[TEARDOWN ERROR] Failed staging shutdown token: {queue_err}\033[0m",
                        flush=True,
                    )

            # 2. SECOND: Allow the background multiprocessing queue to empty outstanding items naturally
            if hasattr(self, "render_queue") and self.render_queue is not None:
                try:
                    print(
                        "\n\033[94m[STAGE 2/3] Finalizing file output. Draining remaining frames from IPC memory lane...\033[0m",
                        flush=True,
                    )
                    t_drain = time.perf_counter()
                    # Wait up to 15 seconds for the worker process to pull every remaining matrix out of the pipe
                    while (
                        not self.render_queue.empty()
                        and (time.perf_counter() - t_drain) < 15.0
                    ):
                        time.sleep(
                            0.01
                        )  # Micro-yield grants immediate execution priority to the background process core
                except Exception:
                    pass

            # 3. THIRD: Block the main thread safely and allow the detached process to finish disk I/O operations
            if hasattr(self, "render_proc") and self.render_proc is not None:
                try:
                    join_duration = 0.0
                    if self.render_proc.is_alive():
                        print(
                            "\033[94m[STAGE 2/2] Main thread blocking. Synchronizing hard drive file headers...\033[0m",
                            flush=True,
                        )
                        t_join_start = time.perf_counter()
                        self.render_proc.join(
                            timeout=15.0
                        )  # Safe timeout lets video containers finalize cleanly
                        join_duration = time.perf_counter() - t_join_start
                    else:
                        print(
                            "\033[94m[STAGE 2/2] Main thread clear...\033[0m",
                            flush=True,
                        )

                    if self.render_proc.is_alive():
                        print(
                            "\n\033[91m[CRITICAL STALL] Render worker process hung on disk write! Forcing termination...\033[0m",
                            flush=True,
                        )
                        self.render_proc.terminate()
                        self.render_proc.join()
                    else:
                        print(
                            f"\033[92m[FINAL] Video compilation finished cleanly! Hard drive commit took {join_duration:.2f} seconds.\033[0m",
                            flush=True,
                        )

                    # self.render_proc.close()
                except Exception as proc_err:
                    print(
                        f"\033[91m[TEARDOWN ERROR] Failed closing background process container: {proc_err}\033[0m",
                        flush=True,
                    )
                setattr(self, "render_proc", None)

            render_proc_handle = getattr(self, "render_proc", None)
            if render_proc_handle is not None:
                try:
                    # 1. Check if it is a Process vs Thread
                    if hasattr(render_proc_handle, "terminate"):
                        # Production/Multiprocessing fallback path
                        if render_proc_handle.is_alive():
                            render_proc_handle.terminate()
                            render_proc_handle.join()
                        render_proc_handle.close()
                    else:
                        # Thread-Safe Path: Standard background threading handles
                        # do not require manual termination or descriptor closure hooks.
                        if render_proc_handle.is_alive():
                            render_proc_handle.join(timeout=0.2)

                    print(
                        "\033[92m[SUCCESS] Background rendering thread container released cleanly.\033[0m",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[TEARDOWN WARN] Safely ignoring lingering thread reference handles: {e}",
                        flush=True,
                    )
                finally:
                    setattr(self, "render_proc", None)

            # 4. FOURTH: Clean up remaining pipeline executors and queues safely
            if hasattr(self, "render_queue") and self.render_queue is not None:
                try:
                    self.render_queue.close()
                    self.render_queue.cancel_join_thread()
                except Exception:
                    pass
                setattr(self, "render_queue", None)

            # if hasattr(self, "async_writer") and self.async_writer is not None:
            #     self.async_writer.release()

            if hasattr(self, "executor") and self.executor is not None:
                try:
                    self.executor.shutdown(wait=True)
                except Exception:
                    pass
                setattr(self, "executor", None)

            self.status = "DONE"
            self._is_stopped = True
            print(
                "[STOPPING] Teardown complete. Releasing session cleanly.\n",
                flush=True,
            )

    def model_warmup(self, H=640, W=640):
        H, W = int(H), int(W)
        # Move the dummy input creation inside a no_grad block
        with torch.no_grad():
            print(f"Starting warmup for {self.name}...", flush=True)
            dummy_input = torch.zeros((1, 3, H, W)).to(self.device_input)

            # Perform iterations directly on the main thread
            for i in range(5):
                _ = self.run_model(
                    dummy_input,
                    imgsz=(H, W),
                    batch=1,
                    device_input=self.device_input,
                    stream=STREAM_ARG,
                )
                if i == 0 and hasattr(self.model, "predictor") and self.model.predictor:
                    self.cached_predictor = self.model.predictor

            # Force GPU to finish before returning
            if self.device_input == "cuda":
                torch.cuda.synchronize()

            # Pin a lightweight tensor to prevent downstream empty_cache() calls
            # from destroying the compiled model weight layouts in memory
            self.__persistent_vram_lock = torch.zeros((1,), device=self.device_input)

        print(f"Warmup complete for {self.name}", flush=True)

    def get_executor_backlog(self):
        """Returns the number of tasks currently waiting in the thread pool queue."""
        # access the internal queue of the executor
        # return self.executor._work_queue.qsize()
        if getattr(self, "executor", None) is None:
            return 0
        try:
            # Handle standard ThreadPoolExecutor and custom queue wrappers cleanly
            if hasattr(self.executor, "_work_queue"):
                return self.executor._work_queue.qsize()
            return 0
        except Exception:
            return 0

    def get_clip_executor_backlog(self):
        """Returns the number of tasks currently waiting in the thread pool queue."""
        # access the internal queue of the clip_executor
        # return self.clip_executor._work_queue.qsize()
        if getattr(self, "clip_executor", None) is None:
            return 0
        try:
            if hasattr(self.clip_executor, "_work_queue"):
                return self.clip_executor._work_queue.qsize()
            return 0
        except Exception:
            return 0

    def check_disk_usage(self, path, min_gb=0.5):
        """Returns True if there is at least min_gb available at path."""
        try:
            total, used, free = shutil.disk_usage(path)
            # Convert bytes to Gigabytes
            free_gb = free / (2**30)
            return free_gb > min_gb
        except Exception as e:
            print(f"[EXCEPTION] Disk check error: {e}")
            return False

    # Gets frame W and H details
    def get_frameWH(self):
        if (self.frame_height * self.frame_width) < (
            self.config.MODEL_H * self.config.MODEL_W
        ):
            new_sizeHW = check_imgsz(
                [self.config.MODEL_H, self.config.MODEL_W]
            )  # expects hxw
        else:
            new_sizeHW = check_imgsz(
                [self.frame_height, self.frame_width]
            )  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]
        self.height = new_sizeWH[1]

        # Configure scaling for 8K-to-Model coordinate mapping
        self.resize_h, self.resize_w = [self.config.MODEL_H, self.config.MODEL_W]
        self.scale_x = self.frame_width / self.config.MODEL_W
        self.scale_y = self.frame_height / self.config.MODEL_H

    def update_frame(self, stat_start_time):
        if self.device_input == "cuda":
            torch.cuda.synchronize()

        self.stat_frame_count += 1
        self.elapsed_display_time += time.perf_counter() - stat_start_time
        # if elapsed > 0.5:
        self.stat_fps = round(self.stat_frame_count / self.elapsed_display_time, 1)

    def is_processing(self):
        """Returns True if any part of the pipeline is still active."""
        if not self.reader.stopped:
            return True

        if (
            self.process_thread is not None and self.process_thread.is_alive()
        ):  # or not self.reader.frame_queue.empty():
            return True

        if not self.reader.frame_queue.empty():
            return True

        q_size = self.write_queue.qsize()
        if self.process_thread is not None:
            print(
                f"[STATUS] Writer: {self.write_count}/{self.frame_count_target} | Q: {q_size} | Inf: {'Alive' if self.process_thread.is_alive() else 'Dead'}",
                end="\r",
                flush=True,
            )
        else:
            print(
                f"[STATUS] Writer: {self.write_count}/{self.frame_count_target} | Q: {q_size}",
                end="\r",
                flush=True,
            )

        if not self.write_queue.empty():
            return True
        if self.write_count < self.frame_count_target:
            # q_size = self.write_queue.qsize()
            # print(f"[DRAIN] Writer: {self.write_count}/{self.frame_count} | Queue: {q_size}", end="\r", flush=True)
            return True
        return False

    def run_model(
        self,
        frame,
        imgsz=(BASE_PIPELINE_CONFIG.MODEL_H, BASE_PIPELINE_CONFIG.MODEL_W),
        batch=1,
        device_input="cuda",
        stream=False,
    ):
        # --- DEBUG VERIFICATION ---
        # if torch.is_tensor(frame):
        #     # We want to see [N, 3, 640, 640] where N > 1
        #     print(f"[DEBUG] Tensor Input Shape: {frame.shape}")
        # elif isinstance(frame, list):
        #     print(f"[DEBUG] List Input Length: {len(frame)}")

        # print(f"[DEBUG] Stream Mode: {stream} | Requested Batch: {batch}")
        # ---------------------------
        if isinstance(frame, torch.Tensor):
            # Ensure on the right device
            frame = frame.to(device_input)

            # Make sure input is multiple of 32
            h, w = frame.shape[-2:]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32

            if pad_h > 0 or pad_w > 0:
                # F.pad for 4D tensor (B, C, H, W) uses (left, right, top, bottom)
                frame = F.pad(frame, (0, pad_w, 0, pad_h), value=0)

        # with torch.inference_mode():
        #     torch.set_num_threads(1)
        #     results = self.model.predict(
        #         frame,
        #         imgsz=imgsz,
        #         batch=batch,
        #         device=device_input,
        #         verbose=False,
        #         stream=stream,
        #         conf=self.config.DETECTION_THRESHOLD,
        #         max_det=self.config.MAX_DETECTIONS,
        #         rect=(batch == 1),  # False,  #
        #     )
        # return results
        with torch.inference_mode():
            if device_input == "cpu":
                # --- SAFE HIGH-SPEED FRAMEWORK INGESTION GATE ---
                # If input is an NCHW PyTorch tensor or transposed array view,
                # decode it back into a standard list of un-transposed [H, W, C] uint8
                # image frames before handing it over to the predictor engine.
                if isinstance(frame, torch.Tensor):
                    # Convert NCHW tensor back to standard NHWC layout standard
                    np_frames = frame.detach().cpu().permute(0, 2, 3, 1).numpy()
                    # Rescale back to uint8 pixel range
                    np_frames = (np_frames * 255.0).astype(np.uint8)
                    # Unpack the batch array dimensions into a standard list of standalone images
                    # ingestion_source = [np.ascontiguousarray(img) for img in np_frames]
                    np_frames_aligned = np.copy(np_frames, order="C")
                    ingestion_source = list(np_frames_aligned)
                else:
                    ingestion_source = frame
            else:
                ingestion_source = frame

            if getattr(self, "cached_predictor", None) is not None:
                results = self.cached_predictor(frame)
            else:
                # Execute prediction using the safe public API path.
                # Passing raw standard [H, W, C] frame matrices handles all sigmoid bounding box
                # decoders and confidence multipliers natively, correcting the top-left clumping bug.
                results = self.model.predict(
                    ingestion_source,
                    imgsz=imgsz,
                    batch=batch,
                    device=device_input,
                    verbose=False,
                    stream=stream,
                    conf=self.config.DETECTION_THRESHOLD,
                    max_det=self.config.MAX_DETECTIONS,
                    rect=(batch == 1),
                    profile=False,  # Disable interior torch.cuda.synchronize() gates completely!
                )
            return results

    def _check_shm_safety(self, threshold_percent=90):
        """
        Scans /dev/shm and deletes the oldest .mp4 files if usage exceeds threshold.
        This prevents the 8K stream from crashing the entire container.
        """
        # Check current usage of the RAM disk
        usage = shutil.disk_usage("/dev/shm")
        percent_used = (usage.used / usage.total) * 100

        if percent_used > threshold_percent:
            print(
                f" [CRITICAL] /dev/shm usage at {percent_used:.1f}%. Purging old clips..."
            )

            # Get all .mp4 files in /dev/shm sorted by oldest first
            shm_path = Path("/dev/shm")
            clips = sorted(shm_path.glob("*.mp4"), key=lambda x: x.stat().st_mtime)

            # Delete files until we are under 70% usage or run out of files
            for clip in clips:
                try:
                    # Don't delete the file the current writer is actively using!
                    # if str(clip) == self.tmp_file:
                    #     continue

                    clip.unlink()
                    print(f"[PURGE] Deleted {clip.name} to free RAM.")

                    # Re-check usage after each deletion
                    usage = shutil.disk_usage("/dev/shm")
                    if (usage.used / usage.total) * 100 < 70:
                        break
                except Exception as e:
                    print(f"[EXCEPTION] Could not purge {clip}: {e}")

    def run_realtime_inference(self, sf_enabled):
        """Producer: Maintains the target FPS and updates clip IDs."""
        # Calculate a dynamic limit: tolerate 0.5 seconds of lag.
        # If target_fps is 15, the limit is 7. If target_fps is 30, the limit is 15.
        self.dynamic_limit = max(2, int(0.5 * self.target_fps))
        last_frame_time = time.perf_counter()
        while self.active:
            try:
                # FRAME RETRIEVAL ---------------------------------------------
                try:
                    ret, frame_8k, frame_num = self.reader.read()
                    if not ret or frame_8k is None:
                        print(
                            "[CLIPPER-TARGET] End of video file source detected. Initiating graceful shutdown.",
                            flush=True,
                        )
                        # if self.reader is None or (
                        #     hasattr(self.reader, "stopped") and self.reader.stopped
                        # ):
                        if self.device_input == "cuda":
                            torch.cuda.synchronize()

                        if hasattr(self, "executor") and self.executor is not None:
                            while self.get_executor_backlog() > 0:
                                time.sleep(0.01)

                        # Instead of only processing the active clip_id, loop through every clip
                        # listed in all_metadata to ensure nothing is dropped at shutdown.
                        global all_metadata, clip_completion_tracker
                        active_tracked_keys = list(all_metadata.keys())

                        for target_clip_key in active_tracked_keys:
                            # Reconstruct the explicit path structure for each target clip segment
                            target_clip_filename = (
                                f"{self.config.SHARED_OUTPUT}/{target_clip_key}"
                            )

                            if target_clip_key not in clip_completion_tracker:
                                clip_completion_tracker[target_clip_key] = {
                                    "video": False,
                                    "meta": False,
                                    "start_time": time.time(),
                                }

                            # Mark tracking convergence fields to True concurrently
                            clip_completion_tracker[target_clip_key]["video"] = True
                            clip_completion_tracker[target_clip_key]["meta"] = True

                            print(
                                f" [CLIPPER-TARGET] Finalizing terminal sync barrier for: {target_clip_key}",
                                flush=True,
                            )

                            # Safely delegate the queue pushing directly to your validation handler
                            self._evaluate_barrier_and_dispatch(
                                target_clip_key,
                                target_clip_filename,
                                self.resize_w,
                                self.resize_h,
                            )

                        # clip_id = (self.frame_count - 1) // self.max_frames_per_clip
                        # clip_filename = f"{self.config.SHARED_OUTPUT}/{self.name}_{clip_id:03d}.mp4"
                        # clip_key = Path(clip_filename).name
                        # global clip_completion_tracker
                        # if "clip_completion_tracker" in globals() or "clip_completion_tracker" in locals():
                        #     if clip_key not in clip_completion_tracker:
                        #         clip_completion_tracker[clip_key] = {
                        #             "video": False,
                        #             "meta": False,
                        #             "start_time": time.time(),
                        #         }
                        #     # Force both verification boundaries to True concurrently
                        #     clip_completion_tracker[clip_key]["video"] = True
                        #     clip_completion_tracker[clip_key]["meta"] = True

                        # print(
                        #     f" [BARRIER-SEAL] All metadata extracted for {clip_key}. Evaluating convergence...",
                        #     flush=True,
                        # )
                        # self._evaluate_barrier_and_dispatch(
                        #     clip_key,
                        #     clip_filename,
                        #     self.resize_w,
                        #     self.resize_h,
                        # )

                        # global send_metadata_queue, all_metadata
                        # if clip_key in all_metadata:
                        #     # Signal to process metadata for previous cli
                        #     if (
                        #         "send_metadata_queue" in globals()
                        #         or "send_metadata_queue" in locals()
                        #     ):
                        #         try:
                        #             send_metadata_queue.put(
                        #                 (
                        #                     clip_filename,
                        #                     self.resize_w,
                        #                     self.resize_h,
                        #                 )
                        #             )
                        #         except Exception as queue_err:
                        #             print(
                        #                 f"[CLIPPER-WARN] Metadata queue push skipped or unallocated: {queue_err}",
                        #                 flush=True,
                        #             )

                        self.active = False
                        break
                        # continue
                    # if torch.is_tensor(device_frame):
                    #     # Tell the main thread's stream to wait for the upload stream's completion
                    #     torch.cuda.current_stream().wait_stream(self.reader.upload_stream)
                except queue.Empty:
                    if getattr(self.reader, "reconnect_failed", False):
                        self.active = False
                        break
                    time.sleep(0.002)
                    continue
                except Exception:
                    traceback.print_exc()

            except queue.Empty:
                if getattr(self.reader, "reconnect_failed", False):
                    self.active = False
                    break
                time.sleep(0.002)
                continue

            calculated_clip_id = (frame_num - 1) // self.max_frames_per_clip
            # self.clip_id = calculated_clip_id
            # print(f"frame_num: {frame_num} calculated_clip_id: {calculated_clip_id}", flush=True)
            stat_start_time = time.perf_counter()  # timing to display detection
            self.frame_count += 1
            self.frame_count_target += 1
            self.frame_in_clip_count += 1
            # is_target_frame = True  # float(frame_num) >= self.next_process_idx

            # # Create a dedicated memory surface immediately on the Producer line.
            # # This isolates the VRAM grid before the reader can overwrite it.
            # if self.device_input == "cuda" and torch.is_tensor(device_frame):
            #     isolated_device_frame = device_frame.clone().contiguous()
            # elif isinstance(device_frame, np.ndarray):
            #     # isolated_device_frame = device_frame.copy()
            #     # Zero-copy memory map assignment directly into the dedicated ingestion slot
            #     self.ingest_ring[self.ingest_ring_idx][:] = device_frame
            #     isolated_device_frame = self.ingest_ring[self.ingest_ring_idx]

            #     # Safely step to the next slot in the ring buffer
            #     self.ingest_ring_idx = (
            #         self.ingest_ring_idx + 1
            #     ) % self.ingest_ring_depth
            # else:
            #     isolated_device_frame = device_frame

            # Determine if this frame should be AI or Raw based on backlog
            # But ALWAYS submit to the executor to maintain frame order.
            backlog = self.get_executor_backlog()

            while backlog > 4 and self.active:
                time.sleep(0.005)
                backlog = self.get_executor_backlog()

            def wrapped_fn(*args):
                if self.device_input == "cuda":
                    # Ensure the worker thread switches to your targeted pipeline execution timeline
                    # torch.cuda.set_stream(self.inference_stream)
                    # dev_frame, f_num, target_flag, stat_start_time = args
                    dev_frame, f_num, start_time, clip_id = args
                    # isolated_device_frame = (
                    #     dev_frame.clone()
                    #     if torch.is_tensor(dev_frame)
                    #     else dev_frame.copy()
                    # )
                    # isolated_device_frame = dev_frame

                    with torch.cuda.stream(self.inference_stream):
                        self.pipeline_fn(dev_frame, f_num, start_time, clip_id)

                    # Force a non-blocking device barrier to ensure operations have fully hit VRAM
                    # before releasing the thread context
                    # self.inference_stream.synchronize()
                    # Force the device to catch up before dropping thread scope references
                    # self.inference_stream.synchronize()

                    # Explicitly drop the variable frames from the localized stack frame
                    # if "isolated_device_frame" in locals():
                    #     del isolated_device_frame
                    # if "dev_frame" in locals():
                    #     del dev_frame
                else:
                    # dev_frame, f_num, target_flag, stat_start_time = args
                    dev_frame, f_num, start_time, clip_id = args
                    # isolated_device_frame = (
                    #     dev_frame.clone()
                    #     if torch.is_tensor(dev_frame)
                    #     else dev_frame.copy()
                    # )
                    # isolated_device_frame = dev_frame
                    self.pipeline_fn(dev_frame, f_num, start_time, clip_id)

            # if is_target_frame:  # timing to display detection
            #     self.next_process_idx += self.step_size
            # Handoff to AI and Writer
            # if self.active:
            # Clone the tensor buffer immediately on the producer thread
            # to prevent upstream overwrite races by the next reader iteration.
            # isolated_device_frame = device_frame.clone() if torch.is_tensor(device_frame) else device_frame.copy()
            self.executor.submit(
                # pipeline_fn,
                wrapped_fn,
                frame_8k,
                self.frame_count,
                # is_target_frame,
                stat_start_time,
                calculated_clip_id,
            )
            # else:
            #     # Process background execution context for skipped frames
            #     self.pipeline_fn(device_frame, frame_num, is_target_frame)

            # if self.device_input == "cuda":
            #     torch.cuda.synchronize()

            # --- PRECISE CLOCK SYNC ---
            # This prevents the producer from "lapping" the consumer
            # and building that jumpy backlog in the first place.
            elapsed = time.perf_counter() - last_frame_time
            if elapsed < self.frame_interval:
                # time.sleep(self.frame_interval - elapsed)
                # Subtract a small epsilon (0.001) for OS scheduling overhead
                sleep_duration = max(0, self.frame_interval - elapsed - 0.0015)
                if sleep_duration > 0.001:
                    time.sleep(sleep_duration)
            last_frame_time = time.perf_counter()

            self.update_frame(stat_start_time)
            self.last_heartbeat = time.time()

        self.stop()

    def filter_contained_boxes(self, boxes, overlap_thresh=0.9):
        """
        Vectorized IoA filter: Removes boxes if most of their area is inside another box.
        """
        if boxes.shape[0] <= 1:
            return boxes

        # Calculate Areas
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        areas = w * h
        valid_mask = (
            (w < (self.resize_w * self.config.ROI_MAX_RELATIVE_SIZE_RATIO))
            & (h < (self.resize_h * self.config.ROI_MAX_RELATIVE_SIZE_RATIO))
            & (w > 0)
            & (h > 0)
            & (areas >= self.min_contour_area)
        )
        boxes = boxes[valid_mask]
        areas = areas[valid_mask]

        # Compute all-to-all Intersections [N, N]
        lt = torch.max(boxes.unsqueeze(1)[:, :, :2], boxes.unsqueeze(0)[:, :, :2])
        rb = torch.min(boxes.unsqueeze(1)[:, :, 2:], boxes.unsqueeze(0)[:, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter_area = wh[:, :, 0] * wh[:, :, 1]

        # Intersection over Area (How much of Box A is in Box B)
        # ioa[i, j] = (Box i ∩ Box j) / Area(i)
        ioa = inter_area / (areas.unsqueeze(1) + 1e-6)

        # Filter logic:
        # Only remove if Box J is LARGER than Box I and overlap is high
        diag = torch.eye(boxes.shape[0], device=boxes.device, dtype=torch.bool)
        larger_mask = areas.unsqueeze(0) >= areas.unsqueeze(1)

        to_remove = (ioa > overlap_thresh) & larger_mask & ~diag
        return boxes[~to_remove.any(dim=1)]

    def get_gpu_rois_by_area(self, mask, max_candidates=100):
        # Extract true spatial constraints straight from the active mask object footprint
        if torch.is_tensor(mask):
            mask_h, mask_w = mask.shape[-2:]
        elif isinstance(mask, cv2.cuda.GpuMat):
            # cv2.cuda.GpuMat.size() returns a tuple of (width, height) standard formatting
            mask_w, mask_h = mask.size()
        else:
            mask_h, mask_w = mask.shape[:2]

        # This prevents find_contours_gpu_equivalent from mutating the mask variables used by other threads.
        if isinstance(mask, cv2.cuda.GpuMat):
            # .clone() allocates a new C++ memory surface and forces full continuity
            isolated_kernel_mask = mask.clone()
        elif torch.is_tensor(mask):
            isolated_kernel_mask = mask.clone().contiguous()
        else:
            isolated_kernel_mask = mask.copy()

        # Get raw boxes from mask (Direct VRAM bridge)
        boxes_gpu = find_contours_gpu_equivalent(
            isolated_kernel_mask,
            stream=self.bgs_stream,
            limit_640=640 * 1.5,
        )

        # --- FIX: ELIMINATE STREAM RACE ---
        if boxes_gpu is None or len(boxes_gpu) == 0:
            return torch.empty((0, 4), device=self.device_input)

        # Wrap existing GPU memory as a float tensor (Zero Copy)
        # raw_boxes = torch.as_tensor(boxes_gpu, device=self.device_input).float()
        if self.device_input == "cuda":
            # Wrap the native device handle and IMMEDIATELY append .clone()
            # This allocates a brand new, physically isolated VRAM block to secure the bounding boxes
            raw_boxes = (
                torch.as_tensor(boxes_gpu, device=self.device_input).float().clone()
            )
        else:
            raw_boxes = torch.as_tensor(boxes_gpu, device=self.device_input).float()

        if raw_boxes is not None and len(raw_boxes) > 0:
            PADDING_PX = 5
            # Assumes merged_boxes_tensor is a 2D tensor of shape [N, 4] containing [x1, y1, x2, y2]
            # Create a matching subtraction/addition padding mask tensor
            # Subtract from x1, y1 (indices 0, 1) and add to x2, y2 (indices 2, 3)
            padding_mask = torch.tensor(
                [-PADDING_PX, -PADDING_PX, PADDING_PX, PADDING_PX],
                device=raw_boxes.device,
                dtype=raw_boxes.dtype,
            )

            # Apply padding to all bounding boxes concurrently via broad-vector math
            padded_tensor = raw_boxes + padding_mask

            # Guard rails: Clamp boundaries in-place to stay safely within the 8K master frame
            # Indices 0 and 2 are X coordinates bounded by frame width; 1 and 3 are Y coordinates bounded by frame height
            padded_tensor[:, 0].clamp_(min=0, max=self.resize_w)
            padded_tensor[:, 1].clamp_(min=0, max=self.resize_h)
            padded_tensor[:, 2].clamp_(min=0, max=self.resize_w)
            padded_tensor[:, 3].clamp_(min=0, max=self.resize_h)

            # Re-assign back to your pipeline's tracking variable
            raw_boxes = padded_tensor

        # Vectorized Pre-Filter (Removes noise blobs before merging)
        w = raw_boxes[:, 2] - raw_boxes[:, 0]
        h = raw_boxes[:, 3] - raw_boxes[:, 1]
        mask_filter = (w * h > self.min_contour_area) & (w < mask_w) & (h < mask_h)
        raw_boxes = raw_boxes[mask_filter]

        # Prevents N^2 distance matrix from exploding during high noise
        if raw_boxes.shape[0] > max_candidates:
            # Prioritize the largest blobs (most likely to be drones)
            areas = (raw_boxes[:, 2] - raw_boxes[:, 0]) * (
                raw_boxes[:, 3] - raw_boxes[:, 1]
            )
            _, indices = torch.topk(areas, max_candidates)
            raw_boxes = raw_boxes[indices]
        return raw_boxes

    def get_cpu_rois(self, frame, frameNum, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes_xywh = [
            list(cv2.boundingRect(c))
            for c in contours
            if cv2.contourArea(c) > self.min_contour_area
        ]

        raw_boxes = [
            [
                max(0, int(x) - PADDING_PX),
                max(0, int(y) - PADDING_PX),
                min(self.resize_w, int(x + w) + PADDING_PX),
                min(self.resize_h, int(y + h) + PADDING_PX),
            ]
            for x, y, w, h in raw_boxes_xywh
        ]
        # raw_boxes = [[x, y, x + w, y + h] for x, y, w, h in raw_boxes_xywh]

        if len(raw_boxes) < 1:
            return torch.empty((0, 4), device=self.device_input)

        if len(raw_boxes) > 1:
            limit_640 = (640 * 2.0) / self.scale_x
            raw_boxes = merge_boxes_cpu(
                raw_boxes, gap_limit=self.dist_thresh_640, size_limit=limit_640
            )

        raw_boxes_640p = torch.tensor(raw_boxes, device=self.device_input).float()

        clean_640p = self.filter_contained_boxes(
            raw_boxes_640p, overlap_thresh=self.config.ROI_CONTAINMENT_THRESH
        )

        if clean_640p.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        # Scale to 8K space
        return clean_640p * self.scales_tensor

        # Project clusters back to 8K and build centered YOLO windows
        # final_8k_rois = scale_clusters_to_8k(
        #     merged_640, frame_w=self.frame_width, frame_h=self.frame_height
        # )
        # return torch.tensor(final_8k_rois, device=self.device_input).float()

    def get_detections(
        self, frame, frame_id, thickness=2, device_input="cuda", merged=None
    ):
        """
        Processes full-frame 8K targets or aggregates smart-filtered bounding-box regions
        into uniform tensor arrays for batched inference execution.
        """
        metadata = {}
        is_cuda = device_input == "cuda"
        num_objs = 0

        # =====================================================================
        # 🚀 PATH 1: FULL RESOLUTION TRACK (sf_enabled = False)
        # =====================================================================
        if merged is None:
            with torch.inference_mode():
                if isinstance(frame, torch.Tensor):
                    # # Transpose to channel-first shape format [1, C, H, W] if layout is trailing
                    # if frame.ndim == 3 and frame.shape[-1] == 3:
                    #     frame = frame.permute(2, 0, 1).unsqueeze(0).clone().contiguous()
                    # elif frame.ndim == 3:
                    #     frame = frame.unsqueeze(0).clone().contiguous()

                    # # Create a memory-contiguous layout view and cast cleanly inside VRAM
                    # frame = frame.contiguous()

                    # Transpose to channel-first shape format [1, C, H, W] if layout is trailing
                    if frame.ndim == 3 and frame.shape[-1] == 3:
                        # CRITICAL BUG FIX: Appending .clone().contiguous() creates a brand new,
                        # physically isolated tensor memory block. This breaks the link to
                        # temporary buffers, preventing the VS Code debugger from crashing on evaluation.
                        frame = frame.permute(2, 0, 1).unsqueeze(0).clone().contiguous()
                    elif frame.ndim == 3:
                        frame = frame.unsqueeze(0).clone().contiguous()
                    else:
                        # Ensure any multi-dimensional batch views are physically packed
                        frame = frame.clone().contiguous()

                    if frame.dtype == torch.uint8:
                        frame = frame.to(
                            device_input, dtype=torch.float16, non_blocking=True
                        )
                        frame.mul_(1.0 / 255.0)  # Safe in-place float normalization
                    else:
                        frame = frame.to(device_input, non_blocking=True)

                    img_size = frame.shape[-2:]
                else:
                    # CPU / Host NumPy NDArray fallback
                    img_size = frame.shape[:2]

                # img_size = (self.resize_h, self.resize_w)
                H, W = img_size
                scale_display_x = self.resize_w / W  # 640 / 8192
                scale_display_y = self.resize_h / H  # 640 / 4608
                results = self.run_model(
                    frame,
                    imgsz=img_size,
                    batch=1,
                    device_input=device_input,
                    stream=STREAM_ARG,
                )

            # Extract full resolution detections
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for idx, box in enumerate(boxes):
                        # coords = box.xywh[0].cpu().tolist()  # [x_center, y_center, width, height]
                        # cls_id = int(box.cls[0].cpu().item())
                        # conf = float(box.conf[0].cpu().item())
                        coords = (
                            box.xywh.cpu().squeeze().tolist()
                        )  # Converts [x_center, y_center, w, h] safely
                        cls_id = int(box.cls.cpu().item())
                        class_name = self.label_source[cls_id]
                        confidence = float(box.conf.cpu().item())

                        # Guard against un-squeezed structural lists
                        if isinstance(coords[0], list):
                            coords = coords[0]

                        # Convert center bounds coordinates back to upper-left origin layout standard
                        # and scale to 640x640
                        disp_x = (coords[0] - (coords[2] / 2.0)) * scale_display_x
                        disp_y = (coords[1] - (coords[3] / 2.0)) * scale_display_y
                        disp_w = coords[2] * scale_display_x
                        disp_h = coords[3] * scale_display_y

                        if disp_w > 2 and disp_h > 2:
                            # Resized
                            object_res = [
                                int(disp_x),  # int(abs_x1 * scale_x),
                                int(disp_y),  # int(abs_y1 * scale_y),
                                int(disp_h),  # int(height * scale_y),
                                int(disp_w),  # int(width * scale_x),
                                class_name,
                                confidence,
                                int(self.resize_h),
                                int(self.resize_w),
                            ]

                            obj_id = len(metadata)
                            num_objs += 1
                            framenum_str = f"{frame_id:04d}_{obj_id:04d}"
                            metadata[framenum_str] = {
                                "frameId": int(frame_id),
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

            if "boxes" in locals():
                del boxes
            if "results" in locals():
                del results

            del frame
            return metadata, num_objs

        # =====================================================================
        # PATH 2: SMART FILTER ROIs TRACK (sf_enabled = True) -> PADDED ASPECT PRESERVING 📦
        # =====================================================================
        else:
            roi_patches = []
            patch_coordinates = []
            max_batch_size = getattr(self.config, "MODEL_MAX_BATCH_SIZE", 64)

            # 1. Canvas Matrix Pre-processing Isolation
            if is_cuda and isinstance(frame, torch.Tensor):
                src_tensor = frame.squeeze(0) if frame.ndim == 4 else frame
                if src_tensor.shape[-1] == 3:
                    src_tensor = src_tensor.permute(2, 0, 1)  # Force [C, H, W]
                src_h, src_w = src_tensor.shape[-2:]
            else:
                src_tensor = np.asarray(frame)
                src_h, src_w = src_tensor.shape[:2]

            # Target layout constraints
            th, tw = self.resize_h, self.resize_w

            # 2. Extract, Aspect-Scale, and Pad Regions of Interest (ROIs)
            for box in merged:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(src_w, x2), min(src_h, y2)

                box_w, box_h = x2 - x1, y2 - y1
                if box_w < 8 or box_h < 8:
                    continue  # Filter out invalid spatial artifacts

                if is_cuda and isinstance(src_tensor, torch.Tensor):
                    with torch.no_grad():
                        crop = src_tensor[:, y1:y2, x1:x2]

                        # Calculate aspect-preserving scale factor
                        scale = min(tw / box_w, th / box_h)
                        nw, nh = int(box_w * scale), int(box_h * scale)
                        nw, nh = max(1, nw), max(1, nh)

                        # Dynamic scaling: downsample or upsample using clean bilinear grids
                        if (box_h, box_w) != (nh, nw):
                            crop_resized = F.interpolate(
                                crop.unsqueeze(0).float(),
                                size=(nh, nw),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0)
                        else:
                            crop_resized = crop.float()

                        # Force the canvas tracking allocation to occur safely inside your active stream scope context
                        with torch.cuda.stream(self.inference_stream):
                            # Instantiate a clean, pre-allocated padded evaluation canvas context
                            padded_canvas = torch.zeros(
                                (3, th, tw), dtype=torch.float32, device=device_input
                            )

                            # Center the aspect-scaled crop onto the dark zero-padded mask grid
                            dx = (tw - nw) // 2
                            dy = (th - nh) // 2
                            padded_canvas[:, dy : dy + nh, dx : dx + nw] = crop_resized

                            # Convert directly to half-precision to speed up pipeline passes
                            roi_patches.append(padded_canvas.to(torch.half))
                else:
                    # CPU Path: Aspect-preserving resize and padding via OpenCV
                    crop = src_tensor[y1:y2, x1:x2]
                    scale = min(tw / box_w, th / box_h)
                    nw, nh = max(1, int(box_w * scale)), max(1, int(box_h * scale))

                    crop_resized = cv2.resize(
                        crop, (nw, nh), interpolation=cv2.INTER_NEAREST
                    )

                    # Center-pad the array block with zeros (black)
                    # padded_canvas = np.zeros((th, tw, 3), dtype=np.uint8)
                    padded_canvas = np.empty((th, tw, 3), dtype=np.uint8)
                    padded_canvas.fill(0)
                    dx = (tw - nw) // 2
                    dy = (th - nh) // 2
                    padded_canvas[dy : dy + nh, dx : dx + nw] = crop_resized
                    roi_patches.append(padded_canvas)

                # Store the scaling shifts to map detections back to the 8K coordinate grid accurately
                patch_coordinates.append((x1, y1, box_w, box_h, scale, dx, dy))

            if not roi_patches:
                return {}, 0

            # 3. Process Patches via Multi-Cam Inference Batch Factory
            results_pool = []
            for i in range(0, len(roi_patches), max_batch_size):
                batch_slices = roi_patches[i : i + max_batch_size]
                current_batch_len = len(batch_slices)

                if is_cuda and isinstance(batch_slices[0], torch.Tensor):
                    with torch.inference_mode():
                        torch.cuda.set_stream(self.inference_stream)
                        inference_batch = torch.stack(batch_slices).to(
                            device_input, dtype=torch.float16, non_blocking=True
                        )
                        inference_batch.mul_(1.0 / 255.0)  # In-place GPU normalization

                        batch_res = self.run_model(
                            inference_batch,
                            imgsz=(th, tw),
                            batch=current_batch_len,
                            device_input=device_input,
                            stream=STREAM_ARG,
                        )
                        results_pool.extend(batch_res)
                        del inference_batch
                else:
                    with torch.inference_mode():
                        # np_batch = np.stack(batch_slices, dtype=np.float32)  #.astype(np.float32) / 255.0
                        # Pre-allocate the complete batch block surface memory up front:
                        th, tw = self.resize_h, self.resize_w
                        current_batch_len = len(batch_slices)
                        # np_batch = np.zeros((current_chunk_len, th, tw, 3), dtype=np.float32)

                        # Stack the list of [H, W, C] patches into a 4D array [B, H, W, C]
                        np_batch_uint8 = np.stack(batch_slices)

                        # Transpose from NHWC [B, H, W, C] to NCHW [B, C, H, W] natively in NumPy
                        np_batch_nchw = np.transpose(np_batch_uint8, (0, 3, 1, 2))

                        # Cast to float32 and normalize concurrently using contiguous memory layout
                        # np_batch_contiguous = np.ascontiguousarray(np_batch_nchw, dtype=np.float32)
                        # np_batch_contiguous *= (1.0 / 255.0)

                        # 1. Stack the list of [H, W, C] patches into a 4D array [B, H, W, C]
                        np_batch_uint8 = np.stack(batch_slices)

                        # 2. Transpose from NHWC [B, H, W, C] to NCHW [B, C, H, W] natively in NumPy
                        # np_batch_nchw = np.transpose(np_batch_uint8, (0, 3, 1, 2))

                        # # 3. Cast to float32 and normalize concurrently using an in-place pointer
                        # np_batch_float = np_batch_nchw.astype(np.float32)
                        # np_batch_float *= 1.0 / 255.0

                        # # --- CRITICAL MEMORY ALIGNMENT LOCK ---
                        # # Force the contiguous memory layer allocation AFTER all mathematical manipulations are complete.
                        # # This ensures the underlying C-memory arrays are perfectly packed and sequential, allowing
                        # # OpenVINO to run zero-copy pointer lookups and bypassing the 48.13ms serialization overhead.
                        # np_batch_aligned = np.ascontiguousarray(np_batch_float)

                        # # 4. Expose the memory-aligned array directly to PyTorch as a host tensor view
                        # inference_batch = torch.from_numpy(np_batch_aligned).to(
                        #     device_input
                        # )

                        np_batch_ready = np.ascontiguousarray(np_batch_uint8)
                        np_batch_nchw = np.transpose(np_batch_ready, (0, 3, 1, 2))
                        np_batch_float = np_batch_nchw.astype(np.float32)
                        np_batch_float *= 1.0 / 255.0
                        inference_batch = torch.from_numpy(np_batch_float).to(
                            device_input
                        )

                        batch_res = self.run_model(
                            inference_batch,
                            imgsz=(th, tw),
                            batch=current_batch_len,
                            device_input=device_input,
                            stream=STREAM_ARG,
                        )
                        results_pool.extend(batch_res)
                        del inference_batch

            # =====================================================================
            # UNIFIED STREAM BARRIER: Sync ONCE after all hardware tasks are queued
            # =====================================================================
            if is_cuda and hasattr(self, "inference_stream"):
                self.inference_stream.synchronize()

            # 4. Map Patch Bounding Boxes back onto the Global 8K Frame Coordinates Map
            scale_display_x = tw / float(self.frame_width)
            scale_display_y = th / float(self.frame_height)

            for idx, res in enumerate(results_pool):
                # Calculate the exact matching absolute global coordinate array slot index
                ox, oy, o_width, o_height, scale_f, pad_x, pad_y = patch_coordinates[
                    idx
                ]
                if res.boxes is not None and len(res.boxes) > 0:
                    all_xyxy = res.boxes.xyxy.cpu().numpy()
                    all_clss = res.boxes.cls.cpu().numpy().astype(int)
                    all_confs = res.boxes.conf.cpu().numpy().astype(float)

                    for j in range(len(all_xyxy)):
                        lx1, ly1, lx2, ly2 = all_xyxy[j]
                        class_name = self.label_source[all_clss[j]]
                        confidence = all_confs[j]

                        # Reverse the centering padding offset values
                        lx1_unpadded = lx1 - pad_x
                        ly1_unpadded = ly1 - pad_y
                        lx2_unpadded = lx2 - pad_x
                        ly2_unpadded = ly2 - pad_y

                        # Reverse the aspect ratio scale shift to map back to absolute 8K coordinates
                        global_x1 = ox + (lx1_unpadded / scale_f)
                        global_y1 = oy + (ly1_unpadded / scale_f)
                        global_x2 = ox + (lx2_unpadded / scale_f)
                        global_y2 = oy + (ly2_unpadded / scale_f)

                        # Project directly onto the 640x640 display monitoring layout canvas
                        disp_x = global_x1 * scale_display_x
                        disp_y = global_y1 * scale_display_y
                        disp_w = (global_x2 - global_x1) * scale_display_x
                        disp_h = (global_y2 - global_y1) * scale_display_y

                        if disp_w > 0 and disp_h > 0:
                            object_res = [
                                int(disp_x),
                                int(disp_y),
                                int(disp_h),
                                int(disp_w),
                                class_name,
                                confidence,
                                int(th),
                                int(tw),
                            ]
                            obj_id = len(metadata)
                            num_objs += 1
                            framenum_str = f"{frame_id:04d}_{obj_id:04d}"
                            metadata[framenum_str] = {
                                "frameId": int(frame_id),
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

            if "results_pool" in locals():
                del results_pool
            if "roi_patches" in locals():
                roi_patches.clear()
                del roi_patches

            del frame
            del src_tensor
            return metadata, num_objs

    def get_gpu_rois(self, frame, frameNum, mask):
        # If more than 20% of the screen is moving, don't bother with crops
        # if current_coverage > 0.6:
        #     return torch.tensor([[0, 0, self.frame_width, self.frame_height]], device=self.device_input)

        limit_640 = (
            640 * 2.0
        ) / self.scale_x  # 40  # self.config.ROI_MERGE_SIZE_LIMIT / self.scale_x
        raw_boxes = self.get_gpu_rois_by_area(
            mask, max_candidates=50
        )  # , limit_640=limit_640)
        # padding = 5  # self.config.ROI_BB_FULL_RES_PADDING /  self.scale_x
        # raw_boxes[:, 0] -= padding
        # raw_boxes[:, 1] -= padding
        # raw_boxes[:, 2] += padding
        # raw_boxes[:, 3] += padding

        if raw_boxes.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        if raw_boxes.shape[0] > 1:
            raw_boxes = merge_boxes_gpu(
                raw_boxes,
                gap_limit=self.dist_thresh_640,
                size_limit=limit_640,
            )

        clean_640p = self.filter_contained_boxes(
            raw_boxes, overlap_thresh=self.config.ROI_CONTAINMENT_THRESH
        )

        if clean_640p.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        # # Scale to 8K space
        # # return clean_640p * self.scales_tensor
        # # margin = 0.10
        # # offsets = (clean_640p[:, 2:] - clean_640p[:, :2]) * margin
        # # clean_640p[:, :2] -= offsets
        # # clean_640p[:, 2:] += offsets
        # # 1. Add 30-pixel 'breathing room' (in 640p space)
        # # padding = 40  # self.config.ROI_BB_FULL_RES_PADDING /  self.scale_x
        # # clean_640p[:, 0] -= padding
        # # clean_640p[:, 1] -= padding
        # # clean_640p[:, 2] += padding
        # # clean_640p[:, 3] += padding

        # # 2. Re-merge the padded boxes (connects nearby drones into one clean crop)
        # clean_640p = merge_boxes_gpu(
        #     clean_640p,
        #     gap_limit=self.dist_thresh_640,
        #     size_limit=limit_640,  # self.config.ROI_MERGE_SIZE_LIMIT / self.scale_x,
        # )

        # # Scale to 8K and clamp
        # clean_full = clean_640p * self.scales_tensor
        # # clean_full[:, [0, 2]] = clean_full[:, [0, 2]].clamp(0, self.frame_width)
        # # clean_full[:, [1, 3]] = clean_full[:, [1, 3]].clamp(0, self.frame_height)
        # return clean_full

        xmin = clean_640p[:, 0]
        ymin = clean_640p[:, 1]
        # w = clean_640p[:, 2]
        # h = clean_640p[:, 3]
        # xmax = xmin + w
        # ymax = ymin + h
        xmax = clean_640p[:, 2]
        ymax = clean_640p[:, 3]

        # Stack into standard format layout [xmin, ymin, xmax, ymax]
        standard_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)

        # 4. Scale to absolute 8K workspace dimensions accurately
        return standard_boxes * self.scales_tensor

    def get_gpu_roisv2(self, frame, frameNum, mask):
        # 1. MORPHOLOGICAL NOISE CLEANUP (Crucial for the hillside)
        # Apply an in-place GPU Opening filter to dissolve the tiny white spots
        # while keeping the larger, dense drone clusters perfectly intact.
        if isinstance(mask, cv2.cuda.GpuMat):
            # 3x3 or 5x5 Ellipse/Rect kernel eliminates high-frequency noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph_filter = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_OPEN, mask.type(), kernel
            )
            clean_mask = morph_filter.apply(mask)
        else:
            kernel = np.ones((3, 3), np.uint8)
            clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 2. Extract raw candidate boxes using the clean mask matrix surface
        limit_640 = (640 * 2.0) / self.scale_x
        raw_boxes = self.get_gpu_rois_by_area(
            clean_mask, max_candidates=100
        )  # Increased limit to 100

        if raw_boxes.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        if raw_boxes.shape[0] > 1:
            raw_boxes = merge_boxes_gpu(
                raw_boxes,
                gap_limit=self.dist_thresh_640,
                size_limit=limit_640,
            )

        clean_640p = self.filter_contained_boxes(
            raw_boxes, overlap_thresh=self.config.ROI_CONTAINMENT_THRESH
        )

        if clean_640p.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        # 3. CORRECT [X, Y, W, H] TO [X1, Y1, X2, y2] MATH
        # Map the original width and height parameters back to endpoint coordinates
        xmin = clean_640p[:, 0]
        ymin = clean_640p[:, 1]
        w = clean_640p[:, 2]
        h = clean_640p[:, 3]

        xmax = xmin + w
        ymax = ymin + h

        # Pack into full master coordinate system arrays
        standard_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)

        # 4. Project coordinates back to 8K master canvas space accurately
        return standard_boxes * self.scales_tensor

    def get_gpu_roisv3(self, frame, frameNum, mask):
        # 1. Extract raw candidate boxes from the mask (Already [x1, y1, x2, y2])
        limit_640 = (640 * 2.0) / self.scale_x
        raw_boxes = self.get_gpu_rois_by_area(mask, max_candidates=50)

        if raw_boxes.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        # 2. Merge overlapping fragments directly on the GPU
        if raw_boxes.shape[0] > 1:
            raw_boxes = merge_boxes_gpu(
                raw_boxes,
                gap_limit=self.dist_thresh_640,
                size_limit=limit_640,
            )

        # 3. Filter out containment or noise artifacts
        clean_640p = self.filter_contained_boxes(
            raw_boxes, overlap_thresh=self.config.ROI_CONTAINMENT_THRESH
        )

        if clean_640p.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        # 4. Direct 8K Space Scaling Matrix Multiplication
        # Matches the clean CPU architecture on Page 31, Line 2050
        return clean_640p * self.scales_tensor

    def frame2video(
        self,
        device_frame,
        frameNum,
        metadata_or_bbs,
        class_list,
        # metrics,
    ):
        scale_display_x = self.disp_w / 640
        scale_display_y = self.disp_h / 640
        try:
            if self.device_input == "cuda" and torch.is_tensor(device_frame):
                # 2. Fast Inline VRAM Downscaling into our static memory slot
                # Reshape to (Batch, Channel, Height, Width) seamlessly without duplicating data
                gpu_tensor = device_frame[None, :].permute(0, 3, 1, 2).float()

                resized_tensor = torch.nn.functional.interpolate(
                    gpu_tensor,
                    size=(self.disp_h, self.disp_w),
                    mode="nearest",
                )

                # CONVERTS TO CPU EARLIER (WORKS but lowers fps)
                # hwc_byte_tensor = resized_tensor.squeeze(0).permute(1, 2, 0).byte()
                # cpu_tensor = hwc_byte_tensor.cpu()
                # bgr_contiguous = cpu_tensor.contiguous()

                # Using copy_() avoids creating any runtime allocations or memory-stride drift.
                self.static_gpu_byte_bchw.copy_(resized_tensor.byte())

                # Safely reshape the pre-allocated, locked GPU memory layout back to standard HWC array topology.
                # Since the underlying array layout is strictly contiguous, this permute is guaranteed
                # to be zero-copy on the GPU and free from memory race stalls.
                bgr_contiguous = self.static_gpu_byte_bchw.squeeze(0).permute(1, 2, 0)

                # 1. Grab the current free pinned slot tracking variables from your reader
                d2h_idx = self.reader.d2h_selector
                pinned_tensor_buf = self.reader.d2h_buffers[d2h_idx]

                # c_idx = self.canvas_selector
                # current_canvas = self.static_host_canvases[c_idx]

                # host_tensor_view = torch.as_tensor(current_canvas, device="cpu")

                # 3. Non-blocking asynchronous PCIe DMA Download directly into our page-locked canvas
                with torch.cuda.stream(self.reader.download_stream):
                    # pinned_tensor = torch.from_numpy(current_canvas).cuda()
                    pinned_tensor_buf.copy_(bgr_contiguous, non_blocking=True)

                # Synchronize only the side download stream layout channel
                self.reader.download_stream.synchronize()

                cpu_360p_frame = self.reader.d2h_numpys[d2h_idx]

                # 5. Non-Blocking Push straight to your background AsyncVideoWriter thread pool
                # We pass a direct .copy() slice so the hot loop can instantly reuse the pinned ring buffer
                display_frame = np.array(cpu_360p_frame, copy=True, order="C")

                if display_frame is not None and display_frame.shape[-1] == 3:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                # --- Draw Detection Overlays ---
                if isinstance(metadata_or_bbs, dict):
                    # Object Mode (YOLO Structs)
                    display_frame = get_metadata_overlay(
                        display_frame,
                        metadata_or_bbs,
                        class_list,
                        (scale_display_x, scale_display_y),
                        (self.disp_w, self.disp_h),
                        is_bgr=True,
                    )

                elif metadata_or_bbs is not None:
                    # # Motion / Smart Filtering Overlay Path
                    display_frame = get_bb_overlay(
                        display_frame,
                        metadata_or_bbs,
                        (scale_display_x, scale_display_y),
                        (self.disp_w, self.disp_h),
                    )

                self.async_writer.write_frame(display_frame)
                # self.canvas_selector = 1 - self.canvas_selector
                self.reader.d2h_selector = 1 - self.reader.d2h_selector

            else:
                # Reusable baseline track for CPU execution mappings
                display_frame = cv2.resize(
                    device_frame,
                    (self.disp_w, self.disp_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                # if metrics["bbs"] is not None:
                #     for box in metrics["bbs"]:
                #         cv2.rectangle(cpu_resized, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # --- Draw Detection Overlays ---
                if isinstance(metadata_or_bbs, dict):
                    # Object Mode (YOLO Structs)
                    display_frame = get_metadata_overlay(
                        display_frame,
                        metadata_or_bbs,
                        class_list,
                        (scale_display_x, scale_display_y),
                        (self.disp_w, self.disp_h),
                        is_bgr=True,
                    )

                elif metadata_or_bbs is not None:
                    # # Motion / Smart Filtering Overlay Path
                    display_frame = get_bb_overlay(
                        display_frame,
                        metadata_or_bbs,
                        (scale_display_x, scale_display_y),
                        (self.disp_w, self.disp_h),
                    )
                self.async_writer.write_frame(display_frame)
        except Exception:
            traceback.print_exc()
        return

    def frame2output(self, device_frame, frame_num, metadata_or_bbs, class_list):
        """
        Renders drone detection bounding boxes onto the canvas
        before dispatching to the live UI shared memory stream.
        """
        if not self.active:
            return
        scale_display_x = self.disp_w / 640
        scale_display_y = self.disp_h / 640
        try:
            if self.device_input == "cuda" and torch.is_tensor(device_frame):
                # 2. Fast Inline VRAM Downscaling into our static memory slot
                # Reshape to (Batch, Channel, Height, Width) seamlessly without duplicating data
                gpu_tensor = device_frame[None, :].permute(0, 3, 1, 2).float()

                resized_tensor = torch.nn.functional.interpolate(
                    gpu_tensor,
                    size=(self.disp_h, self.disp_w),
                    mode="nearest",
                )

                # CONVERTS TO CPU EARLIER (WORKS but lowers fps)
                # hwc_byte_tensor = resized_tensor.squeeze(0).permute(1, 2, 0).byte()
                # cpu_tensor = hwc_byte_tensor.cpu()
                # bgr_contiguous = cpu_tensor.contiguous()

                # Using copy_() avoids creating any runtime allocations or memory-stride drift.
                self.static_gpu_byte_bchw.copy_(resized_tensor.byte())

                # Safely reshape the pre-allocated, locked GPU memory layout back to standard HWC array topology.
                # Since the underlying array layout is strictly contiguous, this permute is guaranteed
                # to be zero-copy on the GPU and free from memory race stalls.
                bgr_contiguous = self.static_gpu_byte_bchw.squeeze(0).permute(1, 2, 0)

                # 1. Grab the current free pinned slot tracking variables from your reader
                d2h_idx = self.reader.d2h_selector
                pinned_tensor_buf = self.reader.d2h_buffers[d2h_idx]

                # c_idx = self.canvas_selector
                # current_canvas = self.static_host_canvases[c_idx]

                # host_tensor_view = torch.as_tensor(current_canvas, device="cpu")

                # 3. Non-blocking asynchronous PCIe DMA Download directly into our page-locked canvas
                with torch.cuda.stream(self.reader.download_stream):
                    # pinned_tensor = torch.from_numpy(current_canvas).cuda()
                    pinned_tensor_buf.copy_(bgr_contiguous, non_blocking=True)

                # Synchronize only the side download stream layout channel
                self.reader.download_stream.synchronize()

                cpu_360p_frame = self.reader.d2h_numpys[d2h_idx]

                # 5. Non-Blocking Push straight to your background AsyncVideoWriter thread pool
                # We pass a direct .copy() slice so the hot loop can instantly reuse the pinned ring buffer
                display_frame = np.array(cpu_360p_frame, copy=True, order="C")

                if display_frame is not None and display_frame.shape[-1] == 3:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                # --- Draw Detection Overlays ---
                if isinstance(metadata_or_bbs, dict):
                    # Object Mode (YOLO Structs)
                    display_frame = get_metadata_overlay(
                        display_frame,
                        metadata_or_bbs,
                        class_list,
                        (scale_display_x, scale_display_y),
                        (self.disp_w, self.disp_h),
                        is_bgr=True,
                    )

                elif metadata_or_bbs is not None:
                    # # Motion / Smart Filtering Overlay Path
                    display_frame = get_bb_overlay(
                        display_frame,
                        metadata_or_bbs,
                        (scale_display_x, scale_display_y),
                        (self.disp_w, self.disp_h),
                    )

                self.async_writer.write_frame(display_frame, frame_num)
                # self.canvas_selector = 1 - self.canvas_selector
                self.reader.d2h_selector = 1 - self.reader.d2h_selector

            else:
                # Reusable baseline track for CPU execution mappings
                display_frame = cv2.resize(
                    device_frame,
                    (self.disp_w, self.disp_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                # if metrics["bbs"] is not None:
                #     for box in metrics["bbs"]:
                #         cv2.rectangle(cpu_resized, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # --- Draw Detection Overlays ---
                if isinstance(metadata_or_bbs, dict):
                    # Object Mode (YOLO Structs)
                    display_frame = get_metadata_overlay(
                        display_frame,
                        metadata_or_bbs,
                        class_list,
                        (scale_display_x, scale_display_y),
                        (self.disp_w, self.disp_h),
                        is_bgr=True,
                    )

                elif metadata_or_bbs is not None:
                    # # Motion / Smart Filtering Overlay Path
                    display_frame = get_bb_overlay(
                        display_frame,
                        metadata_or_bbs,
                        (scale_display_x, scale_display_y),
                        (self.disp_w, self.disp_h),
                    )
                self.async_writer.write_frame(display_frame, frame_num)
        except Exception:
            traceback.print_exc()
        # return

    def pipeline_fn(
        self,
        device_frame,
        overall_frame_num,
        # is_target_frame,
        stat_start_time,
        current_clip_id,
    ):
        global all_metadata
        # current_clip_id = self.clip_id
        current_clip_key = f"{self.name}_{current_clip_id:03d}.mp4"
        current_clip_path = f"{self.config.SHARED_OUTPUT}/{current_clip_key}"
        # --- MOTION MASK GENERATION GATE ---
        if self.config.sf_enabled:
            if self.device_input == "cuda":
                bgs_input_frame = (
                    device_frame.byte()
                    if torch.is_tensor(device_frame)
                    else device_frame
                )
                inf_data = self.rbtd_full_gpu(bgs_input_frame)
                # torch.cuda.current_stream().synchronize()
                # if isinstance(inf_data, dict) and "mask" in inf_data:
                #     if torch.is_tensor(inf_data["mask"]):
                #         inf_data["mask"] = inf_data["mask"].contiguous()
                # inf_data = self.rbtd_full_gpu(device_frame)
            else:
                inf_data = self.rbtd_full_cpu(device_frame)
        else:
            inf_data = {}

        # --- PIPELINE AT TARGET RATE ---
        # if not is_target_frame:
        #     return

        # if is_target_frame:
        # self.next_process_idx += self.step_size
        # self.frame_count_target += 1  # 1-indexed
        # self.frame_in_clip_count += 1
        inf_data["frameNum"] = overall_frame_num  # self.frame_count_target

        # --- CLIP GENERATION ---
        if self.config.ENABLE_QUERYING:
            if self.config.DEBUG_FLAG and (
                self.frame_in_clip_count % 15 == 0 or self.frame_in_clip_count == 1
            ):
                print(
                    f"[CLIPPER] Frame progress tracking index: {self.frame_in_clip_count}/{self.max_frames_per_clip} (Overall Frame: {overall_frame_num})",
                    flush=True,
                )

            self.prep_frame_for_video(device_frame, overall_frame_num)

            if self.frame_in_clip_count > self.max_frames_per_clip:
                global clip_completion_tracker
                if current_clip_key not in clip_completion_tracker:
                    clip_completion_tracker[current_clip_key] = {
                        "video": False,
                        "meta": False,
                        "start_time": time.time(),
                    }

                clip_completion_tracker[current_clip_key]["meta"] = True
                print(
                    f" [BARRIER-SEAL] All metadata extracted for {current_clip_key}. Evaluating convergence...",
                    flush=True,
                )
                self._evaluate_barrier_and_dispatch(
                    current_clip_key,
                    current_clip_path,
                    self.resize_w,
                    self.resize_h,
                )

                self.start_new_clip(current_clip_id)

        if not self.config.DISABLE_DETECTION:
            # --- FULL-RESOLUTION ROI EXTRACTION MAPS ---
            bbs_full_res = None
            if self.config.sf_enabled:
                if self.device_input == "cuda":
                    bbs_full_res = self.get_gpu_rois(
                        inf_data["full_frame"],
                        inf_data["frameNum"],
                        inf_data["mask"],
                    )
                else:
                    bbs_full_res = self.get_cpu_rois(
                        inf_data["full_frame"],
                        inf_data["frameNum"],
                        inf_data["mask"],
                    )

            # Isolate raw coordinate matrices out of device graphs to prevent exit race conditions
            clean_bbs = []
            if self.config.sf_enabled and bbs_full_res is not None:
                if torch.is_tensor(bbs_full_res):
                    clean_bbs = bbs_full_res.detach().cpu().numpy()
                else:
                    clean_bbs = np.array(bbs_full_res)

            # If inline optimizations left the tensor in channels-first format (B, C, H, W),
            # restore it to standard standard layout standard (H, W, C) so get_detections
            # can decode bounding boxes accurately without corrupting weights.
            det_frame = device_frame
            # Cast float canvases back to standard unsigned byte tracking arrays
            # so get_detections can safely run its internal float normalizations.
            if torch.is_tensor(det_frame):
                det_frame = det_frame.byte()

                if det_frame.ndim == 4:
                    det_frame = det_frame.squeeze(0).permute(1, 2, 0)
                elif det_frame.shape[0] == 3:
                    det_frame = det_frame.permute(1, 2, 0)

                # print(f"[DEBUG] {current_clip_key}: {len(clean_bbs)} ROIs detected!")

            if self.config.DETECTION_TYPE != "motion":
                # # Object Mode: Run YOLO and prepare metadata
                # if "full_frame" in inf_data:
                #     det_frame = inf_data["full_frame"]
                # else:
                #     det_frame = (
                #         device_frame.clone()
                #         if torch.is_tensor(device_frame)
                #         else device_frame.copy()
                #     )

                merged = clean_bbs if self.config.sf_enabled else None
                # num_bbs = 0 if merged is None else len(clean_bbs)
                # print(f"[DEBUG] {current_clip_key} 'merged' num bbs: {num_bbs}")
                metadata, num_objs = self.get_detections(
                    det_frame,
                    self.frame_in_clip_count,  # self.frame_count_target,
                    merged=merged,
                    thickness=self.config.THICKNESS,
                    device_input=self.config.device_input,
                )

                # num_objs = len(list(metadata.keys()))
                self.total_objects_detected += num_objs
                if self.config.DEBUG_FLAG:
                    meta_keys = ", ".join(list(metadata.keys()))
                    print(
                        f"[DEBUG] {current_clip_key} metadata keys: {meta_keys}",
                        flush=True,
                    )
                if current_clip_key not in all_metadata:
                    all_metadata[current_clip_key] = {"object": {}, "face": {}}

                all_metadata[current_clip_key]["object"].update(metadata)
                # data_to_draw = metadata
                # if self.config.DEBUG_FLAG or
                if overall_frame_num % 50 == 0:
                    print(
                        f"[METADATA-DEBUG A] Logged tracking structures for Frame #{overall_frame_num} into dictionary key: {current_clip_key}. Number detections: {self.total_objects_detected} (+{num_objs})",
                        flush=True,
                    )

                # print(f"Sending to queue", flush=True)

            if self.device_input == "cuda":
                # torch.cuda.synchronize()
                self.inference_stream.synchronize()

            data_to_draw = (
                clean_bbs if self.config.DETECTION_TYPE == "motion" else metadata
            )

            if self.config.TEST_MODE:
                self.frame2video(
                    det_frame,
                    inf_data["frameNum"],
                    data_to_draw,
                    self.label_source,
                )
            else:
                self.frame2output(
                    det_frame,
                    inf_data["frameNum"],
                    data_to_draw,
                    self.label_source,
                )

            # self.update_frame(stat_start_time)

    # VIDEO CLIPPING
    def start_new_clip(self, clip_id):
        """
        Seals the current AI tracking state layout and safely moves the instance metadata references
        to the next sequential file block segment index.
        """
        global clip_completion_tracker, all_metadata

        # Capture context pointers prior to counter mutation steps
        old_clip_id = clip_id  # self.clip_id
        old_clip_key = f"{self.name}_{old_clip_id:03d}.mp4"
        # old_clip_path = f"{self.config.SHARED_OUTPUT}/{old_clip_key}"

        print(
            f" [CLIPPER] Rotating AI context engine tracking timeline layer. Sealing metadata for: {old_clip_key}",
            flush=True,
        )

        # Seal the AI processing side of the tracker
        # if old_clip_key not in clip_completion_tracker:
        #     clip_completion_tracker[old_clip_key] = {"video": False, "meta": False, "start_time": time.time()}

        # clip_completion_tracker[old_clip_key]["meta"] = True

        # # Evaluate the barrier in case the video segment completed before the AI loop reached this gate
        # self._evaluate_barrier_and_dispatch(old_clip_key, old_clip_path, self.resize_w, self.resize_h)

        # Mutate tracker instance metrics parameters for the upcoming segment chunk window
        # self.clip_id += 1
        self.frame_in_clip_count = 1
        self._check_shm_safety(threshold_percent=90)

        log_to_logger(
            f"New clip created: clip frame {self.frame_in_clip_count} ({self.frame_count_target})) of {self.max_frames_per_clip}",
            level="info",
        )

    def prep_frame_for_video(self, device_frame, frame_num):
        # Stops the handler from starting zombie threads during stop() flushes
        if not self.active or self._is_stopped:
            return

        if not hasattr(self, "write_queue") or self.write_queue is None:
            print(
                " [CLIPPER-INIT] Missing write_queue footprint. Provisioning runtime workspace buffer...",
                flush=True,
            )
            self.write_queue = queue.Queue(maxsize=300)
            self.writer_done = False

        if not self.config.TEST_MODE and (
            not hasattr(self, "send_metadata_queue") or self.send_metadata_queue is None
        ):
            print(
                " [CLIPPER-INIT] Binding instance metadata reference array layer dynamically...",
                flush=True,
            )
            self.send_metadata_queue = queue.Queue()

        if not hasattr(self, "stop_writer") or self.stop_writer is None:
            self.stop_writer = threading.Event()

        if (
            not hasattr(self, "writer_thread")
            or self.writer_thread is None
            or not self.writer_thread.is_alive()
        ):
            print(
                " [CLIPPER-INIT] Target worker runtime thread is offline. Provisioning core consumer loop thread...",
                flush=True,
            )
            self.writer_thread = threading.Thread(
                target=self.video_writer_core_loop,
                args=(self.stop_writer,),
                daemon=True,
            )
            self.writer_thread.start()

        if getattr(self, "video_writer", None) is None:
            print(
                " [CLIPPER-INIT] Downstream execution handle is blank. Initializing FFmpeg subprocess daemon...",
                flush=True,
            )
            self._initialize_writer()

        self.clip_executor.submit(self._async_clipper_worker, device_frame, frame_num)

    def _async_clipper_worker(self, device_frame, frame_num):
        try:
            if self.device_input == "cuda":
                with torch.cuda.stream(self.processing_stream):
                    if device_frame.shape[-1] == 3:
                        gpu_ch_first = device_frame.permute(2, 0, 1).contiguous()
                    else:
                        gpu_ch_first = device_frame.contiguous()

                    self.gpu_float_staging[0, :, :, :].copy_(
                        gpu_ch_first, non_blocking=True
                    )

                    gpu_resized = F.interpolate(
                        self.gpu_float_staging,
                        size=(self.resize_h, self.resize_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                    gpu_final = gpu_resized.clamp(0, 255).to(torch.uint8)

                    # If the tensor shape follows PyTorch's standard format (Channels, Height, Width),
                    # we reverse the channel order [0, 1, 2] to [2, 1, 0] (RGB -> BGR)
                    if gpu_final.ndim == 3 and gpu_final.shape[0] == 3:
                        gpu_final = gpu_final[[2, 1, 0], :, :]

                    gpu_contiguous = gpu_final.permute(1, 2, 0).contiguous()

                    active_tensor = self.pinned_tensors[self.gpu_ring_idx]
                    active_tensor.copy_(gpu_contiguous, non_blocking=True)

                if self.slot_events is not None:
                    self.slot_events[self.gpu_ring_idx].record(self.processing_stream)

                self.write_queue.put(
                    {
                        "ring_slot_idx": self.gpu_ring_idx,
                        "frame_num": frame_num,
                        "pipe_handle": self.video_writer,
                    }
                )
                self.gpu_ring_idx = (self.gpu_ring_idx + 1) % self.ring_depth
            else:
                active_matrix = self.pinned_matrices[self.cpu_ring_idx]
                cv2.resize(
                    device_frame,
                    (self.resize_w, self.resize_h),
                    dst=active_matrix,
                    interpolation=cv2.INTER_NEAREST,
                )

                self.write_queue.put(
                    {
                        "ring_slot_idx": self.cpu_ring_idx,
                        "frame_num": frame_num,
                        "pipe_handle": self.video_writer,
                    }
                )
                self.cpu_ring_idx = (self.cpu_ring_idx + 1) % self.ring_depth

        except Exception as e:
            print(
                f"[CRITICAL-CLIPPER-WORKER] Resizing execution loop dropped: {e}",
                flush=True,
            )
            traceback.print_exc()

    def _initialize_writer(self):
        """
        Spawns a persistent background FFmpeg subprocess with native segment-splitting
        capabilities, entirely bypassing the high-overhead Python-side cv2.VideoWriter lifecycle.
        """
        # self.clip_filename_pattern = f"{self.config.SHARED_OUTPUT}/{self.name}_%03d.mp4"
        # self.clip_key = f"{self.name}_{self.clip_id:03d}.mp4"

        # Safe parameter array construction passed directly to kernel, avoiding subshell expansion failures
        clip_duration = int(self.config.CLIP_DURATION)
        ffmpeg_args = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.resize_w}x{self.resize_h}",
            "-r",
            str(int(self.target_fps)),
            "-i",
            "-",
            "-c:v",
            "libx264",  # "mpeg4",  # Or "libx264" if you prefer H.264
            "-crf",
            "23",
            "-f",
            "mpegts",
            "-movflags",
            "faststart",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{clip_duration})",
            "-f",
            "segment",
            "-segment_time",
            f"{clip_duration}",
            "-reset_timestamps",
            "1",
            "-segment_format",
            "mp4",
            # CRITICAL: Force fragmented headers so every chunk has a valid moov atom instantly
            "-segment_format_options",
            "movflags=frag_keyframe+empty_moov+default_base_moof",
            self.clip_filename_pattern,
        ]

        # print(f" [FFMPEG-INIT] Spawning binary pipeline targeted at: {self.clip_filename_pattern}", flush=True)

        try:
            # log_dir = "/home/logs"
            # os.makedirs(log_dir, exist_ok=True)
            # err_log = open(f"{log_dir}/ffmpeg_handler_{self.name}.log", "w")

            self.ffmpeg_proc = subprocess.Popen(
                ffmpeg_args,
                shell=False,  # Secure token delivery
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                # stderr=err_log,
                stderr=subprocess.PIPE,  # Connected directly to high-speed RAM string buffer streaming
                text=False,
                bufsize=0,
            )
            self.video_writer = self.ffmpeg_proc.stdin
            print(
                " [FFMPEG-INIT] Subprocess online. Log stream parser engine initialization sequencing...",
                flush=True,
            )

            # Fire memory loop monitor
            # self.log_parser_thread = threading.Thread(
            #     target=self._ffmpeg_log_parser_loop, daemon=True
            # )
            # self.log_parser_thread.start()

            log_to_logger(
                f"Persistent native-segmenting FFmpeg writer spawned for stream: {self.name}",
                level="info",
            )
        except Exception as e:
            print(
                f"[CRITICAL-FFMPEG] Process spawn aborted at kernel boundary: {e}",
                flush=True,
            )
            self.video_writer = None

    def _ffmpeg_log_parser_loop(self):
        """
        Memory-piped console stream parser. Intercepts segment generation boundary
        milestone markers straight out of VRAM/RAM buffers with zero disk I/O cost.
        """
        global clip_completion_tracker

        while self.active and self.ffmpeg_proc and self.ffmpeg_proc.stderr:
            try:
                raw_line = self.ffmpeg_proc.stderr.readline()
                if not raw_line:
                    break
            except (ValueError, OSError):
                # Safely intercept when stop() forcefully closes the pipe out-of-band
                break
            line = raw_line.decode("utf-8", errors="ignore")

            if "Opening '" in line and "' for writing" in line:
                try:
                    # Isolate text target strings direct from streaming arrays
                    parts = line.split("Opening '")
                    if len(parts) > 1:
                        full_path = parts[1].split("' for writing")[0]
                        target_filename = os.path.basename(full_path)

                        # Parsing string names means the *previous* segment index is finished writing to disk!
                        # Deduce the previous filename key string mathematically
                        try:
                            name_part, ext_part = os.path.splitext(target_filename)
                            prefix, index_str = name_part.rsplit("_", 1)
                            prev_index = int(index_str) - 1
                            if prev_index >= 0:
                                completed_clip_key = (
                                    f"{prefix}_{prev_index:03d}{ext_part}"
                                )
                                completed_clip_path = (
                                    f"{self.config.SHARED_OUTPUT}/{completed_clip_key}"
                                )

                                # =========================================================================
                                # 🛡️ HARD FLUSH & CLOSE GUARD: Wait for OS File Handlers to Stabilize
                                # =========================================================================
                                file_stable = False
                                timeout_gate = 3.0  # Cap the wait window at 3 seconds to avoid blocking AI pipelines
                                start_sync_time = time.time()
                                last_size = -1

                                while (time.time() - start_sync_time) < timeout_gate:
                                    if os.path.exists(completed_clip_path):
                                        try:
                                            current_size = os.path.getsize(
                                                completed_clip_path
                                            )
                                            # Ensure the file isn't empty AND its byte size has stopped fluctuating
                                            if (
                                                current_size > 0
                                                and current_size == last_size
                                            ):
                                                # Final security check: Test if we can open the file exclusively
                                                # (Confirms FFmpeg or OS has released its write-lock completely)
                                                with open(
                                                    completed_clip_path, "rb+"
                                                ) as _:
                                                    pass
                                                file_stable = True
                                                break
                                            last_size = current_size
                                        except IOError:
                                            # File is still locked by the OS writer, loop and wait
                                            pass
                                    time.sleep(
                                        0.05
                                    )  # Rest the polling thread to preserve CPU cycles

                                if not file_stable:
                                    print(
                                        f" [PARSER-WARN] IO Flush timeout exceeded for {completed_clip_key}. Forcing dispatch anyway.",
                                        flush=True,
                                    )
                                # =========================================================================

                                # Atomic barrier lookup operation update
                                if completed_clip_key not in clip_completion_tracker:
                                    clip_completion_tracker[completed_clip_key] = {
                                        "video": False,
                                        "meta": False,
                                        "start_time": time.time(),
                                    }

                                clip_completion_tracker[completed_clip_key]["video"] = (
                                    True
                                )
                                if self.config.DEBUG_FLAG:
                                    print(
                                        f"[PARSER] Memory pipe intercepted completed segment confirmation: {completed_clip_key}",
                                        flush=True,
                                    )

                                # Run convergence evaluation pass
                                self._evaluate_barrier_and_dispatch(
                                    completed_clip_key,
                                    completed_clip_path,
                                    self.resize_w,
                                    self.resize_h,
                                )
                        except Exception as calc_err:
                            print(
                                f"[PARSER-WARN] Index calculation lookback anomaly skipped: {calc_err}",
                                flush=True,
                            )

                except Exception as parse_err:
                    print(
                        f"[PARSER-ERROR] Failed to extract target token patterns out of logging stream: {parse_err}",
                        flush=True,
                    )
                    continue
        if self.config.DEBUG_FLAG:
            print(
                " [LOG-PARSER] Memory loop pipe interface closed down smoothly.",
                flush=True,
            )

    def video_writer_core_loop(self, stop_evt):
        """
        Thread-safe background min-heap consumer with adaptive sequence hole recovery.
        """
        print(
            " [WRITER-LOOP] Background tracking consumer loop active and polling memory queues...",
            flush=True,
        )
        try:
            while not stop_evt.is_set() or not self.write_queue.empty():
                try:
                    data = None
                    try:
                        data = self.write_queue.get(timeout=0.02)
                    except (queue.Empty, AttributeError):
                        continue

                    if data is None:
                        continue

                    control_data = data.get("control")
                    if control_data == "FLUSH":
                        print(
                            " [WRITER-LOOP] Intercepted downstream engine flush token code signature.",
                            flush=True,
                        )
                        if "pipe_handle" in data and data["pipe_handle"]:
                            try:
                                data["pipe_handle"].close()
                            except Exception:
                                pass
                        self.write_queue.task_done()
                        continue

                    slot_target = data.get("ring_slot_idx")
                    sock_handle = data.get("pipe_handle")
                    # frame_num = data.get("frame_num")

                    if (
                        sock_handle is None
                        and getattr(self, "video_writer", None) is not None
                    ):
                        sock_handle = self.video_writer

                    if slot_target is not None and sock_handle is not None:
                        if self.device_input == "cuda" and self.slot_events is not None:
                            self.slot_events[slot_target].synchronize()

                        if (
                            self.ffmpeg_proc is None
                            or self.ffmpeg_proc.poll() is not None
                        ):
                            print(
                                " [WRITER-WARN] Downstream execution loop pipe was broken out-of-band. Launching recovery routine...",
                                flush=True,
                            )
                            self._initialize_writer()
                            sock_handle = self.video_writer
                            if sock_handle is None:
                                self.write_queue.task_done()
                                continue

                        try:
                            raw_buffer_view = memoryview(
                                self.pinned_matrices[slot_target]
                            )
                            sock_handle.write(raw_buffer_view)
                            sock_handle.flush()

                        except (OSError, ValueError) as pipe_err:
                            print(
                                f" [PIPE-ERROR] Write operation dropped on ring index slot {slot_target}: {pipe_err}",
                                flush=True,
                            )
                            pass

                        self.write_queue.task_done()
                    else:
                        self.write_queue.task_done()

                except Exception as e:
                    print(
                        f"[WRITER-EXCEPTION] Worker engine cycle processing failure: {e}",
                        flush=True,
                    )
                    continue

            try:
                if hasattr(self, "socket_path") and os.path.exists(self.socket_path):
                    os.remove(self.socket_path)
            except Exception:
                pass

            self.writer_done = True
            print(
                " [WRITER-LOOP] Thread pool queue completely drained. Processing safe exit termination sequence...",
                flush=True,
            )

        except Exception as fatal_err:
            print(
                f"[FATAL-WRITER-CRASH] Unhandled background crash: {fatal_err}",
                flush=True,
            )
            traceback.print_exc()

    def _evaluate_barrier_and_dispatch(self, clip_key, clip_path, frame_w, frame_h):
        """
        Synchronized atomic thread barrier. Evaluates component convergence and
        dispatches the unified media asset payload directly to VDMS ingestion queues.
        """
        global clip_completion_tracker, all_metadata, send_metadata_queue

        if clip_key not in clip_completion_tracker:
            clip_completion_tracker[clip_key] = {
                "video": False,
                "meta": False,
                "start_time": time.time(),
            }

        tracker = clip_completion_tracker[clip_key]

        # Check convergence layout: Trigger upload sequence only if both pipelines sealed operations
        if tracker["video"] and tracker["meta"]:
            if self.config.DEBUG_FLAG:
                print(
                    f" [BARRIER-CONVERGENCE] Fully synchronized state reached for asset: {clip_key}",
                    flush=True,
                )

            # Extract and unmap metadata tracking payloads safely from shared RAM memory space
            clip_metadata = all_metadata.pop(clip_key, None)
            clip_completion_tracker.pop(clip_key, None)

            if not self.config.TEST_MODE and clip_metadata:
                # Re-insert the fully constructed framework into the active VDMS queue worker pool
                if (
                    hasattr(self, "send_metadata_queue")
                    and self.send_metadata_queue is not None
                ):
                    self.send_metadata_queue.put((clip_path, frame_w, frame_h))
                else:
                    global send_metadata_queue
                    send_metadata_queue.put((clip_path, frame_w, frame_h))

                print(
                    f" [BARRIER-INGEST] Unified data packages successfully submitted for DB processing: {clip_key}",
                    flush=True,
                )
            elif not self.config.TEST_MODE:
                print(
                    f" [BARRIER-WARN] Synchronization completed but all_metadata structure for {clip_key} was empty!",
                    flush=True,
                )
        elif not self.config.TEST_MODE:
            waiting_on = (
                "video segment closure"
                if not tracker["video"]
                else "AI frame processing execution"
            )
            print(
                f" [BARRIER-WAIT] {clip_key}: Milestone checked. Awaiting {waiting_on} before issuing DB ingestion call.",
                flush=True,
            )


class GPUStreamHandler(DeviceBaseHandler):
    def allocate_gpu(self):
        """
        Allocates persistent GpuMat buffers and CUDA streams to
        enable zero-copy GPU processing.
        """
        self.stream = cv2.cuda.Stream()
        self.ingest_stream = torch.cuda.Stream()
        self.inference_stream = torch.cuda.Stream()
        self.bgs_stream = cv2.cuda.Stream()
        self.gpu_fullres_frame = cv2.cuda.GpuMat(
            self.frame_height, self.frame_width, cv2.CV_8UC3
        )
        # self.resized_gpumat = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3)
        self.resized_frame = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3)
        self.resized_frame.setTo(0, self.bgs_stream)
        self.fgMask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        self.prev_bkgd = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        if self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD == "and":
            self.prev_bkgd.setTo((1,))
        else:
            self.prev_bkgd.setTo((0,))
        self.mask_history = deque(
            maxlen=self.config.BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE
        )
        self.mask_history.append(self.prev_bkgd)
        self.gpu_threshold_dst_frame = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )
        self.gpu_morphed_frame = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )

        self.upload_stream = cv2.cuda.Stream()
        self.upload_event = cv2.cuda.Event()

        # self.queue_capacity = int(2 * self.target_fps)  # 60
        self.num_buffers = int(2 * self.target_fps)  # self.queue_capacity + 5
        self.gpu_buffer_pool = [
            cv2.cuda.GpuMat(self.frame_height, self.frame_width, cv2.CV_8UC3)
            for _ in range(self.num_buffers)
        ]
        self.buffer_idx = 0

        self.frame_buffer_pool = [
            torch.empty(
                (3, self.frame_height, self.frame_width),
                dtype=torch.uint8,
                device="cuda",
            )
            for _ in range(2)
        ]
        self.pool_idx = 0

        # Create a matching pool of pinned host memory for the 8K frames
        self.host_buffer_pool = [
            cv2.cuda.HostMem(self.frame_height, self.frame_width, cv2.CV_8UC3)
            for _ in range(self.num_buffers)
        ]

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

        # Allocate a permanent float32/float16 channel layout space directly on VRAM
        self.static_gpu_360p = torch.empty(
            (1, 3, self.disp_h, self.disp_w),
            dtype=torch.float32,
            device="cuda",
        )
        self.static_gpu_byte_bchw = torch.empty(
            (1, 3, self.disp_h, self.disp_w),
            dtype=torch.uint8,
            device="cuda",
        ).contiguous()

        # Create two isolated tracking canvases to handle the ping-pong data stream
        self.static_host_canvases = [
            np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8),
            np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8),
        ]
        self.canvas_selector = 0

        # Register BOTH buffers as page-locked memory
        cv2.cuda.registerPageLocked(self.static_host_canvases[0])
        cv2.cuda.registerPageLocked(self.static_host_canvases[1])

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
        # history = int(2 * self.target_fps)  # 300  # int(5 * self.target_fps)
        self.lr = self.config.BKGD_SUB_MOG2_LR  # 1 / history
        self.backSub = cv2.cuda.createBackgroundSubtractorMOG2(
            history=self.config.BKGD_SUB_MOG2_HISTORY,  # Clear ghosts of fast drones in ~2 seconds (2*fps)
            varThreshold=int(
                1.15 * self.config.BKGD_SUB_MOG2_VARTHRESHOLD
            ),  # High threshold to ignore "shimmer" and compression noise  # default 16
            # CUDA implementation of MOG2 often requires a higher varThreshold to achieve the same "cleanliness" as the CPU (15-20%)
            detectShadows=self.config.BKGD_SUB_MOG2_DETECTSHADOWS,  # default True
        )

        self.dilate_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8U, self.dilate_kernel
        )
        self.dilate_filter_for_enhanced_mask = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8UC1, self.dilate_kernel_for_enhanced_mask
        )
        # # self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # self.morph_filter = cv2.cuda.createMorphologyFilter(
        #     cv2.MORPH_DILATE, cv2.CV_8UC1, self.morph_kernel
        # )
        # self.labels_gpu = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_32S)
        self.labels_gpu = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8U)
        self.labels_gpu.setTo(0, self.bgs_stream)

    def gpu_warmup(self):
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
                self.config.THRESHOLD_VALUE,
                self.config.THRESHOLD_MAX_VALUE,
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

        torch.cuda.empty_cache()

    def cleanup_gpu(self):
        """
        Explicitly releases all GPU-allocated memory to prevent
        VRAM leaks in 8K concurrent streams.
        """
        # Iterate through class attributes to explicitly release VRAM.
        for attr_name in list(self.__dict__.keys()):
            attr_value = getattr(self, attr_name)

            # Check if the attribute is a GpuMat
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

        if hasattr(self, "gpu_morphed_frame") and self.gpu_morphed_frame is not None:
            try:
                self.gpu_morphed_frame.release()
            except Exception:
                self.gpu_morphed_frame = None

        if hasattr(self, "labels_gpu") and self.labels_gpu is not None:
            try:
                self.labels_gpu.release()
            except Exception:
                self.labels_gpu = None

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

        if hasattr(self, "gpu_crop_batch"):
            for mat in self.gpu_crop_batch:
                if isinstance(mat, cv2.cuda.GpuMat):
                    mat.release()
            self.gpu_crop_batch = []

        if hasattr(self, "stream"):
            self.stream.waitForCompletion()

        self.pinned_downloaded_resizedframe_np = None
        self.gpu_threshold_dst_frame = None
        self.gpu_morphed_frame = None
        self.pinned_downloaded_frame_np = None

        # Handle specific buffers (like your Ping-Pong lists)
        # if hasattr(self, "encode_buffers"):
        #     self.encode_buffers.clear()

        # Clear the BGS history
        if hasattr(self, "mask_history"):
            self.mask_history.clear()

        # Optional: Final flush of the CUDA caching allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_background_subtraction_gpuv1(
        self, include_history=True, method="and", stream=None
    ):
        stream = stream if isinstance(stream, cv2.cuda.Stream) else self.stream
        self.fgMask = self.backSub.apply(
            self.resized_frame, float(self.lr), stream=stream
        )

        if include_history:
            # If this is the first run, clone the mask instead of ANDing with an empty/white buffer
            # if len(self.mask_history) < 1:
            #     self.prev_bkgd.setTo(255, stream)  # Clear the initial white buffer

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
                # bitor = cv2.cuda.bitwise_or(
                #     self.fgMask, self.prev_bkgd, stream=stream
                # )
                # bitand = cv2.cuda.bitwise_and(
                #     self.fgMask, self.prev_bkgd, stream=stream
                # )
                # not_bitand = cv2.cuda.bitwise_not(self.prev_bkgd, stream=stream)
                # self.fgMask = cv2.cuda.subtract(self.fgMask, bitand, stream=stream)
                self.fgMask = cv2.cuda.bitwise_or(
                    self.fgMask, self.prev_bkgd, stream=stream
                )
                # self.fgMask = cv2.cuda.bitwise_or(
                #     self.fgMask, self.mask_history[-2], stream=stream
                # )
                # if method == "or":
                #     self.fgMask = cv2.cuda.bitwise_and(
                #         self.fgMask, self.prev_bkgd, stream=stream
                #     )
                # else:
                #     self.fgMask = cv2.cuda.bitwise_or(
                #         self.fgMask, self.prev_bkgd, stream=stream
                #     )

    def apply_background_subtraction_gpu(
        self, motion_input, include_history=True, method="and", stream=None
    ):
        stream = stream if isinstance(stream, cv2.cuda.Stream) else self.stream
        raw_mask = self.backSub.apply(
            motion_input,
            0.005,  # float(self.lr),
            stream=stream,
        )

        if include_history:
            # If this is the first run, clone the mask instead of ANDing with an empty/white buffer
            # if len(self.mask_history) < 1:
            #     self.prev_bkgd.setTo(255, stream)  # Clear the initial white buffer
            self.prev_bkgd.setTo(0, stream=self.bgs_stream)
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

            self.mask_history.append(raw_mask.clone())
            raw_mask = cv2.cuda.bitwise_or(
                raw_mask, self.prev_bkgd, stream=self.bgs_stream
            )
            return raw_mask

            # min_val, max_val, _, _ = cv2.cuda.minMaxLoc(self.prev_bkgd)
            # if max_val != min_val and max_val > 0:
            #     # bitor = cv2.cuda.bitwise_or(
            #     #     self.fgMask, self.prev_bkgd, stream=stream
            #     # )
            #     # bitand = cv2.cuda.bitwise_and(
            #     #     self.fgMask, self.prev_bkgd, stream=stream
            #     # )
            #     # not_bitand = cv2.cuda.bitwise_not(self.prev_bkgd, stream=stream)
            #     # self.fgMask = cv2.cuda.subtract(self.fgMask, bitand, stream=stream)
            #     self.fgMask = cv2.cuda.bitwise_or(
            #         self.fgMask, self.prev_bkgd, stream=stream
            #     )
            #     # self.fgMask = cv2.cuda.bitwise_or(
            #     #     self.fgMask, self.mask_history[-2], stream=stream
            #     # )
            #     # if method == "or":
            #     #     self.fgMask = cv2.cuda.bitwise_and(
            #     #         self.fgMask, self.prev_bkgd, stream=stream
            #     #     )
            #     # else:
            #     #     self.fgMask = cv2.cuda.bitwise_or(
            #     #         self.fgMask, self.prev_bkgd, stream=stream
            #     #     )

    def rbtd_full_gpuv1(self, frame):
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
        # stream = self.stream

        with torch.no_grad():  # , torch.cuda.stream(self.bgs_stream):
            # frame is [3, 4320, 7680]
            resized_torch = F.interpolate(
                frame.unsqueeze(0).float(),
                size=(self.resize_h, self.resize_w),
                mode="nearest",
                # mode="bilinear",
                # align_corners=False,
            ).squeeze(0)
            # resized_torch = F.interpolate(
            #     frame.to(memory_format=torch.channels_last),
            #     size=(self.resize_h, self.resize_w),
            #     mode="nearest",
            #     # align_corners=False
            # )

            # CRITICAL: Change format from [3, 640, 640] to [640, 640, 3]
            # .contiguous() is mandatory here to reorganize the actual memory bits
            resized_torch = resized_torch.permute(1, 2, 0).byte().contiguous()
            gpu_mat_view = cv2.cuda.createGpuMatFromCudaMemory(
                self.resize_h, self.resize_w, cv2.CV_8UC3, resized_torch.data_ptr()
            )

            # Bridge the TINY 640p frame to OpenCV
            # Moving 8K (100MB) to CPU takes 114ms.
            # Moving 640p (1.2MB) to CPU takes <0.2ms.
            # This preserves your 15 FPS target.
            # small_cpu = resized_torch.permute(1, 2, 0).cpu().numpy()
            # self.resized_frame.upload(small_cpu)
            # gpu_mat_bridge = torch2gpumat(resized_torch.byte())
            gpu_mat_view.copyTo(self.bgs_stream, self.resized_frame)
            self.apply_background_subtraction_gpu(
                include_history=self.config.BKGD_SUB_INCLUDE_HISTORY,
                method=self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD,
                stream=self.bgs_stream,
            )

        # Clean up the motion mask using Thresholding and Morphology (Dilation)
        # Fused Cleanup (Threshold + Dilate)
        # This replaces: cv2.cuda.threshold AND cv2.cuda.dilate
        w, h = self.fgMask.size()
        pitch = self.fgMask.step
        tpb = (16, 16)
        bpg = ((w + 15) // 16, (h + 15) // 16)

        # Bridge GpuMat to CuPy for the kernel
        fg_ptr = self.fgMask.cudaPtr()
        with cupy.cuda.ExternalStream(self.bgs_stream.cudaPtr()):
            fg_cp = cupy.ndarray(
                (h, w),
                dtype=cupy.uint8,
                memptr=cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(fg_ptr, pitch * h, self), 0
                ),
                strides=(pitch, 1),
            )

            # Launch Fused Kernel
            labels_ptr = self.labels_gpu.cudaPtr()
            # Ensure this matches the size of fg_cp
            labels_cp = cupy.ndarray(
                (h, w),
                dtype=cupy.uint8,
                memptr=cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(labels_ptr, self.labels_gpu.step * h, self),
                    0,
                ),
                strides=(self.labels_gpu.step, 1),
            )
            # with cupy.cuda.ExternalStream(self.stream.cudaPtr()):
            # Launch your kernel on the same physical GPU stream as OpenCV BGS
            DETECTION_ACCEL_KERNEL(
                bpg,
                tpb,
                (fg_cp, labels_cp, pitch, w, h, self.config.THRESHOLD_VALUE),
            )
            # Ensure CuPy is done before the stream returns to OpenCV/PyTorch
            # cupy.cuda.get_current_stream().synchronize()

        # if self.debug_range[0] <= self.frame_count <= self.debug_range[1]:
        #     after_cpu = self.labels_gpu.download()
        #     # Multiply by 50 to make distinct object labels visible
        #     visible_labels = (after_cpu * 50).clip(0, 255).astype(np.uint8)
        #     cv2.imwrite(
        #         f"{self.out_imgdir}/mask_AFTER_frame_{self.frame_count}.png",
        #         visible_labels,
        #     )

        return {
            # "frameNum": frameNum,  # overall frame
            "mask": self.labels_gpu,  # GpuMat pointer to cleaned mask
            # "full_frame": frame,  # Kept for high-res cropping
            "full_frame": frame,  # current_gpu_frame,
        }

    def rbtd_full_gpu(self, device_frame):
        """
        Asynchronously handles downsampling, conversion, background subtraction, and
        morphology while writing out intermediate files for stage-by-stage debugging.
        """
        if self.config.DEBUG_FLAG:
            # Create a dedicated directory structure for this stage's debug snapshots
            stage_debug_dir = self.result_dir / "debug_stages" / self._testMethodName
            stage_debug_dir.mkdir(parents=True, exist_ok=True)
        f_num = self.frame_count_target

        # 1. BRIDGE THE PYTORCH TO OPENCV VRAM GAP (ZERO-COPY & STRIDE-ALIGNED)
        if torch.is_tensor(device_frame):
            h_raw, w_raw, ch = device_frame.shape
            cuda_mem_ptr = device_frame.data_ptr()
            cv_type = cv2.CV_8UC3 if ch == 3 else cv2.CV_8UC1

            # Explicitly pass structural step row pitch to avoid memory tearing artifacts
            row_step_bytes = device_frame.stride()[0] * device_frame.element_size()
            src_gpu_mat = cv2.cuda.createGpuMatFromCudaMemory(
                h_raw, w_raw, cv_type, cuda_mem_ptr, step=row_step_bytes
            )
        else:
            src_gpu_mat = device_frame
            ch = src_gpu_mat.channels()

        # [STAGE 1 DEBUG] Save incoming source frame immediately after GpuMat mapping
        if self.config.DEBUG_FLAG and f_num <= self.config.DEBUG_FRAME_LIMIT:
            src_cpu = src_gpu_mat.download()
            # If the source is in RGB, flip channels to BGR so cv2.imwrite saves true colors
            if ch == 3:
                src_cpu = cv2.cvtColor(src_cpu, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(stage_debug_dir / f"frame_{f_num:04d}_stage1_src.jpg"), src_cpu
            )

        # 2. INSTANTIATE GPUMAT CONTROLLERS WITH STRIDED LAYOUT HEADERS
        recycled_resize_mat = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC3 if ch == 3 else cv2.CV_8UC1
        )
        gray_resize_mat = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        raw_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        thresh_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)
        clean_mask = cv2.cuda.GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC1)

        # 3. RUN ASYNCHRONOUS DOWN-SAMPLING GATE
        cv2.cuda.resize(
            src_gpu_mat,
            dst=recycled_resize_mat,
            dsize=(self.resize_w, self.resize_h),
            interpolation=cv2.INTER_NEAREST,
            stream=self.bgs_stream,
        )

        # [STAGE 2 DEBUG] Check Downsampled Frame
        if self.config.DEBUG_FLAG and f_num <= self.config.DEBUG_FRAME_LIMIT:
            self.bgs_stream.waitForCompletion()  # Synchronize stream briefly to download safely
            resize_cpu = recycled_resize_mat.download()
            if ch == 3:
                resize_cpu = cv2.cvtColor(resize_cpu, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(stage_debug_dir / f"frame_{f_num:04d}_stage2_resize.jpg"),
                resize_cpu,
            )

        # 4. COLOR LAYOUT CORRECTION (COLLAPSE CHANNELS SAFELY)
        if recycled_resize_mat.channels() == 3:
            # NOTE: If your input tensor is RGB, use COLOR_RGB2GRAY.
            # If it's already converted to BGR inside your framework handlers, keep COLOR_BGR2GRAY.
            cv2.cuda.cvtColor(
                recycled_resize_mat,
                cv2.COLOR_BGR2GRAY,
                dst=gray_resize_mat,
                stream=self.bgs_stream,
            )
            motion_input = gray_resize_mat
        else:
            motion_input = recycled_resize_mat

        # [STAGE 3 DEBUG] Check Grayscale Input going into Background Subtractor
        if self.config.DEBUG_FLAG and f_num <= self.config.DEBUG_FRAME_LIMIT:
            self.bgs_stream.waitForCompletion()
            gray_cpu = motion_input.download()
            cv2.imwrite(
                str(stage_debug_dir / f"frame_{f_num:04d}_stage3_gray.jpg"), gray_cpu
            )

        # 5. EXECUTE BACKGROUND SUBTRACTION
        # raw_mask = self.backSub.apply(
        #     motion_input,
        #     0.005,  # float(self.lr)
        #     stream=self.bgs_stream,
        # )

        raw_mask = self.apply_background_subtraction_gpu(
            motion_input,
            include_history=self.config.BKGD_SUB_INCLUDE_HISTORY,
            method=self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD,
            stream=self.bgs_stream,
        )
        # [STAGE 4 DEBUG] Check Raw Output directly from Subtractor Kernel
        # if self.config.DEBUG_FLAG and f_num <= self.config.DEBUG_FRAME_LIMIT:
        #     self.bgs_stream.waitForCompletion()
        #     raw_mask_cpu = raw_mask.download()
        #     cv2.imwrite(str(stage_debug_dir / f"frame_{f_num:04d}_stage4_raw_mask.jpg"), raw_mask_cpu)

        # 6. MASK HISTORY ACCUMULATION LIFECYCLE
        # include_history = self.config.BKGD_SUB_INCLUDE_HISTORY
        # method = self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD
        # if include_history:
        #     self.prev_bkgd.setTo(0, stream=self.bgs_stream)
        #     for m in list(self.mask_history):
        #         dilated = self.dilate_filter_for_enhanced_mask.apply(
        #             m, stream=self.bgs_stream
        #         )
        #         if method == "or":
        #             cv2.cuda.bitwise_or(
        #                 self.prev_bkgd, dilated, self.prev_bkgd, stream=self.bgs_stream
        #             )
        #         else:
        #             cv2.cuda.bitwise_and(
        #                 self.prev_bkgd, dilated, self.prev_bkgd, stream=self.bgs_stream
        #             )

        #     self.mask_history.append(raw_mask.clone())
        #     raw_mask = cv2.cuda.bitwise_or(
        #         raw_mask, self.prev_bkgd, stream=self.bgs_stream
        #     )

        # [STAGE 5 DEBUG] Check Mask after History ORing/ANDing steps
        if self.config.DEBUG_FLAG and f_num <= self.config.DEBUG_FRAME_LIMIT:
            self.bgs_stream.waitForCompletion()
            hist_mask_cpu = raw_mask.download()
            cv2.imwrite(
                str(stage_debug_dir / f"frame_{f_num:04d}_stage5_history_mask.jpg"),
                hist_mask_cpu,
            )

        # 7. MORPHOLOGICAL TRANSFORMATIONS & BINARY FILTERS
        cv2.cuda.threshold(
            raw_mask,
            self.config.THRESHOLD_VALUE,
            self.config.THRESHOLD_MAX_VALUE,
            cv2.THRESH_BINARY,
            thresh_mask,
            stream=self.bgs_stream,
        )

        # [STAGE 6 DEBUG] Check Binary Threshold Output
        if self.config.DEBUG_FLAG and f_num <= self.config.DEBUG_FRAME_LIMIT:
            self.bgs_stream.waitForCompletion()
            thresh_cpu = thresh_mask.download()
            cv2.imwrite(
                str(stage_debug_dir / f"frame_{f_num:04d}_stage6_threshold.jpg"),
                thresh_cpu,
            )

        self.dilate_filter.apply(thresh_mask, clean_mask, self.bgs_stream)

        # 8. ENFORCE INDEPENDENT WORKSPACE MEMORY VIEWS
        isolated_kernel_mask = cv2.cuda.GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )
        clean_mask.copyTo(dst=isolated_kernel_mask, stream=self.bgs_stream)

        # [STAGE 7 DEBUG] Check Final Dilated Output Mask
        if self.config.DEBUG_FLAG and f_num <= self.config.DEBUG_FRAME_LIMIT:
            self.bgs_stream.waitForCompletion()
            final_mask_cpu = isolated_kernel_mask.download()
            cv2.imwrite(
                str(stage_debug_dir / f"frame_{f_num:04d}_stage7_final_mask.jpg"),
                final_mask_cpu,
            )

        return {
            "mask": isolated_kernel_mask,
            "full_frame": device_frame,
        }

    def findContours_gpu(self, mask, method="fused"):
        # self.stream.waitForCompletion()
        h, w = mask.size()
        ptr = mask.cudaPtr()
        pitch = mask.step
        mask_cp = cupy.ndarray(
            (h, w),
            dtype=cupy.uint8,
            memptr=cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(ptr, pitch * h, self), 0
            ),
            strides=(pitch, 1),
        )

        labeled, num_labels = cupyx.scipy.ndimage.label(mask_cp, output=cupy.int32)
        # labeled = labeled.astype(cupy.int32)
        # labeled.strides is in bytes. For int32, we need row-start in bytes.
        labeled_pitch_bytes = labeled.strides[0]

        if num_labels == 0:
            return torch.empty((0, 4), device="cuda")

        # Pre-allocate bounds with sentinels
        x1, y1 = (
            cupy.full((num_labels + 1,), w, dtype=cupy.int32),
            cupy.full((num_labels + 1,), h, dtype=cupy.int32),
        )
        x2, y2 = (
            cupy.full((num_labels + 1,), -1, dtype=cupy.int32),
            cupy.full((num_labels + 1,), -1, dtype=cupy.int32),
        )

        # BOUNDS_KERNEL(((w+15)//16, (h+15)//16), (16, 16), (labeled, w, h, num_labels, x1, y1, x2, y2))
        BOUNDS_KERNEL(
            ((w + 15) // 16, (h + 15) // 16),
            (16, 16),
            (
                labeled.data.ptr,
                labeled_pitch_bytes,
                w,
                h,
                num_labels,
                x1.data.ptr,
                y1.data.ptr,
                x2.data.ptr,
                y2.data.ptr,
            ),
        )
        # cupy.cuda.get_current_stream().synchronize()
        # cupy.cuda.Stream.null.synchronize()
        # Convert to torch
        boxes = torch.stack(
            [
                torch.as_tensor(x1[1:], device="cuda"),
                torch.as_tensor(y1[1:], device="cuda"),
                torch.as_tensor(x2[1:], device="cuda"),
                torch.as_tensor(y2[1:], device="cuda"),
            ],
            dim=1,
        ).float()

        # --- CRITICAL FIX: Filter out boxes that weren't updated by the kernel ---
        # A valid box must have x2 >= x1
        valid_mask = boxes[:, 2] >= boxes[:, 0]
        boxes = boxes[valid_mask]

        # if boxes.shape[0] > 0:
        #     print(f"[GPU] Found {boxes.shape[0]} boxes", flush=True)
        return boxes


class CPUStreamHandler(DeviceBaseHandler):
    def allocate_cpu(self):
        # pass
        self.resized_frame = np.zeros((3, self.resize_h, self.resize_w), dtype="uint8")
        # cv2.cuda.createContinuous(
        #     self.resize_h, self.resize_w, cv2.CV_8UC3
        # )

        self.fgMask = np.zeros(
            (self.resize_h, self.resize_w), dtype="uint8"
        )  # For resize

        if self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD == "and":
            self.prev_bkgd = np.ones(
                (self.resize_h, self.resize_w), dtype="uint8"
            )  # * 255
        else:
            self.prev_bkgd = np.zeros((self.resize_h, self.resize_w), dtype="uint8")

        # self.prev_bkgd = np.ones((self.resize_h, self.resize_w), dtype="uint8") * 255

        self.mask_history = deque(
            maxlen=self.config.BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE
        )
        self.mask_history.append(self.prev_bkgd)

    def prepare_cpu_pipeline(self):  # , method="mog2"):
        self.operation_device_map = PipelineMapping()  # "full_cpu"
        self.device_input = self.operation_device_map.detection_device

        self.allocate_cpu()

        # Subtraction
        # if method == "knn":
        #     history = 300  # int(5 * self.target_fps)
        #     background_thresh = 350
        #     NSamples = 10
        #     kNNSamples = 2
        #     self.lr = 1 / history

        #     self.backSub = cv2.createBackgroundSubtractorKNN(
        #         history=history,  # default 500
        #         dist2Threshold=background_thresh,  # default 400
        #         detectShadows=False,  # default True
        #     )
        #     self.backSub.setkNNSamples(kNNSamples)
        #     self.backSub.setNSamples(NSamples)
        # elif method == "mog2":
        # history = int(2 * self.target_fps)
        # background_thresh = 10
        self.lr = self.config.BKGD_SUB_MOG2_LR

        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=self.config.BKGD_SUB_MOG2_HISTORY,  # Clear ghosts of fast drones in ~2 seconds (2*fps)
            varThreshold=self.config.BKGD_SUB_MOG2_VARTHRESHOLD,  # High threshold to ignore "shimmer" and compression noise  # default 16
            detectShadows=self.config.BKGD_SUB_MOG2_DETECTSHADOWS,  # default True
        )
        # else:
        #     raise ValueError(f"Provided method ({method}) is not available.")

    def cleanup_cpu(self):
        """
        Purges large 8K NumPy buffers and CPU-based AI resources.
        """
        # Nullify specific class references to allow Garbage Collection
        self.executor = None
        self.clip_executor = None
        self.reader = None
        self.latest_processed_frame = None

        # Clear the Ping-Pong buffers (up to 200MB of RAM)
        # if hasattr(self, "encode_buffers"):
        #     self.encode_buffers.clear()

        # Explicitly nullify large arrays to trigger Garbage Collection
        self.resized_frame = None
        self.fgMask = None
        self.prev_bkgd = None

        # Clear the BGS history
        if hasattr(self, "mask_history"):
            self.mask_history.clear()

    def apply_background_subtraction_cpuv1(self, include_history=True, method="and"):
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

    def apply_background_subtraction_cpu(
        self, motion_input, include_history=True, method="and"
    ):
        raw_mask = self.backSub.apply(
            motion_input, learningRate=self.config.BKGD_SUB_MOG2_LR
        )

        if include_history:
            # If this is the first run, clone the mask instead of ANDing with an empty/white buffer
            # if len(self.mask_history) < 1:
            #     self.prev_bkgd.setTo(0, stream)  # Clear the initial white buffer
            self.prev_bkgd = np.zeros((self.resize_h, self.resize_w), dtype="uint8")
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

            self.mask_history.append(raw_mask.copy())

            # if (
            #     self.prev_bkgd.max() != self.prev_bkgd.min()
            #     and self.prev_bkgd.max() > 0
            # ):
            #     combined_mask_bool = (self.fgMask > 0) | (self.prev_bkgd > 0)
            #     self.fgMask = combined_mask_bool.astype(np.uint8) * 255
            raw_mask = cv2.bitwise_or(raw_mask, self.prev_bkgd)
            return raw_mask

    def rbtd_full_cpuv1(self, frame):
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
        # if self.config.ENABLE_QUERYING and self.video_writer:
        #     self.write_queue.put(self.cpu_resized_frame.copy())

        # Apply Background Subtraction on CPU
        self.apply_background_subtraction_cpu(
            include_history=self.config.BKGD_SUB_INCLUDE_HISTORY,
            method=self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD,
        )

        # ----------------
        # NOISE FILTERING: Median Blur
        # Kills small single-pixel noise specs
        # self.fgMask = cv2.medianBlur(self.fgMask, 5)

        # # MORPHOLOGY: Opening (Erode then Dilate)
        # kernel = np.ones((3,3), np.uint8)

        # # Remove small specs
        # self.fgMask = cv2.erode(self.fgMask, kernel, iterations=2)

        # # Re-expand and connect nearby moving pixels
        # self.fgMask = cv2.dilate(self.fgMask, kernel, iterations=2)
        # return {"mask": self.fgMask, "full_frame": frame}
        # ----------------

        # Clean up the motion mask using Thresholding and Morphology (Dilation)
        _, mask = cv2.threshold(
            self.fgMask,
            self.config.THRESHOLD_VALUE,
            self.config.THRESHOLD_MAX_VALUE,
            cv2.THRESH_BINARY,
        )
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)

        return {
            # "frameNum": frameNum,  # overall frame
            "mask": mask,
            "full_frame": frame,  # Kept for high-res cropping
        }

    # def rbtd_full_cpu(self, frame):
    #     # --- PHASE 1: NATIVE CACHE-PINNED RESIZE ---
    #     # Fetch the exact background resolution constraints natively required by your config
    #     # We assume self.resize_w and self.resize_h reflect your locked tracking target dimensions
    #     target_w, target_h = self.resize_w, self.resize_h

    #     # Safe Multi-Threaded Canvas Initialization Guard:
    #     # Dynamically instantiate the static persistent memory matrices on the first frame pass.
    #     # This completely stops the Python heap engine from allocating new arrays on subsequent frames.
    #     if not hasattr(self, "_pinned_small_frame") or self._pinned_small_frame.shape[
    #         :2
    #     ] != (target_h, target_w):
    #         self._pinned_small_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    #         self._pinned_fg_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    #         self._pinned_dilated_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    #         self._pinned_global_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    #     # Force a zero-allocation resize into our pre-allocated, sequential cache-line matrix buffer
    #     cv2.resize(
    #         frame,
    #         (target_w, target_h),
    #         dst=self._pinned_small_frame,
    #         interpolation=cv2.INTER_NEAREST,
    #     )

    #     # --- PHASE 2: STRIPPED SINGLE-THREAD BACKGROUND ARITHMETIC ---
    #     # Run background subtraction straight into our static memory address lane
    #     # We pass your configured learning rate (BKGD_SUB_MOG2_LR) to lock the temporal parameters
    #     self._pinned_fg_mask = self.backSub.apply(
    #         self._pinned_small_frame,
    #         # dst=self._pinned_fg_mask,
    #         learningRate=self.config.BKGD_SUB_MOG2_LR,
    #     )

    #     # Execute morphology dilation inline within our zero-allocation ring workspace [PDF: 0.1.18]
    #     cv2.dilate(
    #         self._pinned_fg_mask,
    #         self.dilate_kernel,
    #         dst=self._pinned_dilated_mask,
    #         iterations=1,
    #     )

    #     return {"full_frame": frame, "mask": self._pinned_dilated_mask}

    def rbtd_full_cpu(self, frame):
        # --- PHASE 1: NATIVE CACHE-PINNED RESIZE ---
        # Fetch the exact background resolution constraints natively required by your config
        # We assume self.resize_w and self.resize_h reflect your locked tracking target dimensions
        target_w, target_h = self.resize_w, self.resize_h

        # Safe Multi-Threaded Canvas Initialization Guard:
        # Dynamically instantiate the static persistent memory matrices on the first frame pass.
        # This completely stops the Python heap engine from allocating new arrays on subsequent frames.
        if not hasattr(self, "_pinned_small_frame") or self._pinned_small_frame.shape[
            :2
        ] != (target_h, target_w):
            self._pinned_small_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            self._pinned_fg_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            self._pinned_dilated_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            self._pinned_global_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Force a zero-allocation resize into our pre-allocated, sequential cache-line matrix buffer
        cv2.resize(
            frame,
            (target_w, target_h),
            dst=self._pinned_small_frame,
            interpolation=cv2.INTER_NEAREST,
        )

        # --- PHASE 2: STRIPPED SINGLE-THREAD BACKGROUND ARITHMETIC ---
        # Run background subtraction straight into our static memory address lane
        # We pass your configured learning rate (BKGD_SUB_MOG2_LR) to lock the temporal parameters
        # self._pinned_fg_mask = self.backSub.apply(
        #     self._pinned_small_frame,
        #     # dst=self._pinned_fg_mask,
        #     learningRate=self.config.BKGD_SUB_MOG2_LR,
        # )

        self._pinned_fg_mask = self.apply_background_subtraction_cpu(
            self._pinned_small_frame,
            include_history=self.config.BKGD_SUB_INCLUDE_HISTORY,
            method=self.config.BKGD_SUB_INCLUDE_HISTORY_METHOD,
        )

        # Execute morphology dilation inline within our zero-allocation ring workspace [PDF: 0.1.18]
        cv2.dilate(
            self._pinned_fg_mask,
            self.dilate_kernel,
            dst=self._pinned_dilated_mask,
            iterations=1,
        )

        return {"full_frame": frame, "mask": self._pinned_dilated_mask}
