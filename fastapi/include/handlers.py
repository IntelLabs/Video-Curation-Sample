import asyncio
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
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
main_app_logger = logging.getLogger()


# ----- PIPELINE CONFIGURATION -----
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Force OpenCV to use a single thread for its operations.
# This prevents internal OpenCV threads from "racing" against AI logic.
# cv2.setNumThreads(1)

# Force OpenCV to run sequentially to prevent context-switching overhead
cv2.setNumThreads(0)

from include.utils import (
    PipelineConfig,
    PipelineMapping,
    draw_label,
    get_detection_color,
    get_display_frame_in_bytes,
    metadata2vdms_with_retry,
    tensor2opencv,
)

BASE_PIPELINE_CONFIG = PipelineConfig(
    SHARED_MODEL=os.getenv("SHARED_MODEL", False),
    ENABLE_QUERYING=os.getenv("ENABLE_QUERYING", False),
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
# ENABLE_QUERYING = os.getenv("ENABLE_QUERYING", False)
if BASE_PIPELINE_CONFIG.ENABLE_QUERYING:
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
                video_backlog = (
                    streamer.write_queue.qsize()
                    if streamer.config.ENABLE_QUERYING
                    else 0
                )
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
                        if BASE_PIPELINE_CONFIG.DEBUG == "1":
                            print(f"CLEANUP: Removing {name} from active_streams")
                        streamer.stop()
                        app.state.active_streams.pop(name, None)

        # --- Synchronization Data Purge ---
        if BASE_PIPELINE_CONFIG.ENABLE_QUERYING:
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
        torch.cuda.empty_cache()
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
# from include.utils import nv12_to_rgb_torch
# def nv12_to_rgb_torch(
#     nv12_tensor, h, w, is_h264_8k=False, out_buffer=None, is_bgr=True
# ):
#     """
#     Highly optimized NV12 to RGB conversion for 8K.
#     - Uses FP16 for math to save VRAM.
#     - Performs in-place operations.
#     - Supports pre-allocated output buffers.
#     """
#     if is_h264_8k:
#         if is_bgr:
#             nv12_tensor = nv12_tensor[[2, 1, 0], :, :]
#         # return nv12_tensor
#         if out_buffer is not None:
#             # COPY directly into the pool buffer (Zero Allocation)
#             out_buffer.copy_(nv12_tensor.contiguous())
#             return out_buffer

#         return nv12_tensor.contiguous()

#     with torch.no_grad():
#         # Extract Y and UV planes (Zero-copy views)
#         y = nv12_tensor[:h, :w].unsqueeze(0).half()
#         uv = (
#             nv12_tensor[h:, :w]
#             .reshape(h // 2, w // 2, 2)
#             .permute(2, 0, 1)
#             .unsqueeze(0)
#             .half()
#         )

#         # Upsample UV to match Y (Chroma upsampling 4:2:0 -> 4:4:4)
#         # We use nearest neighbor because it's fastest for 8K
#         uv_up = F.interpolate(uv, size=(h, w), mode="nearest") #, align_corners=False)

#         y = (y - 16.0) * 1.164
#         u = uv_up[:, 0, :, :] - 128
#         v = uv_up[:, 1, :, :] - 128
#         # y = y - 16

#         # Fused YUV to RGB Conversion (BT.709 Coefficients)
#         # Using in-place addition/multiplication to avoid creating temporary 8K tensors
#         # r = y + 1.5748 * v
#         # g = y - 0.1873 * u - 0.4681 * v
#         # b = y + 1.8556 * u
#         r = y + 1.793 * v
#         g = y - 0.213 * u - 0.533 * v
#         b = y + 2.112 * u

#         # Stack into final shape [3, H, W]
#         # If out_buffer is provided (from our Tensor Pool), we write directly into it
#         if not is_bgr:
#             colored_img = torch.cat([r, g, b], dim=0)
#         else:
#             colored_img = torch.cat([b, g, r], dim=0)

#         # Final Clamp and Byte conversion
#         # We use .clamp_() with an underscore for in-place modification
#         colored_img.clamp_(0, 255)

#         if out_buffer is not None:
#             # COPY directly into the pool buffer (Zero Allocation)
#             out_buffer.copy_(colored_img.to(torch.uint8))
#             return out_buffer

#         return colored_img.to(torch.uint8)


def nv12_to_rgb_torch(
    nv12_tensor, h, w, is_h264_8k=False, out_buffer=None, is_bgr=True
):
    """
    Highly optimized NV12/YUV to RGB/BGR conversion.
    Fixes the 'Blue Frame' by applying proper YUV-to-RGB matrix math.
    """
    with torch.no_grad():
        if is_h264_8k:
            # 8K Path: Expects planar data [C, H, W]
            # y: [1, H, W], uv: [1, H, W] (already resized or needs upsampling)
            # y = nv12_tensor[0:1, :, :].half()
            # # If your 8K source has separate U and V, adjust indices [1:2] and [2:3]
            # u = nv12_tensor[1:2, :, :].half()
            # v = nv12_tensor[2:3, :, :].half() # Fallback if interleaved
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
    if BASE_PIPELINE_CONFIG.DEBUG_FLAG:
        print(
            f"[TIMING],start_release_clip,{clip_key},{time.time()}",
            flush=True,
        )

    if _out_vid is not None:
        _out_vid.release()

    if BASE_PIPELINE_CONFIG.DEBUG_FLAG:
        print(
            f"[TIMING],end_release_clip,{clip_key},{time.time()}",
            flush=True,
        )

    # Re-encode video in order to seek via ffmpeg later
    GENERAL_OPTS = "-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -flush_packets 0"  #  -filter:v fps={target_fps}
    CONVERSION = f"-c:v libx264 -preset ultrafast -filter:v fps=fps={target_fps}"  # "-c:v libx264 -preset medium"
    reencode_cmd = f"nice -n 19 ffmpeg -y -i {tmp_file} {GENERAL_OPTS} {CONVERSION} -crf 23 -c:a copy {clip_filename}"

    # Re-encode command with background-friendly flags
    # nice -n 19: Lowers CPU priority to avoid stutters in the main app
    # -threads 2: Limits core usage
    # -preset ultrafast: Fastest possible h264 encoding
    # -crf 28: Slightly higher than default (23) for even less CPU work
    # reencode_cmd = (
    #     f"nice -n 19 ffmpeg -y -i {tmp_file} "
    #     f"-threads 2 -c:v libx264 -preset ultrafast -crf 28 "
    #     f"-filter:v fps=fps={target_fps} -c:a copy -tune zerolatency "
    #     f"-hide_banner -loglevel error {clip_filename}"
    # )

    try:
        cmd_list = shlex.split(reencode_cmd)

        if BASE_PIPELINE_CONFIG.DEBUG_FLAG:
            print(
                f"[TIMING],start_reencode,{clip_key},{time.time()}",
                flush=True,
            )

        subprocess.run(cmd_list, check=True)
        end_time = time.time()

        if BASE_PIPELINE_CONFIG.DEBUG_FLAG:
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


def send_metadata(
    VDMS_POOL=None,
    DEBUG_FLAG=BASE_PIPELINE_CONFIG.DEBUG_FLAG,
    INGESTION=BASE_PIPELINE_CONFIG.INGESTION,
    TEST_MODE=BASE_PIPELINE_CONFIG.TEST_MODE,
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
                    VDMS_POOL=VDMS_POOL,
                    DEBUG_FLAG=DEBUG_FLAG,
                    INGESTION=INGESTION,
                    TEST_MODE=TEST_MODE,
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


# ----- STREAM HANDLERS -----
class TensorPool:
    def __init__(self, h, w, device="cuda"):
        # We only need 2 slots: one for the current frame being processed,
        # and one for the next frame being decoded.
        self.frames = [
            torch.empty((3, h, w), dtype=torch.half, device=device) for _ in range(2)
        ]
        self.current_idx = 0

    def get_buffer(self):
        buf = self.frames[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.frames)
        return buf


# def merge_nearby_boxes_640_gpu(raw_boxes_640p, gap_limit=10):
#     # Vectorized Merging (GPU Equivalent to merge_nearby_boxes_640)
#     # Instead of a loop, we use a distance matrix to find clusters
#     if raw_boxes_640p.shape[0] > 1:
#         # Calculate centroids of the 640p boxes
#         centers = (raw_boxes_640p[:, :2] + raw_boxes_640p[:, 2:]) / 2

#         # Compute all-to-all distance matrix [N, N]
#         dist_matrix = torch.cdist(centers, centers)

#         # Adjacency: True if boxes are within the threshold
#         adj = dist_matrix < gap_limit

#         # simple connected components grouping (Iterative bitwise OR)
#         # This groups the swarm clusters in VRAM
#         n = raw_boxes_640p.shape[0]
#         components = torch.arange(n, device=raw_boxes_640p.device)
#         for _ in range(5):
#             components = torch.max(adj * components, dim=1)[0]

#         # Extract merged bounding boxes for each cluster
#         unique_comps = components.unique()
#         merged_list = []
#         for comp_id in unique_comps:
#             cluster = raw_boxes_640p[components == comp_id]
#             new_box = torch.cat(
#                 [cluster[:, :2].min(dim=0)[0], cluster[:, 2:].max(dim=0)[0]]
#             ).unsqueeze(0)
#             merged_list.append(new_box)

#         raw_boxes_640p = torch.cat(merged_list, dim=0)
#     return raw_boxes_640p


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


def rendering_worker(
    queue,
    shared_details,
    ready_idx,
    reader_active_idx,
    frame_lengths,
    signal_queue,
    display_size,
    quality,
):
    disp_w, disp_h = display_size
    # Attach to both buffers
    shm_names = shared_details["shm_names"]
    worker_shms = [mp.shared_memory.SharedMemory(name=n) for n in shm_names]
    num_shms = len(shm_names)

    # Get shm
    # shm_name = shared_details.get("shm_name")
    # try:
    #     shm = mp.shared_memory.SharedMemory(name=shm_name)
    # except Exception as e:
    #     print(f"[WORKER] SHM attach failed: {e}", flush=True)
    #     return

    try:
        while True:
            item = queue.get()
            if item is None:  # Sentinel value to stop the worker
                break

            # frame is display size
            # metadata in resized res
            display_frame, frameNum, metadata_or_bbs, class_list = item

            # display_size = (self.resize_h, self.resize_w)
            # display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)

            scale_display_x = disp_w / 640
            scale_display_y = disp_h / 640
            if isinstance(metadata_or_bbs, dict):
                # Case: Object Detection
                for _, obj in metadata_or_bbs.items():
                    bbox = obj["bbox"]
                    x, y, w, h = (
                        bbox["x"] * scale_display_x,
                        bbox["y"] * scale_display_y,
                        bbox["width"] * scale_display_x,
                        bbox["height"] * scale_display_y,
                    )
                    class_name = bbox["object"]
                    class_id = class_list.index(class_name)
                    confidence = bbox["object_det"]["confidence"]

                    bb_color = get_detection_color(class_id, is_bgr=True)
                    label = f"{class_name} {confidence:.2f}"

                    x = max(0, int(x))
                    y = max(0, int(y))
                    w = min(disp_w, int(w))
                    h = min(disp_h, int(h))
                    # Draw on a copy of the frame to avoid modifying the original
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), bb_color, 2)
                    # Replace draw_label if it's accessible; otherwise use cv2.putText
                    # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    draw_label(display_frame, label, (x, y), color=bb_color, padding=5)
            elif metadata_or_bbs is not None:
                # Case: Motion Detections Only (SF Path)
                for box in metadata_or_bbs:
                    # if torch.is_tensor(box):
                    #     box = box.tolist()

                    if torch.is_tensor(box):
                        x1, y1, x2, y2 = box.to(torch.int).cpu().tolist()
                    else:
                        x1, y1, x2, y2 = map(int, box)

                    x1, y1, x2, y2 = (
                        x1 * scale_display_x,
                        y1 * scale_display_y,
                        x2 * scale_display_x,
                        y2 * scale_display_y,
                    )
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(disp_w, int(x2))
                    y2 = min(disp_h, int(y2))
                    # Draw on a copy of the frame to avoid modifying the original
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 225), 2)

            # writer.write(display_frame)
            if frameNum > shared_details["last_id"]:  # self.last_delivered_frame_id:
                frame_bytes = get_display_frame_in_bytes(
                    display_frame,
                    display_size=display_size,
                    quality=quality,
                    return_bytes=True,
                )
                if frame_bytes:
                    # THE HARD GUARD: If the reader is currently touching RAM, skip this write.
                    # This prevents the '1-minute' scramble by ensuring zero memory overlap.
                    # if signal_queue.full():
                    #     continue
                    frame_len = len(frame_bytes)

                    forbidden_idx = [ready_idx.value, reader_active_idx.value]
                    available_idx = [
                        i for i in range(num_shms) if i not in forbidden_idx
                    ]

                    if not available_idx:
                        continue

                    # Write to the buffer that is NOT currently 'ready'
                    # write_idx = (shared_details["buffer_idx"] + 1) % 2
                    # write_idx = 1 if ready_idx.value == 0 else 0
                    # current_ready = ready_idx.value
                    # write_idx = (current_ready + 1) % 3
                    write_idx = available_idx[0]
                    shm = worker_shms[write_idx]

                    # Zero-copy write to RAM
                    shm.buf[:frame_len] = frame_bytes

                    frame_lengths[write_idx] = frame_len
                    # shared_details["buffer_idx"] = write_idx
                    ready_idx.value = write_idx
                    shared_details["last_id"] = frameNum
                    # self.last_frame_id = frameNum
                    # self.last_heartbeat = time.time()
                    # Signal the FastAPI generator that a new frame is ready
                    # self.loop.call_soon_threadsafe(self.frame_ready_event.set)
                    # self.mp_frame_ready_event.set()
                    # try:
                    #     signal_queue.put_nowait(True)
                    # except Exception:
                    #     pass
                    signal_queue.put(True)

        # END While

    except Exception as e:
        print(f"Error while rendering display: {e}")
    finally:
        for s in worker_shms:
            s.close()


class DeviceBaseHandler:
    def __init__(
        self, source, name, active_streams, config=BASE_PIPELINE_CONFIG, **kwargs
    ):
        self.name = name
        self.source = source
        self.active = True
        self.active_streams = active_streams
        self.config = config

        self.loop = asyncio.get_event_loop()
        self.frame_ready_event = asyncio.Event()
        self._is_stopped = False  # 🛡️ Shutdown guard
        self._stop_lock = threading.Lock()  # 🔒 Local lock for this instance
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
        self.stat_start_time = time.perf_counter()
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
        # Determine minimum contour size relative to frame resolution
        self.min_contour_area = int(
            (self.config.ROI_MIN_AREA_RATIO * self.resize_w)
            * (self.config.ROI_MIN_AREA_RATIO * self.resize_h)
        )  # 207

        self.dist_thresh_8k = max(
            self.config.ROI_DISTANCE_THRESH_RATIO * self.frame_width,
            self.config.ROI_DISTANCE_THRESH_RATIO * self.frame_height,
        )
        self.dist_thresh_640 = max(
            self.config.ROI_DISTANCE_THRESH_RATIO * self.resize_w,
            self.config.ROI_DISTANCE_THRESH_RATIO * self.resize_h,
        )  # 0.05 * self.resize_w
        self.scales_tensor = torch.tensor(
            [self.scale_x, self.scale_y, self.scale_x, self.scale_y],
            # device="cpu",
            device=self.device_input,
        )

        self.disp_w, self.disp_h = self.config.DISPLAY_FRAME_SIZE

        # Performance Tracking
        self.total_objects_detected = 0
        self.frame_count = 0  # Frame count for videos
        self.frame_count_target = 0
        self.next_process_idx = 0.0
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

        if self.config.ENABLE_QUERYING:
            # Thread-safe queue for the resized frames (640x640)
            # maxlen=300 allows for a 20-second buffer in case of extreme disk lag
            # Non-blocking queue for frames and control signals
            self.write_queue = queue.Queue(maxsize=300)
            self.writer_done = False

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
        if hasattr(self, "reader"):
            del self.reader

        # Add a tiny sleep or garbage collect to ensure the GPU handle is released
        gc.collect()
        torch.cuda.empty_cache()  # Clear any remaining context

        if self.device_input == "cuda":
            from include.readers import GPUHybridReader

            self.gpu_id = 0
            self.reader = GPUHybridReader(
                source=self.source,
                target_fps=target_fps,
                clip_duration=clip_duration,
                gpu_id=self.gpu_id,
                only_target_frames=True,
            )
        else:
            from include.readers import CPUHybridReader

            self.reader = CPUHybridReader(
                source=self.source,
                target_fps=target_fps,
                clip_duration=clip_duration,
                only_target_frames=True,
            )

        self.input_fps = self.reader.input_fps
        self.target_fps = self.reader.target_fps
        self.step_size = self.input_fps / self.target_fps
        self.frame_skip = self.reader.frame_skip
        self.max_frames_per_clip = self.reader.max_frames_per_clip
        self.frame_interval = self.reader.frame_interval
        self.frame_width = self.reader.frame_width
        self.frame_height = self.reader.frame_height
        self.numFrames = self.reader.numFrames
        self.duration_s = self.numFrames / self.input_fps
        self.expected_num_frames = int(self.duration_s * self.target_fps)
        self.get_frameWH()

    def prepare_pipeline(self):
        if self.device_input == "cuda":
            self.prepare_gpu_pipeline()
            if len(self.active_streams) == 0:
                self.gpu_warmup()
        else:
            self.prepare_cpu_pipeline()

    def setup_threads(self):
        # Shared 10MB memory for display
        self.setup_shared_memory()

        # self.pool = TensorPool(self.frame_height, self.frame_width)

        # Executor for Async YOLO tasks and FFmpeg re-encoding
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        #  Select the correct pipeline function
        if self.device.lower() == "gpu":
            if self.config.sf_enabled:
                pipeline_fn = self.sf_gpu_pipeline_fn
            else:
                pipeline_fn = self.yolo_gpu_pipeline_fn

            # pipeline_fn = self.pipeline_fn_gpu if sf_enabled else self.yolo_gpu_pipeline_fn
        else:
            if self.config.sf_enabled:
                pipeline_fn = self.sf_cpu_pipeline_fn
            else:
                pipeline_fn = self.yolo_cpu_pipeline_fn

        print(
            f"pipeline_fn: {pipeline_fn.__class__}\tTEST_MODE: {self.config.TEST_MODE}",
            flush=True,
        )
        # Producer: Handles acquisition and AI metadata logs
        self.process_thread = threading.Thread(
            target=self.run_realtime_inference,
            args=(self.config.sf_enabled, pipeline_fn),
            daemon=True,
        )

        self.signal_queue = mp.Queue(maxsize=1)
        self.render_queue = mp.Queue(maxsize=5)
        self.render_proc = mp.Process(
            target=rendering_worker,
            args=(
                self.render_queue,
                self.shared_details,
                self.ready_buffer_idx,
                self.reader_active_idx,
                self.shm_frame_lengths,
                self.signal_queue,
                (self.disp_w, self.disp_h),
                self.config.DISPLAY_FRAME_QUALITY,
            ),
        )

        def display_signal_sync():
            while self.active:
                # Wait for signal
                # if self.mp_frame_ready_event.wait(timeout=1.0):
                #     self.mp_frame_ready_event.clear()
                try:
                    _ = self.signal_queue.get(timeout=1.0)
                    # print(f"[DEBUG]: Signal received in FastAPI process for {self.name}", flush=True)
                    # Wake FastAPI async loop in main thread
                    self.loop.call_soon_threadsafe(self.frame_ready_event.set)
                except queue.Empty:
                    continue

        self.display_proc = threading.Thread(target=display_signal_sync, daemon=True)

        if self.config.ENABLE_QUERYING:
            # NEW: Dedicated I/O pool for Disk/GPU transfers (Higher worker count for 8K)
            self.io_executor = ThreadPoolExecutor(max_workers=8)

            # Dedicated FFmpeg pool so re-encoding doesn't slow down live AI
            self.ffmpeg_executor = ThreadPoolExecutor(max_workers=2)

            # Sends metadata to VDMS
            self.metadata_thread = threading.Thread(
                target=send_metadata,
                args=(
                    VDMSPool(self.config.DBHOST, self.config.DBPORT, size=10),
                    self.config.DEBUG_FLAG,
                    self.config.INGESTION,
                    self.config.TEST_MODE,
                ),
                daemon=True,
            )

            # Consumer: Handles GPU-to-CPU download and Disk I/O (Writing resized frames to RAM disk)
            self.writer_thread = threading.Thread(
                target=self._video_writer, daemon=True
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

        for idx in range(num_shms):
            # self.shm = mp.shared_memory.SharedMemory(create=True, size=10*1024*1024)
            shm_name = f"shm_{self.name}_{idx}_{os.getpid()}"
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

        self.render_proc.start()

        self.display_proc.start()

        if self.config.ENABLE_QUERYING:
            self._initialize_writer()

        # Small delay to allow the reader's deque to populate
        time.sleep(0.1)

        # Start the producer and consumer threads
        if not self.process_thread.is_alive():
            self.process_thread.start()

        if self.config.ENABLE_QUERYING and not self.metadata_thread.is_alive():
            self.metadata_thread.start()

        if self.config.ENABLE_QUERYING and not self.writer_thread.is_alive():
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

        # This ensures the next /dashboard_stats call won't see this stream
        if self.name in self.active_streams:
            self.active_streams.pop(self.name, None)

        # Signal threads to stop
        self.active = False
        print(f" [STOP] Initiating shutdown for {self.name}", flush=True)

        # Drain queues (writer_thread, render_proc)
        if hasattr(self, "write_queue"):
            self.write_queue.put(None)
        if hasattr(self, "render_queue"):
            self.render_queue.put(None)

        # Shutdown ThreadPools
        # wait=True ensures no 'zombie' threads are left accessing GpuMats
        for pool_name in ["executor", "io_executor", "ffmpeg_executor"]:
            if hasattr(self, pool_name):
                getattr(self, pool_name).shutdown(wait=True)

        # Clean shutdown for daemon threads
        for thread_name in ["writer_thread", "metadata_thread", "display_proc"]:
            thread = getattr(self, thread_name, None)
            if thread and thread.is_alive():
                thread.join(timeout=2)

        # MP cleanup
        if hasattr(self, "render_proc"):
            if self.render_proc.is_alive():
                self.render_proc.join(timeout=10)

                # If it still hasn't closed, force terminate
                if self.render_proc.is_alive():
                    self.render_proc.terminate()

            # Explicitly clear the queue and close the feeder thread
            self.render_queue.close()
            self.render_queue.join_thread()

        # Shared memory cleanup
        if hasattr(self, "shms"):
            for shm in self.shms:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

        if hasattr(self, "manager"):
            self.manager.shutdown()

        # Force the final clip to rotate even if under 10 seconds
        if self.config.ENABLE_QUERYING and self.video_writer:
            print(f" [STOP] Forcing final clip rotation for {self.name}", flush=True)
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
        if hasattr(self, "cap"):
            self.cap.release()
            self.cap = None

        # Stop reader thread
        # if hasattr(self, "reader"):
        #     self.reader.stop()
        if hasattr(self, "reader") and self.reader is not None:
            self.reader.stop()
            self.reader = None

        # # Signal the renderer to stop and wait for it to finish
        # if hasattr(self, "render_queue"):
        #     self.render_queue.put(None)
        #     self.render_proc.join(timeout=10)

        #     # If it still hasn't closed, force terminate
        #     if self.render_proc.is_alive():
        #         self.render_proc.terminate()

        #     # Explicitly clear the queue and close the feeder thread
        #     self.render_queue.close()
        #     self.render_queue.join_thread()

        if hasattr(self, "write_queue"):
            # Signal the writer queue to unblock the worker thread
            self.write_queue.put(None)

            # Wait for the _video_writer thread to finish writing remaining frames
            timeout = 10
            start_wait = time.time()
            while not self.writer_done and (time.time() - start_wait < timeout):
                time.sleep(0.1)

            if not self.writer_done:
                print(f"[WARNING] Writer drain timed out for {self.name}.")

        # Unblock any waiting FastAPI generators
        self.frame_ready_event.set()

        #  Nullify the model reference to trigger automatic cleanup
        if hasattr(self, "model") and self.model is not None:
            # Check if it's an Ultralytics model with a predictor
            predictor = getattr(self.model, "predictor", None)
            if predictor is not None:
                try:
                    predictor.results = []
                except AttributeError:
                    pass
            self.model = None

        # Force Python to run destructors NOW while streams are still alive
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Final sync to clear event queue
            time.sleep(0.2)

        # Purge HW Buffers
        self._is_stopped = True
        if self.device == "GPU":
            self.cleanup_gpu()
        else:
            self.cleanup_cpu()

        #  Hard clear the GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Critical for multi-process/multiprocessing cleanup

        self._check_shm_safety(threshold_percent=0)  # Forced cleanup of current tmp
        print(f" [STOP] {self.name} resources fully released.", flush=True)

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
                    stream=False,
                )

            # Force GPU to finish before returning
            if self.device_input == "cuda":
                torch.cuda.synchronize()

        print(f"Warmup complete for {self.name}", flush=True)

    def get_executor_backlog(self):
        """Returns the number of tasks currently waiting in the thread pool queue."""
        # access the internal queue of the executor
        return self.executor._work_queue.qsize()

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

    # # Gets video fps and framecount
    # def get_fps_and_framecnt(self, cap, target_fps, clip_duration=None):
    #     self.input_fps = int(cap.get(cv2.CAP_PROP_FPS))  # hardware fps
    #     # print(f"in fps: {sself.input_fps} target fps: {target_fps}")
    #     if self.input_fps == 0:  # Case when FPS isn't available
    #         self.input_fps = manual_fps_calculation(cap, num_frames=10)
    #         print(f"new in fps: {self.input_fps}")

    #     self.target_fps = (
    #         target_fps
    #         if target_fps not in [None, 0] and self.input_fps > target_fps
    #         else self.input_fps
    #     )

    #     self.frame_skip = int(self.input_fps / self.target_fps)
    #     if self.frame_skip < 1:
    #         self.frame_skip = 1
    #     # self.skip_count = self.frame_skip - 1

    #     if clip_duration is None:
    #         frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #         clip_duration = frame_count / self.input_fps
    #     self.max_frames_per_clip = int(self.target_fps * float(clip_duration))
    #     self.frame_interval = 1.0 / self.target_fps  # 0.0666s
    #     print(
    #         f"in fps: {self.input_fps} self.target fps: {self.target_fps} self.frame_skip: {self.frame_skip}"
    #     )

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

    def update_frame(self):
        self.stat_frame_count += 1
        elapsed = time.perf_counter() - self.stat_start_time
        if elapsed > 0.5:
            self.stat_fps = round(self.stat_frame_count / elapsed, 1)

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

        results = self.model.predict(
            frame,
            imgsz=imgsz,
            batch=batch,
            device=device_input,
            verbose=False,
            stream=stream,
            conf=self.config.DETECTION_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            rect=True,  # False,  #
        )
        return results

    def _encode_and_signal(self, data, frame_num):
        """Worker task for JPEG encoding. Optimized for zero-copy GPU transfers."""
        if data is None:
            return

        if isinstance(data, cv2.cuda.GpuMat):
            # GPU PATH: Use existing 8K GPU pointer. Resize on GPU (Instant) 🏎️
            cv2.cuda.resize(
                data,
                (self.disp_w, self.disp_h),
                stream=self.encode_stream,
                dst=self.gpu_display_frame,
            )
            # Only download the small (640x360) frame, not the 8K frame!
            display_frame = self.gpu_display_frame.download(self.encode_stream)
        else:
            # CPU PATH: Resize and force memory contiguity for faster encoding
            display_frame = cv2.resize(
                data, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST
            )
            display_frame = np.ascontiguousarray(display_frame)

        # Drop quality to 35. This reduces the bytes FastAPI has to push to the browser.
        success, buffer = cv2.imencode(
            ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 35]
        )

        if success:
            self.latest_processed_frame = buffer.tobytes()
            self.last_delivered_frame_id = frame_num
            self.last_heartbeat = time.time()
            # Non-blocking signal to FastAPI
            self.loop.call_soon_threadsafe(self.frame_ready_event.set)

    def update_ui_fallback(self, frame, frame_num):
        # If backlog is very high, drop JPEG quality to 25 to clear the 'pause' faster
        backlog = self.get_executor_backlog()
        adaptive_quality = (
            20
            if backlog > (self.dynamic_limit * 2)
            else self.config.DISPLAY_FRAME_QUALITY
        )

        # FALLBACK: If AI is busy, worker thread encodes raw frame for the UI.
        # This offloads the 40ms CPU cost from the main Producer loop.
        display_frame = cv2.resize(
            frame, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST
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

    # def process_frame_async(
    #     self, clip_filename, frame, overall_frame_num, skip_ai=False
    # ):
    #     """
    #     Worker function to run heavy AI tasks (Resize, Bkgd Sub, YOLO)
    #     in the background without blocking the video reader.
    #     """
    #     try:
    #         # is_target_frame = frame_num % self.frame_skip == 0
    #         is_target_frame = overall_frame_num >= self.next_process_idx
    #         inf_data = None
    #         with torch.no_grad():
    #             try:
    #                 # inf_data = self.rbtd_full_gpu(frame)
    #                 if self.device_input == "cuda":
    #                     inf_data = self.rbtd_full_gpu(frame)  # , frame_num)
    #                 else:
    #                     inf_data = self.rbtd_full_cpu(frame)  # , frame_num)

    #                 bbs_full_res = []
    #                 if is_target_frame:
    #                     self.next_process_idx += self.step_size
    #                     self.frame_count_target += 1  # 1-indexed

    #                     if inf_data:
    #                         if self.config.ENABLE_QUERYING:
    #                             try:
    #                                 # self.write_queue.put_nowait((frame_num, frame_to_queue))
    #                                 self.write_queue.put(
    #                                     # (frame_num, frame_to_queue)
    #                                     {
    #                                         "frame": self.resized_frame.clone(),
    #                                         "writer_to_finalize": self.video_writer,
    #                                     }
    #                                 )
    #                             except queue.Full:
    #                                 # This tells you that the writer is the bottleneck,
    #                                 # but it won't stall your detection latency.
    #                                 pass

    #                         inf_data["frameNum"] = self.frame_count_target
    #                         self.async_yolo_task(
    #                             clip_filename, inf_data, self.config.ROI_RETURN_BYTES
    #                         )

    #                     else:  # elif frame_num % 3 == 0:
    #                         # else:
    #                         # NO MOTION DETECTED FALLBACK:
    #                         # We call update_ui_fallback from INSIDE this worker thread.
    #                         # This preserves the exact queue order.
    #                         # self.update_ui_fallback(frame, frame_num)

    #                         # Pass the GpuMat directly to the encoder pool
    #                         # display_source = inf_data["full_frame"] if inf_data else frame
    #                         self.executor.submit(
    #                             self._encode_and_signal, frame, self.frame_count_target
    #                         )
    #                     # bbs_full_res = self.get_gpu_rois(frame, frame_num, inf_data["mask"])
    #                     # frame_to_queue = (inf_data["full_frame"], bbs_full_res)

    #                     # try:
    #                     #     # self.write_queue.put_nowait((frame_num, frame_to_queue))
    #                     #     self.write_queue.put((frame_num, frame_to_queue))
    #                     # except queue.Full:
    #                     #     # This tells you that the writer is the bottleneck,
    #                     #     # but it won't stall your detection latency.
    #                     #     pass

    #             except Exception:
    #                 e = traceback.format_exc()
    #                 print(
    #                     f"Pipeline failure for {self.name} at frame {overall_frame_num}: {e}",
    #                     flush=True,
    #                 )
    #     except Exception:
    #         logging.error(f"Pipeline failure for {self.name}: {traceback.format_exc()}")
    #     finally:
    #         # CRITICAL: Manually clear the 8K frame reference from this thread
    #         # This ensures the 96MB is released before the next task starts
    #         del frame
    #         if "inf_data" in locals():
    #             del inf_data

    # def run_realtime_inferencev1(self):
    #     """Producer: Maintains the target FPS and updates clip IDs."""
    #     last_frame_time = time.perf_counter()
    #     while self.active:
    #         frame = self.reader.read()
    #         if frame is not None:
    #             self.frame_count += 1

    #             # Calculate a dynamic limit: tolerate 0.5 seconds of lag.
    #             # If target_fps is 15, the limit is 7. If target_fps is 30, the limit is 15.
    #             self.dynamic_limit = max(2, int(0.5 * self.target_fps))

    #             # Determine if this frame should be AI or Raw based on backlog
    #             # But ALWAYS submit to the executor to maintain frame order.
    #             backlog = self.get_executor_backlog()

    #             while backlog > 4 and self.active:
    #                 time.sleep(0.005)
    #                 backlog = self.get_executor_backlog()

    #             # Use a "Skip AI" flag instead of a "continue" skip
    #             skip_ai = backlog > self.dynamic_limit

    #             self.frame_in_clip_count += 1

    #             # Immediate Rotation (Ensures logs match the new key instantly)
    #             if (
    #                 self.config.ENABLE_QUERYING
    #                 and self.frame_in_clip_count > self.max_frames_per_clip
    #             ):
    #                 # Perform safety check BEFORE starting the next 185MB clip
    #                 # self._check_shm_safety(threshold_percent=90)

    #                 # Pass old state to consumer before updating
    #                 old_writer = self.video_writer
    #                 old_key, old_tmp, old_final, old_frame_in_clip_count = (
    #                     self.clip_key,
    #                     self.tmp_file,
    #                     self.clip_filename,
    #                     self.frame_in_clip_count,
    #                 )

    #                 self.write_queue.put(
    #                     {
    #                         "control": "ROTATE",
    #                         "old_key": old_key,
    #                         "old_tmp": old_tmp,
    #                         "old_final": old_final,
    #                         "frame_in_clip_count": old_frame_in_clip_count,
    #                         "writer_to_finalize": old_writer,
    #                     }
    #                 )

    #                 self.clip_id += 1
    #                 self.frame_in_clip_count = 1
    #                 # self.video_writer = None # Nullify on main thread
    #                 self._initialize_writer()
    #                 # self.clip_id += 1
    #                 # self.frame_in_clip_count = 1
    #                 # self.video_writer = None # Nullify on main thread
    #                 # self._initialize_writer()
    #                 self._check_shm_safety(threshold_percent=90)

    #             # Handoff to AI and Writer
    #             if self.active:
    #                 self.executor.submit(
    #                     self.process_frame_async,
    #                     self.clip_filename,
    #                     frame,
    #                     self.frame_count,
    #                     skip_ai,
    #                 )

    #             # --- PRECISE CLOCK SYNC ---
    #             # This prevents the producer from "lapping" the consumer
    #             # and building that jumpy backlog in the first place.
    #             elapsed = time.perf_counter() - last_frame_time
    #             if elapsed < self.frame_interval:
    #                 # time.sleep(self.frame_interval - elapsed)
    #                 # Subtract a small epsilon (0.001) for OS scheduling overhead
    #                 time.sleep(max(0, self.frame_interval - elapsed - 0.0015))
    #             last_frame_time = time.perf_counter()

    #             self.update_frame()
    #             self.last_heartbeat = time.time()

    #         elif self.reader.stopped:
    #             self.active = False
    #             break
    #         else:
    #             time.sleep(0.01)
    #     self.stop()

    def start_new_clip(self):
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

    # def rendering_worker(self, queue, shared_details):
    #     # Get shm
    #     shm_name = shared_details.get("shm_name")
    #     try:
    #         shm = mp.shared_memory.SharedMemory(name=shm_name)
    #     except Exception:
    #         return

    #     try:
    #         while True:
    #             item = queue.get()
    #             if item is None:  # Sentinel value to stop the worker
    #                 break

    #             # frame is resized
    #             # metadata in resized res
    #             display_frame, frameNum, metadata_or_bbs, class_list = item

    #             # display_size = (self.resize_h, self.resize_w)
    #             # display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)

    #             if isinstance(metadata_or_bbs, dict):
    #                 # Case: Object Detection
    #                 for _, obj in metadata_or_bbs.items():
    #                     bbox = obj["bbox"]
    #                     x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    #                     class_name = bbox["object"]
    #                     class_id = class_list.index(class_name)
    #                     confidence = bbox["object_det"]["confidence"]

    #                     bb_color = get_detection_color(class_id, is_bgr=True)
    #                     label = f"{class_name} {confidence:.2f}"

    #                     # Draw on a copy of the frame to avoid modifying the original
    #                     cv2.rectangle(display_frame, (x, y), (x + w, y + h), bb_color, 2)
    #                     # Replace draw_label if it's accessible; otherwise use cv2.putText
    #                     # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #                     draw_label(display_frame, label, (x, y), color=bb_color, padding=5)
    #             elif metadata_or_bbs is not None:
    #                 # Case: Motion Detections Only (SF Path)
    #                 for box in metadata_or_bbs:
    #                     if torch.is_tensor(box):
    #                         box = box.tolist()

    #                     x1, y1, x2, y2 = map(int, box)
    #                     # Draw on a copy of the frame to avoid modifying the original
    #                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 225), 2)

    #             # writer.write(display_frame)
    #             if frameNum > shared_details["last_id"]:  # self.last_delivered_frame_id:
    #                 frame_bytes = get_display_frame_in_bytes(
    #                     display_frame,
    #                     display_size=(self.disp_w, self.disp_h),
    #                     quality=self.config.DISPLAY_FRAME_QUALITY,
    #                     return_bytes=True,
    #                 )
    #                 if frame_bytes:
    #                     frame_len = len(frame_bytes)
    #                     # Zero-copy write to RAM
    #                     shm.buf[:frame_len] = frame_bytes
    #                     shared_details["frame_length"] = frame_len
    #                     shared_details["last_id"] = frameNum
    #                     # self.last_frame_id = frameNum
    #                     # self.last_heartbeat = time.time()
    #                     # Signal the FastAPI generator that a new frame is ready
    #                     # self.loop.call_soon_threadsafe(self.frame_ready_event.set)
    #                     # self.mp_frame_ready_event.set()
    #                     try:
    #                         self.signal_queue.put_nowait(True)
    #                     except Exception:
    #                         pass

    #         # END While
    #     finally:
    #         shm.close()

    def run_realtime_inference(self, sf_enabled, pipeline_fn):
        """Producer: Maintains the target FPS and updates clip IDs."""
        # Calculate a dynamic limit: tolerate 0.5 seconds of lag.
        # If target_fps is 15, the limit is 7. If target_fps is 30, the limit is 15.
        self.dynamic_limit = max(2, int(0.5 * self.target_fps))
        last_frame_time = time.perf_counter()
        while self.active:
            device_frame, frame_num = self.reader.read()
            if device_frame is None:
                if self.reader.stopped:
                    self.active = False
                    break
                continue

            # Keep 8K frame on GPU (Skip CPU conversion for non-target frames)
            # if self.device_input == "cuda" and not self.reader.is_h264_8k:
            #     with torch.cuda.stream(self.ingest_stream):
            #         device_frame = nv12_to_rgb_torch(
            #             device_frame, self.frame_height, self.frame_width#  , is_bgr=False
            #         )
            #     self.ingest_stream.synchronize()

            # if device_frame is not None:
            self.frame_count += 1
            is_target_frame = frame_num >= self.next_process_idx

            # Determine if this frame should be AI or Raw based on backlog
            # But ALWAYS submit to the executor to maintain frame order.
            backlog = self.get_executor_backlog()

            while backlog > 4 and self.active:
                time.sleep(0.005)
                backlog = self.get_executor_backlog()

            # Use a "Skip AI" flag instead of a "continue" skip
            # skip_ai = backlog > self.dynamic_limit

            # self.frame_in_clip_count += 1

            # # Immediate Rotation (Ensures logs match the new key instantly)
            # if (
            #     self.config.ENABLE_QUERYING
            #     and self.frame_in_clip_count > self.max_frames_per_clip
            # ):
            #     # Perform safety check BEFORE starting the next 185MB clip
            #     # self._check_shm_safety(threshold_percent=90)

            #     # Pass old state to consumer before updating
            #     old_writer = self.video_writer
            #     old_key, old_tmp, old_final, old_frame_in_clip_count = (
            #         self.clip_key,
            #         self.tmp_file,
            #         self.clip_filename,
            #         self.frame_in_clip_count,
            #     )

            #     self.write_queue.put(
            #         {
            #             "control": "ROTATE",
            #             "old_key": old_key,
            #             "old_tmp": old_tmp,
            #             "old_final": old_final,
            #             "frame_in_clip_count": old_frame_in_clip_count,
            #             "writer_to_finalize": old_writer,
            #         }
            #     )

            #     self.clip_id += 1
            #     self.frame_in_clip_count = 1
            #     # self.video_writer = None # Nullify on main thread
            #     self._initialize_writer()
            #     # self.clip_id += 1
            #     # self.frame_in_clip_count = 1
            #     # self.video_writer = None # Nullify on main thread
            #     # self._initialize_writer()
            #     self._check_shm_safety(threshold_percent=90)

            def wrapped_fn(*args):
                with torch.cuda.stream(self.inference_stream):
                    pipeline_fn(*args)

            # Handoff to AI and Writer
            if self.active:
                self.executor.submit(
                    # pipeline_fn,
                    wrapped_fn,
                    device_frame,
                    frame_num,
                    is_target_frame,
                )

            # --- PRECISE CLOCK SYNC ---
            # This prevents the producer from "lapping" the consumer
            # and building that jumpy backlog in the first place.
            elapsed = time.perf_counter() - last_frame_time
            if elapsed < self.frame_interval:
                # time.sleep(self.frame_interval - elapsed)
                # Subtract a small epsilon (0.001) for OS scheduling overhead
                time.sleep(max(0, self.frame_interval - elapsed - 0.0015))
            last_frame_time = time.perf_counter()

            self.update_frame()
            self.last_heartbeat = time.time()

        self.stop()

    def _initialize_writer(self):
        """Sets up a new VideoWriter on a RAM disk for near-zero latency."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # clip_filename has a unique name indicating clip # and timestamp
        # if "://" not in str(self.source):
        self.clip_filename = f"{self.config.SHARED_OUTPUT}/{self.name}_{self.clip_id}_{timestamp}_{os.getpid()}.mp4"
        # else:
        # self.clip_filename = f"{self.config.SHARED_OUTPUT}/{self.name}_{timestamp}.mp4"

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
        # Non-blocking download if using GPU
        if hasattr(frame_payload, "download"):
            # Use the dedicated encode_stream to prevent blocking the AI stream
            frame = frame_payload.download(self.encode_stream)
            # self.encode_stream.waitForCompletion()
        else:
            frame = frame_payload
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        # FORCE RESIZE to match (resize_w, resize_h) exactly
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

                if data.get("control") == "ROTATE":
                    # Release the writer only AFTER all frames in queue are written
                    writer_to_close = data.get("writer_to_finalize")
                    # if writer_to_close:
                    #     # writer_to_close.release()
                    #     # Finalize in background so we can immediately start the next clip
                    #     self.io_executor.submit(writer_to_close.release)

                    # clip_metadata = all_metadata.pop(data["old_key"], {"object": {}, "face": {}})

                    # # Submit the background re-encode task
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

    # def get_detections_for_contours_bbs(
    #     self,
    #     frameNum,
    #     frame_raw,
    #     merged,
    #     thickness=BASE_PIPELINE_CONFIG.THICKNESS,
    #     device_input="cuda",
    #     return_bytes=True,
    # ):
    #     """
    #     Motion-Triggered YOLO Inference Logic.

    #     Instead of running YOLO on a massive 8K frame, this function:
    #     1. Extracts bounding boxes from the motion contours.
    #     2. Merges nearby boxes into optimal 640x640 crops.
    #     3. Runs a single batch inference on those crops.
    #     4. Maps detection coordinates back to the original 8K space.

    #     Args:
    #         foi (np.ndarray): 'Frame of Interest' (the 8K raw frame).
    #         contours (list): Contours extracted from the motion mask.
    #     """
    #     stream_name = self.name
    #     num_objs = 0
    #     metadata = dict()
    #     cropped_imgs, cropped_coords = [], []

    #     if torch.is_tensor(frame_raw):
    #         # Case: Tensor (3, H, W) -> NumPy (H, W, 3)
    #         # Move to CPU
    #         # .permute(1, 2, 0) transforms [C, H, W] -> [H, W, C] (OpenCV format)
    #         # .byte() converts from FP16 (Pool) back to uint8 (Required for drawing)
    #         # .contiguous() fixes the memory layout for OpenCV C++ compatibility
    #         foi = frame_raw.permute(1, 2, 0).byte().cpu().numpy()
    #         # foi = frame_raw.permute(1, 2, 0).to(torch.uint8).cpu().numpy().copy()
    #         # foi = np.ascontiguousarray(foi)
    #     elif isinstance(frame_raw, np.ndarray) and frame_raw.shape[0] == 3:
    #         # Case: NumPy (3, H, W) -> NumPy (H, W, 3)
    #         foi = frame_raw.transpose(1, 2, 0)
    #     else:
    #         # foi = np.ascontiguousarray(frame_raw)
    #         foi = frame_raw
    #     # Always force contiguity for OpenCV C++ compatibility
    #     foi = np.ascontiguousarray(foi)
    #     H, W = foi.shape[:2]  # Unpack once

    #     # if torch.is_tensor(frame_raw):
    #     #     # Move to CPU
    #     #     foi = frame_raw.cpu()

    #     #     # Fix Shape: PyTorch [C, H, W] -> OpenCV [H, W, C]
    #     #     if foi.ndim == 3 and foi.shape[0] == 3:
    #     #         foi = foi.permute(1, 2, 0)

    #     #     foi = foi.numpy()

    #     #     # Fix Color: RGB (from hardware decoder) -> BGR (for OpenCV)
    #     #     foi = cv2.cvtColor(foi, cv2.COLOR_RGB2BGR)

    #     # elif isinstance(frame_raw, cv2.cuda.GpuMat):
    #     #     # Standard OpenCV CUDA download
    #     #     foi = frame_raw.download()
    #     # else:
    #     #     foi = frame_raw

    #     if merged is None or len(merged) == 0:
    #         adaptive_quality = (
    #             30 if self.config.ENABLE_QUERYING else self.config.DISPLAY_FRAME_QUALITY
    #         )
    #         frame_bytes = get_display_frame_in_bytes(
    #             foi,
    #             display_size=(self.disp_w, self.disp_h),
    #             quality=adaptive_quality,
    #             return_bytes=return_bytes,
    #         )
    #         return metadata, frame_bytes  # num_objs, predictions

    #     # Filter small noise and convert contours to 8K-space bounding boxes
    #     # if device_input == "cpu":
    #     #     merged = self.filter_rois(sorted_bbs, dist_thresh=self.dist_thresh_8k)
    #     # else:
    #     #     merged = sorted_bbs

    #     # dist_thresh = min(0.05 * W, 0.05 * H)
    #     # merged = self.filter_rois(bbs_full_res, dist_thresh=dist_thresh)

    #     # Extract crops at full-resolution
    #     crop_cnt = 0
    #     for i in range(len(merged)):
    #         x1, y1, x2, y2 = [int(x) for x in merged[i]]
    #         if (
    #             (x2 - x1) > 31
    #             and (y2 - y1) > 31
    #             and (x2 - x1) < self.frame_width
    #             and (y2 - y1) < self.frame_height
    #         ):
    #             crop_cnt += 1
    #             if self.config.DETECTION_TYPE == "motion":
    #                 # print(f"DEBUG: foi shape {foi.shape}, dtype {foi.dtype}, contiguous {foi.flags['C_CONTIGUOUS']}")
    #                 # self.stop()
    #                 foi = cv2.rectangle(
    #                     foi,
    #                     (x1, y1),
    #                     (x2, y2),
    #                     (0, 0, 255),
    #                     thickness,
    #                 )
    #             else:
    #                 crop = foi[y1:y2, x1:x2]
    #                 cropped_imgs.append(crop)
    #                 cropped_coords.append((x1, y1))

    #             if crop_cnt == self.config.MODEL_MAX_BATCH_SIZE:
    #                 # logging.warning(
    #                 #     f"⚠️ [LIMIT] {self.name} found {len(cropped_imgs)} contours. Capping to 64 for TensorRT."
    #                 # )
    #                 # cropped_imgs = cropped_imgs[:self.config.MODEL_MAX_BATCH_SIZE]
    #                 # cropped_coords = cropped_coords[:self.config.MODEL_MAX_BATCH_SIZE]
    #                 break

    #     if not cropped_imgs or self.config.DETECTION_TYPE == "motion":
    #         frame_bytes = get_display_frame_in_bytes(
    #             foi,
    #             display_size=(self.disp_w, self.disp_h),
    #             quality=self.config.DISPLAY_FRAME_QUALITY,
    #             return_bytes=return_bytes,
    #         )
    #         return metadata, frame_bytes  # num_objs, predictions

    #     # if len(cropped_imgs) > self.config.MODEL_MAX_BATCH_SIZE:
    #     #     logging.warning(
    #     #         f"⚠️ [LIMIT] {self.name} found {len(cropped_imgs)} contours. Capping to 64 for TensorRT."
    #     #     )
    #     #     cropped_imgs = cropped_imgs[:self.config.MODEL_MAX_BATCH_SIZE]
    #     #     cropped_coords = cropped_coords[:self.config.MODEL_MAX_BATCH_SIZE]

    #     # Ensure all previous GPU tasks (CuPy/OpenCV) are finished before YOLO starts
    #     if device_input == "cuda":
    #         torch.cuda.synchronize()

    #     # Run Inference (Keep stream=False as it is stable)
    #     # results = self.run_model(
    #     #     cropped_imgs,
    #     #     imgsz=(self.resize_h, self.resize_w),
    #     #     batch=len(cropped_imgs),
    #     #     device_input=device_input,
    #     #     stream=False,   # With true, it treats crops 1 by 1True,
    #     # )

    #     # --- REFACTORED BATCH INFERENCE (Line 2209) ---
    #     max_batch = getattr(self.config, "MODEL_MAX_BATCH_SIZE", 32)  # e.g., 64
    #     all_results = []

    #     # Split the 95+ crops into chunks (e.g., 64, then 31)
    #     for i in range(0, len(cropped_imgs), max_batch):
    #         batch_imgs = cropped_imgs[i : i + max_batch]

    #         # Run batch inference with stream=False to ensure parallel execution
    #         batch_results = self.run_model(
    #             batch_imgs,
    #             imgsz=(self.resize_h, self.resize_w),
    #             batch=len(batch_imgs),
    #             device_input=device_input,
    #             stream=False,  # Crucial: False enables actual hardware batching
    #         )
    #         all_results.extend(list(batch_results))

    #     # Now replace 'results' with our combined list for the rest of your loop
    #     results = all_results
    #     # ---------------------------

    #     # Process results and draw 8K-space overlays
    #     for ridx, r in enumerate(list(results)):
    #         if r.boxes is None or len(r.boxes) == 0:
    #             continue

    #         # Move to CPU in one bulk operation per crop
    #         boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    #         clss = r.boxes.cls.cpu().numpy().astype(int)
    #         confs = r.boxes.conf.cpu().numpy()
    #         off_x, off_y = cropped_coords[ridx]

    #         for j in range(len(boxes)):
    #             num_objs += 1
    #             bx1, by1, bx2, by2 = boxes[j]
    #             abs_x1, abs_y1 = off_x + bx1, off_y + by1
    #             abs_x2, abs_y2 = off_x + bx2, off_y + by2
    #             class_id = clss[j]
    #             class_name = self.label_source[class_id]
    #             confidence = confs[j]
    #             if confidence > self.config.DETECTION_THRESHOLD:
    #                 if not self.config.OMIT_DETECTIONS_FLAG:
    #                     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    #                     print(
    #                         # f"[OBJECT DETECTION] {class_name} detected in frame {frameNum} (Total detected: {current_cnt})",
    #                         f"[{timestamp}] {stream_name} DETECTION on Frame {frameNum}: {class_name} detected",
    #                         flush=True,
    #                     )

    #                 bb_color = get_detection_color(class_id, is_bgr=True)

    #                 foi = cv2.rectangle(
    #                     foi,
    #                     (abs_x1, abs_y1),
    #                     (abs_x2, abs_y2),
    #                     bb_color,
    #                     thickness,
    #                 )
    #                 label = f"{class_name} {confidence:.2f}"
    #                 draw_label(foi, label, (abs_x1, abs_y1), color=bb_color, padding=5)

    #                 height = min(abs_y2, H) - max(0, abs_y1)
    #                 width = min(abs_x2, W) - max(0, abs_x1)
    #                 # object_res = [
    #                 #     abs_x1,
    #                 #     abs_y1,
    #                 #     height,
    #                 #     width,
    #                 #     class_name,
    #                 #     confidence,
    #                 #     H,
    #                 #     W,
    #                 # ]

    #                 # Resized
    #                 scale_x = self.resize_w / W
    #                 scale_y = self.resize_h / H
    #                 object_res = [
    #                     int(abs_x1 * scale_x),
    #                     int(abs_y1 * scale_y),
    #                     int(height * scale_y),
    #                     int(width * scale_x),
    #                     class_name,
    #                     confidence,
    #                     int(self.resize_h),
    #                     int(self.resize_w),
    #                 ]

    #                 framenum_str = f"{frameNum:04d}_{j:04d}"
    #                 if self.config.DEBUG_FLAG:
    #                     meta_str = ",".join(
    #                         [str(o) for o in object_res + [framenum_str]]
    #                     )
    #                     print(f"[{stream_name} METADATA],{meta_str}", flush=True)

    #                 # Full Res
    #                 metadata[framenum_str] = {
    #                     "frameId": frameNum,
    #                     "bbId": framenum_str,
    #                     "bbox": {
    #                         "x": int(object_res[0]),
    #                         "y": int(object_res[1]),
    #                         "height": int(object_res[2]),
    #                         "width": int(object_res[3]),
    #                         "object": str(object_res[4]),
    #                         "object_det": {
    #                             "confidence": float(object_res[5]),
    #                             "frameH": int(object_res[6]),
    #                             "frameW": int(object_res[7]),
    #                         },
    #                     },
    #                 }

    #     # Queue frame for display (reduce quality for 8K bandwidth)
    #     frame_bytes = get_display_frame_in_bytes(
    #         foi,
    #         display_size=(self.disp_w, self.disp_h),
    #         quality=self.config.DISPLAY_FRAME_QUALITY,
    #         return_bytes=return_bytes,
    #     )

    #     return metadata, frame_bytes

    def get_detectionsv2(
        self,
        frame_raw,  # BGR
        frameNum,
        merged=None,
        thickness=2,
        device_input="cuda",
    ):
        metadata = {}
        is_torch = torch.is_tensor(frame_raw)
        # H, W = frame_raw.shape[:2]  # Unpack once
        #  FIX: Get H/W correctly for both Numpy [H, W, C] and Tensor [C, H, W]
        if is_torch:
            H, W = frame_raw.shape[-2:]  # Gets last two dims
            # Single flip to RGB for YOLO
            frame_rgb = frame_raw.flip(-3)
        else:
            H, W = frame_raw.shape[:2]
            frame_rgb = frame_raw[:, :, ::-1]

        input_list = []
        offsets = []  # Tracks where each crop started in 8K space

        # FULL FRAME INFERENCE
        if merged is None:
            # Run Inference (Keep stream=False as it is stable)
            # results = self.run_model(
            #     frame_rgb,  # Should be RGB
            #     imgsz=(H, W),
            #     batch=1,
            #     device_input=device_input,
            #     stream=False,
            # )
            # input_data = frame_rgb.unsqueeze(0) if is_torch else [frame_rgb]
            # offsets.append((0, 0))
            imgsz = (H, W)
            if torch.is_tensor(frame_raw):
                # Swap channels (-3 is the C dim) and normalize
                frame_input = (
                    # frame_raw.float() / 255.0
                    frame_raw.flip(-3).float() / 255.0
                )
                if frame_input.ndim == 3:
                    frame_input = frame_input.unsqueeze(0)
            else:
                frame_input = frame_raw

            # Run Inference (Keep stream=False as it is stable)
            results = self.run_model(
                frame_input,  # Should be RGB
                imgsz=imgsz,
                batch=1,
                device_input=device_input,
                stream=False,
            )

        # ROI INFERENCE
        else:
            # cropped_batch = []
            # cropped_coords = []
            # PREPARE CROPS (Path-specific Optimization)
            half = int(self.config.MODEL_W / 2)
            for box in merged:
                x1, y1, x2, y2 = [int(val) for val in box]

                # 1. Find Center of Motion
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 2. Define 640x640 Window (Extended Scope)
                # This removes resizing and provides native resolution context
                nx1 = max(0, min(cx - half, W - 640))
                ny1 = max(0, min(cy - half, H - 640))
                nx2, ny2 = nx1 + 640, ny1 + 640

                offsets.append((nx1, ny1))

                # 3. Slice Crop
                if is_torch:
                    crop = frame_rgb[:, ny1:ny2, nx1:nx2].unsqueeze(
                        0
                    )  # .to(torch.half)
                    # input_list.append(crop)
                else:
                    crop = frame_rgb[ny1:ny2, nx1:nx2]
                    # input_list.append(crop)
                crop = crop.flip(-3).float() / 255.0
                input_list.append(crop)

            input_data = torch.cat(input_list, dim=0) if is_torch else input_list
            imgsz = 640

            # results = self.run_model(input_data, imgsz=imgsz, batch=len(offsets), device_input=device_input, stream=False)
            # CHUNKED BATCH INFERENCE
            # Process in chunks of MODEL_MAX_BATCH_SIZE to stay within TensorRT/OpenVINO profile limits
            results = []
            for i in range(0, len(input_data), self.config.MODEL_MAX_BATCH_SIZE):
                chunk = input_data[i : i + self.config.MODEL_MAX_BATCH_SIZE]
                chunk_results = self.run_model(
                    chunk,  # Should be RGB
                    imgsz=(self.resize_h, self.resize_w),
                    batch=len(chunk),
                    device_input=device_input,
                    stream=False,  # stream=False is critical
                )
                results.extend(list(chunk_results))

        # Process results and draw 8K-space overlays
        # Display scales for the final 640x640 stretched output
        num_objs = 0
        scale_display_x = self.resize_w / W  # 640 / 8192
        scale_display_y = self.resize_h / H  # 640 / 4608
        for ridx, r in enumerate(list(results)):
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Determine the ROI expansion ratio for this specific crop
            off_x, off_y = offsets[ridx]

            # Move to CPU in one bulk operation per crop
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for j in range(len(boxes)):
                num_objs += 1

                bx1, by1, bx2, by2 = boxes[j]
                abs_x1, abs_y1 = off_x + bx1, off_y + by1
                abs_x2, abs_y2 = off_x + bx2, off_y + by2

                # Map to 640x640 Display pixels (Apply the non-uniform stretch)
                disp_x = abs_x1 * scale_display_x
                disp_y = abs_y1 * scale_display_y
                disp_w = (abs_x2 - abs_x1) * scale_display_x
                disp_h = (abs_y2 - abs_y1) * scale_display_y

                class_id = clss[j]
                class_name = self.label_source[class_id]
                confidence = confs[j]

                if not self.config.OMIT_DETECTIONS_FLAG:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    print(
                        # f"[OBJECT DETECTION] {class_name} detected in frame {frameNum} (Total detected: {current_cnt})",
                        f"[{timestamp}] {self.name} DETECTION on Frame {frameNum}: {class_name} detected",
                        flush=True,
                    )

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

                framenum_str = f"{frameNum:04d}_{j:04d}"
                # if self.config.DEBUG_FLAG:
                #     meta_str = ",".join([str(o) for o in object_res + [framenum_str]])
                #     print(f"[{self.name} METADATA],{meta_str}", flush=True)

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

        return metadata, None

    def get_detections(
        self,
        frame_raw,  # BGR
        frameNum,
        merged=None,
        thickness=2,
        device_input="cuda",
    ):
        metadata = {}
        # H, W = frame_raw.shape[:2]  # Unpack once
        #  FIX: Get H/W correctly for both Numpy [H, W, C] and Tensor [C, H, W]
        if torch.is_tensor(frame_raw):
            H, W = frame_raw.shape[-2:]  # Gets last two dims
        else:
            H, W = frame_raw.shape[:2]

        if merged is None:
            if torch.is_tensor(frame_raw):
                # Swap channels (-3 is the C dim) and normalize
                frame_input = (
                    # frame_raw.float() / 255.0
                    frame_raw.flip(-3).float() / 255.0
                )
                if frame_input.ndim == 3:
                    frame_input = frame_input.unsqueeze(0)
            else:
                frame_input = frame_raw

            # Run Inference (Keep stream=False as it is stable)
            results = self.run_model(
                frame_input,  # Should be RGB
                imgsz=(H, W),
                batch=1,
                device_input=device_input,
                stream=False,
            )
        else:
            cropped_batch = []
            cropped_coords = []
            # PREPARE CROPS (Path-specific Optimization)
            if device_input == "cuda" and torch.is_tensor(frame_raw):
                # GPU PATH: Zero-copy slicing + Hardware Interpolation
                for box in merged:
                    x1, y1, x2, y2 = [int(val) for val in box]
                    cropped_coords.append((x1, y1))
                    crop = frame_raw[:, y1:y2, x1:x2].unsqueeze(0)
                    # crop_float = crop.float() / 255.0
                    crop_float = crop.flip(-3).float() / 255.0
                    # F.interpolate on A6000 handles 100+ crops in ~2ms
                    crop_resized = F.interpolate(
                        crop_float,
                        size=(self.resize_h, self.resize_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    cropped_batch.append(crop_resized.to(torch.half))

                # Consolidate into 4D Tensor for Parallel Hardware Batching
                input_data = torch.cat(cropped_batch, dim=0) if cropped_batch else None
            else:
                # CPU PATH: OpenCV Batching
                foi_cpu = (
                    frame_raw.permute(1, 2, 0).byte().cpu().numpy()
                    if torch.is_tensor(frame_raw)
                    else frame_raw
                )
                for box in merged:
                    x1, y1, x2, y2 = [int(val) for val in box]
                    cropped_coords.append((x1, y1))
                    crop = foi_cpu[y1:y2, x1:x2]
                    # Batching on CPU still needs resized inputs
                    crop_resized = cv2.resize(
                        crop,
                        (self.resize_w, self.resize_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cropped_batch.append(crop_resized)

                input_data = cropped_batch  # List of arrays for OpenVINO/CPU batching

            if not cropped_batch:
                return {}, None

            if self.config.DEBUG_FLAG:
                self.debug_save_crops(cropped_batch, frameNum)

            # CHUNKED BATCH INFERENCE
            # Process in chunks of MODEL_MAX_BATCH_SIZE to stay within TensorRT/OpenVINO profile limits
            results = []
            for i in range(0, len(input_data), self.config.MODEL_MAX_BATCH_SIZE):
                chunk = input_data[i : i + self.config.MODEL_MAX_BATCH_SIZE]
                chunk_results = self.run_model(
                    chunk,  # Should be RGB
                    imgsz=(self.resize_h, self.resize_w),
                    batch=len(chunk),
                    device_input=device_input,
                    stream=False,  # stream=False is critical
                )
                results.extend(list(chunk_results))

        # Process results and draw 8K-space overlays
        # Display scales for the final 640x640 stretched output
        num_objs = 0
        scale_display_x = self.resize_w / W  # 640 / 8192
        scale_display_y = self.resize_h / H  # 640 / 4608
        for ridx, r in enumerate(list(results)):
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Determine the ROI expansion ratio for this specific crop
            if merged is not None:
                x1_8k, y1_8k, x2_8k, y2_8k = merged[ridx]
                off_x, off_y = x1_8k, y1_8k
                # Ratio: How many 8K pixels does one inference pixel represent?
                roi_ratio_x = (x2_8k - x1_8k) / self.resize_w
                roi_ratio_y = (y2_8k - y1_8k) / self.resize_h

            else:
                off_x, off_y = 0, 0
                roi_ratio_x, roi_ratio_y = 1.0, 1.0

            # Move to CPU in one bulk operation per crop
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for j in range(len(boxes)):
                num_objs += 1

                bx1, by1, bx2, by2 = boxes[j]
                # abs_x1, abs_y1 = off_x + bx1, off_y + by1
                # abs_x2, abs_y2 = off_x + bx2, off_y + by2

                # # Map to absolute 8K pixels (Scale crop-coords to 8K then add offset)
                # abs_x1 = off_x + (bx1 * roi_ratio_x)
                # abs_y1 = off_y + (by1 * roi_ratio_y)
                # abs_x2 = off_x + (bx2 * roi_ratio_x)
                # abs_y2 = off_y + (by2 * roi_ratio_y)

                # Map to absolute 8K pixels
                abs_x1 = off_x + (bx1 * roi_ratio_x)
                abs_y1 = off_y + (by1 * roi_ratio_y)
                abs_x2 = off_x + (bx2 * roi_ratio_x)
                abs_y2 = off_y + (by2 * roi_ratio_y)

                # Map to 640x640 Display pixels (Apply the non-uniform stretch)
                # disp_x = int(abs_x1 * scale_display_x)
                # disp_y = int(abs_y1 * scale_display_y)
                # disp_w = int((abs_x2 - abs_x1) * scale_display_x)
                # disp_h = int((abs_y2 - abs_y1) * scale_display_y)
                disp_x = abs_x1 * scale_display_x
                disp_y = abs_y1 * scale_display_y
                disp_w = (abs_x2 - abs_x1) * scale_display_x
                disp_h = (abs_y2 - abs_y1) * scale_display_y

                class_id = clss[j]
                class_name = self.label_source[class_id]
                confidence = confs[j]

                if not self.config.OMIT_DETECTIONS_FLAG:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    print(
                        # f"[OBJECT DETECTION] {class_name} detected in frame {frameNum} (Total detected: {current_cnt})",
                        f"[{timestamp}] {self.name} DETECTION on Frame {frameNum}: {class_name} detected",
                        flush=True,
                    )

                # if not self.config.TEST_MODE:
                #     bb_color = get_detection_color(class_id, is_bgr=True)

                #     foi = cv2.rectangle(
                #         foi,
                #         (abs_x1, abs_y1),
                #         (abs_x2, abs_y2),
                #         bb_color,
                #         thickness,
                #     )
                #     label = f"{class_name} {confidence:.2f}"
                #     draw_label(foi, label, (abs_x1, abs_y1), color=bb_color, padding=5)

                # height = min(abs_y2, H) - max(0, abs_y1)
                # width = min(abs_x2, W) - max(0, abs_x1)

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

                framenum_str = f"{frameNum:04d}_{j:04d}"
                # if self.config.DEBUG_FLAG:
                #     meta_str = ",".join([str(o) for o in object_res + [framenum_str]])
                #     print(f"[{self.name} METADATA],{meta_str}", flush=True)

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

        return metadata, None

        # # Queue frame for display (reduce quality for 8K bandwidth)
        # frame_bytes = get_display_frame_in_bytes(
        #     foi,
        #     display_size=(self.disp_w, self.disp_h),
        #     quality=self.config.DISPLAY_FRAME_QUALITY,
        #     return_bytes=True,
        # )

        # return metadata, frame_bytes

    def get_gpu_rois_by_area(self, mask, max_candidates=100):
        # Get raw boxes from mask (Direct VRAM bridge)
        boxes_gpu = find_contours_gpu_equivalent(mask, stream=self.bgs_stream)

        if boxes_gpu is None or len(boxes_gpu) == 0:
            return torch.empty((0, 4), device=self.device_input)

        # Wrap existing GPU memory as a float tensor (Zero Copy)
        raw_boxes = torch.as_tensor(boxes_gpu, device=self.device_input).float()

        # Vectorized Pre-Filter (Removes noise blobs before merging)
        w = raw_boxes[:, 2] - raw_boxes[:, 0]
        h = raw_boxes[:, 3] - raw_boxes[:, 1]
        mask_filter = (
            (w * h > self.min_contour_area) & (w < self.resize_w) & (h < self.resize_h)
        )
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

    def get_gpu_rois(self, frame, frameNum, mask):
        raw_boxes = self.get_gpu_rois_by_area(mask)

        if raw_boxes.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        if raw_boxes.shape[0] > 1:
            raw_boxes = merge_boxes_gpu(raw_boxes, gap_limit=self.dist_thresh_640)

        clean_640p = self.filter_contained_boxes(
            raw_boxes, overlap_thresh=self.config.ROI_CONTAINMENT_THRESH
        )

        if clean_640p.shape[0] < 1:
            return torch.empty((0, 4), device=self.device_input)

        # Scale to 8K space
        return clean_640p * self.scales_tensor

    def get_cpu_rois(self, frame, frameNum, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes_xywh = [
            list(cv2.boundingRect(c))
            for c in contours
            if cv2.contourArea(c) > self.min_contour_area
        ]
        raw_boxes = [[x, y, x + w, y + h] for x, y, w, h in raw_boxes_xywh]

        if len(raw_boxes) < 1:
            return torch.empty((0, 4), device=self.device_input)

        if len(raw_boxes) > 1:
            raw_boxes = merge_boxes_cpu(raw_boxes, gap_limit=self.dist_thresh_640)

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

    # def contour2predictions(
    #     self,
    #     clip_filename,
    #     frameNum,  # Frame used in metadata
    #     mask,
    #     frame,
    #     device_input="cpu",
    #     return_bytes=True,
    #     return_prediction_count=False,
    # ):
    #     """
    #     The 'Glue' function that connects Motion Detection to AI Inference.

    #     Args:
    #         mask (GpuMat/np.ndarray): The motion mask (from BGS).
    #         frame (np.ndarray): The original 8K frame.
    #     """
    #     global all_metadata, video_ready_list
    #     clip_key = Path(self.clip_filename).name
    #     metadata = dict()
    #     # try:
    #     if device_input == "cpu":
    #         bbs_full_res = self.get_cpu_rois(frame, frameNum, mask)
    #         # Pass contours to the YOLO detection logic
    #         # metadata = dict()
    #         metadata, frame_bytes = self.get_detections_for_contours_bbs(
    #             frameNum,  # Frame used in metadata
    #             frame,
    #             bbs_full_res,
    #             thickness=self.config.THICKNESS,
    #             device_input=device_input,
    #             return_bytes=return_bytes,
    #         )
    #     else:
    #         # bbs_full_res = self.get_gpu_rois(frame, frameNum, mask)
    #         # HIGH-SPEED GPU PATH
    #         # Get raw 640p boxes from mask
    #         # raw_boxes_640p = self.findContours_gpu(mask, method="opencv")

    #         # # Scale to 8K and Merge with 640x640 constraint
    #         # # raw_boxes_640p * self.scales_tensor maps them to 8K coordinates
    #         # bbs_8k = self.gpu_merge_and_filter(
    #         #     raw_boxes_640p * self.scales_tensor,
    #         #     dist_thresh=self.dist_thresh_8k
    #         # )
    #         bbs_full_res = self.get_gpu_rois(frame, frameNum, mask)

    #         # Parallel Crop & Dynamic Batch Inference
    #         # 'frame_raw' is the GpuMat 8K frame uploaded in rbtd_full_gpu
    #         # metadata, annotated_gpu_mat = self.get_detections_parallel(
    #         #     frame, frameNum, mask
    #         # )

    #         # This ensures all rectangles and tiles are finished before the download starts
    #         # self.stream.waitForCompletion()

    #         # # Final single download of the annotated 8K frame
    #         # frame_cpu = frame.download()

    #         metadata, frame_bytes = self.get_detections_for_contours_bbs(
    #             frameNum,  # Frame used in metadata
    #             frame,  # frame_cpu,
    #             bbs_full_res,
    #             thickness=self.config.THICKNESS,
    #             device_input=device_input,
    #             return_bytes=return_bytes,
    #         )

    #         # This ensures all rectangles and tiles are finished before the download starts
    #         # self.stream.waitForCompletion()

    #         # # Final single download of the annotated 8K frame
    #         # annotated_frame_cpu = annotated_gpu_mat.download()

    #         # # Convert to display bytes for the FastAPI streamer
    #         # frame_bytes = get_display_frame_in_bytes(
    #         #     annotated_frame_cpu,
    #         #     display_size=(self.disp_w, self.disp_h),
    #         #     quality=self.config.DISPLAY_FRAME_QUALITY,
    #         # )

    #     # Write frame
    #     # self.video_writer.write(frame)
    #     # self.video_writer.write(self.cpu_resized_frame)

    #     # # Pass contours to the YOLO detection logic
    #     # # metadata = dict()
    #     # metadata, frame_bytes = self.get_detections_for_contours_bbs(
    #     #     frameNum,  # Frame used in metadata
    #     #     frame,
    #     #     bbs_full_res,
    #     #     thickness=self.config.THICKNESS,
    #     #     device_input=device_input,
    #     #     return_bytes=return_bytes,
    #     # )

    #     # Update global metadata for storage (Database/JSON)
    #     if metadata and self.config.ENABLE_QUERYING:
    #         all_metadata.setdefault(
    #             clip_key,
    #             {
    #                 "object": {},
    #                 "face": {},
    #             },
    #         )
    #         all_metadata[clip_key]["object"].update(metadata)
    #         # print(f"self.clip_key: {self.clip_key} objs: {list(metadata.keys())}")

    #     is_last_frame = (frameNum % self.max_frames_per_clip) == 0
    #     if is_last_frame and self.config.ENABLE_QUERYING:
    #         # Update the tracker that AI work for this clip is done
    #         check_and_dispatch_to_vdms(
    #             clip_filename, self.resize_w, self.resize_h, component="meta"
    #         )

    #     if return_prediction_count:
    #         return frame_bytes, len(metadata.keys())
    #     # frame_bytes returned even if no metadata available
    #     return frame_bytes
    #     # except Exception:
    #     #     e = traceback.format_exc()
    #     #     print(f"Error getting predictions: {e}", flush=True)
    #     #     return frame

    # def async_yolo_task(self, clip_filename, data, return_bytes=True):
    #     """
    #     Orchestrates the AI pipeline for a single frame.

    #     Steps:
    #     1. (if GPU) Synchronizes CUDA stream and downloads the mask.
    #     2. Executes contour-based YOLO detection.
    #     3. Handoffs the frame to the background JPEG encoder.
    #     """
    #     try:
    #         frameNum = data["frameNum"]  # overall frame

    #         # Use pre-allocated pinned memory from BaseHandler for 8K mask download
    #         # if self.device_input == "cuda":
    #         #     # self.stream.waitForCompletion()
    #         #     self.bgs_stream.waitForCompletion()
    #         #     # Ensure the download uses the instance-specific CUDA stream
    #         #     self.pinned_downloaded_frame_np = data["mask"].download(self.stream)
    #         #     # data["mask"].download(self.stream, self.pinned_downloaded_frame_np)

    #         # Run contour-based YOLO logic and draw overlays
    #         frame_bytes = self.contour2predictions(
    #             clip_filename,
    #             # Frame used in metadata; index 1
    #             ((frameNum - 1) % self.max_frames_per_clip) + 1,
    #             # self.pinned_downloaded_frame_np
    #             # if self.device_input == "cuda"
    #             # else data["mask"],
    #             data["mask"],
    #             data["full_frame"],
    #             device_input=self.device_input,
    #             return_bytes=return_bytes,
    #         )

    #         # Thread-safe update of the latest frame for FastAPI StreamingResponse
    #         if return_bytes and frameNum > self.last_delivered_frame_id:
    #             if frame_bytes:
    #                 self.latest_processed_frame = frame_bytes
    #                 self.last_delivered_frame_id = frameNum
    #             self.last_frame_id = frameNum
    #             self.last_heartbeat = time.time()
    #             # Signal the FastAPI generator that a new frame is ready
    #             self.loop.call_soon_threadsafe(self.frame_ready_event.set)

    #         # Offload JPEG encoding to a background worker to release the GIL
    #         # if not return_bytes:
    #         #     # Rotate between buffers so the encoder has a 'locked' memory space
    #         #     self.buf_idx = (self.buf_idx + 1) % 2
    #         #     target_buf = self.encode_buffers[self.buf_idx]

    #         #     # Perform a deep memory copy into the isolated buffer
    #         #     # np.copyto(target_buf, data["full_frame"])
    #         #     # np.copyto(target_buf, data["full_frame"], casting='unsafe')
    #         #     # target_buf[:] = data["full_frame"]
    #         #     target_buf[:] = np.ascontiguousarray(data["full_frame"])

    #         #     # Submit to background worker to bypass the GIL
    #         #     self.executor.submit(self._encode_and_signal, target_buf, frameNum)

    #     except Exception:
    #         logging.error(f"Async YOLO Error in {self.name}: {traceback.format_exc()}")

    # def merge_nearby_boxes(self, boxes, gap_limit=50):
    #     """
    #     Greedily merges boxes that are within gap_limit of each other.
    #     Input: List of [x1, y1, x2, y2]
    #     Output: List of merged [x1, y1, x2, y2]
    #     """
    #     if not boxes:
    #         return []

    #     # Sort by X-coordinate (O(N log N))
    #     # This is the secret to low latency—it prevents O(N^2) comparisons
    #     boxes = sorted(boxes, key=lambda x: x[0])
    #     merged = []

    #     while boxes:
    #         # Take the first box and try to grow it
    #         curr = boxes.pop(0)

    #         i = 0
    #         while i < len(boxes):
    #             test = boxes[i]

    #             # Optimization: Since we are sorted by X, if the X-gap is too large,
    #             # no remaining box in the list can possibly merge with 'curr'
    #             if test[0] - curr[2] > gap_limit:
    #                 break

    #             # Check Y-axis proximity
    #             # (Checks if the vertical gap between boxes is within the limit)
    #             y_dist = max(0, test[1] - curr[3], curr[1] - test[3])

    #             if y_dist <= gap_limit:
    #                 # Merge 'test' into 'curr'
    #                 curr[0] = min(curr[0], test[0])
    #                 curr[1] = min(curr[1], test[1])
    #                 curr[2] = max(curr[2], test[2])
    #                 curr[3] = max(curr[3], test[3])
    #                 # Remove from list and reset search to check new boundaries
    #                 boxes.pop(i)
    #                 i = 0
    #             else:
    #                 i += 1

    #         merged.append(curr)

    #     return merged

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

    def sf_cpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
        # num_objs = 0
        # metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Smart Filtering (Resize / Background Subtraction / Threshold / Dilate)
        # t_start = time.perf_counter()
        inf_data = self.rbtd_full_cpu(device_frame)
        # metrics["sf_time"] = (time.perf_counter() - t_start) * 1000

        # Only keep frames for TARGET_FPS
        # Videos are written for target frames only
        if is_target_frame:
            self.next_process_idx += self.step_size
            self.frame_count_target += 1  # 1-indexed

            # Immediate Rotation (Ensures logs match the new key instantly)
            if (
                self.config.ENABLE_QUERYING
                and self.frame_in_clip_count > self.max_frames_per_clip
            ):
                self.start_new_clip()

            metadata = {}
            bbs_to_send = []
            data_to_draw = []
            cpu_resized = None
            if inf_data:
                if self.config.ENABLE_QUERYING:
                    try:
                        # self.write_queue.put_nowait((frame_num, frame_to_queue))
                        self.write_queue.put(
                            # (frame_num, frame_to_queue)
                            {
                                "frame": self.resized_frame.clone(),
                                "writer_to_finalize": self.video_writer,
                            }
                        )
                    except queue.Full:
                        # This tells you that the writer is the bottleneck,
                        # but it won't stall your detection latency.
                        pass
                inf_data["frameNum"] = self.frame_count_target

                # Get
                # t_start = time.perf_counter()
                bbs_full_res = self.get_cpu_rois(
                    inf_data["full_frame"],
                    self.frame_count_target,
                    inf_data["mask"],
                )
                # metrics["roi_time"] = (time.perf_counter() - t_start) * 1000
                # metrics["bbs"] = bbs_full_res

                if self.config.DETECTION_TYPE == "motion":
                    display_source = (
                        inf_data["full_frame"]
                        if (inf_data and "full_frame" in inf_data)
                        else device_frame
                    )
                    cpu_resized = cv2.resize(
                        display_source,
                        (self.resize_w, self.resize_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # t_start = time.perf_counter()
                if self.config.DETECTION_TYPE == "motion":
                    # send bbnot self.config.TEST_MODE:
                    # disp_frame = tensor2opencv(cpu_resized, self.config.device_input)

                    # Resize for display and send to background worker
                    # disp_size = (self.disp_w, self.disp_h)
                    # disp_size = (self.resize_w, self.resize_h)
                    if (
                        bbs_full_res is not None
                        and bbs_full_res.ndim == 2
                        and bbs_full_res.size(0) > 0
                    ):
                        scaled_resized_bbs = bbs_full_res / self.scales_tensor
                        bbs_to_send = scaled_resized_bbs.cpu().tolist()
                    else:
                        bbs_to_send = []

                    # bbs_to_send = scaled_resized_bbs.cpu().tolist() if torch.is_tensor(scaled_resized_bbs) else scaled_resized_bbs
                    # self.render_queue.put((disp_frame, bbs_to_send, self.label_source))
                    data_to_draw = bbs_to_send
                    # num_objs = len(bbs_full_res)
                else:
                    # Get detection for BBs at original resolution
                    metadata, frame_bytes = self.get_detections(
                        inf_data["full_frame"] if inf_data else device_frame,
                        self.frame_count_target,  # Frame used in metadata
                        merged=bbs_full_res,
                        thickness=self.config.THICKNESS,
                        device_input=self.config.device_input,
                    )
                    data_to_draw = metadata
                    # num_objs = len(metadata.keys())

                # metrics["det_time"] = (time.perf_counter() - t_start) * 1000

            # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
            if not self.config.TEST_MODE:
                display_source = (
                    inf_data["full_frame"]
                    if (inf_data and "full_frame" in inf_data)
                    else device_frame
                )
                cpu_resized = cv2.resize(display_source, (self.resize_w, self.resize_h))
                disp_frame = tensor2opencv(
                    cpu_resized, self.config.device_input, is_bgr=True
                )
                torch.cuda.current_stream().synchronize()
                self.render_queue.put(
                    (disp_frame, inf_data["frameNum"], data_to_draw, self.label_source)
                )

        # return num_objs, metrics

    def sf_gpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
        # num_objs = 0
        # metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # # Smart Filtering (Resize / Background Subtraction / Threshold / Dilate)
        # sf_start, sf_end = (
        #     torch.cuda.Event(enable_timing=True),
        #     torch.cuda.Event(enable_timing=True),
        # )
        # sf_start.record()
        inf_data = self.rbtd_full_gpu(device_frame)
        # sf_end.record()
        # self.gpu_event_buffer["sf"].append((sf_start, sf_end))

        # Only keep frames for TARGET_FPS
        # Videos are written for target frames only
        if is_target_frame:
            self.next_process_idx += self.step_size
            self.frame_count_target += 1  # 1-indexed

            # Immediate Rotation (Ensures logs match the new key instantly)
            if (
                self.config.ENABLE_QUERYING
                and self.frame_in_clip_count > self.max_frames_per_clip
            ):
                self.start_new_clip()

            metadata = {}
            bbs_to_send = []
            data_to_draw = []
            if inf_data:
                inf_data["frameNum"] = self.frame_count_target

                if self.config.ENABLE_QUERYING:
                    try:
                        # self.write_queue.put_nowait((frame_num, frame_to_queue))
                        self.write_queue.put(
                            # (frame_num, frame_to_queue)
                            {
                                "frame": self.resized_frame.clone(),
                                "writer_to_finalize": self.video_writer,
                            }
                        )
                    except queue.Full:
                        # This tells you that the writer is the bottleneck,
                        # but it won't stall your detection latency.
                        pass

                # Get ROIs - time.perf_counter accurate since called self.bgs_stream.waitForCompletion()
                # t_start = time.perf_counter()
                bbs_full_res = self.get_gpu_rois(
                    inf_data["full_frame"],
                    self.frame_count_target,
                    inf_data["mask"],
                )
                # metrics["roi_time"] = (time.perf_counter() - t_start) * 1000
                # # self.gpu_event_buffer["roi"].append((roi_start, roi_end))
                # metrics["bbs"] = bbs_full_res

                # if self.config.DEBUG_FLAG:
                #     # self.resized_frame.download(self.bgs_stream, self.pinned_downloaded_resizedframe_np)
                #     self.bgs_stream.waitForCompletion()
                #     display_source = inf_data["mask"]
                #     self.debug_save_mask(
                #         display_source, self.frame_count_target, rois=bbs_full_res
                #     )

                # if self.config.DEBUG_FLAG:
                #     display_source = (
                #         inf_data["full_frame"]
                #         if (inf_data and "full_frame" in inf_data)
                #         else device_frame
                #     )
                #     gpu_resized = F.interpolate(
                #         display_source.unsqueeze(0).float(),
                #         size=(self.resize_h, self.resize_w),
                #         mode="bilinear",
                #         align_corners=False,
                #     )
                #     self.debug_save_img_roi(
                #         gpu_resized, bbs_full_res, self.frame_count_target
                #     )

                # det_start, det_end = (
                #     torch.cuda.Event(enable_timing=True),
                #     torch.cuda.Event(enable_timing=True),
                # )
                # det_start.record()
                if self.config.DETECTION_TYPE == "motion":
                    # Motion Mode: Prepare boxes for drawing
                    if (
                        bbs_full_res is not None
                        and bbs_full_res.ndim == 2
                        and bbs_full_res.size(0) > 0
                    ):
                        scaled_resized_bbs = bbs_full_res / self.scales_tensor
                        bbs_to_send = scaled_resized_bbs.cpu().tolist()
                    else:
                        bbs_to_send = []
                    data_to_draw = bbs_to_send
                    # num_objs = len(bbs_to_send)
                else:
                    # Object Mode: Run YOLO and prepare metadata
                    det_frame = (
                        inf_data["full_frame"] if inf_data else device_frame
                    )  # RGB
                    metadata, _ = self.get_detections(
                        det_frame,
                        self.frame_count_target,
                        merged=bbs_full_res,
                        thickness=self.config.THICKNESS,
                        device_input=self.config.device_input,
                    )
                    data_to_draw = metadata
                    # num_objs = len(metadata.keys())

                # det_end.record()
                # self.gpu_event_buffer["det"].append((det_start, det_end))

            # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
            if not self.config.TEST_MODE:
                # print(f"Sending to queue", flush=True)
                display_source = (
                    inf_data["full_frame"]
                    if (inf_data and "full_frame" in inf_data)
                    else device_frame
                )
                gpu_resized = F.interpolate(
                    display_source.unsqueeze(0).float(),
                    size=(self.disp_h, self.disp_w),  # (self.resize_h, self.resize_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # RGB
                disp_frame = tensor2opencv(
                    gpu_resized, self.config.device_input, is_bgr=True
                )
                # disp_frame = self.letterbox_gpu(display_source, target_size=self.resize_h)

                # self.inference_stream.synchronize()
                # if hasattr(self, 'bgs_stream'):
                #     self.bgs_stream.waitForCompletion()
                # current_data = data_to_draw if is_target_frame else ({} if isinstance(data_to_draw, dict) else [])
                # self.render_queue.put(
                # try:
                #     self.render_queue.put_nowait(
                #         (disp_frame, inf_data["frameNum"], data_to_draw, self.label_source)
                #     )
                # except queue.Full:
                #     pass
                self.render_queue.put(
                    (disp_frame, inf_data["frameNum"], data_to_draw, self.label_source)
                )

            # Async metrics stay at 0 for fairness; collected at the end of the video
        #     metrics["sf_time"] = 0
        #     metrics["det_time"] = 0
        # return num_objs, metrics

    def yolo_cpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
        # num_objs = 0
        metadata = {}
        # metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Only keep frames for TARGET_FPS
        if is_target_frame:
            self.next_process_idx += self.step_size
            self.frame_count_target += 1  # 1-indexed
            frameNum = self.frame_count_target

            # Immediate Rotation (Ensures logs match the new key instantly)
            if (
                self.config.ENABLE_QUERYING
                and self.frame_in_clip_count > self.max_frames_per_clip
            ):
                self.start_new_clip()

            if self.config.ENABLE_QUERYING:
                try:
                    cpu_resized = cv2.resize(
                        device_frame, (self.resize_w, self.resize_h)
                    )
                    disp_frame = tensor2opencv(
                        cpu_resized, self.config.device_input, is_bgr=True
                    )

                    # self.write_queue.put_nowait((frame_num, frame_to_queue))
                    self.write_queue.put(
                        # (frame_num, frame_to_queue)
                        {
                            "frame": disp_frame,
                            "writer_to_finalize": self.video_writer,
                        }
                    )
                except queue.Full:
                    # This tells you that the writer is the bottleneck,
                    # but it won't stall your detection latency.
                    pass

            # Get detection at original resolution (No SF)
            # t_start = time.perf_counter()
            metadata, _ = self.get_detections(
                device_frame,
                self.frame_count_target,
                thickness=self.config.THICKNESS,
                device_input=self.config.device_input,
            )
            # metrics["det_time"] = (time.perf_counter() - t_start) * 1000
            # num_objs = len(metadata.keys())

            if not self.config.TEST_MODE:
                cpu_resized = cv2.resize(device_frame, (self.resize_w, self.resize_h))
                disp_frame = tensor2opencv(
                    cpu_resized, self.config.device_input, is_bgr=True
                )
                self.render_queue.put(
                    (disp_frame, frameNum, metadata, self.label_source)
                )

        # return num_objs, metrics

    def yolo_gpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
        # num_objs = 0
        metadata = {}
        # metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Only keep frames for TARGET_FPS
        if is_target_frame:
            self.next_process_idx += self.step_size
            self.frame_count_target += 1  # 1-indexed
            frameNum = self.frame_count_target

            # Immediate Rotation (Ensures logs match the new key instantly)
            if (
                self.config.ENABLE_QUERYING
                and self.frame_in_clip_count > self.max_frames_per_clip
            ):
                self.start_new_clip()

            if self.config.ENABLE_QUERYING:
                try:
                    gpu_resized = F.interpolate(
                        device_frame.unsqueeze(0).float(),
                        size=(self.resize_h, self.resize_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    disp_frame = tensor2opencv(
                        gpu_resized, self.config.device_input, is_bgr=True
                    )
                    # self.write_queue.put_nowait((frame_num, frame_to_queue))
                    self.write_queue.put(
                        # (frame_num, frame_to_queue)
                        {
                            "frame": disp_frame,
                            "writer_to_finalize": self.video_writer,
                        }
                    )
                except queue.Full:
                    # This tells you that the writer is the bottleneck,
                    # but it won't stall your detection latency.
                    pass

            # Get detection at original resolution (No SF)
            # det_start, det_end = (
            #     torch.cuda.Event(enable_timing=True),
            #     torch.cuda.Event(enable_timing=True),
            # )
            # det_start.record()
            metadata, _ = self.get_detections(
                device_frame,
                self.frame_count_target,
                thickness=self.config.THICKNESS,
                device_input=self.config.device_input,
            )
            # num_objs = len(metadata.keys())
            # det_end.record()
            # self.gpu_event_buffer["det"].append((det_start, det_end))

            # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
            if not self.config.TEST_MODE:
                gpu_resized = F.interpolate(
                    device_frame.unsqueeze(0).float(),
                    size=(self.resize_h, self.resize_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                disp_frame = tensor2opencv(
                    gpu_resized, self.config.device_input, is_bgr=True
                )
                self.render_queue.put(
                    (disp_frame, frameNum, metadata, self.label_source)
                )

            # Async metrics stay at 0 for fairness; collected at the end of the video
        #     metrics["sf_time"] = 0
        #     metrics["det_time"] = 0
        # return num_objs, metrics


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

    def apply_background_subtraction_gpu(
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

    # def run_realtime_inference(self):
    #     """
    #     Main execution loop for the filtering test.
    #     Tracks performance metrics including processing time and average FPS.
    #     """
    #     # print(f"Inference thread started for {self.name}...\n", flush=True)

    #     # Initialize timing statistics
    #     # last_frame_time = time.perf_counter()
    #     # self.frame_count = 0
    #     self.frame_count_target = 0

    #     while self.active:  # or not self.reader.frame_queue.empty():
    #         nv12_gpu_tensor, frame_num = self.reader.read()

    #         # Get current depth to monitor the backlog
    #         # q_depth = self.reader.frame_queue.qsize()

    #         if nv12_gpu_tensor is not None:
    #             target_buf = self.frame_buffer_pool[self.pool_idx]
    #             self.pool_idx = (self.pool_idx + 1) % 2
    #             # Use float comparison to ensure exactly 15fps output
    #             # if frame_num >= self.next_process_idx:
    #             #     # Update the pointer for the next frame to catch
    #             #     self.next_process_idx += self.step_size
    #             # self.frame_count += 1

    #             # print(
    #             #     f"Processing frame {self.frame_count} | Queue Depth: {q_depth}\n",
    #             #     flush=True,
    #             # )

    #             # Tells the driver not to swap this memory, allowing 2x faster GPU upload
    #             # cv2.cuda.registerPageLocked(frame)

    #             frame = nv12_to_rgb_torch(
    #                 nv12_gpu_tensor,
    #                 self.frame_height,
    #                 self.frame_width,
    #                 is_h264_8k=self.reader.is_h264_8k,
    #                 out_buffer=target_buf,
    #             )

    #             # Clean up the source immediately
    #             del nv12_gpu_tensor

    #             # if not self.reader.is_h264_8k:
    #             #     frame = nv12_to_rgb_torch(
    #             #         nv12_gpu_tensor, self.frame_height, self.frame_width
    #             #     )
    #             # else:
    #             #     frame = nv12_gpu_tensor

    #             # Calculate a dynamic limit: tolerate 0.5 seconds of lag.
    #             # If target_fps is 15, the limit is 7. If target_fps is 30, the limit is 15.
    #             self.dynamic_limit = max(2, int(0.5 * self.target_fps))

    #             # Determine if this frame should be AI or Raw based on backlog
    #             # But ALWAYS submit to the executor to maintain frame order.
    #             backlog = self.get_executor_backlog()

    #             # Use a "Skip AI" flag instead of a "continue" skip
    #             skip_ai = backlog > self.dynamic_limit

    #             # self.frame_count += 1
    #             self.frame_in_clip_count += 1

    #             # Immediate Rotation (Ensures logs match the new key instantly)
    #             if (
    #                 self.config.ENABLE_QUERYING
    #                 and self.frame_in_clip_count > self.max_frames_per_clip
    #             ):
    #                 # Perform safety check BEFORE starting the next 185MB clip
    #                 # self._check_shm_safety(threshold_percent=90)

    #                 # Pass old state to consumer before updating
    #                 old_writer = self.video_writer
    #                 old_key, old_tmp, old_final, old_frame_in_clip_count = (
    #                     self.clip_key,
    #                     self.tmp_file,
    #                     self.clip_filename,
    #                     self.frame_in_clip_count,
    #                 )

    #                 self.write_queue.put(
    #                     {
    #                         "control": "ROTATE",
    #                         "old_key": old_key,
    #                         "old_tmp": old_tmp,
    #                         "old_final": old_final,
    #                         "frame_in_clip_count": old_frame_in_clip_count,
    #                         "writer_to_finalize": old_writer,
    #                     }
    #                 )

    #                 self.clip_id += 1
    #                 self.frame_in_clip_count = 1
    #                 # self.video_writer = None # Nullify on main thread
    #                 self._initialize_writer()
    #                 # self.clip_id += 1
    #                 # self.frame_in_clip_count = 1
    #                 # self.video_writer = None # Nullify on main thread
    #                 # self._initialize_writer()
    #                 self._check_shm_safety(threshold_percent=90)

    #             # self.process_frame_async(frame, self.frame_count)
    #             self.executor.submit(
    #                 self.process_frame_async,
    #                 self.clip_filename,
    #                 frame,
    #                 frame_num,
    #                 skip_ai,
    #             )

    #             # self.frame_count += 1
    #             self.update_frame()
    #             self.last_heartbeat = time.time()
    #             self.frame_count = frame_num
    #             if self.frame_count % 100 == 0:
    #                 torch.cuda.empty_cache()  # Periodically flush "reserved" memory
    #         elif self.is_processing():  # not self.write_queue.empty():
    #             continue
    #         elif self.reader.stopped:
    #             # if self.write_count >= self.frame_count:
    #             #     print(
    #             #         f"Sending final signal to writer ({self.write_count}/{self.frame_count})"
    #             #     )
    #             #     self.write_queue.put((None, None))
    #             #     break
    #             self.active = False
    #             break

    #     # self.total_time = time.perf_counter() - start_time
    #     self.stop()

    def rbtd_full_gpu(self, frame):
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

        # self.bgs_stream.waitForCompletion()
        # Clean up the motion mask using Thresholding and Morphology (Dilation)
        # cv2.cuda.threshold(
        #     self.fgMask,
        #     self.config.THRESHOLD_VALUE,
        #     255,
        #     cv2.THRESH_BINARY,
        #     dst=self.gpu_threshold_dst_frame,
        #     stream=self.bgs_stream,
        # )
        # self.dilate_filter.apply(
        #     self.gpu_threshold_dst_frame, dst=self.labels_gpu, stream=self.bgs_stream
        # )

        # return {
        #     # "frameNum": frameNum,  # overall frame
        #     "mask": self.labels_gpu,  # GpuMat pointer to cleaned mask
        #     # "full_frame": frame,  # Kept for high-res cropping
        #     "full_frame": frame,  # current_gpu_frame,
        # }

        # if self.config.ENABLE_QUERYING and self.video_writer:  # and not self.video_queue.full():
        #     self.pinned_downloaded_resizedframe_np = self.resized_frame.download(stream)
        #     # self.resized_frame.download(self.stream, self.pinned_downloaded_resizedframe_np)
        #     #     self.video_writer.write(self.pinned_downloaded_resizedframe_np)
        #     self.write_queue.put(self.pinned_downloaded_resizedframe_np.copy())

        # if (
        #     (self.debug_range[0] * self.target_fps)
        #     <= self.frame_count
        #     <= (self.debug_range[1] * self.target_fps)
        # ):
        #     mask_cpu = self.fgMask.download()  # fgMask is a GpuMat from the backend
        #     mask_path = f"{self.out_imgdir}/mask_frame_{self.frame_count}.png"
        #     cv2.imwrite(mask_path, mask_cpu)
        #     print(f"[DEBUG] Saved BGS Mask: {mask_path}")

        # --------------
        # NOISE FILTERING: Median Blur (Kills pixel noise)
        # if not hasattr(self, 'median_filter'):
        #     self.median_filter = cv2.cuda.createMedianFilter(cv2.CV_8UC1, 5)

        # # Syntax: apply(src, dst, stream)
        # self.median_filter.apply(self.fgMask, self.fgMask, self.bgs_stream)

        # # MORPHOLOGY: Erode then Dilate (Opening)
        # if not hasattr(self, 'erode_filter'):
        #     # 3x3 kernel is sufficient when combined with a median filter
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        #     # In OpenCV 4.x, use createMorphologyFilter for both Erode and Dilate
        #     self.erode_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, kernel)
        #     self.dilate_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel)

        # # Shave off noise (Erode)
        # self.erode_filter.apply(self.fgMask, self.fgMask, self.bgs_stream)

        # # Restore object size (Dilate)
        # self.dilate_filter.apply(self.fgMask, self.fgMask, self.bgs_stream)

        # return {"mask": self.fgMask, "full_frame": frame}
        # --------------

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

    def rbtd_full_cpu(self, frame):
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
