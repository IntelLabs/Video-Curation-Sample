import asyncio
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys

# Force FFmpeg to use more threads for decoding
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from fastapi import FastAPI

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|hwaccel;cuda|threads;2|probesize;32|analyzeduration;0"
)


# from process_stream import extract_metadata_from_results, release_clip_and_reencode, retry_query
from include.utils import (
    CODE_DIR,
    CUSTOM_MODEL_FLAG,
    DEBUG,
    DEBUG_FLAG,
    DETECTION_THRESHOLD,
    DEVICE,
    MODEL_H,
    MODEL_NAME,
    MODEL_PRECISION,
    MODEL_W,
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
CLIP_DURATION = 10  # seconds
KERNEL_RATIO = 0.05  # 0.03 # .05  # .025
MASK_MAX_VALUE = 255
MASK_THRESHOLD_VALUE = 127
MAX_DETECTIONS = 100
MAX_WORKERS = 4
# DISPLAY_FRAME_SIZE = (1280, 720)
# DISPLAY_FRAME_SIZE = (640, 360)
# DISPLAY_FRAME_SIZE = (854, 480)
DISPLAY_FRAME_SIZE = (960, 540)
DISPLAY_FRAME_QUALITY = 50
ENABLE_QUERYING = False

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
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
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
        # Iterating over a list of keys to avoid "dictionary changed size" error
        for name in list(app.state.active_streams.keys()):
            streamer = app.state.active_streams[name]

            # Check if the stream is marked inactive OR timed out
            # streamer.active should be False when the video source ends
            if not streamer.active or (now - streamer.last_heartbeat > 30):
                if DEBUG == "1":
                    print(f"CLEANUP: Removing {name} from active_streams")
                streamer.stop()
                del app.state.active_streams[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This is the ONLY place this should be initialized
    if not hasattr(app.state, "active_streams"):
        app.state.active_streams = {}
        app.state.status = "Ready"
        # app.state.model = YOLO(model_path, verbose=False, task="detect")
        # app.state.model_lock = threading.Lock()
    asyncio.create_task(auto_cleanup_janitor(app))
    if DEBUG == "1":
        print(f"--- APP STARTUP | PID: {os.getpid()} | STATE READY ---")
    yield
    # Cleanup logic here...
    for s in app.state.active_streams.values():
        s.stop()
    app.state.status = "Stopped"


class VideoStreamHandler:
    def __init__(self, source, name, active_streams):
        self.model = YOLO(model_path, verbose=False, task="detect")
        self.name = name
        self.source = source
        self.active = True
        self.active_streams = active_streams

        # 1. Capture setup
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        self.get_fps_and_framecnt()
        self.get_frameWH()

        # 2. Performance Tracking
        self.stat_start_time = time.perf_counter()
        self.stat_frame_count = 0
        self.stat_fps = 0
        self.latest_processed_frame = None
        self.last_heartbeat = time.time()
        self.last_frame_id = 0

        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1, mp4v
        self.clip_id = 0
        self.clip_filename = ""
        self.clip_key = ""
        self.tmp_file = ""

        self.resize_h, self.resize_w = [MODEL_H, MODEL_W]
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.numFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.scale_x = self.frame_width / MODEL_W
        self.scale_y = self.frame_height / MODEL_H
        self.min_contour_area = int(
            (0.005 * self.frame_width) * (0.005 * self.frame_height)
        )  # 207

        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dilate_kernel_for_enhanced_mask = np.ones((21, 21), np.uint8)

        # Device based setup
        if DEVICE == "GPU":
            self.prepare_gpu_pipeline()
            self.warmup()
        else:
            self.operation_device_map = PipelineMapping(
                detection_device="cpu"
            )  # No CUDA HERE
            self.prepare_cpu_pipeline()

        # 3. Start dedicated inference thread
        self.model_warmup()
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.process_thread = threading.Thread(
            target=self.run_realtime_inference, daemon=True
        )
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

    def allocate_gpu(self, bkgd_mask_queue_size=3):
        self.resized_frame = cv2.cuda_GpuMat(self.resize_h, self.resize_w, cv2.CV_8UC3)

        self.stream = cv2.cuda.Stream()

        self.gpu_fullres_frame = cv2.cuda_GpuMat(
            self.frame_height, self.frame_width, cv2.CV_8UC3
        )

        self.pinned_downloaded_resizedframe_np = cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC3
        )

        self.fgMask = cv2.cuda_GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )  # For resize

        self.prev_bkgd = cv2.cuda_GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )  # For resize
        self.prev_bkgd.setTo((255,))

        self.mask_history = deque(maxlen=bkgd_mask_queue_size)
        self.mask_history.append(self.prev_bkgd)

        self.gpu_threshold_dst_frame = cv2.cuda_GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )
        cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1, self.gpu_threshold_dst_frame
        )

        self.gpu_morphed_frame = cv2.cuda_GpuMat(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )  # For resize
        cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1, self.gpu_morphed_frame
        )

        self.pinned_downloaded_frame_np = cv2.cuda.createContinuous(
            self.resize_h, self.resize_w, cv2.CV_8UC1
        )

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

    def warmup(self):
        # WARM UP (Crucial for first-run latency)
        # JIT kernels are compiled on the first call
        self.gpu_warmup_frame = cv2.cuda_GpuMat(
            self.frame_height, self.frame_width, cv2.CV_8U
        )
        self.gpu_warmup_input_frame = cv2.cuda_GpuMat(
            self.frame_height, self.frame_width, cv2.CV_8U
        )
        self.gpu_warmup_input_frame_np = cv2.cuda.createContinuous(
            self.frame_height, self.frame_width, cv2.CV_8UC3
        )
        self.gpu_warmup_input_frame_np[:] = [255, 0, 0]
        cv2.cuda.createContinuous(
            self.frame_height, self.frame_width, cv2.CV_8U, self.gpu_warmup_frame
        )
        cv2.cuda.createContinuous(
            self.frame_height, self.frame_width, cv2.CV_8U, self.gpu_warmup_input_frame
        )

        self.gpu_warmup_input_frame.upload(self.gpu_warmup_input_frame_np)
        cv2.cuda.cvtColor(
            self.gpu_warmup_input_frame,
            cv2.COLOR_BGR2GRAY,
            stream=self.stream,
            dst=self.gpu_warmup_frame,
        )
        self.stream.waitForCompletion()

    def model_warmup(self):
        print("Starting warmup...")
        dummy_input = torch.zeros((1, 3, self.resize_h, self.resize_w)).to(
            self.device_input
        )  # Match your benchmark size
        for _ in range(20):
            _ = self.model(dummy_input, verbose=False)

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

    def run_pipeline(self, frame, frameNume, repeat_count=1):
        # if self.device_input == "cpu":
        #     return self.test_full_cpu_detection_gpu(
        #         frame, frameNume, repeat_count=repeat_count
        #     )
        # else:
        return self.test_rbtd_detection_gpu(frame, frameNume, repeat_count=repeat_count)

    def async_yolo_task(self, data):
        """Heavy lifting moved to ThreadPoolExecutor"""
        try:
            if self.device_input == "cuda":
                self.pinned_downloaded_frame_np = data["mask"].download(self.stream)
                frame_bytes = self.contour2predictions(
                    data["frameNum"],
                    self.pinned_downloaded_frame_np,
                    data["full_frame"],
                    device_input=self.device_input,
                    repeat_count=data["repeat_count"],
                )
            else:
                frame_bytes = self.contour2predictions(
                    data["frameNum"],
                    data["mask"],
                    data["full_frame"],
                    device_input=self.device_input,
                    repeat_count=data["repeat_count"],
                )
            self.latest_processed_frame = frame_bytes
            self.last_heartbeat = time.time()
            self.last_frame_id += 1
        except Exception:
            e = traceback.format_exc()
            print(f"Async YOLO Error: {e}")

    def process_frame_async(self, frame, frame_num):
        """
        Worker function to run heavy AI tasks (Resize, Bkgd Sub, YOLO)
        in the background without blocking the video reader.
        """
        try:
            # Calls your existing Page 22 logic (run_pipeline)
            inf_data = self.run_pipeline(frame, frame_num + 1)

            if inf_data:
                # Calls your Page 20 async_yolo_task to handle mask download/inference
                self.async_yolo_task(inf_data)

        except Exception:
            e = traceback.format_exc()
            print(f"ERROR: process_frame_async failed for {self.name}: {e}")

    def run_realtime_inference(self):
        """
        Main loop: Initializes the model in this thread to fix CUDA context issues.
        """
        print(f"Inference thread started for {self.name}...")

        # --- CRITICAL: Initialize model INSIDE the thread ---
        # This binds the GPU context to this thread specifically.
        # import torch
        # self.model = YOLO(model_path, verbose=False, task="detect")
        # self.model.to('cuda') # Explicitly move to GPU in this thread

        target_interval = 1.0 / self.target_fps
        last_process_time = time.time()

        while self.active:
            # 1. REAL-TIME SYNC: Clear stale frames from buffer
            # while True:
            grabbed = self.cap.grab()
            if not grabbed:
                self.active = False
                break

            now = time.time()
            if now - last_process_time < target_interval:
                continue

            success, frame = self.cap.retrieve()
            if not success or frame is None:
                continue

            last_process_time = now

            # 3. DECOUPLED AI: Only submit to AI if the worker queue is not backed up
            # This prevents 'lag' if the AI is slower than the video feed
            if self.get_executor_backlog() < MAX_WORKERS:
                # Move the heavy 'run_pipeline' call into a background worker
                self.executor.submit(
                    self.process_frame_async, frame.copy(), self.stat_frame_count
                )
            else:
                # If AI is busy, still update the display with the raw frame
                # so the dashboard video stays smooth and fluid
                _, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, DISPLAY_FRAME_QUALITY]
                )
                self.latest_processed_frame = buffer.tobytes()
                self.last_frame_id += 1  # Ensure the generator sees this 'clean' frame

            self.update_frame()
            self.last_heartbeat = time.time()

        self.stop()
        # Add this line to remove it from the dashboard immediately:
        if self.name in self.active_streams:  # noqa: F821
            del self.active_streams[self.name]  # noqa: F821

    def test_rbtd_detection_gpu(self, frame, frameNum, repeat_count=1):
        # Resize directly into the pre-allocated Pinned Memory
        # This avoids a temporary CPU allocation
        H, W = self.resize_h, self.resize_w
        # self.cpu_resized_frame = cv2.resize(frame, (W, H))
        # self.video_writer.write(self.cpu_resized_frame)
        self.gpu_fullres_frame.upload(frame, self.stream)
        cv2.cuda.resize(
            self.gpu_fullres_frame,
            (W, H),
            stream=self.stream,
            dst=self.resized_frame,
            interpolation=cv2.INTER_NEAREST,
        )
        if ENABLE_QUERYING and self.video_writer:  # and not self.video_queue.full():
            self.pinned_downloaded_resizedframe_np = self.resized_frame.download(
                self.stream
            )
            # self.resized_frame.download(self.stream, self.pinned_downloaded_resizedframe_np)
            for _ in range(repeat_count):
                # self.video_queue.put((self.video_writer, self.pinned_downloaded_resizedframe_np.copy()))
                self.video_writer.write(self.pinned_downloaded_resizedframe_np)

        # Background Subtraction on GPU
        self.fgMask = self.backSub.apply(
            self.resized_frame, float(self.lr), stream=self.stream
        )

        for m in list(self.mask_history):
            # Dilate the historical mask on GPU
            dilated = self.dilate_filter_for_enhanced_mask.apply(m)
            # Bitwise AND on GPU
            cv2.cuda.bitwise_and(self.prev_bkgd, dilated, self.prev_bkgd)
            # dilated = cv2.dilate(m, self.dilate_kernel_for_enhanced_mask, iterations=1)
            # cv2.bitwise_and(prev_bkgd, dilated, dst=prev_bkgd)
        self.mask_history.append(self.fgMask.clone())
        min_val, max_val, _, _ = cv2.cuda.minMaxLoc(self.prev_bkgd)

        if max_val != min_val:
            self.fgMask = cv2.cuda.bitwise_or(self.fgMask, self.prev_bkgd)

        # Thresholding
        cv2.cuda.threshold(
            self.fgMask,
            MASK_THRESHOLD_VALUE,
            MASK_MAX_VALUE,
            cv2.THRESH_BINARY,
            self.gpu_threshold_dst_frame,
            self.stream,
        )

        # mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)
        self.dilate_filter.apply(
            self.gpu_threshold_dst_frame, self.gpu_morphed_frame, self.stream
        )

        return {
            "frameNum": frameNum,
            "mask": self.gpu_morphed_frame,
            "full_frame": frame,  # Original for cropping
            "repeat_count": repeat_count,
        }

    def test_full_cpu_detection_gpu(self, frame, frameNum, repeat_count=1):
        # Resize directly into the pre-allocated Pinned Memory
        # This avoids a temporary CPU allocation
        H, W = self.resize_h, self.resize_w
        self.cpu_resized_frame = cv2.resize(
            frame, (W, H), interpolation=cv2.INTER_NEAREST
        )
        if ENABLE_QUERYING:
            for _ in range(repeat_count):
                self.video_writer.write(self.cpu_resized_frame)

        # Background Subtraction on CPU
        fgMask = self.backSub.apply(self.cpu_resized_frame, learningRate=self.lr)

        prev_bkgd = np.ones_like(fgMask)  # AND
        for m in self.mask_history:
            # Dilate the historical mask
            dilated = cv2.dilate(m, self.dilate_kernel_for_enhanced_mask, iterations=1)
            cv2.bitwise_and(prev_bkgd, dilated, dst=prev_bkgd)
        self.mask_history.append(fgMask)

        if prev_bkgd.max() != prev_bkgd.min():
            combined_mask_bool = (fgMask > 0) | (prev_bkgd > 0)

            # Convert the boolean array back to uint8 with 0 and 255 values
            fgMask = combined_mask_bool.astype(np.uint8) * 255

        # Thresholding
        _, mask = cv2.threshold(
            fgMask, MASK_THRESHOLD_VALUE, MASK_MAX_VALUE, cv2.THRESH_BINARY
        )

        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)

        return {
            "frameNum": frameNum,
            "mask": mask,
            "full_frame": frame,  # Original for cropping
            "repeat_count": repeat_count,
        }

    def update_frame(self):
        self.stat_frame_count += 1
        elapsed = time.perf_counter() - self.stat_start_time
        if elapsed > 1.0:
            self.stat_fps = self.stat_frame_count / elapsed

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
    def get_fps_and_framecnt(self):
        self.input_fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # hardware fps
        if self.input_fps == 0:  # Case when FPS isn't available
            self.input_fps = manual_fps_calculation(self.name, num_frames=10)

        self.target_fps = TARGET_FPS if self.input_fps > TARGET_FPS else self.input_fps
        self.frame_skip = int(self.input_fps / self.target_fps)
        if self.frame_skip < 1:
            self.frame_skip = 1

        self.MAX_FRAMES_PER_CLIP = int(self.target_fps * CLIP_DURATION)
        self.target_interval = 1.0 / self.target_fps  # 0.0666s

        if DEBUG == "1":
            print(f"FPS of {self.name} input stream: {self.input_fps}", flush=True)
            print(f"FPS of {self.name} output mp4: {self.target_fps}", flush=True)

        # Frame count for videos
        self.frame_count = None
        if "://" not in str(self.source):
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    def get_detections_for_contours_bbs(
        self, frameNum, foi, contours, thickness=2, device_input="cuda"
    ):
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

        # Filter and Sort in one go (Minimize Python-to-C++ crossings)
        raw_bbs = []
        padding = 64
        for c in contours:
            area = cv2.contourArea(c)
            x1, y1, w, h = cv2.boundingRect(c)
            if (
                area > self.min_contour_area
            ):  # and area / (w*h) >=0.3:  # and 0.5 < (w / h) < 2.0: # w/ solidity & aspect
                xx1 = max(0, int((x1 * self.scale_x)) - padding)
                yy1 = max(0, int((y1 * self.scale_y)) - padding)
                xx2 = min(W, int(((x1 + w) * self.scale_x)) + padding)
                yy2 = min(H, int(((y1 + h) * self.scale_y)) + padding)
                raw_bbs.append([area, [xx1, yy1, xx2, yy2]])
        bbs_full_res = sorted(
            [pair[1] for pair in raw_bbs if pair[0] > self.min_contour_area],
            key=lambda x: x[0],
            reverse=True,
        )[:MAX_DETECTIONS]

        dist_thresh = min(0.05 * W, 0.05 * H)
        merged = merge_boxes_limit(
            bbs_full_res, dist_threshold=dist_thresh, size_limit=640
        )

        merged = filter_contained_boxes(merged, containment_thresh=0.9)

        # for cnt, area in merged:
        for x1, y1, x2, y2 in merged:
            if (
                x2 > x1
                and y2 > y1
                and (x2 - x1) < self.frame_width
                and (y2 - y1) < self.frame_height
            ):
                crop = foi[y1:y2, x1:x2]
                if crop.size > 0:
                    cropped_imgs.append(crop)
                    cropped_coords.append((x1, y1))

        if not cropped_imgs:
            frame_bytes = get_display_frame_in_bytes(
                foi,
                self.frame_width,
                display_size=DISPLAY_FRAME_SIZE,
                quality=DISPLAY_FRAME_QUALITY,
            )
            return metadata, frame_bytes  # num_objs, predictions

        # 2. Inference (Keep stream=False as it is stable)
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

        label_source = (
            self.model.names if hasattr(self.model, "names") else YOLO_CLASS_NAMES
        )

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
                class_name = label_source[class_id]
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

        # Queue frame for display (reduce quality slightly to 80 for 8K bandwidth)
        frame_bytes = get_display_frame_in_bytes(
            foi,
            self.frame_width,
            display_size=DISPLAY_FRAME_SIZE,
            quality=DISPLAY_FRAME_QUALITY,
        )

        return metadata, frame_bytes

    def contour2predictions(
        self, frameNum, mask, frame, device_input="cpu", repeat_count=1
    ):
        # source = self.source
        # stream_name = self.name
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Write frame
        # self.video_writer.write(frame)
        # if self.video_writer:
        #     for _ in range(repeat_count):
        # self.video_writer.write(self.cpu_resized_frame)

        # num_objs = 0
        # predictions = []
        metadata = dict()
        if contours:
            metadata, frame_bytes = self.get_detections_for_contours_bbs(
                frameNum, frame, contours, thickness=2, device_input=device_input
            )

            if metadata:
                all_metadata.setdefault(
                    self.clip_key,
                    {
                        "object": {},
                        "face": {},
                    },
                )
                all_metadata[self.clip_key]["object"].update(metadata)
            # all_metadata[clip_key]["face"].update(metadata_face)
        return frame_bytes
