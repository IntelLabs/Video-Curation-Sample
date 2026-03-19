import asyncio
import logging
import os
import queue
import shlex
import subprocess
import sys

# Force FFmpeg to use more threads for decoding
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

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
    NUM_USUABLE_CPUS,
    OMIT_DETECTIONS_FLAG,
    SHARED_OUTPUT,
    TARGET_FPS,
    TMP_LOCATION,
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
main_app_logger = logging.getLogger("fastapi_app")
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.INFO)


# ----- SPECIAL VARIABLES -----
CLIP_DURATION = 10  # seconds
KERNEL_RATIO = 0.05  # 0.03 # .05  # .025
MASK_MAX_VALUE = 255
MASK_THRESHOLD_VALUE = 127
MAX_DETECTIONS = 100
# MAX_FRAMES_PER_CLIP = int(TARGET_FPS * CLIP_DURATION)  # 150 frames
# MODEL_PRECISION = "FP16"
# MODEL_W, MODEL_H = (640, 640)
# NUM_USUABLE_CPUS = 2
# OMIT_DETECTIONS_FLAG = str2bool(os.getenv("OMIT_DETECTIONS_FLAG", False))
# SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
# Path(SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
# TEST_MODE = str2bool(os.getenv("TEST_FLAG", False))
# TMP_LOCATION = os.getenv("TMP_LOCATION", "/var/www/cache/")

if CUSTOM_MODEL_FLAG:
    model_path = f"{CODE_DIR}/resources/models/ultralytics/custom_models/{MODEL_NAME}"
else:
    model_path = f"{CODE_DIR}/resources/models/ultralytics/{MODEL_NAME}/{MODEL_PRECISION}/{MODEL_NAME}"
    # model_path = f"{CODE_DIR}/{MODEL_NAME}"

if DEVICE == "GPU":
    model_path += ".engine"
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


class VideoStreamHandler:
    def __init__(self, source, name):
        self.model = YOLO(model_path, verbose=False, task="detect")
        self.source = source
        self.name = name
        # self.cap = cv2.VideoCapture(source)
        # Use CAP_FFMPEG and increase internal buffer for RTSP
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Force the hardware buffer to 1 so we don't lag
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Set a timeout so it doesn't hang
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)

        # self.stat_start_time = time.perf_counter()
        self.stat_frame_count = 0
        self.stat_fps = 0

        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1, mp4v
        self.clip_id = 0
        self.clip_filename = ""
        self.clip_key = ""
        self.tmp_file = ""

        self.active = True
        self.frame = None
        self.latest_processed_frame = None
        self.last_write_time = time.time()
        self.last_frame_id = 0  # Increment this in your DETECTION loop
        self.sent_frame_id = -1  # Track what the BROWSER has already seen

        self.resize_h, self.resize_w = [MODEL_H, MODEL_W]
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.numFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.get_fps_and_framecnt()

        self.get_frameWH()

        self.scale_x = self.frame_width / MODEL_W
        self.scale_y = self.frame_height / MODEL_H
        self.min_contour_area = int(
            (0.005 * self.frame_width) * (0.005 * self.frame_height)
        )  # 207

        self.operation_device_map = PipelineMapping(
            detection_device="cpu"
        )  # No CUDA HERE
        self.device_input = (
            self.operation_device_map.detection_device
            if self.operation_device_map.detection_device == "cpu"
            else "cuda"
        )

        self.cpu_resized_frame = None

        # Subtraction
        history = 300  # int(5 * self.fps)
        background_thresh = 350
        NSamples = 10
        kNNSamples = 2
        self.lr = (
            -1
        )  # .01  #-1  # 0.001  #1 / (5 * self.fps)  # -1  # 0.01  # 1 / history
        bkgd_mask_queue_size = 3
        self.backSub_cpu = cv2.createBackgroundSubtractorKNN(
            history=history,  # default 500
            dist2Threshold=background_thresh,  # default 400
            detectShadows=False,  # default True
        )
        self.backSub_cpu.setkNNSamples(kNNSamples)
        self.backSub_cpu.setNSamples(NSamples)

        prev_bkgd = np.zeros((MODEL_H, MODEL_W), dtype="uint8")
        self.mask_history = deque(maxlen=bkgd_mask_queue_size)
        self.mask_history.append(prev_bkgd)

        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dilate_kernel_for_enhanced_mask = np.ones((21, 21), np.uint8)

        # Create ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=NUM_USUABLE_CPUS)
        self.metadata_thread = threading.Thread(target=self.send_metadata, daemon=True)

        self.thread = threading.Thread(target=self.update, daemon=True)

        self.process_thread = threading.Thread(target=self.run_inference, daemon=True)
        self.start()

    def start(self):
        # self.t = []
        # self.t.append(
        #     self.executor.submit(
        #         send_metadata,
        #     )
        # )
        self.metadata_thread.start()
        self.thread.start()
        self.process_thread.start()

    # def stop(self):
    #     self.active = False
    #     if self.video_writer:
    #         self.release_clip_and_reencode()
    #     # for t in as_completed(self.t):
    #     #     try:
    #     #         _ = t.result()
    #     #     except Exception as t_e:
    #     #         print(f"[DEBUG] Exception occurred in thread: {t_e}")
    #     send_metadata_queue.put(None)
    #     self.metadata_thread.join()

    #     self.executor.shutdown(wait=True, cancel_futures=False)
    #     # self.thread.join()
    #     # self.process_thread.join()
    #     self.cap.release()

    def stop(self):
        self.active = False  # Signals the while loops to exit

        # Release the VideoWriter if it exists
        if self.video_writer:
            self.release_clip_and_reencode()

        # Close the OpenCV capture
        if self.cap:
            self.cap.release()

        # Join threads if you want to be 100% sure they are closed
        # self.thread.join(timeout=1.0)
        # self.process_thread.join(timeout=1.0)

    def update_frame(self):
        self.stat_frame_count += 1
        elapsed = time.perf_counter() - self.stat_start_time
        if elapsed > 0.0:  # Update FPS every second
            self.stat_fps = self.stat_frame_count / elapsed
            # To keep it "real-time" and not a lifetime average, reset:
            # self.stat_start_time = time.perf_counter()
            # self.stat_frame_count = 0

    def send_metadata(self):
        # This loop runs in its own dedicated threading.Thread
        while True:
            try:
                # Blocks until something is in the queue
                queue_details = send_metadata_queue.get()

                if queue_details is None:  # Sentinel to shut down
                    break

                (clip_key, clip_filename, width, height) = queue_details

                clip_data = all_metadata.get(clip_key)

                if clip_data:
                    # Use the EXECUTOR to fire off the heavy metadata sending
                    # This returns immediately so the loop can grab the next item
                    future = self.executor.submit(
                        metadata2vdms,
                        clip_key,
                        clip_filename,
                        clip_data,
                        width,
                        height,
                    )
                    # Track success
                    future.add_done_callback(handle_done)

                    # Clean up dict entry after submitting to the thread
                    # Note: If metadata2vdms needs the data, pass it in (as done above)
                    del all_metadata[clip_key]

                # Mark the task as done in the queue
                send_metadata_queue.task_done()

            except Exception as e:
                print(f"Queue Error: {e}")

    # Gets video fps and framecount
    def get_fps_and_framecnt(self):
        self.input_fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # hardware fps
        if self.input_fps == 0:  # Case when FPs isn't available
            self.input_fps = manual_fps_calculation(self.name, num_frames=10)

        self.target_fps = TARGET_FPS if self.input_fps > TARGET_FPS else self.input_fps
        self.frame_skip = int(self.input_fps / self.target_fps)
        if self.frame_skip < 1:
            self.frame_skip = 1

        self.MAX_FRAMES_PER_CLIP = int(self.target_fps * CLIP_DURATION)
        self.target_interval = 1.0 / self.target_fps  # 0.0666s

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

    def run_inference(self):
        print(f"Inference thread for {self.name} started...")

        # 1. Initialize the start time and a counter for frames actually written
        start_time = time.time()
        self.stat_start_time = time.perf_counter()
        total_frames_written = 0

        while self.active:
            # 2. Calculate how many frames SHOULD be in the file by now
            elapsed_real_time = time.time() - start_time
            expected_total_frames = int(elapsed_real_time * self.target_fps)

            # 3. Determine the "Gap": How many frames do we need to write to stay in sync?
            # If processing took 200ms, slots_to_fill will be ~3.
            slots_to_fill = expected_total_frames - total_frames_written

            # if slots_to_fill > 0:
            slots_to_fill = 1
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            #  Handle Video Writing (Cycle every 10 seconds)
            # clip_frameNum = (expected_total_frames - 1) % self.MAX_FRAMES_PER_CLIP
            clip_frameNum = self.stat_frame_count % self.MAX_FRAMES_PER_CLIP
            if clip_frameNum == 0 or total_frames_written == 0:
                print(f"frameNum: {expected_total_frames} ({clip_frameNum})")
                if self.video_writer:
                    self.release_clip_and_reencode()
                if "://" not in str(self.source):
                    self.clip_filename = (
                        f"{SHARED_OUTPUT}/{self.name}_{self.clip_id}.mp4"
                    )
                else:
                    self.clip_filename = (
                        f"{SHARED_OUTPUT}/{self.name}_{time.time()}.mp4"
                    )

                self.tmp_file = TMP_LOCATION + self.clip_filename.split("/")[-1]
                self.clip_key = Path(self.clip_filename).name

                # timestamp = int(time.time())
                # filename = f"clip_{timestamp}.mp4"
                self.video_writer = cv2.VideoWriter(
                    self.tmp_file,
                    self.fourcc,
                    self.target_fps,
                    (MODEL_W, MODEL_H),
                    # (self.width, self.height)
                    # (self.frame_width, self.frame_height),
                )
                main_app_logger.info(f"Started new clip: {self.tmp_file}")

            try:
                # 4. PASS THE REPEAT COUNT to your function
                # This ensures the video length matches the stopwatch
                frame_bytes = self.test_full_cpu_detection_gpu(
                    frame, self.stat_frame_count + 1, repeat_count=slots_to_fill
                )
                if frame_bytes is None:
                    print(
                        f"CRITICAL: {self.name} detection returned NULL bytes. Check OpenVINO logs."
                    )
                else:
                    print(
                        f"SUCCESS: {self.name} pushed {len(frame_bytes)} bytes to memory."
                    )
                self.latest_processed_frame = frame_bytes
                self.last_heartbeat = time.time()
                self.last_frame_id += 1
                total_frames_written = expected_total_frames
                self.frame = None
                self.update_frame()
            except Exception as e:
                print(f"Inference Error: {e}")
            # else:
            #     # 6. We are ahead of the clock; wait for the next 66ms window
            #     # time.sleep(0.005)
            #     pass

    # def run_inference(self):
    #     print(f"Inference thread for {self.name} started...")

    #     # This counter now perfectly represents a 15fps clock
    #     frame_counter = 0

    #     while self.active:
    #         frame = self.get_frame()
    #         if frame is None:
    #             time.sleep(0.005)
    #             continue

    #         # frame_counter will now hit exactly 150 every 10 seconds
    #         frame_counter += 1

    #         #  Handle Video Writing (Cycle every 10 seconds)
    #         clip_frameNum = (frame_counter - 1) % self.MAX_FRAMES_PER_CLIP
    #         if clip_frameNum == 0:
    #             print(f"frameNum: {frame_counter} ({clip_frameNum})")
    #             if self.video_writer:
    #                 self.release_clip_and_reencode()
    #             if "://" not in str(self.source):
    #                 self.clip_filename = f"{SHARED_OUTPUT}/{self.name}_{self.clip_id}.mp4"
    #             else:
    #                 self.clip_filename = f"{SHARED_OUTPUT}/{self.name}_{time.time()}.mp4"

    #             self.tmp_file = TMP_LOCATION + self.clip_filename.split("/")[-1]
    #             self.clip_key = Path(self.clip_filename).name

    #             # timestamp = int(time.time())
    #             # filename = f"clip_{timestamp}.mp4"
    #             self.video_writer = cv2.VideoWriter(
    #                 self.tmp_file,
    #                 self.fourcc,
    #                 self.target_fps,
    #                 (MODEL_W, MODEL_H),
    #                 # (self.width, self.height)
    #                 # (self.frame_width, self.frame_height),
    #             )
    #             main_app_logger.info(f"Started new clip: {self.tmp_file}")

    #         try:

    #             # This triggers your 10s clip rotation (frameNum % 150 == 0)
    #             self.test_full_cpu_detection_gpu(frame, frame_counter)

    #             # self.latest_processed_frame = frame_bytes # From your detection function
    #             self.frame = None # Signal Reader for next frame

    #         except Exception as e:
    #             print(f"Inference Error: {e}")

    # def run_inference(self):
    #     print(f"Inference thread for {self.name} started...")

    #     target_fps = 15.0
    #     target_interval = 1.0 / target_fps  # 0.0666s

    #     start_time = time.time()
    #     total_frames_written = 0

    #     # Store the last frame to use as a "filler" if we lag
    #     last_processed_annotated = None

    #     while self.active:
    #         # 1. Calculate how many frames SHOULD be in the file by now
    #         elapsed_real_time = time.time() - start_time
    #         expected_total_frames = int(elapsed_real_time * target_fps)

    #         # 2. Are we behind the clock?
    #         if expected_total_frames > total_frames_written:
    #             # How many frames do we need to write to CATCH UP to the clock?
    #             frames_to_catch_up = expected_total_frames - total_frames_written

    #             frame = self.get_frame()
    #             if frame is not None:

    #                 #  Handle Video Writing (Cycle every 10 seconds)
    #                 clip_frameNum = (expected_total_frames - 1) % self.MAX_FRAMES_PER_CLIP
    #                 if clip_frameNum == 0:
    #                     print(f"frameNum: {expected_total_frames} ({clip_frameNum})")
    #                     if self.video_writer:
    #                         self.release_clip_and_reencode()
    #                     if "://" not in str(self.source):
    #                         self.clip_filename = f"{SHARED_OUTPUT}/{self.name}_{self.clip_id}.mp4"
    #                     else:
    #                         self.clip_filename = f"{SHARED_OUTPUT}/{self.name}_{time.time()}.mp4"

    #                     self.tmp_file = TMP_LOCATION + self.clip_filename.split("/")[-1]
    #                     self.clip_key = Path(self.clip_filename).name

    #                     # timestamp = int(time.time())
    #                     # filename = f"clip_{timestamp}.mp4"
    #                     self.video_writer = cv2.VideoWriter(
    #                         self.tmp_file,
    #                         self.fourcc,
    #                         self.target_fps,
    #                         (MODEL_W, MODEL_H),
    #                         # (self.width, self.height)
    #                         # (self.frame_width, self.frame_height),
    #                     )
    #                     main_app_logger.info(f"Started new clip: {self.tmp_file}")
    #                 try:
    #                     # 3. Process the latest frame (YOLO/KNN)
    #                     # This might take 100ms (more than one 66ms tick)
    #                     self.test_full_cpu_detection_gpu(frame, expected_total_frames)
    #                     # self.latest_processed_frame = frame_bytes

    #                     # 4. WRITER SYNC:
    #                     # If we missed 3 'ticks' while processing, write this frame 3 times.
    #                     # This is the "Brake" that stops the fast-forward.
    #                     if self.video_writer:
    #                         for _ in range(frames_to_catch_up):
    #                             self.video_writer.write(self.cpu_resized_frame)

    #                     total_frames_written = expected_total_frames
    #                     self.frame = None
    #                 except Exception as e:
    #                     print(f"Inference Error: {e}")
    #             else:
    #                 # No new frame from reader yet, but we MUST keep the clock moving
    #                 # Write the previous frame again to maintain video duration
    #                 if self.video_writer and total_frames_written > 0:
    #                     for _ in range(frames_to_catch_up):
    #                         self.video_writer.write(self.cpu_resized_frame)
    #                     total_frames_written = expected_total_frames
    #                 time.sleep(0.01)
    #         else:
    #             # 5. We are ahead of the clock (CPU is fast); wait for the next 66ms tick
    #             time.sleep(0.005)

    # def update(self):
    #     # Calculate exactly how much time should pass between 15fps frames
    #     # target_interval = 1.0 / self.target_fps  # 0.0666s
    #     last_grab_time = time.time()

    #     while self.active:
    #         # 1. Fast-forward the internal buffer using grab()
    #         # This clears out the 21fps 'junk' frames
    #         success = self.cap.grab()
    #         if not success:
    #             self.active = False
    #             break

    #         # 2. Check if enough time has passed to 'retrieve' a 15fps frame
    #         current_time = time.time()
    #         if current_time - last_grab_time >= self.target_interval:
    #             success, frame = self.cap.retrieve()
    #             if success:
    #                 self.frame = frame
    #                 last_grab_time = current_time

    #         time.sleep(0.001)

    def update(self):
        print(f"READER THREAD: Started for {self.name}")
        # The 'step' is 1.4 (21 in / 15 out)
        step = self.input_fps / self.target_fps
        # This tracks where the next 'keeper' frame is in the 21fps timeline
        next_keeper_idx = 0.0
        current_idx = 0

        while self.active:
            if not self.cap.isOpened():
                self.active = False
                break

            # 1. Skip (grab) frames until we reach the next 15fps 'slot'
            while current_idx < int(next_keeper_idx):
                if not self.cap.grab():
                    self.active = False
                    return
                current_idx += 1

            # 2. Retrieve (decode) the keeper frame
            success, frame = self.cap.read()  # read() includes grab + retrieve
            if not success:
                self.active = False
                break

            current_idx += 1
            # Advance the 'keeper' mark by 1.4
            next_keeper_idx += step

            # 3. Hand off the frame to the inference thread
            self.frame = frame
            # time.sleep(0.001)

        # --- AUTO-CLEANUP ---
        # Remove itself from the global dictionary so the dashboard knows it's gone
        if self.name in app.state.active_streams:
            app.state.active_streams[self.name].stop()
            del app.state.active_streams[self.name]

    # def update(self):
    #     # 1/15 = 0.066s interval
    #     target_interval = 1.0 / self.target_fps
    #     last_yield_time = time.time()

    #     while self.active:
    #         # 1. Grab (don't decode) as fast as possible to clear the buffer
    #         success = self.cap.grab()
    #         if not success:
    #             self.active = False
    #             break

    #         # 2. Only retrieve (decode) if 66ms has passed
    #         current_time = time.time()
    #         if current_time - last_yield_time >= target_interval:
    #             success, frame = self.cap.retrieve()
    #             if success:
    #                 self.frame = frame
    #                 last_yield_time = current_time

    #         time.sleep(0.001)

    def get_frame(self):
        return self.frame

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
                foi, self.frame_width, display_size=(1280, 720), quality=50
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

                    cv2.rectangle(
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
            foi, self.frame_width, display_size=(1280, 720), quality=50
        )

        return metadata, frame_bytes

    def release_clip_and_reencode(self):
        if self.video_writer is not None:
            threading.Thread(
                target=save_and_finalize_clip,
                args=(
                    self.clip_key,
                    self.video_writer,
                    self.clip_filename,
                    self.tmp_file,
                    self.target_fps,
                    MODEL_W,
                    MODEL_H,
                    # self.frame_width,
                    # self.frame_height,
                ),
                daemon=True,
            ).start()

            self.video_writer = None
            self.clip_id += 1

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

    def test_full_cpu_detection_gpu(self, frame, frameNum, repeat_count=1):
        # Resize directly into the pre-allocated Pinned Memory
        # This avoids a temporary CPU allocation
        H, W = self.resize_h, self.resize_w
        self.cpu_resized_frame = cv2.resize(frame, (W, H))
        self.video_writer.write(self.cpu_resized_frame)

        # Background Subtraction on CPU
        fgMask = self.backSub_cpu.apply(self.cpu_resized_frame, learningRate=self.lr)

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

        # Get Contours & Run Inference on detection_device
        device_input = (
            self.operation_device_map.detection_device
            if self.operation_device_map.detection_device == "cpu"
            else "cuda"
        )

        # num_objs, predictions =
        frame_bytes = self.contour2predictions(
            frameNum, mask, frame, device_input=device_input, repeat_count=repeat_count
        )
        return frame_bytes


class StreamRequest(BaseModel):
    url: str
    name: str


# --------------- APP -------------------
from contextlib import asynccontextmanager


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
                print(f"CLEANUP: Removing {name} from active_streams")
                streamer.stop()
                del app.state.active_streams[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This is the ONLY place this should be initialized
    if not hasattr(app.state, "active_streams"):
        app.state.active_streams = {}
    asyncio.create_task(auto_cleanup_janitor(app))
    print(f"--- APP STARTUP | PID: {os.getpid()} | STATE READY ---")
    yield
    # Cleanup logic here...
    for s in app.state.active_streams.values():
        s.stop()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.post("/stream")
async def stream_video(
    # url: str = Query(..., description="RTSP URL or Local File Path"),
    # name: str = Query(..., description="Name of stream"),
    data: StreamRequest,
):
    url, name = data.url, data.name
    # Start background thread
    if name not in app.state.active_streams:
        print(f"Starting background worker for {name}...")
        app.state.active_streams[name] = VideoStreamHandler(url, name)
    # DEBUG START
    curr_keys = list(app.state.active_streams.keys())
    print(
        f"stream DEBUG VIEW | PID: {os.getpid()} | Looking for: {name} | Found Keys: {curr_keys}"
    )
    # DEBUG END

    return {"status": "started", "keys": list(app.state.active_streams.keys())}


# @app.get("/view_stream", name="view_stream")
# async def view_stream(name: str):
#     # DEBUG START
#     curr_keys = list(app.state.active_streams.keys())
#     print(
#         f"view_stream DEBUG VIEW | PID: {os.getpid()} | Looking for: {name} | Found Keys: {curr_keys}"
#     )
#     # DEBUG END
#     # This will now find 'test_vid' because it's the same memory!
#     streamer = app.state.active_streams.get(name)

#     if not streamer:
#         raise HTTPException(status_code=404, detail="Stream not found")

#     async def get_frames():
#         while streamer.active:
#             frame_bytes = streamer.latest_processed_frame
#             if frame_bytes is None:
#                 print(f"DEBUG: {streamer.name} frame is still None...")
#                 await asyncio.sleep(0.1)  # Wait for first inference to finish
#                 continue

#             streamer.update_frame()

#             yield (
#                 b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
#             )
#             await asyncio.sleep(0.06)

#     return StreamingResponse(
#         get_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
#     )


@app.get("/debug_frame/{name}")
async def debug_frame(name: str):
    streamer = app.state.active_streams.get(name)
    if not streamer:
        return {"error": "not found"}
    # DEBUG START
    curr_keys = list(app.state.active_streams.keys())
    print(
        f"debug_frame DEBUG VIEW | PID: {os.getpid()} | Looking for: {name} | Found Keys: {curr_keys}"
    )
    # DEBUG END
    return {
        "active": streamer.active,
        "has_frame": streamer.latest_processed_frame is not None,
        "frame_size": len(streamer.latest_processed_frame)
        if streamer.latest_processed_frame
        else 0,
    }


@app.get("/stream_list")
async def get_stream_list(request: Request):
    """Returns a list of currently active stream names."""
    return list(request.app.state.active_streams.keys())


# New endpoint to provide stats
@app.get("/stream_stats")
async def get_stats(request: Request):
    # Return a dict mapping camera_id to its metrics
    return {
        cam_id: {"fps": round(state.stat_fps, 1), "frames": state.stat_frame_count}
        for cam_id, state in request.app.state.active_streams.items()
    }


@app.get("/")
async def index(request: Request):
    """Renders the dashboard."""
    print(f"Active Streams: {app.state.active_streams.keys()}")  # Check your terminal!
    curr_keys = list(app.state.active_streams.keys())
    return templates.TemplateResponse(
        "index.html", {"request": request, "cameras": curr_keys}
    )


# @app.get("/view_stream", name="view_stream")
# async def view_stream(name: str, request: Request):
#     # Initialize state if it doesn't exist
#     # if name not in app.state.active_streams:
#     #     app.state.active_streams[name] = CameraState()

#     async def frame_generator():
#         try:
#             while True:
#                 # Check if client disconnected to stop processing immediately
#                 if await request.is_disconnected(): #
#                     break

#                 frame_bytes = app.state.active_streams[name].latest_processed_frame
#                 if frame_bytes is None:
#                     print(f"DEBUG: {app.state.active_streams[name].name} frame is still None...")
#                     await asyncio.sleep(0.1)  # Wait for first inference to finish
#                     continue

#                 app.state.active_streams[name].update_frame()

#                 # app.state.active_streams[name].frame_count += 1
#                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         finally:
#             # THIS IS THE FIX: Remove the camera from the list when stream ends
#             print(f"Stream ended: {name}. Cleaning up.")
#             if name in app.state.active_streams:
#                 del app.state.active_streams[name]

#     return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/view_stream", name="view_stream")
async def view_stream(name: str, request: Request):
    if name not in request.app.state.active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    streamer = request.app.state.active_streams.get(name)
    if not streamer:
        raise HTTPException(status_code=404)

    async def frame_generator():
        # try:
        while streamer.active:
            if await request.is_disconnected():
                break
            # 2. Update Heartbeat for Auto-Cleanup
            # streamer.last_heartbeat = time.time()
            # 3. Only send a frame if a NEW one is ready
            if streamer.latest_processed_frame:
                # streamer.latest_processed_frame must be raw JPEG bytes
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + streamer.latest_processed_frame
                    + b"\r\n"
                )

            # ONLY process if there is a NEW frame from the detector
            # if streamer.last_frame_id > streamer.sent_frame_id:
            #     frame_bytes = streamer.latest_processed_frame
            #     if frame_bytes:
            #         # Sync IDs so we don't send this one again
            #         streamer.sent_frame_id = streamer.last_frame_id

            #         # Now this count is HONEST: 1 count = 1 unique AI frame
            #         streamer.update_frame()

            #         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
            #                 frame_bytes + b"\r\n")

            # Tiny sleep (1ms) to prevent 100% CPU usage while waiting
            # for the next unique frame to arrive from the detector.
            await asyncio.sleep(0.001)

        # finally:
        #     if name in request.app.state.active_streams:
        #         del request.app.state.active_streams[name]

    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/stop_stream/{name}")  # or @app.delete
async def stop_stream(name: str, request: Request):
    """Gracefully stops a background stream and cleans up memory."""
    streamer = request.app.state.active_streams.get(name)

    if not streamer:
        raise HTTPException(status_code=404, detail=f"Stream '{name}' not found.")

    # 1. Trigger the internal stop (releases CV2 cap and joins threads)
    streamer.stop()

    # 2. Remove from the shared state
    del request.app.state.active_streams[name]

    print(f"--- CLEANUP | Stream '{name}' stopped and removed. ---")
    return {"status": "stopped", "camera": name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
