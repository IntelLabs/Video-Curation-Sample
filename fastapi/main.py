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
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from random import randint

import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|hwaccel;cuda|threads;2|probesize;32|analyzeduration;0"
)


# from process_stream import extract_metadata_from_results, release_clip_and_reencode, retry_query
from include.utils import (
    UDF_HOST,
    UDF_PORT,
    YOLO_CLASS_NAMES,
    PipelineMapping,
    draw_label,
    filter_contained_boxes,
    get_detection_color,
    merge_boxes_limit,
    retry_query,
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
def str2bool(in_val):
    if isinstance(in_val, bool):
        return in_val

    if not isinstance(in_val, str):
        raise ValueError(f"{in_val} is not a bool or string")

    if in_val.title() == "True":
        return True
    else:
        return False


CLIP_DURATION = 10  # seconds
CODE_DIR = os.getenv("CODE_DIR", "/home")
CUSTOM_MODEL_FLAG = str2bool(os.getenv("CUSTOM_MODEL_FLAG", False))
DBHOST = os.getenv("DBHOST", "vdms-service")
DEBUG = os.getenv("DEBUG", "0")
DEBUG_FLAG = True if DEBUG == "1" else False
DETECTION_THRESHOLD = 0.25
DEVICE = os.getenv("DEVICE", "CPU")
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ~0.0667 seconds
INGESTION = os.getenv("INGESTION", "object,face")
KERNEL_RATIO = 0.05  # 0.03 # .05  # .025
MASK_MAX_VALUE = 255
MASK_THRESHOLD_VALUE = 127
MAX_DETECTIONS = 100
MAX_FRAMES_PER_CLIP = int(TARGET_FPS * CLIP_DURATION)  # 150 frames
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n")
MODEL_PRECISION = "FP16"
MODEL_W, MODEL_H = (640, 640)
NUM_USUABLE_CPUS = 2
OMIT_DETECTIONS_FLAG = str2bool(os.getenv("OMIT_DETECTIONS_FLAG", False))
SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
Path(SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
TEST_MODE = str2bool(os.getenv("TEST_FLAG", False))
TMP_LOCATION = os.getenv("TMP_LOCATION", "/var/www/cache/")

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


# Manual FPS calculation if OpenCV reports 0
def manual_fps_calculation(src, num_frames=10):
    vid_obj = cv2.VideoCapture(src)

    frame_count = 0
    start_t = time.time()

    while frame_count < num_frames:
        grabbed, frame = vid_obj.read()

        if not grabbed:
            break

        frame_count += 1

    end_t = time.time()
    vid_obj.release()

    elapsed_t = end_t - start_t

    if elapsed_t > 0:
        return frame_count / elapsed_t
    else:
        return 0


# Generate and run UDF query
def get_udf_query(
    filename_path,
    properties,
    ingest_mode,
    new_size,
    id="udf_metadata",
    metadata=None,
    test_mode=TEST_MODE,
):
    query = {
        "AddVideo": {
            "from_file_path": str(filename_path),  # from_server_file
            "is_local_file": True,
            "properties": properties,
            "operations": [
                {
                    "type": "syncremoteOp",  # "remoteOp",
                    "url": f"http://{UDF_HOST}:{UDF_PORT}/video",
                    "options": {
                        "id": id,
                        "otype": ingest_mode,
                        "media_type": "video",
                        "input_sizeWH": new_size,
                        "filename": properties["Name"],
                        "ingestion": 1,
                    },
                }
            ],
        }
    }

    if id == "udf_metadata" and metadata is not None:
        query["AddVideo"]["operations"][0]["options"]["metadata"] = metadata

    if test_mode:
        return

    filename = str(Path(filename_path).name)
    if DEBUG_FLAG:
        print(
            f"[TIMING],start_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
    try:
        res = retry_query([query], sleep_timer=randint(1, 5))

        if DEBUG_FLAG:
            print(
                f"[TIMING],end_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
                flush=True,
            )
            print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
            print(f"[DEBUG] {filename} INGEST_VIDEO RESPONSE: {res}", flush=True)
    except Exception:
        e = traceback.format_exc()
        print(f"[DEBUG] VDMS Query Exception: {e}", flush=True)


def _sort_dict_by_frame(in_dict):
    def _by_int(key):
        return tuple(int(k) for k in key.split("_"))

    return dict(sorted(in_dict.items(), key=lambda x: _by_int(x[0])))


# method to send metadata to VDMS once clip is saved
def metadata2vdms(
    clip_key,
    clip_filename,
    clip_metadata,
    width,
    height,
):
    if DEBUG == "1":
        print(
            f"[TIMING],start_clip_metadata,{clip_key},{time.time()}",
            flush=True,
        )

    # Send metadata to UDF
    properties = {
        "Name": clip_key,  # .split("/")[-1],
        "category": "video_path_rop",
    }

    combined_metadata = clip_metadata["object"] if "object" in clip_metadata else {}
    if "face" in clip_metadata:
        for face_frameidx_bbidx, value in clip_metadata["face"].items():
            face_frameidx, face_bbidx = face_frameidx_bbidx.split("_")
            max_obj_idx = 0
            for obj_frameidx_bbidx in combined_metadata:
                if face_frameidx in obj_frameidx_bbidx:
                    _, obj_bbidx_ = obj_frameidx_bbidx.split("_")
                    max_obj_idx = max(max_obj_idx, int(obj_bbidx_))

            if max_obj_idx > 0:
                new_face_bbidx = max_obj_idx + 1
                new_key = f"{face_frameidx}_{new_face_bbidx:04d}"
                combined_metadata[new_key] = value
                combined_metadata[new_key]["bbId"] = new_key
            else:
                combined_metadata[face_frameidx_bbidx] = value

    combined_metadata = _sort_dict_by_frame(combined_metadata)
    get_udf_query(
        clip_filename,
        properties,
        INGESTION.replace(",", "+"),
        (width, height),
        id="udf_metadata",
        metadata=combined_metadata,
        test_mode=TEST_MODE,
    )

    if DEBUG == "1":
        print(
            f"[TIMING],end_clip_metadata,{clip_key},{time.time()}",
            flush=True,
        )


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


# --------------- APP -------------------
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This is the ONLY place this should be initialized
    if not hasattr(app.state, "active_streams"):
        app.state.active_streams = {}
    print(f"--- APP STARTUP | PID: {os.getpid()} | STATE READY ---")
    yield
    # Cleanup logic here...


app = FastAPI(lifespan=lifespan)


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

        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.clip_id = 0
        self.clip_filename = ""
        self.clip_key = ""
        self.tmp_file = ""

        self.active = True
        self.frame = None
        self.latest_processed_frame = None
        self.last_write_time = time.time()

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

        self.thread = threading.Thread(target=self.update, daemon=True)

        self.process_thread = threading.Thread(target=self.run_inference, daemon=True)
        self.start()

    # Gets video fps and framecount
    def get_fps_and_framecnt(self):
        self.input_fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # hardware fps
        if self.input_fps == 0:  # Case when FPs isn't available
            self.input_fps = manual_fps_calculation(self.stream_id, num_frames=10)

        self.target_fps = TARGET_FPS if self.input_fps > TARGET_FPS else self.input_fps
        self.frame_skip = int(self.input_fps / self.target_fps)
        if self.frame_skip < 1:
            self.frame_skip = 1

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

    # def run_inference(self):
    #     # # streamer = VideoStreamHandler(source_url, name)

    #     # # Skip frames if they aren't fresh
    #     last_frame_time = time.time()
    #     frame_counter = 0
    #     print(f"Inference thread for {self.name} started...")

    #     while self.active:
    #         current_time = time.time()
    #         # Only process if 66ms has passed
    #         if current_time - last_frame_time < FRAME_INTERVAL:
    #             time.sleep(0.001)
    #             continue

    #         # 1. Get frame from the update() thread
    #         frame = self.get_frame()
    #         if frame is None:
    #             time.sleep(0.01)
    #             continue

    #         # if success:
    #         try:
    #             frame_bytes = self.test_full_cpu_detection_gpu(frame, frame_counter + 1)
    #             if frame_bytes is not None:
    #                 frame_counter += 1
    #                 self.latest_processed_frame = frame_bytes
    #             else:
    #                 print("DEBUG: test_full_cpu_detection_gpu returned None")

    #             # Reset frame to signal we are ready for the next one
    #             last_frame_time = current_time
    #             self.frame = None

    #         except Exception as e:
    #             print(f"Inference Error: {e}")

    #         time.sleep(0.001)

    # def run_inference(self):
    #     frames_written = 0
    #     frame_counter = 0
    #     print(f"Inference thread for {self.name} started...")

    #     last_frame_time = time.time()
    #     while self.active:
    #         # elapsed_t = time.time() - last_frame_time
    #         # expected_frames = int(elapsed_t * TARGET_FPS)

    #         # Only process if 66ms has passed
    #         if frame_counter + 1 > frames_written:
    #             print(f"expected_frames > frames_written: {frame_counter + 1} > {frames_written}")
    #             frame = self.get_frame()
    #             if frame is None:
    #                 time.sleep(0.01)
    #                 continue

    #         try:
    #             frameNum = frame_counter + 1
    #             frame_bytes = self.test_full_cpu_detection_gpu(frame, frameNum )
    #             frames_written = frameNum
    #             self.latest_processed_frame = frame_bytes
    #             self.frame = None
    #             frame_counter += 1

    #         except Exception as e:
    #             print(f"Inference Error: {e}")

    #         time.sleep(0.001)

    # def run_inference(self):
    #     start_time = time.time()
    #     frames_accounted_for = 0

    #     while self.active:
    #         current_time = time.time()
    #         elapsed_real_time = current_time - start_time
    #         expected_total_frames = int(elapsed_real_time * self.target_fps)

    #         # How many 15fps 'slots' have passed since we last processed?
    #         # If processing took 133ms, this will be 2.
    #         frames_to_write = expected_total_frames - frames_accounted_for

    #         if frames_to_write > 0:
    #             frame = self.get_frame()
    #             if frame is None:
    #                 time.sleep(0.01)
    #                 continue

    #             try:
    #                 # PASS THE REPEAT COUNT to your function
    #                 frame_bytes = self.test_full_cpu_detection_gpu(
    #                     frame,
    #                     expected_total_frames,
    #                     repeat_count=frames_to_write
    #                 )

    #                 frames_accounted_for = expected_total_frames
    #                 self.latest_processed_frame = frame_bytes
    #                 self.frame = None
    #             except Exception as e:
    #                 print(f"Inference Error: {e}")
    #         else:
    #             time.sleep(0.005) # Wait for the next 66ms slot

    def run_inference(self):
        print(f"Inference thread for {self.name} started...")

        # 1. Initialize the start time and a counter for frames actually written
        start_time = time.time()
        total_frames_written = 0
        target_fps = 15.0  # This must match your VideoWriter TARGET_FPS

        while self.active:
            # 2. Calculate how many frames SHOULD be in the file by now
            elapsed_real_time = time.time() - start_time
            expected_total_frames = int(elapsed_real_time * target_fps)

            # 3. Determine the "Gap": How many frames do we need to write to stay in sync?
            # If processing took 200ms, slots_to_fill will be ~3.
            slots_to_fill = expected_total_frames - total_frames_written

            if slots_to_fill > 0:
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    # 4. PASS THE REPEAT COUNT to your function
                    # This ensures the video length matches the stopwatch
                    self.test_full_cpu_detection_gpu(
                        frame, expected_total_frames, repeat_count=slots_to_fill
                    )

                    # 5. Sync the counter
                    total_frames_written = expected_total_frames
                    self.frame = None
                except Exception as e:
                    print(f"Inference Error: {e}")
            else:
                # 6. We are ahead of the clock; wait for the next 66ms window
                time.sleep(0.005)

    # def update(self):
    #     while self.active:
    #         if not self.cap.isOpened():
    #             print(f"[ERROR] Camera {self.name} lost connection.")
    #             self.active = False
    #             break

    #         # # 1. Check if the frame was successfully grabbed
    #         # grabbed = self.cap.grab()
    #         # if not grabbed:
    #         #     print(f"[INFO] Stream ended or disconnected: {self.name}")
    #         #     self.active = False # <--- TRIGGER SHUTDOWN
    #         #     break

    #         # # 2. Only retrieve if main loop is ready
    #         # if self.frame is None:
    #         #     success, frame = self.cap.retrieve()
    #         #     if not success:
    #         #         self.active = False
    #         #         break
    #         #     self.frame = frame

    #         # time.sleep(0.001)
    #         success, frame = self.cap.read()
    #         if success:
    #             self.frame = frame
    #             # print("READER THREAD: Frame captured") # Silent once working
    #         else:
    #             # print("READER THREAD: Failed to read from file!")
    #             # time.sleep(1)
    #             self.active = False
    #             break

    #     # 3. Clean up hardware handles automatically
    #     # self.stop()

    # def update(self):
    #     print(f"READER THREAD: Started for {self.name}")
    #     while self.active:
    #         if not self.cap.isOpened():
    #             self.active = False
    #             break

    #         # 1. Calculate how many frames to SKIP
    #         # We want to skip 'self.frame_skip - 1' frames
    #         for _ in range(self.frame_skip - 1):
    #             # grab() is 5x faster than read() because it doesn't decode
    #             if not self.cap.grab():
    #                 self.active = False
    #                 return

    #         # 2. Only decode (read/retrieve) the ONE frame we actually want
    #         success, frame = self.cap.read()

    #         if not success:
    #             print(f"READER THREAD: {self.name} reached end of file.")
    #             self.active = False
    #             break

    #         # 3. Hand the decoded frame to the inference thread
    #         # No 'if skip_frame_num' logic needed here anymore
    #         self.frame = frame

    #         # 4. Tiny sleep to prevent this thread from starving the YOLO thread
    #         time.sleep(0.001)

    def update(self):
        # Calculate exactly how much time should pass between 15fps frames
        target_interval = 1.0 / self.target_fps  # 0.0666s
        last_grab_time = time.time()

        while self.active:
            # 1. Fast-forward the internal buffer using grab()
            # This clears out the 21fps 'junk' frames
            success = self.cap.grab()
            if not success:
                self.active = False
                break

            # 2. Check if enough time has passed to 'retrieve' a 15fps frame
            current_time = time.time()
            if current_time - last_grab_time >= target_interval:
                success, frame = self.cap.retrieve()
                if success:
                    self.frame = frame
                    last_grab_time = current_time

            time.sleep(0.001)

    def get_frame(self):
        return self.frame

    def new_get_detections_for_contours_bbs(
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
            if self.frame_width > 1280:
                display_frame = cv2.resize(
                    foi, (1280, 720), interpolation=cv2.INTER_AREA
                )
                _, buffer = cv2.imencode(
                    ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                )
            else:
                _, buffer = cv2.imencode(
                    ".jpg", foi, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                )
            frame_bytes = buffer.tobytes()
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
        if self.frame_width > 1280:
            display_frame = cv2.resize(foi, (1280, 720), interpolation=cv2.INTER_AREA)
            ret, buffer = cv2.imencode(
                ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            )
        else:
            ret, buffer = cv2.imencode(".jpg", foi, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ret:
            frame_bytes = buffer.tobytes()
        else:
            frame_bytes = None

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

    def new_contour2predictions(
        self, frameNum, mask, frame, device_input="cpu", repeat_count=1
    ):
        source = self.source
        stream_name = self.name
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #  Handle Video Writing (Cycle every 10 seconds)
        clip_frameNum = (frameNum - 1) % MAX_FRAMES_PER_CLIP
        if clip_frameNum == 0:
            print(f"frameNum: {frameNum} ({clip_frameNum})")
            if self.video_writer:
                self.release_clip_and_reencode()
            if "://" not in str(source):
                self.clip_filename = f"{SHARED_OUTPUT}/{stream_name}_{self.clip_id}.mp4"
            else:
                self.clip_filename = f"{SHARED_OUTPUT}/{stream_name}_{time.time()}.mp4"

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

        # 3. Write frame
        # self.video_writer.write(frame)
        if self.video_writer:
            for _ in range(repeat_count):
                self.video_writer.write(self.cpu_resized_frame)

        # num_objs = 0
        # predictions = []
        metadata = dict()
        if contours:
            metadata, frame_bytes = self.new_get_detections_for_contours_bbs(
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
        frame_bytes = self.new_contour2predictions(
            frameNum, mask, frame, device_input=device_input, repeat_count=repeat_count
        )
        return frame_bytes

    def start(self):
        self.t = []
        self.t.append(
            self.executor.submit(
                send_metadata,
            )
        )
        self.thread.start()
        self.process_thread.start()

    def stop(self):
        self.active = False
        for t in as_completed(self.t):
            try:
                _ = t.result()
            except Exception as t_e:
                print(f"[DEBUG] Exception occurred in thread: {t_e}")

        self.cap.release()


class StreamRequest(BaseModel):
    url: str
    name: str


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


@app.get("/view_stream")
async def view_stream(name: str):
    # DEBUG START
    curr_keys = list(app.state.active_streams.keys())
    print(
        f"view_stream DEBUG VIEW | PID: {os.getpid()} | Looking for: {name} | Found Keys: {curr_keys}"
    )
    # DEBUG END
    # This will now find 'test_vid' because it's the same memory!
    streamer = app.state.active_streams.get(name)

    if not streamer:
        raise HTTPException(status_code=404, detail="Stream not found")

    async def get_frames():
        while streamer.active:
            frame_bytes = streamer.latest_processed_frame
            if frame_bytes is None:
                print(f"DEBUG: {streamer.name} frame is still None...")
                await asyncio.sleep(0.1)  # Wait for first inference to finish
                continue

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            await asyncio.sleep(0.06)

    return StreamingResponse(
        get_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
