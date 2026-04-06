# Copyright (C) 2025 Intel Corporation

import os
import shlex
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from random import randint

import cupy
import cv2
import numpy as np
from pydantic import BaseModel

# import streamlit as st
from ultralytics import YOLO

import vdms

"""
GENERAL DEFINITIONS/FUNCTIONS
"""


def safely_join_path(base_dir, add_path):
    safe_base = os.path.abspath(base_dir)
    candidate_path = os.path.abspath(os.path.join(safe_base, add_path))
    if not candidate_path.startswith(safe_base + os.sep):
        raise ValueError(f"Invalid path: {candidate_path}")
    return candidate_path


def str2bool(in_val):
    if isinstance(in_val, bool):
        return in_val

    if not isinstance(in_val, str):
        raise ValueError(f"{in_val} is not a bool or string")

    if in_val.title() == "True":
        return True
    else:
        return False


PROJECT_PATH = Path(__file__).parent.parent
CONDITION_OPTIONS = ["", ">", ">=", "<", "<=", "==", "!="]

AVAILABLE_MODELS = [  # Default model listed first
    "Ultralytics-yolo11n-ov-FP16",
    "Ultralytics-yolo11n-pt-FP16",
]

YOLO_BATCH_SIZE = 1
ENABLE_VDMS = os.getenv("ENABLE_VDMS", True)
DBPORT = 55555
DETECTION_THRESHOLD = 0.25
DEVICE_OV = "AUTO"
DYNAMIC_FLAG = True
FILETYPES = ["mp4", "avi"]
HALF_FLAG = True
IOU_THRESHOLD = 0.7
MODEL_PRECISION = "FP16"
MODEL_W, MODEL_H = (640, 640)
NUM_USUABLE_CPUS = 2
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ~0.0667 seconds
WRITER_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # avc1, mp4v

CODE_DIR = os.getenv("CODE_DIR", "/home")
CUSTOM_MODEL_FLAG = str2bool(os.getenv("CUSTOM_MODEL_FLAG", False))
DBHOST = os.getenv("DBHOST", "vdms-service")
DEBUG = os.getenv("DEBUG", "0")
DEBUG_FLAG = True if DEBUG == "1" else False
# DEVICE = os.environ.get("DEVICE", "CPU")
DEVICE = os.getenv("DEVICE", "CPU")
device_input = DEVICE.lower() if DEVICE == "CPU" else "cuda"
INGESTION = os.getenv("INGESTION", "object,face")
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n")
OMIT_DETECTIONS_FLAG = str2bool(os.getenv("OMIT_DETECTIONS_FLAG", False))
RESIZE_FLAG = str2bool(os.getenv("RESIZE_FLAG", False))
SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
Path(SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
TEST_MODE = str2bool(os.getenv("TEST_FLAG", False))
TMP_LOCATION = os.getenv("TMP_LOCATION", "/var/www/cache")
UDF_HOST = os.getenv("UDF_HOST", "udf-service")
UDF_PORT = 5011

if DEVICE == "GPU":
    EXPORT_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[!] USING GPU")
else:
    EXPORT_BATCH_SIZE = int(os.environ.get("CPU_BATCH_SIZE", 1))  # 8
    print("[!] USING CPU")

# if not TEST_MODE:
#     db = vdms.vdms()
#     db.connect(DBHOST, DBPORT)
import queue


class VDMSPool:
    def __init__(self, host, port, size=5):
        self.host = host
        self.port = port
        self.size = size
        self.pool = queue.Queue(maxsize=size)

        # Pre-populate the pool with authenticated connections
        for _ in range(size):
            self.pool.put(self._create_connection())

    def _create_connection(self):
        client = vdms.vdms()
        client.connect(self.host, self.port)
        return client

    def get_connection(self):
        # Borrow a connection (blocks if pool is empty)
        return self.pool.get(block=True, timeout=10)

    def return_connection(self, conn):
        # Put the connection back for reuse
        self.pool.put(conn)


if ENABLE_VDMS:  # == True:
    VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)

LOCKTIMEOUT_RETRIES = 5
ERR_KEYWORDS = [
    "timeout",
    "null search iterator",
    "outoftransactions",
    "internal server",
]


MASK_THRESHOLD_VALUE = 127
MASK_MAX_VALUE = 255
MAX_DETECTIONS = 100

# Plot variables
THICKNESS_SCALE_FACTOR = 1e-3
FONT_SCALE_FACTOR = 1e-3

bbox_kernel = cupy.ElementwiseKernel(
    "S label_image, int32 width",
    "raw T bboxes",
    """
    if (label_image > 0) {
        int label = (int)label_image;

        int y = i / width;
        int x = i % width;
        // Atomic operations to find min/max coordinates
        atomicMin(&bboxes[label * 4 + 0], y); // min_y
        atomicMin(&bboxes[label * 4 + 1], x); // min_x
        atomicMax(&bboxes[label * 4 + 2], y); // max_y
        atomicMax(&bboxes[label * 4 + 3], x); // max_x
    }
    """,
    "bbox_kernel",
)

bbox_area_kernel = cupy.ElementwiseKernel(
    "S label_image, int32 width",
    "raw T bboxes, raw T areas",
    """
    if (label_image > 0) {
        int label = (int)label_image;
        int y = i / width;
        int x = i % width;

        // 1. Update Bounding Box
        atomicMin(&bboxes[label * 4 + 0], y); // min_y
        atomicMin(&bboxes[label * 4 + 1], x); // min_x
        atomicMax(&bboxes[label * 4 + 2], y); // max_y
        atomicMax(&bboxes[label * 4 + 3], x); // max_x

        // 2. Increment Area (Count pixels)
        atomicAdd(&areas[label], 1);
    }
    """,
    "bbox_area_kernel",
)

threshold_dilate_fused_kernel = cupy.ElementwiseKernel(
    "T mask, int32 threshold, int32 width, int32 height",
    "raw T morphed",
    """
    // 1. Threshold
    bool is_active = mask > threshold;

    // 2. Simple 3x3 Dilation (Fuse directly into output)
    if (is_active) {
        int y = i / width;
        int x = i % width;

        // 2. 3x3 Dilation Expansion
        // Writes 255 to the neighbors of any pixel above threshold
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = y + dy;
                int nx = x + dx;
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    morphed[ny * width + nx] = 255;
                }
            }
        }
    }
    """,
    "threshold_dilate_fused",
)


YOLO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

PLOT_HEXS = (
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
)

DETECTION_COLORS = []
for h in PLOT_HEXS:
    DETECTION_COLORS.append(
        tuple(int(f"#{h}"[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
    )


def get_detection_color(index, is_bgr=False):
    ind = int(index) % len(PLOT_HEXS)
    color = DETECTION_COLORS[ind]
    if is_bgr:
        return (color[2], color[1], color[0])
    else:
        return color


def get_line_thickness(npixels, ref_pixels=(1280 * 720)):
    ref_thickness = 1
    factor = npixels / ref_pixels
    thickness = int(ref_thickness * factor)
    if thickness < 1:
        thickness = 1
    return thickness


def draw_label(
    image,
    label,
    txt_bt_lft_corner,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    color=(255, 255, 255),
    padding=5,
):
    height, width, _ = image.shape

    # Scale font and thickness based on the image's smaller dimension
    scaled_font_scale = min(width, height) * FONT_SCALE_FACTOR
    scaled_thickness = max(1, ceil(min(width, height) * THICKNESS_SCALE_FACTOR))

    # Get text size and define position for the label background
    (label_W, label_H), baseline = cv2.getTextSize(
        label, font_face, scaled_font_scale, scaled_thickness
    )
    label_y1 = (txt_bt_lft_corner[0], txt_bt_lft_corner[1] - label_H - padding)
    label_y2 = (
        txt_bt_lft_corner[0] + label_W + padding,
        txt_bt_lft_corner[1] + baseline,
    )
    cv2.rectangle(image, label_y1, label_y2, color, -1)

    # Print label
    cv2.putText(
        image,
        label,
        (txt_bt_lft_corner[0] + padding // 2, txt_bt_lft_corner[1] - padding // 2),
        font_face,
        scaled_font_scale,
        (0, 0, 0),  # Black text
        scaled_thickness,
        cv2.LINE_AA,
    )


def retry_query(
    query, local_db=None, num_retries: int = LOCKTIMEOUT_RETRIES, sleep_timer: int = 0
):
    # global db
    db = local_db if local_db else vdms.vdms().connect(DBHOST, DBPORT)
    for ridx in range(num_retries + 1):
        response, _ = db.query(query, [[]])
        if "FailedCommand" in response[0] and any(
            k in response[0]["info"].lower() for k in ERR_KEYWORDS
        ):
            err = response[0]["info"]
            if DEBUG == "1":
                query_type = list(query[0].keys())[0]
                print(
                    f"DEBUG [process_stream Attempt #{ridx}] Received '{err}' for {query_type} query",
                    flush=True,
                )
            if sleep_timer > 0:
                time.sleep(sleep_timer)
        else:
            if DEBUG == "1":
                print(
                    f"[DEBUG process_stream] Successful query response: {response}",
                    flush=True,
                )
            break  # Continue
    return response


def format_df_value(value):
    if value is None:
        return value
    if value.isdigit():
        if "." in value:
            return float(value)
        else:
            return int(value)
    return value


def get_model(
    MODEL_NAME,
    model_dir,
    run_platform,
    device_input,
    batch=1,
    half_flag=True,
    dynamic_flag=True,
):
    final_model_path = f"{model_dir}/{MODEL_NAME}.pt"
    pt_detection_model = YOLO(final_model_path, verbose=False, task="detect")
    if run_platform == "ov":
        final_model_path = f"{model_dir}/{MODEL_NAME}_openvino_model/"
        if not Path(final_model_path).exists():
            pt_detection_model.export(
                format="openvino",
                half=half_flag,
                dynamic=dynamic_flag,
                device=device_input,
                batch=batch,
            )

        object_detection_model = YOLO(
            final_model_path,
            verbose=False,
            task="detect",
        )

        # det_ov_model = core.read_model(final_model_path+"yolo11n.xml")
        # ov_config = {hints.performance_mode: hints.PerformanceMode.LATENCY}
        # if device == "GPU":
        #     ov_config["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "YES"
        # compiled_model = core.compile_model(det_ov_model, device, ov_config)
        # object_detection_model.predictor.model.ov_compiled_model = compiled_model

    elif run_platform == "engine":
        final_model_path = f"{model_dir}/{MODEL_NAME}.engine"
        if not Path(final_model_path).exists():
            pt_detection_model.export(
                format="engine",
                half=half_flag,
                imgsz=[7680, 4320],  # Max dimensions (8K-[W,H]-[7680,4320])
                dynamic=dynamic_flag,
                device=device_input,
                simplify=True,
                batch=batch,
            )

        object_detection_model = YOLO(
            final_model_path,
            verbose=False,
            task="detect",
        )

    elif run_platform == "onnx":
        from torch import cuda
        from ultralytics.utils.checks import check_requirements

        check_requirements(
            "onnxruntime-gpu"
            if cuda.is_available() and device_input != "cpu"
            else "onnxruntime"
        )

        final_model_path = f"{model_dir}/{MODEL_NAME}.onnx"
        if not Path(final_model_path).exists():
            pt_detection_model.export(
                format="onnx",
                half=half_flag,
                dynamic=dynamic_flag,
                device=device_input,
                simplify=True,
                batch=batch,
            )

        object_detection_model = YOLO(final_model_path, verbose=False, task="detect")

    elif run_platform == "pt":
        object_detection_model = pt_detection_model
        if device_input == "cuda":
            object_detection_model.to("cuda")
        else:
            object_detection_model.to(device_input)

    else:
        raise ValueError(f"[!] Model for {run_platform} is not implemented.")

    return (
        object_detection_model,
        final_model_path,
        list(object_detection_model.names.values()),
    )


def get_models(model_tag: str, model_dir=PROJECT_PATH / "models"):  # , _st_sidebar):
    # FW-Model Name-TYPE
    fw, model_name, model_fw, model_precision = model_tag.split("-")

    if fw == "Ultralytics":
        model, model_path, labels = get_model(
            model_name,
            model_dir / f"ultralytics/{model_precision}",
            model_fw,
            device_input,
            batch=EXPORT_BATCH_SIZE,
            half_flag=HALF_FLAG,
            dynamic_flag=DYNAMIC_FLAG,
        )
    else:
        raise ValueError(f"Model ({model_tag}) not implemented")

    return model, model_path, labels


def get_display_frame_in_bytes(
    foi, display_size=(960, 540), quality=50, return_bytes=True, device="CPU"
):
    H, W = foi.shape[:2]
    dH, dW = display_size
    if H == dH and W == dW:
        ret, buffer = cv2.imencode(
            ".jpg", foi, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        # print(f"[get_display_frame_in_bytes] display_size: {foi.shape}", flush=True)
    else:
        display_frame = cv2.resize(foi, display_size, interpolation=cv2.INTER_NEAREST)
        ret, buffer = cv2.imencode(
            ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        # print(f"[get_display_frame_in_bytes] display_size: {display_frame.shape}", flush=True)
    if ret and return_bytes:
        frame_bytes = buffer.tobytes()
    elif ret:
        frame_bytes = buffer
    else:
        frame_bytes = None

    return frame_bytes


def gpumat2cupy(gpu_mat):
    """Bridge OpenCV GpuMat to CuPy without copying data."""
    # 1. Get properties from GpuMat
    w, h = gpu_mat.size()
    # Check if it's 3-channel (CV_8UC3) or 1-channel (CV_8UC1)
    channels = 3 if gpu_mat.type() == cv2.CV_8UC3 else 1

    if channels == 3:
        shape = (h, w, 3)
        # strides = (bytes_per_row, bytes_per_pixel, bytes_per_channel)
        strides = (gpu_mat.step, 3, 1)
    else:
        shape = (h, w)
        strides = (gpu_mat.step, 1)

    # 2. Map OpenCV types to CuPy typestrs
    # CV_8UC1 is 'u1' (unsigned 1-byte), etc.
    # type_map = {cv2.CV_8U: "|u1", cv2.CV_32F: "<f4", cv2.CV_8UC1: "|u1"}

    # 3. Create the __cuda_array_interface__ dictionary
    # This tells CuPy where the data is and how it's shaped
    if_dict = {
        "version": 3,
        "shape": shape,
        "typestr": "|u1",
        # "descr": [("", type_map.get(gpu_mat.type(), "|u1"))],
        "data": (gpu_mat.cudaPtr(), False),  # (Pointer, Read-only)
        "strides": strides,
    }

    # 4. Create a dummy object with the interface and wrap it in CuPy
    class Holder:
        pass

    holder = Holder()
    holder.__cuda_array_interface__ = if_dict
    return cupy.asarray(holder)


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
    local_db=None,
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
        res = retry_query([query], local_db=local_db, sleep_timer=randint(1, 5))

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

    db = VDMS_POOL.get_connection()
    try:
        get_udf_query(
            clip_filename,
            properties,
            INGESTION.replace(",", "+"),
            (width, height),
            id="udf_metadata",
            metadata=combined_metadata,
            test_mode=TEST_MODE,
            local_db=db,
        )

        if DEBUG == "1":
            print(
                f"[TIMING],end_clip_metadata,{clip_key},{time.time()}",
                flush=True,
            )
    finally:
        VDMS_POOL.return_connection(db)


# Extract metadata from object model results
def extract_metadata_from_results(
    stream_name, frameNum, results, img_size, fps=TARGET_FPS
):
    fW, fH = img_size
    metadata = dict()
    try:
        for _, result in enumerate(results):
            # GET METADATA FOR CLIP
            boxes = result.boxes.cpu()
            oidx = 0
            for box in boxes:
                confidence = float(box.conf.item())
                if confidence > DETECTION_THRESHOLD:
                    class_id = int(box.cls.item())
                    class_name = str(result.names[class_id])

                    if not OMIT_DETECTIONS_FLAG:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        print(
                            # f"[OBJECT DETECTION] {class_name} detected in frame {frameNum} (Total detected: {current_cnt})",
                            f"[{timestamp}] {stream_name} DETECTION on Frame {frameNum}: {class_name} detected",
                            flush=True,
                        )
                    x1, y1, x2, y2 = box.xyxy.tolist()[0]
                    height = min(y2, fH) - max(0, y1)
                    width = min(x2, fW) - max(0, x1)
                    object_res = [
                        x1,
                        y1,
                        height,
                        width,
                        result.names[class_id],
                        confidence,
                        fH,
                        fW,
                    ]

                    framenum_str = f"{frameNum:04d}_{oidx:04d}"
                    if DEBUG_FLAG:
                        meta_str = ",".join(
                            [str(o) for o in object_res + [framenum_str]]
                        )
                        print(f"[{stream_name} METADATA],{meta_str}", flush=True)

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
                                "frameH": int(fH),
                                "frameW": int(fW),
                            },
                        },
                    }
                    oidx += 1

    except Exception:
        e = traceback.format_exc()
        print(f"Error in {stream_name} extract_metadata_from_results: {e}", flush=True)

    return metadata


# Release Video Writer object and re-encode video to seek via ffmpeg later
def release_clip_and_reencode(clip_key, _out_vid, clip_filename, tmp_file, target_fps):
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
    _out_vid = None

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
    return _out_vid


# def merge_boxes_limit(bbs_full_res, dist_threshold=50, size_limit=640):
#     """
#     boxes: list of [x1, y1, x2, y2]
#     dist_threshold: max distance between boxes to consider them 'connected'
#     size_limit: max width/height for a merged box
#     """
#     if len(bbs_full_res) == 0:
#         return []

#     rects = np.array(bbs_full_res)
#     num_boxes = len(rects)
#     parent = list(range(num_boxes))

#     def find(i):
#         if parent[i] == i:
#             return i
#         parent[i] = find(parent[i])
#         return parent[i]

#     def union(i, j):
#         root_i, root_j = find(i), find(j)
#         if root_i != root_j:
#             # Check if merging exceeds size limit
#             temp_x1 = min(rects[root_i][0], rects[root_j][0])
#             temp_y1 = min(rects[root_i][1], rects[root_j][1])
#             temp_x2 = max(rects[root_i][2], rects[root_j][2])
#             temp_y2 = max(rects[root_i][3], rects[root_j][3])

#             if (temp_x2 - temp_x1 <= size_limit) and (temp_y2 - temp_y1 <= size_limit):
#                 parent[root_i] = root_j
#                 # Update the root rectangle to the new merged bounds
#                 rects[root_j] = [temp_x1, temp_y1, temp_x2, temp_y2]

#     # 2. Compare boxes (Optimized: only check nearby ones if sorted by X)
#     for i in range(num_boxes):
#         for j in range(i + 1, num_boxes):
#             # Proximity check (Manhattan distance or check if boxes are 'close')
#             dx = max(0, max(rects[i][0], rects[j][0]) - min(rects[i][2], rects[j][2]))
#             dy = max(0, max(rects[i][1], rects[j][1]) - min(rects[i][3], rects[j][3]))

#             if dx < dist_threshold and dy < dist_threshold:
#                 union(i, j)

#     # 3. Extract unique merged boxes
#     final_boxes = []
#     unique_roots = set()
#     for i in range(num_boxes):
#         root = find(i)
#         if root not in unique_roots:
#             unique_roots.add(root)
#             final_boxes.append(rects[root])

#     return final_boxes


def merge_boxes_limit(bbs_full_res, dist_threshold=50, size_limit=640):
    if not bbs_full_res:
        return []

    # 🏎️ Optimization 1: Sort by X to allow early exit
    bbs_full_res = sorted(bbs_full_res, key=lambda x: x[0])
    num_boxes = len(bbs_full_res)
    merged = []
    used = [False] * num_boxes

    for i in range(num_boxes):
        if used[i]:
            continue

        curr = bbs_full_res[i]
        used[i] = True

        # Greedy merge with neighbors
        for j in range(i + 1, num_boxes):
            if used[j]:
                continue

            # 🏎️ Optimization 2: Early Exit
            # If the next box starts further away than the threshold,
            # no subsequent boxes can possibly merge.
            if bbs_full_res[j][0] - curr[2] > dist_threshold:
                break

            other = bbs_full_res[j]
            # Check Y proximity
            dy = max(0, max(curr[1], other[1]) - min(curr[3], other[3]))

            if dy < dist_threshold:
                # Calculate potential merge
                nx1, ny1 = min(curr[0], other[0]), min(curr[1], other[1])
                nx2, ny2 = max(curr[2], other[2]), max(curr[3], other[3])

                if (nx2 - nx1 <= size_limit) and (ny2 - ny1 <= size_limit):
                    curr = [nx1, ny1, nx2, ny2]
                    used[j] = True
        merged.append(curr)
    return merged


# def filter_contained_boxes(boxes, containment_thresh=0.90):
#     """
#     Deletes redundant boxes that are mostly inside another larger box.
#     """
#     if not boxes:
#         return []

#     # 1. Sort by area (Largest boxes first)
#     boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
#     keep = []

#     for child in boxes:
#         is_contained = False
#         for parent in keep:
#             # Intersection coordinates
#             ix1, iy1 = max(child[0], parent[0]), max(child[1], parent[1])
#             ix2, iy2 = min(child[2], parent[2]), min(child[3], parent[3])

#             if ix2 > ix1 and iy2 > iy1:
#                 inter_area = (ix2 - ix1) * (iy2 - iy1)
#                 child_area = (child[2] - child[0]) * (child[3] - child[1])

#                 # If child is 90% inside a larger box, it's redundant
#                 if inter_area / child_area >= containment_thresh:
#                     is_contained = True
#                     break

#         if not is_contained:
#             keep.append(child)

#     return keep


def filter_contained_boxes(boxes, containment_thresh=0.90):
    if len(boxes) < 2:
        return boxes

    # Convert to NumPy for vectorized math
    objs = np.array(boxes)
    areas = (objs[:, 2] - objs[:, 0]) * (objs[:, 3] - objs[:, 1])

    # Sort by area descending
    order = areas.argsort()[::-1]
    objs = objs[order]
    areas = areas[order]

    keep = []
    idx_list = np.arange(len(objs))

    while len(idx_list) > 0:
        i = idx_list[0]
        keep.append(objs[i].tolist())
        if len(idx_list) == 1:
            break

        # Vectorized Intersection over Union (IoU) / Containment
        others = objs[idx_list[1:]]
        ix1 = np.maximum(objs[i, 0], others[:, 0])
        iy1 = np.maximum(objs[i, 1], others[:, 1])
        ix2 = np.minimum(objs[i, 2], others[:, 2])
        iy2 = np.minimum(objs[i, 3], others[:, 3])

        iw = np.maximum(0, ix2 - ix1)
        ih = np.maximum(0, iy2 - iy1)
        inter_area = iw * ih

        # Calculate how much 'others' are contained within 'i'
        containment = inter_area / areas[idx_list[1:]]

        # Only keep boxes that are NOT mostly contained within the current box
        idx_list = idx_list[1:][containment < containment_thresh]

    return keep


@dataclass
class PipelineMapping:
    resize_device: str.lower = "cpu"
    bkgd_subtraction_device: str.lower = "cpu"
    threshold_device: str.lower = "cpu"
    erodeAndDilate_device: str.lower = "cpu"
    contour_device: str.lower = "cpu"
    detection_device: str.lower = "cpu"


class StreamRequest(BaseModel):
    url: str
    name: str
