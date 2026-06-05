# Copyright (C) 2025 Intel Corporation

import os
import queue
import subprocess
import time
import traceback
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from random import randint

import cupy
import cupyx.scipy
import cupyx.scipy.ndimage
import cv2
import numpy as np
import torch
from include.default_configs import (
    BKGD_SUB_INCLUDE_HISTORY,
    BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE,
    BKGD_SUB_INCLUDE_HISTORY_METHOD,
    BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE,
    BKGD_SUB_MOG2_DETECTSHADOWS,
    BKGD_SUB_MOG2_HISTORY,
    BKGD_SUB_MOG2_LR,
    BKGD_SUB_MOG2_VARTHRESHOLD,
    CLIP_DURATION_DEFAULT,
    CODE_DIR_DEFAULT,
    CUSTOM_MODEL_FLAG_DEFAULT,
    DBHOST_DEFAULT,
    DBPORT_DEFAULT,
    DEBUG_DEFAULT,
    DETECTION_THRESHOLD_DEFAULT,
    DETECTION_TYPE_DEFAULT,
    DEVICE_DEFAULT,
    DILATE_KERNEL_SIZE,
    DISPLAY_FRAME_QUALITY,
    DISPLAY_FRAME_SIZE,
    INGESTION_DEFAULT,
    MAX_DETECTIONS,
    MAX_WORKERS,
    MODEL_H,
    MODEL_MAX_BATCH_SIZE,
    MODEL_NAME_DEFAULT,
    MODEL_PRECISION,
    MODEL_W,
    OMIT_DETECTIONS_FLAG_DEFAULT,
    RESIZE_FLAG_DEFAULT,
    ROI_BB_FULL_RES_PADDING,
    ROI_CONTAINMENT_THRESH,
    ROI_DISTANCE_THRESH_RATIO,
    ROI_MAX_RELATIVE_SIZE_RATIO,
    ROI_MERGE_SIZE_LIMIT,
    ROI_MIN_AREA_RATIO,
    SHARED_OUTPUT_DEFAULT,
    SMART_FILTERING_ENABLED,
    SMART_FILTERING_PIXEL_CONSTRAINT,
    TARGET_FPS,
    TEST_MODE_DEFAULT,
    THICKNESS,
    THRESHOLD_MAX_VALUE,
    THRESHOLD_VALUE,
    TMP_LOCATION_DEFAULT,
    UDF_HOST_DEFAULT,
    UDF_PORT_DEFAULT,
)
from pydantic import BaseModel

# import streamlit as st
import vdms

"""
GENERAL DEFINITIONS/FUNCTIONS
"""


def merge_boxes_gpu(raw_boxes, gap_limit=10):
    device_input = "cuda"
    x1, y1, x2, y2 = raw_boxes.unbind(1)

    # Calculate pairwise gaps [N, N] using broadcasted subtraction
    h_gaps = torch.max(
        torch.zeros(1, device=device_input),
        torch.max(x1.unsqueeze(0) - x2.unsqueeze(1), x1.unsqueeze(1) - x2.unsqueeze(0)),
    )
    v_gaps = torch.max(
        torch.zeros(1, device=device_input),
        torch.max(y1.unsqueeze(0) - y2.unsqueeze(1), y1.unsqueeze(1) - y2.unsqueeze(0)),
    )

    adj = (h_gaps < gap_limit) & (v_gaps < gap_limit)

    # Parallel Connected Components (Iterative grouping)
    components = torch.arange(raw_boxes.shape[0], device=device_input)
    for _ in range(3):  # Reduced iterations for lower latency
        components = torch.max(adj * components, dim=1)[0]

    # Fused Cluster Extraction
    unique_ids = components.unique()
    raw_boxes = torch.stack(
        [
            torch.cat(
                [
                    raw_boxes[components == i, :2].min(0)[0],
                    raw_boxes[components == i, 2:].max(0)[0],
                ]
            )
            for i in unique_ids
        ]
    )
    return raw_boxes


def merge_boxes_cpu(boxes, gap_limit=10):
    """
    Greedy merge in 640x640 space to consolidate swarm fragments.
    Input: List of [x1, y1, x2, y2] within [0, 640]
    """
    if not boxes:
        return []

    # O(N log N) sort by X for early exit optimization
    boxes = sorted(boxes, key=lambda x: x[0])
    merged = []

    while boxes:
        curr = boxes.pop(0)
        i = 0
        while i < len(boxes):
            test = boxes[i]
            # Early exit: horizontal gap exceeds limit
            if test[0] - curr[2] > gap_limit:
                break

            # Check vertical gap
            y_dist = max(0, test[1] - curr[3], curr[3] - test[1])
            if y_dist <= gap_limit:
                # Expand curr box to include test
                curr = [
                    min(curr[0], test[0]),
                    min(curr[1], test[1]),
                    max(curr[2], test[2]),
                    max(curr[3], test[3]),
                ]
                boxes.pop(i)
                i = 0  # Re-check boundaries
            else:
                i += 1
        merged.append(curr)
    return merged


def get_freest_gpu():
    # Queries free memory from nvidia-smi
    command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
    memory_free = [
        int(x)
        for x in subprocess.check_output(command.split()).decode("ascii").split("\n")
        if x
    ]

    # Return index of GPU with maximum free memory
    return memory_free.index(max(memory_free))


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

DEBUG_FLAG_DEFAULT = True if DEBUG_DEFAULT == "1" else False

LOCKTIMEOUT_RETRIES = 5


class PipelineConfig:
    def __init__(self, **kwargs):
        # Fallback to env var if not explicitly passed

        # GENERAL
        self.CODE_DIR = kwargs.get("CODE_DIR", CODE_DIR_DEFAULT)
        self.CUSTOM_MODEL_FLAG = str2bool(
            kwargs.get("CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT)
        )
        self.DEBUG = kwargs.get("DEBUG", DEBUG_DEFAULT)
        self.DEBUG_FRAME_LIMIT = int(kwargs.get("DEBUG_FRAME_LIMIT", 100))
        self.DEVICE = kwargs.get("DEVICE", DEVICE_DEFAULT)
        self.MAX_WORKERS = int(kwargs.get("MAX_WORKERS", MAX_WORKERS))
        self.OMIT_DETECTIONS_FLAG = str2bool(
            kwargs.get("OMIT_DETECTIONS_FLAG", OMIT_DETECTIONS_FLAG_DEFAULT)
        )
        self.SHARED_OUTPUT = kwargs.get("SHARED_OUTPUT", SHARED_OUTPUT_DEFAULT)
        self.TEST_MODE = str2bool(kwargs.get("TEST_MODE", TEST_MODE_DEFAULT))
        self.TMP_LOCATION = kwargs.get("TMP_LOCATION", TMP_LOCATION_DEFAULT)

        # VIDEO WRITER
        CLIP_DURATION = kwargs.get("CLIP_DURATION", CLIP_DURATION_DEFAULT)
        target_fps = kwargs.get("TARGET_FPS", TARGET_FPS)
        self.CLIP_DURATION = (
            None if CLIP_DURATION in ["None", None] else float(CLIP_DURATION)
        )
        self.TARGET_FPS = None if target_fps in [None, "None"] else float(target_fps)

        # VDMS
        self.DBHOST = kwargs.get("DBHOST", DBHOST_DEFAULT)
        self.DBPORT = int(kwargs.get("DBPORT", DBPORT_DEFAULT))
        self.ENABLE_QUERYING = str2bool(kwargs.get("ENABLE_QUERYING", False))
        self.INGESTION = kwargs.get("INGESTION", INGESTION_DEFAULT)
        self.UDF_HOST = kwargs.get("UDF_HOST", UDF_HOST_DEFAULT)
        self.UDF_PORT = int(kwargs.get("UDF_PORT", UDF_PORT_DEFAULT))

        # MODEL
        self.DETECTION_THRESHOLD = float(
            kwargs.get("DETECTION_THRESHOLD", DETECTION_THRESHOLD_DEFAULT)
        )
        self.MAX_DETECTIONS = int(kwargs.get("MAX_DETECTIONS", MAX_DETECTIONS))
        self.MODEL_H = int(kwargs.get("MODEL_H", MODEL_H))
        self.MODEL_W = int(kwargs.get("MODEL_W", MODEL_W))
        self.MODEL_MAX_BATCH_SIZE = int(
            kwargs.get("MODEL_MAX_BATCH_SIZE", MODEL_MAX_BATCH_SIZE)
        )
        self.MODEL_NAME = kwargs.get("MODEL_NAME", MODEL_NAME_DEFAULT)
        self.MODEL_PRECISION = kwargs.get("MODEL_PRECISION", MODEL_PRECISION)
        self.SHARED_MODEL = kwargs.get("SHARED_MODEL", False)

        # PIPELINE
        self.DISABLE_DETECTION = kwargs.get("DISABLE_DETECTION", False)
        self.SMART_FILTERING_PIXEL_CONSTRAINT = SMART_FILTERING_PIXEL_CONSTRAINT
        self.BKGD_SUB_INCLUDE_HISTORY = BKGD_SUB_INCLUDE_HISTORY
        self.BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE = (
            BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE
        )
        self.BKGD_SUB_INCLUDE_HISTORY_METHOD = BKGD_SUB_INCLUDE_HISTORY_METHOD
        self.BKGD_SUB_MOG2_DETECTSHADOWS = BKGD_SUB_MOG2_DETECTSHADOWS
        self.BKGD_SUB_MOG2_HISTORY = BKGD_SUB_MOG2_HISTORY
        self.BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE = (
            BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE
        )
        self.BKGD_SUB_MOG2_LR = BKGD_SUB_MOG2_LR
        self.BKGD_SUB_MOG2_VARTHRESHOLD = BKGD_SUB_MOG2_VARTHRESHOLD
        self.DILATE_KERNEL_SIZE = DILATE_KERNEL_SIZE
        self.RESIZE_FLAG = str2bool(kwargs.get("RESIZE_FLAG", RESIZE_FLAG_DEFAULT))
        self.ROI_BB_FULL_RES_PADDING = int(
            kwargs.get("ROI_BB_FULL_RES_PADDING", ROI_BB_FULL_RES_PADDING)
        )
        self.ROI_MAX_RELATIVE_SIZE_RATIO = float(
            kwargs.get("ROI_MAX_RELATIVE_SIZE_RATIO", ROI_MAX_RELATIVE_SIZE_RATIO)
        )
        self.ROI_MERGE_SIZE_LIMIT = int(
            kwargs.get("ROI_MERGE_SIZE_LIMIT", ROI_MERGE_SIZE_LIMIT)
        )
        self.ROI_MIN_AREA_RATIO = ROI_MIN_AREA_RATIO
        self.ROI_DISTANCE_THRESH_RATIO = ROI_DISTANCE_THRESH_RATIO
        self.ROI_CONTAINMENT_THRESH = ROI_CONTAINMENT_THRESH
        self.ROI_RETURN_BYTES = str2bool(kwargs.get("ROI_RETURN_BYTES", True))
        self.THRESHOLD_MAX_VALUE = int(
            kwargs.get("THRESHOLD_MAX_VALUE", THRESHOLD_MAX_VALUE)
        )
        self.THRESHOLD_VALUE = int(kwargs.get("THRESHOLD_VALUE", THRESHOLD_VALUE))

        # VISUALIZATION
        self.DETECTION_TYPE = kwargs.get("DETECTION_TYPE", DETECTION_TYPE_DEFAULT)
        self.DISPLAY_FRAME_QUALITY = int(
            kwargs.get("DISPLAY_FRAME_QUALITY", DISPLAY_FRAME_QUALITY)
        )
        self.DISPLAY_FRAME_SIZE = kwargs.get("DISPLAY_FRAME_SIZE", DISPLAY_FRAME_SIZE)
        self.THICKNESS = int(kwargs.get("THICKNESS", THICKNESS))

        # VARS WITH DEPENDENCIES
        Path(self.SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
        self.device_input = self.DEVICE.lower() if self.DEVICE == "CPU" else "cuda"
        self.DEBUG_FLAG = True if self.DEBUG == "1" else False

        if self.DETECTION_TYPE == "motion" and self.ENABLE_QUERYING:
            self.ENABLE_QUERYING = False
            self.DISPLAY_FRAME_QUALITY = 100
            self.THICKNESS = 10

        self.sf_enabled = kwargs.get("SMART_FILTERING_ENABLED", SMART_FILTERING_ENABLED)
        if self.CUSTOM_MODEL_FLAG:
            self.model_path = f"{self.CODE_DIR}/resources/models/ultralytics/custom_models/{self.MODEL_NAME}"
        else:
            self.model_path = f"{self.CODE_DIR}/resources/models/ultralytics/{self.MODEL_NAME}/{self.MODEL_PRECISION}/{self.MODEL_NAME}"

        if not self.sf_enabled:
            self.model_path += "_noSF"

        if self.DEVICE == "GPU":
            self.model_path += ".engine"

            # Force PyTorch to initialize the CUDA context
            if torch.cuda.is_available():
                best_gpu_index = get_freest_gpu()
                os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_index)
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
        else:
            self.model_path += "_openvino_model/"


class VDMSPool:
    def __init__(self, host, port, size=5):
        self.host = host
        self.port = port
        self.size = size
        self.pool = queue.Queue(maxsize=size)
        self.populate()

    def populate(self):
        # Pre-populate the pool with authenticated connections
        for _ in range(self.size):
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


# device_input_DEFAULT = DEVICE_DEFAULT.lower() if DEVICE_DEFAULT == "CPU" else "cuda"
# Path(SHARED_OUTPUT_DEFAULT).mkdir(parents=True, exist_ok=True)

# VDMS_POOL = None

ERR_KEYWORDS = [
    "timeout",
    "null search iterator",
    "outoftransactions",
    "internal server",
]


# Plot variables
THICKNESS_SCALE_FACTOR = 1e-3
FONT_SCALE_FACTOR = 1e-3


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


# if DEVICE == "GPU":
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

# This CUDA C++ code finds min/max for all labels in ONE pass over the mask
DETECTION_ACCEL_KERNEL = cupy.RawKernel(
    r"""
    extern "C" __global__
    void fast_detect(const unsigned char* bgs_mask, unsigned char* out_mask, int pitch, int w, int h, float thresh) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x > 0 && x < w-1 && y > 0 && y < h-1) {
            unsigned char val = bgs_mask[y * pitch + x];
            unsigned char res = (val > thresh) ? 255 : 0;
            if (res == 0) {
                if (bgs_mask[(y-1)*pitch + x] > thresh || bgs_mask[(y+1)*pitch + x] > thresh ||
                    bgs_mask[y*pitch + (x-1)] > thresh || bgs_mask[y*pitch + (x+1)] > thresh) {
                    res = 255;
                }
            }
            out_mask[y * pitch + x] = res;
        }
    }
    """,
    "fast_detect",
)
# DETECTION_ACCEL_KERNEL = cupy.RawKernel(
#     r"""
# extern "C" __global__
# void fast_detect(const unsigned char* bgs_mask, unsigned char* out_mask, int pitch, int w, int h, float thresh) {
#     int x = blockIdx.x * blockDim.x + threadIdx.x;
#     int y = blockIdx.y * blockDim.y + threadIdx.y;

#     if (x >= 2 && x < w-2 && y >= 2 && y < h-2) {
#         unsigned char val = bgs_mask[y * pitch + x];
#         unsigned char res = (val > thresh) ? 255 : 0;

#         if (res == 0) {
#             bool found = false;
#             for (int dy = -2; dy <= 2; dy++) {
#                 for (int dx = -2; dx <= 2; dx++) {
#                     // CIRCULAR CHECK: Skip the far corners of the 5x5 square
#                     // This mimics cv2.MORPH_ELLIPSE and prevents over-merging
#                     if (abs(dx) == 2 && abs(dy) == 2) continue;

#                     if (bgs_mask[(y + dy) * pitch + (x + dx)] > thresh) {
#                         found = true;
#                         break;
#                     }
#                 }
#                 if (found) break;
#             }
#             if (found) res = 255;
#         }
#         out_mask[y * pitch + x] = res;
#     }
# }
# """,
#     "fast_detect",
# )

# Fused Kernel: Single-pass Bounding Box Extraction with Stride Support
BOUNDS_KERNEL = cupy.RawKernel(
    r"""
extern "C" __global__
void find_bounds(const unsigned char* labeled_ptr, int step, int width, int height, int num_labels, int* x1, int* y1, int* x2, int* y2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int* row = (const int*)(labeled_ptr + y * step);
        int label = row[x];
        if (label > 0 && label <= num_labels) {
            atomicMin(&x1[label], x);
            atomicMin(&y1[label], y);
            atomicMax(&x2[label], x);
            atomicMax(&y2[label], y);
        }
    }
}
""",
    "find_bounds",
)


# def tensor2opencv(frame_raw, device_input, is_bgr=True):
#     # GPU Path (Always RGB -> needs swap to BGR)
#     if device_input == "cuda":
#         if hasattr(frame_raw, "permute"):
#             # [C, H, W] -> [H, W, C]
#             frame_cpu = (
#                 frame_raw.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
#             )
#         else:
#             frame_cpu = np.ascontiguousarray(frame_raw, dtype=np.uint8)

#         if not is_bgr:
#             # GPU reader outputs RGB, so we MUST swap to BGR for VideoWriter
#             frame_cpu = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)

#     # Handle CPU Path (Check if swap is needed)
#     else:
#         frame_cpu = np.ascontiguousarray(frame_raw, dtype=np.uint8)
#         if frame_cpu.shape[0] == 3:
#             frame_cpu = frame_cpu.transpose(1, 2, 0)

#         if not is_bgr:
#             frame_cpu = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)
#     return frame_cpu


def tensor2opencv(frame_source, device_input, is_bgr=True, resize_h=640, resize_w=640):
    if torch.is_tensor(frame_source):
        # .contiguous() is CRITICAL here to fix the "shredded" look
        temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
        img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
    elif hasattr(frame_source, "download"):
        img_cpu = frame_source.download()
    else:
        img_cpu = np.ascontiguousarray(frame_source)

    #  Fix Shape: restore spatial grid if flattened
    if img_cpu.ndim == 3 and img_cpu.shape[0] == 1:
        img_cpu = img_cpu.reshape((resize_h, resize_w, 3))

    #  Fix Visibility: ONLY multiply if it's actually floating point
    # If uint8 is multiplied by 255, it wraps around and creates "neon" colors
    if img_cpu.dtype != np.uint8:
        if img_cpu.max() <= 1.0:
            img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
        else:
            img_cpu = img_cpu.astype(np.uint8)

    # Color Space: Standardize to BGR for imwrite
    if not is_bgr:
        if len(img_cpu.shape) == 3:
            # Swap RGB (Torch/Decoder) -> BGR (OpenCV)
            # img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
            # Only swap if the source is RGB (GPU Path)
            # CPU path is already BGR from OpenCV reader
            if device_input == "cuda":
                img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
            else:
                # Ensure it's contiguous for saving
                img_cpu = np.ascontiguousarray(img_cpu)
        else:
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)

    return img_cpu


def gpumat2cupy(gpu_mat):
    """Bridge OpenCV GpuMat to CuPy without copying data."""
    # Get properties from GpuMat
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

    # Map OpenCV types to CuPy typestrs
    # CV_8UC1 is 'u1' (unsigned 1-byte), etc.
    # type_map = {cv2.CV_8U: "|u1", cv2.CV_32F: "<f4", cv2.CV_8UC1: "|u1"}

    # Create the __cuda_array_interface__ dictionary
    # This tells CuPy where the data is and how it's shaped
    if_dict = {
        "version": 3,
        "shape": shape,
        "typestr": "|u1",
        # "descr": [("", type_map.get(gpu_mat.type(), "|u1"))],
        "data": (gpu_mat.cudaPtr(), False),  # (Pointer, Read-only)
        "strides": strides,
    }

    # Create a dummy object with the interface and wrap it in CuPy
    class Holder:
        pass

    holder = Holder()
    holder.__cuda_array_interface__ = if_dict
    return cupy.asarray(holder)


def torch2gpumat(tensor):
    """
    Creates an OpenCV GpuMat pointing to the same memory as a PyTorch tensor.
    ZERO-COPY: No data is moved; only the memory address is shared.
    """
    # Ensure tensor is [H, W, C] and contiguous for OpenCV
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0).contiguous()

    # Bridge to CuPy (Zero-Copy)
    cp_arr = cupy.asanyarray(tensor)

    # Wrap in GpuMat
    # cv2.CV_8UC3 for uint8, CV_32FC3 for float
    dtype = cv2.CV_8UC3 if tensor.dtype == torch.uint8 else cv2.CV_32FC3

    gpumat = cv2.cuda_GpuMat(
        tensor.shape[1],  # Width
        tensor.shape[0],  # Height
        dtype,
        cp_arr.data.ptr,
    )
    return gpumat


# This kernel calculates the [x1, y1, x2, y2] for every detected object label
BOUNDS_KERNEL_CODE = r"""
extern "C" __global__
void get_bounds(const int* labeled, int pitch, int w, int h, int num_labels,
                int* x1, int* y1, int* x2, int* y2) {
    // Calculate the unique 2D pixel coordinates (x, y) for this specific thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary Check: Ensure the thread isn't trying to read outside the image dimensions
    if (x < w && y < h) {
        // Use the pitch to correctly calculate the memory offset
        int label = labeled[y * pitch + x];

        if (label > 0 && label <= num_labels) {
            // Atomically update the bounding box for this specific label
            atomicMin(&x1[label], x);
            atomicMin(&y1[label], y);
            // +1 ensures the box captures the full pixel and matches OpenCV ROI logic
            atomicMax(&x2[label], x + 1);
            atomicMax(&y2[label], y + 1);
        }
    }
}
"""

# Compile the kernel once
get_bounds_kernel = cupy.RawKernel(BOUNDS_KERNEL_CODE, "get_bounds")


def find_contours_gpu_equivalent(mask_gpu_mat, stream=None):
    """
    GPU equivalent to cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Returns: torch.Tensor [N, 4] containing (x1, y1, x2, y2) in analysis space.
    """
    # Bridge OpenCV GpuMat to CuPy (Zero-Copy)
    w, h = mask_gpu_mat.size()
    ptr = mask_gpu_mat.cudaPtr()
    pitch_bytes = mask_gpu_mat.step

    mask_cp = cupy.ndarray(
        (h, w),
        dtype=cupy.uint8,
        memptr=cupy.cuda.MemoryPointer(
            cupy.cuda.UnownedMemory(ptr, pitch_bytes * h, mask_gpu_mat), 0
        ),
        strides=(pitch_bytes, 1),
    )

    # Use the stream pointer if provided, otherwise default to 0 (Null Stream)
    stream_ptr = stream.cudaPtr() if stream else 0

    with cupy.cuda.ExternalStream(stream_ptr):
        # Labeling (Equivalent to finding connected components)
        # labeled is an int32 array where every 'blob' has a unique number
        structure = cupy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        labeled, num_labels = cupyx.scipy.ndimage.label(mask_cp, structure=structure)

        if num_labels == 0:
            return torch.empty((0, 4), device="cuda")

        # Setup Bounding Box Buffers
        x1 = cupy.full((num_labels + 1,), w, dtype=cupy.int32)
        y1 = cupy.full((num_labels + 1,), h, dtype=cupy.int32)
        x2 = cupy.full((num_labels + 1,), -1, dtype=cupy.int32)
        y2 = cupy.full((num_labels + 1,), -1, dtype=cupy.int32)

        # Run the Bounds Kernel
        # IMPORTANT: Use labeled.strides[0]//4 to get the pitch in elements
        pitch_elements = labeled.strides[0] // 4
        tpb = (16, 16)
        bpg = ((w + tpb[0] - 1) // tpb[0], (h + tpb[1] - 1) // tpb[1])

        get_bounds_kernel(
            bpg, tpb, (labeled, pitch_elements, w, h, num_labels, x1, y1, x2, y2)
        )

    # Stack and return as Torch Tensor for YOLO/Drawing
    # We skip index 0 as it represents the background (black)
    # boxes = torch.stack(
    #     [
    #         torch.as_tensor(x1[1:], device="cuda"),
    #         torch.as_tensor(y1[1:], device="cuda"),
    #         torch.as_tensor(x2[1:], device="cuda"),
    #         torch.as_tensor(y2[1:], device="cuda"),
    #     ],
    #     dim=1,
    # ).float()
    boxes = cupy.column_stack((x1[1:], y1[1:], x2[1:], y2[1:]))

    return torch.as_tensor(boxes, device="cuda").float()


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
    query,
    local_db=None,
    num_retries: int = LOCKTIMEOUT_RETRIES,
    sleep_timer: int = 0,
    DBHOST=DBHOST_DEFAULT,
    DBPORT=DBPORT_DEFAULT,
    DEBUG_FLAG=DEBUG_FLAG_DEFAULT,
):
    # global db
    db = local_db if local_db else vdms.vdms().connect(DBHOST, DBPORT)
    for ridx in range(num_retries + 1):
        response, _ = db.query(query, [[]])
        if "FailedCommand" in response[0] and any(
            k in response[0]["info"].lower() for k in ERR_KEYWORDS
        ):
            err = response[0]["info"]
            if DEBUG_FLAG:
                query_type = list(query[0].keys())[0]
                print(
                    f"DEBUG [process_stream Attempt #{ridx}] Received '{err}' for {query_type} query",
                    flush=True,
                )
            if sleep_timer > 0:
                time.sleep(sleep_timer)
        else:
            if DEBUG_FLAG:
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


def get_display_frame_in_bytes(
    foi, display_size=(960, 540), quality=50, return_bytes=True, device="CPU"
):  # Expects BGR
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


# Manual FPS calculation if OpenCV reports 0
def manual_fps_calculation(src, num_frames=10):
    if isinstance(src, cv2.VideoCapture):
        vid_obj = src
    else:
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


# def nv12_to_rgb_torch(nv12_tensor, height, width):
#     """
#     Fast GPU-based NV12 to RGB conversion using PyTorch.
#     Input: [H*1.5, W] uint8 tensor on GPU
#     Output: [3, H, W] float32 tensor on GPU (Normalized 0.0-1.0)
#     """
#     # Extract Y (Luma) plane: [0:height, :]
#     y = nv12_tensor[:height, :].float()

#     # Extract UV (Chroma) plane: [height:, :]
#     # UV is interleaved: U V U V ...
#     uv = nv12_tensor[height:, :].reshape(height // 2, width // 2, 2).float()

#     # Upsample UV to match Y dimensions using Bilinear interpolation
#     # This stretches the color to match the 8K detail
#     uv_upsampled = torch.nn.functional.interpolate(
#         uv.permute(2, 0, 1).unsqueeze(0),
#         size=(height, width),
#         mode="bilinear",
#         align_corners=False,
#     ).squeeze(0)

#     u = uv_upsampled[0]
#     v = uv_upsampled[1]

#     # YUV to RGB Conversion Matrix (BT.709 for 8K/HD)
#     # Shift values to be zero-centered
#     y = (y - 16) * 1.164
#     u = u - 128
#     v = v - 128

#     r = y + 1.793 * v
#     g = y - 0.213 * u - 0.533 * v
#     b = y + 2.112 * u

#     # Stack and Clamp
#     rgb = torch.stack([r, g, b], dim=0)
#     return torch.clamp(rgb, 0, 255).byte()  # Return as [3, 4320, 7680] uint8


def rgb_to_nv12_torch(rgb_tensor):
    """
    Fast GPU conversion from RGB to NV12 using PyTorch.
    Input: [3, H, W] uint8 tensor on GPU
    Output: [H*1.5, W] uint8 tensor on GPU (NV12 format)
    """
    _, h, w = rgb_tensor.shape
    rgb = rgb_tensor.float()

    # BT.709 RGB to YUV coefficients (standard for HD/8K)
    # Y plane (Luma)
    y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    # U and V planes (Chroma)
    u = -0.1146 * rgb[0] - 0.3854 * rgb[1] + 0.5000 * rgb[2] + 128
    v = 0.5000 * rgb[0] - 0.4542 * rgb[1] - 0.0458 * rgb[2] + 128

    # Subsample Chroma (4:2:0)
    # We take every 2nd pixel to shrink U and V to half-resolution
    u_sub = u[::2, ::2]
    v_sub = v[::2, ::2]

    # Interleave U and V (NV12 requirement)
    # Reshape to [H/2, W] by placing U and V side-by-side at each pixel
    uv_interleaved = torch.stack((u_sub, v_sub), dim=2).reshape(h // 2, w)

    # Combine Y and UV planes
    # Resulting shape: [H + H/2, W] -> [1.5H, W]
    nv12 = torch.cat([y, uv_interleaved], dim=0)

    return torch.clamp(nv12, 0, 255).byte()


# Generate and run UDF query
def get_udf_query(
    filename_path,
    properties,
    ingest_mode,
    new_size,
    id="udf_metadata",
    metadata=None,
    test_mode=TEST_MODE_DEFAULT,
    local_db=None,
    UDF_HOST=UDF_HOST_DEFAULT,
    UDF_PORT=UDF_PORT_DEFAULT,
    DEBUG_FLAG=DEBUG_FLAG_DEFAULT,
    DBHOST=DBHOST_DEFAULT,
    DBPORT=DBPORT_DEFAULT,
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
        res = retry_query(
            [query],
            local_db=local_db,
            sleep_timer=randint(1, 5),
            DBHOST=DBHOST,
            DBPORT=DBPORT,
            DEBUG_FLAG=DEBUG_FLAG,
        )

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
    VDMS_POOL: VDMSPool = None,
    DEBUG_FLAG=DEBUG_FLAG_DEFAULT,
    INGESTION=INGESTION_DEFAULT,
    TEST_MODE=TEST_MODE_DEFAULT,
    UDF_HOST=UDF_HOST_DEFAULT,
    UDF_PORT=UDF_PORT_DEFAULT,
    DBHOST=DBHOST_DEFAULT,
    DBPORT=DBPORT_DEFAULT,
):
    # global VDMS_POOL

    if VDMS_POOL is None:
        # VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)
        VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)

    if DEBUG_FLAG:
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
            UDF_HOST=UDF_HOST,
            UDF_PORT=UDF_PORT,
            DEBUG_FLAG=DEBUG_FLAG,
            DBHOST=DBHOST,
            DBPORT=DBPORT,
        )

        if DEBUG_FLAG:
            print(
                f"[TIMING],end_clip_metadata,{clip_key},{time.time()}",
                flush=True,
            )
    finally:
        VDMS_POOL.return_connection(db)


# method to send metadata to VDMS once clip is saved w/ retry mechanism
def metadata2vdms_with_retry(
    clip_key,
    clip_filename,
    clip_metadata,
    width,
    height,
    max_retries=LOCKTIMEOUT_RETRIES,
    VDMS_POOL: VDMSPool = None,
    DEBUG_FLAG=DEBUG_FLAG_DEFAULT,
    INGESTION=INGESTION_DEFAULT,
    TEST_MODE=TEST_MODE_DEFAULT,
    UDF_HOST=UDF_HOST_DEFAULT,
    UDF_PORT=UDF_PORT_DEFAULT,
    DBHOST=DBHOST_DEFAULT,
    DBPORT=DBPORT_DEFAULT,
):
    """
    Attempts to send metadata to VDMS with exponential backoff.
    """
    if VDMS_POOL is None:
        # VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)
        VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Attempt the actual upload (using your existing utility)
            success = metadata2vdms(
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
            if success:
                print(f" [VDMS] Successfully uploaded {clip_key}")
                return True
        except Exception as e:
            retry_count += 1
            wait_time = 2**retry_count  # 2s, 4s, 8s, 16s...
            print(
                f" [RETRY] VDMS upload failed for {clip_key} (Attempt {retry_count}/{max_retries}). "
                f"Retrying in {wait_time}s... Error: {e}"
            )
            time.sleep(wait_time)

    print(f" [FAILED] Could not send {clip_key} to VDMS after {max_retries} attempts.")
    return False


def merge_boxes_limit(boxes, dist_threshold=25, min_area=32, max_size=640):
    if len(boxes) == 0:
        return []

    # EARLY FILTER: Remove noise (dots/specks) immediately
    # area = width * height
    valid_boxes = []
    for b in boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        if (w * h) >= min_area:
            valid_boxes.append(list(b))

    boxes = valid_boxes
    merged_any = True

    while merged_any:
        merged_any = False
        new_boxes = []

        while boxes:
            current = boxes.pop(0)
            # has_merged = False

            for i, other in enumerate(boxes):
                # Check Proximity: Are they close enough to consider?
                # (Expanding 'current' by distance_threshold for the check)
                if not (
                    current[2] + dist_threshold < other[0]
                    or other[2] + dist_threshold < current[0]
                    or current[3] + dist_threshold < other[1]
                    or other[3] + dist_threshold < current[1]
                ):
                    # Potential Dimensions: Calculate what the new box would be
                    new_x1 = min(current[0], other[0])
                    new_y1 = min(current[1], other[1])
                    new_x2 = max(current[2], other[2])
                    new_y2 = max(current[3], other[3])

                    new_w = new_x2 - new_x1
                    new_h = new_y2 - new_y1

                    # Size Constraint: Only merge if it doesn't exceed the limit
                    if new_w <= max_size and new_h <= max_size:
                        current = [new_x1, new_y1, new_x2, new_y2]
                        boxes.pop(i)
                        # has_merged = True
                        merged_any = True
                        break

            new_boxes.append(current)
        boxes = new_boxes

    return boxes


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
