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

import cv2
import numpy as np

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
UDF_PORT = 5011
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
RESIZE_FLAG = str2bool(os.getenv("RESIZE_FLAG", False))
SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
Path(SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
TEST_MODE = str2bool(os.getenv("TEST_FLAG", False))
TMP_LOCATION = os.getenv("TMP_LOCATION", "/var/www/cache/")
UDF_HOST = os.getenv("UDF_HOST", "fastapi-service")

if DEVICE == "GPU":
    EXPORT_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[!] USING GPU")
else:
    EXPORT_BATCH_SIZE = int(os.environ.get("CPU_BATCH_SIZE", 1))  # 8
    print("[!] USING CPU")

if not TEST_MODE:
    db = vdms.vdms()
    db.connect(DBHOST, DBPORT)

LOCKTIMEOUT_RETRIES = 5
ERR_KEYWORDS = [
    "timeout",
    "null search iterator",
    "outoftransactions",
    "internal server",
]


def retry_query(query, num_retries: int = LOCKTIMEOUT_RETRIES, sleep_timer: int = 0):
    global db
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


# # This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
# def download_file(url, download_to: Path, expected_size=None):
#     # Don't download the file twice.
#     # (If possible, verify the download using the file length.)
#     if download_to.exists():
#         if expected_size:
#             if download_to.stat().st_size == expected_size:
#                 return
#         else:
#             st.info(f"{url} is already downloaded.")
#             if not st.button("Download again?"):
#                 return

#     download_to.parent.mkdir(parents=True, exist_ok=True)

#     # These are handles to two visual elements to animate.
#     weights_warning, progress_bar = None, None
#     try:
#         weights_warning = st.warning("Downloading %s..." % url)
#         progress_bar = st.progress(0)
#         with open(download_to, "wb") as output_file:
#             with urllib.request.urlopen(url) as response:
#                 length = int(response.info()["Content-Length"])
#                 counter = 0.0
#                 MEGABYTES = 2.0**20.0
#                 while True:
#                     data = response.read(8192)
#                     if not data:
#                         break
#                     counter += len(data)
#                     output_file.write(data)

#                     # We perform animation by overwriting the elements.
#                     weights_warning.warning(
#                         "Downloading %s... (%6.2f/%6.2f MB)"
#                         % (url, counter / MEGABYTES, length / MEGABYTES)
#                     )
#                     progress_bar.progress(min(counter / length, 1.0))
#     # Finally, we remove these visual elements by calling .empty().
#     finally:
# if weights_warning is not None:
#     weights_warning.empty()
# if progress_bar is not None:
#     progress_bar.empty()


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


# def retry_query(query, num_retries: int = LOCKTIMEOUT_RETRIES, sleep_timer: int = 0):
#     global db
#     for ridx in range(num_retries + 1):
#         response, _ = db.query(query, [[]])
#         if "FailedCommand" in response[0] and any(
#             k in response[0]["info"].lower() for k in ERR_KEYWORDS
#         ):
#             err = response[0]["info"]
#             if DEBUG == "1":
#                 query_type = list(query[0].keys())[0]
#                 print(
#                     f"DEBUG [process_stream Attempt #{ridx}] Received '{err}' for {query_type} query",
#                     flush=True,
#                 )
#             if sleep_timer > 0:
#                 time.sleep(sleep_timer)
#         else:
#             if DEBUG == "1":
#                 print(
#                     f"[DEBUG process_stream] Successful query response: {response}",
#                     flush=True,
#                 )
#             break  # Continue
#     return response


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


# """
# VDMS RELATED FUNCTIONS
# """


# def vdms_connection_status(dbhost: str = "localhost", dbport: int = 55555):
#     db = vdms.vdms()
#     availability_status = False
#     try:
#         availability_status = db.connect(dbhost, int(dbport))
#     except Exception:
#         pass
#     return availability_status


# def initialize_vdms_df():
#     st.session_state.vdms_instance_df = pd.DataFrame(
#         columns=["Container Name", "Hostname", "Port", "Status"]
#     )


# def search_for_vdms_instances():
#     if st.button("Search for VDMS demo instances"):
#         with st.spinner("Processing..."):
#             info_str = get_vdms_instances()

#             st.info(info_str)


# def get_docker_status(dbhost="localhost", dbport=55555):
#     try:
#         get_containers_cmd = 'docker ps --filter name=vdms_log_pipeline --format "table {{.ID}}\t{{.Names}}\t{{.Ports}}"'
#         output = subprocess.check_output(get_containers_cmd, shell=True)
#         condition = "vdms_log_pipeline" in output.decode("utf-8")
#     except Exception:
#         db = vdms.vdms()
#         condition = db.connect(dbhost, dbport)
#     if condition:
#         return "Deployed"
#     else:
#         return "Not Deployed"


# def get_vdms_instances():
#     get_containers_cmd = 'docker ps --filter name=vdms_.*_demo_test --format "table {{.ID}}\t{{.Names}}\t{{.Ports}}"'
#     output = subprocess.check_output(get_containers_cmd, shell=True)

#     initialize_vdms_df()

#     lines = [line for line in output.decode("utf-8").split("\n")[1:] if line]
#     line_names = ["ID", "Container Name", "Ports"]
#     new_instances = 0
#     for line in lines:
#         line_split = line.split()
#         line_data = line_split[:2] + ["".join(line_split[2:])]

#         name = line_data[line_names.index("Container Name")]
#         hostname = "localhost"
#         port_str = line_data[line_names.index("Ports")]
#         end_idx = port_str.find("->55555/tcp")
#         port = port_str[port_str.find(":") + 1 : end_idx]
#         availability_status = vdms_connection_status(hostname, int(port))

#         new_data = pd.DataFrame(
#             {
#                 "Container Name": [name],
#                 "Hostname": [hostname],
#                 "Port": [port],
#                 "Status": ["Connected" if availability_status else "Not Available"],
#             }
#         )
#         st.session_state.vdms_instance_df = pd.concat(
#             [st.session_state.vdms_instance_df, new_data], ignore_index=True
#         )
#         new_instances += 1

#     info_str = f"{new_instances} VDMS instances found"

#     st.session_state.vdms_instance_df.sort_values(
#         by=["Container Name"], inplace=True, ignore_index=True
#     )
#     st.session_state.vdms_instance_df.drop_duplicates(inplace=True, ignore_index=True)

#     return info_str


# def add_vdms_instance_buttons(vdms_details):
#     dbhost = vdms_details[1]
#     dbport = vdms_details[2]

#     left_column, right_column = st.columns([0.5, 0.5])

#     right_column.checkbox("Kill & Restart if DB exists", key="Kill_restart")

#     if left_column.form_submit_button("Add", use_container_width=True):
#         if all([arg != "" for arg in [dbhost, dbport]]):
#             matching_idx = st.session_state.vdms_instance_df.loc[
#                 (st.session_state.vdms_instance_df["Hostname"] == dbhost)
#                 & (st.session_state.vdms_instance_df["Port"] == dbport)
#             ].index.tolist()

#             if not st.session_state.Kill_restart and len(matching_idx) != 0:
#                 vdms_method = "Use existing DB"
#             else:
#                 vdms_method = "Fresh DB"

#             if len(matching_idx) > 0:
#                 instance_num = matching_idx[0]
#                 container_name = st.session_state.vdms_instance_df.at[
#                     instance_num, "Container Name"
#                 ]

#                 # Start instance locally
#                 _ = start_vdms_docker(
#                     container_name=container_name,
#                     project_path=PROJECT_PATH,
#                     dbport=int(dbport),
#                     vdms_method=vdms_method,
#                 )
#                 info_str = (
#                     f"Instance #{instance_num} ({container_name}) already running"
#                 )
#                 if vdms_method == "Fresh DB":
#                     info_str += " but redeploying"
#                 st.info(info_str)

#             else:
#                 instance_num = st.session_state.vdms_instance_df.shape[0] + 1
#                 container_name = f"vdms_{instance_num}_demo_test"

#                 # Start instance locally
#                 _ = start_vdms_docker(
#                     container_name=container_name,
#                     project_path=PROJECT_PATH,
#                     dbport=int(dbport),
#                     vdms_method=vdms_method,
#                 )
#                 vdms_details[0] = container_name

#                 # Check is available for connection
#                 availability_status = vdms_connection_status(dbhost, int(dbport))
#                 vdms_details[-1] = (
#                     "Connected" if availability_status else "Not Available"
#                 )

#                 if vdms_details[-1] == "Connected":
#                     st.session_state.vdms_instance_df.loc[instance_num] = vdms_details
#                     info_str = f"Instance #{instance_num} ({container_name}) added"
#                     st.info(info_str)
#                 else:
#                     st.error("Cannot connect to instance; Check server")

#         else:
#             st.error("Must provide VDMS Port")

#     st.session_state.vdms_instance_df.sort_values(
#         by=["Container Name"], inplace=True, ignore_index=True
#     )
#     st.session_state.vdms_instance_df.drop_duplicates(inplace=True, ignore_index=True)


# def add_vdms_instance():
#     col_count = st.session_state.vdms_instance_df.shape[1]

#     with st.form(key="add form", clear_on_submit=True):
#         cols = st.columns(1)
#         vdms_details = []

#         col_idx = 0
#         for i in range(col_count):
#             value = ""
#             if st.session_state.vdms_instance_df.columns[i] in ["Hostname", "Port"]:
#                 if st.session_state.vdms_instance_df.columns[i] == "Hostname":
#                     # Only local deployment supported for demo
#                     value = "localhost"

#                 if st.session_state.vdms_instance_df.columns[i] == "Port":
#                     value = str(
#                         cols[col_idx].text_input(
#                             st.session_state.vdms_instance_df.columns[i]
#                         )
#                     )

#             vdms_details.append(value)

#         add_vdms_instance_buttons(vdms_details)


# def populate_vdms_instances():
#     if "vdms_instance_df" not in st.session_state:
#         initialize_vdms_df()

#     st.markdown("1. If instances already deployed, search for them below.")
#     search_for_vdms_instances()
#     st.markdown("\n\n")

#     st.markdown("2. Provide local port to deploy instance.")
#     add_vdms_instance()
#     st.markdown("\n\n")

#     st.markdown("### VDMS Instances")
#     st.dataframe(st.session_state.vdms_instance_df, use_container_width=True)


MASK_THRESHOLD_VALUE = 127
MASK_MAX_VALUE = 255
MAX_DETECTIONS = 100

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


@dataclass
class PipelineMapping:
    resize_device: str.lower = "cpu"
    bkgd_subtraction_device: str.lower = "cpu"
    threshold_device: str.lower = "cpu"
    erodeAndDilate_device: str.lower = "cpu"
    contour_device: str.lower = "cpu"
    detection_device: str.lower = "cpu"


def merge_boxes_limit(bbs_full_res, dist_threshold=50, size_limit=640):
    """
    boxes: list of [x1, y1, x2, y2]
    dist_threshold: max distance between boxes to consider them 'connected'
    size_limit: max width/height for a merged box
    """
    if len(bbs_full_res) == 0:
        return []

    rects = np.array(bbs_full_res)
    num_boxes = len(rects)
    parent = list(range(num_boxes))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            # Check if merging exceeds size limit
            temp_x1 = min(rects[root_i][0], rects[root_j][0])
            temp_y1 = min(rects[root_i][1], rects[root_j][1])
            temp_x2 = max(rects[root_i][2], rects[root_j][2])
            temp_y2 = max(rects[root_i][3], rects[root_j][3])

            if (temp_x2 - temp_x1 <= size_limit) and (temp_y2 - temp_y1 <= size_limit):
                parent[root_i] = root_j
                # Update the root rectangle to the new merged bounds
                rects[root_j] = [temp_x1, temp_y1, temp_x2, temp_y2]

    # 2. Compare boxes (Optimized: only check nearby ones if sorted by X)
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            # Proximity check (Manhattan distance or check if boxes are 'close')
            dx = max(0, max(rects[i][0], rects[j][0]) - min(rects[i][2], rects[j][2]))
            dy = max(0, max(rects[i][1], rects[j][1]) - min(rects[i][3], rects[j][3]))

            if dx < dist_threshold and dy < dist_threshold:
                union(i, j)

    # 3. Extract unique merged boxes
    final_boxes = []
    unique_roots = set()
    for i in range(num_boxes):
        root = find(i)
        if root not in unique_roots:
            unique_roots.add(root)
            final_boxes.append(rects[root])

    return final_boxes


def filter_contained_boxes(boxes, containment_thresh=0.90):
    """
    Deletes redundant boxes that are mostly inside another larger box.
    """
    if not boxes:
        return []

    # 1. Sort by area (Largest boxes first)
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep = []

    for child in boxes:
        is_contained = False
        for parent in keep:
            # Intersection coordinates
            ix1, iy1 = max(child[0], parent[0]), max(child[1], parent[1])
            ix2, iy2 = min(child[2], parent[2]), min(child[3], parent[3])

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                child_area = (child[2] - child[0]) * (child[3] - child[1])

                # If child is 90% inside a larger box, it's redundant
                if inter_area / child_area >= containment_thresh:
                    is_contained = True
                    break

        if not is_contained:
            keep.append(child)

    return keep


# from random import randint
# # Generate and run UDF query
# def get_udf_query(
#     filename_path,
#     properties,
#     ingest_mode,
#     new_size,
#     id="udf_metadata",
#     metadata=None,
#     test_mode=TEST_MODE,
# ):
#     query = {
#         "AddVideo": {
#             "from_file_path": str(filename_path),  # from_server_file
#             "is_local_file": True,
#             "properties": properties,
#             "operations": [
#                 {
#                     "type": "syncremoteOp",  # "remoteOp",
#                     "url": f"http://{UDF_HOST}:{UDF_PORT}/video",
#                     "options": {
#                         "id": id,
#                         "otype": ingest_mode,
#                         "media_type": "video",
#                         "input_sizeWH": new_size,
#                         "filename": properties["Name"],
#                         "ingestion": 1,
#                     },
#                 }
#             ],
#         }
#     }

#     if id == "udf_metadata" and metadata is not None:
#         query["AddVideo"]["operations"][0]["options"]["metadata"] = metadata

#     if test_mode:
#         return

#     filename = str(Path(filename_path).name)
#     if DEBUG_FLAG:
#         print(
#             f"[TIMING],start_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
#             flush=True,
#         )
#     try:
#         res = retry_query([query], sleep_timer=randint(1, 5))

#         if DEBUG_FLAG:
#             print(
#                 f"[TIMING],end_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
#                 flush=True,
#             )
#             print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
#             print(f"[DEBUG] {filename} INGEST_VIDEO RESPONSE: {res}", flush=True)
#     except Exception:
#         e = traceback.format_exc()
#         print(f"[DEBUG] VDMS Query Exception: {e}", flush=True)

#     # elapsed_time = time.time() - start_t

#     # db.disconnect()
#     # del db


# def _sort_dict_by_frame(in_dict):
#     def _by_int(key):
#         return tuple(int(k) for k in key.split("_"))

#     return dict(sorted(in_dict.items(), key=lambda x: _by_int(x[0])))


# # method to send metadata to VDMS once clip is saved
# def metadata2vdms(
#     clip_key,
#     clip_filename,
#     clip_metadata,
#     width,
#     height,
# ):
#     if DEBUG == "1":
#         print(
#             f"[TIMING],start_clip_metadata,{clip_key},{time.time()}",
#             flush=True,
#         )

#     # Send metadata to UDF
#     properties = {
#         "Name": clip_key,  # .split("/")[-1],
#         "category": "video_path_rop",
#     }

#     combined_metadata = clip_metadata["object"] if "object" in clip_metadata else {}
#     if "face" in clip_metadata:
#         for face_frameidx_bbidx, value in clip_metadata["face"].items():
#             face_frameidx, face_bbidx = face_frameidx_bbidx.split("_")
#             max_obj_idx = 0
#             for obj_frameidx_bbidx in combined_metadata:
#                 if face_frameidx in obj_frameidx_bbidx:
#                     _, obj_bbidx_ = obj_frameidx_bbidx.split("_")
#                     max_obj_idx = max(max_obj_idx, int(obj_bbidx_))

#             if max_obj_idx > 0:
#                 new_face_bbidx = max_obj_idx + 1
#                 new_key = f"{face_frameidx}_{new_face_bbidx:04d}"
#                 combined_metadata[new_key] = value
#                 combined_metadata[new_key]["bbId"] = new_key
#             else:
#                 combined_metadata[face_frameidx_bbidx] = value

#     combined_metadata = _sort_dict_by_frame(combined_metadata)
#     get_udf_query(
#         clip_filename,
#         properties,
#         INGESTION.replace(",", "+"),
#         (width, height),
#         id="udf_metadata",
#         metadata=combined_metadata,
#         test_mode=TEST_MODE,
#     )

#     if DEBUG == "1":
#         print(
#             f"[TIMING],end_clip_metadata,{clip_key},{time.time()}",
#             flush=True,
#         )


# # method to create clips (read frame write to file; add name to list)
# # def send_metadata():
# #     global all_metadata
# #     clip_filename = ""
# #     clip_key = ""
# #     width = 0
# #     height = 0
# #     while True:
# #         try:
# #             queue_details = send_metadata_queue.get()
# #             if queue_details is None:
# #                 break

# #             (clip_key, clip_filename, width, height) = queue_details

# #             metadata2vdms(
# #                 clip_key,
# #                 clip_filename,
# #                 all_metadata[clip_key],
# #                 width,
# #                 height,
# #             )
# #             del all_metadata[clip_key]

# #         except queue.Empty:
# #             pass


# class StreamProcessor:
#     def __init__(self, model_path, source, name):
#         self.model = YOLO(model_path, task="detect")
#         self.cap = cv2.VideoCapture(source)
#         self.name = name
#         self.source =source

#         self.video_writer = None
#         self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         self.clip_id = 0
#         self.clip_filename = ""
#         self.clip_key = ""
#         self.tmp_file = ""

#         self.resize_h, self.resize_w = [MODEL_H, MODEL_W]
#         self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.numFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
#         self.scale_x = self.frame_width / MODEL_W
#         self.scale_y = self.frame_height / MODEL_H
#         self.min_contour_area = int((0.005 * self.frame_width) * (0.005 * self.frame_height) )  # 207

#         self.operation_device_map = PipelineMapping(detection_device="cpu")   # No CUDA HERE
#         self.device_input = (
#             self.operation_device_map.detection_device
#             if self.operation_device_map.detection_device == "cpu"
#             else "cuda"
#         )

#         self.cpu_resized_frame = None

#         # Subtraction
#         history= 300  # int(5 * self.fps)
#         background_thresh = 350
#         NSamples = 10
#         kNNSamples = 2
#         self.lr = -1  #.01  #-1  # 0.001  #1 / (5 * self.fps)  # -1  # 0.01  # 1 / history
#         bkgd_mask_queue_size = 3
#         self.backSub_cpu = cv2.createBackgroundSubtractorKNN(
#             history=history,                    # default 500
#             dist2Threshold=background_thresh,   # default 400
#             detectShadows=False,                # default True
#         )
#         self.backSub_cpu.setkNNSamples(kNNSamples)
#         self.backSub_cpu.setNSamples(NSamples)

#         prev_bkgd = np.zeros((MODEL_H, MODEL_W), dtype="uint8")
#         self.mask_history = deque(maxlen=bkgd_mask_queue_size)
#         self.mask_history.append(prev_bkgd)

#         self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         self.dilate_kernel_for_enhanced_mask = np.ones((21,21), np.uint8)

#         # Create ThreadPoolExecutor
#         self.executor = ThreadPoolExecutor(max_workers=NUM_USUABLE_CPUS)

#     def new_get_detections_for_contours_bbs(
#         self, frameNum, foi, contours, thickness=2, device_input="cuda"
#     ):
#         global active_streams
#         # source = self.source
#         stream_name = self.name
#         num_objs = 0
#         # predictions = []
#         metadata = dict()
#         cropped_imgs, cropped_coords = [], []
#         H, W = foi.shape[:2]  # Unpack once
#         bbs_full_res = []

#         # Filter and Sort in one go (Minimize Python-to-C++ crossings)
#         raw_bbs=[]
#         padding = 64
#         for c in contours:
#             area = cv2.contourArea(c)
#             x1,y1,w,h = cv2.boundingRect(c)
#             if area > self.min_contour_area: # and area / (w*h) >=0.3:  # and 0.5 < (w / h) < 2.0: # w/ solidity & aspect
#                 xx1 = max(0, int((x1 * self.scale_x)) - padding)
#                 yy1 = max(0, int((y1 * self.scale_y)) - padding)
#                 xx2 = min(W, int(((x1+w) * self.scale_x)) + padding)
#                 yy2 = min(H, int(((y1+h) * self.scale_y)) + padding)
#                 raw_bbs.append([area,[xx1,yy1,xx2,yy2]])
#         bbs_full_res = sorted(
#             [pair[1] for pair in raw_bbs if pair[0] > self.min_contour_area],
#             key=lambda x: x[0],
#             reverse=True,
#         )[:MAX_DETECTIONS]

#         dist_thresh = min(0.05*W,0.05*H)
#         merged = merge_boxes_limit(bbs_full_res, dist_threshold=dist_thresh, size_limit=640)

#         merged = filter_contained_boxes(merged, containment_thresh=0.9)

#         # for cnt, area in merged:
#         for (x1, y1, x2, y2) in merged:

#             if x2 > x1 and y2 > y1 and (x2-x1) < self.frame_width and (y2-y1) < self.frame_height :
#                 crop = foi[y1:y2, x1:x2]
#                 if crop.size > 0:
#                     cropped_imgs.append(crop)
#                     cropped_coords.append((x1, y1))

#         if not cropped_imgs:
#             return metadata  #num_objs, predictions

#         # 2. Inference (Keep stream=False as it is stable)
#         results = self.model.predict(
#             cropped_imgs,
#             imgsz=MODEL_W,
#             batch=len(cropped_imgs),
#             device=device_input,
#             verbose=False,
#             stream=True,
#             max_det=MAX_DETECTIONS,
#             # classes=[0],  # only "person",
#             # conf=0.45,
#         )

#         label_source = (
#             self.model.names if hasattr(self.model, "names") else YOLO_CLASS_NAMES
#         )

#         for ridx, r in enumerate(results):
#             if r.boxes is None or len(r.boxes) == 0:
#                 continue

#             # Move to CPU in one bulk operation per crop
#             boxes = r.boxes.xyxy.cpu().numpy().astype(int)
#             clss = r.boxes.cls.cpu().numpy().astype(int)
#             confs = r.boxes.conf.cpu().numpy()
#             off_x, off_y = cropped_coords[ridx]

#             for j in range(len(boxes)):
#                 num_objs += 1
#                 bx1, by1, bx2, by2 = boxes[j]
#                 abs_x1, abs_y1 = off_x + bx1, off_y + by1
#                 abs_x2, abs_y2 = off_x + bx2, off_y + by2
#                 class_id = clss[j]
#                 class_name = label_source[class_id]
#                 confidence = confs[j]
#                 if confidence > DETECTION_THRESHOLD:
#                     if not OMIT_DETECTIONS_FLAG:
#                         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
#                         print(
#                             # f"[OBJECT DETECTION] {class_name} detected in frame {frameNum} (Total detected: {current_cnt})",
#                             f"[{timestamp}] {stream_name} DETECTION on Frame {frameNum}: {class_name} detected",
#                             flush=True,
#                         )

#                     bb_color = get_detection_color(class_id, is_bgr=True)

#                     cv2.rectangle(
#                         foi,
#                         (abs_x1, abs_y1),
#                         (abs_x2, abs_y2),
#                         bb_color,
#                         thickness,
#                     )
#                     label = f"{class_name} {confidence:.2f}"
#                     draw_label(foi, label, (abs_x1, abs_y1), color=bb_color, padding=5)

#                     height = min(abs_y2, H) - max(0, abs_y1)
#                     width = min(abs_x2, W) - max(0, abs_x1)
#                     object_res = [
#                         abs_x1,
#                         abs_y1,
#                         height,
#                         width,
#                         class_name,
#                         confidence,
#                         H,
#                         W,
#                     ]

#                     framenum_str = f"{frameNum:04d}_{j:04d}"
#                     if DEBUG_FLAG:
#                         meta_str = ",".join(
#                             [str(o) for o in object_res + [framenum_str]]
#                         )
#                         print(f"[{stream_name} METADATA],{meta_str}", flush=True)

#                     metadata[framenum_str] = {
#                         "frameId": frameNum,
#                         "bbId": framenum_str,
#                         "bbox": {
#                             "x": int(object_res[0]),
#                             "y": int(object_res[1]),
#                             "height": int(object_res[2]),
#                             "width": int(object_res[3]),
#                             "object": str(object_res[4]),
#                             "object_det": {
#                                 "confidence": float(object_res[5]),
#                                 "frameH": int(H),
#                                 "frameW": int(W),
#                             },
#                         },
#                     }

#             # annotated_frame = r.plot()

#         # Queue frame for display (reduce quality slightly to 80 for 8K bandwidth)
#         if self.frame_width > 1280:
#             display_frame = cv2.resize(foi, (1280, 720), interpolation=cv2.INTER_AREA)
#             _, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
#         else:
#             _, buffer = cv2.imencode(".jpg", foi, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
#         frame_bytes = buffer.tobytes()

#         # Maintain only the freshest frame in the queue
#         if active_streams[self.name].full():
#             try:
#                 active_streams[self.name].get_nowait()
#             except Exception:
#                 pass
#         active_streams[self.name].put(frame_bytes)

#         try:
#             # Use block=False so the inference doesn't wait if the UI is slow
#             active_streams[self.name].put(frame_bytes, block=False)
#         except:
#             pass # Skip frame if queue is full

#         # #  Handle Video Writing (Cycle every 10 seconds)
#         # clip_frameNum = (frameNum - 1) % MAX_FRAMES_PER_CLIP
#         # print(f"frameNum: {frameNum} ({clip_frameNum})")
#         # if clip_frameNum == 0:
#         #     if self.video_writer:
#         #         # video_writer.release()
#         #         self.video_writer = release_clip_and_reencode(
#         #             self.clip_key, self.video_writer, self.clip_filename, self.tmp_file, TARGET_FPS
#         #         )

#         #         send_metadata_queue.put(
#         #             (
#         #                 self.clip_key,
#         #                 self.clip_filename,
#         #                 self.frame_width,
#         #                 self.frame_height,
#         #             )
#         #         )
#         #         self.clip_id += 1

#         #     if "://" not in str(source):
#         #         self.clip_filename = f"{SHARED_OUTPUT}/{stream_name}_{self.clip_id}.mp4"
#         #     else:
#         #         self.clip_filename = f"{SHARED_OUTPUT}/{stream_name}_{time.time()}.mp4"

#         #     self.tmp_file = TMP_LOCATION + self.clip_filename.split("/")[-1]
#         #     self.clip_key = Path(self.clip_filename).name

#         #     # timestamp = int(time.time())
#         #     # filename = f"clip_{timestamp}.mp4"
#         #     self.video_writer = cv2.VideoWriter(self.tmp_file, self.fourcc, TARGET_FPS, (width, height))
#         #     main_app_logger.info(f"Started new clip: {self.tmp_file}")

#         # # 3. Write frame
#         # self.video_writer.write(foi)
#         # frame_counter += 1
#         return metadata


#         # if not results:
#         #     return num_objs, predictions

#         # # 3. Post-processing
#         # label_source = (
#         #     self.model.names if hasattr(self.model, "names") else YOLO_CLASS_NAMES
#         # )
#         # predictions = []
#         # for ridx, r in enumerate(results):
#         #     if r.boxes is None or len(r.boxes) == 0:
#         #         continue

#         #     # Move to CPU in one bulk operation per crop
#         #     boxes = r.boxes.xyxy.cpu().numpy().astype(int)
#         #     clss = r.boxes.cls.cpu().numpy().astype(int)
#         #     confs = r.boxes.conf.cpu().numpy()
#         #     off_x, off_y = cropped_coords[ridx]

#         #     for j in range(len(boxes)):
#         #         num_objs += 1
#         #         bx1, by1, bx2, by2 = boxes[j]
#         #         abs_x1, abs_y1 = off_x + bx1, off_y + by1
#         #         class_id = clss[j]
#         #         bb_color = get_detection_color(class_id, is_bgr=True)

#         #         cv2.rectangle(
#         #             foi,
#         #             (abs_x1, abs_y1),
#         #             (off_x + bx2, off_y + by2),
#         #             bb_color,
#         #             thickness,
#         #         )
#         #         label = f"{label_source[class_id]} {confs[j]:.2f}"
#         #         draw_label(foi, label, (abs_x1, abs_y1), color=bb_color, padding=5)
#         #         predictions.append(
#         #             [class_id, abs_x1, abs_y1, off_x + bx2, off_y + by2, confs[j]]
#         #         )

#         # return num_objs, predictions

#     def new_contour2predictions(self, frameNum, mask, frame, device_input="cpu"):
#         # global all_metadata
#         manager, active_streams, all_metadata, send_metadata_queue = get_manager_stuff()
#         source = self.source
#         stream_name = self.name
#         # Find movement areas
#         # if cv2.countNonZero(mask) > (mask.size * 0.5):
#         #     return 0, []
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         #  Handle Video Writing (Cycle every 10 seconds)
#         clip_frameNum = (frameNum - 1) % MAX_FRAMES_PER_CLIP
#         print(f"frameNum: {frameNum} ({clip_frameNum})")
#         if clip_frameNum == 0:
#             if self.video_writer:
#                 # video_writer.release()
#                 self.video_writer = release_clip_and_reencode(
#                     self.clip_key, self.video_writer, self.clip_filename, self.tmp_file, TARGET_FPS
#                 )

#                 send_metadata_queue.put(
#                     (
#                         self.clip_key,
#                         self.clip_filename,
#                         self.frame_width,
#                         self.frame_height,
#                     )
#                 )
#                 self.clip_id += 1

#             if "://" not in str(source):
#                 self.clip_filename = f"{SHARED_OUTPUT}/{stream_name}_{self.clip_id}.mp4"
#             else:
#                 self.clip_filename = f"{SHARED_OUTPUT}/{stream_name}_{time.time()}.mp4"

#             self.tmp_file = TMP_LOCATION + self.clip_filename.split("/")[-1]
#             self.clip_key = Path(self.clip_filename).name

#             # timestamp = int(time.time())
#             # filename = f"clip_{timestamp}.mp4"
#             self.video_writer = cv2.VideoWriter(self.tmp_file, self.fourcc, TARGET_FPS, (self.frame_width, self.frame_height))
#             main_app_logger.info(f"Started new clip: {self.tmp_file}")

#         # 3. Write frame
#         self.video_writer.write(frame)

#         # num_objs = 0
#         # predictions = []
#         metadata = dict()
#         if contours:
#             # Pass contours directly to the detection logic
#             # Skip the 'get_bb_mask' and 'morphologyEx' on full frames
#             # num_objs, predictions =
#             # self.new_get_detections_for_contours_bbs(
#             #     queue,
#             #     frame,
#             #     contours,
#             #     device_input=device_input,
#             # )

#             metadata = self.new_get_detections_for_contours_bbs(
#                 frameNum, frame, contours, thickness=2, device_input=device_input
#             )

#             if metadata:
#                 all_metadata.setdefault(
#                     self.clip_key,
#                     {
#                         "object": {},
#                         "face": {},
#                     }
#                 )
#                 all_metadata[self.clip_key]["object"].update(metadata)
#             # all_metadata[clip_key]["face"].update(metadata_face)
#         # return metadata
#         # return num_objs, predictions

#     def test_full_cpu_detection_gpu(self, frame, frameNum):
#         # Resize directly into the pre-allocated Pinned Memory
#         # This avoids a temporary CPU allocation
#         H, W = self.resize_h, self.resize_w
#         self.cpu_resized_frame = cv2.resize(frame, (W, H))

#         # if frameNum == 1:
#         #     # Do the same for CPU if needed (OpenCV does this internally, but seeding helps)
#         #     for _ in range(self.backSub_cpu.getNSamples()):
#         #         self.backSub_cpu.apply(self.cpu_resized_frame, learningRate=1.0)

#         # Background Subtraction on CPU
#         fgMask = self.backSub_cpu.apply(self.cpu_resized_frame, learningRate=self.lr)

#         # Skip detection/prediction for the first 10-15 frames of a new stream
#         # Just update the background, don't run the rest of the pipeline
#         # if frameNum < self.fps/2:
#         #     return 0, []

#         prev_bkgd = np.ones_like(fgMask)  # AND
#         for m in self.mask_history:
#             # Dilate the historical mask
#             dilated = cv2.dilate(m, self.dilate_kernel_for_enhanced_mask, iterations=1)
#             cv2.bitwise_and(prev_bkgd, dilated, dst=prev_bkgd)
#         self.mask_history.append(fgMask)

#         if prev_bkgd.max()!=prev_bkgd.min():
#             combined_mask_bool = (fgMask > 0) | (prev_bkgd > 0)

#             # Convert the boolean array back to uint8 with 0 and 255 values
#             fgMask = combined_mask_bool.astype(np.uint8) * 255

#         # Thresholding
#         _, mask = cv2.threshold(
#             fgMask, MASK_THRESHOLD_VALUE, MASK_MAX_VALUE, cv2.THRESH_BINARY
#         )

#         mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)

#         # Get Contours & Run Inference on detection_device
#         device_input = (
#             self.operation_device_map.detection_device
#             if self.operation_device_map.detection_device == "cpu"
#             else "cuda"
#         )

#         # num_objs, predictions =
#         self.new_contour2predictions(frameNum, mask, frame, device_input=device_input)

#     # method to start thread
#     def start(self):
#         self.stopped = False
#         self.t = []
#         # self.t.append(
#         #     self.executor.submit(
#         #         self.get_frames,
#         #     )
#         # )
#         self.t.append(
#             self.executor.submit(
#                 send_metadata,
#             )
#         )

#     # method to stop reading frames
#     def stop(self):
#         for t in as_completed(self.t):
#             try:
#                 _ = t.result()
#             except Exception as t_e:
#                 print(f"[DEBUG] Exception occurred in thread: {t_e}")

#         # self.stopped = True
#         self.cap.release()
