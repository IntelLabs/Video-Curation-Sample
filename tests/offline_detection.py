import os
import shlex
import subprocess
import time
from pathlib import Path

import cv2  # OpenCV library
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz


def str2bool(in_val):
    if isinstance(in_val, bool):
        return in_val

    if not isinstance(in_val, str):
        raise ValueError(f"{in_val} is not a bool or string")

    if in_val.title() == "True":
        return True
    else:
        return False


PROJECT_PATH = Path(__file__).parent.parent.absolute()

# CV2_INTERPOLATION = cv2.INTER_AREA
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1, mp4v, AVC1
# IN_SOURCE = os.environ["IN_SOURCE"]
# kkhost = os.environ["KKHOST"]
# MODEL_PRECISION_face = "FP16"
# Path(SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
# REPO_DIR = Path(__file__).parent.parent
# TEST_VIDEO_PATH = REPO_DIR / "video/archive_custom/video8K__test-8k-26s.mp4"
# tmp_dir = "/var/www/archive"
# video_store_dir = "/var/www/mp4"
CODE_DIR = os.getenv("CODE_DIR", str(PROJECT_PATH / "tests"))
DBHOST = "vdms-service"  # os.environ["DBHOST"]
DBPORT = 55555
# DEBUG = os.environ["DEBUG"]
DEBUG = os.getenv("DEBUG", "0")
DEBUG_FLAG = True if DEBUG == "1" else False
# DEVICE = os.environ["DEVICE"]
DEVICE = os.getenv("DEVICE", "CPU")
DEVICE_OV = "AUTO"
# INGESTION = os.environ["INGESTION"]
INGESTION = os.getenv("INGESTION", "object")
MODEL_PRECISION = "FP16"
MODEL_W, MODEL_H = (640, 640)
RESIZE_FLAG = str2bool(os.getenv("RESIZE_FLAG", False))
SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
TARGET_FPS = 15  # 15  30
TEST_MODE = str2bool(os.getenv("TEST_FLAG", False))
UDF_HOST = "video-service"
UDF_PORT = 5011

if DEVICE == "GPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device_input = DEVICE.lower() if DEVICE == "CPU" else os.environ["CUDA_VISIBLE_DEVICES"]

RESOLUTION_NAME_BY_WH = {
    "1280x736": "1K",
    "3840x2176": "4K",
    "7680x4320": "8K",
    "640x640": "640x640",
}

batch_size = 1
detection_threshold = 0.25  # 0.7
half_flag = True
iou_threshold = 0.7  # 0.9  # 0.5

model_path = f"{CODE_DIR}/resources/models/ultralytics/yolo11/{MODEL_PRECISION}/yolo11n_openvino_model"
model = YOLO(model_path, verbose=False, task="detect")


# ---------- Overlay FPS and System Usage ----------
def get_font_dims(h, w, fontScale=7e-4, thicknessScale=1e-3):
    # h, w = frame.shape[:2]
    fontScale = min(h, w) * fontScale
    thickness = max(
        1, int(min(h, w) * thicknessScale)
    )  # ceil(min(h,w) * thicknessScale)
    return fontScale, thickness


def overlay_info(frame, fps, fontScale=0.5, thickness=1):
    # cpu = psutil.cpu_percent()
    # mem = psutil.virtual_memory().percent
    h, w = frame.shape[:2]
    res = RESOLUTION_NAME_BY_WH[f"{int(w)}x{int(h)}"]
    # text = f"RESOLUTION: {res} | FPS: {fps:.1f} | CPU: {cpu}% | MEM: {mem}%"
    text = f"RESOLUTION: {res} | FPS: {fps:.1f}"
    box_offset = int(0.1 * h)  #  30
    text_offset = int(box_offset / 3)  # int(0.02 * h)  # 10

    # Bottom of video
    # cv2.rectangle(frame, (0, h - box_offset), (w, h), (0, 0, 0), -1)
    # cv2.putText(
    #     frame, text, (text_offset, h - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    # )

    # Top of video
    # cv2.rectangle(frame, (0, 0), (w, box_offset), (0, 0, 0), -1)

    cv2.putText(
        frame,
        text,
        (text_offset, box_offset - text_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        (0, 0, 0),  # (255, 255, 255),
        thickness,
    )
    return frame


# ---------- Processor ----------
def processor(args):  # (camera_src, camera_name=None):
    camera_src = args.input
    result_video = args.output
    tmp_file = args.tmp_output

    # processing frames in input stream
    num_frames_processed = 0

    # open file
    video_obj = cv2.VideoCapture(camera_src, cv2.CAP_FFMPEG)

    # get FPS of input video
    fps = int(video_obj.get(cv2.CAP_PROP_FPS))
    input_width = int(video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if RESIZE_FLAG or ((input_height * input_width) < (MODEL_H * MODEL_W)):
        new_sizeHW = check_imgsz([MODEL_H, MODEL_W])  # expects hxw
        result_video = str(result_video).replace(".mp4", ".resized.mp4")
        tmp_file = str(tmp_file).replace(".mp4", ".resized.mp4")
    else:
        new_sizeHW = check_imgsz([input_height, input_width])  # expects hxw

    new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

    width = new_sizeWH[0]  # self.video_obj.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = new_sizeWH[1]

    fontScale, thickness = get_font_dims(
        height, width, fontScale=1.5e-3, thicknessScale=4e-3
    )  # 1e-3)  # 3e-3

    # define VideoWriter object
    out = cv2.VideoWriter(tmp_file, fourcc, fps, (width, height))

    # read and write frams for output video
    start = time.time()
    while video_obj.isOpened():
        grabbed, frame = video_obj.read()
        if not grabbed:
            break

        frameNum = int(video_obj.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Processing frame {frameNum}", flush=True)

        img_size = (width, height)
        if frame.shape != img_size:
            frame = cv2.resize(frame, img_size)

        results = model.predict(
            frame,
            imgsz=(img_size[1], img_size[0]),
            batch=batch_size,
            conf=detection_threshold,
            iou=iou_threshold,
            half=half_flag,
            device=device_input,
            verbose=False,
            stream=True,
        )

        for result in results:
            annotated = result.plot(font_size=8)
            annotated = overlay_info(annotated, fps, fontScale, thickness)

        out.write(annotated)
        num_frames_processed += 1

    end = time.time()

    # release resources
    video_obj.release()
    out.release()
    cv2.destroyAllWindows()

    # printing time elapsed and fps
    elapsed = end - start
    # fps = num_frames_processed/elapsed
    print(
        "FPS: {}, # Frames: {}, Elapsed Time: {} ".format(
            fps, num_frames_processed, elapsed
        ),
        flush=True,
    )

    # Re-encode video in order to seek via ffmpeg later
    GENERAL_OPTS = "-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -flush_packets 0"  #  -filter:v fps={self.target_fps}
    CONVERSION = f"-c:v libx264 -preset ultrafast -filter:v fps=fps={fps}"  # "-c:v libx264 -preset medium"
    reencode_cmd = f"ffmpeg -y -i {tmp_file} {GENERAL_OPTS} {CONVERSION} -crf 23 -c:a copy {result_video}"
    cmd_list = shlex.split(reencode_cmd)
    subprocess.run(cmd_list, check=True)
    os.remove(tmp_file)


def get_inputs():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=Path,
        required=True,
        help="Path to video to process with YOLO",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=Path,
        required=True,
        help="Directory to store resulting video with annotations",
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.input.exists():
        stream_name = args.input.stem
    else:
        stream_name = str(args.input).split("/")[-1]

    args.tmp_output = args.output / f"_{stream_name}_annotated.mp4"
    args.output = args.output / f"{stream_name}_annotated.mp4"

    return args


if __name__ == "__main__":
    args = get_inputs()
    processor(args)
