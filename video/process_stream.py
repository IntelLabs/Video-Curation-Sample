import multiprocessing as mp
import os
import queue
import shlex
import subprocess
import sys
import time  # time library
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock  # library for multi-threading

import cv2  # OpenCV library
import psutil
from openvino.runtime import Core
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

import vdms


def _sort_dict_by_frame(in_dict):
    def _by_int(key):
        return tuple(int(k) for k in key.split("_"))

    return dict(sorted(in_dict.items(), key=lambda x: _by_int(x[0])))


def str2bool(in_val):
    if isinstance(in_val, bool):
        return in_val

    if not isinstance(in_val, str):
        raise ValueError(f"{in_val} is not a bool or string")

    if in_val.title() == "True":
        return True
    else:
        return False


""" GENERAL VARIABLES """
CODE_DIR = os.getenv("CODE_DIR", "/home")
CUSTOM_MODEL_FLAG = str2bool(os.getenv("CUSTOM_MODEL_FLAG", False))
DBHOST = os.getenv("DBHOST", "vdms-service")
DEBUG = os.getenv("DEBUG", "0")
DEVICE = os.getenv("DEVICE", "CPU")
INGESTION = os.getenv("INGESTION", "object,face")
# NUM_UDFS = int(os.getenv("NCURATIONS", 1))
RESIZE_FLAG = str2bool(os.getenv("RESIZE_FLAG", False))
SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
TEST_MODE = str2bool(os.getenv("TEST_FLAG", False))
TMP_LOCATION = os.getenv("TMP_LOCATION", "/var/www/streams/")
UDF_HOST = os.getenv("UDF_HOST", "video-service")
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n")
# vdms_pool_size = 10

LOCKTIMEOUT_RETRIES = 5
ERR_KEYWORDS = ["timeout", "null search iterator", "outoftransactions"]

BATCH_SIZE = 1
DBPORT = 55555
DEBUG_FLAG = True if DEBUG == "1" else False
DETECTION_THRESHOLD = 0.25  # 0.7
DEVICE_OV = "AUTO"
HALF_FLAG = True
IOU_THRESHOLD = 0.7  # 0.9  # 0.5
MODEL_PRECISION = "FP16"
MODEL_W, MODEL_H = (640, 640)
TARGET_FPS = 15  # 15  30
UDF_PORT = 5011
WRITER_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # avc1, mp4v, AVC1

if CUSTOM_MODEL_FLAG:
    model_path = f"{CODE_DIR}/resources/models/ultralytics/custom_models/{MODEL_NAME}"
else:
    model_path = f"{CODE_DIR}/resources/models/ultralytics/{MODEL_NAME}/{MODEL_PRECISION}/{MODEL_NAME}"

# if hasattr(os, "process_cpu_count"):
#     num_usuable_cpus = os.process_cpu_count()
# elif hasattr(os, "sched_getaffinity"):
#     num_usuable_cpus = len(os.sched_getaffinity(0))
# else:
num_usuable_cpus = os.cpu_count()

if DEVICE == "GPU":
    model_path += ".engine"
    batch_size = int(os.environ.get("GPU_BATCH_SIZE", 1))
else:
    model_path += "_openvino_model/"
    batch_size = int(os.environ.get("CPU_BATCH_SIZE", 1))  # 8


def retry_query(db, query, num_retries=LOCKTIMEOUT_RETRIES, sleep_timer: int = 0):
    # ridx = 0
    # while True:
    for ridx in range(num_retries + 1):
        response, _ = db.query(query, [[]])
        if "FailedCommand" in response[0] and any(
            k in response[0]["info"].lower() for k in ERR_KEYWORDS
        ):
            err = response[0]["info"]
            if DEBUG == "1":
                query_type = list(query[0].keys())[0]
                print(
                    # f"DEBUG [process_stream Attempt #{ridx}] Received '{err}' for {query}",
                    f"DEBUG [process_stream Attempt #{ridx}] Received '{err}' for {query_type} query",
                    flush=True,
                )
            if sleep_timer > 0:
                time.sleep(sleep_timer)
            # ridx += 1
            # pass  # Rerun
        else:
            if DEBUG == "1":
                print(
                    f"[DEBUG process_stream] Successful query response: {response}",
                    flush=True,
                )
            break  # Continue
    return response


""" MODEL DEFINITIONS """
model = YOLO(model_path, verbose=False, task="detect")


ie = Core()
face_detection_model_xml = f"{CODE_DIR}/resources/models/intel/face-detection-adas-0001/{MODEL_PRECISION}/face-detection-adas-0001.xml"
face_detection_model = ie.read_model(
    model=face_detection_model_xml,
    weights=face_detection_model_xml.replace(".xml", ".bin"),
)
# face_det_w, face_det_h = 672, 384
_, face_det_c, face_det_h, face_det_w = face_detection_model.inputs[0].shape
face_det_compiled_model = ie.compile_model(face_detection_model, DEVICE_OV)

age_gender_classification_model_xml = f"{CODE_DIR}/resources/models/intel/age-gender-recognition-retail-0013/{MODEL_PRECISION}/age-gender-recognition-retail-0013.xml"
age_gender_classification_model = ie.read_model(
    model=age_gender_classification_model_xml,
    weights=age_gender_classification_model_xml.replace(".xml", ".bin"),
)
# ag_w, ag_h = 62, 62
_, ag_c, ag_h, ag_w = age_gender_classification_model.inputs[0].shape
ag_compiled_model = ie.compile_model(age_gender_classification_model, DEVICE_OV)

emotions_classification_model_xml = f"{CODE_DIR}/resources/models/intel/emotions-recognition-retail-0003/{MODEL_PRECISION}/emotions-recognition-retail-0003.xml"
emotions_classification_model = ie.read_model(
    model=emotions_classification_model_xml,
    weights=emotions_classification_model_xml.replace(".xml", ".bin"),
)
# em_w, em_h = 64, 64
_, em_c, em_h, em_w = emotions_classification_model.inputs[0].shape
em_compiled_model = ie.compile_model(emotions_classification_model, DEVICE_OV)


""" DETECTION FUNCTIONS """


# Detect faces from frame
def face_detection(stream_name, frameNum, frame, img_size):
    W, H = img_size
    bs = 1
    # Model expects BGRA
    # face detect -> age-gender -> emotions
    global face_det_compiled_model, ag_compiled_model, em_compiled_model
    genders = ["female", "male"]
    emotions = ["neutral", "happy", "sad", "surprise", "anger"]

    # input_layer = face_det_compiled_model.input(0)
    # Resize expects HWC
    input_image = cv2.resize(
        frame, (face_det_w, face_det_h), interpolation=cv2.INTER_AREA
    )
    input_image = input_image.transpose(2, 0, 1)  # Shape: CHW
    input_image = input_image.reshape((bs, face_det_c, face_det_h, face_det_w))

    output_layer = face_det_compiled_model.output(0)
    result = face_det_compiled_model([input_image])[output_layer]

    # Process the detections
    faces = []
    metadata = dict()
    oidx = 1
    for detection in result[0][0]:
        confidence = float(detection[2])
        if confidence > DETECTION_THRESHOLD:
            # Draw a bounding box around the face
            x1 = int(detection[3] * frame.shape[1])
            if x1 < 0:
                x1 = 0

            y1 = int(detection[4] * frame.shape[0])
            if y1 < 0:
                y1 = 0

            x2 = int(detection[5] * frame.shape[1])
            if x2 > frame.shape[1] - 1:
                x2 = frame.shape[1] - 1

            y2 = int(detection[6] * frame.shape[0])
            if y2 > frame.shape[0] - 1:
                y2 = frame.shape[0] - 1

            height = y2 - y1
            width = x2 - x1

            face_roi = frame[y1:y2, x1:x2]
            # print(face_roi.shape)
            age = gender = emotion = None
            try:
                ag_face_blob = cv2.resize(
                    face_roi, (ag_w, ag_h), interpolation=cv2.INTER_AREA
                )
                ag_face_blob = ag_face_blob.transpose((2, 0, 1))
                ag_face_blob = ag_face_blob.reshape((bs, ag_c, ag_h, ag_w))
                ag_result = ag_compiled_model([ag_face_blob])
                age = int(ag_result["fc3_a"].flatten()[0] * 100)
                gender = str(genders[ag_result["prob"].argmax()])
            except Exception as e:
                print(f"Error occurred: {e}. Skipping age-gender model", flush=True)

            try:
                em_face_blob = cv2.resize(
                    face_roi, (em_w, em_h), interpolation=cv2.INTER_AREA
                )
                em_face_blob = em_face_blob.transpose((2, 0, 1))
                em_face_blob = em_face_blob.reshape((bs, em_c, em_h, em_w))
                em_result = em_compiled_model([em_face_blob])[
                    em_compiled_model.output(0)
                ]
                emotion = str(emotions[em_result.argmax()])
            except Exception as e:
                print(f"Error occurred: {e}. Skipping emotion model", flush=True)
            face_res = [x1, y1, height, width, age, gender, emotion, confidence, H, W]
            # print(face_res)
            faces.append(face_res)

            tdict = {
                "x": int(face_res[0]),
                "y": int(face_res[1]),
                "height": int(face_res[2]),
                "width": int(face_res[3]),
                "object": "face",
                "object_det": {
                    "age": int(face_res[4]),
                    "gender": str(face_res[5]),
                    "emotion": str(face_res[6]),
                    "confidence": float(face_res[7]),
                    "frameH": int(H),
                    "frameW": int(W),
                },
            }
            # framenum_str = f"{frameNum}_{oidx}"
            framenum_str = f"{frameNum:04d}_{oidx:04d}"
            if DEBUG_FLAG:
                meta_str = ",".join([str(o) for o in face_res + [framenum_str]])
                print(f"[{stream_name} METADATA],{meta_str}", flush=True)

            metadata[framenum_str] = {
                "frameId": frameNum,
                "bbId": framenum_str,
                "bbox": tdict,
            }
            oidx += 1

    return metadata


# Extract metadata from object model results
def extract_metadata_from_results(
    stream_name, frameNum, results, img_size, fps=TARGET_FPS
):
    fW, fH = img_size
    metadata = dict()
    try:
        for bidx, result in enumerate(results):
            annotated = result.plot()
            annotated = overlay_info(annotated, fps)

            # GET METADATA FOR CLIP
            boxes = result.boxes.cpu()
            oidx = 0
            for box in boxes:
                confidence = float(box.conf.item())
                if confidence > DETECTION_THRESHOLD:
                    class_id = int(box.cls.item())
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
                    class_name = str(object_res[4])
                    # OBJ_COUNTER.setdefault(class_name, 0)
                    # OBJ_COUNTER[class_name] += 1
                    # current_cnt = OBJ_COUNTER[class_name]
                    print(
                        # f"[OBJECT DETECTION] {class_name} detected in frame {frameNum} (Total detected: {current_cnt})",
                        f"[{stream_name} OBJECT DETECTION] {class_name} detected",
                        flush=True,
                    )

                    tdict = {
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
                    }

                    # framenum_str = f"{frameNum}_{oidx}"
                    framenum_str = f"{frameNum:04d}_{oidx:04d}"
                    if DEBUG_FLAG:
                        meta_str = ",".join(
                            [str(o) for o in object_res + [framenum_str]]
                        )
                        print(f"[{stream_name} METADATA],{meta_str}", flush=True)

                    metadata[framenum_str] = {
                        "frameId": frameNum,
                        "bbId": framenum_str,
                        "bbox": tdict,
                    }
                    oidx += 1

    except Exception:
        e = traceback.format_exc()
        print(f"Error in {stream_name} extract_metadata_from_results: {e}", flush=True)

    return annotated, metadata


# Inference Function
def infer_worker(
    stream_name,
    frameNum,
    frame,
    # model_path,
    img_size,
    INGESTION,
    fps=TARGET_FPS,
    return_annotated=False,
):  # img_size:(W,H)
    global model

    height, width = frame.shape[:2]
    if (width, height) != img_size:
        frame = cv2.resize(frame, img_size)

    annotated = None
    metadata = {}
    metadata_face = {}
    if "object" in INGESTION:
        results = model.predict(
            frame,
            imgsz=(img_size[1], img_size[0]),
            batch=BATCH_SIZE,
            conf=DETECTION_THRESHOLD,
            iou=IOU_THRESHOLD,
            half=HALF_FLAG,
            device=DEVICE,
            verbose=False,
            stream=True,
        )

        annotated, metadata = extract_metadata_from_results(
            stream_name, frameNum, results, img_size, fps=fps
        )

    if "face" in INGESTION:
        metadata_face = face_detection(stream_name, frameNum, frame, img_size)

    if return_annotated:
        return annotated, metadata, metadata_face
    else:
        return frame, metadata, metadata_face


""" HELPFUL FUNCTIONS """


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
    # db,
    # start_t,
    filename_path,
    properties,
    ingest_mode,
    new_size,
    id="udf_metadata",
    metadata=None,
    test_mode=TEST_MODE,
):
    # global dbs
    query = {
        "AddVideo": {
            "from_file_path": str(filename_path),  # from_server_file
            "is_local_file": True,
            "properties": properties,
            "operations": [
                {
                    "type": "remoteOp",
                    "url": f"http://{UDF_HOST}:{UDF_PORT}/video",
                    "options": {
                        "id": id,
                        "otype": ingest_mode,
                        "media_type": "video",
                        "input_sizeWH": new_size,
                        "filename": properties["Name"],
                    },
                }
            ],
        }
    }

    if id == "udf_metadata" and metadata is not None:
        # print(f"udf_metadata metadata: {metadata}", flush=True)
        query["AddVideo"]["operations"][0]["options"]["metadata"] = (
            metadata  # json.dumps(metadata)
        )

    if test_mode:
        # print(f"{filename_path} Query: {query}", flush=True)
        return

    # video_blob = []
    # with open(filename_path, "rb") as fd:
    #     video_blob.append(fd.read())
    # return query, video_blob

    filename = str(Path(filename_path).name)
    db = vdms.vdms()
    if not db.is_connected():
        db.connect(DBHOST, DBPORT)
    if DEBUG_FLAG:
        print(
            f"[TIMING],start_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
    try:
        # res, res_arr = db.query([query], [video_blob])
        res = retry_query(db, [query], num_retries=10, sleep_timer=5)

        if DEBUG_FLAG:
            print(
                f"[TIMING],end_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
                flush=True,
            )
            print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
            print(f"[DEBUG] {filename} INGEST_VIDEO RESPONSE: {res}", flush=True)
            # print(f"[DEBUG] Used client: {dn_name}", flush=True)
            # print(f"[DEBUG] Elapsed ingest_video time: {elapsed_time} sec", flush=True)
    except Exception:
        e = traceback.format_exc()
        print(f"[DEBUG] VDMS Query Exception: {e}", flush=True)
        # print(f"[DEBUG] failed query: {query}", flush=True)

    # elapsed_time = time.time() - start_t

    db.disconnect()
    del db


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


# ---------- Overlay FPS and System Usage ----------
def overlay_info(frame, fps):
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    text = f"FPS: {fps:.1f} | CPU: {cpu}% | MEM: {mem}%"
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    return frame


""" CLASSES """


# defining a helper class for implementing multi-threading
class VideoStream:
    # initialization method
    def __init__(self, src, fps=TARGET_FPS, fourcc=WRITER_FOURCC, camera_name=None):
        self.stream_id = src  # default is 0 for main camera
        self.fourcc = fourcc

        if "://" in str(self.stream_id):
            # self.src_id = src.split("/")[-1]
            if camera_name is not None:
                self.stream_name = camera_name
            else:
                self.stream_name = str(self.stream_id).split("/")[-1]
        else:
            # self.src_id = Path(src).stem
            self.stream_name = Path(self.stream_id).stem

        # os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"]="vcodec;x264|preset;medium|crf;23"
        if self.stream_id.startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        # Check that object is opened successfully
        self.connect_to_stream(time_limit_mins=5)
        if DEBUG == "1":
            print(
                f"[TIMING],Start processing,{self.stream_name},{time.time()}",
                flush=True,
            )

        self.setup_stream(fps)

        # Create ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=num_usuable_cpus)

    # method to start thread
    def start(self):
        self.stopped = False
        # [t.start() for t in self.t]
        self.t = []
        self.t.append(
            self.executor.submit(
                self.get_frames,
            )
        )
        self.t.append(
            self.executor.submit(
                self.get_clips,
            )
        )
        self.t.append(
            self.executor.submit(
                self.run_inference,
            )
        )
        # self.t.append(
        #     self.executor.submit(
        #         self.process_clip_metadata,
        #     )
        # )

    # method to stop reading frames
    def stop(self):
        # [t.join() for t in self.t]
        for t in as_completed(self.t):
            try:
                _ = t.result()
            except Exception as t_e:
                print(f"[DEBUG] Exception occurred in thread: {t_e}")

        self.stopped = True
        self.video_obj.release()

    # method to open stream/video within 5min (default) limit
    def connect_to_stream(self, time_limit_mins=5):
        # opening video capture stream
        self.video_obj = cv2.VideoCapture(self.stream_id, cv2.CAP_FFMPEG)

        stream_available = False
        time_limit_secs = time_limit_mins * 60
        connect_time = time.time()
        while not stream_available:
            if self.video_obj.isOpened():
                stream_available = True
            elif self.stream_id.startswith("rtsp"):
                if time.time() - connect_time < time_limit_secs:
                    self.video_obj = cv2.VideoCapture(self.stream_id, cv2.CAP_FFMPEG)
                else:
                    print(
                        f"Exceeds {time_limit_mins} mins limit to connect to {self.stream_name}. Exiting ..."
                    )
                    exit(1)

    # Gets video fps and framecount
    def get_fps_and_framecnt(self, fps):
        self.input_fps = int(self.video_obj.get(cv2.CAP_PROP_FPS))  # hardware fps
        if self.input_fps == 0:  # Case when FPs isn't available
            self.input_fps = manual_fps_calculation(self.stream_id, num_frames=10)

        self.target_fps = fps if self.input_fps > fps else self.input_fps
        self.frame_skip = int(self.input_fps / self.target_fps)
        if self.frame_skip < 1:
            self.frame_skip = 1

        print(f"FPS of {self.stream_name} input stream: {self.input_fps}", flush=True)
        print(f"FPS of {self.stream_name} output mp4: {self.target_fps}", flush=True)

        # Frame count for videos
        self.frame_count = None
        if "://" not in str(self.stream_id):
            self.frame_count = int(self.video_obj.get(cv2.CAP_PROP_FRAME_COUNT))

    # Gets frame W and H details
    def get_frameWH(self):
        input_width = int(self.video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(self.video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if RESIZE_FLAG or ((input_height * input_width) < (MODEL_H * MODEL_W)):
            new_sizeHW = check_imgsz([MODEL_H, MODEL_W])  # expects hxw
        else:
            new_sizeHW = check_imgsz([input_height, input_width])  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]
        self.height = new_sizeWH[1]

    # Sets up important info for stream
    def setup_stream(self, fps):
        self._lock = Lock()
        self.frame_queue = mp.Queue()
        self.clip_queue = mp.Queue()
        self.inference_queue = mp.Queue()
        # self.metadata_queue = mp.Queue()
        self.retrieved_frames = 0
        self.num_frames_processed = 0

        self.get_fps_and_framecnt(fps)

        self.get_frameWH()

        self._out_vid = None
        self.all_metadata = {}
        self.clip_end_frame = {}
        self.clip_filename = ""
        self.clip_frame_count = 0
        self.clip_frame_inds = []
        self.clip_id = 0
        self.clip_length_in_secs = 10
        self.clip_total_frames = int(float(self.clip_length_in_secs * self.target_fps))

    # method to process a frame
    def get_frames(self):
        clip_frame_idx = 0
        clip_id = 0
        _out_vid = None
        if DEBUG == "1":
            print(
                f"[TIMING],start_get_frames,{self.stream_name},{time.time()}",
                flush=True,
            )
        while True:
            if self.stopped:
                break

            grabbed, frame = self.video_obj.read()  # Read next frame
            # cv2.waitKey(int(1000 / self.input_fps))

            if not grabbed:  # or frame is None:
                # print(f"[Exiting {self.stream_name}] No more frames to read", flush=True)
                self.stopped = True
                self.frame_queue.put(None)
                self.inference_queue.put(None)
                break

            # else:
            frameNum = int(self.video_obj.get(cv2.CAP_PROP_POS_FRAMES))
            skip_frame_num = (frameNum - 1) % self.frame_skip

            if clip_frame_idx % self.clip_total_frames == 0:
                if "://" not in str(self.stream_id):
                    clip_filename = f"{SHARED_OUTPUT}/{self.stream_name}_{clip_id}.mp4"
                else:
                    clip_filename = (
                        f"{SHARED_OUTPUT}/{self.stream_name}_{time.time()}.mp4"
                    )

                tmp_file = TMP_LOCATION + clip_filename.split("/")[-1]

            if skip_frame_num == 0:
                h, w = frame.shape[:2]
                if (w, h) != (self.width, self.height):
                    frame = cv2.resize(frame, (self.width, self.height))

                queue_details = (
                    frameNum,  # Overall frame number
                    clip_frame_idx % self.clip_total_frames,  # Frame index in clip
                    clip_id,  # Clip number
                    clip_filename,
                    tmp_file,
                    frame,  # Frame
                )
                self.frame_queue.put(queue_details)
                self.inference_queue.put(queue_details)
                self.retrieved_frames += 1

                clip_frame_idx += 1
                if clip_frame_idx % self.clip_total_frames == 0:
                    clip_id += 1

            # delay_ms = int(1000 / self.input_fps)
            # cv2.waitKey(delay_ms)
            # time.sleep(delay_ms / 1000 )

        if DEBUG == "1":
            print(
                f"[TIMING],end_get_frames,{self.stream_name},{time.time()}", flush=True
            )

    # method to create clips
    def get_clips(self):
        _out_vid = None
        clip_frame_idx = 0
        clip_key = ""
        clip_filename = ""
        tmp_file = ""
        clip_id = 0
        frameNum = 0
        while True:
            try:
                queue_details = self.frame_queue.get()  # read()  # noqa: F841
                if queue_details is None:
                    if _out_vid is not None:
                        frame_count = clip_frame_idx + 1
                        if DEBUG == "1":
                            print(
                                f"[DEBUG] Clip {clip_key} (clip_id: {clip_id}) contains {frame_count} frames (end of stream)",
                                flush=True,
                            )
                        _out_vid = release_clip_and_reencode(
                            clip_key,
                            _out_vid,
                            clip_filename,
                            tmp_file,
                            self.target_fps,
                        )
                        if DEBUG == "1":
                            print(
                                f"[TIMING],end_get_clips,{clip_key},{time.time()}",
                                flush=True,
                            )
                        self.clip_end_frame[clip_key] = frameNum
                    break

                frameNum, clip_frame_idx, clip_id, clip_filename, tmp_file, frame = (
                    queue_details
                )
                clip_key = Path(clip_filename).name

                if clip_frame_idx == 0:
                    if DEBUG == "1":
                        print(
                            f"[TIMING],start_get_clips,{clip_key},{time.time()}",
                            flush=True,
                        )
                    _out_vid = cv2.VideoWriter(
                        tmp_file,
                        fourcc=self.fourcc,
                        fps=self.target_fps,
                        frameSize=(self.width, self.height),
                    )
                    if DEBUG == "1":
                        print(
                            f"[TIMING],Start new clip,{clip_key},{time.time()}",
                            flush=True,
                        )
                _out_vid.write(frame)

                if clip_frame_idx == self.clip_total_frames - 1:
                    frame_count = clip_frame_idx + 1
                    if DEBUG == "1":
                        print(
                            f"[DEBUG] Clip {clip_key} (clip_id: {clip_id}) contains {frame_count} frames",
                            flush=True,
                        )
                    _out_vid = release_clip_and_reencode(
                        clip_key,
                        _out_vid,
                        clip_filename,
                        tmp_file,
                        self.target_fps,
                    )
                    if DEBUG == "1":
                        print(
                            f"[TIMING],end_get_clips,{clip_key},{time.time()}",
                            flush=True,
                        )

                    self.clip_end_frame[clip_key] = frameNum
            except queue.Empty:
                pass

    # method to run inference on frame
    def run_inference(self):
        while True:
            try:
                queue_details = self.inference_queue.get()
                # if queue_details is None:
                #     self.metadata_queue.put(None)
                #     break

                frameNum, clip_frame_idx, clip_id, clip_filename, tmp_file, frame = (
                    queue_details
                )
                clip_key = Path(clip_filename).name
                if DEBUG == "1":
                    print(
                        f"[TIMING],start_infer_worker,{clip_key},{time.time()}",
                        flush=True,
                    )
                _, metadata, metadata_face = infer_worker(
                    self.stream_name,
                    clip_frame_idx,
                    frame,
                    # model_path,
                    (self.width, self.height),
                    INGESTION,
                    fps=self.target_fps,
                )
                if DEBUG == "1":
                    print(
                        f"[TIMING],end_infer_worker,{clip_key},{time.time()}",
                        flush=True,
                    )
                self.all_metadata.setdefault(clip_key, {})
                self.all_metadata[clip_key].setdefault("object", {})
                self.all_metadata[clip_key]["object"].update(metadata)
                self.all_metadata[clip_key].setdefault("face", {})
                self.all_metadata[clip_key]["face"].update(metadata_face)
                self.num_frames_processed += 1

                # Make sure clip is processed; frame added to dict after writing clip
                lastFrame = 0
                while True:
                    if clip_key in self.clip_end_frame:
                        lastFrame = self.clip_end_frame[clip_key]
                        break

                # while True:
                if lastFrame == frameNum:
                    # process clip metadata
                    queue_details = (
                        clip_key,
                        clip_filename,
                        self.all_metadata[clip_key],
                        self.width,
                        self.height,
                    )
                    # self.metadata_queue.put(queue_details)
                    clip_key, clip_filename, clip_metadata, width, height = (
                        queue_details
                    )
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
                    # ingest_mode= "object"
                    for ingest_mode in INGESTION.split(","):
                        get_udf_query(
                            # db,
                            # start_t,
                            clip_filename,
                            properties,
                            ingest_mode,
                            (width, height),
                            id="udf_metadata",
                            metadata=clip_metadata[ingest_mode],
                            test_mode=TEST_MODE,
                        )

                    # all_metadata = (
                    #     clip_metadata["object"] if "object" in clip_metadata else {}
                    # )
                    # if "face" in clip_metadata:
                    #     for face_frameidx_bbidx, value in clip_metadata["face"].items():
                    #         face_frameidx, face_bbidx = face_frameidx_bbidx.split("_")
                    #         max_obj_idx = 0
                    #         for obj_frameidx_bbidx in all_metadata:
                    #             if face_frameidx in obj_frameidx_bbidx:
                    #                 _, obj_bbidx_ = obj_frameidx_bbidx.split("_")
                    #                 max_obj_idx = max(max_obj_idx, int(obj_bbidx_))

                    #         if max_obj_idx > 0:
                    #             new_face_bbidx = max_obj_idx + 1
                    #             new_key = f"{face_frameidx}_{new_face_bbidx:04d}"
                    #             all_metadata[new_key] = value
                    #             all_metadata[new_key]["bbId"] = new_key
                    #         else:
                    #             all_metadata[face_frameidx_bbidx] = value

                    # all_metadata = _sort_dict_by_frame(all_metadata)
                    # get_udf_query(
                    #     clip_filename,
                    #     properties,
                    #     INGESTION.replace(",", " "),
                    #     (width, height),
                    #     id="udf_metadata",
                    #     metadata=all_metadata,
                    #     test_mode=TEST_MODE,
                    # )

                    if DEBUG == "1":
                        print(
                            f"[TIMING],end_clip_metadata,{clip_key},{time.time()}",
                            flush=True,
                        )
                # else:
                #     break
            except queue.Empty:
                pass

    # Process metadata for clip
    def process_clip_metadata(self):
        """
        clip must be ready
        all clip frames inference ready
        """
        # db = vdms.vdms()

        while True:
            try:
                queue_details = self.metadata_queue.get()  # noqa: F841

                if queue_details is None:
                    break

                clip_key, clip_filename, clip_metadata, width, height = queue_details
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
                # ingest_mode= "object"
                # for ingest_mode in INGESTION.split(","):
                #     get_udf_query(
                #         # db,
                #         # start_t,
                #         clip_filename,
                #         properties,
                #         ingest_mode,
                #         (width, height),
                #         id="udf_metadata",
                #         metadata=clip_metadata[ingest_mode],
                #         test_mode=TEST_MODE,
                #     )

                all_metadata = (
                    clip_metadata["object"] if "object" in clip_metadata else {}
                )
                if "face" in clip_metadata:
                    for face_frameidx_bbidx, value in clip_metadata["face"].items():
                        face_frameidx, face_bbidx = face_frameidx_bbidx.split("_")
                        max_obj_idx = 0
                        for obj_frameidx_bbidx in all_metadata:
                            if face_frameidx in obj_frameidx_bbidx:
                                _, obj_bbidx_ = obj_frameidx_bbidx.split("_")
                                max_obj_idx = max(max_obj_idx, int(obj_bbidx_))

                        if max_obj_idx > 0:
                            new_face_bbidx = max_obj_idx + 1
                            new_key = f"{face_frameidx}_{new_face_bbidx:04d}"
                            all_metadata[new_key] = value
                            all_metadata[new_key]["bbId"] = new_key
                        else:
                            all_metadata[face_frameidx_bbidx] = value

                all_metadata = _sort_dict_by_frame(all_metadata)
                get_udf_query(
                    clip_filename,
                    properties,
                    INGESTION.replace(",", " "),
                    (width, height),
                    id="udf_metadata",
                    metadata=all_metadata,
                    test_mode=TEST_MODE,
                )

                if DEBUG == "1":
                    print(
                        f"[TIMING],end_clip_metadata,{clip_key},{time.time()}",
                        flush=True,
                    )
            except queue.Empty:
                pass

        # db.disconnect()


""" MAIN FUNCTION """


def processor(camera_src, camera_name=None):
    start = time.time()

    # initializing and starting multi-threaded webcam input stream
    webcam_stream = VideoStream(str(camera_src), camera_name=camera_name)

    # Start retrieving frames and add to queue
    webcam_stream.start()

    webcam_stream.stop()  # stop the webcam stream

    end = time.time()

    # printing time elapsed and fps
    elapsed = end - start
    if DEBUG == "1":
        print(
            "[DEBUG] Stream name:{}, FPS: {} , Elapsed Time: {}, Num. Retrieved Frames: {}, Num. Processed Frames: {}".format(
                webcam_stream.stream_name,
                webcam_stream.target_fps,
                elapsed,
                webcam_stream.retrieved_frames,
                webcam_stream.num_frames_processed,
            ),
            flush=True,
        )

        print(
            f"[TIMING],Completed processing,{webcam_stream.stream_name},{end}",
            flush=True,
        )

    # closing all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        processor(sys.argv[1], camera_name=sys.argv[2])

    elif len(sys.argv) == 2:
        processor(sys.argv[1])

    else:
        raise ValueError("Invalid input. Please provide video path or camera URL")
