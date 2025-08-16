import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time  # time library
import traceback
from pathlib import Path
from threading import Lock, Thread  # library for multi-threading

import cv2  # OpenCV library
import psutil
from openvino.runtime import Core
from segment_archive import str2bool
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

import vdms

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
CODE_DIR = os.getenv("CODE_DIR", "/home")
DBHOST = "vdms-service"  # os.environ["DBHOST"]
DBPORT = 55555
DEBUG = os.environ["DEBUG"]
DEBUG_FLAG = True if DEBUG == "1" else False
DEVICE = os.environ["DEVICE"]
DEVICE_OV = "AUTO"
INGESTION = os.environ["INGESTION"]
MODEL_PRECISION = "FP16"
MODEL_W, MODEL_H = (640, 640)
RESIZE_FLAG = str2bool(os.getenv("RESIZE_FLAG", False))
SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
TMP_LOCATION = os.getenv("TMP_LOCATION", "/var/www/streams/")
TARGET_FPS = 15  # 15  30
TEST_MODE = str2bool(os.getenv("TEST_FLAG", False))
UDF_HOST = "video-service"
UDF_PORT = 5011

batch_size = 1
detection_threshold = 0.25  # 0.7
half_flag = True
iou_threshold = 0.7  # 0.9  # 0.5

model_path = f"{CODE_DIR}/resources/models/ultralytics/yolo11/{MODEL_PRECISION}/yolo11n_openvino_model"
model = YOLO(model_path, verbose=False, task="detect")


def extract_metadata(stream_name, frameNum, results, img_size, fps=TARGET_FPS):
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
                if confidence > detection_threshold:
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

                    framenum_str = f"{frameNum}_{oidx}"
                    if DEBUG_FLAG:
                        meta_str = ",".join(
                            [str(o) for o in object_res + [framenum_str]]
                        )
                        print(f"[{stream_name} METADATA],{meta_str}", flush=True)

                    metadata[framenum_str] = {
                        "frameId": frameNum,
                        "bbox": tdict,
                    }
                    oidx += 1

    except Exception:
        e = traceback.format_exc()
        print(f"Error in {stream_name} extract_metadata: {e}", flush=True)

    return annotated, metadata


def get_udf_query(
    # start_t,
    filename_path,
    properties,
    ingest_mode,
    new_size,
    id="metadata_callback",
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
                        # "id": "metadata_callback",
                        # "id": "metadata_splitter_callback",
                        "otype": ingest_mode,
                        "media_type": "video",
                        # "fps": properties["fps"],
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

    video_blob = []
    # with open(filename_path, "rb") as fd:
    #     video_blob.append(fd.read())
    # return query, video_blob

    filename = str(Path(filename_path).name)
    # dn_name = filename.split("__")[0]

    # if dn_name not in dbs:
    #     dbs[dn_name] = vdms.vdms()
    #     dbs[dn_name].connect(DBHOST, DBPORT)
    # elif not dbs[dn_name].is_connected():
    #     dbs[dn_name].connect(DBHOST, DBPORT)
    db = vdms.vdms()
    db.connect(DBHOST, DBPORT)
    if DEBUG_FLAG:
        print(
            f"[TIMING],start_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
    try:
        res, res_arr = db.query([query], [video_blob])

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
        if confidence > detection_threshold:
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
            framenum_str = f"{frameNum}_{oidx}"
            if DEBUG_FLAG:
                meta_str = ",".join([str(o) for o in face_res + [framenum_str]])
                print(f"[{stream_name} METADATA],{meta_str}", flush=True)

            metadata[framenum_str] = {"frameId": frameNum, "bbox": tdict}
            oidx += 1

    return metadata


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


def metadata_to_udf(clip_key, clip_filename, all_metadata, width, height):
    # Send metadata to UDF
    properties = {
        "Name": clip_key,  # .split("/")[-1],
        "category": "video_path_rop",
    }
    # ingest_mode= "object"
    for ingest_mode in INGESTION.split(","):
        get_udf_query(
            # start_t,
            clip_filename,
            properties,
            ingest_mode,
            (width, height),
            id="udf_metadata",
            metadata=all_metadata[ingest_mode],
            test_mode=TEST_MODE,
        )


def write_clip_and_process(clip_key, _out_vid, clip_filename, tmp_file, target_fps):
    _out_vid.release()
    _out_vid = None

    # Re-encode video in order to seek via ffmpeg later
    GENERAL_OPTS = "-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -flush_packets 0"  #  -filter:v fps={target_fps}
    CONVERSION = f"-c:v libx264 -preset ultrafast -filter:v fps=fps={target_fps}"  # "-c:v libx264 -preset medium"
    reencode_cmd = f"ffmpeg -y -i {tmp_file} {GENERAL_OPTS} {CONVERSION} -crf 23 -c:a copy {clip_filename}"
    cmd_list = shlex.split(reencode_cmd)
    subprocess.run(cmd_list, check=True)

    # filename = str(Path(clip_filename).name)
    print(f"[TIMING],Save clip,{clip_key},{time.time()}", flush=True)
    os.remove(tmp_file)
    # self.clip_id += 1

    # # Send metadata to UDF
    # properties = {
    #     "Name": clip_key,  # .split("/")[-1],
    #     "category": "video_path_rop",
    # }
    # # ingest_mode= "object"
    # for ingest_mode in INGESTION.split(","):
    #     get_udf_query(
    #         # start_t,
    #         clip_filename,
    #         properties,
    #         ingest_mode,
    #         (width, height),
    #         id="udf_metadata",
    #         metadata=all_metadata[ingest_mode],
    #         test_mode=TEST_MODE,
    #     )

    # clip_filename = ""
    # self.clip_frame_count = 0
    return _out_vid


# defining a helper class for implementing multi-threading
class VideoStream:
    # initialization method
    def __init__(self, src, fps=TARGET_FPS, fourcc=fourcc, camera_name=None):
        self.stream_id = src  # default is 0 for main camera

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

        print(f"[TIMING],Start processing,{self.stream_name},{time.time()}", flush=True)

        self.setup_stream(fps)

        # reading a single frame from stream for initializing
        # self.grabbed, self.frame = self.video_obj.read()
        # if self.grabbed is False:
        #     print(f"[Exiting {self.stream_name}] No more frames to read", flush=True)
        #     exit(0)
        # with self._lock:
        # grabbed, frame = self.video_obj.read()
        # frameNum = int(self.video_obj.get(cv2.CAP_PROP_POS_FRAMES))
        # self.process_frame(frame, frameNum)
        # self.retrieved_frames += 1

        # self.stopped is initialized to False
        self.stopped = True

        # self.process_frame()

        # thread instantiation
        # self.t = Thread(target=self.update, args=(), daemon=True)
        # self.t.daemon = True  # daemon threads run in background
        self.t = Thread(target=self.get_frames, args=(), daemon=True)

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

    def get_frameWH(self):
        input_width = int(self.video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(self.video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if RESIZE_FLAG or ((input_height * input_width) < (MODEL_H * MODEL_W)):
            new_sizeHW = check_imgsz([MODEL_H, MODEL_W])  # expects hxw
        else:
            new_sizeHW = check_imgsz([input_height, input_width])  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]  # self.video_obj.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = new_sizeWH[1]

    # Sets up important info for stream
    def setup_stream(self, fps):
        self._lock = Lock()
        self.file_queue = mp.Queue()
        self.retrieved_frames = 0

        self.get_fps_and_framecnt(fps)

        self.get_frameWH()

        self._out_vid = None
        self.all_metadata = {}
        self.clip_filename = ""
        self.clip_frame_count = 0
        self.clip_frame_inds = []
        self.clip_id = 0
        self.clip_length_in_secs = 10
        self.clip_total_frames = int(float(self.clip_length_in_secs * self.target_fps))
        self.fourcc = fourcc

    # method to process a frame
    def get_frames(self):
        clip_frame_idx = 0
        clip_id = 0
        _out_vid = None
        while True:
            if self.stopped:
                break

            grabbed, frame = self.video_obj.read()  # Read next frame

            if not grabbed or frame is None:
                # print(f"[Exiting {self.stream_name}] No more frames to read", flush=True)
                self.stopped = True
                self.file_queue.put(None)
                # break
            else:
                self.retrieved_frames += 1
                frameNum = int(self.video_obj.get(cv2.CAP_PROP_POS_FRAMES))
                skip_frame_num = (frameNum - 1) % self.frame_skip

                if clip_frame_idx % self.clip_total_frames == 0:
                    if "://" not in str(self.stream_id):
                        clip_filename = (
                            f"{SHARED_OUTPUT}/{self.stream_name}_{clip_id}.mp4"
                        )
                    else:
                        clip_filename = (
                            f"{SHARED_OUTPUT}/{self.stream_name}_{time.time()}.mp4"
                        )

                    tmp_file = TMP_LOCATION + clip_filename.split("/")[-1]

                if skip_frame_num == 0:
                    queue_details = (
                        frameNum,  # Overall frame number
                        clip_frame_idx % self.clip_total_frames,  # Frame index in clip
                        clip_id,  # Clip number
                        clip_filename,
                        tmp_file,
                        frame,  # Frame
                    )
                    self.file_queue.put(queue_details)
                    self.retrieved_frames += 1
                    clip_frame_idx += 1
                    if clip_frame_idx % self.clip_total_frames == 0:
                        clip_id += 1

    # method to return latest read frame
    def read(self):
        # return self.frame
        queue_details = self.file_queue.get()
        return queue_details

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method to stop reading frames
    def stop(self):
        # self.t.join()
        self.stopped = True
        self.video_obj.release()


# ---------- Inference Function ----------
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
    if frame.shape != img_size:
        frame = cv2.resize(frame, img_size)

    annotated = None
    metadata = {}
    metadata_face = {}
    if "object" in INGESTION:
        results = model.predict(
            frame,
            imgsz=(img_size[1], img_size[0]),
            batch=batch_size,
            conf=detection_threshold,
            iou=iou_threshold,
            half=half_flag,
            device=DEVICE,
            verbose=False,
            stream=True,
        )
        # results = model.predict(frame, verbose=False, device=DEVICE)
        # result_queue.put((cam_id, frame, frameNum, results[0]))
        # print(f"result_queue (cam_id, frame, frameNum, results[0]): {cam_id}, frame, {frameNum}, results[0]")
        # try:
        annotated, metadata = extract_metadata(
            stream_name, frameNum, results, img_size, fps=fps
        )

    if "face" in INGESTION:
        metadata_face = face_detection(stream_name, frameNum, frame, img_size)

    if return_annotated:
        return annotated, metadata, metadata_face
    else:
        return frame, metadata, metadata_face


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


# ---------- Processor ----------
def processor(camera_src, camera_name=None):
    start = time.time()

    # initializing and starting multi-threaded webcam input stream
    webcam_stream = VideoStream(str(camera_src), camera_name=camera_name)

    # Start retrieving frames and add to queue
    webcam_stream.start()

    # processing frames in input stream
    num_frames_processed = 0

    # # start = time.time()
    _out_vid = None
    clip_frame_idx = 0
    clip_key = ""
    clip_filename = ""
    tmp_file = ""
    clip_id = 0
    while True:
        # if webcam_stream.stopped:
        #     break
        # else:
        queue_details = webcam_stream.read()  # noqa: F841
        if queue_details is None:
            if _out_vid is not None:
                frame_count = clip_frame_idx + 1
                print(
                    f"[DEBUG] Clip {clip_key} (clip_id: {clip_id}) contains {frame_count} frames (end of stream)",
                    flush=True,
                )
                _out_vid = write_clip_and_process(
                    clip_key,
                    _out_vid,
                    clip_filename,
                    tmp_file,
                    webcam_stream.target_fps,
                )
                metadata_to_udf(
                    clip_key,
                    clip_filename,
                    webcam_stream.all_metadata[clip_key],
                    webcam_stream.width,
                    webcam_stream.height,
                )
            break

        frameNum, clip_frame_idx, clip_id, clip_filename, tmp_file, frame = (
            queue_details
        )

        # print(f"frameNum: {frameNum}, clip_frame_idx: {clip_frame_idx}, clip_id: {clip_id}", flush=True)

        if clip_frame_idx == 0:
            _out_vid = cv2.VideoWriter(
                tmp_file,
                fourcc=webcam_stream.fourcc,
                fps=webcam_stream.target_fps,
                frameSize=(webcam_stream.width, webcam_stream.height),
            )
            print(
                f"[TIMING],Start new clip,{Path(clip_filename).name},{time.time()}",
                flush=True,
            )

        clip_key = Path(clip_filename).name
        annotated, metadata, metadata_face = infer_worker(
            webcam_stream.stream_name,
            clip_frame_idx,
            frame,
            # model_path,
            (webcam_stream.width, webcam_stream.height),
            INGESTION,
            fps=webcam_stream.target_fps,
        )
        webcam_stream.all_metadata.setdefault(clip_key, {})
        webcam_stream.all_metadata[clip_key].setdefault("object", {})
        webcam_stream.all_metadata[clip_key]["object"].update(metadata)
        webcam_stream.all_metadata[clip_key].setdefault("face", {})
        webcam_stream.all_metadata[clip_key]["face"].update(metadata_face)
        _out_vid.write(annotated)
        num_frames_processed += 1

        if clip_frame_idx == webcam_stream.clip_total_frames - 1:
            frame_count = clip_frame_idx + 1
            print(
                f"[DEBUG] Clip {clip_key} (clip_id: {clip_id}) contains {frame_count} frames",
                flush=True,
            )
            _out_vid = write_clip_and_process(
                clip_key,
                _out_vid,
                clip_filename,
                tmp_file,
                webcam_stream.target_fps,
            )
            metadata_to_udf(
                clip_key,
                clip_filename,
                webcam_stream.all_metadata[clip_key],
                webcam_stream.width,
                webcam_stream.height,
            )

    # if _out_vid is not None:
    #     frame_count = clip_frame_idx + 1
    #     print(f"[DEBUG] Clip {clip_key} (clip_id: {clip_id}) contains {frame_count} frames (end of stream)", flush=True)
    #     _out_vid = write_clip_and_process(
    #         clip_key, _out_vid, clip_filename, tmp_file, webcam_stream.target_fps,
    #     )
    #     metadata_to_udf(clip_key, clip_filename, webcam_stream.all_metadata[clip_key], webcam_stream.width, webcam_stream.height)

    webcam_stream.stop()  # stop the webcam stream

    end = time.time()

    # printing time elapsed and fps
    elapsed = end - start
    # fps = num_frames_processed/elapsed
    print(
        "[DEBUG] Stream name:{}, FPS: {} , Elapsed Time: {}, Num. Retrieved Frames: {}, Num. Processed Frames: {}".format(
            webcam_stream.stream_name,
            webcam_stream.target_fps,
            elapsed,
            webcam_stream.retrieved_frames,
            num_frames_processed,
        ),
        flush=True,
    )

    print(
        f"[TIMING],Completed processing,{webcam_stream.stream_name},{end}", flush=True
    )

    # closing all windows
    # webcam_stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        processor(sys.argv[1], camera_name=sys.argv[2])

    elif len(sys.argv) == 2:
        processor(sys.argv[1])

    else:
        raise ValueError("Invalid input. Please provide video path or camera URL")
