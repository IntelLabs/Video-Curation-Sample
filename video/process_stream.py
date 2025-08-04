import os
import shlex
import subprocess
import sys
import time  # time library
import traceback
from pathlib import Path
from threading import Thread  # library for multi-threading

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


def extract_metadata(frameNum, results, img_size, fps=TARGET_FPS):
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
                        print(f"[METADATA],{meta_str}", flush=True)

                    metadata[framenum_str] = {
                        "frameId": frameNum,
                        "bbox": tdict,
                    }
                    oidx += 1

    except Exception:
        e = traceback.format_exc()
        print(f"Error in extract_metadata: {e}", flush=True)

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
    except Exception:
        e = traceback.format_exc()
        print(f"[DEBUG] VDMS Query Exception: {e}", flush=True)
        # print(f"[DEBUG] failed query: {query}", flush=True)

    # elapsed_time = time.time() - start_t

    if DEBUG_FLAG:
        print(
            f"[TIMING],end_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
        print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
        print(f"[DEBUG] INGEST_VIDEO RESPONSE: {res}", flush=True)
        # print(f"[DEBUG] Used client: {dn_name}", flush=True)
        # print(f"[DEBUG] Elapsed ingest_video time: {elapsed_time} sec", flush=True)
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


def face_detection(frameNum, frame, img_size):
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
                age = int(ag_result["fc3_a"].flatten() * 100)
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
                print(f"[METADATA],{meta_str}", flush=True)

            metadata[framenum_str] = {"frameId": frameNum, "bbox": tdict}
            oidx += 1

    return metadata


# defining a helper class for implementing multi-threading
class VideoStream:
    # initialization method
    def __init__(self, src, fps=TARGET_FPS, fourcc=fourcc, camera_name=None):
        self.stream_id = src  # default is 0 for main camera

        # os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"]="vcodec;x264|preset;medium|crf;23"
        if self.stream_id.startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        # opening video capture stream
        self.video_obj = cv2.VideoCapture(self.stream_id, cv2.CAP_FFMPEG)
        # if self.video_obj.isOpened() is False :
        #     print("[Exiting]: Error accessing webcam stream.")
        #     exit(0)

        # Check that object is opened successfully
        stream_available = False
        while not stream_available:
            if self.video_obj.isOpened():
                stream_available = True

        self.input_fps = int(self.video_obj.get(cv2.CAP_PROP_FPS))  # hardware fps
        input_width = int(self.video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(self.video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.target_fps = fps if self.input_fps > fps else self.input_fps

        self.frame_skip = int(self.input_fps / self.target_fps)

        print("FPS of input stream: {}".format(self.input_fps), flush=True)
        print("FPS of output mp4: {}".format(self.target_fps), flush=True)

        self.clip_length_in_secs = 10
        self.clip_total_frames = int(float(self.clip_length_in_secs * self.target_fps))

        if RESIZE_FLAG or ((input_height * input_width) < (MODEL_H * MODEL_W)):
            new_sizeHW = check_imgsz([MODEL_H, MODEL_W])  # expects hxw
        else:
            new_sizeHW = check_imgsz([input_height, input_width])  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]  # self.video_obj.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = new_sizeWH[1]  # self.video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._out_vid = None
        self.clip_filename = ""
        self.fourcc = fourcc
        self.clip_id = 0
        self.clip_frame_count = 0
        self.clip_frame_inds = []

        self.all_metadata = {}
        self.stream_name = camera_name
        if "://" in str(self.stream_id):
            # self.src_id = src.split("/")[-1]
            if camera_name is not None:
                self.file_prefix = camera_name
            else:
                self.file_prefix = str(self.stream_id).split("/")[-1]
            self.frame_count = None
        else:
            # self.src_id = Path(src).stem
            self.file_prefix = Path(self.stream_id).stem
            self.frame_count = int(self.video_obj.get(cv2.CAP_PROP_FRAME_COUNT))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.video_obj.read()
        if self.grabbed is False:
            print("[Exiting] No more frames to read", flush=True)
            exit(0)

        # self.stopped is initialized to False
        self.stopped = True

        self.process_frame()

        # thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method to process a frame
    def process_frame(self):
        frameNum = int(self.video_obj.get(cv2.CAP_PROP_POS_FRAMES))
        mod_frameNum = ((frameNum - 1) % self.clip_total_frames) + 1
        skip_frame_num = (frameNum - 1) % self.frame_skip
        # incr_clip = False
        # print(f"mod_frameNum: {mod_frameNum}\tframeNum: {frameNum}", flush=True)

        if (skip_frame_num == 0) and len(self.clip_frame_inds) < self.clip_total_frames:
            if self._out_vid is None and len(self.clip_frame_inds) == 0:
                self.clip_filename = (
                    f"{SHARED_OUTPUT}/{self.file_prefix}_{self.clip_id}.mp4"
                )
                self.clip_frame_count = 0
                tmp_file = "/var/www/streams/" + self.clip_filename.split("/")[-1]
                self._out_vid = cv2.VideoWriter(
                    tmp_file,
                    fourcc=self.fourcc,
                    fps=self.target_fps,
                    frameSize=(self.width, self.height),
                )
                print(f"Create clip @ frame {frameNum}", flush=True)
            self.clip_frame_inds.append(frameNum)
            # clip_frame_count = len(self.clip_frame_inds)
            # print(f"frameNum: {frameNum}\tmod_frameNum: {mod_frameNum}\tskip_frame_num: {skip_frame_num}\tclip_frame_count: {clip_frame_count}", flush=True)

        annotated, metadata, metadata_face = infer_worker(
            mod_frameNum,
            self.frame,
            model_path,
            (self.width, self.height),
            INGESTION,
            fps=self.target_fps,
        )
        clip_key = Path(self.clip_filename).name
        self.all_metadata.setdefault(clip_key, {})
        self.all_metadata[clip_key].setdefault("object", {})
        self.all_metadata[clip_key]["object"].update(metadata)
        self.all_metadata[clip_key].setdefault("face", {})
        self.all_metadata[clip_key]["face"].update(metadata_face)

        if (self._out_vid is not None) and (skip_frame_num == 0):
            self._out_vid.write(annotated)

        if len(self.clip_frame_inds) == self.clip_total_frames or (
            frameNum == self.frame_count
        ):
            minf = min(self.clip_frame_inds)
            maxf = max(self.clip_frame_inds)
            frame_count = len(self.clip_frame_inds)
            print(f"Clip contains {frame_count} frames {minf} - {maxf}", flush=True)
            self.clip_frame_inds = []
            if self._out_vid is not None:
                # self._out_vid.write(self.frame)
                self._out_vid.release()

                # Re-encode video in order to seek via ffmpeg later
                tmp_file = "/var/www/streams/" + self.clip_filename.split("/")[-1]
                reencode_cmd = f"ffmpeg -y -i {tmp_file} -c:v libx264 -preset medium -crf 23 -c:a copy {self.clip_filename}"
                cmd_list = shlex.split(reencode_cmd)
                subprocess.run(cmd_list, check=True)

                print("Saved ", self.clip_filename, flush=True)
                os.remove(tmp_file)
                self.clip_id += 1

                # Send metadata to UDF
                properties = {
                    "Name": Path(self.clip_filename).name,  # .split("/")[-1],
                    "category": "video_path_rop",
                }
                # ingest_mode= "object"
                for ingest_mode in INGESTION.split(","):
                    get_udf_query(
                        # start_t,
                        self.clip_filename,
                        properties,
                        ingest_mode,
                        (self.width, self.height),
                        id="udf_metadata",
                        metadata=self.all_metadata[clip_key][ingest_mode],
                        test_mode=TEST_MODE,
                    )

            self._out_vid = None
            self.clip_filename = ""
            self.clip_frame_count = 0

        # if ((frameNum - 1) % self.clip_total_frames < (self.clip_total_frames - 1)):
        #     if self._out_vid is None:
        #         self.clip_filename = f"{SHARED_OUTPUT}/{self.file_prefix}_{self.clip_id}.mp4"
        #         self.clip_frame_count = 0
        #         self._out_vid = cv2.VideoWriter(
        #         self.clip_filename,
        #         fourcc=self.fourcc,
        #         fps=self.target_fps,
        #         frameSize=(self.width, self.height),
        #     )
        #     # self._out_vid.write(self.frame)

        # annotated, metadata, metadata_face = infer_worker(mod_frameNum, self.frame, model_path, (self.width, self.height), INGESTION, fps=self.target_fps)
        # clip_key = Path(self.clip_filename).name
        # self.all_metadata.setdefault(clip_key, {})
        # self.all_metadata[clip_key].setdefault("object", {})
        # self.all_metadata[clip_key]["object"].update(metadata)
        # self.all_metadata[clip_key].setdefault("face", {})
        # self.all_metadata[clip_key]["face"].update(metadata_face)

        # if (self._out_vid is not None) and (self.clip_frame_count % self.frame_skip == 0):
        #     self._out_vid.write(annotated)

        # self.clip_frame_count += 1

        # if (
        #     (mod_frameNum - 1) % self.clip_total_frames == (self.clip_total_frames - 1)
        # ) or (frameNum == self.frame_count):
        #     if self._out_vid is not None:
        #         # self._out_vid.write(self.frame)
        #         self._out_vid.release()
        #         print(f"Saved ", self.clip_filename, flush=True)
        #         self.clip_id += 1

        #         # Send metadata to UDF
        #         properties = {
        #             "Name": Path(self.clip_filename).name,  # .split("/")[-1],
        #             "category": "video_path_rop",
        #         }
        #         # ingest_mode= "object"
        #         for ingest_mode in INGESTION.split(","):
        #             get_udf_query(
        #                 # start_t,
        #                 self.clip_filename,
        #                 properties,
        #                 ingest_mode,
        #                 (self.width, self.height),
        #                 id="udf_metadata",
        #                 metadata=self.all_metadata[clip_key][ingest_mode],
        #                 # test_mode=True,
        #             )

        #     self._out_vid = None
        #     self.clip_filename = ""
        #     self.clip_frame_count = 0

    # method passed to thread to read next available frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.video_obj.read()
            if not self.grabbed or self.frame is None:
                print("[Exiting] No more frames to read", flush=True)
                self.stopped = True
                break

            self.process_frame()

        if self._out_vid is not None:
            self._out_vid.release()
            print("Saved ", self.clip_filename, flush=True)
            self.clip_id += 1

    # method to return latest read frame
    def read(self):
        return self.frame

    # method to stop reading frames
    def stop(self):
        self.stopped = True
        self.video_obj.release()


# ---------- Inference Function ----------
def infer_worker(
    frameNum,
    frame,
    model_path,
    img_size,
    INGESTION,
    fps=TARGET_FPS,
    return_annotated=False,
):  # img_size:(W,H)
    global model
    # model = YOLO(model_path, verbose=False, task="detect")
    # model.fuse()  # Model fusion speeds up inference (.pt only)
    # while True:
    # item = frame_queue.get()
    # if item[1] is None:
    #     print("Inference worker received None, exiting.", flush=True)
    #     break
    # cam_id, frame, frameNum = item
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
        annotated, metadata = extract_metadata(frameNum, results, img_size, fps=fps)

    if "face" in INGESTION:
        metadata_face = face_detection(frameNum, frame, img_size)

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
    # camera_srcs = [TEST_VIDEO_PATH]  # Video path or camera URL
    # camera_srcs = ["rtsp://0.0.0.0:8554/live1"]
    # camera_srcs = ["udp://0.0.0.0:30009"]
    # camera_ids = []
    # for src in camera_srcs:
    #     if "://" in str(src):
    #         camera_ids.append(str(src).split("/")[-1])
    #     else:
    #         camera_ids.append(Path(src).stem)

    # start = time.time()
    # initializing and starting multi-threaded webcam input stream
    webcam_stream = VideoStream(
        str(camera_src), camera_name=camera_name
    )  # 0 id for main camera
    webcam_stream.start()
    # processing frames in input stream
    num_frames_processed = 0

    start = time.time()
    while True:
        if webcam_stream.stopped is True:
            break
        else:
            frame = webcam_stream.read()  # noqa: F841

        # frameNum = int(webcam_stream.video_obj.get(cv2.CAP_PROP_POS_FRAMES))

        # adding a delay for simulating video processing time
        # delay = 0.03 # delay value in seconds
        # time.sleep(delay)

        num_frames_processed += 1
        # displaying frame
        # cv2.imshow('frame' , frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # end = time.time()

    # if webcam_stream._out_vid is not None:
    #     webcam_stream._out_vid.release()
    #     print(f"Saved ", webcam_stream.clip_filename)
    webcam_stream.stop()  # stop the webcam stream

    end = time.time()

    # printing time elapsed and fps
    elapsed = end - start
    # fps = num_frames_processed/elapsed
    print(
        "FPS: {} , Elapsed Time: {} ".format(webcam_stream.target_fps, elapsed),
        flush=True,
    )
    # closing all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if isinstance(sys.argv[1], dict):
            processor(sys.argv[1], camera_name=sys.argv[2])
            print(f"Completed processing {sys.argv[2]}", flush=True)
        else:
            processor(sys.argv[1])
            print(f"Completed processing {sys.argv[1]}", flush=True)
    else:
        raise ValueError("Invalid input. Please provide video path or camera URL")
