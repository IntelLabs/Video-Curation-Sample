import json
import os
import uuid

import cv2
from openvino.runtime import Core
from ultralytics import YOLO
import time

detection_threshold = 0.7
model_w, model_h = (640, 640)
model_precision_object = "FP16"  # FP32, FP16
model_name = "yolo11"  # yolo11, yolov8
half_flag = True
dynamic_flag = True
batch_size = 1

model_precision_face = "FP16"
CV2_INTERPOLATION = cv2.INTER_AREA

yolo_path = f"/home/resources/models/ultralytics/{model_name}/{model_precision_object}/{model_name}n"

device = os.environ.get("DEVICE", "CPU")
DEBUG = os.environ["DEBUG"]
if device == "GPU":
    yolo_path += ".engine/"
else:
    yolo_path += "_openvino_model/"


object_detection_model = YOLO(
    yolo_path,
    verbose=False,
    task="detect",
)


def yolo_object_detection(frame):
    H, W, C = frame.shape
    global object_detection_model
    results = object_detection_model(
        frame, verbose=False, stream=True, conf=detection_threshold
    )

    # Draw bounding boxes on the image
    objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf.item())
            if confidence > detection_threshold:
                class_id = box.cls.item()
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                height = min(y2, H - 1) - max(0, y1)
                width = min(x2, W - 1) - max(0, x1)
                object_res = [
                    x1,
                    y1,
                    height,
                    width,
                    object_detection_model.names[class_id],
                    confidence,
                ]
                # print(object_res)
                objects.append(object_res)

    return objects


ie = Core()
face_detection_model_xml = f"/home/resources/models/intel/face-detection-adas-0001/{model_precision_face}/face-detection-adas-0001.xml"
face_detection_model = ie.read_model(
    model=face_detection_model_xml,
    weights=face_detection_model_xml.replace(".xml", ".bin"),
)
# face_det_w, face_det_h = 672, 384
_, face_det_c, face_det_h, face_det_w = face_detection_model.inputs[0].shape
face_det_compiled_model = ie.compile_model(face_detection_model, device)

age_gender_classification_model_xml = f"/home/resources/models/intel/age-gender-recognition-retail-0013/{model_precision_face}/age-gender-recognition-retail-0013.xml"
age_gender_classification_model = ie.read_model(
    model=age_gender_classification_model_xml,
    weights=age_gender_classification_model_xml.replace(".xml", ".bin"),
)
# ag_w, ag_h = 62, 62
_, ag_c, ag_h, ag_w = age_gender_classification_model.inputs[0].shape
ag_compiled_model = ie.compile_model(age_gender_classification_model, device)

emotions_classification_model_xml = f"/home/resources/models/intel/emotions-recognition-retail-0003/{model_precision_face}/emotions-recognition-retail-0003.xml"
emotions_classification_model = ie.read_model(
    model=emotions_classification_model_xml,
    weights=emotions_classification_model_xml.replace(".xml", ".bin"),
)
# em_w, em_h = 64, 64
_, em_c, em_h, em_w = emotions_classification_model.inputs[0].shape
em_compiled_model = ie.compile_model(emotions_classification_model, device)


def face_detection(frame):
    H, W, C = frame.shape
    # Model expects BGRA
    # face detect -> age-gender -> emotions
    global face_det_compiled_model, ag_compiled_model, em_compiled_model
    genders = ["female", "male"]
    emotions = ["neutral", "happy", "sad", "surprise", "anger"]

    # input_layer = face_det_compiled_model.input(0)
    # Resize expects HWC
    input_image = cv2.resize(
        frame, (face_det_w, face_det_h), interpolation=CV2_INTERPOLATION
    )
    input_image = input_image.transpose(2, 0, 1)  # Shape: CHW
    input_image = input_image.reshape((batch_size, face_det_c, face_det_h, face_det_w))

    output_layer = face_det_compiled_model.output(0)
    result = face_det_compiled_model([input_image])[output_layer]

    # Process the detections
    faces = []
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
                    face_roi, (ag_w, ag_h), interpolation=CV2_INTERPOLATION
                )
                ag_face_blob = ag_face_blob.transpose((2, 0, 1))
                ag_face_blob = ag_face_blob.reshape((batch_size, ag_c, ag_h, ag_w))
                ag_result = ag_compiled_model([ag_face_blob])
                age = int(ag_result["fc3_a"].flatten() * 100)
                gender = str(genders[ag_result["prob"].argmax()])
            except Exception as e:
                print(f"Error occurred: {e}. Skipping age-gender model")

            try:
                em_face_blob = cv2.resize(
                    face_roi, (em_w, em_h), interpolation=CV2_INTERPOLATION
                )
                em_face_blob = em_face_blob.transpose((2, 0, 1))
                em_face_blob = em_face_blob.reshape((batch_size, em_c, em_h, em_w))
                em_result = em_compiled_model([em_face_blob])[
                    em_compiled_model.output(0)
                ]
                emotion = str(emotions[em_result.argmax()])
            except Exception as e:
                print(f"Error occurred: {e}. Skipping emotion model")
            face_res = [x1, y1, height, width, age, gender, emotion, confidence]
            # print(face_res)
            faces.append(face_res)
    return faces


# SUPPORTS VIDEOS ONLY
def run(ipfilename, format, options, resize_input=False):
    if DEBUG == "1":
        print(f"[TIMING],start_udf_metadata,{ipfilename},"+str(time.time()), flush=True)
    metadata = dict()
    video_obj = cv2.VideoCapture(ipfilename)

    if not video_obj.isOpened():
        print(f"[!] Error opening video ({ipfilename})")

    while True:
        (grabbed, frame) = video_obj.read()
        if not grabbed:
            break  # No more frames are read

        frameNum = int(video_obj.get(cv2.CAP_PROP_POS_FRAMES))

        if resize_input:
            frame = cv2.resize(
                frame, (model_w, model_h), interpolation=CV2_INTERPOLATION
            )

        H, W, _ = frame.shape
        if frame is not None and options["otype"] == "face":
            # face detection for each frame
            faces = face_detection(frame)
            for face in faces:
                tdict = {
                    "x": int(face[0]),
                    "y": int(face[1]),
                    "height": int(face[2]),
                    "width": int(face[3]),
                    "object": "face",
                    "object_det": {
                        "age": int(face[4]),
                        "gender": str(face[5]),
                        "emotion": str(face[6]),
                        "confidence": float(face[7]),
                    },
                }

                metadata[frameNum] = {"frameId": frameNum, "bbox": tdict}

        elif frame is not None:
            # object detection
            objects = yolo_object_detection(frame)
            for object in objects:
                tdict = {
                    "x": int(object[0]),
                    "y": int(object[1]),
                    "height": int(object[2]),
                    "width": int(object[3]),
                    "object": str(object[4]),
                    "object_det": {"confidence": float(object[5])},
                }

                metadata[frameNum] = {"frameId": frameNum, "bbox": tdict}
                meta_str = ",".join([str(o) for o in object])
                if DEBUG == "1":
                    print(f"[METADATA],{meta_str}", flush=True)

    video_obj.release()

    response = {"opFile": ipfilename, "metadata": metadata}

    jsonfile = "jsonfile" + uuid.uuid1().hex + ".json"
    with open(jsonfile, "w") as f:
        json.dump(response, f, indent=4)

    if DEBUG == "1":
        print(f"[TIMING],end_udf_metadata,{ipfilename},"+str(time.time()), flush=True)
    return ipfilename, jsonfile
