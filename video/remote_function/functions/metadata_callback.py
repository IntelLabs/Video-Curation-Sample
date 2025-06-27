import asyncio
import json
import os
import time
import uuid

import cv2
from openvino.runtime import Core
from ultralytics import YOLO

detection_threshold = 0.7
iou_threshold = 0.5
model_w, model_h = (640, 640)
model_precision_object = "FP16"
model_name = "yolo11"
half_flag = True
dynamic_flag = True

model_precision_face = "FP16"
CV2_INTERPOLATION = cv2.INTER_AREA
DEVICE = os.environ.get("DEVICE", "CPU")
DEVICE_OV = "AUTO"
DEBUG = os.environ.get("DEBUG", "0")
device_input = DEVICE.lower() if DEVICE == "CPU" else 0

yolo_path = f"/home/resources/models/ultralytics/{model_name}/{model_precision_object}/{model_name}n"

if DEVICE == "GPU":
    yolo_path += ".engine"
    batch_size = 1
else:
    yolo_path += "_openvino_model/"
    batch_size = 8


""" MODEL DEFINITIONS """

ie = Core()
face_detection_model_xml = f"/home/resources/models/intel/face-detection-adas-0001/{model_precision_face}/face-detection-adas-0001.xml"
face_detection_model = ie.read_model(
    model=face_detection_model_xml,
    weights=face_detection_model_xml.replace(".xml", ".bin"),
)
# face_det_w, face_det_h = 672, 384
_, face_det_c, face_det_h, face_det_w = face_detection_model.inputs[0].shape
face_det_compiled_model = ie.compile_model(face_detection_model, DEVICE_OV)

age_gender_classification_model_xml = f"/home/resources/models/intel/age-gender-recognition-retail-0013/{model_precision_face}/age-gender-recognition-retail-0013.xml"
age_gender_classification_model = ie.read_model(
    model=age_gender_classification_model_xml,
    weights=age_gender_classification_model_xml.replace(".xml", ".bin"),
)
# ag_w, ag_h = 62, 62
_, ag_c, ag_h, ag_w = age_gender_classification_model.inputs[0].shape
ag_compiled_model = ie.compile_model(age_gender_classification_model, DEVICE_OV)

emotions_classification_model_xml = f"/home/resources/models/intel/emotions-recognition-retail-0003/{model_precision_face}/emotions-recognition-retail-0003.xml"
emotions_classification_model = ie.read_model(
    model=emotions_classification_model_xml,
    weights=emotions_classification_model_xml.replace(".xml", ".bin"),
)
# em_w, em_h = 64, 64
_, em_c, em_h, em_w = emotions_classification_model.inputs[0].shape
em_compiled_model = ie.compile_model(emotions_classification_model, DEVICE_OV)


""" DETECTION FUNCTIONS """


def face_detection(frame, H, W):
    bs = 1
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
    input_image = input_image.reshape((bs, face_det_c, face_det_h, face_det_w))

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
                ag_face_blob = ag_face_blob.reshape((bs, ag_c, ag_h, ag_w))
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
                em_face_blob = em_face_blob.reshape((bs, em_c, em_h, em_w))
                em_result = em_compiled_model([em_face_blob])[
                    em_compiled_model.output(0)
                ]
                emotion = str(emotions[em_result.argmax()])
            except Exception as e:
                print(f"Error occurred: {e}. Skipping emotion model")
            face_res = [x1, y1, height, width, age, gender, emotion, confidence, H, W]
            # print(face_res)
            faces.append(face_res)

    return faces


""" MAIN FUNCTION """


def run(ipfilename, format, options, tmp_dir_path):
    METADATA = dict()
    W, H = options["input_sizeWH"]

    async def update_face_metadata(results, framenum):
        for oidx, face in enumerate(results):
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
                    "frameH": int(H),
                    "frameW": int(W),
                },
            }
            framenum_str = f"{framenum}_{oidx}"
            if DEBUG == "1":
                meta_str = ",".join([str(o) for o in face + [framenum_str]])
                print(f"[METADATA],{meta_str}", flush=True)

            METADATA[framenum_str] = {"frameId": framenum, "bbox": tdict}

    if DEBUG == "1":
        print(
            f"[TIMING],start_udf_metadata,{ipfilename}," + str(time.time()), flush=True
        )

    video_obj = cv2.VideoCapture(ipfilename)

    if not video_obj.isOpened():
        print(f"[!] Error opening video ({ipfilename})")

    fW, fH = (
        int(video_obj.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    if DEBUG == "1":
        otype = options["otype"]
        print(
            f"[TIMING],start_{otype}_metadata_extraction,{ipfilename},"
            + str(time.time()),
            flush=True,
        )
    if options["otype"] == "face":  # Face Detection
        while True:
            (grabbed, frame) = video_obj.read()
            if not grabbed:
                break  # No more frames are read
            frameNum = int(video_obj.get(cv2.CAP_PROP_POS_FRAMES))

            if (W, H) != (fW, fH):
                frame = cv2.resize(frame, (W, H), interpolation=CV2_INTERPOLATION)

            results = face_detection(frame, H, W)
            asyncio.run(update_face_metadata(results, frameNum))

    else:
        object_detection_model = YOLO(
            yolo_path,
            verbose=False,
            task="detect",
        )

        def extract_metadata(predictor):
            # all_objects = []
            # all_object_dicts = []
            all_frame_nums = []
            for bidx, result in enumerate(predictor.results):
                framenum = int(
                    predictor.batch[2][bidx].split("frame ")[-1].split("/")[0]
                )  # Access the frame number
                all_frame_nums.append(framenum)
                fH, fW = result.orig_shape
                boxes = result.boxes.cpu()
                # objects = []
                # dicts = []
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
                        # print(object_res)
                        # objects.append(object_res)

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

                        framenum_str = f"{framenum}_{oidx}"
                        if DEBUG == "1":
                            meta_str = ",".join(
                                [str(o) for o in object_res + [framenum_str]]
                            )
                            print(f"[METADATA],{meta_str}", flush=True)

                        METADATA[framenum_str] = {"frameId": framenum, "bbox": tdict}
                        oidx += 1
                #         dicts.append(tdict)
                # all_objects.append(objects)
                # all_object_dicts.append(dicts)

        object_detection_model.add_callback(
            "on_predict_postprocess_end", extract_metadata
        )
        results = object_detection_model.predict(
            ipfilename,
            imgsz=(H, W),
            batch=batch_size,
            conf=detection_threshold,
            iou=iou_threshold,
            half=half_flag,
            device=device_input,
            project=None,
            name=None,
            verbose=False,
            save=False,
            stream=True,
        )
        for result in results:
            pass

    video_obj.release()
    if DEBUG == "1":
        otype = options["otype"]
        print(
            f"[TIMING],end_{otype}_metadata_extraction,{ipfilename},"
            + str(time.time()),
            flush=True,
        )
    print(f"[UDF METADATA FILE (presort metadata)]: {METADATA}")
    metadata = dict(
        sorted(
            METADATA.items(), key=lambda item: int(item[0].split("_")[0]), reverse=False
        )
    )
    print(f"[UDF METADATA FILE (postsort metadata)]: {metadata}")

    response = {"opFile": ipfilename, "metadata": metadata}

    jsonfile = "jsonfile" + uuid.uuid1().hex + ".json"
    with open(jsonfile, "w") as f:
        json.dump(response, f, indent=4)
    print(f"[UDF METADATA FILE (response)]: {response}")

    if DEBUG == "1":
        num_detections = len(metadata.keys())
        print(f"[TIMING],end_udf_metadata,{ipfilename}," + str(time.time()), flush=True)

        print(
            f"[METADATA_INFO],{ipfilename},{otype},{num_detections},{W},{H}", flush=True
        )

    return ipfilename, jsonfile
