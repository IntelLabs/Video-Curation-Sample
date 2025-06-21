import os
import time
from pathlib import Path

import cv2
from openvino.runtime import Core
from ultralytics import YOLO

base_resource_dir = os.environ.get("base_resource_dir", "/home/resources")
DEBUG = os.environ["DEBUG"]
CV2_INTERPOLATION = cv2.INTER_AREA
DEVICE = os.environ.get("DEVICE", "CPU")
DEVICE_OV = "AUTO"
METADATA_BATCH_SIZE = int(os.environ.get("METADATA_BATCH_SIZE", 100))
detection_threshold = 0.7
iou_threshold = 0.5
model_precision_object = "FP16"
model_precision_face = "FP16"
model_name = "yolo11"
half_flag = True
dynamic_flag = True
if DEVICE == "GPU":
    batch_size = 1
    run_platform = "engine"
else:
    batch_size = 8
    run_platform = "openvino"

device_input = DEVICE.lower() if DEVICE == "CPU" else 0
batch_data = []
yolo_path = Path(
    f"{base_resource_dir}/models/ultralytics/{model_name}/{model_precision_object}"
)


""" MODEL DEFINITIONS """
ie = Core()
face_detection_model_xml = f"{base_resource_dir}/models/intel/face-detection-adas-0001/{model_precision_face}/face-detection-adas-0001.xml"
face_detection_model = ie.read_model(
    model=face_detection_model_xml,
    weights=face_detection_model_xml.replace(".xml", ".bin"),
)
# face_det_w, face_det_h = 672, 384
_, face_det_c, face_det_h, face_det_w = face_detection_model.inputs[0].shape
face_det_compiled_model = ie.compile_model(face_detection_model, DEVICE_OV)

age_gender_classification_model_xml = f"{base_resource_dir}/models/intel/age-gender-recognition-retail-0013/{model_precision_face}/age-gender-recognition-retail-0013.xml"
age_gender_classification_model = ie.read_model(
    model=age_gender_classification_model_xml,
    weights=age_gender_classification_model_xml.replace(".xml", ".bin"),
)
# ag_w, ag_h = 62, 62
_, ag_c, ag_h, ag_w = age_gender_classification_model.inputs[0].shape
ag_compiled_model = ie.compile_model(age_gender_classification_model, DEVICE_OV)

emotions_classification_model_xml = f"{base_resource_dir}/models/intel/emotions-recognition-retail-0003/{model_precision_face}/emotions-recognition-retail-0003.xml"
emotions_classification_model = ie.read_model(
    model=emotions_classification_model_xml,
    weights=emotions_classification_model_xml.replace(".xml", ".bin"),
)
# em_w, em_h = 64, 64
_, em_c, em_h, em_w = emotions_classification_model.inputs[0].shape
em_compiled_model = ie.compile_model(emotions_classification_model, DEVICE_OV)


def get_udf_query(db, filename_path, properties, ingest_mode, new_size):
    query = {
        "AddVideo": {
            "from_file_path": filename_path,  # from_server_file
            "is_local_file": True,
            "properties": properties,
            "operations": [
                {
                    "type": "remoteOp",
                    "url": "http://video-service:5011/video",
                    "options": {
                        # "id": "metadata",
                        # "id": "metadata_async",
                        "id": "metadata_callback",
                        "otype": ingest_mode,
                        "media_type": "video",
                        "fps": properties["fps"],
                        "input_sizeWH": new_size,
                    },
                }
            ],
        }
    }

    video_blob = []
    # with open(filename_path, "rb") as fd:
    #     video_blob.append(fd.read())
    # return query, video_blob

    filename = properties["Name"]
    dn_name = filename.split("__")[0]
    if DEBUG == "1":
        print(
            f"[TIMING],start_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
    res, res_arr = db.query([query], [video_blob])
    if DEBUG == "1":
        print(
            f"[TIMING],end_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
        print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
        print(f"[DEBUG] INGEST_VIDEO RESPONSE: {res}", flush=True)
        print(f"[DEBUG] Used client: {dn_name}", flush=True)


def get_model(model_dir, run_platform, device_input, batch=1):
    final_model_path = f"{model_dir}/{model_name}n.pt"
    pt_detection_model = YOLO(final_model_path, verbose=False, task="detect")
    if run_platform == "openvino":
        pt_detection_model.export(
            format="openvino",
            half=half_flag,
            dynamic=dynamic_flag,
            device=device_input,
            batch=batch,
        )

        final_model_path = f"{model_dir}/{model_name}n_openvino_model/"
        object_detection_model = YOLO(
            final_model_path,
            verbose=False,
            task="detect",
        )

        # det_ov_model = core.read_model(final_model_path+"yolo11n.xml")
        # ov_config = {hints.performance_mode: hints.PerformanceMode.LATENCY}
        # if DEVICE == "GPU":
        #     ov_config["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "YES"
        # compiled_model = core.compile_model(det_ov_model, DEVICE, ov_config)
        # object_detection_model.predictor.model.ov_compiled_model = compiled_model

    elif run_platform == "engine":
        pt_detection_model.export(
            format="engine",
            half=half_flag,
            dynamic=dynamic_flag,
            simplify=True,
            batch=batch,
        )
        # pt_detection_model.export(format='engine')  # Rohit

        final_model_path = f"{model_dir}/{model_name}n.engine/"
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

        final_model_path = f"{model_dir}/{model_name}n.onnx"
        pt_detection_model.export(
            format="onnx",
            half=half_flag,
            dynamic=dynamic_flag,
            device=device_input,
            simplify=True,
            batch=batch,
        )

        object_detection_model = YOLO(final_model_path, verbose=False, task="detect")

    elif run_platform == "pytorch":
        object_detection_model = pt_detection_model
        if DEVICE == "GPU":
            object_detection_model.to("cuda")
        else:
            object_detection_model.to(device_input)

    else:
        raise ValueError(f"[!] Model for {run_platform} is not implemented.")

    return object_detection_model, final_model_path


def insert_bb_data(
    db, filename, data, properties, frame_exist=[], ingest_mode="object"
):
    global batch_data
    frame_props = {
        k: v for k, v in properties.items() if k not in ["category", "frame_count"]
    }
    # frame_props["server_filepath"] = properties["Name"]
    bb_props = {
        k: v for k, v in properties.items() if k not in ["category", "frame_count"]
    }
    # bb_props["server_filepath"] = properties["Name"]
    query = [
        {
            "FindVideo": {
                "_ref": 1,
                "constraints": {"server_filepath": ["==", filename]},
                "results": {"count": "", "list": ["Name", "server_filepath"]},
            }
        }
    ]

    ref = 1
    for b in data:
        framenum, bbidx = b["frameId"].split("_")
        frame_props["frameID"] = int(framenum)
        bb_props["frameID"] = int(framenum)
        bb_props["objectID"] = b["bbox"]["object"]
        bb_props["confidence"] = b["bbox"]["object_det"]["confidence"]
        bb_props["frameH"] = b["bbox"]["object_det"]["frameH"]
        bb_props["frameW"] = b["bbox"]["object_det"]["frameW"]
        if ingest_mode == "face":
            bb_props["age"] = b["bbox"]["object_det"]["age"]
            bb_props["gender"] = b["bbox"]["object_det"]["gender"]
            bb_props["emotion"] = b["bbox"]["object_det"]["emotion"]
        bb_rect = {
            "x": b["bbox"]["x"],
            "y": b["bbox"]["y"],
            "w": b["bbox"]["width"],
            "h": b["bbox"]["height"],
        }

        if int(framenum) in frame_exist:
            ent_command = {
                "FindEntity": {
                    "class": "Frame",
                    "link": {"ref": 1},
                    "_ref": ref + 1,
                    "constraints": {
                        "server_filepath": ["==", filename],
                        "frameID": ["==", frame_props["frameID"]],
                    },
                }
            }
        else:
            ent_command = {
                "AddEntity": {
                    "class": "Frame",
                    "link": {"ref": 1},
                    "_ref": ref + 1,
                    "properties": frame_props,
                }
            }

        query.extend(
            [
                ent_command,
                {
                    "AddBoundingBox": {
                        "link": {"ref": ref + 1},
                        "rectangle": bb_rect,
                        "properties": bb_props,
                    }
                },
            ]
        )
        ref += 1
    res, _ = db.query(query)
    # print(res)


def get_manual_query(db, filename_path, properties, ingest_mode, new_size):
    """
    Run object detection
    Insert data into VDMS using callback
    """
    global batch_data
    W, H = new_size
    filename = properties["server_filepath"]
    dn_name = filename.split("__")[0]
    # async def update_face_metadata(results, framenum):
    #     for oidx, face in enumerate(results):
    #         tdict = {
    #             "x": int(face[0]),
    #             "y": int(face[1]),
    #             "height": int(face[2]),
    #             "width": int(face[3]),
    #             "object": "face",
    #             "object_det": {
    #                 "age": int(face[4]),
    #                 "gender": str(face[5]),
    #                 "emotion": str(face[6]),
    #                 "confidence": float(face[7]),
    #                 "frameH": int(H),
    #                 "frameW": int(W),
    #             },
    #         }
    #         framenum_str = f"{framenum}_{oidx}"
    #         if DEBUG == "1":
    #             meta_str = ",".join([str(o) for o in face + [framenum_str]])
    #             print(f"[METADATA],{meta_str}", flush=True)

    #         METADATA[framenum_str] = {"frameId": framenum, "bbox": tdict}

    def face_detection(frame, framenum, H, W, frame_exist):
        bs = 1
        # Model expects BGRA
        # face detect -> age-gender -> emotions
        global batch_data
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
        # faces = []
        oidx = 0
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
                face_res = [
                    x1,
                    y1,
                    height,
                    width,
                    age,
                    gender,
                    emotion,
                    confidence,
                    H,
                    W,
                ]
                # print(face_res)
                # faces.append(face_res)

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
                framenum_str = f"{framenum}_{oidx}"
                if DEBUG == "1":
                    meta_str = ",".join([str(o) for o in face_res + [framenum_str]])
                    print(f"[METADATA],{meta_str}", flush=True)

                batch_data.append({"frameId": framenum_str, "bbox": tdict})
                if len(batch_data) == METADATA_BATCH_SIZE:
                    # RUN INSERT QUERY
                    insert_bb_data(
                        db,
                        filename,
                        batch_data,
                        properties,
                        frame_exist=frame_exist,
                        ingest_mode=ingest_mode,
                    )

                    batch_data = []
                oidx += 1

        # return faces

    if DEBUG == "1":
        print(
            f"[TIMING],start_manual_ingest_{ingest_mode},{filename},"
            + str(time.time()),
            flush=True,
        )
    # dn_name = filename.split("__")[0]
    # Check if video node exists; If not, add
    vid_query = [
        {
            "FindVideo": {
                "_ref": 1,
                "constraints": {"server_filepath": ["==", filename]},
                "results": {"count": ""},
            }
        },
        {
            "FindEntity": {
                "class": "Frame",
                "link": {"ref": 1},
                "constraints": {
                    "server_filepath": ["==", filename],
                },
                "results": {"count": "", "list": ["frameID"]},
            }
        },
    ]
    vid_res, _ = db.query(vid_query)

    # Only add if it doesn't exist
    frame_exist = []
    if (
        "FailedCommand" in vid_res[0]
        or vid_res[0]["FindVideo"]["status"] != 0
        or vid_res[0]["FindVideo"]["returned"] == 0
    ):
        add_query = {
            "AddVideo": {
                "codec": "h264",
                "container": "mp4",
                "properties": properties,
                # "from_file_path": filename_path,  # from_server_file
                # "is_local_file": True,
            }
        }

        video_blob = []
        with open(filename_path, "rb") as fd:
            video_blob.append(fd.read())

        response, _ = db.query([add_query], [video_blob])
        # print(response[0])
    else:
        for ent in vid_res[1]["FindEntity"]["entities"]:
            frame_exist.append(int(ent["frameID"]))

    if DEBUG == "1":
        print(
            f"[TIMING],start_{ingest_mode}_metadata_extraction,{filename},"
            + str(time.time()),
            flush=True,
        )

    video_obj = cv2.VideoCapture(filename_path)

    if not video_obj.isOpened():
        print(f"[!] Error opening video ({filename})")

    fW, fH = (
        int(video_obj.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    if ingest_mode == "face":
        while True:
            (grabbed, frame) = video_obj.read()
            if not grabbed:
                break  # No more frames are read
            frameNum = int(video_obj.get(cv2.CAP_PROP_POS_FRAMES))

            if (W, H) != (fW, fH):
                frame = cv2.resize(frame, (W, H), interpolation=CV2_INTERPOLATION)

            # results = face_detection(frame, H, W)
            # asyncio.run(update_face_metadata(results, frameNum))
            face_detection(frame, frameNum, H, W, frame_exist)

        if batch_data != []:
            # RUN QUERY
            insert_bb_data(
                db,
                filename,
                batch_data,
                properties,
                frame_exist=frame_exist,
                ingest_mode=ingest_mode,
            )
            batch_data = []
    else:
        object_detection_model, _ = get_model(
            yolo_path, run_platform, device_input, batch=batch_size
        )

        def extract_metadata(predictor):
            global batch_data
            # all_objects = []
            # all_object_dicts = []
            all_frame_nums = []
            # batch_data = []
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

                        batch_data.append({"frameId": framenum_str, "bbox": tdict})
                        if len(batch_data) == METADATA_BATCH_SIZE:
                            # RUN INSERT QUERY
                            insert_bb_data(
                                db,
                                filename,
                                batch_data,
                                properties,
                                frame_exist=frame_exist,
                                ingest_mode=ingest_mode,
                            )

                            batch_data = []
                        oidx += 1

        def flush_metadata(predictor):
            global batch_data
            if batch_data != []:
                # RUN QUERY
                insert_bb_data(
                    db,
                    filename,
                    batch_data,
                    properties,
                    frame_exist=frame_exist,
                    ingest_mode=ingest_mode,
                )
                batch_data = []

        object_detection_model.add_callback(
            "on_predict_postprocess_end", extract_metadata
        )
        object_detection_model.add_callback("on_predict_end", flush_metadata)
        results = object_detection_model.predict(
            filename_path,
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
        print(
            f"[TIMING],end_{ingest_mode}_metadata_extraction,{filename},"
            + str(time.time()),
            flush=True,
        )

        print(
            f"[TIMING],end_manual_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
        print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
        print(f"[DEBUG] Used client: {dn_name}", flush=True)
