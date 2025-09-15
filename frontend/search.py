#!/usr/bin/python3

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from random import randint
from urllib.parse import quote, unquote

from merge_iv import merge_iv
from requests import get
from tornado import gen, web
from tornado.concurrent import run_on_executor

import vdms

dbhost = os.environ["DBHOST"]
vdhost = os.environ["VDHOST"]
DEBUG = os.environ["DEBUG"]

LOCKTIMEOUT_RETRIES = 5
ERR_KEYWORDS = [
    "timeout",
    "null search iterator",
    "outoftransactions",
    "no entities found",
]


def retry_query(db, query, num_retries=LOCKTIMEOUT_RETRIES, sleep_timer: int = 0):
    for ridx in range(num_retries + 1):
        response, _ = db.query(query, [[]])
        if (
            "FailedCommand" in response[0]
            and any(k in response[0]["info"].lower() for k in ERR_KEYWORDS)
        ) or (
            ("FindVideo" in response[0])
            and ("info" in response[0]["FindVideo"])
            and any(k in response[0]["FindVideo"]["info"].lower() for k in ERR_KEYWORDS)
        ):
            if "FailedCommand" in response[0]:
                err = response[0]["info"]
            else:
                err = response[0]["FindVideo"]["info"]
            if DEBUG == "1":
                print(
                    f"[DEBUG search Attempt #{ridx}] Received '{err}' for {query} ",
                    flush=True,
                )
            if sleep_timer > 0:
                time.sleep(sleep_timer)
        else:
            if DEBUG == "1":
                print(
                    f"[DEBUG search] Successful query response: {response}",
                    flush=True,
                )
            break  # Continue
    return response


class SearchHandler(web.RequestHandler):
    def __init__(self, app, request, **kwargs):
        super(SearchHandler, self).__init__(app, request, **kwargs)
        self.executor = ThreadPoolExecutor(8)
        self._vdms = vdms.vdms()
        while True:
            try:
                self._vdms.connect(dbhost)
                break
            except Exception as e:
                print("Exception: " + str(e), flush=True)
            time.sleep(10)

    def check_origin(self, origin):
        return True

    def _value(self, query1, key):
        for kv in query1["params"]:
            if kv["name"] == key:
                return kv["value"]
        return None

    def _construct_query(self, line_queries, ref):
        q_vid = {
            "FindVideo": {
                "_ref": ref,
                "constraints": {
                    "category": ["==", "video_path_rop"],
                },
                "results": {"list": ["Name"], "blob": False},
            }
        }
        q_vid2 = {
            "FindEntity": {
                "_ref": ref + 1,
                "class": "Frame",
                "results": {"list": ["server_filepath", "frameID"]},
                "link": {"ref": ref},
            }
        }
        q_frame = {
            "FindBoundingBox": {
                "link": {"ref": ref + 1},
                "results": {
                    "list": [
                        "objectID",
                        "server_filepath",
                        "frameID",
                        "VD:x1",
                        "VD:y1",
                        "VD:width",
                        "VD:height",
                        "frameW",
                        "frameH",
                        "confidence",
                    ]
                },
            }
        }

        for icon_query in line_queries:
            if icon_query["name"] == "video":
                name = self._value(icon_query, "Video Name")
                if name != "*" and name != "":
                    q_frame["FindBoundingBox"].update(
                        {
                            "constraints": {
                                "server_filepath": ["==", name],
                            },
                        }
                    )
                    q_vid2["FindEntity"].update(
                        {
                            "constraints": {
                                "server_filepath": ["==", name],
                            },
                        }
                    )
                else:
                    return [q_vid, q_vid2]

            if icon_query["name"] == "object":
                obj_name = self._value(icon_query, "Object List")
                frame = self._value(icon_query, "Frame ID")
                frame_cond = self._value(icon_query, "Frame Condition")

                q_frame["FindBoundingBox"].setdefault("constraints", {})
                if "objectID" in q_frame["FindBoundingBox"]["constraints"]:
                    if (
                        obj_name
                        not in q_frame["FindBoundingBox"]["constraints"]["objectID"]
                    ):
                        q_frame["FindBoundingBox"]["constraints"]["objectID"].extend(
                            ["==", obj_name]
                        )
                else:
                    q_frame["FindBoundingBox"]["constraints"]["objectID"] = [
                        "==",
                        obj_name,
                    ]

                if frame != "skip":
                    q_frame["FindBoundingBox"]["constraints"]["frameID"] = [
                        frame_cond,
                        int(frame.strip()),
                    ]
                    q_vid2["FindEntity"].setdefault("constraints", {})
                    q_vid2["FindEntity"]["constraints"]["frameID"] = [
                        frame_cond,
                        int(frame.strip()),
                    ]

            if icon_query["name"] == "person":
                q_frame["FindBoundingBox"].setdefault("constraints", {})
                q_frame["FindBoundingBox"]["constraints"]["age"] = [
                    ">=",
                    int(self._value(icon_query, "Age Min")),
                    "<=",
                    int(self._value(icon_query, "Age Max")),
                ]

                emotion = self._value(icon_query, "Emotion List")
                if emotion != "skip":
                    q_frame["FindBoundingBox"]["constraints"]["emotion"] = [
                        "==",
                        emotion,
                    ]

                gender = self._value(icon_query, "Gender")
                if gender != "skip":
                    q_frame["FindBoundingBox"]["constraints"]["gender"] = ["==", gender]

                if "objectID" in q_frame["FindBoundingBox"]["constraints"]:
                    if (
                        "face"
                        not in q_frame["FindBoundingBox"]["constraints"]["objectID"]
                    ):
                        q_frame["FindBoundingBox"]["constraints"]["objectID"].extend(
                            ["==", "face"]
                        )
                else:
                    q_frame["FindBoundingBox"]["constraints"]["objectID"] = [
                        "==",
                        "face",
                    ]

        del q_vid2["FindEntity"]["link"]
        return [q_vid2, q_frame]

    def _decode_response(self, response):
        clips = {}
        segs = []
        for i in range(0, len(response), 2):
            if (
                "FindVideo" in response[i]
                and response[i]["FindVideo"]["status"] == 0
                and response[i + 1]["FindEntity"]["status"] == 0
            ):
                entities = response[i + 1]["FindEntity"]["entities"]
                # print(entities)

                uniq_name = []
                for ent in entities:
                    name = ent["server_filepath"]
                    if name not in uniq_name:
                        r = get(vdhost + "/api/info", params={"video": name}).json()
                        duration = r["duration"]  # ent["duration"]
                        seg1c = {
                            "name": name,
                            "stream": quote(
                                "/api/segment/0/" + str(duration) + "/" + name
                            ),
                            "thumbnail": quote("/api/thumbnail/0/" + name + ".png"),
                            "fps": r["fps"],
                            "time": 0,
                            "duration": duration,
                            "offset": 0,
                            "width": r["width"],
                            "height": r["height"],
                            "frames": [x for x in range(0, r["frame_count"])],
                        }
                        segs.append(seg1c)
                        uniq_name.append(name)

            elif (
                "FindBoundingBox" in response[i + 1]
                and response[i + 1]["FindBoundingBox"]["status"] == 0
                and "entities" in response[i + 1]["FindBoundingBox"]
            ):
                entities = response[i + 1]["FindBoundingBox"]["entities"]
                print(f"\t{len(entities)} bbs returned", flush=True)

                for ent_bbox in entities:
                    stream = ent_bbox["server_filepath"]
                    if stream not in clips:
                        r = get(vdhost + "/api/info", params={"video": stream}).json()
                        clips[stream] = {
                            "fps": r["fps"],
                            "duration": r["duration"],
                            "width": r["width"],
                            "height": r["height"],
                            "segs": [],
                            "frames": {},
                        }

                    # time stamp and duration
                    stream1 = clips[stream]
                    ts = float(ent_bbox["frameID"]) / stream1["fps"]

                    # merge segs
                    segmin = 1  # 1, 2
                    seg1 = [
                        max(ts - segmin, 0),
                        min(ts + segmin, stream1["duration"]),
                    ]
                    stream1["segs"] = merge_iv(stream1["segs"], seg1)

                    if ts not in stream1["frames"]:
                        stream1["frames"][ts] = {"time": ts, "objects": []}

                    if "objectID" in ent_bbox:
                        bbc = {
                            "x": ent_bbox["VD:x1"],
                            "y": ent_bbox["VD:y1"],
                            "w": ent_bbox["VD:width"],
                            "h": ent_bbox["VD:height"],
                        }

                        # Normalize BBs to frame size
                        frameW = (
                            ent_bbox["frameW"]
                            if not isinstance(ent_bbox["frameW"], str)
                            else ent_bbox["width"]
                        )
                        frameH = (
                            ent_bbox["frameH"]
                            if not isinstance(ent_bbox["frameH"], str)
                            else ent_bbox["height"]
                        )

                        obj = {
                            "detection": {
                                "bounding_box": {
                                    "x_max": float(bbc["w"] + bbc["x"]) / float(frameW),
                                    "x_min": float(bbc["x"]) / float(frameW),
                                    "y_max": float(bbc["h"] + bbc["y"]) / float(frameH),
                                    "y_min": float(bbc["y"]) / float(frameH),
                                },
                                "label": ent_bbox["objectID"],
                            },
                        }
                        if "confidence" in ent_bbox:
                            obj["detection"]["confidence"] = ent_bbox["confidence"]
                        stream1["frames"][ts]["objects"].append(obj)

                if DEBUG == "1":
                    print("clips:", flush=True)
                    print(clips, flush=True)

                # create segments
                segs = []
                for name in clips:
                    stream1 = clips[name]
                    for seg1 in stream1["segs"]:
                        seg1c = {  # var "data" used in playback.js
                            "name": name,
                            "stream": quote(
                                "/api/segment/"
                                + str(seg1[0])
                                + "/"
                                + str(seg1[1])
                                + "/"
                                + name
                            ),
                            "thumbnail": quote(
                                "/api/thumbnail/" + str(seg1[0]) + "/" + name + ".png"
                            ),
                            "fps": stream1["fps"],
                            "time": seg1[0],
                            "duration": seg1[1] - seg1[0],
                            "offset": 0,
                            "width": stream1["width"],
                            "height": stream1["height"],
                            "frames": [],
                        }
                        for ts in stream1["frames"]:
                            if ts >= seg1[0] and ts <= seg1[1]:
                                stream1["frames"][ts].update(
                                    {"time": (ts - seg1[0]) * 1000}
                                )
                                seg1c["frames"].append(stream1["frames"][ts])
                        segs.append(seg1c)

        if DEBUG == "1":
            print("segs:", flush=True)
            print(segs, flush=True)
        return segs

    def one_shot_query(self, queries: list):
        vdms_response = []
        ref = 1
        if DEBUG == "1":
            print("Queries: ", flush=True)
        for line_query in queries:  # Query per line in Gui
            vdms_query = self._construct_query(line_query, ref)
            if DEBUG == "1":
                print("vdms_query:", flush=True)
                print(vdms_query, flush=True)

            responses = retry_query(self._vdms, vdms_query, sleep_timer=randint(1, 5))
            if DEBUG == "1":
                print("response: ", responses, flush=True)

            ref += 1
            vdms_response.extend(responses)

        return vdms_response

    @run_on_executor
    def _search(self, queries, size):
        if DEBUG == "1":
            print("[TIMING],start_frontend_search,," + str(time.time()), flush=True)
        try:
            vdms_response = self.one_shot_query(queries)
        except Exception as e:
            vdms_response = []
            print("Exception: " + str(e) + "\n" + traceback.format_exc(), flush=True)

        if DEBUG == "1":
            print("VDMS response:", flush=True)
            print(vdms_response, flush=True)

        segs = self._decode_response(vdms_response)
        if DEBUG == "1":
            print("[TIMING],end_frontend_search,," + str(time.time()), flush=True)
        return segs

    @gen.coroutine
    def get(self):
        queries = json.loads(unquote(str(self.get_argument("queries"))))
        size = int(self.get_argument("size"))

        if DEBUG == "1":
            print("queries:", flush=True)
            print(queries, flush=True)

        r = yield self._search(queries, size)
        if isinstance(r, str):
            self.set_status(400, str(r))
            return

        self.write({"response": r})
        self.set_status(200, "OK")
        self.finish()
