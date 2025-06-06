#!/usr/bin/python3

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, unquote

from merge_iv import merge_iv
from tornado import gen, web
from tornado.concurrent import run_on_executor

import vdms

dbhost = os.environ["DBHOST"]
vdhost = os.environ["VDHOST"]
DEBUG = os.environ["DEBUG"]
print(f"DEBUG: {DEBUG}")


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

    def _construct_single_query(self, query1, ref):
        q_vid = {
            "FindVideo": {
                "_ref": ref,
                "constraints": {
                    "category": ["==", "video_path_rop"],
                },
                # "results": {"list": ['video_name', "video_filename"]},
                # "results": {"list": ["fps", "duration", "width", "height"]}
            }
        }
        q_vid2 = {
            "FindVideo": {
                "_ref": ref + 1,
                "frameconstraints": {},
                "results": {"list": ["server_filepath"]},
                "link": {"ref": ref},
            }
        }

        q_frame = {
            "FindVideo": {
                # "metaconstraints" : {
                #     # "objectID" : ["==", "car"],
                #     "objectID" : ["==", "face"],
                #     "emotion" : ["==", "happy"],
                # },
                "link": {"ref": ref + 1},
                # "operations": [
                #     {
                #         "type": "remoteOp",
                #         "url": "http://video-service:5011/video",
                #         "options": {
                #             "id": "framesofinterest",
                #         },
                #     }
                # ],
            }
        }

        if query1["name"] == "video":
            name = self._value(query1, "Video Name")
            if name != "*" and name != "":
                q_vid["FindVideo"].update(
                    {
                        "constraints": {
                            "Name": ["==", name],
                        },
                    }
                )

            elif name == "*":
                q_vid["FindVideo"].update(
                    {
                        "results": {
                            "list": [
                                "Name",
                                "fps",
                                "duration",
                                "width",
                                "height",
                                "frame_count",
                            ],
                        }
                    }
                )
                return [q_vid]

        metaconstraints = {}
        if query1["name"] == "object":
            metaconstraints["objectID"] = ["==", self._value(query1, "Object List")]
            q_frame["FindVideo"].update({"metaconstraints": metaconstraints})

        if query1["name"] == "person":
            metaconstraints["age"] = [
                ">=",
                int(self._value(query1, "Age Min")),
                "<=",
                int(self._value(query1, "Age Max")),
            ]
            metaconstraints["objectID"] = ["==", "face"]

            emotion = self._value(query1, "Emotion List")
            if emotion != "skip":
                metaconstraints["emotion"] = ["==", emotion]

            gender = self._value(query1, "Gender")
            if gender != "skip":
                metaconstraints["gender"] = ["==", gender]

            if len(metaconstraints) > 0:
                q_frame["FindVideo"].update({"metaconstraints": metaconstraints})

        return [q_vid, q_vid2, q_frame]

    def _decode_response(self, response):
        clips = {}
        if len(response) % 3 != 0:
            segs = []
            for i in range(0, len(response), 1):
                if (
                    "FindVideo" in response[i]
                    and response[i]["FindVideo"]["status"] == 0
                ):
                    entities = response[i]["FindVideo"]["entities"]
                    print(entities)

                    for ent in entities:
                        name = ent["Name"]
                        duration = ent["duration"]
                        seg1c = {
                            "name": name,
                            "stream": quote(
                                "/api/segment/0/" + str(duration) + "/" + name
                            ),
                            "thumbnail": quote("/api/thumbnail/0/" + name + ".png"),
                            "fps": ent["fps"],
                            "time": 0,
                            "duration": duration,
                            "offset": 0,
                            "width": ent["width"],
                            "height": ent["height"],
                            "frames": [x for x in range(0, ent["frame_count"])],
                        }
                        segs.append(seg1c)

        else:
            for i in range(0, len(response), 3):
                if (
                    "FindVideo" in response[i]
                    and response[i]["FindVideo"]["status"] == 0
                    and response[i + 2]["FindVideo"]["status"] == 0
                    and "entities" in response[i + 2]["FindVideo"]
                ):
                    entities = response[i + 2]["FindVideo"]["entities"]
                    # if response[i+1]["FindConnection"]["status"]==0 and response[i]["FindBoundingBox"]["status"]==0:
                    #     # connections=response[i+1]["FindConnection"]["connections"]
                    #     bboxes=response[i]["FindBoundingBox"]["entities"]
                    #     # if len(connections)!=len(bboxes): continue
                    print(entities)
                    for j in range(0, len(entities)):
                        for ent_bbox in entities[j]["bbox"]:
                            stream = ent_bbox["server_filepath"]
                            if stream not in clips:
                                clips[stream] = {
                                    "fps": ent_bbox["fps"],
                                    "duration": ent_bbox["duration"],
                                    "width": ent_bbox["width"],
                                    "height": ent_bbox["height"],
                                    "segs": [],
                                    "frames": {},
                                }

                            # time stamp and duration
                            stream1 = clips[stream]
                            ts = float(ent_bbox["frameID"]) / stream1["fps"]

                            # merge segs
                            segmin = 1  # 2
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
                                            "x_max": float(bbc["w"] + bbc["x"])
                                            / float(frameW),
                                            "x_min": float(bbc["x"]) / float(frameW),
                                            "y_max": float(bbc["h"] + bbc["y"])
                                            / float(frameH),
                                            "y_min": float(bbc["y"]) / float(frameH),
                                        },
                                        "label": ent_bbox["objectID"],
                                    },
                                }
                                if "confidence" in ent_bbox:
                                    obj["detection"]["confidence"] = ent_bbox[
                                        "confidence"
                                    ]
                                stream1["frames"][ts]["objects"].append(obj)

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

        print("segs:", flush=True)
        print(segs, flush=True)
        return segs

    def one_shot_query(self, queries: list):
        vdms_response = []
        ref = 1
        print("Queries: ", flush=True)
        for query1 in queries:  # Query per line in Gui
            responses = []
            for q in query1:  # Queries on a single line (one icon)
                # print(f"Icon query: {q}", flush=True)

                # BB & Connection query for each icon
                print("q: ", q)
                vdms_query = self._construct_single_query(q, ref)

                print("vdms_query:", flush=True)
                print(vdms_query, flush=True)

                response, _ = self._vdms.query(vdms_query)
                print("response: ", response)

                responses.append(response)
                ref += 1

            # Single query
            vdms_response.extend(responses[0])

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
        # print("VDMS response:")
        # print(vdms_response, flush=True)
        segs = self._decode_response(vdms_response)
        if DEBUG == "1":
            print("[TIMING],end_frontend_search,," + str(time.time()), flush=True)
        return segs

    @gen.coroutine
    def get(self):
        queries = json.loads(unquote(str(self.get_argument("queries"))))
        size = int(self.get_argument("size"))
        # print("queries",flush=True)
        # print(queries,flush=True)
        r = yield self._search(queries, size)
        if isinstance(r, str):
            self.set_status(400, str(r))
            return

        self.write({"response": r})
        self.set_status(200, "OK")
        self.finish()
