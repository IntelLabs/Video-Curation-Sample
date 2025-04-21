#!/usr/bin/python3

from urllib.parse import unquote,quote
from tornado import web,gen
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from merge_iv import merge_iv
from requests import get
import os
import json
import vdms
import time
import traceback

dbhost=os.environ["DBHOST"]
vdhost=os.environ["VDHOST"]

class SearchHandler(web.RequestHandler):
    def __init__(self, app, request, **kwargs):
        super(SearchHandler, self).__init__(app, request, **kwargs)
        self.executor= ThreadPoolExecutor(8)
        self._vdms=vdms.vdms()
        while True:
            try:
                self._vdms.connect(dbhost)
                break
            except Exception as e:
                print("Exception: "+str(e), flush=True)
            time.sleep(10)

    def check_origin(self, origin):
        return True

    def _value(self, query1, key):
        for kv in query1["params"]:
            if kv["name"]==key: return kv["value"]
        return None

    def _construct_single_query(self, query1, ref):
        q_bbox={
            "FindBoundingBox": {
                "_ref": ref,
                "results": {
                    "list": [ "frameID", "objectID", "video_name", "_coordinates" ],
                },
            },
        }
        q_conn={
            "FindConnection": {
                "class": "Frame2BB",
                "ref2": ref,
                "results": {
                    "list": [ "frameID", "video_name" ],
                },
            },
        }

        if query1["name"]=="video":
            name=self._value(query1, "Video Name")
            if name!="*" and name!="":
                q_conn["FindConnection"].update({
                    "constraints": {
                        "video_name": [ "==", name ],
                    },
                })
        if query1["name"]=="object":
            q_bbox["FindBoundingBox"].update({
                "constraints": {
                    "objectID": [ "==", self._value(query1, "Object List") ],
                },
            })
        if query1["name"]=="person":
            constraints={
                "age": [
                    ">=", self._value(query1, "Age Min"),
                    "<=", self._value(query1, "Age Max"),
                ],
                "objectID": [
                    "==", "face"
                ],
            }
            if "age" not in q_bbox["FindBoundingBox"]["results"]["list"]:
                q_bbox["FindBoundingBox"]["results"]["list"].append("age")

            emotion=self._value(query1, "Emotion List")
            if emotion!="skip":
                constraints["emotion"]=[ "==", emotion ]
                if "emotion" not in q_bbox["FindBoundingBox"]["results"]["list"]:
                    q_bbox["FindBoundingBox"]["results"]["list"].append("emotion")

            gender=self._value(query1, "Gender")
            if gender!="skip":
                constraints["gender"]=[ "==", gender ]
                if "gender" not in q_bbox["FindBoundingBox"]["results"]["list"]:
                    q_bbox["FindBoundingBox"]["results"]["list"].append("gender")

            q_bbox["FindBoundingBox"].update({ "constraints": constraints })

        return [q_bbox, q_conn]

    def _decode_response(self, response):
        clips={}
        for i in range(0,len(response)-1,2):
            if response[i+1]["FindConnection"]["status"]==0 and response[i]["FindBoundingBox"]["status"]==0:
                # connections=response[i+1]["FindConnection"]["connections"]
                bboxes=response[i]["FindBoundingBox"]["entities"]
                # if len(connections)!=len(bboxes): continue

                for j in range(0,len(bboxes)):
                    stream=bboxes[j]["video_name"]
                    if stream not in clips:
                        r=get(vdhost+"/api/info",params={"video":stream}).json()
                        clips[stream]={
                            "fps": r["fps"],
                            "duration": r["duration"],
                            "width": r["width"],
                            "height": r["height"],
                            "segs": [],
                            "frames": {},
                        }

                    # time stamp and duration
                    stream1=clips[stream]
                    ts=float(bboxes[j]["frameID"])/stream1["fps"]

                    # merge segs
                    segmin=1
                    seg1=[max(ts-segmin,0),min(ts+segmin,stream1["duration"])]
                    stream1["segs"]=merge_iv(stream1["segs"], seg1)

                    if ts not in stream1["frames"]:
                        stream1["frames"][ts]={
                            "time": ts,
                            "objects": []
                        }

                    if "objectID" in bboxes[j]:
                        bbc=bboxes[j]["_coordinates"]
                        stream1["frames"][ts]["objects"].append({
                            "detection" : {
                                "bounding_box" : {
                                    "x_max" : float(bbc["w"]+bbc["x"])/float(stream1["width"]),
                                    "x_min" : float(bbc["x"])/float(stream1["width"]),
                                    "y_max" : float(bbc["h"]+bbc["y"])/float(stream1["height"]),
                                    "y_min" : float(bbc["y"])/float(stream1["height"]),
                                },
                                "confidence" : 0.99,
                                "label" : bboxes[j]["objectID"],
                            },
                        })
        print("clips:", flush=True)
        print(clips, flush=True)

        # create segments
        segs=[]
        for name in clips:
            stream1=clips[name]
            for seg1 in stream1["segs"]:
                seg1c={
                    "name": name,
                    "stream": quote("/api/segment/"+str(seg1[0])+"/"+str(seg1[1])+"/"+name),
                    "thumbnail": quote("/api/thumbnail/"+str(seg1[0])+"/"+name+".png"),
                    "fps": stream1["fps"],
                    "time": seg1[0],
                    "duration": seg1[1]-seg1[0],
                    "offset": 0,
                    "width": stream1["width"],
                    "height": stream1["height"],
                    "frames": [],
                }
                for ts in stream1["frames"]:
                    if ts>=seg1[0] and ts<=seg1[1]:
                        stream1["frames"][ts].update({ "time": (ts-seg1[0])*1000 })
                        seg1c["frames"].append(stream1["frames"][ts])
                segs.append(seg1c)

        print("segs:", flush=True)
        print(segs, flush=True)
        return segs

    def find_common_elements(self,list_a, list_b):
        set_a = set(list_a)
        set_b = set(list_b)

        common_elements = set_a.intersection(set_b)
        if len(common_elements) > 0:
            return list(common_elements)
        else:
            return list()

    def get_details_from_BB(self, response):
        bb_entities = response[0]["FindBoundingBox"]["entities"]
        con_entity = response[1]["FindConnection"]["connections"][0]

        bb_info = {}
        for j in range(0,len(bb_entities)):
            video_name=bb_entities[j]["video_name"]
            frameID = bb_entities[j]["frameID"]
            objectID = bb_entities[j]["objectID"]
            con_entity["video_name"] = video_name
            con_entity["frameID"] = frameID

            if video_name not in bb_info:
                bb_info[video_name] = {frameID: [[bb_entities[j], con_entity]]}

            elif frameID not in bb_info[video_name]:
                bb_info[video_name][frameID] = [[bb_entities[j], con_entity]]

            else:
                bb_info[video_name][frameID].append([bb_entities[j], con_entity])
        return bb_info

    def intersect_reponses(self, responses):
        # Find reponse with least number of returned elements
        prev_info = self.get_details_from_BB(responses[0])
        responses_info = []
        for ridx in range(1, len(responses)):
            if prev_info == {}:
                break

            response_info = self.get_details_from_BB(responses[ridx])

            prev_videos = prev_info.keys()
            cur_videos = response_info.keys()
            valid_videos = self.find_common_elements(prev_videos, cur_videos)

            if len(valid_videos) == 0:
                response_info = {}
                break

            for vid in set(prev_videos).difference(set(valid_videos)):
                del prev_info[vid]

            for vid in set(cur_videos).difference(set(valid_videos)):
                del response_info[vid]

            for vid in valid_videos:
                prev_frames = prev_info[vid].keys()
                cur_frames = response_info[vid].keys()
                valid_frames = self.find_common_elements(prev_frames, cur_frames)

                if len(valid_frames) == 0:
                    del response_info[vid]
                    continue

                for frame in set(prev_frames).difference(set(valid_frames)):
                    del prev_info[vid][frame]

                for frame in set(cur_frames).difference(set(valid_frames)):
                    del response_info[vid][frame]

                for frame in valid_frames:
                    response_info[vid][frame].extend(prev_info[vid][frame])

            responses_info.append(response_info)
            prev_info = response_info.copy()

        print("responses_info: ", flush=True)
        print(prev_info, flush=True)

        new_connections=[]
        new_bboxes=[]
        for vid in prev_info.keys():
            for frame in prev_info[vid].keys():
                bb_info = prev_info[vid][frame]
                for j in range(0,len(bb_info)):
                    new_bboxes.append(bb_info[j][0])
                    new_connections.append(bb_info[j][1])

        final_response = responses[0]
        print("Number bboxes: ", len(new_bboxes))
        final_response[0]["FindBoundingBox"]["entities"] = new_bboxes
        final_response[0]["FindBoundingBox"]['returned'] = len(new_bboxes)
        final_response[1]["FindConnection"]["connections"] = new_connections
        final_response[1]["FindConnection"]['returned'] = len(new_connections)
        return final_response

    def one_shot_query(self, queries):
        vdms_response = []
        ref = 1
        print("Queries: ", flush=True)
        for query1 in queries: # Query per line in Gui
            responses = []
            for q in query1: #Queries on a single line (one icon)
                # print(f"Icon query: {q}", flush=True)

                # BB & Connection query for each icon
                vdms_query = self._construct_single_query(q, ref)

                print("vdms_query:", flush=True)
                print(vdms_query, flush=True)

                response, _ =self._vdms.query(vdms_query)

                responses.append(response)
                ref += 1

            if len(responses) > 1 and "FindBoundingBox" in responses[0][0]:
                # And operation; multiple on one line
                # Find common response/frames for all icons on line
                final_response = self.intersect_reponses(responses)
                vdms_response.extend(final_response)
            else:
                # Single query
                vdms_response.extend(responses[0])

        return vdms_response

    @run_on_executor
    def _search(self, queries, size):
        try:
            vdms_response = self.one_shot_query(queries)
        except Exception as e:
            vdms_response = []
            print("Exception: "+str(e)+"\n"+traceback.format_exc(), flush=True)
        # print("VDMS response:")
        # print(vdms_response, flush=True)
        segs = self._decode_response(vdms_response)
        return segs

    @gen.coroutine
    def get(self):
        queries=json.loads(unquote(str(self.get_argument("queries"))))
        size=int(self.get_argument("size"))
        # print("queries",flush=True)
        # print(queries,flush=True)
        r=yield self._search(queries, size)
        if isinstance(r, str):
            self.set_status(400, str(r))
            return

        self.write({"response":r})
        self.set_status(200, 'OK')
        self.finish()


    # def intersect_reponses2(self, responses):
    #     # Find reponse with least number of returned elements
    #     responses_info = []
    #     for response in responses:
    #         bboxes=response[0]["FindBoundingBox"]["entities"]

    #         response_info = {}
    #         for j in range(0,len(bboxes)):
    #             video_name=bboxes[j]["video_name"]
    #             frameID = bboxes[j]["frameID"]
    #             if video_name not in response_info:
    #                 response_info[video_name] = [frameID]
    #             else:
    #                 response_info[video_name].append(frameID)
    #         responses_info.append(response_info)

    #     print("responses_info: ", flush=True)
    #     print(responses_info, flush=True)

    #     # List of videos in all reponses
    #     valid_videos = responses_info[0].keys()
    #     for idx in range(1, len(responses_info)):
    #         valid_videos = self.find_common_elements(valid_videos, responses_info[idx].keys())

    #     if valid_videos is None:
    #         return []

    #     combined_response = {}
    #     for video in valid_videos:
    #         valid_frames = responses_info[0][video]
    #         for idx in range(1, len(responses_info)):
    #             valid_frames = self.find_common_elements(valid_frames, responses_info[idx][video])
    #         if valid_frames:
    #             combined_response[video] = valid_frames

    #     if len(combined_response.keys()) == 0:
    #         return []

    #     new_connections=[]
    #     new_bboxes=[]
    #     for ridx, response in enumerate(responses):
    #         bboxes=response[0]["FindBoundingBox"]["entities"]
    #         connections=response[1]["FindConnection"]["connections"]

    #         for j in range(0,len(bboxes)):
    #             video_name=bboxes[j]["video_name"]
    #             frameID = bboxes[j]["frameID"]
    #             if video_name in combined_response and frameID in combined_response[video_name]:
    #                 connections[j]["video_name"] = video_name
    #                 connections[j]["frameID"] = frameID
    #                 new_connections.append(connections[j])
    #                 new_bboxes.append(bboxes[j])

    #     final_response = responses[0]
    #     print("Number bboxes: ", len(new_bboxes))
    #     final_response[0]["FindBoundingBox"]["entities"] = new_bboxes
    #     final_response[0]["FindBoundingBox"]['returned'] = len(new_bboxes)
    #     final_response[1]["FindConnection"]["connections"] = new_connections
    #     final_response[1]["FindConnection"]['returned'] = len(new_connections)
    #     return final_response