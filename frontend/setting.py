#!/usr/bin/python3

from tornado import web, gen
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

class SettingHandler(web.RequestHandler):
    def __init__(self, app, request, **kwargs):
        super(SettingHandler, self).__init__(app, request, **kwargs)
        self.executor= ThreadPoolExecutor(2)

    def check_origin(self, origin):
        return True

    @run_on_executor
    def _settings(self):
        return {
            "controls": [{
                "name": "person",
                "icon": "images/person.png",
                "description": "Find Person",
                "params": [{
                    "name": "Age Min",
                    "type": "number",
                    "value": 0,
                },{
                    "name": "Age Max",
                    "type": "number",
                    "value": 100,
                },{
                    "name": "Gender",
                    "type": "list",
                    "values": [
                       "skip",
                       "male",
                       "female",
                    ],
                    "value": "skip",
                },{
                    "name": "Emotion List",
                    "type": "list",
                    "values": [ 
                        "skip",
                        "neutral", 
                        "happy", 
                        "sad", 
                        "surprise", 
                        "anger" 
                    ],
                    "value": "skip",
                }],
            },{
                "name": "object",
                "icon": "images/object.png",
                "description": "Find Object",
                "params": [{
                    "name": "Object List",
                    "type": "list",
                    "values": [
                        "aeroplane", "apple", "backpack", "banana", "baseball bat", 
                        "baseball glove", "bear", "bed", "bench", "bicycle", "bird", 
                        "boat", "book", "bottle", "bowl", "broccoli", "bus", "cake", 
                        "car", "carrot", "cat", "cell phone", "chair", "clock", "cow", 
                        "cup", "diningtable", "dog", "donut", "elephant", "fire hydrant", 
                        "fork", "frisbee", "giraffe", "hair drier", "handbag", "horse", 
                        "hot dog", "keyboard", "kite", "knife", "laptop", "microwave", 
                        "motorbike", "mouse", "orange", "oven", "parking meter", "person", 
                        "pizza", "pottedplant", "refrigerator", "remote", "sandwich", 
                        "scissors", "sheep", "sink", "skateboard", "skis", "snowboard", 
                        "sofa", "spoon", "sports ball", "stop sign", "suitcase", 
                        "surfboard", "teddy bear", "tennis racket", "tie", "toaster", 
                        "toilet", "toothbrush", "traffic light", "train", "truck", 
                        "tvmonitor", "umbrella", "vase", "wine glass", "zebra"
                    ],
                    "value": "person",
                }],
            },{
                "name": "video",
                "icon": "images/video.png",
                "description": "Find Video",
                "params": [{
                    "name": "Video Name",
                    "type": "text",
                    "value": "*",
                }],
#            },{
#                "name": "advanced",
#                "icon": "images/advanced.png",
#                "description": "Advanced",
#                "params": [{
#                    "name": "Search Queries",
#                    "type": "text",
#                    "value": "",
#                }],
            }],
        }

    @gen.coroutine
    def get(self):
        settings=yield self._settings()
        self.write(settings)
        self.set_status(200,'OK')

