#!/usr/bin/python3

from concurrent.futures import ThreadPoolExecutor

from tornado import gen, web
from tornado.concurrent import run_on_executor


class SettingHandler(web.RequestHandler):
    def __init__(self, app, request, **kwargs):
        super(SettingHandler, self).__init__(app, request, **kwargs)
        self.executor = ThreadPoolExecutor(2)

    def check_origin(self, origin):
        return True

    @run_on_executor
    def _settings(self):
        return {
            "controls": [
                {
                    "name": "person",
                    "icon": "images/person.png",
                    "description": "Find Person",
                    "params": [
                        {
                            "name": "Age Min",
                            "type": "number",
                            "value": 18,
                        },
                        {
                            "name": "Age Max",
                            "type": "number",
                            "value": 75,
                        },
                        {
                            "name": "Gender",
                            "type": "list",
                            "values": [
                                "skip",
                                "male",
                                "female",
                            ],
                            "value": "skip",
                        },
                        {
                            "name": "Emotion List",
                            "type": "list",
                            "values": [
                                "skip",
                                "neutral",
                                "happy",
                                "sad",
                                "surprise",
                                "anger",
                            ],
                            "value": "skip",
                        },
                    ],
                },
                {
                    "name": "object",
                    "icon": "images/object.png",
                    "description": "Find Object",
                    "params": [
                        {
                            "name": "Object List",
                            "type": "list",
                            "values": [
                                "airplane",
                                "apple",
                                "backpack",
                                "banana",
                                "baseball bat",
                                "baseball glove",
                                "bear",
                                "bed",
                                "bench",
                                "bicycle",
                                "bird",
                                "boat",
                                "book",
                                "bottle",
                                "bowl",
                                "broccoli",
                                "bus",
                                "cake",
                                "car",
                                "carrot",
                                "cat",
                                "cell phone",
                                "chair",
                                "clock",
                                "couch",
                                "cow",
                                "cup",
                                "dining table",
                                "dog",
                                "donut",
                                "elephant",
                                "fire hydrant",
                                "fork",
                                "frisbee",
                                "giraffe",
                                "hair drier",
                                "handbag",
                                "horse",
                                "hot dog",
                                "keyboard",
                                "kite",
                                "knife",
                                "laptop",
                                "microwave",
                                "motorcycle",
                                "mouse",
                                "orange",
                                "oven",
                                "parking meter",
                                "person",
                                "pizza",
                                "potted plant",
                                "refrigerator",
                                "remote",
                                "sandwich",
                                "scissors",
                                "sheep",
                                "sink",
                                "skateboard",
                                "skis",
                                "snowboard",
                                "spoon",
                                "sports ball",
                                "stop sign",
                                "suitcase",
                                "surfboard",
                                "teddy bear",
                                "tennis racket",
                                "tie",
                                "toaster",
                                "toilet",
                                "toothbrush",
                                "traffic light",
                                "train",
                                "truck",
                                "tv",
                                "umbrella",
                                "vase",
                                "wine glass",
                                "zebra",
                            ],
                            "value": "person",
                        },
                        {
                            "name": "Object Count",
                            "type": "list",
                            "values": ["skip"] + [str(x) for x in range(1, 26)],
                            "value": "skip",
                        },
                        {
                            "name": "Object Count Condition",
                            "type": "list",
                            "values": ["==", "<=", "<", ">=", ">"],
                            "value": "==",
                        },
                        {
                            "name": "Frame ID",
                            "type": "text",
                            "value": "skip",
                        },
                        {
                            "name": "Frame Condition",
                            "type": "list",
                            "values": [
                                "==",
                                "<=",
                                ">=",
                            ],
                            "value": "==",
                        },
                    ],
                },
                {
                    "name": "video",
                    "icon": "images/video.png",
                    "description": "Find Video",
                    "params": [
                        {
                            "name": "Video Name",
                            "type": "text",
                            "value": "*",
                        }
                    ],
                },
                # {
                #     "name": "advanced",
                #     "icon": "images/advanced.png",
                #     "description": "Advanced",
                #     "params": [
                #         {
                #             "name": "Search Queries",
                #             "type": "text",
                #             "value": "",
                #         }
                #     ],
                # },
            ],
        }

    @gen.coroutine
    def get(self):
        settings = yield self._settings()
        self.write(settings)
        self.set_status(200, "OK")
