#!/usr/bin/python3

import json
import os
from concurrent.futures import ThreadPoolExecutor

from tornado import gen, httpclient, web

BACKEND_URL = os.getenv("BACKEND_URL", "http://fastapi-service:8000")
FASTAPI_URL = f"{BACKEND_URL}/model_classes"


class SettingHandler(web.RequestHandler):
    def __init__(self, app, request, **kwargs):
        super(SettingHandler, self).__init__(app, request, **kwargs)
        self.executor = ThreadPoolExecutor(2)

    def check_origin(self, origin):
        return True

    @gen.coroutine
    def get_model_classes(self):
        """Asynchronously fetches classes from the FastAPI container."""
        client = httpclient.AsyncHTTPClient()
        default_classes = ["class0"]

        try:
            response = yield client.fetch(FASTAPI_URL)
            data = json.loads(response.body.decode("utf-8"))
            return data.get("classes", default_classes)  # Default fallback if empty
            # with open(MODEL_CLASSES_FILE, "r") as f:
            #     data = json.load(f)
            #     return data.get("classes", default_classes)
        except Exception as e:
            # Fallback to a basic list if the model container is unreachable
            print(f"Error fetching model classes: {e}")
            return default_classes

    # @run_on_executor
    @gen.coroutine
    def _settings(self):
        include_advanced = False
        dynamic_objects = yield self.get_model_classes()

        controls = []
        if "person" in dynamic_objects:
            person_control = {
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
            }
            controls.append(person_control)

        object_control = {
            "name": "object",
            "icon": "images/object.png",
            "description": "Find Object",
            "params": [
                {
                    "name": "Object List",
                    "type": "list",
                    "values": sorted(dynamic_objects),
                    "value": dynamic_objects[0],
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
        }
        controls.append(object_control)

        video_control = {
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
        }
        controls.append(video_control)

        if include_advanced:
            advanced_control = {
                "name": "advanced",
                "icon": "images/advanced.png",
                "description": "Advanced",
                "params": [
                    {
                        "name": "Search Queries",
                        "type": "text",
                        "value": "",
                    }
                ],
            }
            controls.append(advanced_control)

        return {
            "controls": controls,
        }

    @gen.coroutine
    def get(self):
        settings = yield self._settings()
        self.write(settings)
        self.set_status(200, "OK")
