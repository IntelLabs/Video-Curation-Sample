#!/usr/bin/python3

import datetime
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
from tornado import gen, web
from tornado.concurrent import run_on_executor


class WorkloadHandler(web.RequestHandler):
    def __init__(self, app, request, **kwargs):
        super(WorkloadHandler, self).__init__(app, request, **kwargs)
        self.executor = ThreadPoolExecutor(4)

    def check_origin(self, origin):
        return True

    @run_on_executor
    def _workload(self):
        return {
            "time": int(time.mktime(datetime.datetime.now().timetuple()) * 1000),
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory(),
            "disk": psutil.disk_usage("/var/www/mp4"),
        }

    @gen.coroutine
    def get(self):
        workload = yield self._workload()
        self.write(workload)
        self.set_status(200, "OK")
