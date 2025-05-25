#!/usr/bin/env python3

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from shutil import copyfile

from inotify.adapters import Inotify
from ultralytics.utils.checks import check_imgsz

import vdms

# SETUP LOGGER
logger = logging.getLogger(__name__)
fmt_str = "[%(filename)s:line %(lineno)d] %(levelname)s:: %(message)s"
formatter = logging.Formatter(fmt_str)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.info("Video: Watch & Ingest")

topic = "video_curation_sched"
clientid = socket.gethostname()

kkhost = os.environ["KKHOST"]
dbhost = "vdms-service"  # os.environ["DBHOST"]
dbport = 55555
ingestion = os.environ["INGESTION"]
in_source = os.environ["IN_SOURCE"]
DEBUG = os.environ["DEBUG"]
# video_store_dir = "/home/resources"
video_store_dir = "/var/www/mp4"
resize_input = False
model_w, model_h = (640, 640)
dbs = {}


def ingest_video(ingest_mode, filename_path, video_info):
    global dbs

    # filename_path = "1191560.mp4"
    filename = str(Path(filename_path).name)
    dn_name = filename.split("__")[0]
    if dn_name not in dbs:
        dbs[dn_name] = vdms.vdms()
        dbs[dn_name].connect(dbhost, dbport)
    elif not dbs[dn_name].is_connected():
        dbs[dn_name].connect(dbhost, dbport)

    properties = {
        "Name": filename,  # .split("/")[-1],
        "category": "video_path_rop",
    }
    if len(video_info) > 0:
        properties.update(video_info)

    if resize_input or (
        (properties["height"] * properties["width"]) < (model_h * model_w)
    ):
        new_sizeHW = check_imgsz([model_h, model_w])  # expects hxw
    else:
        new_sizeHW = check_imgsz(
            [int(properties["height"]), int(properties["width"])]
        )  # expects hxw

    new_size = (new_sizeHW[1], new_sizeHW[0])

    query = {
        "AddVideo": {
            "from_server_file": filename_path,  # .replace("/home/remote_function/", ""), #from_server_file, from_file_path
            # "is_local_file": True,
            "properties": properties,
            "operations": [
                {
                    "type": "remoteOp",
                    "url": "http://video-service:5011/video",
                    "options": {
                        # "id": "metadata",
                        "id": "metadata_async",
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
    with open(filename_path, "rb") as fd:
        video_blob.append(fd.read())
    if DEBUG == "1":
        print(
            f"[TIMING],start_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
    res, res_arr = dbs[dn_name].query([query], [video_blob])
    if DEBUG == "1":
        print(
            f"[TIMING],end_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
        print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
        print(f"[DEBUG] INGEST_VIDEO RESPONSE: {res}", flush=True)
        print(f"[DEBUG] Used client: {dn_name}", flush=True)


def get_video_details(filename_path):
    if DEBUG == "1":
        filename = Path(filename_path).name
        print(
            f"[TIMING],start_get_video_details,{filename}," + str(time.time()),
            flush=True,
        )
    video_info = {}
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
            "-count_frames",
            "-show_format",
            "-show_streams",
            filename_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        result = json.loads(result.stdout)
        width = result["streams"][0]["width"]
        height = result["streams"][0]["height"]
        duration = result["streams"][0]["duration"]
        fps = eval(result["streams"][0]["r_frame_rate"])
        frame_count = eval(result["streams"][0]["nb_read_frames"])
        # duration = frame_count / fps
        if fps == 0 and (frame_count != 0 and duration != 0):
            fps = frame_count / duration

        video_info = {
            "fps": float(fps),
            "duration": float(duration),
            "width": int(width),
            "height": int(height),
            "frame_count": int(frame_count),
        }
    if DEBUG == "1":
        print(f"[TIMING],end_get_video_details,{filename}," + str(time.time()))
    return video_info


def sort_files_in_directory_by_size(in_dir):
    all_files = []
    for filename in os.listdir(in_dir):
        if filename.endswith(".mp4") and filename.startswith("video"):
            filename_path = os.path.join(video_store_dir, filename)
            if os.path.isfile(filename_path):
                file_size = os.path.getsize(filename_path)
                all_files.append((filename_path, file_size))
    return sorted(all_files, key=lambda item: item[1])


def main(watch_folder=os.getcwd()):
    if DEBUG == "1":
        print("[TIMING],start_watchandsend,," + str(time.time()), flush=True)
    if "videos" in in_source:
        sorted_files = sort_files_in_directory_by_size(video_store_dir)
        for filename_path, _ in sorted_files:
            video_info = get_video_details(filename_path)
            for ingest_mode in ingestion.split(","):
                ingest_video(ingest_mode, filename_path, video_info)

    if "stream" in in_source:
        i = Inotify()
        i.add_watch(watch_folder)

        for event in i.event_gen(yield_nones=False):
            (_, type_names, path, filename) = event
            filename_path = os.path.join(video_store_dir, filename)

            # on file write completion, we publish to topic
            if "IN_CLOSE_WRITE" in type_names and not os.path.exists(filename_path):
                copyfile(os.path.join(path, filename), filename_path)
                if DEBUG == "1":
                    print(
                        f"[DEBUG] COPIED:{path}/{filename}  TO {filename_path}"
                    )  # TODO: Remove
                video_info = get_video_details(filename_path)
                for ingest_mode in ingestion.split(","):
                    ingest_video(ingest_mode, filename_path, video_info)
    if DEBUG == "1":
        print("[TIMING],end_watchandsend,," + str(time.time()), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
