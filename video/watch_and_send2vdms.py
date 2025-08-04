#!/usr/bin/env python3

import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from shutil import copyfile

import cv2
from inotify.adapters import Inotify
from segment_archive import str2bool
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
resize_input = str2bool(os.getenv("RESIZE_FLAG", False))
DEBUG = os.environ["DEBUG"]
# video_store_dir = "/home/resources"
video_store_dir = "/var/www/mp4"
model_w, model_h = (640, 640)
FPS = 15
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
            "from_file_path": filename_path,  # from_server_file
            "is_local_file": True,
            "properties": properties,
            "operations": [
                {
                    "type": "remoteOp",
                    "url": "http://video-service:5011/video",
                    "options": {
                        # "id": "metadata",
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
            "duration": float(duration),  # round(float(duration),4)
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
        if filename.endswith(".mp4"):  # and filename.startswith("video"):
            filename_path = os.path.join(video_store_dir, filename)
            if os.path.isfile(filename_path):
                file_size = os.path.getsize(filename_path)
                all_files.append((filename_path, file_size))
    return sorted(all_files, key=lambda item: item[1])


def write_video(file_queue, frameNum, _out_vid, clip_filename):
    if _out_vid is not None:
        _out_vid.release()
        # print(f"Created video at frameNum {frameNum}", flush=True)

        # Add filename to processing queue
        # file_queue.put(clip_filename)
        if clip_filename not in file_queue and clip_filename not in ["", None]:
            file_queue.append(clip_filename)
        print(f"Added {clip_filename} to queue", flush=True)

        _out_vid = None
        clip_filename = ""
    return _out_vid, clip_filename, file_queue


def video_clip_producer(
    file_queue,
    video_path,
    file_prefix,
    method=None,
    fps=None,
    clip_length_in_secs=10,
    outdir="/var/www/mp4",
):
    # ffmpeg Elapsed time: 253.96045470237732 secs
    # opencv Elapsed time: 215.44384384155273 secs
    print("Splitting video into clips ...")
    all_clips = []
    if method == "opencv":
        video_obj = cv2.VideoCapture(video_path)
        # , cv2.CAP_FFMPEG)

        if (fps is not None) and (float(video_obj.get(cv2.CAP_PROP_FPS)) != float(fps)):
            modified_video_path = "/tmp/" + Path(video_path).name

            # Change FPS of video using ffmpeg
            # GENERAL_OPTS = f"-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -threads 1 -filter:v fps={FPS} -flush_packets 0"
            GENERAL_OPTS = f"-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -filter:v fps={fps} -flush_packets 0"
            VIDEO_OPTS = "-f mpegts -movflags faststart -crf 28"
            cmd_str = (
                f"ffmpeg -y -i {video_path} {GENERAL_OPTS} {VIDEO_OPTS} {modified_video_path}"
                # f"ffmpeg -y -i {video_path} -filter:v fps={FPS} -movflags faststart {modified_video_path}"
            )
            cmd = shlex.split(cmd_str)

            try:
                subprocess.run(cmd, check=True)
                # ffmpeg_result = subprocess.run(cmd, capture_output=True, text=True)
                # if ffmpeg_result.returncode > 0:
                #     print("ffmpeg Error:", ffmpeg_result.stderr, flush=True)

                # Reload video
                video_obj.release()
                video_obj = cv2.VideoCapture(modified_video_path)
            except Exception as e:
                raise ValueError(f"Error occurred while processing video: {e}")

        # Check that object is opened successfully
        stream_available = False
        while not stream_available:
            if video_obj.isOpened():
                stream_available = True

        # Setup VideoWriter
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MJPG mp4v  No: h264 avc1 X264
        _out_vid = None

        clip_filename = ""
        fps = float(video_obj.get(cv2.CAP_PROP_FPS))  # 30
        clip_num = 0
        frame_count = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_total_frames = int(float(clip_length_in_secs * fps))
        while video_obj.isOpened():
            # Read frame
            grabbed, frame = video_obj.read()

            if grabbed:
                frameNum = int(video_obj.get(cv2.CAP_PROP_POS_FRAMES))
                # print(f"Current Frame:\t{frameNum}", flush=True)
                frameWH = (frame.shape[1], frame.shape[0])
                # frameHW = frame.shape[:2]

                # Start video clip
                if (frameNum - 1) % clip_total_frames < (clip_total_frames - 1):
                    if _out_vid is None:
                        # Initialize file
                        clip_filename = Path(outdir) / f"{file_prefix}_{clip_num}.mp4"
                        _out_vid = cv2.VideoWriter(
                            clip_filename,
                            fourcc=fourcc,
                            fps=fps,
                            frameSize=frameWH,
                        )
                        clip_num += 1
                    _out_vid.write(frame)

                # mod_val = (frameNum - 1) % clip_total_frames
                # print(f"frame: {frameNum} of {frame_count}\tmod: {mod_val}", flush=True)

                if ((frameNum - 1) % clip_total_frames == (clip_total_frames - 1)) or (
                    frameNum == frame_count
                ):
                    _out_vid.write(frame)
                    _out_vid, clip_filename, all_clips = write_video(
                        all_clips, frameNum, _out_vid, clip_filename
                    )

            else:
                _out_vid, clip_filename, all_clips = write_video(
                    all_clips, frameNum, _out_vid, clip_filename
                )

                break

        # _out_vid, clip_filename = write_video(file_queue, frameNum, _out_vid, clip_filename)

        # file_queue.put(None)  # Signal end of data
        video_obj.release()
        cv2.destroyAllWindows()

    elif method == "ffmpeg":
        time_segment_half = (
            clip_length_in_secs / 2
        )  # forces a keyframe at t=5,10,15 seconds.
        clip_filename = str(Path(outdir) / f"{file_prefix}_%d.mp4")
        clip_list_path = f"/tmp/{file_prefix}.ffconcat"

        GENERAL_OPTS = f"-flags -global_header -hide_banner -loglevel error -nostats -tune zerolatency -threads 1 -filter:v fps={fps} -flush_packets 0"
        VIDEO_OPTS = (
            "-f mpegts -movflags faststart -crf 28"  # -vcodec libx264   -s 640x360
        )
        SEGMENT_OPTS = f"-map 0  -segment_time {clip_length_in_secs} -force_key_frames expr:gte(t,n_forced*{time_segment_half})"
        SEGMENT_OPTS += f" -f segment -reset_timestamps 1 -segment_list {clip_list_path} -segment_format mp4 {clip_filename}"

        cmd_str = (
            f"ffmpeg -y -i {video_path} {GENERAL_OPTS} {VIDEO_OPTS} {SEGMENT_OPTS}"
        )
        cmd_list = shlex.split(cmd_str)

        subprocess.run(cmd_list, check=True)
        # ffmpeg_result = subprocess.run(cmd_list, capture_output=True, text=True, shell=False)
        # if ffmpeg_result.returncode > 0:
        #     print("ffmpeg Error:", ffmpeg_result.stderr, flush=True)
        # else:
        #     os.remove(filename_path)

        with open(clip_list_path, "r") as stream_list:
            file_keyword = "file "
            for line in stream_list:
                if line.strip().startswith(file_keyword):
                    clip_filename = line[len(file_keyword) :].strip()
                    if clip_filename.startswith("'") and clip_filename.endswith("'"):
                        clip_filename = clip_filename[1:-1]
                    clip_filename = str(Path(outdir) / clip_filename)
                    # process_clip(clip_filename)
                    # file_queue.put(clip_filename)
                    # print(f"Added {clip_filename} to queue", flush=True)
                    all_clips.append(clip_filename)
    else:
        raise ValueError(f"{method} is invalid method. Valid methods: opencv, ffmpeg")

    print(f"\tCreated {len(all_clips)} clips")
    return all_clips


def main(watch_folder=os.getcwd()):
    if DEBUG == "1":
        print("[TIMING],start_watchandsend,," + str(time.time()), flush=True)
    if "videos" in in_source:
        tmp_dir = "/var/www/archive"
        # sorted_files = sort_files_in_directory_by_size("/var/www/archive")
        for filename in os.listdir(tmp_dir):
            if filename.endswith(".mp4"):
                full_filename_path = os.path.join(tmp_dir, filename)
                file_prefix = Path(full_filename_path).stem
                all_clips = video_clip_producer(
                    None,
                    full_filename_path,
                    file_prefix,
                    method="ffmpeg",
                    fps=FPS,
                    clip_length_in_secs=10,
                    outdir=video_store_dir,
                )
                for filename_path in all_clips:
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
