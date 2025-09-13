import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import cv2
import yaml

# from inotify.adapters import Inotify
# from segment_archive import str2bool
from ultralytics.utils.checks import check_imgsz

import vdms

REPO_DIR = Path(__file__).parent.parent
model_w, model_h = (640, 640)


all_labels = [
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
]


category_mapping = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
}


def secs2HMS_str(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60

    time_str = ""
    if hour > 0:
        time_str += f"{hour} hrs"
    if min > 0:
        time_str += f" {min} mins"
    if sec > 0:
        time_str += f" {sec:04f} secs"
    return time_str


def str2bool(in_val):
    if isinstance(in_val, bool):
        return in_val

    if not isinstance(in_val, str):
        raise ValueError(f"{in_val} is not a bool or string")

    if in_val.title() == "True":
        return True
    else:
        return False


def read_config(file_path):
    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


def start_udf_server(port=5011):
    # python3 udf_server.py 5011 . &
    REMOTE_FN_DIR = str(REPO_DIR / "tests/remote_function")
    cmd_str = f"python3 {REMOTE_FN_DIR}/udf_server.py {port} {REMOTE_FN_DIR}"
    cmd = shlex.split(cmd_str)

    udf_process = subprocess.Popen(cmd, preexec_fn=os.setsid)
    # udf_process = subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS)  # Windows

    return udf_process


def kill_udf_server(udf_process):
    try:
        udf_process.kill()
    except Exception:
        try:
            import signal

            udf_process.send_signal(signal.SIGKILL)
        except Exception as e:
            print(f"Error killing UDF process: {e}")


def build_vdms(resize_input=False, device="CPU", in_source="videos"):
    db_image = "lcc_vdms:stream"
    dockerfile_path = REPO_DIR / "vdms/Dockerfile"
    context_path = REPO_DIR / "vdms"
    RESIZE_FLAG = str(resize_input)
    cmd_str = f"docker build --rm --build-arg DEVICE={device} --build-arg IN_SOURCE={in_source} --build-arg RESIZE_FLAG={RESIZE_FLAG} -f {dockerfile_path} -t {db_image} {context_path}"
    cmd = shlex.split(cmd_str)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()


def start_vdms(
    port: int = 55555,
    container_name: str = "ingest_test",
    kill: bool = True,
    debug: bool = False,
):
    # db_image = "intellabs/vdms:latest"
    db_image = "lcc_vdms:stream"

    # rm any containers with same name
    if kill:
        print("REMOVE VDMS DOCKER")
        shutdown_vdms(container_name)

    if kill:
        if not db_image.startswith("intellabs"):
            build_vdms()
        try:
            print("START VDMS DOCKER")
            cmd_str = f"docker run --rm -d --no-healthcheck -p {port}:55555 -v {REPO_DIR}:{REPO_DIR} --name {container_name} {db_image}"
            cmd = shlex.split(cmd_str)
            start_connection = time.time()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, error = process.communicate()
            if debug:
                print(output.decode("utf-8"))
            elapsed_time = time.time() - start_connection
            time.sleep(2)
            print(f"Time to confirm connection: {elapsed_time:0.3f} s\n")
            return f"Started VDMS Container: {container_name}"
        except Exception:
            return f"Error starting {container_name}"
            raise RuntimeError("Could not start container")
    else:
        # print(f"USING RUNNING CONTAINER {container_name}")
        return f"Using VDMS Container: {container_name}"


def shutdown_vdms(
    container_name: str = "ingest_test",
    debug: bool = False,
):
    cmd_str = f"docker kill {container_name}"
    cmd = shlex.split(cmd_str)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if debug:
        print(output.decode("utf-8"))
    time.sleep(1)


def get_video_details(filename_path, debug=False):
    if debug:
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
    if debug:
        print(f"[TIMING],end_get_video_details,{filename}," + str(time.time()))
    return video_info


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


def ingest_video(
    ingest_mode,
    filename_path,
    video_info,
    debug=False,
    resize_input=False,
    dbhost="localhost",
    dbport=55555,
    udfhost="video-service",
    udfport=5011,
):
    # global dbs

    # filename_path = "1191560.mp4"
    filename = str(Path(filename_path).name)
    # dn_name = filename.split("__")[0]
    # if dn_name not in dbs:
    #     dbs[dn_name] = vdms.vdms()
    #     dbs[dn_name].connect(dbhost, dbport)
    # elif not dbs[dn_name].is_connected():
    #     dbs[dn_name].connect(dbhost, dbport)
    db = vdms.vdms()
    db.connect(dbhost, dbport)

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
                    "url": f"http://{udfhost}:{udfport}/video",
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
    if debug:
        print(
            f"[TIMING],start_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
    # res, res_arr = dbs[dn_name].query([query], [video_blob])
    res, res_arr = db.query([query], [video_blob])
    if debug:
        print(
            f"[TIMING],end_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
        print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
        print(f"[DEBUG] INGEST_VIDEO RESPONSE: {res}", flush=True)
        # print(f"[DEBUG] Used client: {dn_name}", flush=True)


# TODO: Fix
def write_video_run_udf(
    file_queue, frameNum, _out_vid, clip_filename, metadata=None, ingest_mode="object"
):
    if _out_vid is not None:
        _out_vid.release()
        # print(f"Created video at frameNum {frameNum}", flush=True)

        # Add filename to processing queue
        # file_queue.put(clip_filename)
        if clip_filename not in file_queue and clip_filename not in ["", None]:
            file_queue.append(clip_filename)
        print(f"Added {clip_filename} to queue", flush=True)

        # Run UDF
        video_info = get_video_details(_out_vid)

        # ingest_video(_out_vid, "object", video_info, run_method="udf_metadata", metadata=metadata)
        ingest_video(
            ingest_mode,
            _out_vid,
            video_info,
            debug=False,
            resize_input=False,
            dbhost="localhost",
            dbport=55555,
            udfhost="video-service",
            udfport=5011,
        )

        _out_vid = None
        clip_filename = ""
    return _out_vid, clip_filename, file_queue


# TODO: Fix
def write_video_run_manual(
    file_queue, frameNum, _out_vid, clip_filename, metadata=None, ingest_mode="object"
):
    if _out_vid is not None:
        _out_vid.release()
        # print(f"Created video at frameNum {frameNum}", flush=True)

        # Add filename to processing queue
        # file_queue.put(clip_filename)
        if clip_filename not in file_queue and clip_filename not in ["", None]:
            file_queue.append(clip_filename)
        print(f"Added {clip_filename} to queue", flush=True)

        # Run UDF
        video_info = get_video_details(_out_vid)
        # ingest_video(_out_vid, "object", video_info, run_method="manual_metadata", metadata=metadata)
        ingest_video(
            ingest_mode,
            _out_vid,
            video_info,
            debug=False,
            resize_input=False,
            dbhost="localhost",
            dbport=55555,
            udfhost="video-service",
            udfport=5011,
        )

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
    outdir=str(REPO_DIR / "tests/results/ingest_test"),
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


# def calculate_distance(
#     row, focal_length, image_sensor_width_mm
# ):  # (actual_object_height, actual_object_width, focal_length_mm, image_sensor_width_mm, image_sensor_width_pixels, pixel_height, pixel_width):
#     label = row["label"]
#     image_sensor_width_pixels = row["Frame W"]
#     pixel_width = row["W"]
#     pixel_height = row["H"]

#     avg_object_dims = avg_object_dimensions_whd.get(
#         label, (1, 1, 1)
#     )  # Assuming 1m x 1m object
#     actual_object_width = avg_object_dims[0]
#     actual_object_height = avg_object_dims[1]

#     # Calculate focal length in pixels
#     focal_length_pixels = (
#         focal_length * image_sensor_width_pixels
#     ) / image_sensor_width_mm

#     # Calculate distance using height
#     distance_height = (actual_object_height * focal_length_pixels) / pixel_height

#     # Calculate distance using width
#     distance_width = (actual_object_width * focal_length_pixels) / pixel_width

#     # Calculate average distance
#     average_distance = (distance_height + distance_width) / 2

#     return average_distance


def _intersect(interval1, interval2):
    """
    Find whether two intervals intersect
    :param interval1: list [a, b], where 'a' is the left border and 'b' is the right border
    :param interval2: list [c, d], where 'c' is the left border and 'd' is the right border
    :return: True if intervals intersect, False otherwise
    """
    if interval1[0] < interval2[0]:
        left = interval1
        right = interval2
    elif interval2[0] < interval1[0]:
        left = interval2
        right = interval1
    else:  # so interval1[0] == interval2[0]
        return True

    if left[1] >= right[0]:
        return True
    else:
        return False


def _merge(interval1, interval2):
    """
    Finds merge of two intersecting intervals. This function should be called only if it's checked that
    intervals intersect, e.g. if "_intersect(interval1, interval2)" is True

    :param interval1: list [a, b], where 'a' is the left border and 'b' is the right border
    :param interval2: list [c, d], where 'c' is the left border and 'd' is the right border
    :return: new interval that contains only both intervals
    """
    return [min(interval1[0], interval2[0]), max(interval1[1], interval2[1])]


def merge_iv(intervals, interval_to_add):
    """
    Adds 'interval_to_add' into 'intervals' list:
    1) as a separate list if 'interval_to_add' intersects with no one in 'intervals' list
    2) as a product of merging with intervals that 'interval_to_add' intersect

    :param intervals: list of some existing intervals, e.g. [[a, b], [c, d], ...]
    :param interval_to_add: interval that must be added to 'intervals' list considering that it can intersect
    with some of intervals in 'intervals' list and then they must be merged into one bigger interval
    :return: list of intervals after adding 'interval_to_add' to 'intervals' with possible merge
    """
    new_intervals = []
    for segment in intervals:
        if _intersect(segment, interval_to_add):
            interval_to_add = _merge(segment, interval_to_add)
            continue
        new_intervals.append(segment)

    new_intervals.append(interval_to_add)
    return new_intervals
