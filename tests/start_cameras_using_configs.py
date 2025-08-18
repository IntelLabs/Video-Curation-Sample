# THIS FILE CREATES MOCK CAMERA_CONFIG.YAML FILES FOR TESTING

import argparse
import multiprocessing as mp
import shlex
import subprocess
import time
from pathlib import Path

import yaml

PROJECT_PATH = Path(__file__).parent.parent

TEST_VIDEOS = [
    PROJECT_PATH / "video/archive_custom/new_videos/Test-People-4k.mp4",
    PROJECT_PATH / "video/archive_custom/new_videos/Test-People-8k.mp4",
    PROJECT_PATH / "video/archive_custom/test-4k-24s.mp4",
    PROJECT_PATH / "video/archive_custom/test-8k-26s.mp4",
    PROJECT_PATH / "video/archive_custom/test2-4k-13s.mp4",
    PROJECT_PATH / "video/archive_custom/test2-8k-9s.mp4",
]


def read_config(file_path):
    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default=PROJECT_PATH / "tests/results/camera_configs/camera_config_1.yaml",
        type=Path,
        help="Path for camera config",
    )

    parser.add_argument(
        "-r",
        "--repeat",
        default=0,
        type=int,
        help="Number of times to repeat video stream [Default: 0]",
    )

    args = parser.parse_args()

    return args


def run_cmds(stream_details):
    camera_name, TEST_VIDEO, details, num_repeats = stream_details
    URL = details["url"]
    filename = Path(TEST_VIDEO).name

    this_camera_details = {}

    fps_dur_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate,duration -of default=noprint_wrappers=1:nokey=1 {TEST_VIDEO}"
    fps_dur_ = subprocess.run(
        shlex.split(fps_dur_cmd), capture_output=True, text=True, check=True
    ).stdout.split("\n")[:2]
    FPS = eval(fps_dur_[0])
    duration_ = float(fps_dur_[1])

    GENERAL_OPTS = "-threads 0 -flags +global_header -hide_banner -loglevel error -nostats -tune zerolatency -flush_packets 0"  #  -filter:v fps={self.target_fps}  -flush_packets 0 -threads 1
    # cmd = f"ffmpeg -re  -stream_loop 1 -i {TEST_VIDEO} {GENERAL_OPTS} -c:v libx264 -c:a aac -r {FPS} -f rtsp {URL}"
    # cmd = f"ffmpeg -re -stream_loop 1 -i {TEST_VIDEO} {GENERAL_OPTS} -filter:v fps={FPS} -preset fast -c:v libx264 -f rtsp -rtsp_transport tcp {URL}"
    # cmd = f"ffmpeg -re -stream_loop 1 -i {TEST_VIDEO} {GENERAL_OPTS} -c:v libx264 -preset ultrafast -filter:v fps=fps={FPS} -f rtsp -rtsp_transport tcp {URL}"
    # cmd = f"ffmpeg -re -stream_loop 1 -i {TEST_VIDEO} {GENERAL_OPTS} -filter:v fps={FPS} -f rtsp -rtsp_transport tcp {URL}"
    # cmd = f"ffmpeg -re -stream_loop 1 -i {TEST_VIDEO} {GENERAL_OPTS} -filter:v fps={FPS} -vsync cfr rtsp1_output.mp4"  #-f rtsp -rtsp_transport tcp {URL}"
    # cmd = f"ffmpeg -re -stream_loop 1 -i {TEST_VIDEO} {GENERAL_OPTS} -filter:v fps={FPS} -f rtsp -rtsp_transport tcp {URL}"
    # cmd = f"ffmpeg -re -i {TEST_VIDEO} -preset medium -tune zerolatency -filter:v fps=fps={FPS} -f rtsp -rtsp_transport tcp {URL}"

    cmd = f"ffmpeg -re -stream_loop {num_repeats} -i {TEST_VIDEO} {GENERAL_OPTS} -filter:v fps=fps={FPS} -f rtsp -rtsp_transport tcp {URL}"
    # cmd = f"ffmpeg -re -i {TEST_VIDEO} {GENERAL_OPTS} -filter:v fps=fps={FPS} -f rtsp -rtsp_transport tcp {URL}"
    cmd_list = shlex.split(cmd)

    # time_expired = False
    # start_cmd = time.time()
    while True:
        # if time_expired:
        #     break

        try:
            print(
                f"Sending {filename} ...\n\tFPS: {FPS}\n\tDuration: {duration_} s",
                flush=True,
            )
            # time_expired = (time.time() - start_cmd) >= (3 * 60)
            start_time = time.time()
            process = subprocess.Popen(
                cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            end_time = time.time()

            if process.returncode != 0:
                print(
                    f"Error sending {camera_name} ({filename}): {stderr.decode()}",
                    flush=True,
                )
            else:
                elapsed_time = end_time - start_time
                print(
                    f"Successfully sent {camera_name} ({filename}) in {elapsed_time} s",
                    flush=True,
                )
                this_camera_details.setdefault(camera_name, {})
                this_camera_details[camera_name]["video"] = TEST_VIDEO
                this_camera_details[camera_name]["FPS"] = FPS
                this_camera_details[camera_name]["duration"] = duration_
                this_camera_details[camera_name]["start_time"] = start_time
                this_camera_details[camera_name]["end_time"] = end_time
                this_camera_details[camera_name]["elapsed_time_s"] = elapsed_time
                break
        except Exception as e:
            print(
                f"Failed to start ffmpeg process for {camera_name} ({filename}): {e}\n",
                flush=True,
            )
    return this_camera_details


def merge_dicts(dict1, dict2):
    new_dict = dict1.copy()
    for k, v in dict2.items():
        if (k in new_dict) and (isinstance(new_dict[k], dict) and isinstance(v, dict)):
            new_dict[k] = merge_dicts(new_dict[k], v)
        else:
            new_dict[k] = v
    return new_dict


def main(args):
    camera_details = read_config(args.config)

    num_repeats = args.repeat

    num_workers = min(len(camera_details.keys()), mp.cpu_count())

    streams_to_process = []
    for camera_idx, (camera_name, details) in enumerate(camera_details.items()):
        video = str(TEST_VIDEOS[camera_idx % len(TEST_VIDEOS)])
        streams_to_process.append((camera_name, video, details, num_repeats))

        # streams_to_process.append((camera_idx, camera_name, details))

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_cmds, streams_to_process)

    for result in results:
        camera_details = merge_dicts(camera_details, result)

    # Write to YAML
    file_path = str(args.config).replace(".yaml", ".videos.yaml")
    with open(file_path, "w") as f:
        yaml.dump(camera_details, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    args = get_input_args()
    main(args)
    print("DONE")
