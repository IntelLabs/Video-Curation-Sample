# importing required libraries
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time  # time library

import yaml
from inotify.adapters import Inotify

num_workers = mp.cpu_count()  #  5,  mp.cpu_count()

# # REPO_DIR = Path(__file__).parent.parent
# # TEST_VIDEO_PATH = REPO_DIR / "video/archive_custom/video8K__test-8k-26s.mp4"
# SHARED_OUTPUT = os.getenv("SHARED_OUTPUT", "/var/www/mp4")
# # Path(SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
# tmp_dir = "/var/www/archive"
# kkhost = os.environ["KKHOST"]
# dbhost = "vdms-service"  # os.environ["DBHOST"]
# dbport = 55555
# ingestion = os.environ["INGESTION"]
in_source = os.environ["IN_SOURCE"]
# resize_input = str2bool(os.getenv("RESIZE_FLAG", False))
DEBUG = os.environ["DEBUG"]
DEBUG_FLAG = True if DEBUG == "1" else False
# # video_store_dir = "/home/resources"
# video_store_dir = "/var/www/mp4"
# model_w, model_h = (640, 640)
# DEVICE = os.environ["DEVICE"]


def run_processor(path_or_url, camera_name=None):
    cmd = [
        sys.executable,
        "/home/process_stream.py",
        path_or_url,
        # camera_name,
    ]
    if camera_name is not None:
        cmd.append(camera_name)

    # process = subprocess.Popen(
    #     cmd,
    # )
    # process.wait()
    subprocess.run(cmd, check=True)


def watch_video_files(queue, watch_dir):
    # Get files already in watch_dir
    for filename in os.listdir(watch_dir):
        if any(filename.endswith(ext) for ext in [".mp4", ".mkv", ".avi"]):
            source = os.path.join(watch_dir, filename)
            print(f"{source} added to queue: {time.time()}", flush=True)
            queue.put((source, None))

    # Watch watch_dir for new files
    i = Inotify()
    i.add_watch(watch_dir)
    print("START ADDING VIDEO FILES TO WATCHED DIRECTORY", flush=True)

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event

        # New file created in watched directory
        # if "IN_CREATE" in type_names:
        if "IN_CLOSE_WRITE" in type_names:
            source = os.path.join(path, filename)
            print(f"{source} added to queue: {time.time()}", flush=True)
            queue.put((source, None))


def retrieve_camera_details(queue, config_path):
    with open(config_path, "r") as inFile:
        config = yaml.safe_load(inFile)

    for camera_name, camera_details in config.items():
        # run_processor(camera_details["url"], camera_name=camera_name)
        source = camera_details["url"]
        print(f"{source} added to queue: {time.time()}", flush=True)
        queue.put((source, camera_name))


def main(watch_folder=os.getcwd()):
    if DEBUG_FLAG:
        print("[TIMING],start_watchandsend,," + str(time.time()), flush=True)

    print(f"in_source: {in_source}", flush=True)
    # if "videos" in in_source:
    #     # all_processes = []
    #     for filename in os.listdir(tmp_dir):
    #         if any(filename.endswith(ext) for ext in [".mp4", ".mkv", ".avi"]):
    #             full_filename_path = os.path.join(tmp_dir, filename)
    #             run_processor(full_filename_path, camera_name=None)

    # if "stream" in in_source:
    #     import yaml

    #     with open("/home/camera_config.yaml", "r") as inFile:
    #         config = yaml.safe_load(inFile)
    #     for camera_name, camera_details in config.items():
    #         run_processor(camera_details["url"], camera_name=camera_name)

    file_queue = mp.Queue()

    # Create a process that monitors new files in watch_folder (added to file_queue)
    watcher_process = mp.Process(
        target=watch_video_files, args=(file_queue, watch_folder)
    )

    # Create a process that retrieves camera info from config file (added to file_queue)
    watcher_process_camera = mp.Process(
        target=retrieve_camera_details, args=(file_queue, "/home/camera_config.yaml")
    )

    watcher_process.start()
    watcher_process_camera.start()

    # Pool of workers to process video clips
    with mp.Pool(processes=num_workers) as pool:
        while True:
            try:
                path_or_url, camera_name = file_queue.get(timeout=0.5)
                # pool.apply(run_processor, (path_or_url, camera_name,))
                pool.apply_async(
                    run_processor,
                    (
                        path_or_url,
                        camera_name,
                    ),
                )
            except queue.Empty:
                pass

    watcher_process.join()
    watcher_process_camera.join()

    if DEBUG_FLAG:
        print("[TIMING],end_watchandsend,," + str(time.time()), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise ValueError("Invalid input. Please provide watch directory.")
