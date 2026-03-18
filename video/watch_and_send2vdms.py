import multiprocessing as mp
import os
import sys
import time
from multiprocessing.managers import BaseManager
from pathlib import Path

import requests
import yaml
from inotify.adapters import Inotify

num_workers = mp.cpu_count() // 2

DEBUG = os.environ["DEBUG"]
DEBUG_FLAG = True if DEBUG == "1" else False

BACKEND_URL = "http://fastapi-service:8000"

# Exit program if queue is empty for 5 minutes
empty_timeout = 5 * 60


# 1. Define the Manager
class QueueManager(BaseManager):
    pass


# ENABLE_STREAMLIT = os.environ["ENABLE_STREAMLIT"]
# if ENABLE_STREAMLIT:
#     from frontend.detection_runner import LiveDetectionRunner

#     n_cols = 2
#     pipeline = LiveDetectionRunner(n_cols=n_cols)
#     pipeline.setup_page()


# def run_processor(path_or_url, camera_name=None):
#     cmd = [
#         sys.executable,
#         "/home/process_stream.py",
#         path_or_url,
#         # camera_name,
#     ]
#     if camera_name is not None:
#         cmd.append(camera_name)

#     subprocess.run(cmd, check=True)


def start_stream_processor(source, camera_name):
    if camera_name is None:
        camera_name = Path(source).stem

    for _ in range(10):
        try:
            # FastAPI stream_processor
            payload = {"url": str(source), "name": camera_name}
            # Data is hidden in the body, no URL-encoding (%2F) mess
            res = requests.post(f"{BACKEND_URL}/stream", json=payload)
            # res = requests.post(
            #     # f"{BACKEND_URL}/stream",
            #     f"{BACKEND_URL}/stream?url={str(source)}&name={camera_name}",
            #     # json={"url": str(source), "name": camera_name},
            #     timeout=10
            # )
            if res.status_code == 200:
                print(f"Started {source} process.")
            return res
        # except requests.exceptions.ConnectionError:
        #     print(f"Connection reset, retrying... {res.json}")
        #     time.sleep(1) # Wait for buffers to clear
        except Exception:  # as e:
            # e = traceback.format_exc()
            # print(f"Error: {e}")
            time.sleep(1)


def watch_video_files(queue, watch_dir):
    # Get files already in watch_dir
    for filename in os.listdir(watch_dir):
        if any(filename.endswith(ext) for ext in [".mp4", ".mkv", ".avi"]):
            source = os.path.join(watch_dir, filename)
            print(f"{source} added to queue: {time.time()}", flush=True)
            queue.put((source, None))
            start_stream_processor(source, None)

    # Watch watch_dir for new files
    i = Inotify()
    i.add_watch(watch_dir)
    print("START ADDING VIDEO FILES TO WATCHED DIRECTORY", flush=True)

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event

        # New file created in watched directory
        # if "IN_CREATE" in type_names:
        if "IN_CLOSE_WRITE" in type_names and any(
            filename.endswith(ext) for ext in [".mp4", ".mkv", ".avi"]
        ):
            source = os.path.join(path, filename)
            print(f"{source} added to queue: {time.time()}", flush=True)
            queue.put((source, None))
            start_stream_processor(source, None)


def retrieve_camera_details(queue, config_path):
    config = None
    with open(config_path, "r") as inFile:
        config = yaml.safe_load(inFile)

    if config is not None:
        for camera_name, camera_details in config.items():
            # run_processor(camera_details["url"], camera_name=camera_name)
            source = camera_details["url"]
            print(f"{source} added to queue: {time.time()}", flush=True)
            queue.put((source, camera_name))
            start_stream_processor(source, camera_name)


def main(watch_folder=os.getcwd()):
    if DEBUG_FLAG:
        print("[TIMING],start_watchandsend,," + str(time.time()), flush=True)

    # 2. Setup the Shared Queue and Manager
    file_queue = mp.Queue()
    QueueManager.register("get_file_queue", callable=lambda: file_queue)

    # Start the manager on port 5000
    manager = QueueManager(address=("0.0.0.0", 5005), authkey=b"password123")
    manager.start()
    print("Queue Server started at 0.0.0.0:5005")

    # 3. Start worker processes
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

    # if ENABLE_STREAMLIT:
    #     while True:
    #         if not file_queue.empty():
    #             pipeline.setup_stream_section()
    #             break

    # 4. Keep the server alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        watcher_process.terminate()
        watcher_process_camera.terminate()
        manager.shutdown()

    # # Pool of workers to process video clips
    # # empty_queue_start = None
    # empty_queue_start = time.time()
    # i = 0  # stream_id
    # while time.time() - empty_queue_start < empty_timeout:
    #     if not file_queue.empty():
    #         path_or_url, camera_name = file_queue.get(timeout=0.5)
    #         empty_queue_start = None
    #         fastapi_processor = f"{BACKEND_URL}/video_feed/{camera_name}"
    #         if ENABLE_STREAMLIT:
    #             with pipeline.stream_cols[i % 2]:
    #                 pipeline.st.subheader(f"Stream {i}: {camera_name}")
    #                 pipeline.st.image(fastapi_processor + f"?path_or_url={path_or_url}")
    #         else:
    #             # pool.apply_async(
    #             #     run_processor,
    #             #     (
    #             #         path_or_url,
    #             #         camera_name,
    #             #     ),
    #             # )
    #             _ = requests.get(
    #                 fastapi_processor,
    #                 data=[("path_or_url", path_or_url)],
    #                 stream=True,
    #             )

    #         empty_queue_start = time.time()  # Reset timer if item is processed
    #     else:
    #         time.sleep(0.1)  # Sleep briefly to prevent high CPU usage

    # watcher_process.join()
    # watcher_process_camera.join()

    if DEBUG_FLAG:
        print("[TIMING],end_watchandsend,," + str(time.time()), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise ValueError("Invalid input. Please provide watch directory.")
