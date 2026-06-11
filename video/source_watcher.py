import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from multiprocessing.managers import BaseManager
from pathlib import Path

import requests
import yaml
from inotify.adapters import Inotify

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://fastapi-service:8000")
DEBUG = os.environ["DEBUG"]
DEBUG_FLAG = True if DEBUG == "1" else False
ACCEPTED_VIDEO_FORMATS = [".mp4", ".mkv", ".avi"]
num_workers = mp.cpu_count() // 2
empty_timeout = 5 * 60


# --- HELPER FUNCTIONS ---
def is_file_ready(filepath, retries=5, delay=1):
    """
    Checks if a file is fully written and accessible.
    Uses exponential backoff for retries.
    """
    for _ in range(retries):
        try:
            # Try to open the file in append mode to check for locks
            with open(filepath, "r"):
                return True
        except (IOError, OSError):
            logger.warning(f"File {filepath} is locked. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    return False


def connect_to_app():
    # is_ready = False
    # while not is_ready:
    with requests.Session() as session:
        while True:
            try:
                res = session.get(f"{BACKEND_URL}/status")

                if res.status_code == 200:
                    # Expected format: {"cam_1": {"status": "Ready"}, ...}
                    data = res.json()

                    is_ready = any(status == "Ready" for status in data.values())
                    if is_ready:
                        break
            except requests.exceptions.RequestException:
                logger.info("BACKEND: Waiting for fastapi-service ...")

            time.sleep(1)


# --- WATCHERS & MANAGERS ---
class QueueManager(BaseManager):
    pass


def worker_process(queue):
    """Consumes tasks from the queue and sends them to the backend."""
    while True:
        try:
            # Wait for a task; timeout prevents hanging on shutdown
            task = queue.get(timeout=10)
            if task is None:
                break  # Sentinel value to stop worker

            source, camera_name = task
            start_stream_processor(source, camera_name)
        except mp.queues.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker error: {e}")


def watch_video_files(queue, watch_dir):
    # Initial scan of files
    with os.scandir(watch_dir) as entries:
        for entry in entries:
            # entry.is_file() and entry.name are retrieved in one go
            if entry.is_file() and any(
                entry.name.endswith(ext) for ext in ACCEPTED_VIDEO_FORMATS
            ):
                source = entry.path  # Full path is already available

                if is_file_ready(source):
                    queue.put((source, None))
                    logger.info(f"{source} added to queue: {time.time()}")
                    # start_stream_processor(source, None)

    # Watch watch_dir for new files
    i = Inotify()
    i.add_watch(watch_dir)
    logger.info("START ADDING VIDEO FILES TO WATCHED DIRECTORY")

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event

        # New file created in watched directory
        # IN_CLOSE_WRITE is better than IN_CREATE because it triggers
        # only after the writing process finishes and closes the file.
        if "IN_CLOSE_WRITE" in type_names and any(
            filename.endswith(ext) for ext in ACCEPTED_VIDEO_FORMATS
        ):
            source = os.path.join(path, filename)

            if is_file_ready(source):
                queue.put((source, None))
                logger.info(f"{source} added to queue: {time.time()}")
                # start_stream_processor(source, None)
            else:
                logger.error(f"Failed to access {source} after retries. Skipping.")


def retrieve_camera_details(queue, config_path):
    """
    Watches config file and adds only new streams (via camera_name) to queue
    """
    unique_camera_names = set()

    def read_config_and_queue():
        try:
            with open(config_path, "r") as inFile:
                config = yaml.safe_load(inFile)

            if config:
                for camera_name, camera_details in config.items():
                    # run_processor(camera_details["url"], camera_name=camera_name)
                    if isinstance(camera_details, dict) and "url" in camera_details:
                        source = camera_details["url"]
                        if source not in unique_camera_names:
                            logger.info(f"{source} added to queue: {time.time()}")
                            queue.put((source, camera_name))
                            unique_camera_names.add(source)
                            # start_stream_processor(source, camera_name)
                    else:
                        logger.warning(f"Invalid entry: {camera_name}")
        except Exception:
            e = traceback.format_exc()
            logger.info(f"Unexpected error processing config file: {e}")

    # Initial scan of config file
    read_config_and_queue()

    # Watch for changes to file
    i = Inotify()
    config_dir = os.path.dirname(os.path.abspath(config_path))
    i.add_watch(config_dir)

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event

        if filename == os.path.basename(config_path) and "IN_CLOSE_WRITE" in type_names:
            time.sleep(0.5)
            read_config_and_queue()


def unified_watcher(queue, watch_dir, config_path):
    """
    Watches directory for both videos and config changes.
    """
    unique_camera_names = set()
    config_filename = os.path.basename(config_path)

    def read_config():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if config:
                for name, details in config.items():
                    url = details.get("url")
                    if url and url not in unique_camera_names:
                        logger.info(f"CAMERA: {name} added to queue: {time.time()}")
                        queue.put((url, name))
                        unique_camera_names.add(url)
        except Exception as e:
            logger.error(f"CONFIG ERROR: {e}")

    # Initial scan of config file
    read_config()

    # Initial scan of video files
    with os.scandir(watch_dir) as entries:
        for entry in entries:
            if entry.is_file() and any(
                entry.name.endswith(ext) for ext in ACCEPTED_VIDEO_FORMATS
            ):
                if is_file_ready(entry.path):
                    queue.put((entry.path, None))

    # Unified Event Loop
    i = Inotify()
    i.add_watch(watch_dir)
    logger.info(f"UNIFIED WATCHER: Monitoring {watch_dir} for videos and config...")

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event
        full_path = os.path.join(path, filename)

        # Handle Config Update (Supports Atomic Saves/Moves)
        if filename == config_filename:
            if any(ev in type_names for ev in ["IN_CLOSE_WRITE", "IN_MOVED_TO"]):
                time.sleep(0.2)
                read_config()

        # Handle New Video Files
        elif any(filename.endswith(ext) for ext in ACCEPTED_VIDEO_FORMATS):
            if "IN_CLOSE_WRITE" in type_names:
                if is_file_ready(full_path):
                    queue.put((full_path, None))
                    logger.info(f"VIDEO: {full_path} added to queue: {time.time()}")


# --- STREAMER ---
def start_stream_processor(source, camera_name):
    if camera_name is None:
        camera_name = Path(source).stem

    payload = {"url": str(source), "name": camera_name}

    for attempt in range(1, 11):
        try:
            # FastAPI stream_processor
            res = requests.post(f"{BACKEND_URL}/stream", json=payload)

            # Raise an exception for 4xx or 5xx status codes
            res.raise_for_status()

            if res.status_code == 200:
                logger.info(f"Started {source} process on attempt {attempt}.")
                return res

        except requests.exceptions.HTTPError as e:
            # Handle specific API errors (e.g., 400 Bad Request, 500 Internal Server Error)
            logger.info(f"HTTP Error: {e.response.status_code} - {e.response.text}")

            # If it's a client error (4xx), retrying likely won't help
            if 400 <= e.response.status_code < 500:
                break

        except requests.exceptions.RequestException as e:
            # Handle connection issues, timeouts, and DNS errors
            logger.info(f"Connection attempt {attempt} failed: {e}")

        except Exception:
            e = traceback.format_exc()
            logger.info(f"Unexpected error processing {source}: {e}")

        time.sleep(1)


# --- MAIN FUNCTION ---
def main(watch_folder=os.getcwd()):
    # Wait until App is ready
    connect_to_app()

    if DEBUG_FLAG:
        logger.info("[TIMING],start_watchandsend,," + str(time.time()))

    # Setup the Shared Queue and Manager
    file_queue = mp.Queue()
    QueueManager.register("get_file_queue", callable=lambda: file_queue)

    # Start the manager on port 5005
    manager = QueueManager(address=("0.0.0.0", 5005), authkey=b"password123")
    manager.start()
    logger.info("Queue Server started at 0.0.0.0:5005")

    # Define worker processes
    processes = [
        mp.Process(  # Process that retrieves camera info from config file (added to file_queue)
            target=unified_watcher,
            args=(file_queue, watch_folder, "/home/camera_config.yaml"),
            name="UnifiedWatcher",
            daemon=True,
        ),
    ]

    for _ in range(num_workers):
        p = mp.Process(target=worker_process, args=(file_queue,), daemon=True)
        processes.append(p)

    # 4. Start processed and Keep the server alive
    try:
        # Start all processes
        for p in processes:
            p.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping services ...")
    finally:
        # Cleanup
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()  # Ensure fully closed

        manager.shutdown()
        if DEBUG_FLAG:
            logger.info("[TIMING],end_watchandsend,," + str(time.time()))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise ValueError("Invalid input. Please provide watch directory.")
