# ==============================================================================
# IMPORTS
import asyncio
import gc
import inspect
import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.cuda._memory_viz as memory_viz
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from fastapi import FastAPI

sys.path.insert(1, str(Path(__file__).parent.parent))
from include.default_configs import ENABLE_QUERYING_DEFAULT
from include.detectors import GeneralObjectDetector, SmartFilteringObjectDetector
from include.utils import (
    AsyncDisplayVideoWriter,
    AsyncVideoWriter,
    DummyProcess,
    PipelineConfig,
    VDMSPool,
    # safe_unregister_shm,
    analyze_tracemalloc_snapshot,
    default_attr_keys,
    get_bb_overlay,
    get_metadata_overlay,
    # find_contours_gpu_equivalent,
    global_frame_prefetch_worker_v1,
    metadata2vdms_with_retry,
    release_native_linux_heap,
    release_shared_memory,
)

# Force OpenCV to run sequentially to prevent context-switching overhead
cv2.setNumThreads(0)  # Forces OpenCV loops to run strictly sequentially

# ==============================================================================
# LOGGING
logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Suppress low-delay reference block warnings from OpenCV/PyAV/FFmpeg
# os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
# os.environ["OPENCV_LOG_LEVEL"] = "OFF"
logging.getLogger("libav").setLevel(logging.CRITICAL)
logging.getLogger("libav.hevc").setLevel(logging.CRITICAL)
main_app_logger = logging.getLogger(__name__)

# ==============================================================================
# FUNCTIONS

STREAM_ARG = False
# PADDING_PX = 5  #25
PADDING_SCALE = 0.5  # 0.2  #3  #0.05  # 0.045  # 0.05
BASE_PIPELINE_CONFIG = PipelineConfig(
    SHARED_MODEL=os.getenv("SHARED_MODEL", False),
    ENABLE_QUERYING=os.getenv("ENABLE_QUERYING", ENABLE_QUERYING_DEFAULT),
)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp;hwaccel;cuda;threads;auto;low_delay;1;probesize;5000000"
)

ffmpeg_cores = "4,5,6,7"  # Manually define cores

# ----- GLOBAL VARIABLES -----
all_metadata = {}

# Queue for metadata being sent to vdms
send_metadata_queue = queue.Queue()

# Tracks if both components are finished:
#   {"clip_name": {"video": bool, "meta": bool}}
clip_completion_tracker = {}


def log_to_logger(message, level="info"):
    """Safely logs a message to the main application logger."""
    try:
        if level.lower() == "debug":
            main_app_logger.debug(message)
        elif level.lower() == "warning":
            main_app_logger.warning(message)
        else:
            main_app_logger.info(message)
    except Exception:
        pass


from multiprocessing.shared_memory import SharedMemory


def video_writer_core_loop(
    write_queue,
    segment_pattern,
    fps,
    width,
    height,
    clip_duration,
    shm_names,
    frame_bytes,
):
    """
    Original core loop, now running in an isolated process.
    It manages its own FFmpeg pipe and reads frames from the multiprocessing queue.
    """
    # 1. Move your existing FFmpeg command creation here
    # (Adapt these arguments if your original segment command differed slightly)
    # command = [
    #     'ffmpeg', '-y',
    #     '-f', 'rawvideo', '-vcodec', 'rawvideo',
    #     '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
    #     '-i', '-',
    #     '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
    #     '-f', 'segment', '-segment_time', '60', '-reset_timestamps', '1',
    #     segment_pattern
    # ]
    # clip_duration = int(self.config.CLIP_DURATION)
    # Attach to the existing shared memory blocks
    shm_blocks = [SharedMemory(name=name) for name in shm_names]

    frame_buffers = [
        np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
        for shm in shm_blocks
    ]
    # Example: Get the CPU core count and reserve the last few for FFmpeg
    # core_count = os.cpu_count()
    # ffmpeg_cores = "4,5,6,7"  # Manually define cores
    command = [
        "taskset",
        "-c",
        ffmpeg_cores,
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",  # "mpeg4",  # Or "libx264" if you prefer H.264
        "-crf",
        "23",
        "-f",
        "mpegts",
        "-movflags",
        "faststart",
        "-force_key_frames",
        f"expr:gte(t,n_forced*{clip_duration})",
        "-f",
        "segment",
        "-segment_time",
        f"{clip_duration}",
        "-reset_timestamps",
        "1",
        "-segment_format",
        "mp4",
        # CRITICAL: Force fragmented headers so every chunk has a valid moov atom instantly
        "-segment_format_options",
        "movflags=frag_keyframe+empty_moov+default_base_moof",
        segment_pattern,
    ]

    # 2. Start the FFmpeg pipe INSIDE the process
    video_writer = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    try:
        # while True:
        #     # 3. Read the frame array directly from the queue
        #     frame = write_queue.get()

        #     # Poison pill to stop the process cleanly
        #     if frame is None:
        #         break

        #     # =================================================================
        #     # 🛠️ SAFE BYTE EXTRACTION
        #     # =================================================================
        #     if torch.is_tensor(frame):
        #         # Move to CPU, convert to NumPy, and get bytes
        #         raw_bytes = frame.detach().cpu().numpy().tobytes()
        #     elif isinstance(frame, np.ndarray):
        #         raw_bytes = frame.tobytes()
        #     else:
        #         raw_bytes = bytes(frame) # Fallback for raw byte strings
        #     # =================================================================

        #     # Write the raw bytes to the FFmpeg pipe
        #     video_writer.stdin.write(raw_bytes)

        #     # Help garbage collection in the isolated process
        #     del frame, raw_bytes
        while True:
            # Get the SLOT INDEX from the queue (not the full frame)
            slot_idx = write_queue.get()
            if slot_idx is None:
                break

            # 3. Access the frame directly from the shared memory buffer
            frame = frame_buffers[slot_idx]
            video_writer.stdin.write(frame.tobytes())

    except Exception as e:
        logging.error(f"[VIDEO WRITER] Loop exception: {e}")
    finally:
        video_writer.stdin.close()
        video_writer.wait()
        # Clean up process-local attachments
        for shm in shm_blocks:
            shm.close()


# ----- FASTAPI APPLICATION STARTUP/SHUTDOWN -----
# The lifespan parameter handles startup and shutdown
async def auto_cleanup_janitor(app):
    """
    A background task that runs periodically to find and clean up stale or inactive streams.
    This acts as a safety net to prevent resource leaks from streams that do not terminate correctly.
    """
    while True:
        await asyncio.sleep(10)
        now = time.time()

        # --- Stream Monitoring ---
        async with app.state.stream_lock:
            # Iterating over a list of keys to avoid "dictionary changed size" error
            for name, streamer in list(app.state.active_streams.items()):
                # streamer = app.state.active_streams.get(name)
                if not streamer:
                    app.state.active_streams.pop(name, None)
                    continue

                # If the stream handler marked itself inactive or stopped
                # if getattr(streamer, "_is_stopped", False) or not getattr(streamer, "active", True):
                #     main_app_logger.info(f"JANITOR: Removing dead stream {name}")
                #     app.state.active_streams.pop(name, None)
                #     if not streamer._is_stopped:
                #         loop = asyncio.get_event_loop()
                #         loop.run_in_executor(None, streamer.stop)
                #     continue

                if hasattr(
                    streamer, "last_heartbeat"
                ):  # and (getattr(streamer, "active", False) == "RUNNING"):
                    #     # Inactivity heartbeat check (e.g., 5 seconds with no new frames)
                    #     if now - getattr(streamer, "last_heartbeat", now) > 15.0:
                    #         main_app_logger.info(f"JANITOR: Stream {name} timed out. Evicting.")
                    #         app.state.active_streams.pop(name, None)
                    #         if not streamer._is_stopped:
                    #             loop = asyncio.get_event_loop()
                    #             loop.run_in_executor(None, streamer.stop)
                    # continue

                    ai_backlog = streamer.get_executor_backlog()
                    video_backlog = (
                        streamer.write_queue.qsize()
                        if streamer.config.ENABLE_QUERYING
                        else 0
                    )
                    # io_backlog = (
                    #     streamer.io_executor._work_queue.qsize()
                    #     if hasattr(streamer, "io_executor")
                    #     else 0
                    # )

                    # Check if the stream is marked inactive OR timed out
                    # streamer.active should be False when the video source ends
                    is_stale = now - streamer.last_heartbeat > 30  # No activity for 30s
                    is_hung = (
                        now - streamer.last_heartbeat > 90
                    )  # Hard timeout after 90s

                    should_remove = False
                    reason = ""

                    # Conditions to remove stream
                    if streamer._is_stopped:
                        should_remove, reason = True, "Handler already stopped"
                    elif not streamer.active and (
                        ai_backlog == 0 and video_backlog == 0  # and io_backlog == 0
                    ):
                        should_remove, reason = True, "Video ended naturally"
                    elif is_stale and (
                        ai_backlog == 0 and video_backlog == 0  # io_backlog == 0
                    ):
                        should_remove, reason = True, "Browser tab closed/Network lost"
                    elif is_hung:
                        should_remove, reason = True, "Hard timeout for hung processes"

                    if should_remove:
                        async with app.state.stream_lock:
                            if BASE_PIPELINE_CONFIG.DEBUG == "1":
                                main_app_logger.info(
                                    f"CLEANUP: Removing {name} from active_streams: {reason}"
                                )
                            try:
                                if not streamer._is_stopped:
                                    streamer.stop()
                                    streamer.stop_threads(["process_thread"])
                                app.state.active_streams.pop(name, None)
                            finally:
                                del streamer

            gc.collect()  # Final garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages FastAPI startup and shutdown lifecycle.
    Guarantees release on shutdown.
    """

    # --- STARTUP ---
    if not hasattr(app.state, "classes"):
        app.state.classes = None

    if not hasattr(app.state, "active_streams"):
        app.state.active_streams = {}

    app.state.status = "Ready"
    app.state.stream_lock = asyncio.Lock()

    # Warmup shared model (if configured)
    if BASE_PIPELINE_CONFIG.SHARED_MODEL:
        app.state.model = YOLO(
            BASE_PIPELINE_CONFIG.model_path, verbose=False, task="detect"
        )

        device_input = "cuda" if BASE_PIPELINE_CONFIG.DEVICE == "GPU" else "cpu"
        main_app_logger.info("Starting shared model warmup...")
        try:
            dummy_input = torch.zeros(
                (1, 3, BASE_PIPELINE_CONFIG.MODEL_H, BASE_PIPELINE_CONFIG.MODEL_W)
            ).to(device_input)
            for _ in range(5):
                _ = app.state.model(dummy_input, verbose=False)
        finally:
            del dummy_input
            if "cuda" in device_input:
                torch.cuda.empty_cache()
        main_app_logger.info("Shared model warmup and VRAM purge complete.")

    janitor_task = asyncio.create_task(auto_cleanup_janitor(app))

    if BASE_PIPELINE_CONFIG.DEBUG == "1":
        main_app_logger.info(f"--- APP STARTUP | PID: {os.getpid()} | STATE READY ---")

    yield

    # --- CLEANUP ---
    janitor_task.cancel()
    async with app.state.stream_lock:
        for name, streamer in list(app.state.active_streams.items()):
            main_app_logger.info(f"Shutting down stream: {name}")
            streamer.stop()  # Custom stop method defined below
            streamer.stop_threads(["process_thread"])
            app.state.active_streams.pop(name, None)
            del streamer

    # Clear all application state
    app.state.active_streams.clear()
    if hasattr(app.state, "model"):
        del app.state.model

    gc.collect()  # Final garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    app.state.status = "Stopped"


# ----- INGESTION FUNCTIONS -----
def send_metadata(
    VDMS_POOL=None,
    DEBUG_FLAG=BASE_PIPELINE_CONFIG.DEBUG_FLAG,
    INGESTION=BASE_PIPELINE_CONFIG.INGESTION,
    TEST_MODE=BASE_PIPELINE_CONFIG.TEST_MODE,
    UDF_HOST=BASE_PIPELINE_CONFIG.UDF_HOST,
    UDF_PORT=BASE_PIPELINE_CONFIG.UDF_PORT,
    DBHOST=BASE_PIPELINE_CONFIG.DBHOST,
    DBPORT=BASE_PIPELINE_CONFIG.DBPORT,
):
    """
    Consumer thread that sends metadata to VDMS.
    If retries fail, it saves the data to a local JSON 'dead-letter' file.
    """
    if VDMS_POOL is None:
        # VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)
        VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)

    global all_metadata, send_metadata_queue
    clip_filename = ""
    clip_key = ""
    width = 0
    height = 0

    while True:
        queue_details = None
        clip_metadata = None

        try:
            queue_details = send_metadata_queue.get()
            # send_metadata_queue.task_done()
            if queue_details is None:
                main_app_logger.info(
                    "[METADATA-DEBUG D-pill] Poison pill received. Terminating send_metadata thread loops cleanly.",
                )
                break

            # # (clip_key, clip_filename, width, height, clip_metadata) = queue_details
            (clip_filename, width, height) = queue_details
            clip_key = Path(clip_filename).name
            clip_metadata = all_metadata.pop(clip_key, None)

            if clip_metadata:
                main_app_logger.info(
                    f"sending 2 vdms: {clip_key} with {len(clip_metadata)} items."
                )
                success = metadata2vdms_with_retry(
                    clip_key,
                    clip_filename,
                    clip_metadata,
                    width,
                    height,
                    VDMS_POOL=VDMS_POOL,
                    DEBUG_FLAG=DEBUG_FLAG,
                    INGESTION=INGESTION,
                    TEST_MODE=TEST_MODE,
                    UDF_HOST=UDF_HOST,
                    UDF_PORT=UDF_PORT,
                    DBHOST=DBHOST,
                    DBPORT=DBPORT,
                )

                # CUSTOM ERROR HANDLER: Final Failure Fallback
                if not success:
                    error_path = f"{BASE_PIPELINE_CONFIG.CODE_DIR}/failed_metadata/{clip_key}.json"
                    os.makedirs(os.path.dirname(error_path), exist_ok=True)

                    with open(error_path, "w") as f:
                        json.dump(
                            {
                                "clip_filename": clip_filename,
                                "width": width,
                                "height": height,
                                "metadata": clip_metadata,
                                "failed_at": datetime.now().isoformat(),
                            },
                            f,
                        )

                    main_app_logger.error(
                        f" [CRITICAL] Permanent VDMS failure. Data saved to: {error_path}"
                    )

            else:
                main_app_logger.error(
                    f" [MISSING] Metadata for {clip_key} was lost before upload!"
                )

        except Exception as e:
            # pass
            main_app_logger.info(
                f"[EXCEPTION] Exception occurred in send_metadata: {e}"
            )
            traceback.print_exc()
        finally:
            if "queue_details" in locals():
                del queue_details
            if "clip_metadata" in locals():
                del clip_metadata
            send_metadata_queue.task_done()


_KNOWN_HANDLER_METHODS = {}


def get_known_handler_methods(handler_class):
    if handler_class not in _KNOWN_HANDLER_METHODS:
        handler_methods = {}
        for cls in [handler_class, DeviceBaseHandler]:
            for name, attr in inspect.getmembers(cls, predicate=inspect.isfunction):
                handler_methods[name] = attr
        _KNOWN_HANDLER_METHODS[handler_class] = handler_methods
    return _KNOWN_HANDLER_METHODS[handler_class]


def standalone_writer_process(write_queue, fps, width, height, output_path):
    """Runs completely isolated from the GIL and main AI loop."""
    # 1. Initialize FFmpeg INSIDE the new process
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        while True:
            frame_data = write_queue.get()
            if frame_data is None:  # Poison pill to stop
                break

            # Write the raw bytes directly to FFmpeg
            writer.stdin.write(frame_data.tobytes())

    except Exception as e:
        logging.error(f"[WRITER PROCESS] Error: {e}")
    finally:
        writer.stdin.close()
        writer.wait()


# ----- STREAM HANDLERS -----
def get_test_handler(test_class_self, device):  # Resolve concrete handler class type
    HandlerClass = GPUStreamHandler if device == "gpu" else CPUStreamHandler
    # HandlerClass.pipeline_fn = test_class_self.__class__.pipeline_fn

    # Dynamically re-bind backend methods to this execution instance
    handler_classes = [HandlerClass, DeviceBaseHandler]
    handler_methods = get_known_handler_methods(HandlerClass)
    for method_name, method_func in handler_methods.items():
        if method_name in ["pipeline_fn", "config"]:  # "run_realtime_inference"
            continue

        if not hasattr(test_class_self.__class__, method_name):
            setattr(
                test_class_self,
                method_name,
                types.MethodType(method_func, test_class_self),
            )

    orig_methods = list(
        sorted(
            set(
                [
                    k
                    for h in handler_classes + [test_class_self]
                    for k in h.__dict__.keys()
                ]
            )
        )
    )
    return test_class_self, orig_methods


class DeviceBaseHandler:
    def __init__(
        self, source, name, active_streams, config=BASE_PIPELINE_CONFIG, **kwargs
    ):
        self.name = name
        self.source = source
        self.is_rtsp = str(self.source).startswith("rtsp:/")
        self.active = True
        self.active_streams = active_streams
        self.config = config

        sf_name = "SF" if config.sf_enabled else "noSF"
        self._testMethodName = kwargs.get(
            "run_name",
            f"{name}_{sf_name}_{config.DETECTION_TYPE.lower()}_{config.DEVICE.lower()}",
        )

        configstr = "\n".join(
            [f"\t{k}: {v}" for k, v in config.__dict__.items() if not k.startswith("_")]
        )
        main_app_logger.info(f"PipelineConfig: \n{configstr}\n")

        self.loop = asyncio.get_event_loop()
        self.frame_ready_event = asyncio.Event()
        self._is_stopped = False
        self._stop_lock = threading.Lock()  # Local lock for this instance
        self.main_startup_event = mp.Event()

        # From global
        self.device = self.config.DEVICE
        self.device_input = self.config.device_input
        self.disp_w, self.disp_h = self.config.DISPLAY_FRAME_SIZE
        self.resize_h, self.resize_w = [self.config.MODEL_H, self.config.MODEL_W]

        self.setup_reader(
            self.config.TARGET_FPS,
            self.config.CLIP_DURATION,
            startup_event=self.main_startup_event,
        )

        # Kwargs
        # clip_duration = kwargs.get("clip_duration", CLIP_DURATION)
        self.initialize_variables()

        # Should be started before calling setup_threads
        # if hasattr(self.reader, "worker") and not self.reader.worker.is_alive():
        #     self.reader.worker.start()

        # Start dedicated inference thread and timers
        # self.stat_start_time = time.perf_counter() # timing to display frame
        self.setup_threads()
        self.last_heartbeat = time.perf_counter()

    def start(self):
        """
        Starts the decoupled ingestion and inference threads in the correct order.
        """
        # PRE-SYNC: Ensure GPU is idle before timing starts
        if self.device_input == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        main_app_logger.info(f"[START {self.name}] Starting Threads ...")

        # Start the hardware-decoupled reader first
        self.reader.start()

        # if self.config.ENABLE_QUERYING:
        #     self._initialize_writer()

        # Small delay to allow the reader's deque to populate
        time.sleep(0.1)

        # Start the producer and consumer threads
        if hasattr(self, "process_thread") and not self.process_thread.is_alive():
            self.process_thread.start()

        if not self.config.DISABLE_DETECTION:
            if hasattr(self, "render_proc") and hasattr(self.render_proc, "start"):
                self.render_proc.start()

            if hasattr(self, "display_proc") and hasattr(self.display_proc, "start"):
                self.display_proc.start()

        if (
            self.config.ENABLE_QUERYING
            and not self.config.TEST_MODE
            and not self.metadata_thread.is_alive()
        ):
            self.metadata_thread.start()

        if self.config.ENABLE_QUERYING and not self.writer_process.is_alive():
            self.writer_process.start()

        self.last_heartbeat = time.perf_counter()
        return self

    def stop(self):
        """
        A robust, sequential shutdown process that gracefully terminates all threads
        and releases all resources without race conditions, especially for live streams.
        """
        # Force the GPU to finish all active kernels, copy operations,
        # and event processing across all streams.
        # This guarantees no async operations are holding the memory locked!
        if "cuda" in str(self.device_input) and torch.cuda.is_available():
            # main_app_logger.info(f"[STOP {self.name}] Synchronizing all CUDA streams...")
            torch.cuda.synchronize(self.device_input)

        # Use a lock to prevent this complex function from being called by multiple threads at once.
        with self._stop_lock:
            if getattr(self, "_is_stopped", False):
                return

            main_app_logger.info(
                f"[STOP {self.name}] Initiating orderly shutdown sequence..."
            )

            if hasattr(self, "write_queue") and hasattr(self, "writer_process"):
                try:
                    self.write_queue.put(
                        None, timeout=1.0
                    )  # The poison pill for the process
                except (queue.Full, AttributeError):
                    pass

            # Signal all active loops to stop accepting new work.
            self.active = False
            self.prefetch_active = False

            if hasattr(self, "active_streams") and isinstance(
                self.active_streams, dict
            ):
                self.active_streams.pop(self.name, None)

            # Wake up any FastAPI async streaming generators waiting on frame_ready_event
            if (
                hasattr(self, "frame_ready_event")
                and self.frame_ready_event is not None
            ):
                try:
                    if hasattr(self, "loop") and self.loop and self.loop.is_running():
                        self.loop.call_soon_threadsafe(self.frame_ready_event.set)
                    else:
                        self.frame_ready_event.set()
                except Exception:
                    pass

            main_app_logger.info(f"[STOP {self.name}] Stopping writer event...")
            self.set_stop_writer_event(remove=False)

            # Shut down the worker pools. wait=True is crucial.
            # This blocks until all background tasks have finished.
            main_app_logger.info(
                f"[STOP {self.name}] Shutting down worker executors..."
            )
            self.stop_executors(["clip_executor", "executor"])

            # Join all other background threads.
            main_app_logger.info(
                f"[STOP {self.name}] Joining all background threads..."
            )
            self.stop_threads(
                [
                    "process_thread",
                    "writer_process",
                    "metadata_thread",
                    "display_proc",
                    "render_proc",
                ]
            )
            # if (
            #     hasattr(self, "writer_process") and self.writer_process.is_alive()
            # ):  # actually a process
            #     self.writer_process.join(timeout=3.0)
            #     if self.writer_process.is_alive():
            #         self.writer_process.terminate()
            self.unregister_pinned_cuda_data()
            buffer_pools = [
                "shm_buffer_pool",
                "gpu_input",
                "clipper_shm_np_views",
                "pinned_matrices",
                "pinned_tensors",
            ]
            self.clear_buffer_pools(buffer_pools)

            if hasattr(self, "reader") and self.reader is not None:
                if hasattr(self.reader, "pinned_views"):
                    self.reader.pinned_views = []
                if hasattr(self.reader, "_static_gpu_frame_buffer"):
                    self.reader._static_gpu_frame_buffer = None

            if hasattr(self, "reader") and self.reader is not None:
                main_app_logger.info(f"[STOP {self.name}] Stopping reader process...")
                self.reader.stop()
                main_app_logger.info(f"[STOP {self.name}] Reader process stopped.")

            if hasattr(self, "prefetch_threads") and self.prefetch_threads:
                main_app_logger.info(f"[STOP {self.name}] Stopping prefetch threads...")
                for val in self.prefetch_threads:
                    self.stop_thread(val)
                self.prefetch_threads.clear()
                # delattr(self, "prefetch_threads")

            if hasattr(self, "async_writer") and self.async_writer is not None:
                main_app_logger.info(f"[STOP {self.name}] Releasing writer...")
                try:
                    self.async_writer.release()
                except Exception:
                    pass
                setattr(self, "async_writer", None)

            # Clear the buffer that holds direct pointers to the shared memory.
            # This is the most critical step to release the "exported pointers".
            # buffer_pools = ["shm_buffer_pool", "gpu_input"]
            # buffer_pools = [
            #     "shm_buffer_pool",
            #     "gpu_input",
            #     "clipper_shm_np_views",
            #     "pinned_matrices",
            #     "pinned_tensors",
            # ]
            # self.clear_buffer_pools(buffer_pools)
            # if hasattr(self, "shm_buffer_pool") and self.shm_buffer_pool is not None:
            #     main_app_logger.info(
            #         f"[STOP {self.name}] Clearing shared memory buffer pool to release pointers..."
            #     )
            #     for i in range(len(self.shm_buffer_pool)):
            #         self.shm_buffer_pool[i] = None
            #     # self.shm_buffer_pool = None

            # if hasattr(self, "gpu_input") and self.gpu_input is not None:
            #     # main_app_logger.info(
            #     #     f"[STOP {self.name}] Clearing shared memory buffer pool to release pointers..."
            #     # )
            #     for i in range(len(self.gpu_input)):
            #         self.gpu_input[i] = None
            #     self.gpu_input = None
            #     # self.gpu_input.clear()

            # Force instant garbage collection to drop Python buffer exports from memoryview
            gc.collect()
            if "cuda" in str(self.device_input) and torch.cuda.is_available():
                torch.cuda.empty_cache()

            shm_names = [
                "clipper_shm_blocks",
                "reader.shms",
                "shms",
            ]
            main_app_logger.info(f"[STOP {self.name}] Clearing shared memory...")
            self.clear_shared_memory_list(shm_names)

            # if hasattr(self, "clipper_shm_blocks"):
            #     for shm in self.clipper_shm_blocks:
            #         shm.close()
            #         shm.unlink()  # This removes the file from /dev/shm

            # shms_to_close = []  # self.shms
            # if hasattr(self, "reader") and self.reader is not None:
            #     main_app_logger.info(
            #         f"[STOP {self.name}] Calling reader.stop() to clean up worker process..."
            #     )
            #     self.reader.stop()
            #     if hasattr(self.reader, "shms"):
            #         shms_to_close += self.reader.shms
            # self.reader = None

            # if hasattr(sys, "exc_info"):
            #     # This clears the three-element tuple (type, value, traceback)
            #     exc_info = sys.exc_info()
            #     if exc_info[2] is not None: # Check if a traceback object exists
            #         traceback.clear_frames(exc_info[2])
            # # For older python versions, or as an extra measure
            # if hasattr(sys, "exc_clear"):
            #     sys.exc_clear()

            # gc.collect()
            # if "cuda" in str(self.device_input) and torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     if hasattr(torch.cuda, "ipc_collect"):
            #         torch.cuda.ipc_collect()

            # if shms_to_close:
            #     main_app_logger.info(f"[STOP {self.name}] Closing OS shared memory files...")
            #     for shm in shms_to_close:
            #         try:
            #             shm.close()
            #             shm.unlink()
            #         except Exception:
            #             pass
            #     shms_to_close.clear()

            # =========================================================================
            # 🕵️‍♂️ ADVANCED REFERENCE INVESTIGATOR
            # =========================================================================
            # import gc
            # import inspect

            main_app_logger.info("\n" + "=" * 80)
            main_app_logger.info(
                "🕵️‍♂️ [STOP INVESTIGATION] Scanning for active pointers to SharedMemory blocks..."
            )
            # main_app_logger.info("=" * 80)

            # if hasattr(self, "reader") and hasattr(self.reader, "shms"):
            #     for idx, shm in enumerate(self.reader.shms):
            #         main_app_logger.info(f"\n[SHM BLOCK {idx}] Name: {shm.name}")

            #         # Create a function to recursively find the ultimate owner
            #         def find_owner(obj, depth=0):
            #             if depth > 5:  # Safety limit to prevent infinite recursion
            #                 return

            #             referrers = gc.get_referrers(obj)
            #             for ref in referrers:
            #                 # Ignore the current function's local variables
            #                 if ref is referrers or ref is locals():
            #                     continue

            #                 ref_type_name = type(ref).__name__

            #                 # If we find a class instance, we found the owner!
            #                 if hasattr(ref, "__dict__"):
            #                     # Check if it's a class instance and not just a dict
            #                     if not isinstance(ref, (dict, list, tuple, set)):
            #                         for k, v in ref.__dict__.items():
            #                             if v is obj:
            #                                 main_app_logger.info(
            #                                     f"{'  ' * depth}└── Found Owner: Class <{type(ref).__name__}> holds reference in attribute '{k}'"
            #                                 )

            #                 # If the referrer is a list or tuple, recurse deeper
            #                 elif isinstance(ref, (list, tuple)):
            #                     main_app_logger.info(
            #                         f"{'  ' * depth}└── Held by container: <{ref_type_name}> of len {len(ref)}"
            #                     )
            #                     find_owner(ref, depth + 1)

            #         find_owner(shm)
            # main_app_logger.info("=" * 80 + "\n")
            # =========================================================================

            # 8. Clean up any remaining handler resources like queues and events.
            main_app_logger.info(f"[STOP {self.name}] Cleaning up final resources...")
            self.stop_sync_manager()
            self.drain_and_close_queues(
                ["signal_queue", "render_queue", "prefetch_queue", "write_queue"]
            )

            remove_attrs = [
                "_cached_grid_x",
                "_cached_grid_y",
                "ffmpeg_proc",
                "frame_in_clip_count",
                "latest_processed_frame",
                "mp_frame_ready_flag",
                "mp_last_id",
                "processing_stream",
                "reader_active_idx",
                "ready_buffer_idx",
                "render_queue_backlog_counter",
                "render_queue_counter",
                "shm_frame_lengths",
                "stat_start_time",
                "write_queue_backlog_counter",
                "write_queue_counter",
                "static_gpu_360p",
                "static_gpu_byte_bchw",
                "gpu_float_staging",
                "pinned_matrices",
                "pinned_tensors",
            ]
            main_app_logger.info(f"[STOP {self.name}] Removing attributes...")
            self.remove_scalar_attributes(remove_attrs)

            # self.stop_events()
            events = [
                "frame_ready_event",
                "queue_data_ready_event",
                "_d2h_fence",
                "det_end",
                "det_start",
                "frame_ready_event",
                "main_startup_event",
                "queue_data_ready_event",
                "reader_lock",
                "roi_end",
                "roi_start",
                "sf_end",
                "sf_start",
                "worker_tracking_lock",
            ]
            main_app_logger.info(f"[STOP {self.name}] Stopping events...")
            self.stop_events(events, keys_to_skip_deletion=default_attr_keys)

            if hasattr(self, "slot_events") and self.slot_events is not None:
                main_app_logger.info(f"[STOP {self.name}] Stopping slot events...")
                # Grab local reference and immediately nullify instance attribute
                events_to_clean = self.slot_events
                self.slot_events = None
                if events_to_clean is not None:
                    for ev in list(events_to_clean):
                        try:
                            if isinstance(ev, torch.cuda.Event):
                                # Force internal driver handle release
                                del ev
                        except Exception:
                            pass
                    # self.slot_events = None

            if getattr(self, "processor", None):
                delattr(self, "processor")

            if getattr(self, "evaluator", None):
                delattr(self, "evaluator")

            # self.pinned_matrices.clear()
            # self.pinned_tensors.clear()

            self.clean_up_tensors_and_arrays()

            release_native_linux_heap()

            gc.collect()
            if "cuda" in self.device_input and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

            # with self._stop_lock:
            self._is_stopped = True
            self.status = "DONE"
            self.active_streams.pop(self.name, None)
            main_app_logger.info(f"[STOP {self.name}] Shutdown complete.")

    def setup_threads(self):
        """Overrides handlers.py to bind threads dynamically to the test instance."""

        self._cached_grid_y, self._cached_grid_x = torch.meshgrid(
            torch.arange(
                self.resize_h, device=self.device_input
            ),  # Match target tracking resolution
            torch.arange(self.resize_w, device=self.device_input),
            indexing="ij",
        )

        self.setup_shared_memory()  # Natively sets up Manager dictionary and buffers

        # Executor for Async YOLO tasks and FFmpeg re-encoding
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        self.clip_executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        main_app_logger.info(
            f"sf_enabled: {self.config.sf_enabled}\tTEST_MODE: {self.config.TEST_MODE}",
        )

        self.process_thread = threading.Thread(
            target=self.run_realtime_inference,
            kwargs={
                "sf_enabled": self.config.sf_enabled,
                "gt_enabled": getattr(self, "gt_enabled", False),
            },
            daemon=True,
        )

        # # Open up looking-ahead buffer horizons to eliminate 8K queue backpressure stalls
        # self.render_queue_maxsize = 16  # 4
        # self.render_queue = queue.Queue(maxsize=self.render_queue_maxsize)

        # self.signal_queue_maxsize = 32  # 8
        # self.signal_queue = queue.Queue(maxsize=self.signal_queue_maxsize)

        if self.config.TEST_MODE:
            test_dir = os.getenv(
                "TEST_SUITE_RENDER_DIR", str(Path(self.config.SHARED_OUTPUT))
            )
            os.makedirs(test_dir, exist_ok=True)

            if hasattr(self, "video_output_name"):
                video_output_name = self.video_output_name
            else:
                if self.source.startswith("rtsp"):
                    short_name = "rtsp"
                else:
                    short_name = Path(self.source).stem
                video_output_name = f"{self._testMethodName}_{short_name}.mp4"

            self.output_path = os.path.join(test_dir, video_output_name)
            # self.output_path = os.path.join(
            #     test_dir, f"{self.name}_detections_output.mp4"
            # )
            # log_to_logger(
            #     f"[TEST MODE] Detection results saved to: {self.output_path}",
            #     level="info",
            # )

            self.async_writer = AsyncVideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*"avc1"),  # avc1, mp4v
                float(self.target_fps),
                (self.disp_w, self.disp_h),
            )

            # Dummy target alignment to prevent execution signature exceptions
            # self.render_proc = threading.Thread(target=lambda: None, daemon=True)
            # self.display_proc = threading.Thread(target=lambda: None, daemon=True)
            # self.render_proc = DummyProcess()
            # self.display_proc = DummyProcess()
            log_to_logger(
                f"[TEST MODE] Results saved to: {self.output_path}", level="info"
            )
        else:
            self.async_writer = AsyncDisplayVideoWriter(
                float(self.target_fps),
                (self.disp_w, self.disp_h),
                quality=int(self.config.DISPLAY_FRAME_QUALITY),
            )
            self.async_writer.set_handler_context(self)
            # self.async_writer.start_worker()
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    self.loop = asyncio.get_event_loop()
                except RuntimeError:
                    self.loop = None  # Headless fallback context

        if self.config.ENABLE_QUERYING:
            # NEW: Dedicated I/O pool for Disk/GPU transfers (Higher worker count for 8K)
            # self.io_executor = ThreadPoolExecutor(max_workers=8)

            # Dedicated FFmpeg pool so re-encoding doesn't slow down live AI
            # self.ffmpeg_executor = ThreadPoolExecutor(max_workers=2)

            if not self.config.TEST_MODE:
                # Sends metadata to VDMS
                self.metadata_thread = threading.Thread(
                    target=send_metadata,
                    args=(
                        VDMSPool(self.config.DBHOST, self.config.DBPORT, size=10),
                        self.config.DEBUG_FLAG,
                        self.config.INGESTION,
                        self.config.TEST_MODE,
                        self.config.UDF_HOST,
                        self.config.UDF_PORT,
                        self.config.DBHOST,
                        self.config.DBPORT,
                    ),
                    daemon=True,
                )

            # Consumer: Handles GPU-to-CPU download and Disk I/O (Writing resized frames to RAM disk)
            # self.writer_process = threading.Thread(
            #     target=self.video_writer_core_loop,
            #     args=(self.stop_writer,),
            #     daemon=True,
            # )

            # self.render_proc = threading.Thread(target=lambda: None, daemon=True)
            # self.display_proc = threading.Thread(target=lambda: None, daemon=True)
            # self.render_proc = DummyProcess()
            # self.display_proc = DummyProcess()

    def initialize_variables(self):
        self.prefetch_threads = []
        # Track active worker threads atomically to protect end-of-stream drainage
        num_prefetch_workers = 4 if self.device_input == "cpu" else 3
        self.active_workers_count = num_prefetch_workers
        self.worker_tracking_lock = threading.Lock()

        # Override the atomic tracking count to match our active pool allocation
        with self.worker_tracking_lock:
            self.active_workers_count = num_prefetch_workers

        # self.initialize_run_realtime_inference(read_frame_only, num_prefetch_workers)

        # 1. HARD GUARD: Capture reader values and ensure the connection is active
        if not hasattr(self, "result_dir"):
            self.result_dir = self.config.SHARED_OUTPUT
        self.device_index = f"cuda:{self.gpu_id}" if self.device_input else "cpu"
        self.input_fps = self.reader.input_fps
        self.target_fps = self.reader.target_fps
        self.frame_width = self.reader.frame_width
        self.frame_height = self.reader.frame_height
        self.numFrames = self.reader.numFrames
        # self.min_roi_w = int(self.config.ROI_MIN_AREA_RATIO * self.resize_w)
        # self.min_roi_h = int(self.config.ROI_MIN_AREA_RATIO * self.resize_h)
        # self.max_roi_w = int(self.resize_w * self.config.ROI_MAX_RELATIVE_SIZE_RATIO)
        # self.max_roi_h = int(self.resize_h * self.config.ROI_MAX_RELATIVE_SIZE_RATIO)
        # self.max_cached_elements = 100

        if self.input_fps <= 0 or self.frame_width <= 0 or self.frame_height <= 0:
            main_app_logger.error(
                f"[{self.name}] Stream handler fast-fail triggered. Destination "
                f"unreachable or invalid ({self.source}). Terminating pipeline configuration."
            )
            # Instantly stop background threads to prevent zombie process leakage
            if hasattr(self, "reader") and self.reader is not None:
                self.reader.stop()

            raise RuntimeError(
                f"Failed to initialize stream reader endpoint: {self.source}"
            )

        # 2. Proceed with calculation mechanics safely only if values are healthy
        self.step_size = (
            float(self.input_fps) / float(self.target_fps)
            if hasattr(self, "target_fps")
            else 1.0
        )
        self.frame_skip = self.reader.frame_skip
        self.max_frames_per_clip = self.reader.max_frames_per_clip
        self.frame_interval = self.reader.frame_interval

        self.duration_s = self.numFrames / self.input_fps
        self.expected_num_frames = int(self.duration_s * self.target_fps)
        self.get_frameWH()

        # Determine minimum contour size relative to frame resolution
        # self.min_contour_area = int(
        #     (self.min_roi_h)
        #     * (self.min_roi_h)
        # )  # 207

        # self.dist_thresh_8k = max(
        #     self.config.ROI_DISTANCE_THRESH_RATIO * self.frame_width,
        #     self.config.ROI_DISTANCE_THRESH_RATIO * self.frame_height,
        # )
        # multiplier = 2.0 if self.device_input == "cpu" else 1.0
        # self.dist_thresh_640 = (
        #     max(
        #         self.config.ROI_DISTANCE_THRESH_RATIO * self.resize_w,
        #         self.config.ROI_DISTANCE_THRESH_RATIO * self.resize_h,
        #     )
        #     * multiplier
        # )  # 0.05 * self.resize_w
        # self.scales_tensor = torch.tensor(
        #     [self.scale_x, self.scale_y, self.scale_x, self.scale_y],
        #     # device="cpu",
        #     device=self.device_input,
        # )

        # self.disp_w, self.disp_h = self.config.DISPLAY_FRAME_SIZE

        # Performance Tracking
        self.elapsed_display_time = 0.0
        self.frame_count = 0  # Frame count for videos
        self.frame_count_target = 0
        self.last_delivered_frame_id = -1  # Track what was actually sent
        self.last_frame_id = 0
        self.latest_processed_frame = None
        self.next_process_idx = 0.0
        self.stat_fps = 0
        self.stat_frame_count = 0
        self.abs_frame_num = 0
        self.total_objects_detected = 0
        self.is_cuda = self.device.lower() == "gpu" and torch.cuda.is_available()
        self.frame_in_clip_count = 0

        self.writer_done = True

        # Video Clipping
        # self.video_writer = None
        self.ffmpeg_proc = None  # Replaces cv2.VideoWriter completely
        # self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1, mp4v
        # self.clip_id = 0
        # self.clip_filename = ""
        self.clip_filename_pattern = f"{self.config.SHARED_OUTPUT}/{self.name}_%03d.mp4"
        # self.clip_key = f"{self.name}_000.mp4"
        # self.tmp_file = ""
        # self.frame_in_clip_count = 0

        if self.config.ENABLE_QUERYING:
            self.gpu_float_staging = torch.empty(
                (1, 3, self.frame_height, self.frame_width),
                dtype=torch.float16,
                device=self.device_index,
            )
            # Thread-safe queue for the resized frames (640x640)
            # maxlen=300 allows for a 20-second buffer in case of extreme disk lag
            # Non-blocking queue for frames and control signals
            # self.write_queue = queue.Queue(maxsize=int(self.target_fps))  # 300)

            # self.write_queue = queue.Queue(
            #     maxsize=int(self.target_fps / 2)
            #     if self.device_input == "cpu"
            #     else int(self.target_fps / 2)
            # )  # 300)

            # Define the depth of your shared memory ring buffer. 16 is a safe number.
            self.clipper_ring_depth = 16
            self.clipper_shm_frame_bytes = self.config.MODEL_W * self.config.MODEL_H * 3

            # Create the shared memory blocks
            self.clipper_shm_blocks = [
                SharedMemory(create=True, size=self.clipper_shm_frame_bytes)
                for _ in range(self.clipper_ring_depth)
            ]

            # Get the names to pass to the new process
            self.clipper_shm_names = [shm.name for shm in self.clipper_shm_blocks]

            # Create NumPy array views that the main process will use to write data
            self.clipper_shm_np_views = [
                np.ndarray(
                    (self.config.MODEL_H, self.config.MODEL_W, 3),
                    dtype=np.uint8,
                    buffer=shm.buf,
                )
                for shm in self.clipper_shm_blocks
            ]

            # An atomic, thread-safe counter to track the next available slot
            self.clipper_ring_idx = mp.Value("i", 0)
            self.clipper_idx_lock = (
                mp.Lock()
            )  # Lock to prevent race conditions when getting an index

            self.write_queue = mp.Queue(maxsize=self.clipper_ring_depth * 2)
            # if (
            #     not hasattr(self, "writer_process")
            #     or self.writer_process is None
            #     or not self.writer_process.is_alive()
            # ):
            #     main_app_logger.info(
            #         " [CLIPPER-INIT] Target worker runtime thread is offline. Provisioning core consumer loop thread...",
            #     )
            self.writer_process = mp.Process(
                target=video_writer_core_loop,
                args=(
                    self.write_queue,
                    self.clip_filename_pattern,
                    self.target_fps,
                    self.config.MODEL_W,
                    self.config.MODEL_H,
                    int(self.config.CLIP_DURATION),
                    # Pass the shared memory details to the new process
                    self.clipper_shm_names,
                    self.clipper_shm_frame_bytes,
                ),
                daemon=True,
            )
            # self.writer_process.start()
            if not self.config.TEST_MODE:
                self.send_metadata_queue = queue.Queue()
            self.writer_done = False
        self.stop_writer = threading.Event() if self.config.ENABLE_QUERYING else None

        # self._cached_grid_y, self._cached_grid_x = torch.meshgrid(
        #     torch.arange(
        #         self.resize_h, device=self.device_input
        #     ),  # Match target tracking resolution
        #     torch.arange(self.resize_w, device=self.device_input),
        #     indexing="ij",
        # )

        # self.fixed_inference_batch = torch.empty(
        #     (self.config.MODEL_MAX_BATCH_SIZE, 3, self.config.MODEL_H, self.config.MODEL_W),
        #     dtype=torch.half,
        #     device=self.device_input,
        # )
        # Create a deep look-ahead queue buffer that consumes near 0 RAM
        # because it only holds reference pointers to your 6 SHM slots!
        # if self.is_rtsp:
        # prefetch_maxsize = 10
        prefetch_maxsize = 5  # use 4 or 5; 10 risk memlock
        # else:
        #     prefetch_maxsize = (
        #         128 if self.device_input != "cpu" else int(self.target_fps)
        #     )

        self.prefetch_queue = queue.Queue(
            maxsize=prefetch_maxsize,
        )
        # self.prefetch_queue = mp.Queue(maxsize=128)
        # Initialize a thread-safe signaling handle at class setup (run_realtime_inference)
        self.queue_data_ready_event = threading.Event()
        maxsize = (
            self.prefetch_queue.maxsize
            if hasattr(self.prefetch_queue, "maxsize")
            else self.prefetch_queue._maxsize
        )
        # self.shm_buffer_pool = [None] * maxsize
        try:
            self.shm_buffer_pool = [
                # (
                #     torch.empty(
                #         (self.frame_height, self.frame_width, 3), dtype=torch.uint8
                #     ).pin_memory(),
                #     None,  # event
                #     0,  # frame_num
                #     0,  # abs_frame_num
                #     0.0,  # read_latency
                # )
                None
                for _ in range(maxsize)
            ]
        except Exception as e_:
            traceback.print_exc()
            main_app_logger.info(f"[INITIALIZATION ERROR] Error occurred: {e_}")
        self.gpu_input = [
            torch.empty(
                (self.frame_height, self.frame_width, 3),
                device=self.device_input,
                dtype=torch.uint8,
            )  # .pin_memory()
            for _ in range(maxsize)
        ]

        self.prefetch_active = True

        self.prefetch_threads = []

        # self.shm_buffer_pool = [None] * maxsize
        # self.shm_buffer_pool = mp.Manager().list([None] * maxsize)
        # global_shared_pool = mp.Manager().list([None] * maxsize)
        # self.shm_buffer_pool = global_shared_pool

        # Track active worker threads atomically to protect end-of-stream drainage
        num_prefetch_workers = 1  # 4 if self.device_input == "cpu" else 3
        self.active_workers_count = num_prefetch_workers
        self.worker_tracking_lock = threading.RLock()
        # self.obj_counter_l-ock = threading.Lock()

        # Offload the pre-fetch layer straight to a daemon thread context
        # prefetch_thread = threading.Thread(target=frame_prefetch_worker, daemon=True)
        # prefetch_thread.start()
        # num_prefetch_workers = 1 if self.device_input == "cpu" else 3
        # num_prefetch_workers = 3

        # Override the atomic tracking count to match our active pool allocation
        with self.worker_tracking_lock:
            self.active_workers_count = num_prefetch_workers

        # This guarantees frames are read from self.reader sequentially
        # and stay perfectly in order, while letting workers decode in parallel!
        self.reader_lock = threading.RLock()

        # --- PRE-ALLOCATED ZERO-COPY HARDWARE RING WORKSPACE ---
        self.ring_depth = 4  # 8  # _async_clipper_worker
        self.gpu_ring_idx = 0  # _async_clipper_worker
        self.cpu_ring_idx = 0  # _async_clipper_worker

        self.pinned_matrices = []  # _async_clipper_worker, video_writer_core_loop (CPU)
        if self.device_input == "cuda":
            self.pinned_tensors = []  # _async_clipper_worker

            self.static_gpu_byte_bchw = torch.empty(
                (1, 3, self.disp_h, self.disp_w),
                dtype=torch.uint8,
                device="cuda",
            ).contiguous()

        # Pre-allocate 640x640 workspace footprint across CPU and GPU spaces
        for _ in range(
            self.ring_depth
        ):  # _async_clipper_worker, video_writer_core_loop
            mat = np.zeros((self.resize_h, self.resize_w, 3), dtype=np.uint8)
            if self.device_input == "cuda":
                try:
                    cv2.cuda.registerPageLocked(mat)
                except cv2.error:
                    pass
                self.pinned_tensors.append(torch.from_numpy(mat))

            self.pinned_matrices.append(mat)

        # self.init_pipeline_shared_memory()

        # if self.device_input == "cuda" and not hasattr(self, "_cuda_gaussian_filter"):
        #     # Large block blur to dissolve high-frequency single-pixel speckles
        #     # ksize=(15, 15)
        #     # ksize=(11, 11)
        #     ksize = (17, 17)
        #     self._cuda_gaussian_filter = cv2.cuda.createGaussianFilter(
        #         srcType=cv2.CV_8UC1, dstType=cv2.CV_8UC1, ksize=ksize, sigma1=0
        #     )

        # if not hasattr(self, "_filter_scratch_keep"):
        #     # Existing order and boolean validation maps
        #     self._filter_scratch_keep = torch.zeros(
        #         (self.max_cached_elements,), dtype=torch.bool, device=self.device_input
        #     )
        #     self._filter_order_scratch = torch.zeros(
        #         (self.max_cached_elements,), dtype=torch.long, device=self.device_input
        #     )

        #     # Persistent coordinate layers to absorb inner-loop tensor evaluations safely
        #     self._filter_scratch_x1 = torch.zeros(
        #         (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        #     )
        #     self._filter_scratch_y1 = torch.zeros(
        #         (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        #     )
        #     self._filter_scratch_x2 = torch.zeros(
        #         (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        #     )
        #     self._filter_scratch_y2 = torch.zeros(
        #         (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        #     )
        #     self._filter_scratch_ioa = torch.zeros(
        #         (self.max_cached_elements,), dtype=torch.float32, device=self.device_input
        #     )

        # Isolate CUDA tasks using a dedicated stream and independent hardware completion barriers
        self.processing_stream = (
            torch.cuda.Stream() if self.device_input == "cuda" else None
        )

        # Cache the context manager instance persistently
        self.compiled_no_grad_gate = torch.no_grad()

        # Pre-allocated hardware events guarantee completely non-blocking stream isolation
        self.slot_events = (
            [torch.cuda.Event() for _ in range(8)]
            if self.device_input == "cuda"
            else None
        )

        # Allocate a permanent float32/float16 channel layout space directly on VRAM
        if self.device_input == "cuda":
            self.static_gpu_360p = torch.empty(
                (1, 3, self.disp_h, self.disp_w),
                dtype=torch.float32,
                device=self.device_input,
            )
            self.static_gpu_byte_bchw = torch.empty(
                (1, 3, self.disp_h, self.disp_w),
                dtype=torch.uint8,
                device=self.device_input,
            ).contiguous()

    def setup_reader(self, target_fps, clip_duration, startup_event=None):
        # if hasattr(self, "reader"):
        #     del self.reader

        self.gpu_id = 0

        # try:
        determined_queue_size = (
            # 4 if self.is_rtsp else 2  # (0 if self.config.TEST_MODE else 2)
            # 4 if self.is_rtsp else 8
            4  # max: 5
        )
        if self.device_input == "cuda":  # and not self.is_rtsp:
            # Add a tiny sleep or garbage collect to ensure the GPU handle is released
            # gc.collect()
            # torch.cuda.empty_cache()  # Clear any remaining context

            from include.readers import GPUReader

            self.reader = GPUReader(
                source=self.source,
                startup_event=startup_event,
                target_fps=target_fps,
                clip_duration=clip_duration,
                gpu_id=0,  # self.gpu_id,
                queue_size=determined_queue_size,
            )
        else:
            from include.readers import CPUReader

            self.reader = CPUReader(
                source=self.source,
                startup_event=startup_event,
                target_fps=target_fps,
                clip_duration=clip_duration,
                queue_size=determined_queue_size,
            )
        # except Exception as e:
        #     self.reader = None
        #     raise ValueError(f"Stream reader initialization failure: {e}")

    def setup_shared_memory(self):
        self.manager = mp.Manager()

        self.shms = []
        shm_names = []
        num_shms = 3

        # Shared Integer to track which buffer is "Ready" for the UI
        # 'i' for integer, initialized to 0
        self.ready_buffer_idx = mp.Value("i", 0)
        self.reader_active_idx = mp.Value("i", -1)
        self.shm_frame_lengths = mp.Array("i", [0 for _ in range(num_shms)])
        self.mp_last_id = mp.Value("i", -1)
        self.mp_frame_ready_flag = mp.Value("b", False)

        # --- HIGH-SPEED ATOMIC TELEMETRY BACKLOG REGISTERS ---
        self.render_queue_backlog_counter = mp.Value("i", 0)
        self.write_queue_backlog_counter = mp.Value("i", 0)

        display_timestamp = int(time.time_ns())

        for idx in range(num_shms):
            # self.shm = mp.shared_memory.SharedMemory(create=True, size=10*1024*1024)
            shm_name = f"shm_{self.name}_{idx}_{os.getpid()}_{display_timestamp}"
            if shm_name not in shm_names:
                try:
                    SharedMemory(name=shm_name).unlink()
                except FileNotFoundError:
                    pass
                # main_app_logger.info(f"[DEBUG]: Setting up SHM {shm_name}")
                try:
                    shm = SharedMemory(
                        name=shm_name, create=True, size=10 * 1024 * 1024
                    )
                    # safe_unregister_shm(shm.name)
                except FileExistsError:
                    # Attach to existing memory
                    shm = SharedMemory(name=shm_name)
                    # safe_unregister_shm(shm.name)
                except Exception as e:
                    main_app_logger.error(f"Failed to initialize shared memory: {e}")
                    raise
                self.shms.append(shm)
                shm_names.append(shm_name)

                # try:
                #     unregister(shm._name, "shared_memory")
                #     # main_app_logger.info(f"[HANDLER] Successfully unregistered {shm.name} from Resource Tracker.")
                # except Exception:
                #     pass

        self.signal_queue = self.manager.Queue(maxsize=32)
        self.render_queue = self.manager.Queue(maxsize=10)  # old-5

        self.shared_details = self.manager.dict()
        self.shared_details["shm_names"] = shm_names
        # self.shared_details["buffer_idx"] = 0
        # self.shared_details["frame_length"] = [0 for _ in range(num_shms)]
        self.shared_details["last_id"] = -1

    # PIPELINE FUNCTIONS --------------------------------------------

    def update_frame(self, stat_start_time):
        # if self.device_input == "cuda":
        #     torch.cuda.synchronize()

        self.stat_frame_count += 1
        self.elapsed_display_time += time.perf_counter() - stat_start_time
        # self.elapsed_display_time = time.perf_counter() - stat_start_time
        # if elapsed > 0.5:
        self.stat_fps = round(self.stat_frame_count / self.elapsed_display_time, 1)

    @torch.inference_mode()
    def pipeline_fn(
        self,
        device_frame,
        overall_frame_num,
        stat_start_time,
        current_clip_id,
        gt_boxes=None,
        read_frame_only=False,
    ):
        global all_metadata
        current_clip_key = f"{self.name}_{current_clip_id:03d}.mp4"
        current_clip_path = f"{self.config.SHARED_OUTPUT}/{current_clip_key}"

        metadata = {}
        motion_detected = False
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Given input frame, get metadata and metrics
        try:
            if read_frame_only:
                # Get frame, metadata and metrics
                _, det_frame = self.processor.format_bbs_and_frame_4_detection(
                    [], device_frame
                )

            else:
                # --- CLIP GENERATION ---
                if self.config.ENABLE_QUERYING:
                    if (
                        self.config.DEBUG_FLAG
                        and hasattr(self, "max_frames_per_clip")
                        and (
                            self.frame_in_clip_count % 15 == 0
                            or self.frame_in_clip_count == 1
                        )
                    ):
                        main_app_logger.info(
                            f"[CLIPPER] Frame progress tracking index: {self.frame_in_clip_count}/{self.max_frames_per_clip} (Overall Frame: {overall_frame_num})"
                        )

                    self.prep_frame_for_video(device_frame, overall_frame_num)

                    if self.frame_in_clip_count > self.max_frames_per_clip:
                        global clip_completion_tracker
                        if current_clip_key not in clip_completion_tracker:
                            clip_completion_tracker[current_clip_key] = {
                                "video": False,
                                "meta": False,
                                "start_time": time.time(),
                            }

                        clip_completion_tracker[current_clip_key]["meta"] = True
                        # main_app_logger.info(
                        #     f" [BARRIER-SEAL] All metadata extracted for {current_clip_key}. Evaluating convergence...",
                        #
                        # )
                        self._evaluate_barrier_and_dispatch(
                            current_clip_key,
                            current_clip_path,
                            self.resize_w,
                            self.resize_h,
                        )

                        self.start_new_clip(current_clip_id)

                # Run model on frame, get metadata and metrics
                with torch.inference_mode():
                    metrics, metadata, det_frame, motion_detected = self.processor.run(
                        device_frame,
                        overall_frame_num,
                        frame_in_clip_count=self.frame_in_clip_count,
                        gt_boxes=None,
                    )

            num_objs = len(metadata.keys())
            self.total_objects_detected += num_objs

            if self.config.DEBUG_FLAG:
                meta_keys = ", ".join(list(metadata.keys()))
                main_app_logger.info(
                    f"[DEBUG] {current_clip_key} metadata keys: {meta_keys}",
                )

            if num_objs > 0:
                if current_clip_key not in all_metadata:
                    all_metadata[current_clip_key] = {"object": {}, "face": {}}

                all_metadata[current_clip_key]["object"].update(metadata)

            # data_to_draw = metadata
            if self.config.DEBUG_FLAG and overall_frame_num % 50 == 0:
                main_app_logger.info(
                    f"[METADATA-DEBUG A] Logged tracking structures for Frame #{overall_frame_num} into dictionary key: {current_clip_key}. Number detections: {self.total_objects_detected}",
                )

            if self.config.TEST_MODE:
                self.frame2video(
                    det_frame,
                    overall_frame_num,
                    metadata,
                    getattr(self, "label_source", None),
                    stat_start_time,
                )
            else:
                self.frame2output(
                    det_frame,
                    overall_frame_num,
                    metadata,
                    self.label_source,
                    stat_start_time,
                )

            # with self.obj_counter_lock:
            #     self.num_objs += len(metadata.keys())
        # except (TypeError, IndexError):
        #     # App is shutting down and processor/reader not available
        #     self.active = False

        except Exception as e_detection:
            # traceback.print_exc()
            # if self.active:
            traceback.print_exc()
            main_app_logger.info(f"[DETECTION ERROR] Exception: {e_detection}")
            # self.active = False

        finally:
            # del inf_data, bbs_full_res, device_frame, det_frame
            if "det_frame" in locals():
                del det_frame
            if "device_frame" in locals():
                del device_frame
            # if "bbs_full_res" in locals():
            #     del bbs_full_res
            # if "inf_data" in locals():
            #     del inf_data

        return metadata, metrics  # Skip full detection pass

    def initialize_run_realtime_inference(self, read_frame_only, num_prefetch_workers):
        # self.duration_target = 30
        # self.status = "RUNNING"
        self.processor = self._setup_processor(read_frame_only)

        if hasattr(self, "processor") and hasattr(self.processor, "label_source"):
            self.label_source = self.processor.label_source
        # Setup empty list to track background futures natively
        # self._active_inference_futures = []

        if self.is_cuda and not hasattr(self, "sf_start") and self.config.TEST_MODE:
            self.sf_start, self.sf_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.roi_start, self.roi_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.det_start, self.det_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )

        if self.config.TEST_MODE:
            self.component_stats = {
                "sf": [],
                "roi": [],
                "det": [],
                "dma_upload": [],  # Track PCIe Transfer Latencies
                "queue_blocked": [],  # Track GIL Serialization stalls
                "batch_sizes": [],  # Track Smart Filtering density
                "thread_backlog": [],  # Track thread work pool backlog
            }
            self.crops_per_frame_list = []

        # Bind active execution permanently to sub-stream BEFORE entering loop (Bypasses TLS Overhead)
        # if self.device_input == "cuda" and torch.cuda.is_available():
        #     torch.cuda.set_stream(self.inference_stream)
        if hasattr(self, "target_fps") and hasattr(self, "duration_target"):
            self.max_target_frames = int(self.duration_target * float(self.target_fps))
        elif hasattr(self, "numFrames"):
            self.max_target_frames = self.numFrames
        else:
            self.max_target_frames = float("inf")

        # self.dynamic_limit = max(2, int(0.5 * self.target_fps))

        # Initialize a thread-safe signaling handle at class setup (run_realtime_inference)
        self.queue_data_ready_event = threading.Event()

        try:
            for _ in range(num_prefetch_workers):
                # prefetch_thread = mp.Process(
                prefetch_thread = threading.Thread(
                    target=lambda: global_frame_prefetch_worker_v1(self),
                    daemon=True,
                )
                prefetch_thread.start()
                self.prefetch_threads.append(prefetch_thread)
        except Exception as e:
            main_app_logger.critical(
                f"[{self.name}] Failed to start prefetch workers: {e}", exc_info=True
            )
            traceback.print_exc()
            raise

    @torch.inference_mode()
    def run_realtime_inference(
        self,
        sf_enabled=True,
        profiler=None,
        gt_enabled=False,
        read_frame_only=False,
    ):
        self.status = "RUNNING"

        self.initialize_run_realtime_inference(
            read_frame_only, self.active_workers_count
        )

        self.stat_start_time = time.perf_counter()
        # pipeline_start_time = self.stat_start_time
        # last_loop_cycle_timestamp = self.stat_start_time

        missing_frame_cnt = 0
        max_retries = int(self.target_fps)
        # Release the master startup event at the last possible moment.
        self.main_startup_event.set()
        while (
            self.active  # or not self.prefetch_queue.empty()
        ):  # and self.frame_count_target < self.max_target_frames:
            stat_start_time = time.perf_counter()
            try:
                # FRAME RETRIEVAL ---------------------------------------------
                try:
                    safe_frame, frame_details, should_continue = (
                        self._get_frame_from_queue()
                    )

                    if not should_continue:
                        self.active = False  # Signal loop termination
                        break
                    if safe_frame is None:  # and self.is_rtsp:
                        missing_frame_cnt += 1

                        if missing_frame_cnt >= max_retries:
                            self.active = False  # Signal loop termination
                            main_app_logger.info("Too many frames missing. Exiting ...")
                            break
                        continue  # Skip to the next iteration if the frame is invalid

                    # try:
                    #     # ret, slot_idx = self.prefetch_queue.get(block=True, timeout=1.0)
                    #     # ret, slot_idx = self.prefetch_queue.get(block=False)
                    #     ret, slot_idx = self.prefetch_queue.get(block=True, timeout=1.0)
                    #     self.prefetch_queue.task_done()

                    # except queue.Empty:
                    #     with self.worker_tracking_lock:
                    #         if (
                    #             self.active_workers_count == 0
                    #             and self.prefetch_queue.empty()
                    #         ):
                    #             self.active = False
                    #             break
                    #     # If the queue is empty AND the prefetch workers are done, we can exit.
                    #     if not self.active:
                    #         break
                    #     continue

                    # # If self.stop() drops the queues, break out of the thread loop natively.
                    # except (OSError, ValueError, AssertionError):
                    #     main_app_logger.info(
                    #         "[PROCESS THREAD] Ingestion queue disconnected via stop() signal. Breaking loop."
                    #     )
                    #     self.active = False
                    #     break

                    # if ret is False or slot_idx == "END_OF_STREAM":
                    #     main_app_logger.info(
                    #         "[PROCESS THREAD] End of video stream detected. Breaking loop naturally."
                    #     )
                    #     self.active = False
                    #     break

                    # if not self.active:
                    #     break

                    # # if slot_idx == -1 or self.shm_buffer_pool[slot_idx] is None:
                    # #     continue
                    # if (
                    #     slot_idx == -1
                    #     or not hasattr(self, "shm_buffer_pool")
                    #     or self.shm_buffer_pool is None
                    # ):
                    #     continue
                    # if (
                    #     slot_idx >= len(self.shm_buffer_pool)
                    #     or self.shm_buffer_pool[slot_idx] is None
                    # ):
                    #     continue

                    # # Zero-Copy Reference Extraction straight out of the memory slot array
                    # (
                    #     raw_shm_frame,
                    #     current_event,
                    #     frame_num,
                    #     abs_frame_num,
                    #     true_read_latency_secs,
                    # ) = self.shm_buffer_pool[slot_idx]

                    # if frame_num == 0:
                    #     main_app_logger.info(
                    #         f"[VERIFY - CONSUMER] Main loop is officially processing Frame {frame_num}!",
                    #     )

                    # if "cuda" in str(self.device_input):
                    #     safe_frame = self.gpu_input[slot_idx]
                    #     # safe_frame = cpu_tensor.to(self.device_input, non_blocking=True)
                    #     safe_frame.copy_(raw_shm_frame)
                    #     # self.gpu_input[slot_idx].zero_()
                    # else:
                    #     safe_frame = raw_shm_frame

                    # # Record the PyTorch CUDA event on the current stream
                    # if current_event is not None and isinstance(
                    #     current_event, torch.cuda.Event
                    # ):
                    #     # This tells the GPU that the consumer is done reading this slot
                    #     current_event.record(torch.cuda.current_stream())

                    # # Originally 0-based but make 1-based
                    # frame_num += 1
                    # abs_frame_num += 1
                    # self.abs_frame_num = abs_frame_num
                    missing_frame_cnt = 0

                except queue.Empty:
                    if getattr(self.reader, "reconnect_failed", False):
                        self.active = False
                        break
                    if not self.active:
                        break  # Exit on error if shutdown has been initiated
                    time.sleep(0.001)
                    continue

                # FRAME PROCESSING ---------------------------------------------
                # stat_start_time = self.stat_start_time
                self.abs_frame_num = frame_details["abs_frame_num"]
                calculated_clip_id = (
                    frame_details["frame_num"] - 1
                ) // self.max_frames_per_clip
                self.frame_count += 1
                self.frame_count_target += 1
                self.frame_in_clip_count += 1

                # RUN PIPELINE_FN ---------------------------------------------
                # run_pipelinefn_start = time.perf_counter()
                # self.frame_count_target += 1
                # self.frame_in_clip_count += 1

                # if gt_enabled:
                #     target_boxes_array = self.get_frame_gt_boxes(
                #         abs_frame_num, gt_sequence, gt_boxes
                #     )

                metadata_or_bbs, metrics = self.pipeline_fn(
                    safe_frame,
                    frame_details["frame_num"],
                    stat_start_time,
                    calculated_clip_id,
                    read_frame_only=read_frame_only,
                )

                # # FRAME POST PROCESSING ---------------------------------------------

                # Explicitly delete frame variables to free their references
                frame_8k = None
                del frame_8k, safe_frame
                if "metadata_or_bbs" in locals():
                    del metadata_or_bbs

                if self.frame_count % 100 == 0:
                    gc.collect()

                # Force a microscopic micro-yield if executing on CPU.
                # This grants immediate execution priority back to the background cleaning thread,
                # allowing the garbage collector to evict data from RAM instantly!
                if self.device_input == "cpu":
                    time.sleep(
                        0
                    )  # .005)  # 1ms yield breaks core processor starvation lock

                # END -> frame processing
            except torch.cuda.OutOfMemoryError:
                main_app_logger.info("!" * 70)
                main_app_logger.info(
                    "[CRITICAL TEST CRASH] GPU MEMORY CEILING HIT INSIDE RUNNER LOOP!"
                )
                main_app_logger.info(
                    "Freezing allocation history registers and writing diagnostic log..."
                )
                main_app_logger.info("!" * 70)

                try:
                    snapshot_filename = (
                        f"/tmp/test_vram_leak_profile_pid{os.getpid()}.pickle"
                    )
                    torch.cuda.memory._dump_snapshot(snapshot_filename)
                    main_app_logger.info(
                        f"[PROFILER SUCCESSFUL] Snapshot profile written to: {snapshot_filename}"
                    )
                    main_app_logger.info(
                        "--> Drag and drop this file directly into: https://pytorch.org"
                    )
                except Exception as dump_err:
                    main_app_logger.info(
                        f"Failed to record profile data snapshot: {dump_err}"
                    )

                # Force safe system unlinking of background workers to clean up OS handles
                self.active = False
                if hasattr(self, "reader") and self.reader is not None:
                    self.reader.stop()
                raise

            except Exception as e:
                if not self.active or not getattr(self, "prefetch_active", True):
                    main_app_logger.info(
                        "[PROCESS THREAD] System shutdown detected during exception sweep. Exiting thread payload context."
                    )
                    break

                main_app_logger.info(
                    f"[CRITICAL PIPELINE ERROR] Crash on frame: {repr(e)}"
                )
                traceback.print_exc()
                raise e  # Let it break so you can see the exact line number!

        # END -> while self.active and self.frame_count_target < self.max_target_frames:

        # PIPELINE POST PROCESSING ---------------------------------------------
        # )

        self.async_writer.release()

        # CALCULATE PERFORMANCE METRICS ---------------------------------------------
        main_app_logger.info(
            f"Execution Finished. Total Output Frames Written: {self.frame_count_target}"
        )

        # Force early hardware driver sweep before unbinding threads
        if self.device_input == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()

        if profiler is not None:
            profiler.stop()

        self.stop()

    def get_overlay(
        self,
        display_frame,
        metadata_or_bbs,
        class_list,
        scale_display_x,
        scale_display_y,
        color=(0, 0, 255),
    ):
        if isinstance(metadata_or_bbs, dict):
            detached_bbs = {
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in metadata_or_bbs.items()
            }
            # Object Mode (YOLO Structs)
            display_frame = get_metadata_overlay(
                display_frame,
                detached_bbs,
                class_list,
                (scale_display_x, scale_display_y),
                (self.disp_w, self.disp_h),
                is_bgr=True,
            )

        elif metadata_or_bbs is not None:
            # # Motion / Smart Filtering Overlay Path
            display_frame = get_bb_overlay(
                display_frame,
                metadata_or_bbs,
                (scale_display_x, scale_display_y),
                (self.disp_w, self.disp_h),
                color=color,
            )

        return display_frame

    def _get_frame_from_queue(self):
        """
        Retrieves the next available frame data from the shared memory prefetch queue.

        This helper encapsulates the logic for handling an empty queue, stream termination
        signals, and invalid data slots, making the main processing loop cleaner.

        Returns:
            A tuple containing:
            - (torch.Tensor or None): The frame tensor if successful, otherwise None.
            - (dict or None): A dictionary with frame metadata (frame_num, event), or None.
            - (bool): A flag indicating if the processing loop should continue.
        """
        try:
            # Block for a short timeout to wait for a frame
            ret, slot_idx = self.prefetch_queue.get(block=True, timeout=1.0)
            self.prefetch_queue.task_done()
        except queue.Empty:
            # If the queue is empty, check if workers are still active before deciding to exit.
            with self.worker_tracking_lock:
                if self.active_workers_count == 0 and self.prefetch_queue.empty():
                    main_app_logger.info(
                        "[HELPER] Prefetch queue is empty and all workers are done. Signaling stop."
                    )
                    return None, None, False  # Stop
            return None, None, True  # Continue waiting
        except (OSError, ValueError):
            # The queue was likely closed by self.stop(), signal to exit.
            main_app_logger.info("[HELPER] Prefetch queue was closed. Signaling stop.")
            return None, None, False  # Stop

        # Check for end-of-stream signal or invalid data
        if not ret or slot_idx == "END_OF_STREAM":
            main_app_logger.info("[HELPER] Received end-of-stream signal.")
            return None, None, False  # Stop

        if (
            slot_idx == -1
            or not hasattr(self, "shm_buffer_pool")
            or self.shm_buffer_pool is None
            or slot_idx >= len(self.shm_buffer_pool)
            or self.shm_buffer_pool[slot_idx] is None
        ):
            return None, None, True  # Continue, skip this invalid slot

        # --- Zero-Copy Reference Extraction ---
        (
            raw_shm_frame,
            current_event,
            frame_num,
            abs_frame_num,
            true_read_latency_secs,
        ) = self.shm_buffer_pool[slot_idx]
        # self.shm_buffer_pool[slot_idx] = None # Immediately clear the slot

        if frame_num == 0:
            main_app_logger.info(
                f"[VERIFY - CONSUMER] Main loop is officially processing Frame {frame_num}!",
            )

        # Convert numpy array from shared memory to a torch tensor
        # cpu_tensor = torch.from_numpy(raw_shm_frame) if isinstance(raw_shm_frame, np.ndarray) else raw_shm_frame

        # # Move to the correct device (GPU or CPU)
        # if "cuda" in str(self.device_input):
        #     safe_frame = cpu_tensor.to(self.device_input, non_blocking=True)
        # else:
        #     safe_frame = cpu_tensor

        # if "cuda" in str(self.device_input):
        #     safe_frame = self.gpu_input[slot_idx]
        #     # safe_frame = cpu_tensor.to(self.device_input, non_blocking=True)
        #     safe_frame.copy_(raw_shm_frame)
        #     # self.gpu_input[slot_idx].zero_()
        # else:
        safe_frame = raw_shm_frame

        # Record the CUDA event to signal that the consumer is done with this slot
        if current_event and isinstance(current_event, torch.cuda.Event):
            current_event.record(torch.cuda.current_stream())

        frame_details = {
            "frame_num": frame_num + 1,
            "abs_frame_num": abs_frame_num + 1,
            "event": current_event,
            "reader_time": true_read_latency_secs,
        }

        # Explicitly delete intermediate tensors
        del raw_shm_frame

        return safe_frame, frame_details, True

    def _setup_processor(self, read_frame_only):
        """
        Selects and initializes the appropriate object detector based on the configuration.

        This helper encapsulates the logic for creating the AI model (`processor`),
        handling potential initialization errors gracefully.

        Args:
            read_frame_only (bool): If True, no processor is initialized as frames are only being read.

        Returns:
            An initialized detector instance (e.g., SmartFilteringObjectDetector) or None.
        """
        # If we are only reading frames, no AI model is needed.
        # if read_frame_only:
        #     return None

        # Determine the correct detector class based on the smart filtering configuration.
        DetectorClass = (
            SmartFilteringObjectDetector
            if self.config.sf_enabled
            else GeneralObjectDetector
        )
        detector_name = DetectorClass.__name__
        main_app_logger.info(f"[{self.name}] Initializing processor: {detector_name}")

        try:
            # Common configuration for all detectors.
            debug_frame_limit = (
                self.config.DEBUG_FRAME_LIMIT if self.config.DEBUG_FLAG else -1
            )

            processor = DetectorClass(
                config=self.config,
                device=self.device_input,
                timer_enabled=True,
                resize_hw=(self.resize_h, self.resize_w),
                frame_hw=(self.frame_height, self.frame_width),
                target_fps=self.target_fps,
                result_dir=self.result_dir,
                run_name=self._testMethodName,
                debug_frame_limit=debug_frame_limit,
            )
            return processor
        except Exception as e:
            # If the model fails to load, log the error and halt the pipeline.
            main_app_logger.critical(
                f"[{self.name}] CRITICAL: Failed to initialize AI processor '{detector_name}'. Error: {e}",
                exc_info=True,
            )
            # Propagate the error to stop the handler from starting.
            raise

    def frame2video(
        self,
        device_frame,
        frameNum,
        metadata_or_bbs,
        class_list,
        stat_start_time,
        gt_boxes=None,
    ):
        """
        Frame is formated and added to video
        metadata_or_bbs: Expected to be in resize dimensions (640x640)
        """

        # Factor for converting 640 bb dimensions to display dimensions
        scale_display_x = self.disp_w / self.resize_w  # 640
        scale_display_y = self.disp_h / self.resize_h  # 640

        if self.device_input == "cuda":  # and torch.is_tensor(device_frame):
            with torch.inference_mode():
                # 2. Fast Inline VRAM Downscaling into our static memory slot
                # Reshape to (Batch, Channel, Height, Width) seamlessly without duplicating data
                gpu_tensor = device_frame[None].permute(0, 3, 1, 2)

                resized_tensor = torch.nn.functional.interpolate(
                    gpu_tensor,
                    size=(self.disp_h, self.disp_w),
                    mode="nearest",
                )

            # Using copy_() avoids creating any runtime allocations or memory-stride drift.
            self.static_gpu_byte_bchw.copy_(resized_tensor)

            # Safely reshape the pre-allocated, locked GPU memory layout back to standard HWC array topology.
            # Since the underlying array layout is strictly contiguous, this permute is guaranteed
            # to be zero-copy on the GPU and free from memory race stalls.
            bgr_contiguous = self.static_gpu_byte_bchw.squeeze(0).permute(1, 2, 0)

            # 1. Grab the current free pinned slot tracking variables from your reader
            d2h_idx = self.reader.d2h_selector
            pinned_tensor_buf = self.reader.d2h_buffers[d2h_idx]

            # # Synchronize only the side download stream layout channel
            # self.reader.download_stream.synchronize()
            # Initialize a static download sync event on setup_context if not present
            # if not hasattr(self, "_d2h_fence"):
            #     self._d2h_fence = torch.cuda.Event()

            with torch.cuda.stream(self.reader.download_stream):
                pinned_tensor_buf.copy_(
                    bgr_contiguous,
                    non_blocking=True,
                )
            #     self._d2h_fence.record()

            # self._d2h_fence.synchronize()

            cpu_360p_frame = self.reader.d2h_numpys[d2h_idx]

            # 5. Non-Blocking Push straight to your background AsyncVideoWriter thread pool
            # We pass a direct .copy() slice so the hot loop can instantly reuse the pinned ring buffer
            # display_frame = np.array(cpu_360p_frame, copy=True, order="C")
            display_frame = np.asarray(cpu_360p_frame, order="C")
            if display_frame.shape[-1] == 3:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

            # --- Draw Detection Overlays ---
            if gt_boxes is not None:
                # Factor for converting 8K (original) bb dimensions to display dimensions
                scale_display_ox = self.disp_w / self.frame_width
                scale_display_oy = self.disp_h / self.frame_height

                display_frame = self.get_overlay(
                    display_frame,
                    gt_boxes,
                    None,
                    scale_display_ox,
                    scale_display_oy,
                    color=(0, 255, 0),
                )

            display_frame = self.get_overlay(
                display_frame,
                metadata_or_bbs,
                class_list,
                scale_display_x,
                scale_display_y,
            )

            self.async_writer.write_frame(display_frame)
            self.update_frame(stat_start_time)
            # self.canvas_selector = 1 - self.canvas_selector
            self.reader.d2h_selector = 1 - self.reader.d2h_selector

            if (
                self.config.DEBUG_FLAG
                and self.config.TEST_MODE
                and frameNum <= self.config.DEBUG_FRAME_LIMIT
            ):
                stage_debug_dir = (
                    self.result_dir / "debug_stages" / self._testMethodName / "display"
                )
                stage_debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    # str(stage_debug_dir / f"frame_{f_num:04d}_stage6_threshold.jpg"),
                    str(stage_debug_dir / f"frame_{frameNum:04d}_final_frame.jpg"),
                    display_frame,
                )

        else:  # CPU
            # Reusable baseline track for CPU execution mappings
            display_frame = cv2.resize(
                device_frame,
                (self.disp_w, self.disp_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # --- Draw Detection Overlays ---
            if gt_boxes is not None:
                # Factor for converting 8K (original) bb dimensions to display dimensions
                scale_display_ox = self.disp_w / self.frame_width
                scale_display_oy = self.disp_h / self.frame_height

                display_frame = self.get_overlay(
                    display_frame,
                    gt_boxes,
                    None,
                    scale_display_ox,
                    scale_display_oy,
                    color=(0, 255, 0),
                )

            display_frame = self.get_overlay(
                display_frame,
                metadata_or_bbs,
                class_list,
                scale_display_x,
                scale_display_y,
            )
            self.async_writer.write_frame(display_frame)
            self.update_frame(stat_start_time)

        # if "display_frame" in locals():
        #     del display_frame
        # if "metadata_or_bbs" in locals():
        #     del metadata_or_bbs

    def frame2output(
        self, device_frame, frame_num, metadata_or_bbs, class_list, stat_start_time
    ):
        """
        Renders drone detection bounding boxes onto the canvas
        before dispatching to the live UI shared memory stream.
        """
        if not self.active:
            return

        # Factor for converting 640 bb dimensions to display dimensions
        scale_display_x = self.disp_w / self.resize_w  # 640
        scale_display_y = self.disp_h / self.resize_h  # 640

        try:
            if self.device_input == "cuda":  # and torch.is_tensor(device_frame):
                with torch.inference_mode():
                    # main_app_logger.info(f"device_frame.shape: {device_frame.shape}")
                    # main_app_logger.info(f"\tdevice_frame[None, :].shape: {device_frame[None, :].shape}")
                    # 2. Fast Inline VRAM Downscaling into our static memory slot
                    # Reshape to (Batch, Channel, Height, Width) seamlessly without duplicating data
                    gpu_tensor = device_frame[None, :].permute(0, 3, 1, 2).float()

                    resized_tensor = torch.nn.functional.interpolate(
                        gpu_tensor,
                        size=(self.disp_h, self.disp_w),
                        mode="nearest",
                    )

                # CONVERTS TO CPU EARLIER (WORKS but lowers fps)
                # hwc_byte_tensor = resized_tensor.squeeze(0).permute(1, 2, 0).byte()
                # cpu_tensor = hwc_byte_tensor.cpu()
                # bgr_contiguous = cpu_tensor.contiguous()

                # Using copy_() avoids creating any runtime allocations or memory-stride drift.
                self.static_gpu_byte_bchw.copy_(resized_tensor.byte())

                # Safely reshape the pre-allocated, locked GPU memory layout back to standard HWC array topology.
                # Since the underlying array layout is strictly contiguous, this permute is guaranteed
                # to be zero-copy on the GPU and free from memory race stalls.
                bgr_contiguous = self.static_gpu_byte_bchw.squeeze(0).permute(1, 2, 0)

                # 1. Grab the current free pinned slot tracking variables from your reader
                d2h_idx = self.reader.d2h_selector
                pinned_tensor_buf = self.reader.d2h_buffers[d2h_idx]

                # c_idx = self.canvas_selector
                # current_canvas = self.static_host_canvases[c_idx]

                # host_tensor_view = torch.as_tensor(current_canvas, device="cpu")

                # 3. Non-blocking asynchronous PCIe DMA Download directly into our page-locked canvas
                # with torch.cuda.stream(self.reader.download_stream):
                # if (
                #     hasattr(self.reader, "download_stream")
                #     and self.reader.download_stream is not None
                # ):
                #     torch.cuda.set_stream(self.reader.download_stream)
                # pinned_tensor = torch.from_numpy(current_canvas).cuda()
                pinned_tensor_buf.copy_(bgr_contiguous, non_blocking=True)

                # Synchronize only the side download stream layout channel
                # self.reader.download_stream.synchronize()

                cpu_360p_frame = self.reader.d2h_numpys[d2h_idx]

                # 5. Non-Blocking Push straight to your background AsyncVideoWriter thread pool
                # We pass a direct .copy() slice so the hot loop can instantly reuse the pinned ring buffer
                display_frame = np.array(cpu_360p_frame, copy=True, order="C")
                if display_frame is not None and display_frame.shape[-1] == 3:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                # --- Draw Detection Overlays ---
                display_frame = self.get_overlay(
                    display_frame,
                    metadata_or_bbs,
                    class_list,
                    scale_display_x,
                    scale_display_y,
                )

                self.async_writer.write_frame(display_frame, frame_num)
                self.update_frame(stat_start_time)
                # self.canvas_selector = 1 - self.canvas_selector
                self.reader.d2h_selector = 1 - self.reader.d2h_selector

            else:
                # Reusable baseline track for CPU execution mappings
                display_frame = cv2.resize(
                    device_frame,
                    (self.disp_w, self.disp_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                # --- Draw Detection Overlays ---
                display_frame = self.get_overlay(
                    display_frame,
                    metadata_or_bbs,
                    class_list,
                    scale_display_x,
                    scale_display_y,
                )
                self.async_writer.write_frame(display_frame, frame_num)
                self.update_frame(stat_start_time)

        except Exception:
            traceback.print_exc()
        # return

        if "display_frame" in locals():
            del display_frame
        if "metadata_or_bbs" in locals():
            del metadata_or_bbs

    # CLEANUP --------------------------------------------

    def clean_up_tensors_and_arrays_v1(self):
        main_app_logger.info("[CLEANUP] Safely stripping active runtime arrays ...")

        # 1. Force disable tracking states to stop graph allocation recursion
        # torch.set_grad_enabled(False)

        # if hasattr(torch, "jit") and hasattr(torch.jit, "_builtins"):
        #     if isinstance(torch.jit._builtins, dict):
        #         torch.jit._sbuiltins.clear()
        #     else:
        #         torch.jit._builtins = {}

        # 2. Collect references safely using strict type checking
        # This completely avoids pulling unmanaged proxy objects from the heap
        all_live_objects = gc.get_objects()

        target_tensors = []
        target_arrays = []

        for obj in all_live_objects:
            try:
                obj_type = type(obj)
                if isinstance(obj, types.FrameType):
                    frame_info = inspect.getframeinfo(obj)
                    if (
                        "openvino" in frame_info.filename
                        or "openvino.py" in frame_info.filename
                    ):
                        # Clear the frame's local variable dictionary to break circular references
                        obj.f_locals.clear()

                # Check for concrete types to prevent triggering proxy __getattr__ hooks
                if obj_type is torch.Tensor:
                    target_tensors.append(obj)
                elif obj_type is np.ndarray:
                    # Explicitly guard size checks to keep it completely stable
                    if obj.base is None and obj.ndim > 0:
                        target_arrays.append(obj)
            except Exception:
                traceback.print_exc()

        # 3. Truncate discovered references in-place without triggering deletions
        reclaimed_tensors = 0
        for tensor in target_tensors:
            try:
                # Truncate raw storage footprint safely
                tensor.data = torch.empty(0, device=self.device_input)
                reclaimed_tensors += 1
            except Exception:
                traceback.print_exc()

        reclaimed_arrays = 0
        for arr in target_arrays:
            try:
                # Shrink writeable numpy arrays down to 0 bytes safely
                if arr.flags.writeable:
                    arr.resize((0,), refcheck=False)
                    reclaimed_arrays += 1
            except (ValueError, SystemError):
                traceback.print_exc()
            except Exception:
                traceback.print_exc()

        main_app_logger.info(
            f"[CLEANUP] Reclaimed {reclaimed_tensors} tensors and {reclaimed_arrays} arrays safely."
        )

        # Clean local registers immediately
        all_live_objects = None
        target_tensors = None
        target_arrays = None

        if hasattr(sys, "exc_info"):
            sys.exc_clear() if hasattr(sys, "exc_clear") else None
        gc.collect()

    def clean_up_tensors_and_arrays(self):
        """
        Safely releases instance-local memory without corrupting global PyTorch tensors.
        """
        main_app_logger.info(
            f"[CLEANUP {getattr(self, 'name', '')}] Releasing local arrays and flushing caches..."
        )

        # Explicitly release any instance-level tensor references
        instance_tensor_attrs = [
            "static_gpu_360p",
            "static_gpu_byte_bchw",
            "gpu_float_staging",
            "scales_tensor",
            "_cached_grid_x",
            "_cached_grid_y",
        ]
        for attr in instance_tensor_attrs:
            if hasattr(self, attr):
                setattr(self, attr, None)

        if hasattr(sys, "exc_info"):
            try:
                if hasattr(sys, "exc_clear"):
                    sys.exc_clear()
            except Exception:
                pass

        gc.collect()
        if "cuda" in str(self.device_input) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def drain_queues(self, all_queues, wait_time=15.0):
        # Wait (15 seconds) for your background writers to pull remaining matrices out of the pipe
        drain_start = time.perf_counter()
        while time.perf_counter() - drain_start < wait_time:
            still_has_frames = False
            for q_name in all_queues:
                q_val = getattr(self, q_name, None)
                if q_val is not None:
                    try:
                        # Accumulate total remaining frame backlogs across your pipelines
                        # if hasattr(q_val, "qsize"):
                        #     backlog += q_val.qsize()
                        if not q_val.empty():
                            still_has_frames = True
                            break
                    except Exception:
                        pass

            if not still_has_frames:
                main_app_logger.info(
                    "[TEARDOWN] All background queues are empty. Proceeding to safe close.",
                )
                break

            # Yield micro-slices to grant immediate execution priority to background threads
            time.sleep(0.01)

    def remove_scalar_attributes(self, remove_attrs):
        for attr in remove_attrs:
            val = getattr(self, attr, None)
            if val is not None:
                # try:
                #     val.clear()
                # except Exception:
                #     setattr(self, attr, None)

                # # try: delattr(self, attr)
                # # except AttributeError: pass
                try:
                    # THE NATIVE SHIELD: Only call clear() if it has the attribute!
                    if hasattr(val, "clear") and callable(getattr(val, "clear")):
                        val.clear()
                    else:
                        # If it's a primitive type like an int, reset it safely
                        if isinstance(val, int):
                            setattr(self, attr, 0)
                        elif isinstance(val, float):
                            setattr(self, attr, 0.0)
                        elif isinstance(val, dict):
                            setattr(self, attr, {})
                        elif torch.is_tensor(val):
                            val.data = torch.empty(0, device=self.device_input)
                        else:
                            setattr(self, attr, None)
                    if hasattr(self, attr):
                        delattr(self, attr)
                except Exception:
                    # pass
                    traceback.print_exc()

                try:
                    if hasattr(self, attr):
                        delattr(self, attr)
                except Exception:
                    # pass
                    traceback.print_exc()

    def drain_and_close_queues(self, all_queues):
        for key in all_queues:
            val = getattr(self, key, None)
            if val is None:
                continue
            try:
                while not val.empty():
                    val.get_nowait()
            except Exception:
                pass
            try:
                if hasattr(val, "close"):
                    if hasattr(val, "cancel_join_thread"):
                        val.cancel_join_thread()
                    val.close()
            except Exception:
                pass
            # setattr(self, key, None)

    def stop_events(self, events=[], keys_to_skip_deletion=[]):
        for event_attr in events:
            # if hasattr(self, event_attr):
            tgt_event = getattr(self, event_attr, None)
            if tgt_event is None:
                continue
            try:
                if hasattr(tgt_event, "_handle"):
                    tgt_event._handle.close()
                if (
                    isinstance(tgt_event, torch.cuda.Event)
                    and event_attr not in keys_to_skip_deletion
                ):
                    # Force internal driver handle release
                    del tgt_event
            except Exception:
                # pass
                traceback.print_exc()
            setattr(self, event_attr, None)

            if hasattr(self, event_attr) and event_attr not in keys_to_skip_deletion:
                delattr(self, event_attr)

    def stop_thread(self, val):
        # current_thread_id = threading.get_ident()
        # val = getattr(self, targeted_key, None)
        # val = targeted_key_val
        try:
            if hasattr(val, "is_alive") and val.is_alive():
                if val.ident != threading.get_ident():
                    if hasattr(val, "terminate"):
                        val.terminate()
                    val.join(timeout=0.5)
        except Exception:
            if val is None:
                pass
            traceback.print_exc()

    def stop_threads(self, all_targeted_keys):
        current_thread_id = threading.get_ident()
        for attr in all_targeted_keys:
            val = getattr(self, attr, None)
            if val is None:
                continue
            try:
                if hasattr(val, "is_alive") and val.is_alive():
                    if getattr(val, "ident", None) != current_thread_id:
                        # if hasattr(val, "terminate"): val.terminate()
                        # val.join(timeout=0.5)
                        if isinstance(val, (threading.Thread, DummyProcess)):
                            # Threads can only be joined cooperatively via flags (self.active = False)
                            val.join(timeout=0.2)  # Use a tight, rapid timeout gate
                        else:
                            val.join(timeout=3.0)
                            # OS multi-processing blocks can safely accept termination hooks
                            if val.is_alive() and hasattr(val, "terminate"):
                                val.terminate()
            except Exception:
                # pass
                traceback.print_exc()

            # Clean up references immediately to drop tracking counters
            if getattr(val, "ident", None) != current_thread_id:
                setattr(self, attr, None)

    def set_stop_writer_event(self, remove=False):
        if hasattr(self, "stop_writer") and self.stop_writer is not None:
            try:
                self.stop_writer.set()
            except Exception:
                pass
            setattr(self, "stop_writer", None)

            if remove:
                try:
                    delattr(self, "stop_writer")
                except AttributeError:
                    pass

    def stop_executors(self, all_targeted_keys, remove=True):
        for key in list(all_targeted_keys):
            val = getattr(self, key, None)
            if val is None:
                continue
            try:
                val.shutdown(wait=False, cancel_futures=True)
                # Natively strip out the inner worker thread arrays to drop OS handles
                if hasattr(val, "_threads"):
                    val._threads.clear()
            except Exception:
                try:
                    val.shutdown(wait=False)
                except Exception:
                    pass
            setattr(self, key, None)
            if remove:
                delattr(self, key)

    def unregister_pinned_cuda_data_v1(self):
        # Unregister the hidden page-locks inside the ai_pinned_tensors pool
        if hasattr(self, "ai_pinned_tensors") and self.ai_pinned_tensors:
            for tensor in list(self.ai_pinned_tensors):
                try:
                    if torch.is_tensor(tensor):
                        # Extract the shared NumPy view to break the C++ driver lock
                        cv2.cuda.unregisterPageLocked(tensor.numpy())
                except cv2.error:
                    pass
                except Exception:
                    pass

        # Unregister the standard pinned_tensors pool page-locks
        if hasattr(self, "pinned_tensors") and self.pinned_tensors:
            for tensor in list(self.pinned_tensors):
                try:
                    if torch.is_tensor(tensor):
                        cv2.cuda.unregisterPageLocked(tensor.numpy())
                except cv2.error:
                    pass
                except Exception:
                    pass

        # Unregister any standalone page-locked matrices
        if hasattr(self, "pinned_matrices") and self.pinned_matrices:
            for active_mat in list(self.pinned_matrices):
                if getattr(self, "device_input", "cpu") == "cuda":
                    try:
                        # Wrap explicitly to catch the native OpenCV -217 API Exception!
                        cv2.cuda.unregisterPageLocked(active_mat)
                    except cv2.error:
                        # Catch the pointer registry mismatch safely without breaking the stop() execution stream
                        pass
                    except Exception:
                        pass

        # FORCE PYTORCH C++ ALLOCATOR TO DISSOLVE THE HARDWARE BOUNDARIES
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def unregister_pinned_cuda_data(self):
        """
        Safely unregisters OpenCV page-locked host memory from the CUDA driver
        before the underlying arrays are released.
        """
        if getattr(self, "device_input", "cpu") != "cuda":
            return

        if hasattr(self, "pinned_matrices") and self.pinned_matrices:
            for active_mat in list(self.pinned_matrices):
                if active_mat is not None:
                    try:
                        cv2.cuda.unregisterPageLocked(active_mat)
                    except Exception:
                        pass

        if hasattr(self, "pinned_tensors") and self.pinned_tensors:
            for tensor in list(self.pinned_tensors):
                if tensor is not None and torch.is_tensor(tensor):
                    try:
                        cv2.cuda.unregisterPageLocked(tensor.numpy())
                    except Exception:
                        pass

        # Force CUDA driver to finalize unpinning
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def clear_buffer_pools(self, buffer_pools):
        for buffer_pool in buffer_pools:
            val = getattr(self, buffer_pool, None)
            if val is None:
                continue

            setattr(self, buffer_pool, None)
            for i in range(len(val)):
                val[i] = None
            if hasattr(val, "clear"):
                try:
                    val.clear()
                except Exception:
                    pass

    def clear_pinned_data(self):
        tensor_pools = ["ai_pinned_tensors", "pinned_tensors"]
        for pool_attr in tensor_pools:
            pool = getattr(self, pool_attr, None)
            if pool:
                for tensor in list(pool):
                    try:
                        if torch.is_tensor(tensor):
                            # Truncate internal storage mapping layer immediately
                            tensor.data = torch.empty(0, device="cpu")
                    except Exception:
                        pass
                try:
                    pool.clear()
                except Exception:
                    pass

        # 7. UNPIN HARDWARE MATRICES (From initialize_variables)
        if hasattr(self, "pinned_matrices") and self.pinned_matrices:
            try:
                self.pinned_matrices.clear()
            except Exception:
                pass

    def clear_shared_memory_list(self, shm_names: list):
        """
        Cleans up and unlinks shared memory collections, supporting both
        direct attributes ("shms") and nested attributes ("reader.shms").
        """
        for shm_path in shm_names:
            curr_obj = self
            for attr in shm_path.split("."):
                curr_obj = getattr(curr_obj, attr, None)
                if curr_obj is None:
                    break

            val = curr_obj
            if not val:
                continue

            release_shared_memory(list(val))
            # for shm in list(val):
            #     if shm is not None:
            #         try:
            #             # Release buffer view if open
            #             if hasattr(shm, "buf") and shm.buf is not None:
            #                 shm.buf.release()
            #         except Exception:
            #             pass

            #         try:
            #             shm.close()
            #         except Exception:
            #             pass

            #         try:
            #             shm.unlink()
            #         except (FileNotFoundError, AttributeError, OSError):
            #             pass

            # # Clear the container list/collection in-place
            # if hasattr(val, "clear"):
            #     try:
            #         val.clear()
            #     except Exception:
            #         pass

    def clean_up_reader(self):
        """Halts the background prefetch staging worker completely before draining pipeline queues."""
        if hasattr(self, "reader") and self.reader is not None:
            try:
                # 1. IMMEDIATE HALT: Force the background loop condition to fail instantly
                self.reader.stopped = True
                self.prefetch_active = False
                self.active = False

                # 2. Join the prefetch thread gracefully so it finishes its current read and dies
                if hasattr(self, "prefetch_threads") and self.prefetch_threads:
                    for th in self.prefetch_threads:
                        if th.is_alive():
                            th.join(timeout=0.2)
                    self.prefetch_threads.clear()

                # if hasattr(self.reader, "print_breakdown"):
                #     self.reader.print_breakdown()

                # 3. Safe, deadlock-free queue drain loop (Now guaranteed to exit cleanly)
                drain_timeout_start = time.perf_counter()
                while not self.prefetch_queue.empty():
                    try:
                        self.prefetch_queue.get_nowait()
                        self.prefetch_queue.task_done()
                    except Exception:
                        break
                    # Hard escape hatch safety guard to prevent hanging if threads ghost
                    if time.perf_counter() - drain_timeout_start > 0.5:
                        break

                # 4. Trigger standard standalone hardware unpinning routines
                # if hasattr(self.reader, "release_hardware_pins"):
                #     self.reader.release_hardware_pins()

                self.reader.stop()

            except Exception as e:
                main_app_logger.debug(
                    f"Reader object isolation step encountered a soft error: {e}"
                )
            finally:
                self.reader = None

    def unlink_shared_memory(self, all_shm_keys):
        for key in all_shm_keys:
            val = getattr(self, key, None)
            if val is None:
                continue
            for shm in list(val):
                if shm is not None:
                    try:
                        # Drop direct memoryview trackers instantly
                        if hasattr(shm, "buf") and shm.buf is not None:
                            try:
                                shm.buf.release()
                            except Exception:
                                pass

                        # Invalidate private mmap structures to drop pytest stack frame caches
                        if hasattr(shm, "_mmap") and shm._mmap is not None:
                            try:
                                shm._mmap = None
                            except Exception:
                                pass

                        # try:
                        #     shm.__class__.__del__ = lambda self: None
                        # except Exception:
                        #     pass

                        # # Unlink text descriptors out of the core tracker process
                        # shm_descriptor = shm._name if shm._name.startswith("/") else f"/{shm._name}"
                        # try:
                        #     unregister(shm_descriptor, "shared_memory")
                        # except Exception:
                        #     pass

                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass
            try:
                val.clear()
            except Exception:
                pass

    def stop_sync_manager(self):
        """Safely shuts down the background Sync Base Manager process daemon."""
        # if hasattr(self, "shared_details") and self.shared_details is not None:
        #     try: self.shared_details.clear()
        #     except Exception: pass

        if hasattr(self, "manager") and self.manager is not None:
            try:
                if hasattr(self.manager, "_allocated"):
                    for memory_block in list(self.manager._allocated):
                        try:
                            if (
                                hasattr(memory_block, "buf")
                                and memory_block.buf is not None
                            ):
                                memory_block.buf.release()
                            memory_block.close()
                            memory_block.unlink()
                        except Exception:
                            pass
                    try:
                        self.manager._allocated.clear()
                    except Exception:
                        pass
                self.manager.shutdown()

                # Added
                if (
                    hasattr(self.manager, "_process")
                    and self.manager._process is not None
                ):
                    proc = self.manager._process
                    if proc.is_alive():
                        proc.join(timeout=0.2)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()

            except Exception:
                pass
            setattr(self, "manager", None)

    def purge_attributes(self):
        # ─── FORCE LOW-LEVEL OPENVINO C++ HARDWARE PURGE ───
        if hasattr(self, "model") and self.model is not None:
            try:
                # If your backend utilizes an OpenVINO core orchestrator:
                model_wrapper = self.model

                # Check for common OpenVINO runtime handle names inside your core class
                for attr_name in ["core", "_core", "ov_core", "runtime"]:
                    if hasattr(model_wrapper, attr_name):
                        ov_core = getattr(model_wrapper, attr_name)
                        if ov_core is not None:
                            # 1. Force OpenVINO to clear internal cached compiled models and models' graphs
                            if hasattr(ov_core, "get_property"):
                                try:
                                    # Tells OpenVINO to drop its physical memory caching metrics pools
                                    ov_core.set_property({}, {})
                                except Exception:
                                    pass

                            # 2. Trigger explicit Python C++ boundary deconstruction bindings
                            if hasattr(ov_core, "__del__"):
                                ov_core.__del__()

                            setattr(model_wrapper, attr_name, None)
            except Exception:
                pass

        # Un-bind dynamic instance methods to break Pytest's reference hold
        for dynamic_method in ["run_realtime_inference", "pipeline_fn"]:
            if hasattr(self, dynamic_method):
                try:
                    # Wiping out the bound method object instantly frees the frame loops
                    # setattr(self, dynamic_method, None)
                    delattr(self, dynamic_method)
                except Exception:
                    pass

        for cls_attr in ["d2h_buffers", "d2h_numpys", "d2h_selector"]:  # "pipeline_fn",
            if hasattr(self, cls_attr):
                try:
                    setattr(self, cls_attr, None)
                except AttributeError:
                    pass

        # Now it is structurally safe to purge attributes without causing cross-thread collisions
        keys_to_purge = [
            "__persistent_vram_lock",
            "_DeviceBaseHandler__persistent_vram_lock",  # Wipes out the private lock tensor safely
            "ai_gpu_staging",
            "ai_pinned_tensors",
            "ai_shm_names",  # Clear names list references
            "ai_shms",  # Clear handler specific memory arrays
            # "all_preds",
            # "all_targets",
            "bgs_stream",
            "config",  # Force config destruction
            "evaluator",
            "executor",
            "frame_buffer_pool",
            "gpu_buffer_pool",
            "gpu_display_frame",
            "gpu_encoder_8k_buf",
            "gpu_float_staging",
            "host_buffer_pool",
            "inference_stream",
            "model",
            "pinned_downloaded_frame_np",
            "pinned_downloaded_resizedframe_np",
            "pinned_matrices",  # Clear handler registration matrix list
            "pinned_tensors",  # Clear backend tensor mappings
            "process_thread",
            "raw_input",
            "reader",
            "scales_tensor",
            "shms",  # Clear handler specific memory arrays
            "static_gpu_360p",  # Added to secure structural bounds
            "static_gpu_byte_bchw",  # Added to secure structural bounds
            # "static_host_canvases",
            # "io_executor",
        ]
        for key in keys_to_purge:
            if hasattr(self, key):
                try:
                    delattr(self, key)
                except AttributeError:
                    pass

        gc.collect()

    # HELPER FUNCTIONS --------------------------------------------
    def get_executor_backlog(self):
        """Returns the number of tasks currently waiting in the thread pool queue."""
        # access the internal queue of the executor
        # return self.executor._work_queue.qsize()
        if getattr(self, "executor", None) is None:
            return 0
        try:
            # Handle standard ThreadPoolExecutor and custom queue wrappers cleanly
            if hasattr(self.executor, "_work_queue"):
                return self.executor._work_queue.qsize()
            return 0
        except Exception:
            return 0

    def get_clip_executor_backlog(self):
        """Returns the number of tasks currently waiting in the thread pool queue."""
        # access the internal queue of the clip_executor
        # return self.clip_executor._work_queue.qsize()
        if getattr(self, "clip_executor", None) is None:
            return 0
        try:
            if hasattr(self.clip_executor, "_work_queue"):
                return self.clip_executor._work_queue.qsize()
            return 0
        except Exception:
            return 0

    def check_disk_usage(self, path, min_gb=0.5):
        """Returns True if there is at least min_gb available at path."""
        try:
            total, used, free = shutil.disk_usage(path)
            # Convert bytes to Gigabytes
            free_gb = free / (2**30)
            return free_gb > min_gb
        except Exception as e:
            main_app_logger.info(f"[EXCEPTION] Disk check error: {e}")
            return False

    # Gets frame W and H details
    def get_frameWH(self):
        if (self.frame_height * self.frame_width) < (
            self.config.MODEL_H * self.config.MODEL_W
        ):
            new_sizeHW = check_imgsz(
                [self.config.MODEL_H, self.config.MODEL_W]
            )  # expects hxw
        else:
            new_sizeHW = check_imgsz(
                [self.frame_height, self.frame_width]
            )  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]
        self.height = new_sizeWH[1]

        # Configure scaling for 8K-to-Model coordinate mapping
        self.resize_h, self.resize_w = [self.config.MODEL_H, self.config.MODEL_W]
        self.scale_x = self.frame_width / self.config.MODEL_W
        self.scale_y = self.frame_height / self.config.MODEL_H

    def is_processing(self):
        """Returns True if any part of the pipeline is still active."""
        if not self.reader.stopped:
            return True

        if (
            self.process_thread is not None and self.process_thread.is_alive()
        ):  # or not self.reader.frame_queue.empty():
            return True

        if not self.reader.frame_queue.empty():
            return True

        q_size = self.write_queue.qsize()
        if self.process_thread is not None:
            main_app_logger.info(
                f"[STATUS] Writer: {self.write_count}/{self.frame_count_target} | Q: {q_size} | Inf: {'Alive' if self.process_thread.is_alive() else 'Dead'}",
                end="\r",
            )
        else:
            main_app_logger.info(
                f"[STATUS] Writer: {self.write_count}/{self.frame_count_target} | Q: {q_size}",
                end="\r",
            )

        if not self.write_queue.empty():
            return True
        if self.write_count < self.frame_count_target:
            # q_size = self.write_queue.qsize()
            # main_app_logger.info(f"[DRAIN] Writer: {self.write_count}/{self.frame_count} | Queue: {q_size}", end="\r")
            return True
        return False

    def _check_shm_safety(self, threshold_percent=90):
        """
        Scans /dev/shm and deletes the oldest .mp4 files if usage exceeds threshold.
        This prevents the 8K stream from crashing the entire container.
        """
        # Check current usage of the RAM disk
        usage = shutil.disk_usage("/dev/shm")
        percent_used = (usage.used / usage.total) * 100

        if percent_used > threshold_percent:
            main_app_logger.info(
                f" [CRITICAL] /dev/shm usage at {percent_used:.1f}%. Purging old clips..."
            )

            # Get all .mp4 files in /dev/shm sorted by oldest first
            shm_path = Path("/dev/shm")
            clips = sorted(shm_path.glob("*.mp4"), key=lambda x: x.stat().st_mtime)

            # Delete files until we are under 70% usage or run out of files
            for clip in clips:
                try:
                    # Don't delete the file the current writer is actively using!
                    # if str(clip) == self.tmp_file:
                    #     continue

                    clip.unlink()
                    main_app_logger.info(f"[PURGE] Deleted {clip.name} to free RAM.")

                    # Re-check usage after each deletion
                    usage = shutil.disk_usage("/dev/shm")
                    if (usage.used / usage.total) * 100 < 70:
                        break
                except Exception as e:
                    main_app_logger.info(f"[EXCEPTION] Could not purge {clip}: {e}")

    def print_active_gpu_tensor_memory(self):
        main_app_logger.info(
            "=" * 60,
        )
        main_app_logger.info(
            "\033[95m[VRAM INVESTIGATOR] Scanning Active GPU Tensors with Sources:\033[0m"
        )

        # Capture the exact C++ memory registry structure
        try:
            raw_snapshot = torch.cuda.memory._snapshot()
            segments = raw_snapshot.get("segments", [])
        except Exception:
            segments = []
            main_app_logger.info(
                "[WARN] Failed parsing native memory context snapshot.",
            )

        # Map raw block storage memory addresses straight to Python frames
        addr_to_source = {}
        for seg in segments:
            for block in seg.get("blocks", []):
                if block.get("state") == "active_allocated":
                    addr = block.get("address")
                    history = block.get("history", [])
                    if history:
                        # Inspect the deepest frame in the allocation stack
                        frame = history[-1]
                        filename = frame.get("filename", "Unknown")
                        lineno = frame.get("line", 0)
                        func_name = frame.get("name", "unknown_func")
                        addr_to_source[addr] = f"{filename}:{lineno} ({func_name})"

        # Scan the heap via GC and resolve actual backing storage layers
        leaked_tensors = []
        total_detected_bytes = 0

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    t_bytes = obj.element_size() * obj.nelement()
                    total_detected_bytes += t_bytes
                    if t_bytes > 0:
                        leaked_tensors.append(obj)
            except Exception:
                pass

        obj = None  # Break tracking register references

        for i, tensor in enumerate(leaked_tensors):
            t_bytes = tensor.element_size() * tensor.nelement()

            # --- THE FIX: Extract the address of the underlying storage block ---
            try:
                if hasattr(tensor, "untyped_storage"):
                    storage_addr = tensor.untyped_storage().data_ptr()
                elif hasattr(tensor, "storage") and tensor.storage():
                    storage_addr = tensor.storage().data_ptr()
                else:
                    storage_addr = tensor.data_ptr()
            except Exception:
                storage_addr = tensor.data_ptr()

            # Extract absolute code trace locations from our snapshot dictionary
            source_loc = addr_to_source.get(
                storage_addr, "Unknown Native C++ Allocation / Model Context"
            )

            main_app_logger.info(
                f" > Tensor {i:3d} | Shape: {str(list(tensor.shape)):<18} | "
                f"Size: {t_bytes / 1024**2:6.2f} MB | Source: \033[93m{source_loc}\033[0m"
            )

        del leaked_tensors
        main_app_logger.info(
            f"[VRAM INVESTIGATOR] Total Live Tensor Memory: {total_detected_bytes / 1024**2:.2f} MB"
        )
        gc.collect()
        if (
            torch.cuda.is_available()
        ):  # self.device_input == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Flush the pool BEFORE the guard snapshots it

    def print_active_shared_memory(self):
        main_app_logger.info(
            "=" * 60,
        )
        main_app_logger.info(
            "\033[96m[SHM INVESTIGATOR] Scanning Active OS Shared Memory Filesystem Tables:\033[0m"
        )
        try:
            shm_dir = Path("/dev/shm")
            if shm_dir.exists():
                # Extract and inventory all live POSIX memory segments allocated right now
                shm_files = [
                    f
                    for f in shm_dir.iterdir()
                    if f.is_file()
                    and not f.name.startswith("sem.")
                    and not f.name.startswith("psm")
                ]
                main_app_logger.info(
                    f" > Discovered Live OS-Mapped Memory Nodes: {len(shm_files)}"
                )

                for f_path in shm_files:
                    try:
                        f_stat = f_path.stat()
                        size_mb = f_stat.st_size / (1024 * 1024)

                        # Highlight the files using visual color anchors for scannability
                        main_app_logger.info(
                            f"   ⚠️  \033[93m[ALIVE SHM NODE]\033[0m File: {f_path.name:<25} | Size: {size_mb:7.2f} MB"
                        )
                    except Exception:
                        pass
            else:
                main_app_logger.info(
                    "   [ERROR] /dev/shm runtime directory is inaccessible on this host context."
                )
        except Exception as e:
            main_app_logger.info(
                f"   [WARN] Kernel inspection execution pass failed: {e}"
            )

        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()  # Flush the pool BEFORE the guard snapshots it

    def calculate_leaked_memory(
        self, device, video_name, start_allocated, start_reserved
    ):
        _testMethodName = f"{video_name}_{device}"
        main_app_logger.info("=" * 60)
        max_allowed_leak = 1024 * 1024  # 1MB buffer allowance
        msg = "[LEAKAGE INVESTIGATOR] Scanning memory allocations:\n"
        if device == "gpu" and torch.cuda.is_available():
            # torch.cuda.synchronize()

            end_allocated = torch.cuda.memory_allocated(0)
            end_reserved = torch.cuda.memory_reserved(0)

            leak_allocated = end_allocated - start_allocated
            leak_reserved = end_reserved - start_reserved
            # max_allowed_leak = 1024 * 1024  # 1MB buffer allowance

            if leak_allocated > max_allowed_leak:
                msg += (
                    f"\n🔴 GPU Memory Leak Detected for {_testMethodName}!\n"
                    f"Check for dangling references or missing 'del' statements\n\n"
                )
                # else:
                #     msg = "\n"

            msg += (
                f"\tPre-Setup Allocation:  {start_allocated / 1024**2:.2f} MB\n"
                f"\tPost-Teardown Allocation:  {end_allocated / 1024**2:.2f} MB\n"
                f"\tNet Leaked VRAM: {leak_allocated / 1024**2:.2f} MB\n"
                f"\tNet Leaked Reserved Blocks: {leak_reserved / 1024**2:.2f} MB"
            )

            # main_app_logger.info(msg, )
        else:
            process = psutil.Process(os.getpid())
            end_rss = process.memory_info().rss

            # start_allocated and start_reserved must be populated with baseline RSS in each_test_setup
            leak_rss = end_rss - start_allocated

            if leak_rss > max_allowed_leak:
                msg += (
                    f"\n🔴 CPU Memory Leak Detected for {_testMethodName}!\n"
                    f"Check for dangling references or missing 'del' statements\n\n"
                )
            # else:
            #     msg = "\n"

            msg += (
                f" Pre-Setup Host RAM Allocation: {start_allocated / 1024**2:.2f} MB\n"
            )
            msg += f" Post-Teardown Host RAM Allocation: {end_rss / 1024**2:.2f} MB\n"
            msg += f" Net Leaked Host System Memory: {leak_rss / 1024**2:.2f} MB"
        main_app_logger.info(msg)
        main_app_logger.info("=" * 60)

    def diagnostic_profiler(self, device, video_name, start_allocated, start_reserved):
        _testMethodName = f"{video_name}_{device}"
        self.calculate_leaked_memory(
            device, video_name, start_allocated, start_reserved
        )

        # main_app_logger.info("=" * 60, )
        # main_app_logger.info(
        #     f"\n\033[95m[DIAGNOSTICS] Starting Automated Leak Analysis for {_testMethodName}...\033[0m",
        #     ,
        # )

        # 1. Run objgraph to inspect the Python object reference trees before clearing containers
        # try:
        # main_app_logger.info(
        #     "\033[94m[DIAGNOSTICS] Python Object Registry Standings:\033[0m",
        # )
        # objgraph.show_most_common_types(limit=10)

        # Check if the metrics arrays are pinning references inside memory
        # for tracker_attr in ["all_preds", "all_targets"]:
        #     if hasattr(self, tracker_attr):
        #         tgt_list = getattr(self, tracker_attr)
        #         if len(tgt_list) > 0:
        #             graph_path = f"/tmp/backrefs_{tracker_attr}_{device}.png"
        #             main_app_logger.info(
        #                 f"\033[93m[WARN] '{tracker_attr}' contains {len(tgt_list)} entries. Generating reference graph to: {graph_path}\033[0m"
        #             )
        #             objgraph.show_backrefs(
        #                 [tgt_list], max_depth=3, filename=graph_path
        #             )
        # except ImportError:
        #     main_app_logger.info(
        #         "\033[91m[DIAGNOSTICS] 'objgraph' package missing. Skipping reference chain mapping. (pip install objgraph)\033[0m"
        #     )

        # 2. Dump PyTorch Memory Snapshot before clearing VRAM caches
        if device == "gpu" and torch.cuda.is_available():
            try:
                # snapshot_path = f"/tmp/vram_leak_profile_{self._testMethodName}.pickle"
                snapshot_path = self.output_path.replace(".mp4", "_vram_profile.html")
                torch.cuda.memory._dump_snapshot(snapshot_path)
                main_app_logger.info(
                    f"\033[92m[DIAGNOSTICS] VRAM Snapshot Trace generated successfully: {snapshot_path}\033[0m"
                )
                main_app_logger.info(
                    "\033[92m--> Upload this file to https://pytorch.org to inspect leak allocation stacks.\033[0m"
                )

                with open(snapshot_path, "rb") as f:
                    snapshot = pickle.load(f)

                # Print an HTML visualization path map of the allocations
                html_timeline = memory_viz.trace_plot(snapshot)
                html_path = snapshot_path.replace(".pickle", ".html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_timeline)
            except Exception as e:
                main_app_logger.info(
                    f"\033[91m[DIAGNOSTICS] Failed to generate PyTorch memory snapshot: {e}\033[0m"
                )

    def assess_memory(self, device, video_name, start_allocated, start_reserved):
        gc.collect()

        surviving_arrays = [
            obj
            for obj in gc.get_objects()
            # if isinstance(obj, np.ndarray) and obj.size >= 1  #(1920 * 1080)
            if type(obj) is np.ndarray and obj.ndim > 0
        ]
        analyze_tracemalloc_snapshot()

        main_app_logger.info(
            f"[DIAGNOSTICS] Found {len(surviving_arrays)} uncollected large arrays alive in RAM filesystem."
        )

        for i, arr in enumerate(surviving_arrays):
            referrers = gc.get_referrers(arr)
            main_app_logger.info(
                f"  > Array {i} | Shape: {arr.shape} | Pinned by {len(referrers)} references:"
            )
            for ref in referrers:
                if isinstance(ref, dict):
                    main_app_logger.info(
                        f"    - Dict Keys holding this array: {list(ref.keys())[:4]}"
                    )
                else:
                    main_app_logger.info(
                        f"    - Variable holding object layout: {type(ref)}",
                    )
        # ───────────────────────────────────────

        # analyze_tracemalloc_snapshot()

        if device == "gpu":
            self.print_active_gpu_tensor_memory()
        else:
            # --- CPU RAM PATH METRICS ---
            # main_app_logger.info("=" * 60, )
            # main_app_logger.info("\n[RAM INVESTIGATOR] Scanning Host CPU Memory Standings:", )
            # process = psutil.Process(os.getpid())
            # current_rss = process.memory_info().rss / (1024 * 1024)  # Host RAM in MB
            # main_app_logger.info(f" > Current Process Resident Set Size (RSS): {current_rss:.2f} MB", )
            self.print_active_cpu_tensor_memory()

        self.print_active_shared_memory()

        # AUTOMATED DIAGNOSTIC PROFILING PHASE (TRIGGERED ON TEST TEARDOWN)
        self.diagnostic_profiler(device, video_name, start_allocated, start_reserved)

    # VIDEO CLIPPING --------------------------------------------
    def start_new_clip(self, clip_id):
        """
        Seals the current AI tracking state layout and safely moves the instance metadata references
        to the next sequential file block segment index.
        """
        global clip_completion_tracker, all_metadata

        # Mutate tracker instance metrics parameters for the upcoming segment chunk window
        # self.clip_id += 1
        self.frame_in_clip_count = 1
        self._check_shm_safety(threshold_percent=90)

        log_to_logger(
            f"New clip created: clip frame {self.frame_in_clip_count} of {self.max_frames_per_clip} (Overall target frame: {self.frame_count_target})",
            level="info",
        )

    def prep_frame_for_video(self, device_frame, frame_num):
        # Stops the handler from starting zombie threads during stop() flushes
        if not self.active or self._is_stopped:
            return

        if not hasattr(self, "write_queue") or self.write_queue is None:
            main_app_logger.info(
                " [CLIPPER-INIT] Missing write_queue footprint. Provisioning runtime workspace buffer...",
            )
            # self.write_queue = queue.Queue(maxsize=300)
            # write_queue_size = int(self.target_fps / 2) if self.device_input == "cpu" else int(self.target_fps / 2)
            write_queue_size = int(2 * self.target_fps)
            self.write_queue = queue.Queue(maxsize=write_queue_size)  # 300)
            self.writer_done = False

        if not self.config.TEST_MODE and (
            not hasattr(self, "send_metadata_queue") or self.send_metadata_queue is None
        ):
            main_app_logger.info(
                " [CLIPPER-INIT] Binding instance metadata reference array layer dynamically...",
            )
            self.send_metadata_queue = queue.Queue(maxsize=50)

        if not hasattr(self, "stop_writer") or self.stop_writer is None:
            self.stop_writer = threading.Event()

        # if (
        #     not hasattr(self, "writer_process")
        #     or self.writer_process is None
        #     or not self.writer_process.is_alive()
        # ):
        #     main_app_logger.info(
        #         " [CLIPPER-INIT] Target worker runtime thread is offline. Provisioning core consumer loop thread...",
        #     )
        #     self.writer_process = threading.Thread(
        #         target=self.video_writer_core_loop,
        #         args=(self.stop_writer,),
        #         daemon=True,
        #     )
        #     self.writer_process.start()

        # if getattr(self, "video_writer", None) is None:
        #     main_app_logger.info(
        #         " [CLIPPER-INIT] Downstream execution handle is blank. Initializing FFmpeg subprocess daemon...",
        #     )
        #     self._initialize_writer()

        if not self.active or self._is_stopped:
            return

        # self.clip_executor.submit(self._async_clipper_worker, device_frame, frame_num)
        try:
            if hasattr(self, "clip_executor") and self.clip_executor is not None:
                self.clip_executor.submit(
                    self._async_clipper_worker, device_frame, frame_num
                )
        except RuntimeError as e:
            # If the executor was shut down in the fraction of a millisecond
            # between our check and the submit, catch it silently and drop the frame.
            if "cannot schedule new futures" in str(e):
                pass
            else:
                # If it's a different RuntimeError, we still want to see it
                raise e
        except Exception:
            # Catch other unexpected submission errors
            # pass
            traceback.print_exc()

    @torch.inference_mode()
    def _async_clipper_worker_v1(self, device_frame, frame_num):
        """
        A worker that resizes a full-resolution frame and places it in a queue
        for the video writer. Runs in a separate thread.
        """
        # ring_idx = (frame_num - 1) % self.ring_depth
        try:
            # If a shutdown has started, attributes might be gone. Exit cleanly.
            if (
                not self.active
                or not hasattr(self, "processing_stream")
                or self.processing_stream is None
            ):
                return

            if self.device_input == "cuda":
                # with torch.cuda.stream(self.processing_stream):
                if (
                    hasattr(self, "processing_stream")
                    and self.processing_stream is not None
                ):
                    torch.cuda.set_stream(self.processing_stream)
                if device_frame.shape[-1] == 3:
                    gpu_ch_first = device_frame.permute(2, 0, 1).contiguous()
                else:
                    gpu_ch_first = device_frame.contiguous()

                self.gpu_float_staging[0, :, :, :].copy_(
                    gpu_ch_first, non_blocking=True
                )

                gpu_resized = F.interpolate(
                    self.gpu_float_staging,
                    size=(self.resize_h, self.resize_w),
                    # mode="bilinear",
                    # align_corners=False,
                    mode="nearest",
                ).squeeze(0)

                gpu_final = gpu_resized.clamp(0, 255).to(torch.uint8)

                # If the tensor shape follows PyTorch's standard format (Channels, Height, Width),
                # we reverse the channel order [0, 1, 2] to [2, 1, 0] (RGB -> BGR)
                if gpu_final.ndim == 3 and gpu_final.shape[0] == 3:
                    gpu_final = gpu_final[[2, 1, 0], :, :]

                gpu_contiguous = gpu_final.permute(1, 2, 0).contiguous()

                active_tensor = self.pinned_tensors[self.gpu_ring_idx]
                active_tensor.copy_(gpu_contiguous, non_blocking=True)

                if self.slot_events is not None:
                    self.slot_events[self.gpu_ring_idx].record(self.processing_stream)

                # with self.write_queue_backlog_counter.get_lock():
                #     self.write_queue_backlog_counter.value += 1
                # self.write_queue.put(
                #     {
                #         "ring_slot_idx": self.gpu_ring_idx,
                #         "frame_num": frame_num,
                #         "pipe_handle": self.video_writer,
                #     }
                # )
                try:
                    # Put the NumPy array directly into the queue
                    self.write_queue.put_nowait(active_tensor)

                    # (Optional) Increment your backlog counter if you still use it
                    with self.write_queue_backlog_counter.get_lock():
                        self.write_queue_backlog_counter.value += 1
                except queue.Full:
                    pass  # Drop the frame if the writer process is lagging
                self.gpu_ring_idx = (self.gpu_ring_idx + 1) % self.ring_depth
            else:
                active_matrix = self.pinned_matrices[self.cpu_ring_idx]
                cv2.resize(
                    device_frame,
                    (self.resize_w, self.resize_h),
                    dst=active_matrix,
                    interpolation=cv2.INTER_NEAREST,
                )

                # with self.write_queue_backlog_counter.get_lock():
                #     self.write_queue_backlog_counter.value += 1
                # self.write_queue.put(
                #     {
                #         "ring_slot_idx": self.cpu_ring_idx,
                #         "frame_num": frame_num,
                #         "pipe_handle": self.video_writer,
                #     }
                # )
                try:
                    # Put the NumPy array directly into the queue
                    self.write_queue.put_nowait(active_matrix)

                    # (Optional) Increment your backlog counter if you still use it
                    with self.write_queue_backlog_counter.get_lock():
                        self.write_queue_backlog_counter.value += 1
                except queue.Full:
                    pass  # Drop the frame if the writer process is lagging

                self.cpu_ring_idx = (self.cpu_ring_idx + 1) % self.ring_depth

        except Exception as e:
            main_app_logger.info(
                f"[CRITICAL-CLIPPER-WORKER] Resizing execution loop dropped: {e}",
            )
            traceback.print_exc()

        finally:
            # This worker handles large 8K frames. Deleting all local tensor
            # references here is critical to prevent VRAM accumulation.
            if "device_frame" in locals():
                del device_frame
            if "gpu_ch_first" in locals():
                del gpu_ch_first
            if "gpu_resized" in locals():
                del gpu_resized
            if "gpu_final" in locals():
                del gpu_final
            if "gpu_contiguous" in locals():
                del gpu_contiguous

    @torch.inference_mode()
    def _async_clipper_worker(self, device_frame, frame_num):
        """
        A worker that resizes a full-resolution frame and places it in a queue
        for the video writer. Runs in a separate thread.
        """
        # ring_idx = (frame_num - 1) % self.ring_depth
        try:
            # If a shutdown has started, attributes might be gone. Exit cleanly.
            if (
                not self.active
                or not hasattr(self, "processing_stream")
                or self.processing_stream is None
            ):
                return

            if self.device_input == "cuda":
                # with torch.cuda.stream(self.processing_stream):
                if (
                    hasattr(self, "processing_stream")
                    and self.processing_stream is not None
                ):
                    torch.cuda.set_stream(self.processing_stream)
                if device_frame.shape[-1] == 3:
                    gpu_ch_first = device_frame.permute(2, 0, 1).contiguous()
                else:
                    gpu_ch_first = device_frame.contiguous()

                self.gpu_float_staging[0, :, :, :].copy_(
                    gpu_ch_first, non_blocking=True
                )

                gpu_resized = F.interpolate(
                    self.gpu_float_staging,
                    size=(self.resize_h, self.resize_w),
                    # mode="bilinear",
                    # align_corners=False,
                    mode="nearest",
                ).squeeze(0)

                gpu_final = gpu_resized.clamp(0, 255).to(torch.uint8)

                # If the tensor shape follows PyTorch's standard format (Channels, Height, Width),
                # we reverse the channel order [0, 1, 2] to [2, 1, 0] (RGB -> BGR)
                if gpu_final.ndim == 3 and gpu_final.shape[0] == 3:
                    gpu_final = gpu_final[[2, 1, 0], :, :]

                gpu_contiguous = gpu_final.permute(1, 2, 0).contiguous()

                active_tensor = self.pinned_tensors[self.gpu_ring_idx]
                active_tensor.copy_(gpu_contiguous, non_blocking=True)

                if self.slot_events is not None:
                    self.slot_events[self.gpu_ring_idx].record(self.processing_stream)

                # with self.write_queue_backlog_counter.get_lock():
                #     self.write_queue_backlog_counter.value += 1
                # self.write_queue.put(
                #     {
                #         "ring_slot_idx": self.gpu_ring_idx,
                #         "frame_num": frame_num,
                #         "pipe_handle": self.video_writer,
                #     }
                # )

                # try:
                #     # Put the NumPy array directly into the queue
                #     self.write_queue.put_nowait(active_tensor)

                #     # (Optional) Increment your backlog counter if you still use it
                #     with self.write_queue_backlog_counter.get_lock():
                #         self.write_queue_backlog_counter.value += 1
                # except queue.Full:
                #     pass # Drop the frame if the writer process is lagging

                # 1. Get the next available ring buffer slot atomically
                with self.clipper_idx_lock:
                    slot_idx = self.clipper_ring_idx.value
                    self.clipper_ring_idx.value = (
                        self.clipper_ring_idx.value + 1
                    ) % self.clipper_ring_depth

                # 2. Get the NumPy view for that slot and copy the data into it
                target_buffer = self.clipper_shm_np_views[slot_idx]
                np.copyto(target_buffer, active_tensor)

                # 3. Put the INTEGER INDEX into the queue (extremely fast)
                self.write_queue.put_nowait(slot_idx)

                self.gpu_ring_idx = (self.gpu_ring_idx + 1) % self.ring_depth
            else:
                active_matrix = self.pinned_matrices[self.cpu_ring_idx]
                cv2.resize(
                    device_frame,
                    (self.resize_w, self.resize_h),
                    dst=active_matrix,
                    interpolation=cv2.INTER_NEAREST,
                )

                # with self.write_queue_backlog_counter.get_lock():
                #     self.write_queue_backlog_counter.value += 1
                # self.write_queue.put(
                #     {
                #         "ring_slot_idx": self.cpu_ring_idx,
                #         "frame_num": frame_num,
                #         "pipe_handle": self.video_writer,
                #     }
                # )

                # try:
                #     # Put the NumPy array directly into the queue
                #     self.write_queue.put_nowait(active_matrix)

                #     # (Optional) Increment your backlog counter if you still use it
                #     with self.write_queue_backlog_counter.get_lock():
                #         self.write_queue_backlog_counter.value += 1
                # except queue.Full:
                #     pass # Drop the frame if the writer process is lagging

                # 1. Get the next available ring buffer slot atomically
                with self.clipper_idx_lock:
                    slot_idx = self.clipper_ring_idx.value
                    self.clipper_ring_idx.value = (
                        self.clipper_ring_idx.value + 1
                    ) % self.clipper_ring_depth

                # 2. Get the NumPy view for that slot and copy the data into it
                target_buffer = self.clipper_shm_np_views[slot_idx]
                np.copyto(target_buffer, active_matrix)

                # 3. Put the INTEGER INDEX into the queue (extremely fast)
                self.write_queue.put_nowait(slot_idx)

                self.cpu_ring_idx = (self.cpu_ring_idx + 1) % self.ring_depth

        except Exception as e:
            main_app_logger.info(
                f"[CRITICAL-CLIPPER-WORKER] Resizing execution loop dropped: {e}",
            )
            traceback.print_exc()

        finally:
            # This worker handles large 8K frames. Deleting all local tensor
            # references here is critical to prevent VRAM accumulation.
            if "device_frame" in locals():
                del device_frame
            if "gpu_ch_first" in locals():
                del gpu_ch_first
            if "gpu_resized" in locals():
                del gpu_resized
            if "gpu_final" in locals():
                del gpu_final
            if "gpu_contiguous" in locals():
                del gpu_contiguous

    def _initialize_writer(self):
        """
        Spawns a persistent background FFmpeg subprocess with native segment-splitting
        capabilities, entirely bypassing the high-overhead Python-side cv2.VideoWriter lifecycle.
        """
        # self.clip_filename_pattern = f"{self.config.SHARED_OUTPUT}/{self.name}_%03d.mp4"
        # self.clip_key = f"{self.name}_{self.clip_id:03d}.mp4"

        # Safe parameter array construction passed directly to kernel, avoiding subshell expansion failures
        clip_duration = int(self.config.CLIP_DURATION)
        ffmpeg_args = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.resize_w}x{self.resize_h}",
            "-r",
            str(int(self.target_fps)),
            "-i",
            "-",
            "-c:v",
            "libx264",  # "mpeg4",  # Or "libx264" if you prefer H.264
            "-crf",
            "23",
            "-f",
            "mpegts",
            "-movflags",
            "faststart",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{clip_duration})",
            "-f",
            "segment",
            "-segment_time",
            f"{clip_duration}",
            "-reset_timestamps",
            "1",
            "-segment_format",
            "mp4",
            # CRITICAL: Force fragmented headers so every chunk has a valid moov atom instantly
            "-segment_format_options",
            "movflags=frag_keyframe+empty_moov+default_base_moof",
            self.clip_filename_pattern,
        ]

        # main_app_logger.info(f" [FFMPEG-INIT] Spawning binary pipeline targeted at: {self.clip_filename_pattern}")

        try:
            # log_dir = "/home/logs"
            # os.makedirs(log_dir, exist_ok=True)
            # err_log = open(f"{log_dir}/ffmpeg_handler_{self.name}.log", "w")

            self.ffmpeg_proc = subprocess.Popen(
                ffmpeg_args,
                shell=False,  # Secure token delivery
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                # stderr=err_log,
                stderr=subprocess.PIPE,  # Connected directly to high-speed RAM string buffer streaming
                text=False,
                bufsize=0,
            )
            self.video_writer = self.ffmpeg_proc.stdin
            main_app_logger.info(
                " [FFMPEG-INIT] Subprocess online. Log stream parser engine initialization sequencing...",
            )

            # Fire memory loop monitor
            # self.log_parser_thread = threading.Thread(
            #     target=self._ffmpeg_log_parser_loop, daemon=True
            # )
            # self.log_parser_thread.start()

            log_to_logger(
                f"Persistent native-segmenting FFmpeg writer spawned for stream: {self.name}",
                level="info",
            )
        except Exception as e:
            main_app_logger.info(
                f"[CRITICAL-FFMPEG] Process spawn aborted at kernel boundary: {e}",
            )
            self.video_writer = None

    def _ffmpeg_log_parser_loop(self):
        """
        Memory-piped console stream parser. Intercepts segment generation boundary
        milestone markers straight out of VRAM/RAM buffers with zero disk I/O cost.
        """
        global clip_completion_tracker

        while self.active and self.ffmpeg_proc and self.ffmpeg_proc.stderr:
            try:
                raw_line = self.ffmpeg_proc.stderr.readline()
                if not raw_line:
                    break
            except (ValueError, OSError):
                # Safely intercept when stop() forcefully closes the pipe out-of-band
                break
            line = raw_line.decode("utf-8", errors="ignore")

            if "Opening '" in line and "' for writing" in line:
                try:
                    # Isolate text target strings direct from streaming arrays
                    parts = line.split("Opening '")
                    if len(parts) > 1:
                        full_path = parts[1].split("' for writing")[0]
                        target_filename = os.path.basename(full_path)

                        # Parsing string names means the *previous* segment index is finished writing to disk!
                        # Deduce the previous filename key string mathematically
                        try:
                            name_part, ext_part = os.path.splitext(target_filename)
                            prefix, index_str = name_part.rsplit("_", 1)
                            prev_index = int(index_str) - 1
                            if prev_index >= 0:
                                completed_clip_key = (
                                    f"{prefix}_{prev_index:03d}{ext_part}"
                                )
                                completed_clip_path = (
                                    f"{self.config.SHARED_OUTPUT}/{completed_clip_key}"
                                )

                                # =========================================================================
                                # 🛡️ HARD FLUSH & CLOSE GUARD: Wait for OS File Handlers to Stabilize
                                # =========================================================================
                                file_stable = False
                                timeout_gate = 3.0  # Cap the wait window at 3 seconds to avoid blocking AI pipelines
                                start_sync_time = time.time()
                                last_size = -1

                                while (time.time() - start_sync_time) < timeout_gate:
                                    if os.path.exists(completed_clip_path):
                                        try:
                                            current_size = os.path.getsize(
                                                completed_clip_path
                                            )
                                            # Ensure the file isn't empty AND its byte size has stopped fluctuating
                                            if (
                                                current_size > 0
                                                and current_size == last_size
                                            ):
                                                # Final security check: Test if we can open the file exclusively
                                                # (Confirms FFmpeg or OS has released its write-lock completely)
                                                with open(
                                                    completed_clip_path, "rb+"
                                                ) as _:
                                                    pass
                                                file_stable = True
                                                break
                                            last_size = current_size
                                        except IOError:
                                            # File is still locked by the OS writer, loop and wait
                                            pass
                                    time.sleep(
                                        0.05
                                    )  # Rest the polling thread to preserve CPU cycles

                                if not file_stable:
                                    main_app_logger.info(
                                        f" [PARSER-WARN] IO Flush timeout exceeded for {completed_clip_key}. Forcing dispatch anyway.",
                                    )
                                # =========================================================================

                                # Atomic barrier lookup operation update
                                if completed_clip_key not in clip_completion_tracker:
                                    clip_completion_tracker[completed_clip_key] = {
                                        "video": False,
                                        "meta": False,
                                        "start_time": time.time(),
                                    }

                                clip_completion_tracker[completed_clip_key]["video"] = (
                                    True
                                )
                                if self.config.DEBUG_FLAG:
                                    main_app_logger.info(
                                        f"[PARSER] Memory pipe intercepted completed segment confirmation: {completed_clip_key}",
                                    )

                                # Run convergence evaluation pass
                                self._evaluate_barrier_and_dispatch(
                                    completed_clip_key,
                                    completed_clip_path,
                                    self.resize_w,
                                    self.resize_h,
                                )
                        except Exception as calc_err:
                            main_app_logger.info(
                                f"[PARSER-WARN] Index calculation lookback anomaly skipped: {calc_err}",
                            )

                except Exception as parse_err:
                    main_app_logger.info(
                        f"[PARSER-ERROR] Failed to extract target token patterns out of logging stream: {parse_err}",
                    )
                    continue
        if self.config.DEBUG_FLAG:
            main_app_logger.info(
                " [LOG-PARSER] Memory loop pipe interface closed down smoothly.",
            )

    # def video_writer_core_loop(self, stop_evt):
    #     """Thread-safe background min-heap consumer with adaptive sequence hole recovery

    #     and leak-proof lifecycle task tracking signals.
    #     """
    #     main_app_logger.info(
    #         " [WRITER-LOOP] Background tracking consumer loop active and polling memory queues...",
    #     )
    #     try:
    #         # Use a safe helper function to check if the queue is both active and not empty
    #         def is_queue_active_and_populated():
    #             if self.write_queue is None:
    #                 return False
    #             try:
    #                 return not self.write_queue.empty()
    #             except (AttributeError, ValueError):
    #                 # In case the queue was just destroyed by the main thread
    #                 return False

    #         while not stop_evt.is_set() or not is_queue_active_and_populated():
    #             try:
    #                 data = None
    #                 try:
    #                     data = self.write_queue.get(timeout=0.02)
    #                 except (queue.Empty, AttributeError):
    #                     continue

    #                 if data is None:
    #                     continue

    #                 # --- DETACHED PROCESSING LOGIC ENVELOPE ------------------
    #                 try:
    #                     control_data = data.get("control")
    #                     if control_data == "FLUSH":
    #                         main_app_logger.info(
    #                             " [WRITER-LOOP] Intercepted downstream engine flush token code signature.",
    #                         )
    #                         if "pipe_handle" in data and data["pipe_handle"]:
    #                             try:
    #                                 data["pipe_handle"].close()
    #                             except Exception:
    #                                 pass
    #                         continue

    #                     slot_target = data.get("ring_slot_idx")
    #                     sock_handle = data.get("pipe_handle")

    #                     if (
    #                         sock_handle is None
    #                         and getattr(self, "video_writer", None) is not None
    #                     ):
    #                         sock_handle = self.video_writer

    #                     if slot_target is not None and sock_handle is not None:
    #                         if (
    #                             self.device_input == "cuda"
    #                             and self.slot_events is not None
    #                         ):
    #                             self.slot_events[slot_target].synchronize()

    #                         if (
    #                             self.ffmpeg_proc is None
    #                             or self.ffmpeg_proc.poll() is not None
    #                         ):
    #                             main_app_logger.info(
    #                                 " [WRITER-WARN] Downstream execution loop pipe was broken out-of-band. Launching recovery routine...",
    #                             )
    #                             self.initialize_writer()
    #                             sock_handle = self.video_writer
    #                             if sock_handle is None:
    #                                 continue

    #                         try:
    #                             raw_buffer_view = memoryview(
    #                                 self.pinned_matrices[slot_target]
    #                             )
    #                             sock_handle.write(raw_buffer_view)
    #                             sock_handle.flush()
    #                         except (OSError, ValueError) as pipe_err:
    #                             main_app_logger.info(
    #                                 f" [PIPE-ERROR] Write operation dropped on ring index slot {slot_target}: {pipe_err}",
    #                             )
    #                         finally:
    #                             if "raw_buffer_view" in locals():
    #                                 del raw_buffer_view

    #                     del slot_target, sock_handle

    #                     with self.write_queue_backlog_counter.get_lock():
    #                         self.write_queue_backlog_counter.value -= 1
    #                         if (
    #                             self.write_queue_backlog_counter.value % 30 == 0
    #                         ):  # Every ~2 seconds
    #                             gc.collect()
    #                             if torch.cuda.is_available():
    #                                 torch.cuda.empty_cache()

    #                 # --- ENSURE PRECISE TASK TRACKING REGISTRATION ALLOCATION ---
    #                 finally:
    #                     if "data" in locals():
    #                         del data
    #                     # This executes exactly once per queue iteration pass, completely
    #                     # eliminating "task_done() called too many times" exceptions.
    #                     self.write_queue.task_done()

    #             except Exception as e:
    #                 main_app_logger.info(
    #                     f" [WRITER-EXCEPTION] Worker engine cycle processing failure: {e}",
    #                 )
    #                 continue

    #         if hasattr(self, "socket_path") and os.path.exists(self.socket_path):
    #             try:
    #                 os.remove(self.socket_path)
    #             except Exception:
    #                 pass

    #         self.writer_done = True
    #         main_app_logger.info(
    #             " [WRITER-LOOP] Thread pool queue completely drained. Processing safe exit termination sequence...",
    #         )

    #     except Exception as fatal_err:
    #         main_app_logger.info(
    #             f"[FATAL-WRITER-CRASH] Unhandled background crash: {fatal_err}",
    #         )
    #         traceback.print_exc()

    def _evaluate_barrier_and_dispatch(self, clip_key, clip_path, frame_w, frame_h):
        """
        Synchronized atomic thread barrier. Evaluates component convergence and
        dispatches the unified media asset payload directly to VDMS ingestion queues.
        """
        global clip_completion_tracker, all_metadata, send_metadata_queue

        if clip_key not in clip_completion_tracker:
            clip_completion_tracker[clip_key] = {
                "video": False,
                "meta": False,
                "start_time": time.time(),
            }

        tracker = clip_completion_tracker[clip_key]

        # Check convergence layout: Trigger upload sequence only if both pipelines sealed operations
        if tracker["video"] and tracker["meta"]:
            if self.config.DEBUG_FLAG:
                main_app_logger.info(
                    f" [BARRIER-CONVERGENCE] Fully synchronized state reached for asset: {clip_key}",
                )

            # Extract and unmap metadata tracking payloads safely from shared RAM memory space
            # clip_metadata = all_metadata.pop(clip_key, None)
            # clip_completion_tracker.pop(clip_key, None)
            clip_metadata = all_metadata.get(clip_key)
            # Immediately delete the data from the global dictionaries to release VRAM
            if clip_key in all_metadata:
                del all_metadata[clip_key]
            if clip_key in clip_completion_tracker:
                del clip_completion_tracker[clip_key]

            gc.collect()
            if "cuda" in self.device_input:
                torch.cuda.empty_cache()

            if not self.config.TEST_MODE and clip_metadata:
                # Re-insert the fully constructed framework into the active VDMS queue worker pool
                if (
                    hasattr(self, "send_metadata_queue")
                    and self.send_metadata_queue is not None
                ):
                    self.send_metadata_queue.put((clip_path, frame_w, frame_h))
                else:
                    global send_metadata_queue
                    send_metadata_queue.put((clip_path, frame_w, frame_h))

                main_app_logger.info(
                    f" [BARRIER-INGEST] Unified data packages successfully submitted for DB processing: {clip_key}",
                )
            elif not self.config.TEST_MODE:
                main_app_logger.info(
                    f" [BARRIER-WARN] Synchronization completed but all_metadata structure for {clip_key} was empty!",
                )
        elif not self.config.TEST_MODE:
            waiting_on = (
                "video segment closure"
                if not tracker["video"]
                else "AI frame processing execution"
            )
            main_app_logger.info(
                f" [BARRIER-WAIT] {clip_key}: Milestone checked. Awaiting {waiting_on} before issuing DB ingestion call.",
            )


class GPUStreamHandler(DeviceBaseHandler):
    def cleanup_gpu(self):
        """
        Explicitly releases all GPU-allocated memory to prevent
        VRAM leaks in 8K concurrent streams.
        """
        # Iterate through class attributes to explicitly release VRAM.
        for attr_name in list(self.__dict__.keys()):
            attr_value = getattr(self, attr_name)

            # Check if the attribute is a GpuMat
            if isinstance(attr_value, cv2.cuda.GpuMat):
                # 🏎️ Force the NVIDIA driver to deallocate this specific memory segment
                attr_value.release()
                setattr(self, attr_name, None)
                main_app_logger.info(f"✅ Released GpuMat: {attr_name}")

        if hasattr(self, "gpu_fullres_frame") and self.gpu_fullres_frame is not None:
            try:
                self.gpu_fullres_frame.release()
            except Exception:
                self.gpu_fullres_frame = None

        if hasattr(self, "gpu_morphed_frame") and self.gpu_morphed_frame is not None:
            try:
                self.gpu_morphed_frame.release()
            except Exception:
                self.gpu_morphed_frame = None

        if hasattr(self, "labels_gpu") and self.labels_gpu is not None:
            try:
                self.labels_gpu.release()
            except Exception:
                self.labels_gpu = None

        if hasattr(self, "gpu_encoder_8k_buf") and self.gpu_encoder_8k_buf is not None:
            try:
                self.gpu_encoder_8k_buf.release()
            except Exception:
                self.gpu_encoder_8k_buf = None

        if hasattr(self, "gpu_display_frame") and self.gpu_display_frame is not None:
            try:
                self.gpu_display_frame.release()
            except Exception:
                self.gpu_display_frame = None

        if hasattr(self, "gpu_crop_batch"):
            for mat in self.gpu_crop_batch:
                if isinstance(mat, cv2.cuda.GpuMat):
                    mat.release()
            self.gpu_crop_batch = []

        # if hasattr(self, "stream"):
        #     self.stream.waitForCompletion()

        self.pinned_downloaded_resizedframe_np = None
        self.gpu_threshold_dst_frame = None
        self.gpu_morphed_frame = None
        self.pinned_downloaded_frame_np = None

        # Handle specific buffers (like your Ping-Pong lists)
        # if hasattr(self, "encode_buffers"):
        #     self.encode_buffers.clear()

        # Clear the BGS history
        if hasattr(self, "mask_history"):
            self.mask_history.clear()

        # Optional: Final flush of the CUDA caching allocator
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


class CPUStreamHandler(DeviceBaseHandler):
    def cleanup_cpu(self):
        """
        Purges large 8K NumPy buffers and CPU-based AI resources.
        """
        self._pinned_small_frame = None
        self._pinned_fg_mask = None
        self._pinned_blurred_mask = None
        self._pinned_threshold_mask = None
        self._pinned_dilated_mask = None

        # Nullify specific class references to allow Garbage Collection
        self.executor = None
        self.clip_executor = None
        self.reader = None
        self.latest_processed_frame = None

        # Clear the Ping-Pong buffers (up to 200MB of RAM)
        # if hasattr(self, "encode_buffers"):
        #     self.encode_buffers.clear()

        # Explicitly nullify large arrays to trigger Garbage Collection
        self.resized_frame = None
        self.fgMask = None
        self.prev_bkgd = None

        # Clear the BGS history
        if hasattr(self, "mask_history"):
            self.mask_history.clear()
