import warnings

warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")

import asyncio
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import datetime

import psutil
from include.handlers import lifespan
from include.utils import PipelineConfig, StreamRequest, str2bool

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

MODEL_CLASSES_FILE = "/var/www/cache/model_classes.json"

RUN_CONFIG = PipelineConfig(
    CODE_DIR=os.getenv("CODE_DIR", "/home"),
    CUSTOM_MODEL_FLAG=str2bool(os.getenv("CUSTOM_MODEL_FLAG", False)),
    DBHOST=os.getenv("DBHOST", "vdms-service"),
    DEBUG=os.getenv("DEBUG", "0"),
    DEVICE=os.getenv("DEVICE", "CPU"),
    ENABLE_QUERYING=os.getenv("ENABLE_QUERYING", True),
    INGESTION=os.getenv("INGESTION", "object"),
    MODEL_NAME=os.getenv("MODEL_NAME", "yolo11n"),
    OMIT_DETECTIONS_FLAG=str2bool(os.getenv("OMIT_DETECTIONS_FLAG", False)),
    # RESIZE_FLAG=str2bool(os.getenv("RESIZE_FLAG", False)),
    SHARED_MODEL=os.getenv("SHARED_MODEL", False),
    SHARED_OUTPUT=os.getenv("SHARED_OUTPUT", "/var/www/mp4"),
    TEST_MODE=str2bool(os.getenv("TEST_MODE", False)),
    TMP_LOCATION=os.getenv("TMP_LOCATION", "/var/www/cache"),
    UDF_HOST=os.getenv("UDF_HOST", "udf-service"),
    UDF_PORT=5011,
)
if RUN_CONFIG.DEVICE == "GPU":
    from include.handlers import GPUStreamHandler

    VideoStreamHandler = GPUStreamHandler
else:
    from include.handlers import CPUStreamHandler

    VideoStreamHandler = CPUStreamHandler


# if RUN_CONFIG.ENABLE_QUERYING:
#     from include.handlers import (
#         all_metadata,
#         clip_completion_tracker,
#         send_metadata_queue,
#     )

# ----- LOGGING CONFIGURATION -----
# Standardizes logs across the application and uvicorn server
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
main_app_logger = logging.getLogger("fastapi_app")
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.INFO)


# ----- APPLICATION INITIALIZATION -----
# The lifespan parameter handles startup and shutdown
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# ----- APPLICATION ENDPOINTS -----
@app.get("/")
async def index(request: Request):
    """
    Renders the main monitoring dashboard.
    Passes current active stream IDs to the frontend for UI synchronization.
    Args:
        request (Request): The FastAPI request object.

    Returns:
        TemplateResponse: HTML page with the list of currently active camera IDs.
    """
    curr_keys = list(request.app.state.active_streams.keys())
    if RUN_CONFIG.DEBUG_FLAG:
        print(f"Active Streams: {curr_keys}")

    return templates.TemplateResponse(
        request=request, name="index.html", context={"cameras": curr_keys}
    )


@app.post("/stream")
async def stream_video(data: StreamRequest, request: Request):
    """
    Initializes a new VideoStreamHandler for a specific source and starts background processing.
    If the stream is not already active, it starts a background processing thread.
    Args:
        data (StreamRequest): Pydantic model containing the source URL and unique name.
        request (Request): The FastAPI request object to access global state.

    Returns:
        dict: Status message and the updated list of active stream keys.
    """
    url, name = data.url, data.name
    active_streams = request.app.state.active_streams

    # Only initialize if the stream isn't already being processed
    if name not in active_streams:
        print(f"Starting background worker for {name}...")

        # Check if a global model instance should be passed to the handler
        if RUN_CONFIG.SHARED_MODEL:
            handler = VideoStreamHandler(
                url,
                name,
                active_streams,
                config=RUN_CONFIG,
                model=app.state.model,
            )
        else:
            handler = VideoStreamHandler(
                url,
                name,
                active_streams,
                config=RUN_CONFIG,
            )

        # Start the background thread (OpenCV capture + AI inference)
        handler.start()

        # Register the handler in the global state for cross-endpoint access
        app.state.active_streams[name] = handler

    curr_keys = list(app.state.active_streams.keys())
    if RUN_CONFIG.DEBUG_FLAG:
        print(
            f"stream DEBUG VIEW | PID: {os.getpid()} | Looking for: {name} | Found Keys: {curr_keys}"
        )

    return {"status": "started", "keys": curr_keys}


@app.get("/view_stream", name="view_stream")
async def view_stream(name: str, request: Request):
    """
    High-bandwidth MJPEG streaming gateway.
    Uses an asynchronous generator to pipe processed JPEG frames to the browser.
    Args:
        name (str): The unique identifier of the camera stream.
        request (Request): The FastAPI request object.

    Returns:
        StreamingResponse: A multipart/x-mixed-replace stream of JPEG images.
    """
    active_streams = request.app.state.active_streams

    if name not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    streamer = active_streams.get(name)
    if not streamer:
        raise HTTPException(status_code=404)

    async def frame_generator(streamer, request: Request):
        """
        Yields frames only when the background worker signals a new frame is ready.
        Ensures strict chronological order using frame IDs.
        """
        shm_names = streamer.shared_details["shm_names"]
        reader_shms = [mp.shared_memory.SharedMemory(name=n) for n in shm_names]
        last_sent_id = -1
        try:
            while streamer.active:
                # Stop the generator immediately if the browser tab is closed
                if await request.is_disconnected():
                    main_app_logger.info(f"Client disconnected from {name}")
                    break

                # Wait for the background thread to signal that AI processing is complete
                await streamer.frame_ready_event.wait()
                streamer.frame_ready_event.clear()  # Reset for the next frame

                # streamer.reader_busy.value = True
                # target_idx = streamer.ready_buffer_idx.value

                # streamer.reader_active_idx.value = target_idx

                # Frame Synchronization: ensure we don't send duplicate or out-of-order frames
                current_id = streamer.shared_details.get("last_id", -1)
                if current_id > last_sent_id:
                    ready_idx = streamer.ready_buffer_idx.value
                    streamer.reader_active_idx.value = ready_idx
                    frame_len = streamer.shm_frame_lengths[ready_idx]

                    if frame_len > 0:
                        # shm_name = streamer.shared_details["shm_name"]
                        # shm_name = shm_names[ready_idx]
                        # print(f"DEBUG: Displaying SHM {shm_name}")
                        # frame_bytes = streamer.latest_processed_frame
                        try:
                            frame_bytes = bytes(reader_shms[ready_idx].buf[:frame_len])
                        finally:
                            streamer.reader_active_idx.value = -1
                        last_sent_id = current_id
                        streamer.last_heartbeat = time.time()
                        # streamer.reader_busy.value = False

                        # Multipart JPEG delivery with explicit Content-Length for stability
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: "
                            + str(len(frame_bytes)).encode()
                            + b"\r\n\r\n"
                            # b"Content-Length: "
                            # + str(len(frame_bytes)).encode()
                            # + b"\r\n\r\n"
                            + frame_bytes
                            + b"\r\n"
                        )
                        # last_sent_id = current_id

                # Yield control to the event loop to prevent blocking
                await asyncio.sleep(0.001)
        except Exception as e:
            main_app_logger.error(f"Generator Error: {e}")
        # finally:
        #     streamer.reader_busy.value = False # Safety unlock

    return StreamingResponse(
        frame_generator(streamer, request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream_list")
async def get_stream_list(request: Request):
    """Returns a thread-safe list of all currently running stream identifiers."""
    return list(request.app.state.active_streams.keys())


@app.get("/stream_stats")
async def get_stats(request: Request):
    """Provides granular FPS and frame-count metrics for all running streams."""
    return {
        name: {
            "fps": round(streamer.stat_fps, 1),
            "inputfps": round(streamer.input_fps, 1),
            "targetfps": round(streamer.target_fps, 1),
            "frames": streamer.stat_frame_count,
            # "status":
        }
        for name, streamer in request.app.state.active_streams.items()
    }


@app.get("/dashboard_stats")
async def dashboard_stats(request: Request):
    """
    Returns real-time performance metrics for the dashboard overlay,
    including FPS and the background processing backlog.
    """
    stats = {}
    active_streams = request.app.state.active_streams

    for name, streamer in active_streams.items():
        # AI Backlog: Tasks waiting in the ThreadPool
        ai_backlog = streamer.get_executor_backlog()
        # AI Backlog: Tasks waiting in the ThreadPool
        clipper_backlog = streamer.get_clip_executor_backlog()

        # Video Backlog: Frames waiting for Disk I/O
        video_backlog = (
            streamer.write_queue.qsize() if streamer.config.ENABLE_QUERYING else 0
        )

        # IO Backlog: Frames queued for disk storage (if enabled)
        io_backlog = (
            streamer.io_executor._work_queue.qsize()
            if hasattr(streamer, "io_executor")
            else 0
        )

        # Calculate Shared Memory (/dev/shm) usage - critical for Docker/Linux deployments
        shm_usage = shutil.disk_usage("/dev/shm")
        shm_percent = (shm_usage.used / shm_usage.total) * 100

        stats[name] = {
            "fps": round(streamer.stat_fps, 1),
            "inputfps": round(streamer.input_fps, 1),
            "targetfps": round(streamer.target_fps, 1),
            "is_streaming": streamer.active,
            "clipper_backlog": clipper_backlog,
            "ai_backlog": ai_backlog,
            "video_backlog": video_backlog,
            "io_backlog": io_backlog,
            "querying_active": streamer.config.ENABLE_QUERYING,
            "total_frames": streamer.stat_frame_count,
            "shm_usage": f"{shm_percent:0.1f}%",
        }
    return stats


@app.get("/status")
async def get_status(request: Request):
    """Returns the overall system status (e.g., 'Ready', 'Loading', or 'Error')."""
    return {
        "status": request.app.state.status
        if hasattr(request.app.state, "status")
        else "Loading"
    }


@app.post("/stop_stream/{name}")
async def stop_stream(name: str, request: Request):
    """
    Gracefully stops a single stream and releases its hardware/VRAM resources.
    The blocking cleanup logic is offloaded to a separate thread to prevent API hang.
    """
    # Immediately remove from the global state so polling/UI syncs instantly
    streamer = request.app.state.active_streams.pop(name, None)

    if streamer:
        streamer.active = False

        # BACKGROUND CLEANUP: Fire-and-forget the heavy hardware teardown
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, streamer.stop)

        if streamer.config.DEBUG_FLAG:
            print(f"--- CLEANUP | Stream '{name}' stopped and removed. ---")
        return {"status": "stopped", "camera": name}

    return {"status": "not found"}


@app.post("/stop_all")
async def stop_all_streams(request: Request):
    """
    Stops all active cameras and purges hardware resources.
    Uses a list snapshot to safely iterate while modifying the dictionary.
    """
    # Use the global stream_lock to prevent janitor/new-streams from interfering
    async with request.app.state.stream_lock:
        active_streams = request.app.state.active_streams
        active_names = list(active_streams.keys())

        if not active_names:
            return {"status": "success", "message": "No active streams to stop"}

        for name in active_names:
            streamer = active_streams.pop(name, None)
            if streamer:
                # Offload to executor to keep the API responsive
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, streamer.stop)

    return {
        "status": "success",
        "stopped_count": len(active_names),
        "cleared_streams": active_names,
    }


@app.get("/health")
async def health_check(request: Request):
    """
    Diagnostic endpoint for monitoring system stability.
    Returns:
        dict: Hardware metrics, stream backlogs, and executor status.
    """
    # Hardware Metrics
    ram = psutil.virtual_memory()
    # Check RAM disk usage (critical for 8K MJPEG/MP4 buffers)
    shm = shutil.disk_usage("/dev/shm")

    health_data = {
        "status": "online",
        "system_time": datetime.now().isoformat(),
        "hardware": {
            "ram_used_percent": ram.percent,
            "shm_used_percent": round((shm.used / shm.total) * 100, 1),
            "cpu_count": psutil.cpu_count(),
        },
        "active_streams": len(request.app.state.active_streams),
        "stream_details": {},
    }

    # Pipeline Backlogs (Identify bottlenecks)
    for name, streamer in request.app.state.active_streams.items():
        health_data["stream_details"][name] = {
            "ai_backlog": streamer.get_executor_backlog(),
            "io_backlog": streamer.write_queue.qsize()
            if hasattr(streamer, "write_queue")
            else 0,
            "fps_live": streamer.stat_fps,
            "uptime_sec": round(time.perf_counter() - streamer.stat_start_time, 1),
        }

    # Global Sync Health
    # if RUN_CONFIG.ENABLE_QUERYING:
    #     health_data["sync_engine"] = {
    #         "pending_completions": len(clip_completion_tracker),
    #         # "metadata_buffer_size": len(all_metadata),
    #         "vdms_queue_depth": send_metadata_queue.qsize(),
    #     }

    return health_data


@app.get("/model_classes")
async def get_model_classes():
    classes = app.state.classes

    # Use classes already stored
    if classes is not None:
        main_app_logger.info(f"classes: {classes}")
        return {"classes": classes}

    # Read list from JSON file stored at entrypoint
    if os.path.exists(MODEL_CLASSES_FILE):
        with open(MODEL_CLASSES_FILE, "r") as f:
            data = json.load(f)
            classes = data.get("classes", None)
            if classes is not None:
                app.state.classes = classes
                main_app_logger.info(f"classes: {classes}")
                return classes

    # Read from model of active stream
    stream_name = list(app.state.active_streams.keys())
    main_app_logger.info(f"stream_name: {stream_name}")
    # Extracts the dynamic labels from your loaded AI model instance
    if len(stream_name) > 0:
        streamer = app.state.active_streams.get(stream_name[0])
        if hasattr(streamer, "label_sources") and streamer.label_sources:
            classes = list(streamer.label_sources)
            app.state.classes = classes
            main_app_logger.info(f"classes: {classes}")
            return {"classes": classes}

    # Fallback structure matching your old format if no model is loaded
    default_classes = ["class0"]
    return {"classes": default_classes}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
