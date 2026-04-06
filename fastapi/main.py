import warnings

warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")


import asyncio
import logging
import os
import sys
import time

from include.handlers import VideoStreamHandler_WIP as VideoStreamHandler
from include.handlers import lifespan
from include.utils import DEBUG, StreamRequest

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

# from include.handlers import VideoStreamHandler, lifespan  # Choppy replay; up to 11 fps
# from include.handlers import VideoStreamHandler1 as VideoStreamHandler, lifespan  # Choppy replay; up to 11 fps
# from include.handlers import VideoStreamHandler2 as VideoStreamHandler, lifespan  # Really choppy replay w/ slight rewind; up to 15 fps
# from include.handlers import VideoStreamHandler3 as VideoStreamHandler, lifespan  # Choppy replay likw 1; up to 10.5 fps
# from include.handlers import VideoStreamHandler4 as VideoStreamHandler, lifespan  # Choppy; up to 11.4 fps
# from include.handlers import VideoStreamHandler5 as VideoStreamHandler, lifespan
# from include.handlers import VideoStreamHandler6 as VideoStreamHandler, lifespan

# ----- LOGGING CONFIGURATION -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
main_app_logger = logging.getLogger("fastapi_app")
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.INFO)


# ----- APPLICATION INITIALIZATION -----
# The lifespan parameter handles startup (model loading) and shutdown (memory cleanup)
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    """
    Renders the main monitoring dashboard.
    Passes current active stream IDs to the frontend for UI synchronization.
    """
    curr_keys = list(request.app.state.active_streams.keys())
    if DEBUG == "1":
        print(f"Active Streams: {curr_keys}")

    return templates.TemplateResponse(
        request=request, name="index.html", context={"cameras": curr_keys}
    )


@app.post("/stream")
async def stream_video(data: StreamRequest, request: Request):
    """
    Initializes a new VideoStreamHandler for a specific source.
    If the stream is not already active, it starts a background processing thread.
    """
    url, name = data.url, data.name
    active_streams = request.app.state.active_streams

    if name not in active_streams:
        print(f"Starting background worker for {name}...")
        handler = VideoStreamHandler(
            url,
            name,
            active_streams,
            model=app.state.model,
        )
        handler.start()

        # Register the handler in the global state for cross-endpoint access
        app.state.active_streams[name] = handler

    curr_keys = list(app.state.active_streams.keys())
    if DEBUG == "1":
        print(
            f"stream DEBUG VIEW | PID: {os.getpid()} | Looking for: {name} | Found Keys: {curr_keys}"
        )

    return {"status": "started", "keys": curr_keys}


@app.get("/view_stream", name="view_stream")
async def view_stream(name: str, request: Request):
    """
    High-bandwidth MJPEG streaming gateway.
    Uses an asynchronous generator to pipe processed JPEG frames to the browser.
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

                # Sequence check: skip if the frame is older than what we just sent
                if streamer.last_frame_id > last_sent_id:
                    if streamer.latest_processed_frame:
                        frame_bytes = streamer.latest_processed_frame
                        streamer.last_heartbeat = time.time()

                        # Multipart JPEG delivery with explicit Content-Length for stability
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: "
                            + str(len(frame_bytes)).encode()
                            + b"\r\n\r\n"  # <--- Two \r\n
                            + frame_bytes
                            + b"\r\n"  # <--- One \r\n
                        )
                        last_sent_id = streamer.last_frame_id

                # Yield control to the event loop to prevent blocking
                await asyncio.sleep(0.001)
        except Exception as e:
            main_app_logger.error(f"Generator Error: {e}")

    return StreamingResponse(
        frame_generator(streamer, request),
        media_type="multipart/x-mixed-replace;boundary=frame",
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
        stats[name] = {
            "current_fps": round(streamer.stat_fps, 2),
            "reencode_backlog": streamer.get_executor_backlog(),
            "total_frames": streamer.stat_frame_count,
        }

        # is_alive = getattr(streamer, "process_thread", None) and streamer.process_thread.is_alive()

        # Safely get the buffer size under lock to avoid race conditions
        # with streamer.buffer_lock:
        #     buffer_backlog = len(streamer.frame_buffer)

        # stats[name] = {
        #     "status": "Active" if streamer.active else "Inactive",
        #     "thread_alive": is_alive,
        #     "fps": round(streamer.stat_fps, 2),
        #     "total_frames": streamer.stat_frame_count,
        #     "ai_backlog": streamer.get_executor_backlog(), # Tasks in ThreadPool
        #     "display_buffer_size": buffer_backlog,        # Frames waiting for sequence
        #     "next_expected_frame": streamer.next_display_id,
        #     "last_heartbeat": round(time.time() - streamer.last_heartbeat, 2)
        # }
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
    streamer = request.app.state.active_streams.get(name)

    if not streamer:
        raise HTTPException(status_code=404, detail=f"Stream '{name}' not found.")

    # Execute the heavy hardware/thread cleanup off the main event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, streamer.stop)

    # Remove reference from global state to allow Garbage Collection
    app.state.active_streams.pop(name, None)

    if DEBUG == "1":
        print(f"--- CLEANUP | Stream '{name}' stopped and removed. ---")

    return {"status": "stopped", "camera": name}


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

        # for name in active_names:
        #     streamer = active_streams.get(name)
        #     if streamer:
        #         streamer.stop()
        #         app.state.active_streams.pop(name, None)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
