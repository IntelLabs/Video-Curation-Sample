import asyncio
import logging
import os
import sys

# Force FFmpeg to use more threads for decoding
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|hwaccel;cuda|threads;1|probesize;32|analyzeduration;0"
)

from include.handlers import VideoStreamHandler, lifespan
from include.utils import DEBUG, StreamRequest

# ----- SETUP LOGGING -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
main_app_logger = logging.getLogger("fastapi_app")
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.INFO)


# --------------- APP -------------------
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    """Renders the dashboard."""
    if DEBUG == "1":
        print(f"Active Streams: {app.state.active_streams.keys()}")
    curr_keys = list(app.state.active_streams.keys())
    # return templates.TemplateResponse(
    #     "index.html", {"request": request, "cameras": curr_keys}
    # )
    return templates.TemplateResponse(
        request=request, name="index.html", context={"cameras": curr_keys}
    )


@app.post("/stream")
async def stream_video(data: StreamRequest):
    url, name = data.url, data.name
    # Start background thread
    if name not in app.state.active_streams:
        print(f"Starting background worker for {name}...")
        app.state.active_streams[name] = VideoStreamHandler(
            url, name, app.state.active_streams
        )  # , model=app.state.model, lock=app.state.model_lock)
    # DEBUG START
    curr_keys = list(app.state.active_streams.keys())
    if DEBUG == "1":
        print(
            f"stream DEBUG VIEW | PID: {os.getpid()} | Looking for: {name} | Found Keys: {curr_keys}"
        )
    # DEBUG END

    return {"status": "started", "keys": list(app.state.active_streams.keys())}


@app.get("/debug_frame/{name}")
async def debug_frame(name: str):
    streamer = app.state.active_streams.get(name)
    if not streamer:
        return {"error": "not found"}
    # DEBUG START
    curr_keys = list(app.state.active_streams.keys())
    if DEBUG == "1":
        print(
            f"debug_frame DEBUG VIEW | PID: {os.getpid()} | Looking for: {name} | Found Keys: {curr_keys}"
        )
    # DEBUG END
    return {
        "active": streamer.active,
        "has_frame": streamer.latest_processed_frame is not None,
        "frame_size": len(streamer.latest_processed_frame)
        if streamer.latest_processed_frame
        else 0,
    }


@app.get("/stream_list")
async def get_stream_list(request: Request):
    """Returns a list of currently active stream names."""
    return list(request.app.state.active_streams.keys())


@app.get("/stream_stats")
async def get_stats(request: Request):
    # Return a dict mapping camera_id to its metrics
    return {
        name: {
            "fps": round(streamer.stat_fps, 1),
            "frames": streamer.stat_frame_count,
            # "status":
        }
        for name, streamer in request.app.state.active_streams.items()
    }


@app.get("/status")
async def get_status(request: Request):
    # Return a dict mapping camera_id to its metrics
    return {"status": app.state.status if hasattr(app.state, "status") else "Loading"}


@app.get("/view_stream", name="view_stream")
async def view_stream(name: str, request: Request):
    if name not in request.app.state.active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    streamer = request.app.state.active_streams.get(name)
    if not streamer:
        raise HTTPException(status_code=404)

    async def frame_generator():
        # try:
        while streamer.active:
            if await request.is_disconnected():
                break
            # 2. Update Heartbeat for Auto-Cleanup
            streamer.last_heartbeat = time.time()
            # 3. Only send a frame if a NEW one is ready
            if streamer.latest_processed_frame:
                # streamer.latest_processed_frame must be raw JPEG bytes
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + streamer.latest_processed_frame
                    + b"\r\n"
                )

            # Tiny sleep (1ms) to prevent 100% CPU usage while waiting
            # for the next unique frame to arrive from the detector.
            await asyncio.sleep(0.01)

        # finally:
        #     if name in request.app.state.active_streams:
        #         del request.app.state.active_streams[name]

    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/stop_stream/{name}")  # or @app.delete
async def stop_stream(name: str, request: Request):
    """Gracefully stops a background stream and cleans up memory."""
    streamer = request.app.state.active_streams.get(name)

    if not streamer:
        raise HTTPException(status_code=404, detail=f"Stream '{name}' not found.")

    # 1. Trigger the internal stop (releases CV2 cap and joins threads)
    streamer.stop()

    # 2. Remove from the shared state
    del request.app.state.active_streams[name]

    if DEBUG == "1":
        print(f"--- CLEANUP | Stream '{name}' stopped and removed. ---")
    return {"status": "stopped", "camera": name}


@app.get("/dashboard_stats")
async def dashboard_stats(request: Request):
    stats = {}
    for name, streamer in request.app.state.active_streams.items():
        stats[name] = {
            "current_fps": round(streamer.stat_fps, 2),
            "reencode_backlog": streamer.get_executor_backlog(),
            "total_frames": streamer.stat_frame_count,
        }
    return stats


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
