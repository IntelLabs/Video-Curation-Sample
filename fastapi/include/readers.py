import ctypes
import logging
import os
import queue
import sys
import threading
import time
import traceback

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils.checks import check_imgsz

# from fastapi import FastAPI

# ----- SETUP LOGGING -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
main_app_logger = logging.getLogger()


# ----- PIPELINE CONFIGURATION -----
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Force OpenCV to use a single thread for its operations.
# This prevents internal OpenCV threads from "racing" against AI logic.
# cv2.setNumThreads(1)

# Force OpenCV to run sequentially to prevent context-switching overhead
cv2.setNumThreads(0)

# DEVICE = os.getenv("DEVICE", "CPU")
# device_input = DEVICE.lower() if DEVICE == "CPU" else "cuda"

from include.utils import PipelineConfig, manual_fps_calculation, str2bool

BASE_PIPELINE_CONFIG = PipelineConfig(
    CODE_DIR=os.getenv("CODE_DIR", "/home"),
    CUSTOM_MODEL_FLAG=str2bool(os.getenv("CUSTOM_MODEL_FLAG", False)),
    DBHOST=os.getenv("DBHOST", "vdms-service"),
    DEBUG=os.getenv("DEBUG", "0"),
    DEVICE=os.getenv("DEVICE", "CPU"),
    ENABLE_QUERYING=os.getenv("ENABLE_QUERYING", False),
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

# Placeholder for dynamic import for PyNvVideoCodec (GPU package)
nvc = None


# ----- VIDEO READERS -----
class BaseReader:
    def __init__(
        self,
        source,
        target_fps=BASE_PIPELINE_CONFIG.TARGET_FPS,
        clip_duration=BASE_PIPELINE_CONFIG.CLIP_DURATION,
        only_target_frames=True,
    ):
        self.source = source
        self.frame_idx = 0
        self.frame_queue = queue.Queue(maxsize=2)  # 5
        self.stopped = False
        self.target_fps = (
            float(target_fps) if target_fps not in [None, 0] else target_fps
        )
        self.clip_duration = (
            float(clip_duration) if clip_duration not in [None, 0] else clip_duration
        )

    def start(self):
        threading.Thread(target=self.stream_frames, daemon=True).start()
        return self

    def stop(self):
        """Cleanly stop the reader and release resources."""
        self.stopped = True
        self.release()

    def read(self):
        try:
            # If the reader is stopped, don't wait a full second;
            # check immediately to speed up the "Draining" phase.
            wait_time = 0.1 if self.stopped else 2.0
            return self.frame_queue.get(timeout=wait_time)
        except Exception:
            return None, None


class CPUHybridReader(BaseReader):
    """
    Decouples frame acquisition from processing.
    Uses a background thread to ingest frames into a small deque,
    preventing OpenCV buffer lag.
    """

    def __init__(
        self,
        source,
        target_fps=BASE_PIPELINE_CONFIG.TARGET_FPS,
        clip_duration=BASE_PIPELINE_CONFIG.CLIP_DURATION,
        only_target_frames=True,
        MODEL_W=BASE_PIPELINE_CONFIG.MODEL_W,
        MODEL_H=BASE_PIPELINE_CONFIG.MODEL_H,
    ):
        super().__init__(source, target_fps=target_fps, clip_duration=clip_duration)

        target_fps, clip_duration = (self.target_fps, self.clip_duration)
        # options = (
        #     {"rtsp_transport": "tcp", "stimeout": "5000000"}
        #     if str(self.source).startswith("rtsp")
        #     else {}
        # )
        self.MODEL_H = MODEL_H
        self.MODEL_W = MODEL_W

        self.cap = self._create_capture(target_fps, clip_duration)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Force low latency
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

        # self.frame_queue = deque(maxlen=5)  # Keep queue small to stay "real-time"
        # self.frame_queue = queue.Queue(maxsize=5)
        # self.stopped = False
        self.device = "CPU"  # Global from include.utils
        # self.frame_idx = 0
        self.target_frame_idx = 0
        # self.frame_queue = queue.Queue(maxsize=30)

    def _create_capture(self, target_fps, clip_duration):
        """Creates a VideoCapture with stable RTSP options."""
        if not isinstance(self.source, cv2.VideoCapture):
            self.source = str(self.source)
            params = [cv2.CAP_PROP_N_THREADS, 1]
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params=params)
        else:
            cap = self.source
        self.get_fps_and_framecnt(cap, target_fps, clip_duration)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.get_frameWH()
        return cap

    # Gets video details
    def get_fps_and_framecnt(self, cap, target_fps, clip_duration):
        self.input_fps = int(cap.get(cv2.CAP_PROP_FPS))  # hardware fps
        # print(f"in fps: {sself.input_fps} target fps: {target_fps}")
        if self.input_fps == 0:  # Case when FPS isn't available
            self.input_fps = manual_fps_calculation(cap, num_frames=10)
            print(f"new in fps: {self.input_fps}")

        self.target_fps = (
            target_fps
            if target_fps not in [None, 0] and self.input_fps > target_fps
            else self.input_fps
        )

        self.frame_skip = int(self.input_fps / self.target_fps)
        if self.frame_skip < 1:
            self.frame_skip = 1
        # self.skip_count = self.frame_skip - 1

        if clip_duration is None:
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            clip_duration = frame_count / self.input_fps
        self.max_frames_per_clip = int(self.target_fps * float(clip_duration))
        self.frame_interval = 1.0 / self.target_fps  # 0.0666s
        # print(
        #     f"in fps: {self.input_fps} self.target fps: {self.target_fps} self.frame_skip: {self.frame_skip}"
        # )

    # Gets frame W and H details
    def get_frameWH(self):
        if (self.frame_height * self.frame_width) < (self.MODEL_H * self.MODEL_W):
            new_sizeHW = check_imgsz([self.MODEL_H, self.MODEL_W])  # expects hxw
        else:
            new_sizeHW = check_imgsz(
                [self.frame_height, self.frame_width]
            )  # expects hxw

        new_sizeWH = (new_sizeHW[1], new_sizeHW[0])

        self.width = new_sizeWH[0]
        self.height = new_sizeWH[1]

        # Configure scaling for 8K-to-Model coordinate mapping
        self.resize_h, self.resize_w = [self.MODEL_H, self.MODEL_W]
        self.scale_x = self.frame_width / self.MODEL_W
        self.scale_y = self.frame_height / self.MODEL_H

    def stream_frames(self):
        """
        Continuously grabs frames. Throttles local files to maintain
        the target FPS and manages RTSP reconnections.
        """
        is_rtsp = str(self.source).startswith("rtsp")
        max_retries = 5
        retry_cnt = 0

        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if not is_rtsp:
                        self.stopped = True
                        break
                    try:
                        # Logic to recreate the VideoCapture
                        self.cap = self._create_capture(
                            self.target_fps, self.clip_duration
                        )
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        retry_cnt = 0
                    except Exception:
                        retry_cnt += 1
                        if retry_cnt >= max_retries:
                            self.stopped = True
                            break

                        wait_time = retry_cnt * 2
                        main_app_logger.warning(
                            f"CPU reconnect failed. Retry {retry_cnt} in {wait_time}s"
                        )
                        time.sleep(wait_time)
                        continue

                # Fully decode this frame
                ret, frame = self.cap.read()
                if not ret:
                    if not is_rtsp:
                        self.stopped = True
                        break

                    # If a live stream returns No Frame, don't just die—trigger a reconnect
                    main_app_logger.warning("No frame received. Attempting reconnect.")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    continue

                self.frame_queue.put((frame, self.frame_idx))
                self.target_frame_idx += 1
                self.frame_idx += 1
            except Exception as e:
                main_app_logger.error(f"CPU Reader error: {e}")
                time.sleep(1)

    def release(self):
        print("Closing HybridReader...")
        self.stopped = True

        time.sleep(0.2)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class GPUHybridReader(BaseReader):
    """
    Encapsulates a Zero-Copy 8K video pipeline.
    Bridges PyAV (Demuxing) and NVDEC (Hardware Decoding) directly to PyTorch.
    """

    def __init__(
        self,
        source,
        gpu_id=0,
        only_target_frames=True,
        target_fps=BASE_PIPELINE_CONFIG.TARGET_FPS,
        clip_duration=BASE_PIPELINE_CONFIG.CLIP_DURATION,
        MODEL_W=BASE_PIPELINE_CONFIG.MODEL_W,
        MODEL_H=BASE_PIPELINE_CONFIG.MODEL_H,
    ):
        global nvc
        super().__init__(source, target_fps=target_fps, clip_duration=clip_duration)

        if "PyNvVideoCodec" not in sys.modules:
            try:
                import PyNvVideoCodec as nvc

                globals()["nvc"] = nvc

            except ImportError:
                raise ImportError(
                    "GPUHybridReader requires PyNvVideoCodec. Please install."
                )

        self.gpu_id = gpu_id
        self.container = None
        self.nv_dec = None
        self.bsf = None

        # --- Initialize CUDA Context & Bridge ---
        torch.cuda.set_device(self.gpu_id)
        torch.cuda.init()
        _ = torch.zeros(1).cuda()  # Force context creation

        # self.is_opened = self.open(target_fps, clip_duration)
        self.cuda_lib = ctypes.CDLL("libcuda.so.1")
        ctx = ctypes.c_void_p()
        self.cuda_lib.cuCtxGetCurrent(ctypes.byref(ctx))
        self.cuda_ctx_handle = ctx.value if ctx.value is not None else 0

        self.cuda_lib.cuCtxSetCurrent(ctypes.c_void_p(self.cuda_ctx_handle))
        target_fps, clip_duration = (self.target_fps, self.clip_duration)
        self.av_options = (
            {
                "rtsp_transport": "tcp",
                "stimeout": "2000000",  # 2s
                "probesize": "10000000",  # "32000000",  # 32MB for 8K
                "analyzeduration": "5000000",
                "buffer_size": "10240000",  # 10MB socket buffer
            }
            if str(self.source).startswith("rtsp")
            else {}
        )

        # try:
        # self.container = av.open(
        #     self.source,
        #     options={
        #         **self.av_options,
        #         "err_detect": "ignore_err",  # Don't stop demuxing on minor packet errors
        #         "flags": "low_delay",  # Reduce internal buffering
        #     },
        # )
        # self.container.streams.video[0].thread_type = 'AUTO'
        # streams = self.container.streams.get(video=0)
        # self.stream = streams[0] if isinstance(streams, list) else streams

        self.get_stream()

        # Map Codec
        codec_map = {"hevc": nvc.cudaVideoCodec.HEVC, "h264": nvc.cudaVideoCodec.H264}
        self.nvc_codec = codec_map.get(
            self.stream.codec_context.name, nvc.cudaVideoCodec.HEVC
        )

        # self.stream_width = self.stream.width
        # self.stream_height = self.stream.height

        self.frame_width = self.true_width
        self.frame_height = self.true_height

        self.input_fps = self.metadata_fps

        self.target_fps = (
            target_fps
            if target_fps not in [None, 0] and self.input_fps > target_fps
            else self.input_fps
        )
        self.frame_skip = int(self.input_fps / self.target_fps)
        if self.frame_skip < 1:
            self.frame_skip = 1
        self.max_frames_per_clip = (
            None
            if clip_duration is None
            else int(self.target_fps * float(clip_duration))
        )
        self.frame_interval = 1.0 / self.target_fps

        self.numFrames = self.total_frames

        self.raw_input = torch.empty(
            (self.frame_height, self.frame_width, 3), dtype=torch.uint8, device="cuda"
        )
        self.sync_locked = True

    def get_stream(self):
        max_init_retries = 5
        init_retry_cnt = 0
        connected = False
        is_rtsp = str(self.source).startswith("rtsp")

        while not connected:
            try:
                self.container = av.open(
                    self.source,
                    options={
                        **self.av_options,
                        "err_detect": "ignore_err",  # Don't stop demuxing on minor packet errors
                        "flags": "low_delay",  # Reduce internal buffering
                    },
                )
                self.container.streams.video[0].thread_type = "AUTO"
                streams = self.container.streams.get(video=0)
                self.stream = streams[0] if isinstance(streams, list) else streams

                if self.stream:
                    self.stream_width = self.stream.width
                    self.stream_height = self.stream.height
                    connected = True

            except Exception as e:
                init_retry_cnt += 1
                if not is_rtsp or init_retry_cnt >= max_init_retries:
                    main_app_logger.error(
                        f"Critical: Could not open/connect to {self.name}"
                    )
                    raise e

                main_app_logger.warning(
                    f"Failed to connect to stream. Retry {init_retry_cnt}/{max_init_retries}..."
                )
                time.sleep(init_retry_cnt * 2)

    def stream_frames(self):
        """
        Universal background thread for 8K video.
        - RTSP: Reconnects automatically if the stream drops.
        - Files: Processes until EOF and then stops cleanly.
        """
        is_rtsp = str(self.source).startswith("rtsp")
        max_retries = 5
        retry_cnt = 0

        # The outer while loop allows RTSP to recover from network hiccups
        while not self.stopped:
            try:
                # Ensure the container and decoder are active
                # For RTSP, if the demuxer loop below exits, we re-verify the connection here
                if self.container is None:
                    try:
                        self.container = av.open(
                            self.source,
                            options={
                                **self.av_options,
                                "err_detect": "ignore_err",  # Don't stop demuxing on minor packet errors
                                "flags": "low_delay",  # Reduce internal buffering
                            },
                        )
                        retry_cnt = 0
                    except Exception as e:
                        retry_cnt += 1
                        if retry_cnt >= max_retries:
                            # Catch the 404 and stop the thread instead of retrying
                            main_app_logger.info(
                                f"Stream {self.source} ended or not found. Closing reader."
                            )
                            self.stopped = True
                            break
                        wait_time = retry_cnt * 2
                        main_app_logger.warning(
                            f"Connection failed ({e}). Retry {retry_cnt}/{max_retries} in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue

                    streams = self.container.streams.get(video=0)
                    self.stream = streams[0] if isinstance(streams, list) else streams

                # INITIALIZE DECODER if missing (Fixes the NoneType Error)
                if self.nv_dec is None:
                    # Sync with PyTorch
                    self.cuda_lib.cuCtxSetCurrent(ctypes.c_void_p(self.cuda_ctx_handle))
                    torch_stream = torch.cuda.current_stream().cuda_stream

                    # Detect Codec
                    codec_map = {
                        "hevc": nvc.cudaVideoCodec.HEVC,
                        "h264": nvc.cudaVideoCodec.H264,
                    }
                    nvc_codec = codec_map.get(
                        self.stream.codec_context.name, nvc.cudaVideoCodec.HEVC
                    )

                    # Professional cards often support 8K HEVC but are capped at 4K for H.264
                    if nvc_codec == nvc.cudaVideoCodec.HEVC:
                        # Use 8K limits for HEVC
                        hw_max_w, hw_max_h = 8192, 4320
                    else:
                        # Safely default to 4K limits for H.264 and other codecs
                        hw_max_w, hw_max_h = 4096, 4096

                    # Ensure max dimensions are at least as large as current stream
                    hw_max_w = max(hw_max_w, self.stream_width)
                    hw_max_h = max(hw_max_h, self.stream_height)

                    # Determine if the library build supports Ultra Low Latency enums.
                    try:
                        latency_mode = nvc.DisplayDecodeLatencyType.ULTRA_LOW_LATENCY
                    except AttributeError:
                        latency_mode = nvc.DisplayDecodeLatencyType.NATIVE

                    self.nv_dec = nvc.CreateDecoder(
                        gpuid=self.gpu_id,
                        codec=nvc_codec,
                        cudacontext=int(self.cuda_ctx_handle),
                        cudastream=int(torch_stream),
                        usedevicememory=1,
                        maxwidth=hw_max_w,  # Force 8K profile support
                        maxheight=hw_max_h,
                        latency=latency_mode,
                    )
                    print(
                        f"[DEBUG] Hardware Decoder initialized for {self.source}",
                        flush=True,
                    )

                # Re-initialize BitStream Filter for this session
                bsf_map = {"hevc": "hevc_mp4toannexb", "h264": "h264_mp4toannexb"}
                bsf_name = bsf_map.get(self.stream.codec_context.name)
                local_bsf = (
                    av.BitStreamFilterContext(bsf_name, self.stream)
                    if bsf_name
                    else None
                )

                target_fps, clip_duration = (self.target_fps, self.clip_duration)
                self.stream_width = self.stream.width
                self.stream_height = self.stream.height
                self.frame_width = self.true_width
                self.frame_height = self.true_height
                self.input_fps = self.metadata_fps
                self.target_fps = (
                    target_fps
                    if target_fps not in [None, 0] and self.input_fps > target_fps
                    else self.input_fps
                )
                self.frame_skip = int(self.input_fps / self.target_fps)
                if self.frame_skip < 1:
                    self.frame_skip = 1
                self.max_frames_per_clip = (
                    None
                    if clip_duration is None
                    else int(self.target_fps * float(clip_duration))
                )
                self.frame_interval = 1.0 / self.target_fps
                self.numFrames = self.total_frames

                self.is_h264_8k = (
                    self.stream.codec_context.name == "h264"
                    and self.stream_width > 4096
                )

                # --- PATH A: H.264 8K CPU Fallback ---
                if self.is_h264_8k:
                    print(
                        f"[INFO] Starting CPU-Decode Fallback for {self.source}",
                        flush=True,
                    )
                    for frame in self.container.decode(video=0):
                        # if self.stopped:
                        #     break

                        # img_array = frame.to_ndarray(format="rgb24")
                        img_array = frame.to_ndarray(format="bgr24")
                        # gpu_tensor = (
                        #     torch.from_numpy(img_array).to("cuda").permute(2, 0, 1)
                        # )
                        self.raw_input.copy_(torch.from_numpy(img_array))
                        gpu_tensor = self.raw_input.permute(2, 0, 1)

                        self.frame_queue.put((gpu_tensor, self.frame_idx))
                        self.frame_idx += 1

                        # if not is_rtsp and self.frame_idx >= 500:
                        #     break  # File test limit

                # --- PATH B: HEVC Hardware Acceleration (15 FPS Path) ---
                else:
                    print(
                        f"[INFO] Starting HW-Accelerated Pump for {self.source}",
                        flush=True,
                    )
                    for packet in self.container.demux(self.stream):
                        if self.stopped:
                            break

                        # Check for corruption/validity safely
                        is_broken = packet.size == 0 or packet.dts is None
                        is_broken = (
                            is_broken
                            or getattr(packet, "corrupt", False)
                            or getattr(packet, "is_corrupt", False)
                        )

                        if is_broken:
                            print(
                                "[WARNING] Corrupt packet detected. Locking sync.",
                                flush=True,
                            )
                            self.sync_locked = True
                            continue

                        if self.sync_locked:
                            if packet.is_keyframe:
                                if packet.size > 100000:  # 100KB
                                    self.sync_locked = False
                                else:
                                    continue

                        # if self.stopped:
                        #     break
                        if packet.size == 0 or packet.dts is None:
                            continue

                        # Apply Annex B filter
                        filtered_packets = (
                            local_bsf.filter(packet) if local_bsf else [packet]
                        )

                        for filtered_packet in filtered_packets:
                            if self.sync_locked:
                                continue

                            if filtered_packet.size < 10:
                                continue

                            pkt_bytes = bytes(filtered_packet)
                            # Extract raw memory address from the numpy tuple
                            ptr_info = np.frombuffer(
                                pkt_bytes, dtype=np.uint8
                            ).__array_interface__["data"]
                            addr = (
                                ptr_info[0]
                                if isinstance(ptr_info, (tuple, list))
                                else ptr_info
                            )
                            nvc_packet = nvc.PacketData()
                            nvc_packet.bsl_data = int(addr)
                            nvc_packet.bsl = filtered_packet.size

                            try:
                                for decoded_frame in self.nv_dec.Decode(nvc_packet):
                                    try:
                                        # Zero-Copy Bridge: Hardware Surface -> PyTorch Tensor
                                        gpu_tensor = torch.from_dlpack(decoded_frame)
                                        # 2. CONVERSION: Turn NV12 [YUV] into BGR immediately
                                        gpu_tensor = self.nv12_to_bgr_reader(
                                            gpu_tensor,
                                            # self.frame_height,
                                            # self.frame_width,
                                            # is_8k=self.is_h264_8k,
                                        )

                                        # Use .clone() for threaded safety to prevent hardware surface reuse glitches
                                        self.frame_queue.put(
                                            (gpu_tensor.clone(), self.frame_idx)
                                        )
                                        self.frame_idx += 1
                                    finally:
                                        # Immediate VRAM Cleanup
                                        del gpu_tensor
                                        if hasattr(decoded_frame, "Unlock"):
                                            decoded_frame.Unlock()
                                        del decoded_frame

                            except Exception as e:
                                self.sync_locked = True
                                if any(c in str(e) for c in ["700", "208"]):
                                    self.nv_dec = None
                                    break
                        # if not is_rtsp and self.frame_idx >= 500:
                        #     break  # File test limit

                # --- UNIVERSAL EXIT LOGIC ---
                if not is_rtsp:
                    print(
                        f"[DEBUG] Reached End of File: {self.source}. Exiting thread.",
                        flush=True,
                    )
                    self.stopped = True
                    break
                else:
                    # If we reach here and it's RTSP, the demuxer stopped yielding
                    print(
                        "[WARNING] RTSP glitch detected. Closing container and retrying...",
                        flush=True,
                    )
                    if self.container:
                        # Explicitly flush streams before closing
                        for s in self.container.streams:
                            s.codec_context.flush_buffers()
                        self.container.close()
                    self.container = None

                    if local_bsf:
                        try:
                            local_bsf.filter(None)
                        except Exception:
                            pass

                    # CRITICAL: Reset the hardware decoder handle
                    # This forces the 'while' loop to recreate self.nv_dec (Line 741)
                    self.nv_dec = None
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except Exception:
                            break
                    time.sleep(0.5)  # Wait before reconnection attempt

            except Exception as e:
                print(f"[ERROR] Reader Thread Failure: {e}", flush=True)
                traceback.print_exc()
                if not is_rtsp:
                    self.stopped = True
                    break

                self.sync_locked = True
                if self.nv_dec:
                    del self.nv_dec
                    # Force CUDA to synchronize and clear errors
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    self.nv_dec = None
                time.sleep(1.0)  # Backoff for RTSP reconnection

        self.stopped = True

    def release(self):
        """Safely flushes the decoder and closes the connection."""
        print("Closing HybridReader...")
        self.stopped = True  # Signal thread to stop
        time.sleep(0.5)

        if self.nv_dec and self.frame_idx > 0:
            try:
                self.nv_dec.Decode(nvc.PacketData())  # Flush
                del self.nv_dec
                self.nv_dec = None
            except Exception:
                pass
        if self.container:
            self.container.close()
            self.container = None

    def nv12_to_bgr_reader(self, nv12_tensor):  # , h, w, is_8k=False):
        """Internal reader helper to convert raw hardware surfaces to BGR."""
        h, w = self.frame_height, self.frame_width
        is_8k = self.is_h264_8k
        with torch.no_grad():
            if is_8k:
                # 8K Path: Extract and unzipper interleaved UV
                y = nv12_tensor[0:1, :, :].half()
                uv_raw = nv12_tensor[1:2, :, :].half()
                u = uv_raw[:, :, 0::2]
                v = uv_raw[:, :, 1::2]
                u = F.interpolate(u.unsqueeze(0), size=(h, w), mode="nearest").squeeze(
                    0
                )
                v = F.interpolate(v.unsqueeze(0), size=(h, w), mode="nearest").squeeze(
                    0
                )
            else:
                # Standard NV12 Path
                y = nv12_tensor[:h, :w].unsqueeze(0).half()

                uv = (
                    nv12_tensor[h:, :w]
                    .reshape(h // 2, w // 2, 2)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .half()
                )
                uv_up = F.interpolate(uv, size=(h, w), mode="nearest")
                u, v = uv_up[0, 0:1, :, :], uv_up[0, 1:2, :, :]

            # BT.709 Math (Natural Color)
            y = (y - 16.0) * 1.164
            u, v = u - 128.0, v - 128.0

            r = y + 1.793 * v
            g = y - 0.213 * u - 0.533 * v
            b = y + 2.112 * u

            # Stack as BGR [B, G, R] for OpenCV/Browser compatibility
            return torch.cat([b, g, r], dim=0).clamp(0, 255).to(torch.uint8)

    @property
    def true_height(self):
        """
        Returns the logical video height.
        Handles cases where metadata might report the 1.5x NV12 buffer height.
        """
        # if self.nv_dec:
        #     return int(self.nv_dec.Height()) # Get from decoder instead of stream
        # return int(self.stream.height)  # / 1.5)
        return self.stream.height if self.stream else 0

    @property
    def true_width(self):
        # if self.nv_dec:
        #     return self.nv_dec.Width()
        # return self.stream.width
        return self.stream.width if self.stream else 0

    @property
    def metadata_fps(self):
        """Returns the FPS defined in the video metadata/header."""
        if self.stream and self.stream.average_rate:
            return float(self.stream.average_rate)
        return 0.0

    @property
    def total_frames(self):
        """Returns the total number of frames defined in the file metadata."""
        if self.stream:
            # Check nb_frames first
            if self.stream.frames > 0:
                return self.stream.frames
        return 0  # Returns 0 for live streams or files with missing metadata
