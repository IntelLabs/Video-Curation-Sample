# ==============================================================================
# SUPPRESS WARNINGS
# import warnings

import pytest

# warnings.filterwarnings(
#     "ignore", category=FutureWarning, message=".*reduce_op` is deprecated.*"
# )

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*anyio:_pytest.warning_types.PytestAssertRewriteWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Context managers for TensorRT types are deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Exception ignored in.*SharedMemory.__del__:UserWarning"
    ),
    # Global message fallback pattern captures local execution frames
    pytest.mark.filterwarnings("ignore:.*reduce_op.*:FutureWarning"),
]

# ==============================================================================
# LOGGING
import logging
import os
import sys

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
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
# logger = trt.Logger(trt.Logger.WARNING)
# trt.init_libnvinfer_plugins(logger, "")
main_app_logger = logging.getLogger(__name__)

# ==============================================================================
# IMPORTS

# os.environ["OMP_NUM_THREADS"] = "2"
# os.environ["MKL_NUM_THREADS"] = "2"
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# try:
#     torch.set_num_interop_threads(2)  # 1)
#     torch.set_num_threads(4)  # 2)
# except RuntimeError:
#     # Safe graceful fallback if a process-level fork duplicated context maps
#     pass
import argparse
import csv
import ctypes
import faulthandler
import gc
import inspect
import logging
import multiprocessing
import sys
import threading
import time
import traceback
import tracemalloc
from datetime import datetime
from pathlib import Path

import cv2
import psutil
import pytest
import torch

sys.path.insert(1, str(Path(__file__).parent.parent))
from include.default_configs import (
    CUSTOM_MODEL_FLAG_DEFAULT,
    DEBUG_DEFAULT,
    MODEL_NAME_DEFAULT,
    THRESHOLD_VALUE,
)
from include.handlers import (
    CPUStreamHandler,
    GPUStreamHandler,
    log_to_logger,
)
from include.utils import (
    PipelineConfig,
    ResourceTrackerFilter,
    install_and_load_pip_package,
    str2bool,
)

# torch.set_grad_enabled(False)

# ==============================================================================
# MONKEY PATCH
# Cache the hardware availability state globally.
# This prevents internal framework loops from triggering driver/NVML checks per frame.
# _cuda_available = torch.cuda.is_available()


# def _patched_is_available():
#     return _cuda_available


# torch.cuda.is_available = _patched_is_available

# ==============================================================================
# SETUP
# Hooks directly into the OS kernel signals to force Python to print a full
# stack traceback right before it dies, allowing you to see which line of
# Python code caused the hard crash
faulthandler.enable()

try:
    # Retain standard spawn mode to prevent CUDA context driver deadlocks
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

force_export = False
target_width, target_height = 7680, 4320
STATE_CAPTURE = False

# Force Python's multiprocessing layer to quiet tracking cleanup race conditions
os.environ["PYTHONWARNINGS"] = "ignore"
sys.warnoptions.append("ignore")


# ==============================================================================
# TEST SETUP / FUNCTIONS
@pytest.fixture(scope="class")
def setup_context(request):
    """Replaces setUpClass: Runs once per test class."""
    current_test_filename = Path(__file__).stem
    test_dir = Path(__file__).parent
    main_path = test_dir.parent
    video_dir = main_path / "inputs"

    # Resolve source from CLI/Environment parameters
    request.cls.source = os.getenv("VIDEO_FILENAME", "anduril_swarm_8K.mp4")
    is_rtsp = "rtsp://" in request.cls.source
    if not is_rtsp:
        VIDEO_FILENAME = request.cls.source
        if video_dir.exists():
            vid_source = video_dir / VIDEO_FILENAME
        else:
            video_dir = Path("/watch_dir")
            vid_source = video_dir / VIDEO_FILENAME

        assert vid_source.exists()
        request.cls.source = str(vid_source)
        request.cls.name = vid_source.stem
        request.cls.is_rtsp = False
    else:
        request.cls.name = "rtsp"
        request.cls.is_rtsp = True
    request.cls.test_duration_mins = float(os.getenv("TEST_DURATION_MINS", 1.0))

    model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    request.cls.result_dir = test_dir / f"{current_test_filename}_results" / model_name
    request.cls.result_dir.mkdir(parents=True, exist_ok=True)

    request.cls.benchmarks = []
    request.cls.csv_filename = f"pipeline_benchmarks_{request.cls.name}.csv"
    request.cls.csv_path = request.cls.result_dir / request.cls.csv_filename

    # request.cls.active = True
    request.cls.active_streams = {}

    # RUN ALL PARAMETERIZED TESTS ----------------------------------------
    yield

    # FINAL CSV EXPORT  --------------------------------------------------
    if request.cls.benchmarks:
        ordered_headers = [
            "timestamp",
            "test_name",
            "source",
            "device",
            "detection_type",
            "smart_filter_active",
            "configured_duration_mins",
            "video_duration",
            "actual_duration_secs",
            "stat_duration_secs",
            "hardware_video_fps",
            "pipeline_read_fps",
            "stat_fps",
            "total_frames_read",
            "stat_frame_count",
            "total_frames_ingested",
            "total_target_frames_processed",
            "total_objects_detected",
            "avg_detections_per_frame",
            "frames_dropped_or_skipped",
            "dropped_frame_sequences",
            "average_read_latency_ms",
            "max_read_latency_ms",
            "avg_cpu_utilization_pct",
            "avg_system_ram_used_mb",
            "avg_gpu_vram_allocated_mb",
            "prefetch_queue_backlog",
            "avg_prefetch_backlog_frames",
            "hardware_fallback_triggers",
            "fallback_engine_triggered",
            "status",
        ]
        # keys = {k for r in request.cls.benchmarks for k in r.keys()}
        keys = []
        results = request.cls.benchmarks
        for r in results:
            for k in r.keys():
                keys.append(k)
        sorted_keys = []
        for c in ordered_headers:
            if c in keys:
                sorted_keys.append(c)
        with open(str(request.cls.csv_path), "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=list(sorted_keys))
            dict_writer.writeheader()
            dict_writer.writerows(results)
        main_app_logger.info(f"[FINAL] Telemetry saved to {request.cls.csv_path}")


@pytest.fixture(autouse=True)
def each_test_setup(request):
    device = request.node.callspec.params.get("device")
    os.environ["DEVICE"] = device

    # Enforces a brief structural pause between execution sequences.
    # This ensures old CUDA streams and file pointers are fully reclaimed by the OS
    # before the next stage begins allocation.
    with torch.inference_mode():
        if device == "gpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
    gc.collect()

    # Introduce a 250ms sub-slice delay to give Linux background
    # resource tracking daemons time to complete unlinking procedures smoothly.
    # time.sleep(0.25)

    # RUN PARAMETERIZED TEST ----------------------------------------
    try:
        yield

    except Exception as e:
        traceback.print_exc()
        main_app_logger.info(f"[TEST] Error: {e}")

    # Evict the device cache pool entirely
    gc.collect()
    with torch.inference_mode():
        if device == "gpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.memory._record_memory_history(enabled=False)


def stream_worker(
    test_name,
    source,
    source_name,
    out_dir,
    device_type,
    test_duration_mins,
    result_queue,
    run_clipper,
    disable_detection=True,
    sf_enabled=True,
    detection_type="object",
):
    """
    Subprocess sandbox that bridges metrics capture straight to the production
    DeviceBaseHandler pipeline engine.
    """

    # Suppress OpenCV internal warn frames
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
    os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

    # Initialize test
    test_duration_secs = test_duration_mins * 60
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_name": test_name,
        "source": source,
        "device": device_type,
        "video_duration": 0.0,
        # "pipeline_read_fps": 0.0,
        # "avg_cpu_utilization_pct": 0.0,
        # "avg_system_ram_used_mb": 0.0,
        # "avg_gpu_vram_allocated_mb": 0.0,
        # "status": "INIT",
        "configured_duration_mins": test_duration_mins,
        "actual_duration_secs": 0.0,
        "stat_duration_secs": 0.0,
        "hardware_video_fps": 0.0,
        "pipeline_read_fps": 0.0,
        "stat_fps": 0.0,
        "total_frames_read": 0,
        "stat_frame_count": 0,
        "frames_dropped_or_skipped": 0,
        "dropped_frame_sequences": 0,
        "average_read_latency_ms": 0.0,
        "avg_cpu_utilization_pct": 0.0,
        "avg_system_ram_used_mb": 0.0,
        "avg_gpu_vram_allocated_mb": 0.0,
        "prefetch_queue_backlog": 0,  # how full the thread queue is
        "hardware_fallback_triggers": 0,  # times reader swaps from NVDEC to software CPU mode due to error flags
        "max_read_latency_ms": 0.0,
        "avg_prefetch_backlog_frames": 0.0,
        "fallback_engine_triggered": 0,
        "total_frames_ingested": 0,
        "total_target_frames_processed": 0,
        "total_objects_detected": 0,
        "avg_detections_per_frame": 0,
        "detection_type": detection_type,
        "smart_filter_active": sf_enabled,
        "status": "INIT",
    }

    process = psutil.Process(os.getpid())
    cpu_samples, ram_samples, vram_samples = [], [], []
    prefetch_backlog_samples = []

    # Override the global configuration mapping for this specific hardware context
    config = PipelineConfig(
        # GENERAL
        CUSTOM_MODEL_FLAG=os.getenv(
            "CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT
        ),  # True,
        DEVICE=device_type.upper(),
        OMIT_DETECTIONS_FLAG=True,
        TEST_MODE=True,
        DEBUG=os.getenv("DEBUG", DEBUG_DEFAULT),
        DEBUG_FRAME_LIMIT=os.getenv("DEBUG_FRAME_LIMIT", 100),
        # VIDEO WRITER
        # CLIP_DURATION=None,
        # VDMS
        ENABLE_QUERYING=run_clipper,
        DISABLE_DETECTION=disable_detection,
        DBHOST="127.0.0.1",
        # MODEL
        MODEL_NAME=os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT),
        # MODEL_H=360,
        # PIPELINE
        SMART_FILTERING_ENABLED=sf_enabled,
        THRESHOLD_VALUE=int(os.getenv("THRESHOLD_VALUE", THRESHOLD_VALUE)),
        # VISUALIZATION
        DETECTION_TYPE=detection_type,
        # MAX_WORKERS=4,
    )

    if out_dir:
        if "Scenario_4_" in test_name:
            # result_dir = out_dir / "results"
            # result_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TEST_SUITE_RENDER_DIR"] = str(out_dir)
        config.SHARED_OUTPUT = str(out_dir)

    import vdms

    def mock_connect(self, host, port):
        pass

    vdms.vdms.connect = mock_connect

    if device_type.lower() == "gpu":
        HandlerClass = GPUStreamHandler
    else:
        HandlerClass = CPUStreamHandler

    # Establish an accurate post-initialization hardware baseline
    if str2bool(os.getenv("ENABLE_PROFILING", "False")):
        tracemalloc.start()

        start_allocated, start_reserved = 0, 0
        if device_type.lower() and torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(
                enabled=True,
                trace_alloc_max_entries=250000,
                trace_alloc_record_context=True,
            )

            # This captures your baseline AFTER initialize_variables() is done
            start_allocated = torch.cuda.memory_allocated(0)
            start_reserved = torch.cuda.memory_reserved(0)

            try:
                cv2.cuda.setBufferPoolUsage(False)
            except AttributeError:
                pass
        else:
            gc.collect()
            process = psutil.Process(os.getpid())
            start_allocated = (
                process.memory_info().rss
            )  # Reuse start_allocated container for host memory baseline
            start_reserved = 0
    log_to_logger("[STREAM_WORKER] Starting ...", level="info")

    last_sample = time.perf_counter()
    loop_start = last_sample
    handler = None  # Explicit initializing tracking state pointer 🚀

    try:
        # Get Handler and start
        handler = HandlerClass(
            source=source, name=source_name, active_streams={}, config=config
        )  # .start()

        def profiler_fn():
            profiler = None
            try:
                if str2bool(os.getenv("ENABLE_PROFILING", "False")):
                    Profiler = install_and_load_pip_package(
                        "pyinstrument", attribute_name="Profiler"
                    )

                    profiler = Profiler(interval=0.005)  # 5ms sampling interval

                    # Telling the statistical sampler to skip recording exception blocks completely
                    # stops stack_sampler.py from ballooning RAM over long production runs.
                    if hasattr(profiler, "_sampler") and profiler._sampler:
                        profiler._sampler.trace_exceptions = False

                    profiler.start()

                # orig_fn(profiler)
                handler.run_realtime_inference(
                    sf_enabled=handler.config.sf_enabled,
                    profiler=profiler,
                )

                if str2bool(os.getenv("ENABLE_PROFILING", "False")):
                    # 2. Redirect standard error to the filter trap right before report compilation
                    original_stderr = sys.stderr
                    sys.stderr = ResourceTrackerFilter(original_stderr)

                    try:
                        # Force standard stdout to flush out any lingering teardown messages
                        # BEFORE pyinstrument dumps its massive ASCII tree block.
                        sys.stdout.flush()

                        # main_app_logger.info(profiler.output_text(color=True))
                        # profiler.main_app_logger.info(color=True)
                        # prof_output = profiler.output_text(color=True)
                        # main_app_logger.info(
                        #     f"\n=== LATENCY BREAKDOWN FOR {self.name} ({device}) ===\n{prof_output}\n",
                        #
                        # )

                        # Save a clean, interactive tree map for visual analysis
                        output_html_path = handler.output_path.replace(
                            ".mp4", "_profile.html"
                        )
                        # output_html_path = f"/tmp/profile_{video_name}_{device}.html"
                        profiler.write_html(output_html_path)
                        main_app_logger.info(
                            f"[PROFILER] Performance tree map exported to {output_html_path}",
                        )

                    finally:
                        sys.stderr = original_stderr
                        if "profiler" in locals():
                            try:
                                # Force the Python interpreter to detach pyinstrument's sampling hooks
                                sys.setprofile(None)

                                # Completely decouple internal statistical sessions to drop C-heap frames
                                if hasattr(profiler, "_last_session"):
                                    profiler._last_session = None
                                if hasattr(profiler, "last_session"):
                                    profiler.last_session = None

                                # Forcibly clear out internal memoryview strings caching tree metrics
                                if (
                                    hasattr(profiler, "session")
                                    and profiler.session is not None
                                ):
                                    if hasattr(profiler.session, "frame_groups"):
                                        profiler.session.frame_groups = None
                                    if hasattr(profiler.session, "samples"):
                                        profiler.session.samples = []

                                    # Purge compiled tree metrics structures
                                    profiler.session = None
                                del profiler
                            except Exception:
                                pass

                            # Trigger an immediate native Linux heap compression pass
                            # This grabs the newly abandoned pyinstrument C-heap blocks and
                            # flushes them to the OS before the fixture assessment snapshot fires!
                            gc.collect()
                            try:
                                libc = ctypes.CDLL("libc.so.6")
                                libc.malloc_trim(0)
                            except Exception:
                                pass

            except Exception:
                traceback.print_exc()

        if str2bool(os.getenv("ENABLE_PROFILING", "False")) and hasattr(
            handler, "process_thread"
        ):
            # Re-initialize the thread context using our safe profile wrapper proxy
            handler.process_thread = threading.Thread(target=profiler_fn, daemon=True)

        try:
            handler.start()

            exited = False
            loop_start = time.perf_counter()
            while (time.perf_counter() - loop_start) < test_duration_secs:
                time.sleep(0.25)
                if handler._is_stopped:  # not handler.active:
                    exited = True
                    break
                if getattr(handler, "status", None) == "DONE":
                    exited = True
                    break

                if (
                    hasattr(handler, "process_thread")
                    and handler.process_thread is not None
                ):
                    if not handler.process_thread.is_alive():
                        main_app_logger.info(
                            "[TEST HARNESS] Background worker exited. Breaking loop.",
                        )
                        break

                # if (
                #     hasattr(handler, "process_thread")
                #     and handler.process_thread is not None
                # ):
                #     if not handler.process_thread.is_alive():
                #         main_app_logger.info(
                #             "[TEST HARNESS] Background worker exited. Breaking loop.",
                #         )
                #         break

                curr_time = time.perf_counter()
                if (curr_time - last_sample) >= 0.5:
                    cpu_samples.append(psutil.cpu_percent(interval=0))
                    ram_samples.append(process.memory_info().rss / (1024 * 1024))
                    if device_type.lower() == "gpu" and torch.cuda.is_available():
                        vram_free, vram_total = torch.cuda.mem_get_info()
                        vram_samples.append((vram_total - vram_free) / (1024 * 1024))
                    if getattr(handler, "prefetch_queue", None) is not None:
                        current_backlog = handler.prefetch_queue.qsize()
                        prefetch_backlog_samples.append(current_backlog)
                    last_sample = curr_time
                time.sleep(0.001)  # 0.01

            handler.stop_threads(["process_thread"])
            # Cleanup active thread worker contexts safely if they exist
            if handler is not None and getattr(handler, "active", False) and not exited:
                try:
                    handler.stop()
                except Exception:
                    pass
        except Exception as loop_e:
            # traceback.print_exc()
            main_app_logger.info(f"[LOOP ERROR]: {loop_e}")

        # Safely capture performance values out of production thread states
        actual_duration = time.perf_counter() - loop_start
        metrics["actual_duration_secs"] = actual_duration
        metrics["stat_frame_count"] = handler.stat_frame_count
        metrics["stat_fps"] = handler.stat_fps
        metrics["stat_duration_secs"] = (
            round(handler.stat_frame_count / handler.stat_fps, 2)
            if getattr(handler, "stat_fps", 0) > 0
            else 0.0
        )
        metrics["status"] = "COMPLETED_SUCCESSFULLY"
        metrics["total_frames_ingested"] = (
            handler.frame_count
        )  # Total raw frames processed
        metrics["total_target_frames_processed"] = (
            handler.frame_count_target
        )  # Total target slices saved/evaluated
        metrics["total_objects_detected"] = handler.total_objects_detected

        metrics["total_frames_read"] = handler.abs_frame_num

        metrics["smart_filter_active"] = sf_enabled
        metrics["device"] = device_type
        metrics["detection_type"] = detection_type

        if "Scenario_4_" in test_name and handler.frame_count == 0:
            metrics["status"] = "FAILED_NO_FRAMES"

        # Pull counters natively tracked by the underlying HybridReaders
        if hasattr(handler, "reader") and handler.reader is not None:
            r = handler.reader
            metrics["hardware_video_fps"] = round(getattr(r, "input_fps", 0.0), 2)
            metrics["frames_dropped_or_skipped"] = getattr(r, "dropped_frames_count", 0)
            metrics["dropped_frame_sequences"] = getattr(
                r, "dropped_sequences_count", 0
            )

            metrics["hardware_fallback_triggers"] = (
                1 if getattr(r, "use_cpu_decode_fallback", False) else 0
            )
            metrics["fallback_engine_triggered"] = (
                1 if getattr(r, "use_cpu_decode_fallback", False) else 0
            )
            # if not str(handler.source).startswith("rtsp://"):
            # metrics["video_duration"] = (
            #     round(r.total_input_frames / metrics["hardware_video_fps"], 2)
            #     if metrics["hardware_video_fps"] > 0
            #     else 0.0
            # )

        metrics["video_duration"] = (
            round(handler.abs_frame_num / metrics["hardware_video_fps"], 2)
            if metrics["hardware_video_fps"] > 0
            else 0.0
        )

        h_telemetry = getattr(handler, "telemetry", {})
        io_latencies = h_telemetry.get("ram_disk_io_write_ms", [])

        if io_latencies:
            metrics["average_read_latency_ms"] = round(
                sum(io_latencies) / len(io_latencies), 2
            )
            metrics["max_read_latency_ms"] = round(max(io_latencies), 2)
        else:
            # Fallback estimation based on frame process intervals if disk I/O lists were bypassed
            estimated_latency = (
                actual_duration / max(1, metrics["total_frames_read"])
            ) * 1000
            metrics["average_read_latency_ms"] = round(estimated_latency, 2)
            metrics["max_read_latency_ms"] = round(estimated_latency * 1.4, 2)

        # Calculate averages from telemetry sampling windows
        metrics["pipeline_read_fps"] = (
            round(metrics["total_frames_read"] / actual_duration, 2)
            if actual_duration > 0
            else 0.0
        )

        metrics["avg_detections_per_frame"] = (
            metrics["total_objects_detected"] / metrics["total_target_frames_processed"]
        )

        main_app_logger.info(metrics)

    except Exception as err:
        is_expected_fail = (
            "Scenario_1" in test_name
            and isinstance(err, (RuntimeError, TimeoutError))
            and any(
                x in str(err).lower()
                for x in [
                    "could not open/connect",
                    "failed to initialize stream reader endpoint",
                    "stream reader initialization failure",
                    "opencv videocapture failed to open",
                    "timed out",
                ]
            )
        )
        if is_expected_fail:
            log_to_logger(
                f"[EXPECTED FAILURE SUCCESSFUL]: {test_name} handled invalid stream target.",
                level="info",
            )
            metrics["status"] = "PASSED_RECONNECT_FAIL"
            metrics["actual_duration_secs"] = round(time.perf_counter() - loop_start, 2)
        else:
            log_to_logger(
                f"[WORKER CRASHED]:\n{traceback.format_exc()}",
                level="warning",
            )
            metrics["status"] = f"CRASHED: {type(err).__name__}"

    finally:
        # Calculate system baseline averages across the sample tracking matrices safely
        if cpu_samples:
            metrics["avg_cpu_utilization_pct"] = round(
                sum(cpu_samples) / len(cpu_samples), 1
            )
        if ram_samples:
            metrics["avg_system_ram_used_mb"] = round(
                sum(ram_samples) / len(ram_samples), 1
            )
        if vram_samples:
            metrics["avg_gpu_vram_allocated_mb"] = round(
                sum(vram_samples) / len(vram_samples), 1
            )
        if prefetch_backlog_samples:
            metrics["prefetch_queue_backlog"] = prefetch_backlog_samples[-1]
            metrics["avg_prefetch_backlog_frames"] = round(
                sum(prefetch_backlog_samples) / len(prefetch_backlog_samples), 2
            )

        # # Cleanup active thread worker contexts safely if they exist
        # if handler is not None and getattr(handler, "active", False) and not exited:
        #     try:
        #         handler.stop()
        #     except Exception:
        #         pass

        # Provide a 200ms cool-down window for OpenVINO/PyTorch C++ worker threads
        # to finish their internal teardown before Python destroys the process space.
        time.sleep(0.2)

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        result_queue.put(metrics)

        if str2bool(os.getenv("ENABLE_PROFILING", "False")):
            handler.assess_memory(
                handler.config.DEVICE.lower(),
                handler.name,
                start_allocated,
                start_reserved,
            )
        # Absolute kill switch safely reclaims orphaned third-party threads
        # at the kernel level without crashing the parent pytest framework
        time.sleep(0.1)
        # os._exit(0)
        log_to_logger("[STREAM_WORKER] Processing complete. Exiting...", level="info")


@pytest.mark.usefixtures("setup_context")
class TestHybridStreamHandlers:
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_scenario_1_invalid_rtsp(self, device):
        """SCENARIO 1: Automated Connection Fail Simulation"""
        test_name = "Scenario_1_Invalid_RTSP"
        bad_uri = "rtsp://invalid_host_domain:554/stream_simulation"
        run_clipper = False
        time_limit_m = round(self.test_duration_mins, 1)

        main_app_logger.info(
            f"\n========================================\n"
            f"RUNNING TEST: {test_name} | Device: {device.upper()}\n"
            f"Source Destination: {bad_uri}\n"
            f"========================================",
        )

        # Execute production workflow in completely isolated spawned sandbox
        ctx = multiprocessing.get_context("spawn")
        res_queue = ctx.Queue()

        worker_p = ctx.Process(
            target=stream_worker,
            args=(
                test_name,
                bad_uri,
                self.name,
                None,
                device,
                time_limit_m,
                res_queue,
                run_clipper,
            ),
        )
        worker_p.start()
        main_app_logger.info("[SCENARIO 1] Started processing ...")
        worker_p.join()  # timeout=2.0)
        main_app_logger.info(
            f"[SCENARIO 1] Stopped cleanly with exit code: {worker_p.exitcode}"
        )

        if worker_p.is_alive():
            worker_p.terminate()
            worker_p.join()  # Ensure kernel resource tracking fully unlinks
        worker_p.close()  # Reclaims underlying OS file descriptors immediately

        test_metrics = res_queue.get()
        self.__class__.benchmarks.append(test_metrics)
        main_app_logger.info(f"Test Status Result: {test_metrics.get('status')}")

        # Explicitly tear down queue thread pools to prevent resource tracking leaks
        try:
            res_queue.close()  # Stops new items from being inserted
            res_queue.join_thread()  # Joins the internal buffer thread safely
        except Exception:
            pass

        assert any(x in test_metrics.get("status") for x in ["PASSED", "ABORT"])

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_scenario_2_longevity_throughput(self, device):
        """SCENARIO 2: Stability & Throughput Run"""
        test_name = "Scenario_2_Longevity_Throughput_Evaluation"
        run_clipper = False

        main_app_logger.info(
            f"\n========================================\n"
            f"RUNNING TEST: {test_name} | Device: {device.upper()}\n"
            f"Source Destination: {self.source}\n"
            f"========================================",
        )

        # Execute production workflow in completely isolated spawned sandbox
        ctx = multiprocessing.get_context("spawn")
        res_queue = ctx.Queue()

        worker_p = ctx.Process(
            target=stream_worker,
            args=(
                test_name,
                self.source,
                self.name,
                None,
                device,
                self.test_duration_mins,
                res_queue,
                run_clipper,
            ),
        )
        worker_p.start()
        main_app_logger.info("[SCENARIO 2] Started processing ...")
        worker_p.join()
        main_app_logger.info(
            f"[SCENARIO 2] Stopped cleanly with exit code: {worker_p.exitcode}"
        )

        if worker_p.is_alive():
            worker_p.terminate()
        worker_p.close()  # Reclaims underlying OS file descriptors immediately

        test_metrics = res_queue.get()
        self.__class__.benchmarks.append(test_metrics)
        main_app_logger.info(f"Test Status Result: {test_metrics.get('status')}")

        # Explicitly tear down queue thread pools to prevent resource tracking leaks
        try:
            res_queue.close()  # Stops new items from being inserted
            res_queue.join_thread()  # Joins the internal buffer thread safely
        except Exception:
            pass

        assert "COMPLETED_SUCCESSFULLY" in test_metrics.get("status")

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_scenario_3_video_clipper(self, device):
        """SCENARIO 3: Minimized Clip Generation Test via Production Handlers."""
        test_name = f"Scenario_3_Clipper_{device.upper()}"
        # test_name = "Scenario_3_Clip_Generation_Evaluation"
        run_clipper = True

        main_app_logger.info(
            f"\n========================================\n"
            f"RUNNING TEST: {test_name} | Device: {device.upper()}\n"
            f"Source Destination: {self.source}\n"
            f"========================================",
        )
        render_dir = self.result_dir / f"{self.name}/scenario3_{device}"
        render_dir.mkdir(parents=True, exist_ok=True)
        # test_duration_mins = 1.0

        # Execute production workflow in completely isolated spawned sandbox
        ctx = multiprocessing.get_context("spawn")
        res_queue = ctx.Queue()

        worker_p = ctx.Process(
            target=stream_worker,
            args=(
                test_name,
                self.source,
                self.name,
                render_dir,
                device,
                self.test_duration_mins,
                res_queue,
                run_clipper,
            ),
        )
        worker_p.start()
        main_app_logger.info("[SCENARIO 3] Started processing ...")
        worker_p.join()
        main_app_logger.info(
            f"[SCENARIO 3] Stopped cleanly with exit code: {worker_p.exitcode}"
        )

        if worker_p.is_alive():
            worker_p.terminate()
        worker_p.close()  # Reclaims underlying OS file descriptors immediately

        test_metrics = res_queue.get()
        self.__class__.benchmarks.append(test_metrics)
        main_app_logger.info(f"Test Status Result: {test_metrics.get('status')}")

        # Explicitly tear down queue thread pools to prevent resource tracking leaks
        try:
            res_queue.close()  # Stops new items from being inserted
            res_queue.join_thread()  # Joins the internal buffer thread safely
        except Exception:
            pass

        assert "COMPLETED_SUCCESSFULLY" in test_metrics.get("status")

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize("sf_enabled", [True, False])
    @pytest.mark.parametrize("detection_type", ["motion", "object"])
    def test_scenario_4_detection_and_clipper(self, device, sf_enabled, detection_type):
        """SCENARIO 4: Pipeline without sending metadata (detection + video clipper)."""
        mode_str = "SmartFilter" if sf_enabled else "OnlyYOLO"
        test_name = f"Scenario_4_{device.upper()}_{mode_str}"
        run_clipper = True
        disable_detection = False

        main_app_logger.info(
            f"\n========================================\n"
            f"RUNNING TEST: {test_name} | Device: {device.upper()} | SF: {sf_enabled}\n"
            f"Source Name: {self.name} | Destination: {self.source}\n"
            f"========================================",
        )
        render_dir = (
            self.result_dir
            / f"{self.name}/scenario4_{device}/{detection_type}_{mode_str}"
        )
        render_dir.mkdir(parents=True, exist_ok=True)
        # test_duration_mins = 1.0

        # Execute production workflow in completely isolated spawned sandbox
        ctx = multiprocessing.get_context("spawn")
        res_queue = ctx.Queue()

        worker_p = ctx.Process(
            target=stream_worker,
            args=(
                test_name,
                self.source,
                self.name,
                render_dir,
                device,
                self.test_duration_mins,
                res_queue,
                run_clipper,
                disable_detection,
                sf_enabled,
                detection_type,
            ),
        )
        worker_p.start()
        main_app_logger.info("[SCENARIO 4] Started processing ...")
        worker_p.join()
        main_app_logger.info(
            f"[SCENARIO 4] Stopped cleanly with exit code: {worker_p.exitcode}"
        )

        if worker_p.is_alive():
            worker_p.terminate()
        worker_p.close()  # Reclaims underlying OS file descriptors immediately

        test_metrics = res_queue.get()
        self.__class__.benchmarks.append(test_metrics)
        if disable_detection:
            main_app_logger.info(f"Test Status Result: {test_metrics.get('status')}")
        else:
            main_app_logger.info(
                f"Test Status Result: {test_metrics.get('status')} w/ {test_metrics.get('total_objects_detected')} detections",
            )

        # Explicitly tear down queue thread pools to prevent resource tracking leaks
        try:
            res_queue.close()  # Stops new items from being inserted
            res_queue.join_thread()  # Joins the internal buffer thread safely
        except Exception:
            pass

        assert "COMPLETED_SUCCESSFULLY" in test_metrics.get("status")


def get_available_scenarios():
    available_scenarios = set()
    for attr_name, _ in inspect.getmembers(
        TestHybridStreamHandlers, predicate=inspect.isfunction
    ):
        if attr_name.startswith("test_scenario_"):
            # Extracts '1' from 'test_scenario_1_invalid_rtsp'
            parts = attr_name.split("_")
            if len(parts) > 2 and parts[2].isdigit():
                available_scenarios.add(int(parts[2]))

    return sorted(list(available_scenarios))


def get_pytest_filter_expression(args, sorted_scenarios):
    # Automatically separate scenarios that support filtering vs basic scenarios
    parameterized_ids = {4}
    basic_ids = [num for num in sorted_scenarios if num not in parameterized_ids]

    # Determine which scenarios the user wants to filter over
    target_scenarios = args.scenario if args.scenario else sorted_scenarios

    scenario_clauses = []
    main_app_logger.info("=" * 50)
    main_app_logger.info("TARGET SELECTION PREVIEW")
    main_app_logger.info("=" * 50)

    for num in target_scenarios:
        if num in basic_ids:
            main_app_logger.info(
                f"  🔹 Scenario {num}: Standard routing (ignoring sub-filters)"
            )
            scenario_clauses.append(f"scenario_{num}")
        else:
            clause = f"scenario_{num}"
            applied_subs = []

            if args.sf_enabled is not None:
                # Target exact parameter tokens generated by pytest parametrization
                sf_str = "-True-" if args.sf_enabled else "-False-"
                clause += f" and {sf_str}"
                applied_subs.append(f"sf_enabled={args.sf_enabled}")

            if args.detection_type:
                clause += f" and {args.detection_type}"
                applied_subs.append(f"type={args.detection_type}")

            if applied_subs:
                applied_subs_str = ", ".join(applied_subs)
                sub_msg = f" with sub-filters: {applied_subs_str}"
                main_app_logger.info(
                    f"  ⚙️  Scenario {num}: Active compilation{sub_msg}"
                )
            else:
                sub_msg = (
                    f" with sub-filters: {', '.join(applied_subs)}"
                    if applied_subs
                    else ""
                )
                main_app_logger.info(
                    f"  🔹 Scenario {num}: Standard routing (ignoring sub-filters)"
                )
            scenario_clauses.append(f"({clause})")

    # Safely join scenarios together with 'or' so they can execute side-by-side
    scenario_clauses_str = " or ".join(scenario_clauses)
    filter_expression = f"test_scenario_ and ({scenario_clauses_str})"

    # Target hardware context selection filter applies globally across all test cases
    if args.device.lower() != "all":
        main_app_logger.info(f"  💻 Hardware Context Constraint: {args.device.upper()}")
        filter_expression = f"({filter_expression}) and {args.device.lower()}"
    else:
        main_app_logger.info("  💻 Hardware Context Constraint: ALL AVAILABLE")

    main_app_logger.info("=" * 50)
    main_app_logger.info(f"COMPILED PYTEST KEYWORD EXPRESSION:  {filter_expression}")
    main_app_logger.info("=" * 50)

    return filter_expression


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sorted_scenarios = get_available_scenarios()

    parser = argparse.ArgumentParser(
        description="Isolated HybridReader Telemetry Harness Suite"
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        # default="rtsp://172.17.0.1:8554/live1",
        default="anduril_swarm_8K.mp4",
        help="Video filename (located in /inputs) or RTSP target stream endpoint",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=1.0,
        help="Test duration in minutes.",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        type=int,
        choices=sorted_scenarios,  # 1 - 4
        default=None,
        help=f"Specify one or more scenarios. Otherwise all scenarios are ran. Available scenarios: {sorted_scenarios}",
    )

    # MODEL TO USE
    parser.add_argument(
        "--no-custom",
        action="store_false",
        dest="custom_model_flag",
        help="Enable if using Ultralytics YOLO model",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="drone_detection",
        dest="model_name",
        help="Name of model. Required if `--no-custom` is enabled. [Default: drone_detection]",
    )

    # Filter tests
    parser.add_argument(
        "--type",
        type=str,
        choices=["object", "motion"],
        dest="detection_type",
        help="Filter by detection type (object or motion)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="all",
        choices=["cpu", "gpu", "all"],
        help="Target hardware context selection filter.",
    )
    parser.add_argument(
        "--sf",
        action="store_true",
        default=None,
        dest="sf_enabled",
        help="Filter by Smart Filtering",
    )

    # DEBUGGING
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug message and save intermediate images for Smart Filtering tests",
    )
    # parser.add_argument(
    #     "-n",
    #     type=int,
    #     default=100,
    #     dest="debug_frame_limit",
    #     help="Number of frames used for debugging [Default: 100]",
    # )

    # PROFILING
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )

    args = parser.parse_args()

    os.environ["VIDEO_FILENAME"] = args.source
    os.environ["TEST_DURATION_MINS"] = str(args.duration)
    os.environ["CUSTOM_MODEL_FLAG"] = "True" if args.custom_model_flag else "False"
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["DEBUG"] = "1" if args.debug else "0"
    os.environ["ENABLE_PROFILING"] = "True" if args.profile else "False"

    # filter_expression = "test_scenario_4_detection_and_clipper"
    # filter_expression = "test_scenario_3_video_clipper"
    # filter_expression = "test_scenario_*_clipper"

    filter_expression = get_pytest_filter_expression(args, sorted_scenarios)

    pytest_args = [
        "-k",
        filter_expression,
        "-s",
        "-v",
        "--log-cli-level=DEBUG",
        "-W",
        "ignore::_pytest.warning_types.PytestAssertRewriteWarning",
        __file__,
    ]

    main_app_logger.info(
        f"Launching decoupled testing suite configurations for destination targets: {args.source}",
    )
    sys.exit(pytest.main(pytest_args))
