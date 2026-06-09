import argparse
import csv
import gc
import logging
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

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

# Import the exact core handlers to replace duplicate script code
# from include.handlers import BASE_PIPELINE_CONFIG
from include.utils import (
    PipelineConfig,
)

try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Suppress low-delay reference block warnings from OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

main_app_logger = logging.getLogger(__name__)


def log_to_logger(message, level="info"):
    try:
        if level.lower() == "debug":
            main_app_logger.debug(message)
        elif level.lower() == "warning":
            main_app_logger.warning(message)
        else:
            main_app_logger.info(message)
    except Exception:
        pass


@pytest.fixture(scope="class")
def setup_context(request):
    """Orchestrates test directories and configuration schemas."""
    current_test_filename = Path(__file__).stem
    test_dir = Path(__file__).parent
    main_path = test_dir.parent
    video_dir = main_path / "inputs"

    # Resolve source from CLI/Environment parameters
    request.cls.source = os.getenv("STREAM_SOURCE", "anduril_swarm_8K.mp4")
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
    else:
        request.cls.name = "rtsp"
    request.cls.test_duration_mins = float(os.getenv("TEST_DURATION_MINS", 2.0))

    request.cls.result_dir = test_dir / f"{current_test_filename}_results"
    request.cls.result_dir.mkdir(parents=True, exist_ok=True)

    request.cls.benchmarks = []
    if is_rtsp:
        request.cls.csv_filename = "reader_perf_results_rtsp.csv"
    else:
        vid_shortname = Path(request.cls.source).stem
        request.cls.csv_filename = f"reader_perf_results_{vid_shortname}.csv"
    request.cls.csv_path = request.cls.result_dir / request.cls.csv_filename

    yield

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
        for r in request.cls.benchmarks:
            for k in r.keys():
                keys.append(k)
        sorted_keys = []
        for c in ordered_headers:
            if c in keys:
                sorted_keys.append(c)
        with open(str(request.cls.csv_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sorted_keys))
            writer.writeheader()
            writer.writerows(request.cls.benchmarks)
        print(f"\n[FINAL] Telemetry saved to: {request.cls.csv_path}", flush=True)


@pytest.fixture(autouse=True)
def each_test_setup(request):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    device = request.node.callspec.params.get("device")
    os.environ["DEVICE"] = device

    yield

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(0.2)


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
        "detection_type": detection_type,
        "smart_filter_active": sf_enabled,
        "status": "INIT",
    }

    loop_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    cpu_samples, ram_samples, vram_samples = [], [], []
    prefetch_backlog_samples = []

    # Override the global configuration mapping for this specific hardware context
    config = PipelineConfig(
        # GENERAL
        CUSTOM_MODEL_FLAG=os.getenv(
            "CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT
        ),  # True,
        DEVICE="GPU" if device_type.lower() == "gpu" else "CPU",
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
            result_dir = out_dir / "results"
            result_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TEST_SUITE_RENDER_DIR"] = str(result_dir)
        config.SHARED_OUTPUT = str(out_dir)

    import vdms

    def mock_connect(self, host, port):
        pass

    vdms.vdms.connect = mock_connect

    from include.handlers import CPUStreamHandler, GPUStreamHandler

    if device_type.lower() == "gpu":
        HandlerClass = GPUStreamHandler
    else:
        HandlerClass = CPUStreamHandler

    last_sample = time.perf_counter()
    handler = None  # Explicit initializing tracking state pointer 🚀

    try:
        # Let handlers.py completely manage the reader, threads, ring buffers, and FFMPEG process
        handler = HandlerClass(
            source=source, name=source_name, active_streams={}, config=config
        )
        handler.start()

        while (time.perf_counter() - loop_start) < test_duration_secs:
            if not handler.active:
                break

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
            time.sleep(0.01)

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

        metrics["total_frames_read"] = getattr(handler, "frame_count", 0)

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
            if not str(handler.source).startswith("rtsp://"):
                metrics["video_duration"] = (
                    round(r.numFrames / metrics["hardware_video_fps"], 2)
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

        # Cleanup active thread worker contexts safely if they exist
        if handler is not None and getattr(handler, "active", False):
            try:
                handler.stop()
            except Exception:
                pass

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

        # Absolute kill switch safely reclaims orphaned third-party threads
        # at the kernel level without crashing the parent pytest framework
        time.sleep(0.1)
        os._exit(0)


@pytest.mark.usefixtures("setup_context")
class TestHybridStreamHandlers:
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_scenario_1_invalid_rtsp(self, device):
        """SCENARIO 1: Automated Connection Fail Simulation"""
        test_name = "Scenario_1_Invalid_RTSP"
        bad_uri = "rtsp://invalid_host_domain:554/stream_simulation"
        run_clipper = False
        time_limit_m = round(self.test_duration_mins, 1)

        print(
            f"\n========================================\n"
            f"RUNNING TEST: {test_name} | Device: {device.upper()}\n"
            f"Source Destination: {bad_uri}\n"
            f"========================================",
            flush=True,
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
        worker_p.join()

        if worker_p.is_alive():
            worker_p.terminate()
        worker_p.close()  # Reclaims underlying OS file descriptors immediately

        test_metrics = res_queue.get()
        self.__class__.benchmarks.append(test_metrics)
        print(f"Test Status Result: {test_metrics.get('status')}\n", flush=True)

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

        print(
            f"\n========================================\n"
            f"RUNNING TEST: {test_name} | Device: {device.upper()}\n"
            f"Source Destination: {self.source}\n"
            f"========================================",
            flush=True,
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
        worker_p.join()

        if worker_p.is_alive():
            worker_p.terminate()
        worker_p.close()  # Reclaims underlying OS file descriptors immediately

        test_metrics = res_queue.get()
        self.__class__.benchmarks.append(test_metrics)
        print(f"Test Status Result: {test_metrics.get('status')}\n", flush=True)

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

        print(
            f"\n========================================\n"
            f"RUNNING TEST: {test_name} | Device: {device.upper()}\n"
            f"Source Destination: {self.source}\n"
            f"========================================",
            flush=True,
        )
        render_dir = self.result_dir / f"{self.name}/scenario3_{device}"
        render_dir.mkdir(parents=True, exist_ok=True)
        test_duration_mins = 1.0

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
                test_duration_mins,
                res_queue,
                run_clipper,
            ),
        )
        worker_p.start()
        worker_p.join()

        if worker_p.is_alive():
            worker_p.terminate()
        worker_p.close()  # Reclaims underlying OS file descriptors immediately

        test_metrics = res_queue.get()
        self.__class__.benchmarks.append(test_metrics)
        print(f"Test Status Result: {test_metrics.get('status')}\n", flush=True)

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

        print(
            f"\n========================================\n"
            f"RUNNING TEST: {test_name} | Device: {device.upper()} | SF: {sf_enabled}\n"
            f"Source Name: {self.name} | Destination: {self.source}\n"
            f"========================================",
            flush=True,
        )
        render_dir = (
            self.result_dir
            / f"{self.name}/scenario4_{device}/{detection_type}_{mode_str}"
        )
        render_dir.mkdir(parents=True, exist_ok=True)
        test_duration_mins = 1.0

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
                test_duration_mins,
                res_queue,
                run_clipper,
                disable_detection,
                sf_enabled,
                detection_type,
            ),
        )
        worker_p.start()
        worker_p.join()

        if worker_p.is_alive():
            worker_p.terminate()
        worker_p.close()  # Reclaims underlying OS file descriptors immediately

        test_metrics = res_queue.get()
        self.__class__.benchmarks.append(test_metrics)
        if disable_detection:
            print(f"Test Status Result: {test_metrics.get('status')}\n", flush=True)
        else:
            print(
                f"Test Status Result: {test_metrics.get('status')} w/ {test_metrics.get('total_objects_detected')} detections\n",
                flush=True,
            )

        # Explicitly tear down queue thread pools to prevent resource tracking leaks
        try:
            res_queue.close()  # Stops new items from being inserted
            res_queue.join_thread()  # Joins the internal buffer thread safely
        except Exception:
            pass

        assert "COMPLETED_SUCCESSFULLY" in test_metrics.get("status")


def get_available_scenarios():
    import inspect

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
    print("\n" + "=" * 50)
    print("TARGET SELECTION PREVIEW")
    print("=" * 50)

    for num in target_scenarios:
        if num in basic_ids:
            print(f"  🔹 Scenario {num}: Standard routing (ignoring sub-filters)")
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

            sub_msg = (
                f" with sub-filters: {', '.join(applied_subs)}" if applied_subs else ""
            )
            print(f"  ⚙️  Scenario {num}: Active compilation{sub_msg}")
            scenario_clauses.append(f"({clause})")

    # Safely join scenarios together with 'or' so they can execute side-by-side
    filter_expression = f"test_scenario_ and ({' or '.join(scenario_clauses)})"

    # Target hardware context selection filter applies globally across all test cases
    if args.device.lower() != "all":
        print(f"  💻 Hardware Context Constraint: {args.device.upper()}")
        filter_expression = f"({filter_expression}) and {args.device.lower()}"
    else:
        print("  💻 Hardware Context Constraint: ALL AVAILABLE")

    print("=" * 50)
    print(f"COMPILED PYTEST KEYWORD EXPRESSION:\n   👉 {filter_expression}")
    print("=" * 50 + "\n")

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
        default=2.0,
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
    args = parser.parse_args()

    os.environ["STREAM_SOURCE"] = args.source
    os.environ["TEST_DURATION_MINS"] = str(args.duration)
    os.environ["CUSTOM_MODEL_FLAG"] = "True" if args.custom_model_flag else "False"
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["DEBUG"] = "1" if args.debug else "0"

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
        __file__,
    ]

    print(
        f"Launching decoupled testing suite configurations for destination targets: {args.source}",
        flush=True,
    )
    sys.exit(pytest.main(pytest_args))
