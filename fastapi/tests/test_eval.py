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

import argparse
import asyncio
import csv
import ctypes
import faulthandler
import gc
import multiprocessing as mp
import sys
import threading
import time
import traceback
import tracemalloc
from pathlib import Path

import cv2
import psutil
import torch

# import torch

# Retrieve repo packages
REPO_DIR = str(Path(__file__).parent.parent)
sys.path.insert(1, REPO_DIR)
from base_test import (
    BaseTest,
    download_eval_data,
    fps_comparison_chart,
)
from include.default_configs import (
    CUSTOM_MODEL_FLAG_DEFAULT,
    DEBUG_DEFAULT,
    MODEL_NAME_DEFAULT,
    THRESHOLD_VALUE,
)
from include.handlers import (
    get_test_handler,
)
from include.utils import (
    PipelineConfig,
    ResourceTrackerFilter,
    install_and_load_pip_package,
    str2bool,
)
from metrics import DeviceAgnosticOnTheFlyEvaluator

gdown = install_and_load_pip_package("gdown", attribute_name=None)


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
# target_width, target_height = 7680, 4320
STATE_CAPTURE = False

# Force Python's multiprocessing layer to quiet tracking cleanup race conditions
os.environ["PYTHONWARNINGS"] = "ignore"
sys.warnoptions.append("ignore")


# =========================================================================
# ISOLATED BACKGROUND WORKER
# =========================================================================


def isolated_detection_worker(init_args, test_args, res_queue):
    device = test_args["device"]
    detection_type = test_args["detection_type"]
    sf_enabled = test_args["sf_enabled"]
    video_name = test_args["video_name"]
    gt_enabled = test_args["gt_enabled"]

    if device == "gpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    gc.collect()

    # Establish an accurate post-initialization hardware baseline
    if str2bool(os.getenv("ENABLE_PROFILING", "False")):
        tracemalloc.start()

        start_allocated, start_reserved = 0, 0
        if device == "gpu" and torch.cuda.is_available():
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

    try:
        instance, _ = get_test_handler(TestEvalSmartFilteringDetections(), device)

        test_dir = Path(__file__).parent
        video_dir = test_dir / "eval_data/Anti-UAV-Tracking-V0-8K"
        vid_source = video_dir / f"{video_name}/{video_name}.mp4"

        if not vid_source.exists():
            _ = download_eval_data([video_name])

        instance.source = str(vid_source)
        instance.name = vid_source.stem
        instance.result_dir = Path(init_args["result_dir"])
        instance.active_streams = init_args["active_streams"]
        instance.__class__.benchmarks = init_args["benchmarks"]  # Sandbox the metrics

        vid_dir = instance.result_dir / device
        vid_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TEST_SUITE_RENDER_DIR"] = str(vid_dir)

        # if instance.source.startswith("rtsp"):
        #     short_name = "rtsp"
        # else:
        #     short_name = Path(instance.source).stem

        # config definition
        config = PipelineConfig(
            SHARED_OUTPUT=str(instance.result_dir),  # defined in context
            CUSTOM_MODEL_FLAG=os.getenv("CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT),
            DEVICE=device.upper(),
            OMIT_DETECTIONS_FLAG=True,
            TEST_MODE=True,
            DEBUG=os.getenv("DEBUG", DEBUG_DEFAULT),
            DEBUG_FRAME_LIMIT=int(os.getenv("DEBUG_FRAME_LIMIT", 100)),
            ENABLE_QUERYING=False,
            MODEL_NAME=os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT),
            SMART_FILTERING_ENABLED=sf_enabled,
            THRESHOLD_VALUE=int(os.getenv("THRESHOLD_VALUE", THRESHOLD_VALUE)),
            DETECTION_TYPE=detection_type,
        )

        # INITIALIZE CLASS (mimic DeviceBaseHandler.__init__) ------------------------------
        instance.evaluator = DeviceAgnosticOnTheFlyEvaluator(device=device)

        instance.is_rtsp = str(instance.source).startswith("rtsp:/")
        instance.active = True
        instance.config = config

        # kwarg definition
        instance._testMethodName = (
            f"{video_name}_sf_{detection_type}_{device}"
            if sf_enabled
            else f"{video_name}_yolo_{detection_type}_{device}"
        )
        instance.video_output_name = f"{instance._testMethodName}.mp4"
        instance.gt_enabled = gt_enabled

        instance.loop = asyncio.get_event_loop()
        instance.frame_ready_event = asyncio.Event()
        instance._is_stopped = False
        instance._stop_lock = threading.Lock()  # Local lock for this instance
        instance.main_startup_event = mp.Event()

        instance.device = instance.config.DEVICE
        instance.device_input = instance.config.device_input
        instance.disp_w, instance.disp_h = instance.config.DISPLAY_FRAME_SIZE
        instance.resize_h, instance.resize_w = [
            instance.config.MODEL_H,
            instance.config.MODEL_W,
        ]

        instance.setup_reader(
            instance.config.TARGET_FPS,
            instance.config.CLIP_DURATION,
            startup_event=instance.main_startup_event,
        )
        instance.initialize_variables()
        instance.setup_threads()
        instance.last_heartbeat = time.perf_counter()

        if STATE_CAPTURE:
            main_app_logger.info(
                "[DIAGNOSTIC] Registering system state baseline layout..."
            )
            instance.baseline_before_start = instance.capture_state_snapshot()

        def orig_fn(profiler):
            return instance.run_realtime_inference(
                sf_enabled=instance.config.sf_enabled,
                profiler=profiler,
                gt_enabled=gt_enabled,
            )

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

                orig_fn(profiler)

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
                        output_html_path = instance.output_path.replace(
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
            instance, "process_thread"
        ):
            # Re-initialize the thread context using our safe profile wrapper proxy
            instance.process_thread = threading.Thread(target=profiler_fn, daemon=True)

        if gt_enabled:
            instance.VIDEO_GT_DETAILS = download_eval_data([instance.name])
        else:
            instance.VIDEO_GT_DETAILS = None

        instance.start()

        while instance.active or not instance._is_stopped:
            time.sleep(0.25)
            if getattr(instance, "status", None) == "DONE":
                break

            if (
                hasattr(instance, "process_thread")
                and instance.process_thread is not None
            ):
                if not instance.process_thread.is_alive():
                    main_app_logger.info(
                        "[TEST HARNESS] Background worker exited. Breaking loop.",
                    )
                    break

        instance.stop_threads(["process_thread"])

        # Force early hardware driver sweep before unbinding threads
        if instance.device_input == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()

        try:
            if STATE_CAPTURE:
                # --- CAPTURE STATE POINT B (Right before cleanup) ---
                main_app_logger.info(
                    "[DIAGNOSTIC] Gathering active execution workspace layouts..."
                )
                state_after_stop = instance.capture_state_snapshot()

                # 3. Print the granular delta analysis out to your terminal screen
                new_keys_generated, mutated_keys, static_keys = (
                    instance.print_lifecycle_delta(
                        instance.baseline_before_start,
                        state_after_stop,
                        return_keys=True,
                    )
                )
                # self.config.DEBUG_FLAG and
                if len(new_keys_generated) > 0 or len(mutated_keys) > 0:
                    # Inform the blueprint engine to eliminate every difference found between Point A and B
                    instance.stop_blueprint_executor(new_keys_generated, mutated_keys)
        except Exception:
            traceback.print_exc()

        assert instance.status == "DONE"

        # TEARDOWN
        instance.execute_teardown()

        if str2bool(os.getenv("ENABLE_PROFILING", "False")):
            gc.collect()
            if (
                device == "gpu" and torch.cuda.is_available()
            ):  # self.device_input == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            instance.assess_memory(
                device, instance.name, start_allocated, start_reserved
            )

        if len(instance.__class__.benchmarks) > 0:
            final_metrics = instance.__class__.benchmarks[-1]
            res_queue.put({"status": "success", "metrics": final_metrics})
        else:
            res_queue.put({"status": "error", "error": "No benchmarks generated."})

    except Exception as e:
        res_queue.put(
            {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        )
        res_queue.put(
            {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        )


# =========================================================================
# PYTEST TEST HARNESS
# =========================================================================
def pytest_generate_tests(metafunc):
    """
    Pytest Hook: Generates a complete cross-product matrix
    of (video_name x device) dynamically during collection phase.
    """
    if (
        "video_name" in metafunc.fixturenames
        and "device" in metafunc.fixturenames
        and "detection_type" in metafunc.fixturenames
        and "sf_enabled" in metafunc.fixturenames
    ):
        # Define video targets explicitly during collection
        video_names = [f"video{i:02d}" for i in range(9, 21)]
        # video_names = ["video16", "video17", "video12"]  # , "video18"]
        # video_names = ["video17"]  # DEBUG: worst perf
        # video_names = ["video12"]  # DEBUG: worst perf
        # video_names = ["video09"]

        # Read device target filters passed from your CLI parser args block
        # Falls back to both types if running globally via '--device all'
        device_input = os.getenv("TEST_SUITE_DEVICE_FILTER", "all")
        devices = ["cpu", "gpu"] if device_input == "all" else [device_input]

        type_input = os.getenv("TEST_SUITE_DETECTION_FILTER", "all")
        detection_types = ["motion", "object"] if type_input == "all" else [type_input]

        sf_input = os.getenv("TEST_SUITE_SF_FILTER", "None")
        sf_flags = [True, False] if sf_input == "None" else [str2bool(sf_input)]

        # Tell Pytest to create an individual test instance for each discovered name
        metafunc.parametrize("device", devices)
        metafunc.parametrize("sf_enabled", sf_flags)
        metafunc.parametrize("detection_type", detection_types)
        metafunc.parametrize("video_name", video_names)


@pytest.fixture(scope="class")
def setup_context(request):
    """Replaces setUpClass: Runs once per test class."""
    current_test_filename = Path(__file__).stem
    test_dir = Path(__file__).parent

    # model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    # Handler.__init__ (main items)
    request.cls.is_rtsp = False

    # model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    request.cls.result_dir = test_dir / f"{current_test_filename}_drone_results"
    request.cls.result_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark statistics
    request.cls.benchmarks = []
    request.cls.csv_filename = "drone_eval.csv"
    request.cls.csv_path = request.cls.result_dir / request.cls.csv_filename

    # request.cls.active = True
    request.cls.active_streams = {}

    # RUN ALL PARAMETERIZED TESTS ----------------------------------------
    yield

    # FINAL CSV EXPORT  --------------------------------------------------
    if request.cls.benchmarks:
        # Filter and exclude rows that were interrupted or failed initialization due to an early pytest skip
        request.cls.benchmarks = [
            r for r in request.cls.benchmarks if r and "Test Name" in r
        ]
        results = request.cls.benchmarks
        for row in results:
            if "gpu" in row["Test Name"].lower():
                match_name = row["Test Name"].replace("gpu", "cpu")
                cpu_row = next(
                    (r for r in results if r["Test Name"] == match_name), None
                )
                if cpu_row:
                    gpu_fps = float(row["Pipeline FPS (Target frames)"])
                    cpu_fps = float(cpu_row["Pipeline FPS (Target frames)"])
                    speedup = (gpu_fps / cpu_fps) if cpu_fps > 0 else 0
                    row["Pipeline Speedup vs CPU"] = f"{speedup:.2f}x"
                else:
                    row["Pipeline Speedup vs CPU"] = "N/A"
            else:
                row["Pipeline Speedup vs CPU"] = "Baseline (CPU)"

        if results:
            # # Define video targets explicitly during collection
            # video_names = [f"video{i:02d}" for i in range(1, 21)]
            # ALL_VIDEO_GT_DETAILS = download_eval_data(video_names)

            keys = results[0].keys()

            with open(str(request.cls.csv_path), "w", newline="") as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(results)
            main_app_logger.info(f"[FINAL] Benchmarks saved to {request.cls.csv_path}")

            main_app_logger.info("=" * 125)
            main_app_logger.info(
                f"{'Test Name':<25} | {'mAP_10':<5} | {'mAP_10_95':<10} | {'Pipeline FPS (Target)':<21} | {'Avg Frame Reading (ms)':<22} | {'Pipeline Speedup vs CPU':<15}",
            )
            main_app_logger.info("-" * 125)

            for r in results:
                main_app_logger.info(
                    f"{r['Test Name']:<25} | {r['mAP_10']:0.4f} | {r['mAP_10_95']:0.4f}     | {r['Pipeline FPS (Target frames)']:<21} | {r['Avg Frame Reading (ms)']:<22} | {r.get('Pipeline Speedup vs CPU', 'N/A'):<10}"
                )
            main_app_logger.info("=" * 125)

            chart_path = (
                request.cls.result_dir
                / f"{request.cls.csv_filename.replace('.csv', '')}_pipelineFPS.png"
            )
            fps_comparison_chart(
                chart_path, results, fps_key="Pipeline FPS (Video frames)"
            )

            chart_path = (
                request.cls.result_dir
                / f"{request.cls.csv_filename.replace('.csv', '')}_pipelineFPS_target.png"
            )
            fps_comparison_chart(
                chart_path, results, fps_key="Pipeline FPS (Target frames)"
            )

            chart_path = (
                request.cls.result_dir
                / f"{request.cls.csv_filename.replace('.csv', '')}_sfdetectFPS_target.png"
            )
            fps_comparison_chart(chart_path, results, fps_key="SF/Detection FPS")

            chart_path = (
                request.cls.result_dir
                / f"{request.cls.csv_filename.replace('.csv', '')}_displayFPS_target.png"
            )
            fps_comparison_chart(chart_path, results, fps_key="Display FPS")

        else:
            main_app_logger.info(
                "[WARN] Benchmarking matrix empty. Skipping final CSV and chart exports."
            )


@pytest.mark.usefixtures("setup_context")
class TestEvalSmartFilteringDetections(BaseTest):
    """
    Pytest runner that spawns the isolated worker, waits for completion,
    and merges the returned metrics into the main class for CSV/Chart generation.
    """

    benchmarks = []  # Class-level attribute required by _finalize_benchmarks

    # @pytest.mark.parametrize("device", ["gpu", "cpu"])
    # @pytest.mark.parametrize("sf_enabled", [True, False])
    # @pytest.mark.parametrize("detection_type", ["object", "motion"])
    def test_eval(
        self, device, detection_type, sf_enabled, video_name, gt_enabled=True
    ):
        if detection_type == "motion" and not sf_enabled:
            pytest.skip(
                "Pure YOLO mode is structurally invalid for detection_type 'motion'.\n"
            )

        # Pull the values dynamically assigned by the setup_context fixture
        init_args = {
            # "source": self.__class__.source,
            # "name": self.__class__.name,
            "result_dir": str(self.__class__.result_dir),
            "active_streams": {},
            "benchmarks": self.__class__.benchmarks,
        }

        test_args = {
            "device": device,
            "detection_type": detection_type,
            "sf_enabled": sf_enabled,
            "video_name": video_name,
            "gt_enabled": gt_enabled,
        }

        main_app_logger.info(
            f"\n{'=' * 60}\n"
            f"[TEST HARNESS] Spawning isolated high-speed process for {video_name} (SF: {sf_enabled})...\n"
            f"{'=' * 60}"
        )

        # Spawn a pristine Python process (identical to test_pipeline.py's stream_worker)
        ctx = mp.get_context("spawn")
        res_queue = ctx.Queue()

        worker_p = ctx.Process(
            target=isolated_detection_worker, args=(init_args, test_args, res_queue)
        )

        worker_p.start()
        worker_p.join()  # Block Pytest until the isolated pipeline finishes

        # Extract and merge results
        if not res_queue.empty():
            result = res_queue.get()

            if result["status"] == "error":
                pytest.fail(
                    f"Pipeline crashed in worker process:\n{result.get('error')}\n{result.get('traceback')}"
                )
            else:
                metrics = result["metrics"]

                self.__class__.benchmarks.append(metrics)

                main_app_logger.info(
                    f"[TEST HARNESS] Worker returned successfully. Display FPS: {metrics.get('Display FPS')}"
                )

                # Basic functionality assertions
                assert metrics is not None, "Metrics dictionary should not be None."
                assert int(metrics.get("Output Frames", 0)) > 0, (
                    "No frames were written to output."
                )
        else:
            pytest.fail("Worker process died unexpectedly without returning metrics.")


# =========================================================================
# MAIN
# =========================================================================
def get_pytest_filter_expression(args):
    main_app_logger.info("=" * 50)
    main_app_logger.info("TARGET SELECTION PREVIEW")
    main_app_logger.info("=" * 50)

    filter_expression = Path(__file__).stem
    # applied_subs = []

    if args.sf_enabled is not None:
        # Target exact parameter tokens generated by pytest parametrization
        sf_str = "-True-" if args.sf_enabled else "-False-"
        filter_expression += f" and {sf_str}"
        # applied_subs.append(f"sf_enabled={args.sf_enabled}")

    if args.detection_type:
        filter_expression += f" and {args.detection_type}"

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
    # TEST ARGUMENTS
    parser = argparse.ArgumentParser(description="Run Video Detection Pipeline Tests")
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default="anduril_swarm_8K.mp4",
        help="Video filename (located in /inputs)",
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
        "--device",
        type=str,
        default="all",
        choices=["cpu", "gpu", "all"],
        help="Filter by device (cpu or gpu)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["object", "motion"],
        dest="detection_type",
        help="Filter by detection type (object or motion)",
    )
    parser.add_argument(
        "--sf",
        action="store_true",
        default=None,
        dest="sf_enabled",
        help="Filter by Smart Filtering",
    )
    parser.add_argument(
        "--no-sf",
        action="store_false",
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
    parser.add_argument(
        "-n",
        type=int,
        default=100,
        dest="debug_frame_limit",
        help="Number of frames used for debugging [Default: 100]",
    )

    # PROFILING
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )

    args = parser.parse_args()

    # UPDATE ENVIRONMENTAL VARIABLES
    os.environ["VIDEO_FILENAME"] = args.source
    os.environ["CUSTOM_MODEL_FLAG"] = "True" if args.custom_model_flag else "False"
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["DEBUG"] = "1" if args.debug else "0"
    os.environ["DEBUG_FRAME_LIMIT"] = str(args.debug_frame_limit)
    os.environ["ENABLE_PROFILING"] = "True" if args.profile else "False"

    # detection_type, device, sf_enabled
    filter_expression = get_pytest_filter_expression(args)

    # PYTEST COMMAND
    pytest_args = [
        "-k",
        filter_expression,
        "-s",
        "-v",
        # "--log-cli-level=DEBUG",
        "-W",
        "ignore:Exception ignored in.*SharedMemory.__del__:UserWarning",
        # Target the exact module rewrite warning path inside the configuration framework
        "-W",
        "ignore:Module already imported so cannot be rewritten; anyio:_pytest.warning_types.PytestAssertRewriteWarning",
        # Bypass Pytest's internal proxy
        "--capture=no",
        __file__,
    ]

    # main_app_logger.info(f"Launching tests for {args.source}")

    sys.exit(pytest.main(pytest_args))
