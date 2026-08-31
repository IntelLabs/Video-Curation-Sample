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
import asyncio
import csv
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
)
from include.default_configs import (
    CUSTOM_MODEL_FLAG_DEFAULT,
    DEBUG_DEFAULT,
    MODEL_NAME_DEFAULT,
    THRESHOLD_VALUE,
)
from include.detectors import GeneralObjectDetector
from include.handlers import (
    get_test_handler,
)
from include.utils import (
    PipelineConfig,
    str2bool,
)

# objgraph = install_and_load_pip_package("objgraph", attribute_name=None)
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
    # gt_enabled = test_args["gt_enabled"]

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
        instance, _ = get_test_handler(TestModel(), device)

        instance.source = init_args["source"]
        instance.name = init_args["name"]
        instance.result_dir = Path(init_args["result_dir"])
        instance.active_streams = init_args["active_streams"]
        instance.__class__.benchmarks = init_args["benchmarks"]  # Sandbox the metrics

        # vid_dir = instance.result_dir / device
        # vid_dir.mkdir(parents=True, exist_ok=True)
        # os.environ["TEST_SUITE_RENDER_DIR"] = str(vid_dir)

        # if instance.source.startswith("rtsp"):
        #     short_name = "rtsp"
        # else:
        #     short_name = Path(instance.source).stem

        # config definition
        model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
        config = PipelineConfig(
            SHARED_OUTPUT=str(instance.result_dir),  # defined in context
            CUSTOM_MODEL_FLAG=os.getenv("CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT),
            DEVICE=device.upper(),
            OMIT_DETECTIONS_FLAG=True,
            TEST_MODE=True,
            DEBUG=os.getenv("DEBUG", DEBUG_DEFAULT),
            DEBUG_FRAME_LIMIT=int(os.getenv("DEBUG_FRAME_LIMIT", 100)),
            ENABLE_QUERYING=False,
            MODEL_NAME=model_name,
            SMART_FILTERING_ENABLED=sf_enabled,
            THRESHOLD_VALUE=int(os.getenv("THRESHOLD_VALUE", THRESHOLD_VALUE)),
            DETECTION_TYPE=detection_type,
        )

        # INITIALIZE CLASS (mimic DeviceBaseHandler.__init__) ------------------------------
        instance.is_rtsp = str(instance.source).startswith("rtsp:/")
        instance.active = True
        instance.config = config

        # kwarg definition
        instance._testMethodName = f"{model_name}_{detection_type}_{device}"
        # instance.video_output_name = (
        #     f"{instance._testMethodName}_{short_name}.mp4"
        # )

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

        # def profiler_fn():
        #     profiler = None
        #     try:
        #         if str2bool(os.getenv("ENABLE_PROFILING", "False")):
        #             Profiler = install_and_load_pip_package(
        #                 "pyinstrument", attribute_name="Profiler"
        #             )

        #             profiler = Profiler(interval=0.005)  # 5ms sampling interval

        #             # Telling the statistical sampler to skip recording exception blocks completely
        #             # stops stack_sampler.py from ballooning RAM over long production runs.
        #             if hasattr(profiler, "_sampler") and profiler._sampler:
        #                 profiler._sampler.trace_exceptions = False

        #             profiler.start()

        #         # orig_fn(profiler)
        #         instance.run_realtime_inference(
        #             sf_enabled=instance.config.sf_enabled,
        #             profiler=profiler,
        #             gt_enabled=gt_enabled,
        #         )

        #         if str2bool(os.getenv("ENABLE_PROFILING", "False")):
        #             # 2. Redirect standard error to the filter trap right before report compilation
        #             original_stderr = sys.stderr
        #             sys.stderr = ResourceTrackerFilter(original_stderr)

        #             try:
        #                 # Force standard stdout to flush out any lingering teardown messages
        #                 # BEFORE pyinstrument dumps its massive ASCII tree block.
        #                 sys.stdout.flush()

        #                 # main_app_logger.info(profiler.output_text(color=True))
        #                 # profiler.main_app_logger.info(color=True)
        #                 # prof_output = profiler.output_text(color=True)
        #                 # main_app_logger.info(
        #                 #     f"\n=== LATENCY BREAKDOWN FOR {self.name} ({device}) ===\n{prof_output}\n",
        #                 #
        #                 # )

        #                 # Save a clean, interactive tree map for visual analysis
        #                 output_html_path = instance.output_path.replace(
        #                     ".mp4", "_profile.html"
        #                 )
        #                 # output_html_path = f"/tmp/profile_{video_name}_{device}.html"
        #                 profiler.write_html(output_html_path)
        #                 main_app_logger.info(
        #                     f"[PROFILER] Performance tree map exported to {output_html_path}",
        #                 )

        #             finally:
        #                 sys.stderr = original_stderr
        #                 if "profiler" in locals():
        #                     try:
        #                         # Force the Python interpreter to detach pyinstrument's sampling hooks
        #                         sys.setprofile(None)

        #                         # Completely decouple internal statistical sessions to drop C-heap frames
        #                         if hasattr(profiler, "_last_session"):
        #                             profiler._last_session = None
        #                         if hasattr(profiler, "last_session"):
        #                             profiler.last_session = None

        #                         # Forcibly clear out internal memoryview strings caching tree metrics
        #                         if (
        #                             hasattr(profiler, "session")
        #                             and profiler.session is not None
        #                         ):
        #                             if hasattr(profiler.session, "frame_groups"):
        #                                 profiler.session.frame_groups = None
        #                             if hasattr(profiler.session, "samples"):
        #                                 profiler.session.samples = []

        #                             # Purge compiled tree metrics structures
        #                             profiler.session = None
        #                         del profiler
        #                     except Exception:
        #                         pass

        #                     # Trigger an immediate native Linux heap compression pass
        #                     # This grabs the newly abandoned pyinstrument C-heap blocks and
        #                     # flushes them to the OS before the fixture assessment snapshot fires!
        #                     gc.collect()
        #                     try:
        #                         libc = ctypes.CDLL("libc.so.6")
        #                         libc.malloc_trim(0)
        #                     except Exception:
        #                         pass

        #     except Exception:
        #         traceback.print_exc()

        # if str2bool(os.getenv("ENABLE_PROFILING", "False")) and hasattr(
        #     instance, "process_thread"
        # ):
        #     # Re-initialize the thread context using our safe profile wrapper proxy
        #     instance.process_thread = threading.Thread(target=profiler_fn, daemon=True)

        # instance.VIDEO_GT_DETAILS = None
        # instance.duration_target = 30

        # instance.start()

        # while instance.active or not instance._is_stopped:
        #     time.sleep(0.25)
        #     if getattr(instance, "status", None) == "DONE":
        #         break

        #     if hasattr(instance, "process_thread") and instance.process_thread is not None:
        #         if not instance.process_thread.is_alive():
        #             main_app_logger.info(
        #                 "[TEST HARNESS] Background worker exited. Breaking loop.",
        #             )
        #             break

        # instance.stop_threads(["process_thread"])

        #  Run the actual model loader
        instance.processor = GeneralObjectDetector(
            instance.config,
            device=instance.device_input,
            timer_enabled=False,
            resize_hw=(instance.resize_h, instance.resize_w),
            frame_hw=(instance.frame_height, instance.frame_width),
            target_fps=instance.target_fps,
            result_dir=instance.result_dir,
            run_name=instance._testMethodName,
            debug_frame_limit=-1,
        )

        total_session_start = time.perf_counter()

        print("Running model...", flush=True)
        results = instance.processor.model.predict(
            source=instance.source,
            imgsz=(instance.frame_height, instance.frame_width),
            batch=1,
            device=instance.config.device_input,
            stream=any(x in instance.source for x in [".mp4", "rtsp"]),
            conf=instance.config.DETECTION_THRESHOLD,
            iou=instance.config.IOU_THRESHOLD,
            show=False,
            save=True,
            project=str(instance.config.SHARED_OUTPUT),
            name="predict_results",
            exist_ok=True,  # overwrite if folder exists
            data={
                "names": {
                    i: name for i, name in enumerate(instance.processor.label_source)
                }
            },
        )

        frame_cnt = 0
        total_model_preprocess = 0.0
        total_model_inference = 0.0
        total_model_postprocess = 0.0
        for result in results:
            frame_cnt += 1
            # Each iteration computes and yields the next frame's latency
            total_model_preprocess += result.speed.get("preprocess", 0.0)
            total_model_inference += result.speed.get("inference", 0.0)
            total_model_postprocess += result.speed.get("postprocess", 0.0)

        # Capture the true real-world duration across all operational processing layers
        real_world_latency_ms = (time.perf_counter() - total_session_start) * 1000

        print("Summarizing run...", flush=True)
        instance._finalize_benchmarks(
            real_world_latency_ms,
            frame_cnt,
            total_model_preprocess,
            total_model_inference,
            total_model_postprocess,
        )
        instance.stop()

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
@pytest.fixture(scope="class")
def setup_context(request):
    """Replaces setUpClass: Runs once per test class."""
    # Initialize shared paths/results
    current_test_filename = Path(__file__).stem
    test_dir = Path(__file__).parent
    main_path = test_dir.parent
    video_dir = main_path / "inputs"

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

    model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    request.cls.result_dir = test_dir / f"{current_test_filename}_results/{model_name}"
    request.cls.result_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark statistics
    request.cls.benchmarks = []
    request.cls.csv_filename = f"model_benchmarks_{request.cls.name}.csv"
    request.cls.csv_path = request.cls.result_dir / request.cls.csv_filename

    # Initialize class vars
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
        if results:
            keys = results[0].keys()

            with open(str(request.cls.csv_path), "w", newline="") as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(results)

            print(
                f"\n[FINAL] Benchmarks saved to {request.cls.csv_filename}", flush=True
            )


@pytest.mark.usefixtures("setup_context")
class TestModel(BaseTest):
    """
    Pytest runner that spawns the isolated worker, waits for completion,
    and merges the returned metrics into the main class for CSV/Chart generation.
    """

    benchmarks = []  # Class-level attribute required by _finalize_benchmarks

    @pytest.mark.parametrize("device", ["gpu", "cpu"])
    def test_model(self, device):
        # Pull the values dynamically assigned by the setup_context fixture
        init_args = {
            "source": self.__class__.source,
            "name": self.__class__.name,
            "result_dir": str(self.__class__.result_dir),
            "active_streams": {},
            "benchmarks": self.__class__.benchmarks,
        }

        os.environ["DEVICE"] = device
        detection_type = "object"  # request.node.callspec.params.get("detection_type")
        sf_enabled = False  # request.node.callspec.params.get("sf_enabled")
        gt_enabled = False

        test_args = {
            "device": device,
            "detection_type": detection_type,
            "sf_enabled": sf_enabled,
            "gt_enabled": gt_enabled,
        }

        main_app_logger.info(
            f"\n{'=' * 60}\n"
            f"[TEST HARNESS] Spawning isolated high-speed process for {self.__class__.name} (SF: {sf_enabled})...\n"
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
                    f"[TEST HARNESS] Worker returned successfully. Model Est. FPS: {metrics.get('Model Est. FPS')}"
                )

                # Basic functionality assertions
                assert metrics is not None, "Metrics dictionary should not be None."
                assert int(metrics.get("Frames Processed", 0)) > 0, (
                    "No frames were written to output."
                )
        else:
            pytest.fail("Worker process died unexpectedly without returning metrics.")

    # HELPERS --------------------------------------------
    def _finalize_benchmarks(
        self,
        real_world_latency_ms,
        n_frames,
        total_model_preprocess,
        total_model_inference,
        total_model_postprocess,
    ):
        """Aggregates metrics and adds them to the results list."""
        total_model_ms = (
            total_model_preprocess + total_model_inference + total_model_postprocess
        )

        duration_s = n_frames / self.input_fps if self.input_fps > 0 else 0

        real_latency_s = real_world_latency_ms / 1000.0
        real_est_fps = n_frames / real_latency_s if real_latency_s > 0 else 0
        model_fps = n_frames / (total_model_ms / 1000.0) if total_model_ms > 0 else 0

        # Construct dictionary block matching test_detections structure
        stats = {
            "Test Name": self._testMethodName,
            "Detection Type": self.config.DETECTION_TYPE,
            "Device": self.device,
            "Smart Filtering": "Enabled" if self.config.sf_enabled else "Disabled",
            "Video": self.name,
            "Video Duration (s)": f"{duration_s:.4f}",
            "Video FPS": f"{self.input_fps:.2f}",
            "Pipeline Latency (s)": f"{real_latency_s:.2f}",
            "Frames Processed": n_frames,
            "Real Est. FPS": f"{real_est_fps:.2f}",
            "Model Est. FPS": f"{model_fps:.2f}",
            "Model Avg Pre-processing (ms)": f"{total_model_preprocess / n_frames:.2f}",
            "Model Avg Inference (ms)": f"{total_model_inference / n_frames:.2f}",
            "Model Avg Post-processing (ms)": f"{total_model_postprocess / n_frames:.2f}",
        }
        self.__class__.benchmarks.append(stats)
        print(stats, flush=True)

        print(f"\n[{self._testMethodName}] Latency: {real_latency_s:.2f} sec")
        print(f"\n[{self._testMethodName}] Real Est. FPS: {real_est_fps:.2f}")
        print(
            f"\n[{self._testMethodName}] Model Est. FPS: {model_fps:.2f} ({n_frames} frames)"
        )

    def setup_threads(self):
        """Overrides handlers.py to bind threads dynamically to the test instance."""
        pass

    def start(self):
        """
        Starts the decoupled ingestion and inference threads in the correct order.
        """
        # PRE-SYNC: Ensure GPU is idle before timing starts
        if self.device_input == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Start the hardware-decoupled reader first
        # self.reader.start()

        return self


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    # TEST ARGUMENTS
    parser = argparse.ArgumentParser(description="Run Video Detection Pipeline Tests")
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default="anduril_swarm_8K.mp4",
        help="Video filename (located in /inputs) or RTSP target stream endpoint",
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
        choices=["cpu", "gpu"],
        help="Filter by device (cpu or gpu)",
    )

    args = parser.parse_args()

    # UPDATE ENVIRONMENTAL VARIABLES
    os.environ["VIDEO_FILENAME"] = args.source
    os.environ["CUSTOM_MODEL_FLAG"] = "True" if args.custom_model_flag else "False"
    os.environ["MODEL_NAME"] = args.model_name
    # os.environ["DEBUG"] = "1" if args.debug else "0"
    # os.environ["DEBUG_FRAME_LIMIT"] = str(args.debug_frame_limit)

    # detection_type, device, sf_enabled
    run_args = []
    # if args.detection_type:
    #     run_args.append(args.detection_type)
    if args.device:
        run_args.append(args.device)
    # if args.sf_enabled:
    #     run_args.append(str(args.sf_enabled))

    # PYTEST COMMAND
    pytest_args = [
        "-s",
        "-v",
        "--log-cli-level=DEBUG",
        "-W",
        "ignore::_pytest.warning_types.PytestAssertRewriteWarning",
        __file__,
    ]  # -s -v --log-cli-level=DEBUG
    if run_args:
        pytest_args.extend(["-k", " and ".join(run_args)])

    print(f"Launching tests for {args.source}")

    sys.exit(pytest.main(pytest_args))
