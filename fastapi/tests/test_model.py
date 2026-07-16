import argparse
import csv
import gc
import inspect
import logging
import os
import sys
import time
import types
from pathlib import Path

import pytest
import tensorrt as trt
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
    DeviceBaseHandler,
    GPUStreamHandler,
)
from include.models import get_model
from include.utils import PipelineConfig

try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")


DEBUG_FLAG_DEFAULT = True if DEBUG_DEFAULT == "1" else False
os.environ["OMP_NUM_THREADS"] = "1"

force_export = False


@pytest.fixture(scope="class")
def setup_context(request):
    """Replaces setUpClass: Runs once per test class."""
    # Initialize shared paths/results
    current_test_filename = Path(__file__).stem
    test_dir = Path(__file__).parent
    main_path = test_dir.parent
    video_dir = main_path / "inputs"

    request.cls._shared_model = None
    request.cls._shared_model_path = None
    request.cls._shared_model_device = None
    request.cls._shared_model_sf_enabled = None

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

    # Add any shared state to the class
    model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    request.cls.result_dir = test_dir / f"{current_test_filename}_results/{model_name}"
    request.cls.result_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark statistics
    request.cls.benchmarks = []
    request.cls.csv_filename = f"model_benchmarks_{request.cls.name}.csv"
    request.cls.csv_path = request.cls.result_dir / request.cls.csv_filename

    # Initialize class vars
    request.cls.active = True
    request.cls.active_streams = {}

    # RUN ALL PARAMETERIZED TESTS ----------------------------------------
    yield

    # FINAL CSV EXPORT  --------------------------------------------------
    if request.cls.benchmarks:
        results = request.cls.benchmarks
        keys = results[0].keys()

        with open(str(request.cls.csv_path), "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        print(f"\n[FINAL] Benchmarks saved to {request.cls.csv_filename}", flush=True)


@pytest.fixture(autouse=True)
def each_test_setup(request):
    test_class_self = request.instance
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    device = request.node.callspec.params.get("device")
    os.environ["DEVICE"] = device
    detection_type = "object"  # request.node.callspec.params.get("detection_type")
    sf_enabled = False  # request.node.callspec.params.get("sf_enabled")

    model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    test_class_self._testMethodName = f"{model_name}_{detection_type}_{device}"

    render_dir = test_class_self.result_dir / f"{device}"
    render_dir.mkdir(exist_ok=True)

    test_class_self.config = PipelineConfig(
        # GENERAL
        SHARED_OUTPUT=render_dir,  # os.getenv("SHARED_OUTPUT",SHARED_OUTPUT_DEFAULT),
        CUSTOM_MODEL_FLAG=os.getenv("CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT),
        DEVICE=device.upper(),
        OMIT_DETECTIONS_FLAG=True,
        TEST_MODE=False,
        DEBUG=os.getenv("DEBUG", DEBUG_DEFAULT),
        DEBUG_FRAME_LIMIT=os.getenv("DEBUG_FRAME_LIMIT", 100),
        # VIDEO WRITER
        # CLIP_DURATION=None,
        # VDMS
        ENABLE_QUERYING=False,
        DBHOST="0.0.0.0",
        DBPORT=55555,
        # MODEL
        MODEL_NAME=model_name,
        # MODEL_H=360,
        # PIPELINE
        SMART_FILTERING_ENABLED=sf_enabled,
        THRESHOLD_VALUE=int(os.getenv("THRESHOLD_VALUE", THRESHOLD_VALUE)),
        # VISUALIZATION
        DETECTION_TYPE=detection_type,
    )

    test_class_self.device = test_class_self.config.DEVICE
    test_class_self.device_input = test_class_self.config.device_input
    test_class_self.resize_h, test_class_self.resize_w = [
        test_class_self.config.MODEL_H,
        test_class_self.config.MODEL_W,
    ]

    # Resolve concrete handler class type
    HandlerClass = GPUStreamHandler if device == "gpu" else CPUStreamHandler
    # HandlerClass.pipeline_fn = test_class_self.__class__.pipeline_fn

    # Dynamically re-bind backend methods to this execution instance
    handler_classes = [HandlerClass, DeviceBaseHandler]
    all_method_names = set()

    for cls in handler_classes:
        for name, attr in inspect.getmembers(cls, predicate=inspect.isfunction):
            all_method_names.add(name)

    for method_name in all_method_names:
        source_obj = (
            HandlerClass if hasattr(HandlerClass, method_name) else DeviceBaseHandler
        )
        if hasattr(source_obj, method_name):
            raw_func = getattr(source_obj, method_name)
            if (
                not hasattr(test_class_self.__class__, method_name)
                or method_name == "pipeline_fn"
            ):
                setattr(
                    test_class_self,
                    method_name,
                    types.MethodType(raw_func, test_class_self),
                )

    # 3. FORCE NATIVE PIPELINE PROVISIONING
    test_class_self.setup_reader(
        test_class_self.config.TARGET_FPS, test_class_self.config.CLIP_DURATION
    )
    test_class_self.initialize_variables()
    test_class_self.setup_model(None)
    test_class_self.prepare_pipeline()

    # test_class_self.run _realtime_inference = types.MethodType(
    #     test_class_self.__class__.run_realtime_inference, test_class_self
    # )

    test_class_self.setup_threads()

    # RUN PARAMETERIZED TEST ----------------------------------------
    yield

    # tearDown LOGIC --------------------------------------------------
    print(
        f"\n--- [TearDown] Memory Before Cleanup ({test_class_self._testMethodName}) ---"
    )

    # test_class_self.stop()

    #  Nullify the model reference to trigger automatic cleanup
    if hasattr(test_class_self, "model") and test_class_self.model is not None:
        # Check if it's an Ultralytics model with a predictor
        predictor = getattr(test_class_self.model, "predictor", None)
        if predictor is not None:
            try:
                predictor.results = []
            except AttributeError:
                pass
        del test_class_self.model
        test_class_self.model = None

    #  Clear the singleton references to force a reload
    TestModelDetections._shared_model = None
    TestModelDetections._shared_model_path = None
    TestModelDetections._shared_model_device = None

    # Force Python to run destructors NOW while streams are still alive
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Final sync to clear event queue
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Critical for shared memory cleanup
        time.sleep(0.2)


@pytest.mark.usefixtures("setup_context")
class TestModelDetections:
    # SETUP --------------------------------------------

    @pytest.mark.parametrize(
        "device",
        [
            ("cpu"),
            ("gpu"),
        ],
    )
    def test_model(self, device):
        """Unified test runner for all configurations."""

        #  Run the actual model loader
        self.get_model_by_device(device, sf_enabled=self.config.sf_enabled)

        total_session_start = time.perf_counter()

        results = self.model.predict(
            source=self.source,
            imgsz=(self.frame_height, self.frame_width),
            batch=1,
            device=self.config.device_input,
            stream=True,
            conf=self.config.DETECTION_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            show=False,
            save=True,
            project=str(self.config.SHARED_OUTPUT),
            name="predict_results",
            exist_ok=True,  # overwrite if folder exists
            data={"names": {i: name for i, name in enumerate(self.label_source)}},
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

        self._finalize_benchmarks(
            real_world_latency_ms,
            frame_cnt,
            total_model_preprocess,
            total_model_inference,
            total_model_postprocess,
        )

    # HELPERS --------------------------------------------
    def get_model_by_device(self, device, sf_enabled=False):
        """Singleton loader: only loads if device changes or model is missing."""
        if (
            sf_enabled
            and (self.frame_width * self.frame_height)
            <= self.config.SMART_FILTERING_PIXEL_CONSTRAINT
        ):
            sf_enabled = False

        if (
            TestModelDetections._shared_model is not None
            and TestModelDetections._shared_model_device == device
            and TestModelDetections._shared_model_sf_enabled == sf_enabled
        ):
            self.model = TestModelDetections._shared_model
            self.model_path = TestModelDetections._shared_model_path
            return

        run_platform_name = "engine" if "cuda" in self.device_input else "openvino"

        if self.config.CUSTOM_MODEL_FLAG:
            dir_path = "/home/resources/models/ultralytics/custom_models"
        else:
            dir_path = f"/home/resources/models/ultralytics/{self.config.MODEL_NAME}/{self.config.MODEL_PRECISION}"

        (
            TestModelDetections._shared_model,
            TestModelDetections._shared_model_path,
            self.label_source,
        ) = get_model(
            Path(dir_path),
            self.config.MODEL_NAME,
            run_platform_name,
            self.device_input,
            batch=self.config.MODEL_MAX_BATCH_SIZE,
            force_export=force_export,
            sf_enabled=sf_enabled,
            model_h=self.resize_h,
            model_w=self.resize_w,
        )

        TestModelDetections._shared_model_device = device
        TestModelDetections._shared_model_sf_enabled = sf_enabled
        self.model = TestModelDetections._shared_model
        # self.model.half()
        self.model_path = TestModelDetections._shared_model_path

        W, H = self.resize_w, self.resize_h
        if not sf_enabled:
            W, H = self.frame_width, self.frame_height
        self.model_warmup(H, W)

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
        # self.setup_shared_memory()  # Natively sets up Manager dictionary and buffers

        # # Executor for Async YOLO tasks and FFmpeg re-encoding
        # self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        # # self.clip_executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        # print(
        #     f"sf_enabled: {self.config.sf_enabled}\tTEST_MODE: {self.config.TEST_MODE}",
        #     flush=True,
        # )

        # # Open up looking-ahead buffer horizons to eliminate 8K queue backpressure stalls
        # self.signal_queue = mp.Queue(maxsize=128)  # 10)
        # self.render_queue = mp.Queue(maxsize=256)  # 64)

        # test_dir = os.getenv(
        #     "TEST_SUITE_RENDER_DIR", str(Path(self.config.SHARED_OUTPUT))
        # )
        # os.makedirs(test_dir, exist_ok=True)

        # if self.source.startswith("rtsp"):
        #     short_name = "rtsp"
        # else:
        #     short_name = Path(self.source).stem
        # video_output_name = f"{self._testMethodName}_{short_name}_{self.device}.mp4"
        # out_path = os.path.join(test_dir, video_output_name)
        # self.output_path = (
        #     out_path  # f"{self.config.SHARED_OUTPUT}/{short_name}_{self.device}.mp4"
        # )

        # log_to_logger(
        #     f"[TEST MODE] Detection results saved to: {out_path}", level="info"
        # )

        # # =====================================================================
        # # DYNAMIC THREAD TARGET REDIRECTION
        # # =====================================================================
        # # Lambda forces the background processing thread to evaluate your custom staging loop natively
        # self.process_thread = threading.Thread(
        #     target=lambda: self.run_realtime_inference(
        #         sf_enabled=self.config.sf_enabled
        #     ),
        #     daemon=True,
        # )
        # # self.render_proc = mp.Process(  # threading.Thread(
        # #     target=rendering_worker,
        # #     args=(
        # #         self.render_queue,
        # #         (self.disp_w, self.disp_h),
        # #         self.output_path,
        # #         self.target_fps,
        # #     ),
        # #     daemon=True,
        # # )
        # # self.render_proc = threading.Thread(target=lambda: None, daemon=True)
        # # self.display_proc = threading.Thread(target=lambda: None, daemon=True)
        # self.render_proc = DummyProcess()
        # self.display_proc = DummyProcess()

        # self.async_writer = AsyncVideoWriter(
        #     self.output_path,
        #     cv2.VideoWriter_fourcc(*"avc1"),
        #     float(self.target_fps),
        #     (self.disp_w, self.disp_h),
        # )
        # # self.async_writer = None

    def start(self):
        """
        Starts the decoupled ingestion and inference threads in the correct order.
        """
        # PRE-SYNC: Ensure GPU is idle before timing starts
        if self.device_input == "cuda":
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Start the hardware-decoupled reader first
        self.reader.start()

        # if self.config.ENABLE_QUERYING:
        #     self._initialize_writer()

        # # Small delay to allow the reader's deque to populate
        # time.sleep(0.1)

        # # Start the producer and consumer threads
        # if hasattr(self, "process_thread") and not self.process_thread.is_alive():
        #     self.process_thread.start()

        # if not self.config.DISABLE_DETECTION:
        #     self.render_proc.start()

        #     self.display_proc.start()

        # if (
        #     self.config.ENABLE_QUERYING
        #     and not self.config.TEST_MODE
        #     and not self.metadata_thread.is_alive()
        # ):
        #     self.metadata_thread.start()

        # if self.config.ENABLE_QUERYING and not self.writer_thread.is_alive():
        #     self.writer_thread.start()

        return self

    def stop(self):
        """Overrides handlers.py to flush thread objects safely without deadlocking."""
        with self._stop_lock:
            if self._is_stopped:
                return

            print(
                f"\n[TEST HARNESS] Initiating rapid thread flush for {self.name}",
                flush=True,
            )
            self.active = False

            # 1. Rapidly clear the async video writer thread queue first
            # if hasattr(self, "async_writer") and self.async_writer is not None:
            #     try:
            #         self.async_writer.release()
            #     except Exception:
            #         pass

            # =====================================================================
            # DECUPLED MULTI-PROCESS TEARDOWN LIFECYCLE SEQUENCE
            # =====================================================================
            # 1. FIRST: Drain and synchronize your thread pool tasks so EVERY frame drops into the IPC queue
            if hasattr(self, "display_pool") and self.display_pool is not None:
                try:
                    if hasattr(self.display_pool, "_work_queue"):
                        backlog = self.display_pool._work_queue.qsize()
                        if backlog > 0:
                            print(
                                f"\n\033[94m[STAGE 1/4] Synchronizing display pool threads. Draining {backlog} background tasks...\033[0m",
                                flush=True,
                            )
                            backlog_s = time.perf_counter()
                            while (
                                self.display_pool._work_queue.qsize() > 0
                                or len(self.reorder_staging_buffer) > 0
                            ):
                                # Yield a brief slice to let worker threads complete downscaling
                                time.sleep(0.01)
                            backlog_e = time.perf_counter() - backlog_s
                            print(f"\t\033[94mTook {backlog_e} secs\033[0m", flush=True)
                    print(
                        "\033[92m[SUCCESS] All background image transformations completed cleanly.\033[0m",
                        flush=True,
                    )
                    self.display_pool.shutdown(wait=True)
                except Exception as e:
                    print(
                        f"[TEARDOWN ERROR] Thread pool synchronization failed: {e}",
                        flush=True,
                    )
                setattr(self, "display_pool", None)

            # 1. FIRST: Instantly dispatch the shutdown token so the worker process knows to flush entries
            if hasattr(self, "render_queue") and self.render_queue is not None:
                try:
                    print(
                        f"\n\033[94m[STAGE 1/3] Dispatching shutdown sentinel (None) to rendering process worker (Backlog: {self.render_queue.qsize()} frames)...\033[0m",
                        flush=True,
                    )
                    # Delivering the poison pill immediately tells FFmpeg to drain remaining blocks sequentially
                    self.render_queue.put(None, timeout=2.0)
                except Exception as queue_err:
                    print(
                        f"\033[91m[TEARDOWN ERROR] Failed staging shutdown token: {queue_err}\033[0m",
                        flush=True,
                    )

            # 2. SECOND: Allow the background multiprocessing queue to empty outstanding items naturally
            if hasattr(self, "render_queue") and self.render_queue is not None:
                try:
                    print(
                        "\n\033[94m[STAGE 2/3] Finalizing file output. Draining remaining frames from IPC memory lane...\033[0m",
                        flush=True,
                    )
                    t_drain = time.perf_counter()
                    # Wait up to 15 seconds for the worker process to pull every remaining matrix out of the pipe
                    while (
                        not self.render_queue.empty()
                        and (time.perf_counter() - t_drain) < 15.0
                    ):
                        time.sleep(
                            0.01
                        )  # Micro-yield grants immediate execution priority to the background process core
                except Exception:
                    pass

            # 3. THIRD: Block the main thread safely and allow the detached process to finish disk I/O operations
            if hasattr(self, "render_proc") and self.render_proc is not None:
                try:
                    join_duration = 0.0
                    if self.render_proc.is_alive():
                        print(
                            "\033[94m[STAGE 2/2] Main thread blocking. Synchronizing hard drive file headers...\033[0m",
                            flush=True,
                        )
                        t_join_start = time.perf_counter()
                        self.render_proc.join(
                            timeout=15.0
                        )  # Safe timeout lets video containers finalize cleanly
                        join_duration = time.perf_counter() - t_join_start
                    else:
                        print(
                            "\033[94m[STAGE 2/2] Main thread clear...\033[0m",
                            flush=True,
                        )

                    if self.render_proc.is_alive():
                        print(
                            "\n\033[91m[CRITICAL STALL] Render worker process hung on disk write! Forcing termination...\033[0m",
                            flush=True,
                        )
                        self.render_proc.terminate()
                        self.render_proc.join()
                    else:
                        print(
                            f"\033[92m[FINAL] Video compilation finished cleanly! Hard drive commit took {join_duration:.2f} seconds.\033[0m",
                            flush=True,
                        )

                        expected_disk_path = Path(self.output_path)
                        if expected_disk_path.exists():
                            size_mb = expected_disk_path.stat().st_size / (1024 * 1024)
                            print(
                                f"\033[92m[DISK VERIFIED] Destination asset compiled: {expected_disk_path.name} ({size_mb:.2f} MB)\033[0m\n",
                                flush=True,
                            )
                    self.render_proc.close()
                except Exception as proc_err:
                    print(
                        f"\033[91m[TEARDOWN ERROR] Failed closing background process container: {proc_err}\033[0m",
                        flush=True,
                    )
                setattr(self, "render_proc", None)

            # 4. FOURTH: Clean up remaining pipeline executors and queues safely
            if hasattr(self, "render_queue") and self.render_queue is not None:
                try:
                    self.render_queue.close()
                    self.render_queue.cancel_join_thread()
                except Exception:
                    pass
                setattr(self, "render_queue", None)

            if hasattr(self, "executor") and self.executor is not None:
                try:
                    self.executor.shutdown(wait=True)
                except Exception:
                    pass
                setattr(self, "executor", None)

            self.status = "DONE"
            self._is_stopped = True
            print(
                "[TEST HARNESS] Teardown complete. Releasing session cleanly.\n",
                flush=True,
            )


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
    # parser.add_argument(
    #     "--type",
    #     type=str,
    #     choices=["object", "motion"],
    #     dest="detection_type",
    #     help="Filter by detection type (object or motion)",
    # )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        help="Filter by device (cpu or gpu)",
    )
    # parser.add_argument(
    #     "--sf",
    #     action="store_true",
    #     default=None,
    #     dest="sf_enabled",
    #     help="Filter by Smart Filtering",
    # )

    # # DEBUGGING
    # parser.add_argument(
    #     "--debug",
    #     action="store_true",
    #     help="Enable debug message and save intermediate images for Smart Filtering tests",
    # )
    # parser.add_argument(
    #     "-n",
    #     type=int,
    #     default=100,
    #     dest="debug_frame_limit",
    #     help="Number of frames used for debugging [Default: 100]",
    # )

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
        __file__,
    ]  # -s -v --log-cli-level=DEBUG
    if run_args:
        pytest_args.extend(["-k", " and ".join(run_args)])

    print(f"Launching tests for {args.source}")

    sys.exit(pytest.main(pytest_args))
