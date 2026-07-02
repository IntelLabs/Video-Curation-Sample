import argparse
import csv
import gc
import logging
import os
import sys
import time
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
from include.handlers import DeviceBaseHandler
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

    VIDEO_FILENAME = os.getenv("VIDEO_FILENAME", "anduril_swarm_8K.mp4")
    if video_dir.exists():
        request.cls.video_path = video_dir / VIDEO_FILENAME
    else:
        video_dir = Path("/watch_dir")
        request.cls.video_path = video_dir / VIDEO_FILENAME

    # Add any shared state to the class
    model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    request.cls.result_dir = (
        test_dir / f"{current_test_filename}_results/{request.cls.video_path.stem}"
    )
    request.cls.result_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark statistics
    request.cls.benchmarks = []
    request.cls.csv_filename = (
        f"model_benchmarks_{model_name}_{request.cls.video_path.stem}.csv"
    )
    request.cls.csv_path = request.cls.result_dir / request.cls.csv_filename

    # Initialize class vars
    request.cls.name = request.cls.video_path.stem
    request.cls.source = str(request.cls.video_path)
    request.cls.is_rtsp = str(request.cls.source).startswith("rtsp:/")
    request.cls.active = True
    request.cls.active_streams = {}
    request.cls._shared_model = None
    request.cls._shared_model_path = None
    request.cls._shared_model_device = None
    request.cls._shared_model_sf_enabled = None

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

    # setUp LOGIC --------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    device = request.node.callspec.params.get("device")
    os.environ["DEVICE"] = device
    detection_type = "object"  # request.node.callspec.params.get("detection_type")
    sf_enabled = False  # request.node.callspec.params.get("sf_enabled")
    model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    test_class_self._testMethodName = f"{model_name}_{detection_type}_{device}"

    render_dir = test_class_self.result_dir / f"{test_class_self._testMethodName}"
    render_dir.mkdir(exist_ok=True)
    test_class_self.config = PipelineConfig(
        # GENERAL
        SHARED_OUTPUT=render_dir,  # os.getenv("SHARED_OUTPUT",SHARED_OUTPUT_DEFAULT),
        CUSTOM_MODEL_FLAG=os.getenv(
            "CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT
        ),  # True,
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

    test_class_self.setup_reader(
        test_class_self.config.TARGET_FPS, test_class_self.config.CLIP_DURATION
    )

    # RUN PARAMETERIZED TEST ----------------------------------------
    yield

    # tearDown LOGIC --------------------------------------------------
    print(
        f"\n--- [TearDown] Memory Before Cleanup ({test_class_self._testMethodName}) ---"
    )

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
    TestSmartFilteringDetections._shared_model = None
    TestSmartFilteringDetections._shared_model_path = None
    TestSmartFilteringDetections._shared_model_device = None

    # Force Python to run destructors NOW while streams are still alive
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Final sync to clear event queue
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Critical for shared memory cleanup
        time.sleep(0.2)


@pytest.mark.usefixtures("setup_context")
class TestSmartFilteringDetections:
    # SETUP --------------------------------------------

    @pytest.mark.parametrize(
        "device",
        [
            ("cpu"),
            ("gpu"),
        ],
    )
    def test_pipeline(self, device):
        """Unified test runner for all configurations."""

        #  Run the actual model loader
        self.get_model_by_device(device, sf_enabled=self.config.sf_enabled)

        total_session_start = time.perf_counter()

        results = self.model.predict(
            source=self.source,
            conf=self.config.DETECTION_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            show=False,
            imgsz=(self.frame_height, self.frame_width),
            save=True,
            project=str(self.config.SHARED_OUTPUT),
            name=f"pred_{self._testMethodName}_output_video",
            exist_ok=True,  # overwrite if folder exists
            stream=True,
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
            TestSmartFilteringDetections._shared_model is not None
            and TestSmartFilteringDetections._shared_model_device == device
            and TestSmartFilteringDetections._shared_model_sf_enabled == sf_enabled
        ):
            self.model = TestSmartFilteringDetections._shared_model
            self.model_path = TestSmartFilteringDetections._shared_model_path
            return

        run_platform_name = "engine" if "cuda" in self.device_input else "openvino"

        if self.config.CUSTOM_MODEL_FLAG:
            dir_path = "/home/resources/models/ultralytics/custom_models"
        else:
            dir_path = f"/home/resources/models/ultralytics/{self.config.MODEL_NAME}/{self.config.MODEL_PRECISION}"

        (
            TestSmartFilteringDetections._shared_model,
            TestSmartFilteringDetections._shared_model_path,
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

        TestSmartFilteringDetections._shared_model_device = device
        TestSmartFilteringDetections._shared_model_sf_enabled = sf_enabled
        self.model = TestSmartFilteringDetections._shared_model
        # self.model.half()
        self.model_path = TestSmartFilteringDetections._shared_model_path

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
            "Video": self.video_path.name,
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


# INHERIT METHODS FROM HANDLERS -----------------------------------------------------------------
TestSmartFilteringDetections.setup_reader = DeviceBaseHandler.setup_reader
TestSmartFilteringDetections.get_frameWH = DeviceBaseHandler.get_frameWH
TestSmartFilteringDetections.run_model = DeviceBaseHandler.run_model
TestSmartFilteringDetections.model_warmup = DeviceBaseHandler.model_warmup


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
