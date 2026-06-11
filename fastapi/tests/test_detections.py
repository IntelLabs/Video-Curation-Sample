import argparse
import csv
import gc
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorrt as trt
import torch
import torch.nn.functional as F

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
    log_to_logger,
)
from include.handlers import test_rendering_worker as rendering_worker
from include.utils import (
    PipelineConfig,
    tensor2opencv,
)

try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

DEBUG_FLAG_DEFAULT = True if DEBUG_DEFAULT == "1" else False
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
force_export = False


# def render_display_worker(queue, output_path, fps, size):
#     """Background process that draws labels and writes to video."""
#     fourcc = cv2.VideoWriter_fourcc(*"avc1")
#     writer = cv2.VideoWriter(output_path, fourcc, fps, size)

#     if not writer.isOpened():
#         print(f" [ERROR] VideoWriter failed to open: {output_path}")
#         return

#     while True:
#         item = queue.get()
#         if item is None:
#             break

#         display_frame, metadata_or_bbs, class_list = item
#         if (display_frame.shape[1], display_frame.shape[0]) != size:
#             display_frame = cv2.resize(display_frame, size)

#         if isinstance(metadata_or_bbs, dict):
#             for _, obj in metadata_or_bbs.items():
#                 bbox = obj["bbox"]
#                 x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
#                 class_name = bbox["object"]
#                 class_id = class_list.index(class_name) if class_name in class_list else 0
#                 confidence = bbox["object_det"]["confidence"]

#                 bb_color = get_detection_color(class_id, is_bgr=True)
#                 label = f"{class_name} {confidence:.2f}"

#                 cv2.rectangle(display_frame, (x, y), (x + w, y + h), bb_color, 2)
#                 draw_label(display_frame, label, (x, y), color=bb_color, padding=5)
#         elif metadata_or_bbs is not None:
#             for box in metadata_or_bbs:
#                 if torch.is_tensor(box):
#                     box = box.tolist()
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 225), 2)

#         writer.write(display_frame)

#     writer.release()


def fps_comparison_chart(chart_path, results, fps_key="Pipeline FPS (Video frames)"):
    try:
        names = [r["Test Name"] for r in results]
        fps_values = [float(r[fps_key]) for r in results]

        plt.figure(figsize=(10, 6))
        plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

        colors = ["#2ca02c" if "gpu" in n.lower() else "#1f77b4" for n in names]
        bars = plt.bar(names, fps_values, color=colors, zorder=3)
        plt.ylabel("Frames Per Second (FPS)")
        plt.title(f"Performance Comparison: {chart_path.stem}")
        plt.xticks(rotation=45)

        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 1,
                f"{yval:.1f}",
                ha="center",
                va="bottom",
            )

        if fps_values:
            plt.ylim(0, max(fps_values) * 1.2)

        plt.tight_layout()
        plt.savefig(str(chart_path))
        print(f" Comparison chart saved to: {chart_path}")
    except Exception:
        print("Skipping chart generation: error occurred.")


@pytest.fixture(scope="class")
def setup_context(request):
    """Replaces setUpClass: Runs once per test class."""
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

    model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    request.cls.result_dir = (
        test_dir
        / f"{current_test_filename}_results/{model_name}"
        / request.cls.video_path.stem
    )
    request.cls.result_dir.mkdir(parents=True, exist_ok=True)
    request.cls.benchmarks = []
    request.cls.csv_filename = (
        f"pipeline_benchmarks_{model_name}_{request.cls.video_path.stem}.csv"
    )

    request.cls.name = request.cls.video_path.stem
    request.cls.source = str(request.cls.video_path)
    request.cls.active = True
    request.cls.active_streams = {}

    request.cls._shared_model = None
    request.cls._shared_model_path = None
    request.cls._shared_model_device = None
    request.cls._shared_model_sf_enabled = None
    yield

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
                    gpu_fps = float(row["Pipeline FPS (Video frames)"])
                    cpu_fps = float(cpu_row["Pipeline FPS (Video frames)"])
                    speedup = (gpu_fps / cpu_fps) if cpu_fps > 0 else 0
                    row["Pipeline Speedup vs CPU"] = f"{speedup:.2f}x"
                else:
                    row["Pipeline Speedup vs CPU"] = "N/A"
            else:
                row["Pipeline Speedup vs CPU"] = "Baseline (CPU)"

        keys = results[0].keys()
        with open(
            str(request.cls.result_dir / request.cls.csv_filename), "w", newline=""
        ) as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        print(f"\n[FINAL] Benchmarks saved to {request.cls.csv_filename}")

        for r in results:
            print(
                f" > {r['Test Name']}: {r['Pipeline FPS (Video frames)']} FPS | {r['Pipeline FPS (Target frames)']} FPS | {r['sf+roi+det FPS (Target frames)']} FPS | {r['Display FPS']} FPS | Speedup: {r.get('Pipeline Speedup vs CPU', 'N/A')}"
            )

        chart_path = (
            request.cls.result_dir
            / f"{request.cls.csv_filename.replace('.csv', '')}_sfFPS.png"
        )
        fps_comparison_chart(
            chart_path, results, fps_key="sf+roi+det FPS (Video frames)"
        )

        chart_path = (
            request.cls.result_dir
            / f"{request.cls.csv_filename.replace('.csv', '')}_pipelineFPS.png"
        )
        fps_comparison_chart(chart_path, results, fps_key="Pipeline FPS (Video frames)")

        chart_path = (
            request.cls.result_dir
            / f"{request.cls.csv_filename.replace('.csv', '')}_sfFPS_target.png"
        )
        fps_comparison_chart(
            chart_path, results, fps_key="sf+roi+det FPS (Target frames)"
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
            / f"{request.cls.csv_filename.replace('.csv', '')}_displayFPS_target.png"
        )
        fps_comparison_chart(chart_path, results, fps_key="Display FPS")


@pytest.fixture(autouse=True)
def each_test_setup(request):
    test_class_self = request.instance
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    device = request.node.callspec.params.get("device")
    detection_type = request.node.callspec.params.get("detection_type")
    sf_enabled = request.node.callspec.params.get("sf_enabled")

    test_class_self._testMethodName = (
        f"sf_{detection_type}_{device}"
        if sf_enabled
        else f"yolo_{detection_type}_{device}"
    )

    # 1. Re-initialize a fresh configuration object
    test_class_self.config = PipelineConfig(
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

    # video_output_name = f"{test_class_self._testMethodName}_detections_output.mp4"
    vid_dir = test_class_self.result_dir / "results"
    vid_dir.mkdir(parents=True, exist_ok=True)
    # test_video_output_path = os.path.join(str(vid_dir), video_output_name)
    os.environ["TEST_SUITE_RENDER_DIR"] = str(vid_dir)
    test_class_self.config.SHARED_OUTPUT = str(test_class_self.result_dir)

    # Reset core state machine properties before invoking any backend handlers
    test_class_self.active = True
    test_class_self._is_stopped = False

    # FIXING ATTRIBUTEERRORS: Explicitly provision primitives before method re-binding
    import threading
    import time

    test_class_self._stop_lock = threading.Lock()
    # test_class_self.stat_start_time = time.perf_counter() # timing to display detection

    test_class_self.device = test_class_self.config.DEVICE
    test_class_self.device_input = test_class_self.config.device_input
    test_class_self.resize_h, test_class_self.resize_w = [
        test_class_self.config.MODEL_H,
        test_class_self.config.MODEL_W,
    ]

    # Resolve concrete handler class type
    HandlerClass = GPUStreamHandler if device == "gpu" else CPUStreamHandler
    HandlerClass.pipeline_fn = test_class_self.__class__.pipeline_fn

    # 2. Dynamically re-bind backend methods to this execution instance
    import inspect
    import types

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

    # FIXING FILEEXISTSERRORS: Proactively clear lingering POSIX layout blocks in /dev/shm
    from multiprocessing import shared_memory

    for i in range(4):  # Matches your self.ai_ring_depth footprint
        stale_shm_name = f"shm_ai_640_{test_class_self.name}_{i}_{os.getpid()}"
        try:
            # Force link onto lingering handle and unlink it instantly from the OS map
            lingering_shm = shared_memory.SharedMemory(name=stale_shm_name)
            lingering_shm.close()
            lingering_shm.unlink()
        except FileNotFoundError:
            pass

    # 3. FORCE NATIVE PIPELINE PROVISIONING
    test_class_self.setup_reader(
        test_class_self.config.TARGET_FPS, test_class_self.config.CLIP_DURATION
    )
    test_class_self.initialize_variables()
    test_class_self.setup_model(None)
    test_class_self.prepare_pipeline()

    # Reset runtime loop state properties
    test_class_self.render_queue = None
    test_class_self.frame_count_target = 0
    test_class_self.next_process_idx = 0.0
    test_class_self.frame_in_clip_count = 0
    test_class_self.frame_count = 0
    test_class_self.elapsed_display_time = 0.0

    if hasattr(test_class_self, "setup_threads"):
        test_class_self.setup_threads()

    shared_event_buffer = {"sf": [], "roi": [], "det": []}
    test_class_self.gpu_event_buffer = shared_event_buffer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    yield

    # Teardown logic
    if (
        hasattr(test_class_self, "render_queue")
        and test_class_self.render_queue is not None
    ):
        try:
            test_class_self.render_queue.put(None)
        except Exception:
            pass

    render_proc_handle = getattr(test_class_self, "render_proc", None)
    if render_proc_handle is not None and render_proc_handle._started.is_set():
        test_class_self.render_proc.join(timeout=10.0)

    # Signal the state machine to drop out of loops safely
    test_class_self.active = False
    if hasattr(test_class_self, "reader") and test_class_self.reader is not None:
        try:
            test_class_self.reader.stop()
        except Exception:
            pass

    # Execute custom release blocks
    test_class_self.stop()

    # CRITICAL FIX: Block and wait for the background producer thread to die
    # BEFORE we delete the reader attribute from the namespace map.
    producer_thread_handle = getattr(test_class_self, "process_thread", None)
    if producer_thread_handle is not None and producer_thread_handle.is_alive():
        producer_thread_handle.join(timeout=5.0)

    # Now it is structurally safe to purge attributes without causing cross-thread collisions
    keys_to_purge = [
        "reader",
        "process_thread",
        "inference_stream",
        "bgs_stream",
        "model",
        "raw_input",
        "ai_gpu_staging",
        "ai_pinned_tensors",
        "executor",
        "io_executor",
    ]
    for key in keys_to_purge:
        if hasattr(test_class_self, key):
            delattr(test_class_self, key)

    if (
        hasattr(test_class_self, "gpu_event_buffer")
        and test_class_self.gpu_event_buffer
    ):
        test_class_self.gpu_event_buffer.clear()
    if hasattr(test_class_self, "component_stats") and test_class_self.component_stats:
        test_class_self.component_stats.clear()

    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    gc.collect()
    time.sleep(0.2)


@pytest.mark.usefixtures("setup_context")
class TestSmartFilteringDetections:
    # class TestSmartFilteringDetections(DeviceBaseHandler):
    #     """
    #     Unified testing harness.
    #     By inheriting from DeviceBaseHandler, 'self' functions as both
    #     the pytest telemetry harness and the live stream execution runner.
    #     """
    #     # Overriding __init__ to prevent standard initialization collisions with pytest
    #     def __init__(self, *args, **kwargs):
    #         pass
    # SETUP --------------------------------------------

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize("sf_enabled", [True, False])
    @pytest.mark.parametrize("detection_type", ["motion", "object"])
    def test_detections(self, detection_type, device, sf_enabled):
        """Unified test runner for all configurations."""
        if detection_type == "motion" and not sf_enabled:
            pytest.skip(
                "Pure YOLO mode is structurally invalid for detection_type 'motion'."
            )

        #  Run the actual model loader
        # self.get_model_by_device(device, sf_enabled=self.config.sf_enabled)

        # Execute
        self.run_pipeline()  # pipeline_fn)

    def setup_threads(self):
        # Shared 10MB memory for display
        self.setup_shared_memory()

        # Executor for Async YOLO tasks and FFmpeg re-encoding
        # self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        # self.clip_executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        print(
            f"sf_enabled: {self.config.sf_enabled}\tTEST_MODE: {self.config.TEST_MODE}",
            flush=True,
        )

        # Producer: Handles acquisition and AI metadata logs
        # self.process_thread = threading.Thread(
        #     target=self.run_pipeline,
        #     daemon=True,
        # )

        self.signal_queue = mp.Queue(maxsize=1)
        self.render_queue = mp.Queue(maxsize=5)

        # if self.config.TEST_MODE:
        test_dir = os.getenv(
            "TEST_SUITE_RENDER_DIR", str(Path(self.config.SHARED_OUTPUT))
        )
        os.makedirs(test_dir, exist_ok=True)

        video_output_name = f"{self._testMethodName}_detections_output.mp4"
        out_path = os.path.join(test_dir, video_output_name)

        log_to_logger(
            f"[TEST MODE] Detection results saved to: {out_path}", level="info"
        )
        self.render_proc = threading.Thread(
            target=rendering_worker,
            args=(
                self.render_queue,
                (self.disp_w, self.disp_h),
                out_path,
                self.target_fps,
            ),
            daemon=True,
        )

        # Dummy target alignment to prevent execution signature exceptions
        self.display_proc = threading.Thread(target=lambda: None, daemon=True)
        # else:
        #     self.render_proc = mp.Process(
        #         target=rendering_worker,
        #         args=(
        #             self.render_queue,
        #             self.shared_details,
        #             self.ready_buffer_idx,
        #             self.reader_active_idx,
        #             self.shm_frame_lengths,
        #             self.signal_queue,
        #             (self.disp_w, self.disp_h),
        #             self.config.DISPLAY_FRAME_QUALITY,
        #         ),
        #     )

        #     def display_signal_sync():
        #         while self.active:
        #             # Wait for signal
        #             # if self.mp_frame_ready_event.wait(timeout=1.0):
        #             #     self.mp_frame_ready_event.clear()
        #             try:
        #                 _ = self.signal_queue.get(timeout=1.0)
        #                 # print(f"[DEBUG]: Signal received in FastAPI process for {self.name}", flush=True)
        #                 # Wake FastAPI async loop in main thread
        #                 self.loop.call_soon_threadsafe(self.frame_ready_event.set)
        #             except queue.Empty:
        #                 continue

        #     self.display_proc = threading.Thread(
        #         target=display_signal_sync, daemon=True
        #     )

        # if self.config.ENABLE_QUERYING:
        #     # NEW: Dedicated I/O pool for Disk/GPU transfers (Higher worker count for 8K)
        #     self.io_executor = ThreadPoolExecutor(max_workers=8)

        #     # Dedicated FFmpeg pool so re-encoding doesn't slow down live AI
        #     self.ffmpeg_executor = ThreadPoolExecutor(max_workers=2)

        #     if not self.config.TEST_MODE:
        #         # Sends metadata to VDMS
        #         self.metadata_thread = threading.Thread(
        #             target=send_metadata,
        #             args=(
        #                 VDMSPool(self.config.DBHOST, self.config.DBPORT, size=10),
        #                 self.config.DEBUG_FLAG,
        #                 self.config.INGESTION,
        #                 self.config.TEST_MODE,
        #                 self.config.UDF_HOST,
        #                 self.config.UDF_PORT,
        #                 self.config.DBHOST,
        #                 self.config.DBPORT,
        #             ),
        #             daemon=True,
        #         )

        #     # Consumer: Handles GPU-to-CPU download and Disk I/O (Writing resized frames to RAM disk)
        #     self.writer_thread = threading.Thread(
        #         target=self.video_writer_core_loop,
        #         args=(self.stop_writer,),
        #         daemon=True,
        #     )

    # HELPERS --------------------------------------------
    def _print_gpu_mem(self):
        if torch.cuda.is_available():
            # Memory currently used by tensors
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            # Total memory reserved by PyTorch (the "Pool")
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"\tAllocated: {allocated:0.2f} MB")
            print(f"\tReserved:  {reserved:0.2f} MB")
        else:
            print("\tCUDA not available.")

    # def get_model_by_device(self, device, sf_enabled=False):
    #     """Singleton loader: only loads if device changes or model is missing."""
    #     if (
    #         sf_enabled
    #         and (self.frame_width * self.frame_height)
    #         <= self.config.SMART_FILTERING_PIXEL_CONSTRAINT
    #     ):
    #         sf_enabled = False

    #     if (
    #         TestSmartFilteringDetections._shared_model is not None
    #         and TestSmartFilteringDetections._shared_model_device == device
    #         and TestSmartFilteringDetections._shared_model_sf_enabled == sf_enabled
    #     ):
    #         self.model = TestSmartFilteringDetections._shared_model
    #         self.model_path = TestSmartFilteringDetections._shared_model_path
    #         return

    #     run_platform_name = "engine" if "cuda" in self.device_input else "openvino"

    #     if self.config.CUSTOM_MODEL_FLAG:
    #         dir_path = "/home/resources/models/ultralytics/custom_models"
    #     else:
    #         dir_path = f"/home/resources/models/ultralytics/{self.config.MODEL_NAME}/{self.config.MODEL_PRECISION}"

    #     (
    #         TestSmartFilteringDetections._shared_model,
    #         TestSmartFilteringDetections._shared_model_path,
    #         self.label_source,
    #     ) = get_model(
    #         # model_run_key, model_run_config, export=False
    #         Path(dir_path),
    #         self.config.MODEL_NAME,
    #         run_platform_name,
    #         self.device_input,
    #         batch=self.config.MODEL_MAX_BATCH_SIZE,
    #         force_export=force_export,
    #         sf_enabled=sf_enabled,
    #         model_h=self.resize_h,
    #         model_w=self.resize_w,
    #     )
    #     TestSmartFilteringDetections._shared_model_device = device
    #     TestSmartFilteringDetections._shared_model_sf_enabled = sf_enabled
    #     self.model = TestSmartFilteringDetections._shared_model
    #     # self.model.half()
    #     self.model_path = TestSmartFilteringDetections._shared_model_path

    #     W, H = self.resize_w, self.resize_h
    #     if not sf_enabled:
    #         W, H = self.frame_width, self.frame_height
    #     self.model_warmup(H, W)
    #     # self.pipeline_handler.model = self.model
    #     # self.pipeline_handler.model_warmup(H, W)

    # def calculate_unique_coverage(self, merged_boxes, target_w=640, target_h=640):
    #     """
    #     FULLY VECTORIZED: Calculate pixel coverage without Python loops.
    #     Works for any number of boxes (1 to 1000+) with near-zero overhead.
    #     """
    #     if merged_boxes is None or merged_boxes.shape[0] == 0:
    #         return 0.0

    #     # 1. Scaling (Vectorized)
    #     # merged_boxes is [N, 4] -> [x1, y1, x2, y2] in 8K space
    #     scale = torch.tensor(
    #         [
    #             target_w / self.frame_width,
    #             target_h / self.frame_height,
    #             target_w / self.frame_width,
    #             target_h / self.frame_height,
    #         ],
    #         device=self.device_input,
    #     )

    #     # Scale and clamp all boxes at once on the GPU
    #     coords = (merged_boxes * scale).long()
    #     coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, target_w)
    #     coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, target_h)

    #     # 2. Vectorized Mask Filling
    #     # We create a 1D representation of the 640x640 mask for fast indexing
    #     mask = torch.zeros(
    #         target_h * target_w, device=self.device_input, dtype=torch.uint8
    #     )

    #     # For each box, we generate a range of indices and fill them.
    #     # Note: For small N (drones), a loop is okay, but for 'Swarm' noise,
    #     # we use this broadcasted approach:
    #     for i in range(coords.shape[0]):
    #         x1, y1, x2, y2 = coords[i]
    #         # Generate row indices for this box
    #         rows = torch.arange(y1, y2, device=self.device_input).view(-1, 1)
    #         # Calculate mask indices: (y * width) + x
    #         # This fills a horizontal slice of the mask in one GPU operation
    #         indices = (rows * target_w) + torch.arange(x1, x2, device=self.device_input)
    #         mask[indices] = 1

    #     # 3. Final Sum (GPU Reduction)
    #     return (torch.sum(mask).item() / (target_w * target_h)) * 100

    def calculate_unique_coverage(self, merged_boxes, target_w=640, target_h=640):
        """
        TRUE LOOPLESS VECTORIZATION: Calculates combined bounding box pixel coverage
        in a single GPU pass, completely avoiding CPU-GPU synchronization stalls.
        """
        if merged_boxes is None or merged_boxes.shape[0] == 0:
            return 0.0

        scale = torch.tensor(
            [
                target_w / self.frame_width,
                target_h / self.frame_height,
                target_w / self.frame_width,
                target_h / self.frame_height,
            ],
            device=self.device_input,
        )

        coords = (merged_boxes * scale).long()
        coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, target_w)
        coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, target_h)

        x1 = coords[:, 0].view(-1, 1, 1)
        y1 = coords[:, 1].view(-1, 1, 1)
        x2 = coords[:, 2].view(-1, 1, 1)
        y2 = coords[:, 3].view(-1, 1, 1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(target_h, device=self.device_input),
            torch.arange(target_w, device=self.device_input),
            indexing="ij",
        )

        inside_boxes = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)
        unique_mask = torch.any(inside_boxes, dim=0)

        return (torch.sum(unique_mask).item() / (target_w * target_h)) * 100

    def _finalize_benchmarks(
        self,
        # n_frames,
        num_objs,
        total_pipeline_ms,
        real_world_latency_ms,
        coverage_percentages,
        sf_enabled,
        stat_frame_count,
        stat_fps,
    ):
        """Aggregates metrics and adds them to the results list."""
        latency_s = total_pipeline_ms / 1000.0  # Just pipeline (sf + roi_ det)
        est_fps = self.frame_count / latency_s if latency_s > 0 else 0
        est_fps_processed = self.frame_count_target / latency_s if latency_s > 0 else 0
        duration_s = self.frame_count / self.input_fps if self.input_fps > 0 else 0

        real_latency_s = real_world_latency_ms / 1000.0
        real_est_fps = self.frame_count / real_latency_s if real_latency_s > 0 else 0
        real_est_fps_processed = (
            self.frame_count_target / real_latency_s if real_latency_s > 0 else 0
        )

        # Calculate averages for component breakdowns
        avg_sf = (
            sum(self.component_stats["sf"]) / len(self.component_stats["sf"])
            if self.component_stats["sf"]
            else 0
        )
        avg_roi = (
            sum(self.component_stats["roi"]) / len(self.component_stats["roi"])
            if self.component_stats["roi"]
            else 0
        )
        avg_det = (
            sum(self.component_stats["det"]) / len(self.component_stats["det"])
            if self.component_stats["det"]
            else 0
        )
        avg_cov = (
            sum(coverage_percentages) / len(coverage_percentages)
            if coverage_percentages
            else (100.0 if not sf_enabled else 0)
        )

        avg_crops = (
            sum(self.crops_per_frame_list) / len(self.crops_per_frame_list)
            if self.crops_per_frame_list
            else 0
        )

        # Total Latency Sum (SF + ROI + DET)
        total_sum = avg_sf + avg_roi + avg_det

        # Calculate how often we hit a high-motion cap (e.g., 20 crops)
        capped_frames = sum(1 for c in self.crops_per_frame_list if c >= 20)
        cap_rate = (
            (capped_frames / len(self.crops_per_frame_list)) * 100
            if self.crops_per_frame_list
            else 0
        )

        self.__class__.benchmarks.append(
            {
                "Test Name": self._testMethodName,
                "Detection Type": self.config.DETECTION_TYPE,
                "Device": self.device,
                "Smart Filtering": "Enabled" if sf_enabled else "Disabled",
                "Video": self.video_path.name,  # self.name?
                "Video FPS": f"{self.input_fps:.2f}",
                "Video Duration (s)": f"{duration_s:.4f}",
                "Video Frames": self.frame_count,
                "Target Frames": self.frame_count_target,
                "Pipeline Latency (s)": f"{real_latency_s:.2f}",
                "Display Latency (s)": f"{self.elapsed_display_time:.2f}",
                "sf+roi+det Latency (s)": f"{latency_s:.2f}",
                # Includes time by HW decoder (reader) and preparing output video
                "Pipeline FPS (Video frames)": f"{real_est_fps:.2f}",
                "Pipeline FPS (Target frames)": f"{real_est_fps_processed:.2f}",
                # TIming to display frame (read to after send to render queue)
                "Display Frames": stat_frame_count,
                "Display FPS": f"{stat_fps:.2f}",
                # Only sf + roi + det
                "sf+roi+det FPS (Video frames)": f"{est_fps:.2f}",
                "sf+roi+det FPS (Target frames)": f"{est_fps_processed:.2f}",
                "Avg SF (ms)": f"{avg_sf:.2f}",
                "Avg ROI (ms)": f"{avg_roi:.2f}",
                "Avg Obj. Detection (ms)": f"{avg_det:.2f}",
                "Total Breakdown Sum (ms)": f"{total_sum:.2f}",
                "Avg Area Coverage %": f"{avg_cov:.2f}%",
                "Avg Crops/Frame": f"{avg_crops:.1f}",
                "Crop Cap Rate (>20)": f"{cap_rate:.1f}%",
                "Objects Detected": num_objs,
            }
        )

        print(f"\n[{self._testMethodName}] Latency: {latency_s:.2f} sec")
        print(
            f"\n[{self._testMethodName}] Pipeline FPS (Target frames): {real_est_fps_processed:.2f} ({self.frame_count_target} frames)"
        )
        print(
            f"\n[{self._testMethodName}] sf+roi+det FPS (Target frames): {est_fps_processed:.2f} ({self.frame_count_target} frames)"
        )
        print(
            f"\n[{self._testMethodName}] Display FPS (Target frames): {stat_fps:.2f} ({stat_frame_count} frames)"
        )

    # def tensor2opencv_gpu(self, frame_tensor):
    #     """
    #     GPU-native equivalent of tensor2opencv.
    #     Fixes the '3x3 ghosting' and 'ValueError' at 30+ FPS.
    #     """
    #     # 1. Handle Batch Dimension: (1, 3, 640, 640) -> (3, 640, 640)
    #     temp = frame_tensor.squeeze(0) if frame_tensor.ndim == 4 else frame_tensor

    #     # 2. Fix Layout & Shape: (3, 640, 640) -> (640, 640, 3)
    #     # contiguous() physically rearranges pixels in VRAM to interleave colors.
    #     # reshape() ensures we never see the (1, 409600, 3) shape again.
    #     gpu_hwc = temp.permute(1, 2, 0).reshape(640, 640, 3).contiguous()

    #     # 3. Visibility Fix: Scale floats (0.0-1.0) to bytes (0-255) on GPU
    #     if gpu_hwc.dtype != torch.uint8:
    #         gpu_hwc = (gpu_hwc * 255).clamp(0, 255).byte()

    #     # 4. Color Space Fix: RGB -> BGR (Matches OpenCV)
    #     gpu_bgr = gpu_hwc.flip(-1).contiguous()

    #     # 5. Bridge to CuPy (Zero-Copy)
    #     cp_frame = cp.from_dlpack(torch.utils.dlpack.to_dlpack(gpu_bgr))

    #     # 6. Select Double-Buffer from Pool
    #     target_buf_cpu = self.pinned_pool_np[self.pool_idx]
    #     target_buf_gpu = self.cp_pool[self.pool_idx]
    #     self.pool_idx = (self.pool_idx + 1) % 2

    #     if not hasattr(self, "transfer_stream"):
    #         self.transfer_stream = cp.cuda.Stream(non_blocking=True)

    #     with self.transfer_stream:
    #         # cp.copyto is more robust for pinned memory than target[:]
    #         cp.copyto(target_buf_gpu, cp_frame)
    #         # Download into the pinned CPU RAM
    #         target_buf_gpu.get(out=target_buf_cpu)

    #     # 7. Mandatory Sync: Wait for the DMA transfer to hit RAM
    #     self.transfer_stream.synchronize()

    #     # Return a snapshot copy so the worker has private memory for 30 FPS
    #     return target_buf_cpu.copy()

    # DEBUG FUNCTIONS --------------------------------------------
    def debug_save_mask(self, frame_source, frame_num, rois=None):
        debug_dir = self.result_dir / "debug_mask" / self._testMethodName
        debug_dir.mkdir(parents=True, exist_ok=True)

        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        #  Download or copy the data
        if hasattr(frame_source, "download"):
            img_cpu = frame_source.download()
        elif torch.is_tensor(frame_source):
            # .contiguous() fixes the horizontal "shredding"/static look
            temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
            # img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
            if temp.ndim == 1:
                temp = temp.view(self.resize_h, self.resize_w)

            if temp.ndim == 3:
                img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
            else:
                img_cpu = temp.contiguous().cpu().numpy()
        else:
            # For numpy arrays (like your pinned memory), ensure memory is linear
            img_cpu = np.ascontiguousarray(frame_source)

        #  Fix Visibility (Normalization)
        # If float, scale to 0-255. If uint8, leave as is to avoid "neon" colors.
        if img_cpu.dtype != np.uint8:
            if img_cpu.max() <= 1.0:
                img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
            else:
                img_cpu = img_cpu.astype(np.uint8)

        #  Handle Color Space
        # OpenCV imwrite expects BGR. If 3-channel (RGB), swap. If 1-channel, save as-is.
        if len(img_cpu.shape) == 3:
            # img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
            pass
        else:
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)

        # Draw 8K Boxes (Scaled down)
        if rois is not None:
            h_img, w_img = img_cpu.shape[:2]
            scale_x = float(w_img) / self.frame_width
            scale_y = float(h_img) / self.frame_height
            boxes = rois.cpu().tolist() if torch.is_tensor(rois) else rois
            for box in boxes:
                x1, y1, x2, y2 = [
                    int(box[0] * scale_x),
                    int(box[1] * scale_y),
                    int(box[2] * scale_x),
                    int(box[3] * scale_y),
                ]
                cv2.rectangle(img_cpu, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save to disk
        save_path = debug_dir / f"mask_{frame_num:04d}.jpg"
        cv2.imwrite(str(save_path), img_cpu)

    def debug_save_img_roi(self, frame_source, bbs_full_res, frame_num):
        debug_dir = self.result_dir / "debug_analysis" / self._testMethodName
        debug_dir.mkdir(parents=True, exist_ok=True)

        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        #  Download/Copy the frame
        if torch.is_tensor(frame_source):
            # .contiguous() is CRITICAL here to fix the "shredded" look
            temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
            img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
        elif hasattr(frame_source, "download"):
            img_cpu = frame_source.download()
        else:
            img_cpu = np.ascontiguousarray(frame_source)

        #  Fix Shape: restore spatial grid if flattened
        if img_cpu.ndim == 3 and img_cpu.shape[0] == 1:
            img_cpu = img_cpu.reshape((self.resize_h, self.resize_w, 3))

        #  Fix Visibility: ONLY multiply if it's actually floating point
        # If uint8 is multiplied by 255, it wraps around and creates "neon" colors
        if img_cpu.dtype != np.uint8:
            if img_cpu.max() <= 1.0:
                img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
            else:
                img_cpu = img_cpu.astype(np.uint8)

        # Color Space: Standardize to BGR for imwrite
        if len(img_cpu.shape) != 3:
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)

        # Draw 8K Boxes (Scaled down)
        h_img, w_img = img_cpu.shape[:2]
        scale_x = w_img / self.frame_width
        scale_y = h_img / self.frame_height

        if bbs_full_res is not None:
            boxes = (
                bbs_full_res.cpu().tolist()
                if torch.is_tensor(bbs_full_res)
                else bbs_full_res
            )
            for box in boxes:
                x1, y1, x2, y2 = [
                    int(box[0] * scale_x),
                    int(box[1] * scale_y),
                    int(box[2] * scale_x),
                    int(box[3] * scale_y),
                ]
                cv2.rectangle(img_cpu, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(debug_dir / f"analysis_{frame_num:04d}.jpg"), img_cpu)

    def debug_save_crops(self, cropped_batch, frame_num):
        """Saves the first 5 crops of a batch to the results directory."""
        debug_dir = self.result_dir / "debug_crops" / self._testMethodName
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Only save for the first self.config.DEBUG_FRAME_LIMIT frames to avoid disk bloat
        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        for i, crop in enumerate(cropped_batch[: self.config.DEBUG_FRAME_LIMIT]):
            # Convert GPU Tensor [C, H, W] -> NumPy [H, W, C]
            if torch.is_tensor(crop):
                # Reverse normalization (* 255) and permute to BGR
                img = (crop.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
            else:
                img = crop

            cv2.imwrite(str(debug_dir / f"frame_{frame_num}_crop_{i}.jpg"), img)

    def debug_save_img(self, frame_source, frame_num):
        debug_dir = self.result_dir / "debug_test" / self._testMethodName
        debug_dir.mkdir(parents=True, exist_ok=True)

        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        #  Download/Copy the frame
        if torch.is_tensor(frame_source):
            # .contiguous() is CRITICAL here to fix the "shredded" look
            temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
            img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
        elif hasattr(frame_source, "download"):
            img_cpu = frame_source.download()
        else:
            img_cpu = np.ascontiguousarray(frame_source)

        #  Fix Shape: restore spatial grid if flattened
        if img_cpu.ndim == 3 and img_cpu.shape[0] == 1:
            img_cpu = img_cpu.reshape((self.resize_h, self.resize_w, 3))

        #  Fix Visibility: ONLY multiply if it's actually floating point
        # If uint8 is multiplied by 255, it wraps around and creates "neon" colors
        if img_cpu.dtype != np.uint8:
            if img_cpu.max() <= 1.0:
                img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
            else:
                img_cpu = img_cpu.astype(np.uint8)

        # Color Space: Standardize to BGR for imwrite
        if len(img_cpu.shape) == 3:
            pass
        else:
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(debug_dir / f"analysis_{frame_num:04d}.jpg"), img_cpu)

    # # FRAME PROCESSORS --------------------------------------------
    # def run_pipeline(self):  #, pipeline_fn):
    #     # n_frames = 0
    #     num_objs = 0
    #     self.frame_count_target = 0
    #     self.next_process_idx = 0.0
    #     self.frame_in_clip_count = 0
    #     total_pipeline_time_ms = 0.0  # Track pure latency
    #     coverage_percentages = []
    #     self.component_stats = {"sf": [], "roi": [], "det": []}
    #     self.gpu_event_buffer = {"sf": [], "roi": [], "det": []}
    #     self.crops_per_frame_list = []

    #     # PRE-SYNC: Ensure GPU is idle before timing starts
    #     # if self.device_input == "cuda":
    #     #     # Pre-allocate CUDA events for isolated GPU timing
    #     #     # start_event = torch.cuda.Event(enable_timing=True)
    #     #     # end_event = torch.cuda.Event(enable_timing=True)

    #     #     if torch.cuda.is_available():
    #     #         torch.cuda.synchronize()

    #     self.start()
    #     time.sleep(0.1)

    #     # start_time = time.perf_counter()
    #     total_session_start = time.perf_counter()

    #     while self.active:  # and n_frames < 10*self.fps:
    #         # ret, frame = self.cap.read()
    #         device_frame, frame_num = self.reader.read()
    #         if device_frame is None:
    #             if self.reader.stopped:
    #                 self.active = False
    #                 break
    #             continue

    #         # n_frames += 1
    #         self.frame_count += 1
    #         is_target_frame = float(frame_num) >= self.next_process_idx
    #         nob = 0

    #         # if self.device_input == "cuda":
    #         #     # start_eve   nt.record()
    #         #     nob, metrics = self.pipeline_fn(device_frame, frame_num, is_target_frame)
    #         #     # end_event.record()
    #         #     # torch.cuda.synchronize()
    #         #     # total_pipeline_time_ms += start_event.elapsed_time(end_event)
    #         # else:
    #             # start_t = time.perf_counter()
    #         if self.device_input == "cuda":
    #             # Record the reader's current layout availability milestone
    #             curr_event = torch.cuda.Event()
    #             curr_event.record()

    #             # Instruct the isolated inference stream to wait for this specific frame context
    #             self.inference_stream.wait_event(curr_event)
    #             with torch.cuda.stream(self.inference_stream):
    #                 nob, metrics = self.pipeline_fn(device_frame, frame_num, is_target_frame)
    #             # self.inference_stream.synchronize()

    #         else:
    #             nob, metrics = self.pipeline_fn(device_frame, frame_num, is_target_frame)
    #             # total_pipeline_time_ms += (time.perf_counter() - start_t) * 1000

    #         num_objs += nob

    #         if is_target_frame and metrics != {}:
    #             num_crops = len(metrics["bbs"]) if metrics["bbs"] is not None else 0
    #             self.crops_per_frame_list.append(num_crops)

    #             # self.component_stats["roi"].append(metrics["roi_time"])

    #             # if self.device_input != "cuda":
    #             if metrics.get("sf_time"):
    #                 self.component_stats["sf"].append(metrics["sf_time"])
    #             if metrics.get("det_time"):
    #                 self.component_stats["det"].append(metrics["det_time"])
    #             if metrics.get("roi_time"):
    #                 self.component_stats["roi"].append(metrics["roi_time"])

    #             # Calculate coverage OUTSIDE the timed block to prevent interference
    #             if self.config.sf_enabled and metrics.get("bbs") is not None:
    #                 cov = self.calculate_unique_coverage(metrics["bbs"])
    #                 coverage_percentages.append(cov)

    #     # POST-SYNC: Ensure all GPU tasks finished before timing ends
    #     if self.device_input == "cuda":
    #         torch.cuda.synchronize()

    #     # ACTUAL speed including all overhead
    #     real_world_latency_ms = (time.perf_counter() - total_session_start) * 1000

    #     # if self.device_input == "cuda":
    #     #     # Move GPU event timings into component_stats lists
    #     #     for key in ["sf", "det", "roi"]:
    #     #         for start, end in self.gpu_event_buffer[key]:
    #     #             self.component_stats[key].append(start.elapsed_time(end))

    #     # Calculate Pure processing latency
    #     # This excludes: Reader, Tensor2OpenCV, Queueing, and Video Writing
    #     total_pipeline_time_ms = (
    #         sum(self.component_stats["sf"])
    #         + sum(self.component_stats["roi"])
    #         + sum(self.component_stats["det"])
    #     )

    #     # assert num_objs > 0
    #     if num_objs == 0:
    #         print(f" [WARNING] No objects detected for {self._testMethodName}")

    #     self._finalize_benchmarks(
    #         self.frame_count,
    #         num_objs,
    #         total_pipeline_time_ms,
    #         real_world_latency_ms,
    #         coverage_percentages,
    #         self.config.sf_enabled,
    #         self.stat_frame_count,
    #         self.stat_fps,
    #     )

    def run_pipeline(self):
        num_objs = 0
        self.frame_count_target = 0
        self.next_process_idx = 0.0
        self.frame_in_clip_count = 0
        total_pipeline_time_ms = 0.0
        coverage_percentages = []
        self.component_stats = {"sf": [], "roi": [], "det": []}
        self.crops_per_frame_list = []

        self.step_size = (
            float(self.input_fps) / float(self.target_fps)
            if hasattr(self, "target_fps")
            else 1.0
        )

        self.start()
        time.sleep(0.1)

        total_session_start = time.perf_counter()

        while self.active:
            # 1. CAPTURE COMPLETE CYCLE OVERHEAD
            # t_cycle_start = time.perf_counter()

            device_frame, frame_num = self.reader.read()
            if device_frame is None:
                if self.reader is None or (
                    hasattr(self.reader, "stopped") and self.reader.stopped
                ):
                    # --- CRITICAL PIPELINE DRAIN START ---
                    print(
                        "[INFO] Reader reached EOF. Draining asynchronous workers and VRAM queues..."
                    )

                    # 1. Thread Pool Flush: Force the CPU thread to block here until
                    # every single background pipeline task in the executor finishes.
                    if hasattr(self, "executor") and self.executor:
                        self.executor.shutdown(wait=True)

                    # 2. Render Queue Flush: If you utilize an asynchronous video frame
                    # saving worker thread, wait for its queue tasks to bottom out.
                    if hasattr(self, "render_queue") and self.render_queue:
                        while not self.render_queue.empty():
                            time.sleep(0.01)

                    # 3. GPU Hardware Flush: Force the GPU to completely finish
                    # all remaining background subtraction, crops, and YOLO operations.
                    if self.device_input == "cuda":
                        torch.cuda.synchronize()

                    self.active = False
                    break
                continue

            self.stat_start_time = time.perf_counter()  # timing to display detection
            self.frame_count += 1
            # is_target_frame = float(frame_num) >= self.next_process_idx
            # CRITICAL CADENCE DRIFT FIX: If input FPS equals target FPS, or if Smart Filtering
            # is disabled (YOLO baseline), force is_target_frame to ALWAYS be True.
            # This completely bypasses floating-point accumulation drift errors.
            if (
                not self.config.sf_enabled
                or abs(float(self.input_fps) - float(self.target_fps)) < 0.01
            ):
                is_target_frame = True
            else:
                is_target_frame = float(frame_num) >= self.next_process_idx

            # 2. DISPATCH WORKLOADS ASYNCHRONOUSLY
            metrics = {}
            if is_target_frame:
                self.next_process_idx += self.step_size

                if self.device_input == "cuda":
                    # Instantiate a lightweight hardware fence event object
                    curr_event = torch.cuda.Event(enable_timing=False)
                    # curr_event.record()
                    # Record the exact milestone on the default stream right after fetching the frame data
                    curr_event.record(torch.cuda.default_stream())
                    self.inference_stream.wait_event(curr_event)
                    with torch.cuda.stream(self.inference_stream):
                        # 1. Isolate the incoming image buffer canvas
                        isolated_device_frame = (
                            device_frame.clone()
                            if torch.is_tensor(device_frame)
                            else device_frame.copy()
                        )

                        # 2. RUN BACKGROUND SUBTRACTION IMMEDIATELY ON THE PRODUCER TIMELINE
                        # This guarantees that the mask matches this exact frame_num before threads overlap!
                        nob, metrics = self.pipeline_fn(
                            isolated_device_frame,
                            frame_num,
                            is_target_frame,
                            self.stat_start_time,
                        )
                else:
                    # 1. Isolate the incoming image buffer canvas
                    isolated_device_frame = (
                        device_frame.clone()
                        if torch.is_tensor(device_frame)
                        else device_frame.copy()
                    )

                    # 2. RUN BACKGROUND SUBTRACTION IMMEDIATELY ON THE PRODUCER TIMELINE
                    # This guarantees that the mask matches this exact frame_num b
                    nob, metrics = self.pipeline_fn(
                        isolated_device_frame,
                        frame_num,
                        is_target_frame,
                        self.stat_start_time,
                    )

                num_objs += nob
            # else:
            #     # Process background execution context for skipped frames
            #     nob, metrics = self.pipeline_fn(
            #         device_frame, frame_num, is_target_frame, self.stat_start_time
            #     )

            # 3. ENFORCE UNIFIED HARDWARE TIMING BARRIER
            # if self.device_input == "cuda":
            #     torch.cuda.synchronize()

            # t_cycle_end = time.perf_counter()
            # cycle_total_ms = (t_cycle_end - t_cycle_start) * 1000.0

            # 4. ALLOCATE ALL TRACKING TIMINGS ACCURATELY
            # if is_target_frame:
            if metrics != {}:
                num_crops = len(metrics["bbs"]) if metrics["bbs"] is not None else 0
                self.crops_per_frame_list.append(num_crops)

                if self.config.sf_enabled:
                    self.component_stats["sf"].append(metrics["sf_time"])
                    self.component_stats["roi"].append(metrics["roi_time"])
                else:
                    # Full-frame YOLO baseline accounts for total data movement cycle
                    # self.component_stats["det"].append(cycle_total_ms)
                    self.component_stats["sf"].append(0.0)
                    self.component_stats["roi"].append(0.0)

                self.component_stats["det"].append(metrics["det_time"])

                if self.config.sf_enabled and metrics.get("bbs") is not None:
                    cov = self.calculate_unique_coverage(metrics["bbs"])
                    coverage_percentages.append(cov)
            # else:
            #     # CRITICAL METRICS FIX: If Smart Filtering runs on a skipped frame,
            #     # its mask generation overhead MUST be captured and tracked!
            #     if self.config.sf_enabled and metrics != {}:
            #         self.component_stats["sf"].append(metrics["sf_time"])

        if self.device_input == "cuda":
            torch.cuda.synchronize()

        real_world_latency_ms = (time.perf_counter() - total_session_start) * 1000.0

        total_pipeline_time_ms = (
            sum(self.component_stats["sf"])
            + sum(self.component_stats["roi"])
            + sum(self.component_stats["det"])
        )

        if num_objs == 0:
            print(f" [WARNING] No objects detected for {self._testMethodName}")

        self._finalize_benchmarks(
            # self.frame_count,
            num_objs,
            total_pipeline_time_ms,
            real_world_latency_ms,
            coverage_percentages,
            self.config.sf_enabled,
            self.stat_frame_count,
            self.stat_fps,
        )

    # # TESTS --------------------------------------------
    # def pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
    #     num_objs = 0
    #     metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

    #     # Initialize timing event handle pairs
    #     sf_start, sf_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #     roi_start, roi_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #     det_start, det_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    #     # --- 1. MOTION MASK GENERATION GATE ---
    #     if self.config.sf_enabled:
    #         if self.device_input == "cuda":
    #             sf_start.record(self.inference_stream)
    #             inf_data = self.rbtd_full_gpu(device_frame)
    #             sf_end.record(self.inference_stream)
    #         else:
    #             t_start = time.perf_counter()
    #             inf_data = self.rbtd_full_cpu(device_frame)
    #             metrics["sf_time"] = (time.perf_counter() - t_start) * 1000.0
    #     else:
    #         inf_data = {}

    #     # --- PIPELINE AT TARGET RATE ---
    #     if not is_target_frame:
    #         # If skipping, pull outstanding execution records immediately
    #         if self.device_input == "cuda" and self.config.sf_enabled:
    #             self.inference_stream.synchronize()
    #             metrics["sf_time"] = sf_start.elapsed_time(sf_end)
    #         return num_objs, metrics

    #     self.next_process_idx += self.step_size
    #     self.frame_count_target += 1
    #     self.frame_in_clip_count += 1
    #     inf_data["frameNum"] = self.frame_count_target

    #     # --- 2. FULL-RESOLUTION ROI EXTRACTION MAPS ---
    #     bbs_full_res = None
    #     if self.config.sf_enabled:
    #         if self.device_input == "cuda":
    #             roi_start.record(self.inference_stream)
    #             bbs_full_res = self.get_gpu_rois(
    #                 inf_data["full_frame"],
    #                 self.frame_count_target,
    #                 inf_data["mask"],
    #             )
    #             roi_end.record(self.inference_stream)
    #             metrics["bbs"] = bbs_full_res
    #         else:
    #             t_start = time.perf_counter()
    #             bbs_full_res = self.get_cpu_rois(
    #                 inf_data["full_frame"],
    #                 self.frame_count_target,
    #                 inf_data["mask"],
    #             )
    #             metrics["roi_time"] = (time.perf_counter() - t_start) * 1000.0
    #             metrics["bbs"] = bbs_full_res

    #         if self.config.DEBUG_FLAG:
    #             if self.device_input == "cuda":
    #                 torch.cuda.stream(self.inference_stream)
    #                 torch.cuda.synchronize()
    #                 display_source = inf_data["mask"]
    #                 self.debug_save_mask(
    #                     display_source, self.frame_count_target, rois=bbs_full_res
    #                 )

    #     clean_bbs = []
    #     if self.config.sf_enabled and bbs_full_res is not None:
    #         if torch.is_tensor(bbs_full_res):
    #             clean_bbs = bbs_full_res.detach().cpu().numpy()
    #         else:
    #             clean_bbs = np.array(bbs_full_res)

    #     # --- 3. MODEL INFERENCE TIMING BLOCK ---
    #     if self.device_input == "cuda":
    #         det_start.record(self.inference_stream)
    #     else:
    #         t_start = time.perf_counter()

    #     if self.config.DETECTION_TYPE != "motion":
    #         det_frame = inf_data["full_frame"] if "full_frame" in inf_data else device_frame
    #         merged = clean_bbs if self.config.sf_enabled else None
    #         metadata, _ = self.get_detections(
    #             det_frame,
    #             self.frame_in_clip_count,
    #             merged=merged,
    #             thickness=self.config.THICKNESS,
    #             device_input=self.config.device_input,
    #         )
    #         num_objs = len(metadata.keys())
    #     else:
    #         num_objs = len(clean_bbs)
    #         metadata = clean_bbs

    #     if self.device_input == "cuda":
    #         det_end.record(self.inference_stream)

    #         # CRITICAL PERFORMANCE SYNCHRONIZATION POINT:
    #         # We execute a single stream synchronization barrier here at the end of the entire loop.
    #         # This allows the GPU kernels to run overlapped and fully concurrently!
    #         self.inference_stream.synchronize()

    #         # Unpack hardware timings efficiently
    #         if self.config.sf_enabled:
    #             metrics["sf_time"] = sf_start.elapsed_time(sf_end)
    #             metrics["roi_time"] = roi_start.elapsed_time(roi_end)
    #         metrics["det_time"] = det_start.elapsed_time(det_end)
    #         metrics["bbs"] = bbs_full_res
    #     else:
    #         metrics["det_time"] = (time.perf_counter() - t_start) * 1000.0

    #     # --- 4. RENDER WORKER PREPARATION ---
    #     display_source = inf_data["full_frame"] if (inf_data and "full_frame" in inf_data) else device_frame

    #     if self.device_input == "cuda":
    #         gpu_resized = F.interpolate(
    #             display_source.unsqueeze(0).float(),
    #             size=(self.disp_h, self.disp_w),
    #             mode="bilinear",
    #             align_corners=False,
    #         ).squeeze(0).contiguous()
    #         disp_frame = np.copy(tensor2opencv(gpu_resized, self.config.device_input, is_bgr=True))
    #     else:
    #         cpu_resized = cv2.resize(device_frame, (self.disp_w, self.disp_h))
    #         disp_frame = np.copy(tensor2opencv(cpu_resized, self.config.device_input, is_bgr=True))

    #     data_to_draw = clean_bbs if self.config.DETECTION_TYPE == "motion" else metadata

    #     try:
    #         if hasattr(self, "render_queue") and getattr(self, "render_queue", None) is not None and not self.render_queue.full():
    #             self.render_queue.put(
    #                 (
    #                     disp_frame,
    #                     inf_data["frameNum"] if "frameNum" in inf_data else self.frame_count_target,
    #                     data_to_draw,
    #                     self.label_source,
    #                 )
    #             )
    #         if self.config.DEBUG_FLAG:
    #             self.debug_save_img(disp_frame, self.frame_count_target)
    #             self.debug_save_img_roi(disp_frame, bbs_full_res, self.frame_count_target)
    #     except queue.Full:
    #         pass

    #     self.update_frame()
    #     return num_objs, metrics

    def pipeline_fn(
        self, device_frame, overall_frame_num, is_target_frame, stat_start_time
    ):
        num_objs = 0
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Pre-allocate non-blocking event records to extract clean GPU timings
        sf_start, sf_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        roi_start, roi_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        det_start, det_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

        # --- 1. MOTION MASK GENERATION GATE ---
        if self.config.sf_enabled:
            if self.device_input == "cuda":
                sf_start.record(self.inference_stream)
                inf_data = self.rbtd_full_gpu(device_frame)
                sf_end.record(self.inference_stream)
            else:
                t_start = time.perf_counter()
                inf_data = self.rbtd_full_cpu(device_frame)
                metrics["sf_time"] = (time.perf_counter() - t_start) * 1000.0
        else:
            inf_data = {}

        # --- PIPELINE AT TARGET RATE ---
        if not is_target_frame:
            # Safe, non-blocking timing extraction for skipped frames
            if self.device_input == "cuda" and self.config.sf_enabled:
                torch.cuda.synchronize()
                metrics["sf_time"] = sf_start.elapsed_time(sf_end)
            return num_objs, metrics

        self.frame_count_target += 1
        self.frame_in_clip_count += 1
        inf_data["frameNum"] = self.frame_count_target

        # --- 2. FULL-RESOLUTION ROI EXTRACTION MAPS ---
        bbs_full_res = None
        if self.config.sf_enabled:
            if self.device_input == "cuda":
                roi_start.record(self.inference_stream)
                bbs_full_res = self.get_gpu_rois(
                    inf_data["full_frame"],
                    self.frame_count_target,
                    inf_data["mask"],
                )
                roi_end.record(self.inference_stream)
                metrics["bbs"] = bbs_full_res
            else:
                t_start = time.perf_counter()
                bbs_full_res = self.get_cpu_rois(
                    inf_data["full_frame"],
                    self.frame_count_target,
                    inf_data["mask"],
                )
                metrics["roi_time"] = (time.perf_counter() - t_start) * 1000.0
                metrics["bbs"] = bbs_full_res

            if self.config.DEBUG_FLAG:
                # if self.device_input == "cuda":
                #     # Isolate data capturing using an asynchronous memory clone operation
                #     # This safely copies data without dropping your multi-stream execution pipeline concurrency
                #     display_source = inf_data["mask"].clone().to("cpu", non_blocking=True)
                # else:
                display_source = inf_data["mask"]
                #     self.inference_stream.synchronize()
                # display_source = inf_data["mask"]
                self.debug_save_mask(
                    display_source, self.frame_count_target, rois=bbs_full_res
                )

        clean_bbs = []
        if self.config.sf_enabled and bbs_full_res is not None:
            if torch.is_tensor(bbs_full_res):
                clean_bbs = bbs_full_res.detach().cpu().numpy()
            else:
                clean_bbs = np.array(bbs_full_res)

        # --- 3. MODEL INFERENCE TIMING BLOCK ---
        if self.device_input == "cuda":
            det_start.record(self.inference_stream)
        else:
            t_start = time.perf_counter()

        if self.config.DETECTION_TYPE != "motion":
            # det_frame = (
            #     inf_data["full_frame"] if "full_frame" in inf_data else device_frame
            # )
            # Isolate your image buffer array view to prevent upstream reader pointer races
            if "full_frame" in inf_data:
                det_frame = inf_data["full_frame"]
            else:
                det_frame = (
                    device_frame.clone()
                    if torch.is_tensor(device_frame)
                    else device_frame.copy()
                )

            merged = clean_bbs if self.config.sf_enabled else None
            metadata, _ = self.get_detections(
                det_frame,
                self.frame_in_clip_count,
                merged=merged,
                thickness=self.config.THICKNESS,
                device_input=self.config.device_input,
            )
            num_objs = len(metadata.keys())
        else:
            num_objs = len(clean_bbs)
            metadata = clean_bbs

        if self.device_input == "cuda":
            det_end.record(self.inference_stream)

            torch.cuda.synchronize()

            # Extract hardware timings smoothly without stalling mid-run
            if self.config.sf_enabled:
                metrics["sf_time"] = sf_start.elapsed_time(sf_end)
                metrics["roi_time"] = roi_start.elapsed_time(roi_end)
            metrics["det_time"] = det_start.elapsed_time(det_end)
            metrics["bbs"] = bbs_full_res
        else:
            metrics["det_time"] = (time.perf_counter() - t_start) * 1000.0

        # --- 4. RENDER WORKER PREPARATION ---
        display_source = (
            inf_data["full_frame"]
            if (inf_data and "full_frame" in inf_data)
            else device_frame
        )

        if self.device_input == "cuda":
            gpu_resized = (
                F.interpolate(
                    display_source.unsqueeze(0).float(),
                    size=(self.disp_h, self.disp_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .contiguous()
            )
            disp_frame = np.copy(
                tensor2opencv(gpu_resized, self.config.device_input, is_bgr=True)
            )
        else:
            cpu_resized = cv2.resize(device_frame, (self.disp_w, self.disp_h))
            disp_frame = np.copy(
                tensor2opencv(cpu_resized, self.config.device_input, is_bgr=True)
            )

        data_to_draw = clean_bbs if self.config.DETECTION_TYPE == "motion" else metadata

        # try:
        #     if (
        #         hasattr(self, "render_queue")
        #         and getattr(self, "render_queue", None) is not None
        #         and not self.render_queue.full()
        #     ):
        #         self.render_queue.put(
        #             (
        #                 disp_frame,
        #                 inf_data["frameNum"]
        #                 if "frameNum" in inf_data
        #                 else self.frame_count_target,
        #                 data_to_draw,
        #                 self.label_source,
        #             )
        #         )
        #     if self.config.DEBUG_FLAG:
        #         self.debug_save_img(disp_frame, self.frame_count_target)
        #         self.debug_save_img_roi(
        #             disp_frame, bbs_full_res, self.frame_count_target
        #         )
        # except queue.Full:
        #     pass

        if (
            hasattr(self, "render_queue")
            and getattr(self, "render_queue", None) is not None
        ):
            self.render_queue.put(
                (
                    disp_frame,
                    inf_data["frameNum"]
                    if "frameNum" in inf_data
                    else self.frame_count_target,
                    data_to_draw,
                    self.label_source,
                )
            )
        if self.config.DEBUG_FLAG:
            self.debug_save_img(disp_frame, self.frame_count_target)
            self.debug_save_img_roi(disp_frame, bbs_full_res, self.frame_count_target)

        self.update_frame(stat_start_time)
        return num_objs, metrics

    # def sf_cpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
    #     num_objs = 0
    #     metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

    #     # Smart Filtering (Resize / Background Subtraction / Threshold / Dilate)
    #     t_start = time.perf_counter()
    #     inf_data = self.rbtd_full_cpu(device_frame)
    #     metrics["sf_time"] = (time.perf_counter() - t_start) * 1000

    #     # Only keep frames for TARGET_FPS
    #     # Videos are written for target frames only
    #     if is_target_frame:
    #         self.next_process_idx += self.step_size
    #         self.frame_count_target += 1  # 1-indexed

    #         if not inf_data or inf_data["mask"] is None:
    #             return num_objs, metrics

    #         metadata = {}
    #         bbs_to_send = []
    #         data_to_draw = []
    #         if inf_data:
    #             inf_data["frameNum"] = self.frame_count_target

    #             # Get ROIs
    #             t_start = time.perf_counter()
    #             bbs_full_res = self.get_cpu_rois(
    #                 inf_data["full_frame"],
    #                 self.frame_count_target,
    #                 inf_data["mask"],
    #             )
    #             metrics["roi_time"] = (time.perf_counter() - t_start) * 1000
    #             metrics["bbs"] = bbs_full_res

    #             if self.config.DEBUG_FLAG:
    #                 display_source = inf_data["mask"]
    #                 self.debug_save_mask(
    #                     display_source, self.frame_count_target, rois=bbs_full_res
    #                 )

    #             if bbs_full_res is not None and len(bbs_full_res) == 0:
    #                 return num_objs, metrics

    #             # if self.config.DEBUG_FLAG or self.config.DETECTION_TYPE == "motion":
    #             #     display_source = (
    #             #         inf_data["full_frame"]
    #             #         if (inf_data and "full_frame" in inf_data)
    #             #         else device_frame
    #             #     )
    #             #     cpu_resized = cv2.resize(
    #             #         display_source,
    #             #         (self.resize_w, self.resize_h),
    #             #         interpolation=cv2.INTER_NEAREST,
    #             #     )

    #             t_start = time.perf_counter()
    #             if self.config.DETECTION_TYPE == "motion":
    #                 # Motion Mode: Prepare boxes for drawing
    #                 if (
    #                     bbs_full_res is not None
    #                     and bbs_full_res.ndim == 2
    #                     and bbs_full_res.size(0) > 0
    #                 ):
    #                     scaled_resized_bbs = bbs_full_res / self.scales_tensor
    #                     bbs_to_send = scaled_resized_bbs.cpu().tolist()
    #                 else:
    #                     bbs_to_send = []

    #                 data_to_draw = bbs_to_send
    #                 num_objs = len(bbs_full_res)
    #             else:
    #                 # Object Mode: Run YOLO and prepare metadata
    #                 det_frame = inf_data["full_frame"] if inf_data else device_frame
    #                 metadata, _ = self.get_detections(
    #                     det_frame,
    #                     self.frame_count_target,  # Frame used in metadata
    #                     merged=bbs_full_res,
    #                     thickness=self.config.THICKNESS,
    #                     device_input=self.config.device_input,
    #                 )
    #                 data_to_draw = metadata
    #                 num_objs = len(metadata.keys())

    #             metrics["det_time"] = (time.perf_counter() - t_start) * 1000

    #         # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
    #         if not self.config.TEST_MODE:
    #             display_source = (
    #                 inf_data["full_frame"]
    #                 if (inf_data and "full_frame" in inf_data)
    #                 else device_frame
    #             )
    #             cpu_resized = cv2.resize(display_source, (self.resize_w, self.resize_h))
    #             display_frame = tensor2opencv(
    #                 cpu_resized, self.config.device_input, is_bgr=True
    #             )
    #             self.render_queue.put((display_frame, data_to_draw, self.label_source))

    #             if self.config.DEBUG_FLAG:
    #                 cpu_resized = cv2.resize(
    #                     display_source, (self.resize_w, self.resize_h)
    #                 )
    #                 self.debug_save_img(cpu_resized, self.frame_count_target)
    #                 self.debug_save_img_roi(
    #                     cpu_resized, bbs_full_res, self.frame_count_target
    #                 )

    #     return num_objs, metrics

    # def sf_gpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
    #     num_objs = 0
    #     metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

    #     # Smart Filtering (Resize / Background Subtraction / Threshold / Dilate)
    #     sf_start, sf_end = (
    #         torch.cuda.Event(enable_timing=True),
    #         torch.cuda.Event(enable_timing=True),
    #     )
    #     sf_start.record()
    #     inf_data = self.rbtd_full_gpu(device_frame)
    #     sf_end.record()
    #     self.gpu_event_buffer["sf"].append((sf_start, sf_end))

    #     # Only keep frames for TARGET_FPS
    #     # Videos are written for target frames only
    #     if is_target_frame:
    #         self.next_process_idx += self.step_size
    #         self.frame_count_target += 1  # 1-indexed

    #         if not inf_data or inf_data["mask"] is None:
    #             return num_objs, metrics

    #         metadata = {}
    #         bbs_to_send = []
    #         data_to_draw = []
    #         if inf_data:
    #             inf_data["frameNum"] = self.frame_count_target

    #             # Get ROIs
    #             # time.perf_counter accurate since called self.bgs_stream.waitForCompletion()
    #             # t_start = time.perf_counter()
    #             roi_start = torch.cuda.Event(enable_timing=True)
    #             roi_end = torch.cuda.Event(enable_timing=True)
    #             roi_start.record()
    #             bbs_full_res = self.get_gpu_rois(
    #                 inf_data["full_frame"],
    #                 self.frame_count_target,
    #                 inf_data["mask"],
    #             )
    #             # metrics["roi_time"] = (time.perf_counter() - t_start) * 1000
    #             # self.gpu_event_buffer["roi"].append((roi_start, roi_end))
    #             roi_end.record()
    #             self.gpu_event_buffer["roi"].append((roi_start, roi_end))
    #             metrics["bbs"] = bbs_full_res

    #             if self.config.DEBUG_FLAG:
    #                 self.bgs_stream.waitForCompletion()
    #                 display_source = inf_data["mask"]
    #                 self.debug_save_mask(
    #                     display_source, self.frame_count_target, rois=bbs_full_res
    #                 )

    #             if bbs_full_res is not None and len(bbs_full_res) == 0:
    #                 return num_objs, metrics

    #             det_start, det_end = (
    #                 torch.cuda.Event(enable_timing=True),
    #                 torch.cuda.Event(enable_timing=True),
    #             )
    #             det_start.record()
    #             if self.config.DETECTION_TYPE == "motion":
    #                 # Motion Mode: Prepare boxes for drawing
    #                 if (
    #                     bbs_full_res is not None
    #                     and bbs_full_res.ndim == 2
    #                     and bbs_full_res.size(0) > 0
    #                 ):
    #                     scaled_resized_bbs = bbs_full_res / self.scales_tensor
    #                     bbs_to_send = scaled_resized_bbs.detach().cpu().tolist()
    #                 else:
    #                     bbs_to_send = []
    #                 data_to_draw = bbs_to_send
    #                 num_objs = len(bbs_to_send)
    #             else:
    #                 # Object Mode: Run YOLO and prepare metadata
    #                 det_frame = (
    #                     inf_data["full_frame"] if inf_data else device_frame
    #                 )  # RGB
    #                 metadata, _ = self.get_detections(
    #                     det_frame,
    #                     self.frame_count_target,
    #                     merged=bbs_full_res,
    #                     thickness=self.config.THICKNESS,
    #                     device_input=self.config.device_input,
    #                 )
    #                 data_to_draw = metadata
    #                 num_objs = len(metadata.keys())

    #             det_end.record()
    #             self.gpu_event_buffer["det"].append((det_start, det_end))

    #         # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
    #         if not self.config.TEST_MODE:
    #             display_source = (
    #                 inf_data["full_frame"]
    #                 if (inf_data and "full_frame" in inf_data)
    #                 else device_frame
    #             )  # RGB
    #             gpu_resized = F.interpolate(
    #                 display_source.unsqueeze(0).half(),
    #                 size=(self.resize_h, self.resize_w),
    #                 mode="bilinear",
    #                 align_corners=False,
    #             ).squeeze(0)
    #             display_frame = tensor2opencv(
    #                 gpu_resized, self.config.device_input, is_bgr=True
    #             )
    #             # display_frame = self.tensor2opencv_gpu(gpu_resized)
    #             self.render_queue.put((display_frame, data_to_draw, self.label_source))

    #             if self.config.DEBUG_FLAG:
    #                 gpu_resized = F.interpolate(
    #                     display_source.unsqueeze(0).float(),
    #                     size=(self.resize_h, self.resize_w),
    #                     mode="bilinear",
    #                     align_corners=False,
    #                 )  # RGB
    #                 self.debug_save_img(gpu_resized, self.frame_count_target)
    #                 self.debug_save_img_roi(
    #                     gpu_resized, bbs_full_res, self.frame_count_target
    #                 )

    #         # Async metrics stay at 0 for fairness; collected at the end of the video
    #         metrics["sf_time"] = 0
    #         metrics["roi_time"] = 0
    #         metrics["det_time"] = 0
    #     return num_objs, metrics

    # def yolo_cpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
    #     num_objs = 0
    #     metadata = {}
    #     metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

    #     # Only keep frames for TARGET_FPS
    #     if is_target_frame:
    #         self.next_process_idx += self.step_size
    #         self.frame_count_target += 1  # 1-indexed

    #         # Get detection at original resolution (No SF)
    #         t_start = time.perf_counter()
    #         metadata, _ = self.get_detections(
    #             device_frame,
    #             self.frame_count_target,
    #             thickness=self.config.THICKNESS,
    #             device_input=self.config.device_input,
    #         )
    #         num_objs = len(metadata.keys())
    #         metrics["det_time"] = (time.perf_counter() - t_start) * 1000

    #         if not self.config.TEST_MODE:
    #             cpu_resized = cv2.resize(device_frame, (self.resize_w, self.resize_h))
    #             display_frame = tensor2opencv(
    #                 cpu_resized, self.config.device_input, is_bgr=True
    #             )
    #             self.render_queue.put((display_frame, metadata, self.label_source))

    #     return num_objs, metrics

    # def yolo_gpu_pipeline_fn(self, frame_raw, overall_frame_num, is_target_frame):
    #     num_objs = 0
    #     metadata = {}
    #     metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

    #     # Only keep frames for TARGET_FPS
    #     if is_target_frame:
    #         self.next_process_idx += self.step_size
    #         self.frame_count_target += 1  # 1-indexed

    #         # Get detection at original resolution (No SF)
    #         det_start, det_end = (
    #             torch.cuda.Event(enable_timing=True),
    #             torch.cuda.Event(enable_timing=True),
    #         )
    #         det_start.record()
    #         metadata, _ = self.get_detections(
    #             frame_raw,
    #             self.frame_count_target,
    #             thickness=self.config.THICKNESS,
    #             device_input=self.config.device_input,
    #         )
    #         num_objs = len(metadata.keys())
    #         det_end.record()
    #         self.gpu_event_buffer["det"].append((det_start, det_end))

    #         # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
    #         if not self.config.TEST_MODE:
    #             gpu_resized = F.interpolate(
    #                 frame_raw.unsqueeze(0).half(),
    #                 size=(self.resize_h, self.resize_w),
    #                 mode="bilinear",
    #                 align_corners=False,
    #             ).squeeze(0)
    #             display_frame = tensor2opencv(
    #                 gpu_resized, self.config.device_input, is_bgr=True
    #             )
    #             # display_frame = self.tensor2opencv_gpu(gpu_resized)
    #             self.render_queue.put((display_frame, metadata, self.label_source))

    #         # Async metrics stay at 0 for fairness; collected at the end of the video
    #         metrics["sf_time"] = 0
    #         metrics["det_time"] = 0
    #     return num_objs, metrics


# INHERIT METHODS FROM HANDLERS -----------------------------------------------------------------
# TestSmartFilteringDetections.setup_reader = DeviceBaseHandler.setup_reader
# TestSmartFilteringDetections.get_frameWH = DeviceBaseHandler.get_frameWH
# TestSmartFilteringDetections.initialize_variables = (
#     DeviceBaseHandler.initialize_variables
# )
# TestSmartFilteringDetections.filter_contained_boxes = (
#     DeviceBaseHandler.filter_contained_boxes
# )
# TestSmartFilteringDetections.get_detections = DeviceBaseHandler.get_detections
# TestSmartFilteringDetections.prepare_pipeline = DeviceBaseHandler.prepare_pipeline
# TestSmartFilteringDetections.get_gpu_rois_by_area = (
#     DeviceBaseHandler.get_gpu_rois_by_area
# )
# TestSmartFilteringDetections.get_gpu_rois = DeviceBaseHandler.get_gpu_rois
# TestSmartFilteringDetections.get_cpu_rois = DeviceBaseHandler.get_cpu_rois
# TestSmartFilteringDetections.run_model = DeviceBaseHandler.run_model
# TestSmartFilteringDetections.model_warmup = DeviceBaseHandler.model_warmup

# # TestSmartFilteringDetections.cleanup_gpu = GPUStreamHandler.cleanup_gpu
# # TestSmartFilteringDetections.rbtd_full_gpu = GPUStreamHandler.rbtd_full_gpu
# # TestSmartFilteringDetections.prepare_gpu_pipeline = (
# #     GPUStreamHandler.prepare_gpu_pipeline
# # )
# # TestSmartFilteringDetections.allocate_gpu = GPUStreamHandler.allocate_gpu
# TestSmartFilteringDetections.gpu_warmup = GPUStreamHandler.gpu_warmup
# TestSmartFilteringDetections.apply_background_subtraction_gpu = (
#     GPUStreamHandler.apply_background_subtraction_gpu
# )

# TestSmartFilteringDetections.cleanup_cpu = CPUStreamHandler.cleanup_cpu
# # TestSmartFilteringDetections.rbtd_full_cpu = CPUStreamHandler.rbtd_full_cpu
# # TestSmartFilteringDetections.prepare_cpu_pipeline = (
# #     CPUStreamHandler.prepare_cpu_pipeline
# # )
# # TestSmartFilteringDetections.allocate_cpu = CPUStreamHandler.allocate_cpu
# TestSmartFilteringDetections.apply_background_subtraction_cpu = (
#     CPUStreamHandler.apply_background_subtraction_cpu
# )


def get_pytest_filter_expression(args):
    print("\n" + "=" * 50)
    print("TARGET SELECTION PREVIEW")
    print("=" * 50)

    filter_expression = "test_detections"
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
        print(f"  💻 Hardware Context Constraint: {args.device.upper()}")
        filter_expression = f"({filter_expression}) and {args.device.lower()}"
    else:
        print("  💻 Hardware Context Constraint: ALL AVAILABLE")

    print("=" * 50)
    print(f"COMPILED PYTEST KEYWORD EXPRESSION:\n   👉 {filter_expression}")
    print("=" * 50 + "\n")

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
        help="Filter by device (cpu or gpu)",
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
    parser.add_argument(
        "-n",
        type=int,
        default=100,
        dest="debug_frame_limit",
        help="Number of frames used for debugging [Default: 100]",
    )

    args = parser.parse_args()

    # UPDATE ENVIRONMENTAL VARIABLES
    os.environ["VIDEO_FILENAME"] = args.source
    os.environ["CUSTOM_MODEL_FLAG"] = "True" if args.custom_model_flag else "False"
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["DEBUG"] = "1" if args.debug else "0"
    os.environ["DEBUG_FRAME_LIMIT"] = str(args.debug_frame_limit)

    # detection_type, device, sf_enabled
    filter_expression = get_pytest_filter_expression(args)

    # PYTEST COMMAND
    pytest_args = [
        "-k",
        filter_expression,
        "-s",
        "-v",
        # "--log-cli-level=DEBUG",
        __file__,
    ]

    print(f"Launching tests for {args.source}")

    sys.exit(pytest.main(pytest_args))
