import argparse
import gc
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)
import multiprocessing as mp

import cv2
import numpy as np
import pytest
import tensorrt as trt
import torch
import torch.nn.functional as F

try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
import csv
import sys

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
    # nv12_to_rgb_torch
)
from include.models import get_model
from include.utils import (
    PipelineConfig,
    # PipelineMapping,
    draw_label,
    get_detection_color,
    # get_display_frame_in_bytes,
    # nv12_to_rgb_torch,
    tensor2opencv,
)

DEBUG_FLAG_DEFAULT = True if DEBUG_DEFAULT == "1" else False
os.environ["OMP_NUM_THREADS"] = "1"

force_export = False


def rendering_worker(queue, output_path, fps, size):
    """Background process that draws labels and writes to video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    while True:
        item = queue.get()
        if item is None:  # Sentinel value to stop the worker
            break

        # frame is resized
        # metadata in resized res
        display_frame, metadata_or_bbs, class_list = item

        # display_size = (self.resize_h, self.resize_w)
        # display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)

        if isinstance(metadata_or_bbs, dict):
            # Case: Object Detection
            for _, obj in metadata_or_bbs.items():
                bbox = obj["bbox"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                class_name = bbox["object"]
                class_id = class_list.index(class_name)
                confidence = bbox["object_det"]["confidence"]

                bb_color = get_detection_color(class_id, is_bgr=True)
                label = f"{class_name} {confidence:.2f}"

                # Draw on a copy of the frame to avoid modifying the original
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), bb_color, 2)
                # Replace draw_label if it's accessible; otherwise use cv2.putText
                # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                draw_label(display_frame, label, (x, y), color=bb_color, padding=5)
        elif metadata_or_bbs is not None:
            # Case: Motion Detections Only (SF Path)
            for box in metadata_or_bbs:
                if torch.is_tensor(box):
                    box = box.tolist()

                x1, y1, x2, y2 = map(int, box)
                # Draw on a copy of the frame to avoid modifying the original
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 225), 2)

        writer.write(display_frame)

    writer.release()


def fps_comparison_chart(
    chart_path, results, fps_key="Logic/Theo. Est. FPS (All frames)"
):
    try:
        # Extract names and FPS
        names = [r["Test Name"] for r in results]
        fps_values = [float(r[fps_key]) for r in results]

        plt.figure(figsize=(10, 6))
        plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

        colors = ["#2ca02c" if "gpu" in n.lower() else "#1f77b4" for n in names]

        bars = plt.bar(names, fps_values, color=colors, zorder=3)
        plt.ylabel("Frames Per Second (FPS)")
        plt.title(f"Performance Comparison: {chart_path.stem}")
        plt.xticks(rotation=45)

        # Add value labels on top of bars
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
        print(f"📈 Comparison chart saved to: {chart_path}")

    except ImportError:
        print("Skipping chart generation: matplotlib not installed.")


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
        test_dir
        / f"{current_test_filename}_results_{model_name}"
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

    # Initialize Model placeholder
    request.cls._shared_model = None
    request.cls._shared_model_path = None
    request.cls._shared_model_device = None
    request.cls._shared_model_sf_enabled = None
    yield
    # ... teardown logic here (like CSV export) ...
    # Final CSV Export
    # if request.cls.benchmarks:
    #     keys = request.cls.benchmarks[0].keys()
    #     with open(str(request.cls.result_dir / request.cls.csv_filename), "w", newline="") as f:
    #         dict_writer = csv.DictWriter(f, fieldnames=keys)
    #         dict_writer.writeheader()
    #         dict_writer.writerows(request.cls.benchmarks)
    #     print(f"\n[FINAL] Benchmarks saved to {request.cls.csv_filename}")

    # --- Final CSV Export & Speedup Calculation ---
    if request.cls.benchmarks:
        results = request.cls.benchmarks
        for row in results:
            if "gpu" in row["Test Name"].lower():
                # Find matching CPU run (e.g., 'sf_gpu' matches 'sf_cpu')
                match_name = row["Test Name"].replace("gpu", "cpu")
                cpu_row = next(
                    (r for r in results if r["Test Name"] == match_name), None
                )

                if cpu_row:
                    gpu_fps = float(row["Logic/Theo. Est. FPS (All frames)"])
                    cpu_fps = float(cpu_row["Logic/Theo. Est. FPS (All frames)"])
                    speedup = (gpu_fps / cpu_fps) if cpu_fps > 0 else 0
                    row["Speedup vs CPU"] = f"{speedup:.2f}x"
                else:
                    row["Speedup vs CPU"] = "N/A"
            else:
                row["Speedup vs CPU"] = "Baseline (CPU)"

        # Save to CSV with the new column
        keys = results[0].keys()
        with open(
            str(request.cls.result_dir / request.cls.csv_filename), "w", newline=""
        ) as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        print(f"\n[FINAL] Benchmarks saved to {request.cls.csv_filename}")
        # Print a quick summary to the console
        for r in results:
            print(
                f" > {r['Test Name']}: {r['Logic/Theo. Est. FPS (All frames)']} FPS | Speedup: {r.get('Speedup vs CPU', 'N/A')}"
            )

        # --- Generate Comparison Chart ---
        chart_path = (
            request.cls.result_dir
            / f"{request.cls.csv_filename.replace('.csv', '')}_theoFPS.png"
        )
        fps_comparison_chart(
            chart_path, results, fps_key="Logic/Theo. Est. FPS (All frames)"
        )

        chart_path = (
            request.cls.result_dir
            / f"{request.cls.csv_filename.replace('.csv', '')}_realFPS.png"
        )
        fps_comparison_chart(chart_path, results, fps_key="Real Est. FPS (All frames)")

    # Clean up the video handle if still open
    if hasattr(request.cls, "cap") and request.cls.cap.isOpened():
        request.cls.cap.release()


@pytest.fixture(autouse=True)
def each_test_setup(request):
    test_class_self = request.instance

    # --- [setUp logic] ---
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
    test_class_self.config = PipelineConfig(
        # GENERAL
        CUSTOM_MODEL_FLAG=os.getenv(
            "CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT
        ),  # True,
        DEVICE=device.upper(),
        OMIT_DETECTIONS_FLAG=True,
        TEST_MODE=False,
        DEBUG=os.getenv("DEBUG", DEBUG_DEFAULT),
        DEBUG_FRAME_LIMIT=os.getenv("DEBUG_FRAME_LIMIT", 100),
        # VIDEO WRITER
        CLIP_DURATION=None,
        # VDMS
        ENABLE_QUERYING=False,
        # MODEL
        MODEL_NAME=os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT),
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
    test_class_self.initialize_variables()
    test_class_self.prepare_pipeline()

    if device == "gpu" and torch.cuda.is_available():
        # Declare events on 'self' so all methods can see them
        test_class_self.ev_sf_start = torch.cuda.Event(enable_timing=True)
        test_class_self.ev_sf_end = torch.cuda.Event(enable_timing=True)
        test_class_self.ev_det_start = torch.cuda.Event(enable_timing=True)
        test_class_self.ev_det_end = torch.cuda.Event(enable_timing=True)
        test_class_self.gpu_event_buffer = {"sf": [], "roi": [], "det": []}

    # Initialize background video writer
    render_dir = test_class_self.result_dir / "rendered_videos"
    render_dir.mkdir(exist_ok=True)
    test_short_name = test_class_self._testMethodName.replace("test_", "")
    output_path = str(
        render_dir
        / f"annotated_{test_class_self.config.DETECTION_TYPE}_{test_class_self.video_path.stem}__{test_short_name}.mp4"
    )

    test_class_self.render_queue = mp.Queue()
    test_class_self.render_proc = mp.Process(
        target=rendering_worker,
        args=(
            test_class_self.render_queue,
            output_path,
            test_class_self.config.TARGET_FPS,
            (test_class_self.resize_w, test_class_self.resize_h),
        ),
    )
    test_class_self.render_proc.start()

    yield  # <--- The actual test function (e.g., test_pipeline) runs here

    # --- [tearDown logic] ---
    print(
        f"\n--- [TearDown] Memory Before Cleanup ({test_class_self._testMethodName}) ---"
    )
    test_class_self._print_gpu_mem()

    if hasattr(test_class_self, "reader") and test_class_self.reader is not None:
        test_class_self.reader.stop()
        test_class_self.reader = None

    # Signal the renderer to stop and wait for it to finish
    if hasattr(test_class_self, "render_queue"):
        test_class_self.render_queue.put(None)
        test_class_self.render_proc.join(timeout=10)

        # If it still hasn't closed, force terminate
        if test_class_self.render_proc.is_alive():
            test_class_self.render_proc.terminate()

        # Explicitly clear the queue and close the feeder thread
        test_class_self.render_queue.close()
        test_class_self.render_queue.join_thread()

    #  Nullify the model reference to trigger automatic cleanup
    if hasattr(test_class_self, "model") and test_class_self.model is not None:
        # Check if it's an Ultralytics model with a predictor
        predictor = getattr(test_class_self.model, "predictor", None)
        if predictor is not None:
            try:
                predictor.results = []
            except AttributeError:
                pass
        test_class_self.model = None

    #  Clear the singleton references to force a reload
    TestSmartFilteringDetections._shared_model = None
    TestSmartFilteringDetections._shared_model_path = None
    TestSmartFilteringDetections._shared_model_device = None

    # Force Python to run destructors NOW while streams are still alive
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Final sync to clear event queue
        time.sleep(0.2)

    if test_class_self.device == "GPU":
        test_class_self.cleanup_gpu()
    else:
        test_class_self.cleanup_cpu()

    #  Hard clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Critical for multi-process/multiprocessing cleanup

    print("--- [TearDown] Memory After Cleanup ---")
    test_class_self._print_gpu_mem()


@pytest.mark.usefixtures("setup_context")
class TestSmartFilteringDetections:
    # SETUP --------------------------------------------

    @pytest.mark.parametrize(
        "detection_type, device, sf_enabled",
        [
            ("motion", "cpu", True),  # SF enabled
            ("motion", "gpu", True),  # SF enabled
            ("object", "cpu", False),
            ("object", "cpu", True),  # SF enabled
            ("object", "gpu", False),
            ("object", "gpu", True),  # SF enabled
        ],
    )
    def test_pipeline(self, detection_type, device, sf_enabled):
        """Unified test runner for all configurations."""

        #  Run the actual model loader
        self.get_model_by_device(device, sf_enabled=self.config.sf_enabled)

        #  Select the correct pipeline function
        if device == "gpu":
            if (
                self.config.sf_enabled
                and self._shared_model_sf_enabled != self.config.sf_enabled
            ):
                pipeline_fn = self.yolo_gpu_pipeline_fn
            elif self.config.sf_enabled:
                pipeline_fn = self.sf_gpu_pipeline_fn
            else:
                pipeline_fn = self.yolo_gpu_pipeline_fn

            # pipeline_fn = self.pipeline_fn_gpu if sf_enabled else self.yolo_gpu_pipeline_fn
        else:
            if (
                self.config.sf_enabled
                and self._shared_model_sf_enabled != self.config.sf_enabled
            ):
                pipeline_fn = self.yolo_cpu_pipeline_fn
            elif self.config.sf_enabled:
                pipeline_fn = self.sf_cpu_pipeline_fn
            else:
                pipeline_fn = self.yolo_cpu_pipeline_fn
            # pipeline_fn = self.pipeline_fn_cpu if sf_enabled else self.yolo_cpu_pipeline_fn

        # Execute
        self.run_pipeline(pipeline_fn)

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
            # model_run_key, model_run_config, export=False
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
        self.model_path = TestSmartFilteringDetections._shared_model_path

        W, H = self.resize_w, self.resize_h
        if not sf_enabled:
            W, H = self.frame_width, self.frame_height
        self.model_warmup(H, W)

    def calculate_unique_coverage(self, merged_boxes, target_w=640, target_h=640):
        """Calculates unique pixel coverage on a 640p proxy to avoid CPU overhead."""
        if len(merged_boxes) == 0:
            return 0.0

        # Scale factors: 8K -> 640p
        scale_x = target_w / self.frame_width
        scale_y = target_h / self.frame_height

        # Create a small 640p mask (only 0.4 MB vs 33 MB for 8K)
        mask = np.zeros((target_h, target_w), dtype=np.uint8)

        for box in merged_boxes:
            x1, y1, x2, y2 = box
            # Draw filled rectangle on the mask
            cv2.rectangle(
                mask,
                (int(x1 * scale_x), int(y1 * scale_y)),
                (int(x2 * scale_x), int(y2 * scale_y)),
                1,
                -1,
            )

        return (np.count_nonzero(mask) / (target_w * target_h)) * 100

    def _finalize_benchmarks(
        self,
        n_frames,
        num_objs,
        total_pipeline_ms,
        real_world_latency_ms,
        coverage_percentages,
        sf_enabled,
    ):
        """Aggregates metrics and adds them to the results list."""
        latency_s = total_pipeline_ms / 1000.0
        est_fps = n_frames / latency_s if latency_s > 0 else 0
        est_fps_processed = self.frame_count_target / latency_s if latency_s > 0 else 0
        duration_s = n_frames / self.input_fps if self.input_fps > 0 else 0

        real_latency_s = real_world_latency_ms / 1000.0
        real_est_fps = n_frames / real_latency_s if real_latency_s > 0 else 0
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
                "Video Duration (s)": f"{duration_s:.4f}",
                "Video Frames": n_frames,
                "Video FPS": f"{self.input_fps:.2f}",
                "Pipeline Latency (s)": f"{latency_s:.2f}",
                "Frames Processed": self.frame_count_target,
                # Includes time by HW decoder (reader) and preparing output video
                "Real Est. FPS (All frames)": f"{real_est_fps:.2f}",
                "Real Est. FPS (Processed frames)": f"{real_est_fps_processed:.2f}",
                # Isolates algorithm from disk I/O bottlenecks
                "Logic/Theo. Est. FPS (All frames)": f"{est_fps:.2f}",
                "Logic/Theo. Est. FPS (Processed frames)": f"{est_fps_processed:.2f}",
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
            f"\n[{self._testMethodName}] Logic/Theo. Est. FPS (All frames): {est_fps:.2f} ({self.numFrames} frames)"
        )
        print(
            f"\n[{self._testMethodName}] Logic/Theo. Est. FPS (Processed frames): {est_fps_processed:.2f} ({self.frame_count_target} frames)"
        )

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
            img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
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
        if len(img_cpu.shape) == 3:
            # Swap RGB (Torch/Decoder) -> BGR (OpenCV)
            # img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
            # Only swap if the source is RGB (GPU Path)
            # CPU path is already BGR from OpenCV reader
            # if self.device_input == "cuda":
            #     img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
            # else:
            #     # Ensure it's contiguous for saving
            #     img_cpu = np.ascontiguousarray(img_cpu)
            pass
        else:
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

    # FRAME PROCESSORS --------------------------------------------
    def run_pipeline(self, pipeline_fn):
        n_frames = 0
        num_objs = 0
        self.frame_count_target = 0
        total_pipeline_time_ms = 0.0  # Track pure latency
        coverage_percentages = []
        self.component_stats = {"sf": [], "roi": [], "det": []}
        self.crops_per_frame_list = []

        total_session_start = time.perf_counter()
        # PRE-SYNC: Ensure GPU is idle before timing starts
        if self.device_input == "cuda":
            # Pre-allocate CUDA events for isolated GPU timing
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self.reader.start()

        # start_time = time.perf_counter()

        while self.active:  # and n_frames < 10*self.fps:
            # ret, frame = self.cap.read()
            device_frame, frame_num = self.reader.read()
            if device_frame is None:
                if self.reader.stopped:
                    self.active = False
                    break
                continue

            # if frame_num > 150:
            #     break

            # Keep 8K frame on GPU (Skip CPU conversion for non-target frames)
            # self.debug_save_img(device_frame, self.frame_count_target)
            # if self.device_input == "cuda":  # and not self.reader.is_h264_8k:
            #     device_frame = nv12_to_rgb_torch(
            #         device_frame,
            #         self.frame_height,
            #         self.frame_width,
            #         is_h264_8k=self.reader.is_h264_8k,
            #         is_bgr=False,
            #     )
            #     self.debug_save_img(device_frame, self.frame_count_target)

            n_frames += 1
            is_target_frame = frame_num >= self.next_process_idx
            nob = 0

            if self.device_input == "cuda":
                # start_eve   nt.record()
                nob, metrics = pipeline_fn(device_frame, frame_num, is_target_frame)
                # end_event.record()
                # torch.cuda.synchronize()
                # total_pipeline_time_ms += start_event.elapsed_time(end_event)
            else:
                # start_t = time.perf_counter()
                nob, metrics = pipeline_fn(device_frame, frame_num, is_target_frame)
                # total_pipeline_time_ms += (time.perf_counter() - start_t) * 1000

            num_objs += nob

            if is_target_frame and metrics != {}:
                num_crops = len(metrics["bbs"]) if metrics["bbs"] is not None else 0
                self.crops_per_frame_list.append(num_crops)

                self.component_stats["roi"].append(metrics["roi_time"])

                if self.device_input != "cuda":
                    self.component_stats["sf"].append(metrics["sf_time"])
                    self.component_stats["det"].append(metrics["det_time"])

                # Calculate coverage OUTSIDE the timed block to prevent interference
                if self.config.sf_enabled:
                    cov = self.calculate_unique_coverage(metrics["bbs"])
                    coverage_percentages.append(cov)

        # POST-SYNC: Ensure all GPU tasks finished before timing ends
        if self.device_input == "cuda":
            torch.cuda.synchronize()

        # ACTUAL speed including all overhead
        real_world_latency_ms = (time.perf_counter() - total_session_start) * 1000

        if self.device_input == "cuda":
            # Calculate SF Averages
            for start, end in self.gpu_event_buffer["sf"]:
                self.component_stats["sf"].append(start.elapsed_time(end))
            # Calculate Detection Averages
            for start, end in self.gpu_event_buffer["det"]:
                self.component_stats["det"].append(start.elapsed_time(end))
            # ROI
            # for start, end in self.gpu_event_buffer["roi"]:
            #     self.component_stats["roi"].append(start.elapsed_time(end))

        # Calculate Pure processing latency
        # This excludes: Reader, Tensor2OpenCV, Queueing, and Video Writing
        total_pipeline_time_ms = (
            sum(self.component_stats["sf"])
            + sum(self.component_stats["roi"])
            + sum(self.component_stats["det"])
        )

        # assert num_objs > 0
        if num_objs == 0:
            print(f" [WARNING] No objects detected for {self._testMethodName}")

        self._finalize_benchmarks(
            n_frames,
            num_objs,
            total_pipeline_time_ms,
            real_world_latency_ms,
            coverage_percentages,
            self.config.sf_enabled,
        )

    # TESTS --------------------------------------------
    def sf_cpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
        num_objs = 0
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Smart Filtering (Resize / Background Subtraction / Threshold / Dilate)
        t_start = time.perf_counter()
        inf_data = self.rbtd_full_cpu(device_frame)
        metrics["sf_time"] = (time.perf_counter() - t_start) * 1000

        # Only keep frames for TARGET_FPS
        # Videos are written for target frames only
        if is_target_frame:
            self.next_process_idx += self.step_size
            self.frame_count_target += 1  # 1-indexed

            metadata = {}
            bbs_to_send = []
            data_to_draw = []
            cpu_resized = None
            if inf_data:
                inf_data["frameNum"] = self.frame_count_target

                # Get
                t_start = time.perf_counter()
                bbs_full_res = self.get_cpu_rois(
                    inf_data["full_frame"],
                    self.frame_count_target,
                    inf_data["mask"],
                )
                metrics["roi_time"] = (time.perf_counter() - t_start) * 1000
                metrics["bbs"] = bbs_full_res

                if self.config.DEBUG_FLAG:
                    display_source = inf_data["mask"]
                    self.debug_save_mask(
                        display_source, self.frame_count_target, rois=bbs_full_res
                    )

                if self.config.DEBUG_FLAG or self.config.DETECTION_TYPE == "motion":
                    display_source = (
                        inf_data["full_frame"]
                        if (inf_data and "full_frame" in inf_data)
                        else device_frame
                    )
                    cpu_resized = cv2.resize(
                        display_source,
                        (self.resize_w, self.resize_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                if self.config.DEBUG_FLAG:
                    self.debug_save_img_roi(
                        cpu_resized, bbs_full_res, self.frame_count_target
                    )

                t_start = time.perf_counter()
                if self.config.DETECTION_TYPE == "motion":
                    # send bbnot self.config.TEST_MODE:
                    # disp_frame = tensor2opencv(cpu_resized, self.config.device_input)

                    # Resize for display and send to background worker
                    # disp_size = (self.disp_w, self.disp_h)
                    # disp_size = (self.resize_w, self.resize_h)
                    if (
                        bbs_full_res is not None
                        and bbs_full_res.ndim == 2
                        and bbs_full_res.size(0) > 0
                    ):
                        scaled_resized_bbs = bbs_full_res / self.scales_tensor
                        bbs_to_send = scaled_resized_bbs.cpu().tolist()
                    else:
                        bbs_to_send = []

                    # bbs_to_send = scaled_resized_bbs.cpu().tolist() if torch.is_tensor(scaled_resized_bbs) else scaled_resized_bbs
                    # self.render_queue.put((disp_frame, bbs_to_send, self.label_source))
                    data_to_draw = bbs_to_send
                    num_objs = len(bbs_full_res)
                else:
                    # Get detection for BBs at original resolution
                    metadata, frame_bytes = self.get_detections(
                        inf_data["full_frame"] if inf_data else device_frame,
                        self.frame_count_target,  # Frame used in metadata
                        merged=bbs_full_res,
                        thickness=self.config.THICKNESS,
                        device_input=self.config.device_input,
                    )
                    data_to_draw = metadata
                    num_objs = len(metadata.keys())

                metrics["det_time"] = (time.perf_counter() - t_start) * 1000

            # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
            if not self.config.TEST_MODE:
                display_source = (
                    inf_data["full_frame"]
                    if (inf_data and "full_frame" in inf_data)
                    else device_frame
                )
                cpu_resized = cv2.resize(display_source, (self.resize_w, self.resize_h))
                disp_frame = tensor2opencv(
                    cpu_resized, self.config.device_input, is_bgr=True
                )
                self.render_queue.put((disp_frame, data_to_draw, self.label_source))

        return num_objs, metrics

    def sf_gpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
        num_objs = 0
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Smart Filtering (Resize / Background Subtraction / Threshold / Dilate)
        sf_start, sf_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        sf_start.record()
        inf_data = self.rbtd_full_gpu(device_frame)
        sf_end.record()
        self.gpu_event_buffer["sf"].append((sf_start, sf_end))

        # Only keep frames for TARGET_FPS
        # Videos are written for target frames only
        if is_target_frame:
            self.next_process_idx += self.step_size
            self.frame_count_target += 1  # 1-indexed

            metadata = {}
            bbs_to_send = []
            data_to_draw = []
            if inf_data:
                inf_data["frameNum"] = self.frame_count_target

                # Get ROIs - time.perf_counter accurate since called self.bgs_stream.waitForCompletion()
                t_start = time.perf_counter()
                bbs_full_res = self.get_gpu_rois(
                    inf_data["full_frame"],
                    self.frame_count_target,
                    inf_data["mask"],
                )
                metrics["roi_time"] = (time.perf_counter() - t_start) * 1000
                # self.gpu_event_buffer["roi"].append((roi_start, roi_end))
                metrics["bbs"] = bbs_full_res

                if self.config.DEBUG_FLAG:
                    # self.resized_frame.download(self.bgs_stream, self.pinned_downloaded_resizedframe_np)
                    self.bgs_stream.waitForCompletion()
                    display_source = inf_data["mask"]
                    self.debug_save_mask(
                        display_source, self.frame_count_target, rois=bbs_full_res
                    )

                if self.config.DEBUG_FLAG:
                    display_source = (
                        inf_data["full_frame"]
                        if (inf_data and "full_frame" in inf_data)
                        else device_frame
                    )
                    gpu_resized = F.interpolate(
                        display_source.unsqueeze(0).float(),
                        size=(self.resize_h, self.resize_w),
                        mode="bilinear",
                        align_corners=False,
                    )  # RGB
                    self.debug_save_img(gpu_resized, self.frame_count_target)
                    self.debug_save_img_roi(
                        gpu_resized, bbs_full_res, self.frame_count_target
                    )

                det_start, det_end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                det_start.record()
                if self.config.DETECTION_TYPE == "motion":
                    # Motion Mode: Prepare boxes for drawing
                    if (
                        bbs_full_res is not None
                        and bbs_full_res.ndim == 2
                        and bbs_full_res.size(0) > 0
                    ):
                        scaled_resized_bbs = bbs_full_res / self.scales_tensor
                        bbs_to_send = scaled_resized_bbs.cpu().tolist()
                    else:
                        bbs_to_send = []
                    data_to_draw = bbs_to_send
                    num_objs = len(bbs_to_send)
                else:
                    # Object Mode: Run YOLO and prepare metadata
                    det_frame = (
                        inf_data["full_frame"] if inf_data else device_frame
                    )  # RGB
                    # if torch.is_tensor(det_frame):
                    #     det_frame = det_frame.flip(-3)
                    self.debug_save_img(det_frame, self.frame_count_target)
                    metadata, _ = self.get_detections(
                        det_frame,
                        self.frame_count_target,
                        merged=bbs_full_res,
                        thickness=self.config.THICKNESS,
                        device_input=self.config.device_input,
                    )
                    data_to_draw = metadata
                    num_objs = len(metadata.keys())

                det_end.record()
                self.gpu_event_buffer["det"].append((det_start, det_end))

            # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
            if not self.config.TEST_MODE:
                display_source = (
                    inf_data["full_frame"]
                    if (inf_data and "full_frame" in inf_data)
                    else device_frame
                )  # RGB
                # if torch.is_tensor(det_frame):
                # display_source = display_source.flip(-3)
                gpu_resized = F.interpolate(
                    display_source.unsqueeze(0).float(),
                    size=(self.resize_h, self.resize_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                self.debug_save_img(gpu_resized, self.frame_count_target)
                disp_frame = tensor2opencv(
                    gpu_resized, self.config.device_input, is_bgr=True
                )
                # disp_frame = self.letterbox_gpu(display_source, target_size=self.resize_h)

                # current_data = data_to_draw if is_target_frame else ({} if isinstance(data_to_draw, dict) else [])
                self.render_queue.put((disp_frame, data_to_draw, self.label_source))

            # Async metrics stay at 0 for fairness; collected at the end of the video
            metrics["sf_time"] = 0
            metrics["det_time"] = 0
        return num_objs, metrics

    def yolo_cpu_pipeline_fn(self, device_frame, overall_frame_num, is_target_frame):
        num_objs = 0
        metadata = {}
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Only keep frames for TARGET_FPS
        if is_target_frame:
            self.next_process_idx += self.step_size
            self.frame_count_target += 1  # 1-indexed

            # Get detection at original resolution (No SF)
            t_start = time.perf_counter()
            metadata, _ = self.get_detections(
                device_frame,
                self.frame_count_target,
                thickness=self.config.THICKNESS,
                device_input=self.config.device_input,
            )
            metrics["det_time"] = (time.perf_counter() - t_start) * 1000
            num_objs = len(metadata.keys())

            if not self.config.TEST_MODE:
                cpu_resized = cv2.resize(device_frame, (self.resize_w, self.resize_h))
                disp_frame = tensor2opencv(
                    cpu_resized, self.config.device_input, is_bgr=True
                )
                self.render_queue.put((disp_frame, metadata, self.label_source))

        return num_objs, metrics

    def yolo_gpu_pipeline_fn(self, frame_raw, overall_frame_num, is_target_frame):
        num_objs = 0
        metadata = {}
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Only keep frames for TARGET_FPS
        if is_target_frame:
            self.next_process_idx += self.step_size
            self.frame_count_target += 1  # 1-indexed

            # Get detection at original resolution (No SF)
            det_start, det_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            det_start.record()
            metadata, _ = self.get_detections(
                frame_raw,
                self.frame_count_target,
                thickness=self.config.THICKNESS,
                device_input=self.config.device_input,
            )
            num_objs = len(metadata.keys())
            det_end.record()
            self.gpu_event_buffer["det"].append((det_start, det_end))

            # --- OFFLOAD TO QUEUE (Run for EVERY target frame) ---
            if not self.config.TEST_MODE:
                gpu_resized = F.interpolate(
                    frame_raw.unsqueeze(0).float(),
                    size=(self.resize_h, self.resize_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                disp_frame = tensor2opencv(
                    gpu_resized, self.config.device_input, is_bgr=True
                )
                self.render_queue.put((disp_frame, metadata, self.label_source))

            # Async metrics stay at 0 for fairness; collected at the end of the video
            metrics["sf_time"] = 0
            metrics["det_time"] = 0
        return num_objs, metrics


TestSmartFilteringDetections.setup_reader = DeviceBaseHandler.setup_reader
TestSmartFilteringDetections.get_frameWH = DeviceBaseHandler.get_frameWH
TestSmartFilteringDetections.initialize_variables = (
    DeviceBaseHandler.initialize_variables
)
TestSmartFilteringDetections.filter_contained_boxes = (
    DeviceBaseHandler.filter_contained_boxes
)
TestSmartFilteringDetections.get_detections = DeviceBaseHandler.get_detections
TestSmartFilteringDetections.prepare_pipeline = DeviceBaseHandler.prepare_pipeline
TestSmartFilteringDetections.get_gpu_rois_by_area = (
    DeviceBaseHandler.get_gpu_rois_by_area
)
TestSmartFilteringDetections.get_gpu_rois = DeviceBaseHandler.get_gpu_rois
TestSmartFilteringDetections.get_cpu_rois = DeviceBaseHandler.get_cpu_rois
TestSmartFilteringDetections.run_model = DeviceBaseHandler.run_model
TestSmartFilteringDetections.model_warmup = DeviceBaseHandler.model_warmup

TestSmartFilteringDetections.cleanup_gpu = GPUStreamHandler.cleanup_gpu
TestSmartFilteringDetections.rbtd_full_gpu = GPUStreamHandler.rbtd_full_gpu
TestSmartFilteringDetections.prepare_gpu_pipeline = (
    GPUStreamHandler.prepare_gpu_pipeline
)
TestSmartFilteringDetections.allocate_gpu = GPUStreamHandler.allocate_gpu
TestSmartFilteringDetections.gpu_warmup = GPUStreamHandler.gpu_warmup
TestSmartFilteringDetections.apply_background_subtraction_gpu = (
    GPUStreamHandler.apply_background_subtraction_gpu
)

TestSmartFilteringDetections.cleanup_cpu = CPUStreamHandler.cleanup_cpu
TestSmartFilteringDetections.rbtd_full_cpu = CPUStreamHandler.rbtd_full_cpu
TestSmartFilteringDetections.prepare_cpu_pipeline = (
    CPUStreamHandler.prepare_cpu_pipeline
)
TestSmartFilteringDetections.allocate_cpu = CPUStreamHandler.allocate_cpu
TestSmartFilteringDetections.apply_background_subtraction_cpu = (
    CPUStreamHandler.apply_background_subtraction_cpu
)


if __name__ == "__main__":
    # TEST ARGUMENTS
    parser = argparse.ArgumentParser(description="Run Video Detection Pipeline Tests")
    parser.add_argument(
        "-v",
        "--video",
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
        choices=["cpu", "gpu"],
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
    os.environ["VIDEO_FILENAME"] = args.video
    os.environ["CUSTOM_MODEL_FLAG"] = "True" if args.custom_model_flag else "False"
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["DEBUG"] = "1" if args.debug else "0"
    os.environ["DEBUG_FRAME_LIMIT"] = str(args.debug_frame_limit)

    # detection_type, device, sf_enabled
    run_args = []
    if args.detection_type:
        run_args.append(args.detection_type)
    if args.device:
        run_args.append(args.device)
    if args.sf_enabled:
        run_args.append(str(args.sf_enabled))

    # PYTEST COMMAND
    pytest_args = ["-v", __file__]
    if run_args:
        pytest_args.extend(["-k", " and ".join(run_args)])

    print(f"Launching tests for {args.video}")

    sys.exit(pytest.main(pytest_args))
