import argparse
import asyncio
import csv
import faulthandler
import gc
import threading
import time
import traceback

faulthandler.enable()
import inspect
import logging
import multiprocessing as mp
import os
import queue
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorrt as trt
import torch

# # Force all PyTorch extension handles to compile and load BEFORE threads spawn
# if torch.cuda.is_available():
#     _ = torch.zeros(1).cuda()
# import torch._ops
# import torch.utils

# Explicitly invoke any plotting or module style lookups early

sys.path.insert(1, str(Path(__file__).parent.parent))
from include.default_configs import (
    CUSTOM_MODEL_FLAG_DEFAULT,
    DEBUG_DEFAULT,
    MODEL_NAME_DEFAULT,
    THRESHOLD_VALUE,
)
from include.handlers import (
    AsyncVideoWriter,
    CPUStreamHandler,
    DeviceBaseHandler,
    GPUStreamHandler,
    get_bb_overlay,
    get_metadata_overlay,
    log_to_logger,
)
from include.utils import (
    PipelineConfig,
)

try:
    # Retain standard spawn mode to prevent CUDA context driver deadlocks
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

DEBUG_FLAG_DEFAULT = True if DEBUG_DEFAULT == "1" else False
os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())  # "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
force_export = False

# Force OpenCV to run sequentially to prevent context-switching overhead
cv2.setNumThreads(0)  # Forces OpenCV loops to run strictly sequentially


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


class DummyProcess:
    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False

    def close(self):
        pass


@pytest.fixture(scope="class")
def setup_context(request):
    """Replaces setUpClass: Runs once per test class."""
    current_test_filename = Path(__file__).stem
    test_dir = Path(__file__).parent
    main_path = test_dir.parent
    video_dir = main_path / "inputs"

    request.cls._shared_model = None
    request.cls._shared_model_path = None
    request.cls._shared_model_device = None
    request.cls._shared_model_sf_enabled = None

    # model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    # Handler.__init__ (main items)
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
    request.cls.result_dir = (
        test_dir / f"{current_test_filename}_results/{model_name}" / request.cls.name
    )
    request.cls.result_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark statistics
    request.cls.benchmarks = []
    request.cls.csv_filename = (
        f"detections_benchmarks_{model_name}_{request.cls.name}.csv"
    )
    request.cls.csv_path = request.cls.result_dir / request.cls.csv_filename

    request.cls.active = True
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

        keys = results[0].keys()

        with open(str(request.cls.csv_path), "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        print(f"\n[FINAL] Benchmarks saved to {request.cls.csv_path}")

        for r in results:
            print(
                f" > {r['Test Name']}: {r['Pipeline FPS (Video frames)']} FPS | {r['Pipeline FPS (Target frames)']} FPS | {r['Avg Frame Reading (ms)']} ms | Speedup: {r.get('Pipeline Speedup vs CPU', 'N/A')}"
            )

        chart_path = (
            request.cls.result_dir
            / f"{request.cls.csv_filename.replace('.csv', '')}_pipelineFPS.png"
        )
        fps_comparison_chart(chart_path, results, fps_key="Pipeline FPS (Video frames)")

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


@pytest.fixture(autouse=True)
def each_test_setup(request):
    test_class_self = request.instance
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    device = request.node.callspec.params.get("device")
    detection_type = request.node.callspec.params.get("detection_type")
    sf_enabled = request.node.callspec.params.get("sf_enabled")

    if detection_type == "motion" and not sf_enabled:
        pytest.skip(
            "Pure YOLO mode is structurally invalid for detection_type 'motion'.\n"
        )

    test_class_self._testMethodName = (
        f"sf_{detection_type}_{device}"
        if sf_enabled
        else f"yolo_{detection_type}_{device}"
    )

    # video_output_name = f"{test_class_self._testMethodName}_detections_output.mp4"
    vid_dir = test_class_self.result_dir / device
    vid_dir.mkdir(parents=True, exist_ok=True)

    # Handler.__init__ (remaining items)
    # 1. Re-initialize a fresh configuration object
    test_class_self.config = PipelineConfig(
        SHARED_OUTPUT=str(test_class_self.result_dir),
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

    # test_video_output_path = os.path.join(str(vid_dir), video_output_name)
    os.environ["TEST_SUITE_RENDER_DIR"] = str(vid_dir)
    # test_class_self.config.SHARED_OUTPUT = str(test_class_self.result_dir)
    # test_class_self.disp_w, test_class_self.disp_h = (640, 360)
    test_class_self.disp_w, test_class_self.disp_h = (
        test_class_self.config.DISPLAY_FRAME_SIZE
    )

    if device == "gpu":
        # Allocate a permanent float32/float16 channel layout space directly on VRAM
        test_class_self.static_gpu_360p = torch.empty(
            (1, 3, test_class_self.disp_h, test_class_self.disp_w),
            dtype=torch.float32,
            device="cuda",
        )
        test_class_self.static_gpu_byte_bchw = torch.empty(
            (1, 3, test_class_self.disp_h, test_class_self.disp_w),
            dtype=torch.uint8,
            device="cuda",
        ).contiguous()

        # Create two isolated tracking canvases to handle the ping-pong data stream
        test_class_self.static_host_canvases = [
            np.zeros(
                (test_class_self.disp_h, test_class_self.disp_w, 3), dtype=np.uint8
            ),
            np.zeros(
                (test_class_self.disp_h, test_class_self.disp_w, 3), dtype=np.uint8
            ),
        ]
        test_class_self.canvas_selector = 0

        # Register BOTH buffers as page-locked memory
        cv2.cuda.registerPageLocked(test_class_self.static_host_canvases[0])
        cv2.cuda.registerPageLocked(test_class_self.static_host_canvases[1])

    # Reset core state machine properties before invoking any backend handlers
    test_class_self.active = True

    test_class_self.loop = asyncio.get_event_loop()
    test_class_self.frame_ready_event = asyncio.Event()
    test_class_self._is_stopped = False
    test_class_self._stop_lock = threading.Lock()  # Local lock for this instance
    test_class_self.mp_frame_ready_event = mp.Event()
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

    test_class_self.run_realtime_inference = types.MethodType(
        test_class_self.__class__.run_realtime_inference, test_class_self
    )

    test_class_self.setup_threads()

    yield

    # Teardown logic
    test_class_self.active = False

    # Fallback to production reader safety closure
    if hasattr(test_class_self, "reader") and test_class_self.reader is not None:
        try:
            if hasattr(test_class_self.reader, "print_breakdown"):
                test_class_self.reader.print_breakdown()
            test_class_self.reader.stop()
        except Exception:
            pass

    if (
        hasattr(test_class_self, "stop_writer")
        and test_class_self.stop_writer is not None
    ):
        try:
            test_class_self.stop_writer.set()
        except Exception:
            pass

    # --- CRITICAL RE-ORDERED FLUSH GATE ---
    # 1. First, block and wait for the video processing producer thread to completely finish
    # reading all remaining frames from the input video file.
    producer_thread_handle = getattr(test_class_self, "process_thread", None)
    if producer_thread_handle is not None and producer_thread_handle.is_alive():
        producer_thread_handle.join(timeout=0.5)  # 10.0)

    # 2. Block and wait for background asynchronous AI executor workers to finish inference tasks.
    if hasattr(test_class_self, "executor") and test_class_self.executor:
        test_class_self.executor.shutdown(wait=True)

    # 3. Allow the multiprocessing render queue to naturally empty its frames into the encoder pipe.
    if (
        hasattr(test_class_self, "render_queue")
        and test_class_self.render_queue is not None
    ):
        try:
            # Loop safely until the render loop handles the last remaining frames
            # while not test_class_self.render_queue.empty():
            #     time.sleep(0.05)

            # Send a poison pill token ONLY after all threads have fully stopped putting data into it
            # test_class_self.render_queue.put(None, timeout=1.0)
            if (
                hasattr(test_class_self, "render_queue")
                and test_class_self.render_queue is not None
            ):
                try:
                    # Forcefully clear the queue thread tracking matrices
                    test_class_self.render_queue.close()
                    test_class_self.render_queue.cancel_join_thread()
                except Exception:
                    pass
        except Exception:
            pass

    # 4. Gracefully join the background FFmpeg process, avoiding premature kills
    render_proc_handle = getattr(test_class_self, "render_proc", None)
    if render_proc_handle is not None:
        try:
            if render_proc_handle.is_alive():
                # Grant up to 10 seconds for disk I/O synchronization
                render_proc_handle.join(timeout=0.5)  # 10.0)

            if render_proc_handle.is_alive():
                print(
                    "[WARN] Render worker hung during final flush. Forcing termination."
                )
                render_proc_handle.terminate()
                render_proc_handle.join()

            render_proc_handle.close()
        except Exception:
            pass
        setattr(test_class_self, "render_proc", None)

    # 4. Deep System Unlinking of Shared Memory Layers (Parent/Child safe)
    # This loop uses direct reference mapping rather than guessing PIDs
    for shm_pool_attr in ["ai_shms", "shms"]:
        if hasattr(test_class_self, shm_pool_attr):
            shm_pool = getattr(test_class_self, shm_pool_attr)
            if shm_pool:
                for shm_segment in list(shm_pool):
                    try:
                        shm_segment.close()
                        shm_segment.unlink()
                    except Exception:
                        pass
                shm_pool.clear()

    # 5. Clean up the Sync Base Manager Process
    if hasattr(test_class_self, "manager") and test_class_self.manager is not None:
        try:
            test_class_self.manager.shutdown()
        except Exception:
            pass
        test_class_self.manager = None

    # CRITICAL FIX: Block and wait for the background producer thread to die
    # BEFORE we delete the reader attribute from the namespace map.
    producer_thread_handle = getattr(test_class_self, "process_thread", None)
    if producer_thread_handle is not None and producer_thread_handle.is_alive():
        producer_thread_handle.join(timeout=5.0)

    # if hasattr(test_class_self, "ingest_ring") and test_class_self.ingest_ring:
    #     for mat in test_class_self.ingest_ring:
    #         try:
    #             cv2.cuda.unregisterPageLocked(mat)
    #         except Exception:
    #             pass
    #     test_class_self.ingest_ring.clear()

    # if hasattr(test_class_self, "pinned_matrices") and test_class_self.pinned_matrices:
    #     for active_mat in test_class_self.pinned_matrices:
    #         try:
    #             cv2.cuda.unregisterPageLocked(active_mat)
    #         except Exception:
    #             pass
    #     test_class_self.pinned_matrices.clear()
    #     if hasattr(test_class_self, "pinned_tensors"):
    #         test_class_self.pinned_tensors.clear()
    #     if hasattr(test_class_self, "ai_pinned_tensors"):
    #         test_class_self.ai_pinned_tensors.clear()

    try:
        cv2.cuda.unregisterPageLocked(test_class_self.pinned_matrices)
    except Exception:
        pass

    # ADD THIS MECHANISM:
    if hasattr(test_class_self, "static_host_canvas"):
        try:
            cv2.cuda.unregisterPageLocked(test_class_self.static_host_canvas)
        except Exception:
            pass

    if (
        not test_class_self.active
    ):  # Only execute this block during complete class teardown
        if hasattr(test_class_self, "ingest_ring") and test_class_self.ingest_ring:
            for mat in test_class_self.ingest_ring:
                try:
                    cv2.cuda.unregisterPageLocked(mat)
                except Exception:
                    pass
            test_class_self.ingest_ring.clear()

        if (
            hasattr(test_class_self, "pinned_matrices")
            and test_class_self.pinned_matrices
        ):
            for active_mat in test_class_self.pinned_matrices:
                try:
                    cv2.cuda.unregisterPageLocked(active_mat)
                except Exception:
                    pass
            test_class_self.pinned_matrices.clear()

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
        # "io_executor",
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
            if device == "gpu":
                torch.cuda.memory._record_memory_history(enabled=False)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    gc.collect()
    time.sleep(0.2)


@pytest.mark.usefixtures("setup_context")
class TestSmartFilteringDetections:
    # SETUP --------------------------------------------

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize("sf_enabled", [True, False])
    @pytest.mark.parametrize("detection_type", ["motion", "object"])
    def test_detections(self, device, sf_enabled, detection_type):
        """Unified test runner for all configurations."""
        # if detection_type == "motion" and not sf_enabled:
        #     pytest.skip(
        #         "Pure YOLO mode is structurally invalid for detection_type 'motion'.\n"
        #     )

        self.duration_target = 30

        # Trigger your clean overridden thread boot
        self.start()

        print(
            f"\n[TEST HARNESS] Execution Started. Monitoring variables natively for {self.duration_target}s...",
            flush=True,
        )
        # Explicit timer tracking allows background multiprocessing queues to stream uninhibited
        # test_start_time = time.perf_counter()

        while self.active or not self._is_stopped:
            # Check if our active benchmark validation timeline target has been fulfilled
            # if (time.perf_counter() - test_start_time) >= float(self.duration_target) + 15.0:
            #     print("\n[TEST HARNESS] Targeted test runtime duration reached. Initiating clean multi-process flush...", flush=True)
            #     break
            time.sleep(0.1)  # Stay clear of the hot pipeline lane

        assert self.status == "DONE"

    def setup_threads(self):
        """Overrides handlers.py to bind threads dynamically to the test instance."""
        self.setup_shared_memory()  # Natively sets up Manager dictionary and buffers

        # Executor for Async YOLO tasks and FFmpeg re-encoding
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        # self.clip_executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        print(
            f"sf_enabled: {self.config.sf_enabled}\tTEST_MODE: {self.config.TEST_MODE}",
            flush=True,
        )

        # Open up looking-ahead buffer horizons to eliminate 8K queue backpressure stalls
        self.signal_queue = mp.Queue(maxsize=128)  # 10)
        self.render_queue = mp.Queue(maxsize=256)  # 64)

        test_dir = os.getenv(
            "TEST_SUITE_RENDER_DIR", str(Path(self.config.SHARED_OUTPUT))
        )
        os.makedirs(test_dir, exist_ok=True)

        if self.source.startswith("rtsp"):
            short_name = "rtsp"
        else:
            short_name = Path(self.source).stem
        video_output_name = f"{self._testMethodName}_{short_name}.mp4"
        out_path = os.path.join(test_dir, video_output_name)
        self.output_path = (
            out_path  # f"{self.config.SHARED_OUTPUT}/{short_name}_{self.device}.mp4"
        )

        # log_to_logger(
        #     f"[TEST MODE] Detection results saved to: {out_path}", level="info"
        # )

        # =====================================================================
        # DYNAMIC THREAD TARGET REDIRECTION
        # =====================================================================
        # Lambda forces the background processing thread to evaluate your custom staging loop natively
        self.process_thread = threading.Thread(
            target=lambda: self.run_realtime_inference(
                sf_enabled=self.config.sf_enabled
            ),
            daemon=True,
        )
        # self.render_proc = mp.Process(  # threading.Thread(
        #     target=rendering_worker,
        #     args=(
        #         self.render_queue,
        #         (self.disp_w, self.disp_h),
        #         self.output_path,
        #         self.target_fps,
        #     ),
        #     daemon=True,
        # )
        # self.render_proc = threading.Thread(target=lambda: None, daemon=True)
        # self.display_proc = threading.Thread(target=lambda: None, daemon=True)
        self.render_proc = DummyProcess()
        self.display_proc = DummyProcess()

        self.async_writer = AsyncVideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"avc1"),
            float(self.target_fps),
            (self.disp_w, self.disp_h),
        )
        # self.async_writer = None
        log_to_logger(
            f"[TEST MODE] Detection results saved to: {self.output_path}", level="info"
        )

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

        if self.config.ENABLE_QUERYING:
            self._initialize_writer()

        # Small delay to allow the reader's deque to populate
        time.sleep(0.1)

        # Start the producer and consumer threads
        if hasattr(self, "process_thread") and not self.process_thread.is_alive():
            self.process_thread.start()

        if not self.config.DISABLE_DETECTION:
            self.render_proc.start()

            self.display_proc.start()

        if (
            self.config.ENABLE_QUERYING
            and not self.config.TEST_MODE
            and not self.metadata_thread.is_alive()
        ):
            self.metadata_thread.start()

        if self.config.ENABLE_QUERYING and not self.writer_thread.is_alive():
            self.writer_thread.start()

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
        num_objs,
        total_written_frames,
        total_pipeline_ms,
        real_world_latency_ms,
        total_sf_pipeline_ms,
        frame_loop_latency_ms,
        coverage_percentages,
        total_read_time,
        total_queue_saturation,
        queue_saturation_history,
    ):
        """Aggregates metrics and adds them to the results list."""
        sf_enabled = self.config.sf_enabled
        stat_frame_count = self.stat_frame_count
        stat_fps = self.stat_fps

        h2d_label = (
            "Pinned H2D Transfer (PCIe DMA)"
            if self.is_cuda
            else "Data Preparation Overhead"
        )
        # d2h_label = (
        #     "D2H Transfer (PCIe Download)"
        #     if self.is_cuda
        #     else "Array Extraction Overhead"
        # )

        total_pipeline_s = total_pipeline_ms / 1000.0  # Full pipeline

        total_real_pipeline_s = (
            real_world_latency_ms / 1000.0
        )  # Pipeline until last frame sent to writer
        real_est_fps = (
            self.frame_count_target / total_real_pipeline_s
            if total_real_pipeline_s > 0
            else 0
        )

        # if total_written_frames > 0:
        duration_s = (
            self.reader.total_input_frames / self.reader.input_fps
            if self.reader.input_fps > 0
            else 0
        )
        out_duration_s = (
            total_written_frames / self.reader.target_fps
            if self.reader.target_fps > 0
            else 0
        )
        print(
            f"Expected duration: {self.duration_target:.2f} stream_duration_s: {duration_s:.2f} output_duration_s: {out_duration_s:.2f}"
        )

        h2d_display_label = f"Reader {h2d_label}:"
        avg_reader_ms = (total_read_time / total_written_frames) * 1000
        # avg_resize_ms = (total_resize_time / total_written_frames)*1000
        # avg_write_ms = (total_disk_write_overhead / total_written_frames)*1000
        target_frame_fps = total_written_frames / total_pipeline_s
        all_frame_fps = self.reader.total_input_frames / total_pipeline_s
        avg_copy_ms = (self.reader.total_shm_copy_time / total_written_frames) * 1000
        # avg_blocked_ms = (
        #     self.reader.total_queue_wait_time / total_written_frames
        # ) * 1000

        # Avg Blocked MS tracks your downstream GIL blocks and serialization stalls
        avg_blocked_ms = (
            sum(self.component_stats.get("queue_blocked", [0.0]))
            / max(1, len(self.component_stats.get("queue_blocked", [])))
            if self.component_stats.get("queue_blocked")
            else 0.0
        )

        avg_saturation = (total_queue_saturation / total_written_frames) * 100
        peak_saturation = (
            max(queue_saturation_history) * 100 if queue_saturation_history else 0
        )
        target_match_ratio = (
            total_written_frames / (total_pipeline_s * self.reader.target_fps)
        ) * 100

        frame_drop = max(
            0.0,
            (1.0 - (total_written_frames / max(1, self.reader.total_input_frames)))
            * 100.0,
        )
        serial_pressure = (avg_blocked_ms / max(0.1, frame_loop_latency_ms)) * 100.0
        evac_vel = target_frame_fps / max(0.1, all_frame_fps)

        # Isolate the ratio of sequential loop stalls against full execution bounds
        backpressure_index = (avg_blocked_ms / max(0.1, frame_loop_latency_ms)) * 100.0

        if self.is_cuda:
            avg_h2d_ms = (self.reader.total_h2d_time / total_written_frames) * 1000
            avg_prep_ms = 0.0
        else:
            avg_prep_ms = (self.reader.total_h2d_time / total_written_frames) * 1000
            avg_h2d_ms = 0.0

        # Calculate structural frame size payload in Gigabytes (H * W * Channels)
        frame_size_gb = (self.reader.frame_height * self.reader.frame_width * 3) / (
            1024**3
        )

        # 1. Compute Peak H2D PCIe Throughput
        avg_h2d_s = (
            (
                sum(self.component_stats["dma_upload"])
                / len(self.component_stats["dma_upload"])
            )
            / 1000.0
            if self.component_stats["dma_upload"]
            else 0
        )
        if self.is_cuda and avg_h2d_s > 0.00001:
            self.pcie_throughput_gbps = frame_size_gb / avg_h2d_s
        else:
            self.pcie_throughput_gbps = (
                0.0  # Safe fallback baseline for non-PCIe pipelines (CPU)
            )

        # Calculate bus saturation relative to standard Gen4 x16 baseline ceilings (31.5 GB/s)
        # Scale to an index percentage safely
        pcie_bus_saturation = (self.pcie_throughput_gbps / 31.5) * 100.0

        # 2. Extract PyTorch VRAM Fragmentation Delta Metrics
        if self.is_cuda and torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(0) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(0) / 1024**2
            self.vram_efficiency = (
                (peak_alloc / peak_reserved * 100.0) if peak_reserved > 0 else 100.0
            )
        else:
            self.vram_efficiency = (
                0.0  # Clean zero baseline representation for CPU runs
            )

        # Calculate averages for component breakdowns
        # avg_reader_ms = (total_read_time / self.reader.total_input_frames) * 1000
        # avg_reader_ms = (total_read_time / self.frame_count_target) * 1000
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

        # Total Latency Sum (SF + ROI + DET)
        det_latency_ms = avg_sf + avg_roi + avg_det
        det_latency_s = det_latency_ms / 1000.0
        det_est_fps = (total_written_frames / det_latency_s) if det_latency_s > 0 else 0

        # Track model workloads against preprocessing overheads (sf_time + roi_time)
        preprocessing_overhead = avg_sf + avg_roi
        model_cost_density = avg_det / max(0.1, preprocessing_overhead)

        loop_overhead = max(
            0.0, 100.0 - ((det_latency_ms / max(1.0, frame_loop_latency_ms)) * 100.0)
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

        # Calculate how often we hit a high-motion cap (e.g., 20 crops)
        capped_frames = sum(1 for c in self.crops_per_frame_list if c >= 20)
        cap_rate = (
            (capped_frames / len(self.crops_per_frame_list)) * 100
            if self.crops_per_frame_list
            else 0
        )

        # avg_queue = (
        #     sum(self.component_stats["queue_blocked"])
        #     / len(self.component_stats["queue_blocked"])
        #     if self.component_stats["queue_blocked"]
        #     else 0
        # )
        # avg_batch = (
        #     sum(self.component_stats["batch_sizes"])
        #     / len(self.component_stats["batch_sizes"])
        #     if self.component_stats["batch_sizes"]
        #     else 0
        # )

        backlog_list = self.component_stats.get("thread_backlog", [])
        avg_backlog = (
            sum(backlog_list) / len(backlog_list) if len(backlog_list) > 0 else 0.0
        )

        print("\n" + "=" * 60)
        print(
            f"     FULLY-OPTIMIZED ASYNC STAGE ({self.config.DEVICE}) PIPELINE BREAKDOWN  "
        )
        print("=" * 60)
        print(f"Total Output Frames Written:   {total_written_frames}")
        print(f"Total Pipeline Execution Time: {total_pipeline_s:.4f} seconds")
        print(f"Overall Processing Speed:      {target_frame_fps:.2f} FPS")
        print("-" * 60)
        print("MAIN CONSUMER LOOP TIMELINE (SEQUENTIAL OVERHEAD):")
        print(f" 1. Shared Memory Copy to Host:                {avg_copy_ms:6.2f} ms")
        print(
            f" 2. GIL / Downstream Queue Serialization Stalls: {avg_blocked_ms:6.2f} ms"
        )
        print(f" 3. Pure Video Frame File-Ingestion Read:      {avg_reader_ms:6.2f} ms")
        print("-" * 60)
        print("BACKGROUND INGESTION HEALTH (ASYNC METRICS):")
        print(
            f" A. Inbound Stream Decode Speed:               {all_frame_fps:6.2f} FPS"
        )
        print(f" B. {h2d_display_label:<42} {avg_h2d_ms:6.2f} ms")
        print(f" C. Consumer Queue Saturation Density:         {avg_saturation:6.1f}%")
        print(
            f" D. Peak PCIe Upload Throughput:               {self.pcie_throughput_gbps:6.2f} GB/s"
        )
        print(
            f" E. Active Thread Pool Work Backlog:           {avg_backlog:6.1f} tasks"
        )
        print(
            f" F. VRAM Hardware Memory Efficiency (Cache):   {self.vram_efficiency:6.1f}%"
        )
        print("-" * 60)

        # EXPANDED PERFORMANCE INSIGHTS AND DIAGNOSTICS
        print("PIPELINE HEALTH & BEHAVIOR INSIGHTS:")
        if avg_saturation > 80.0:
            pipeline_status = "CHOKED"
            print(
                f" • Status: \033[91m🔴 {pipeline_status}\033[0m (Downstream logic cannot keep up with inbound stream speed)"
            )
        elif avg_saturation > 40.0:
            pipeline_status = "BALANCED"
            print(
                f" • Status: \033[93m🟡 {pipeline_status}\033[0m (Queue buffer is actively pacing consumer workloads)"
            )
        else:
            pipeline_status = "IDLE/LIGHT"
            print(
                f" • Status: \033[92m🟢 {pipeline_status}\033[0m (Consumer loop finishes ahead of background ingestion clock)"
            )

        print(f" • Peak Queue Fullness Reached:               {peak_saturation:.1f}%")
        # Calculate stream drop indicator
        print(f" • Targeted Pacing Delivery Accuracy:        {target_match_ratio:.1f}%")
        print(f" • Downstream Serialization Pressure Index:   {serial_pressure}")
        print(f" • Core Queue Evacuation Velocity Rank:       {evac_vel}")
        print(
            f" • Hardware Compute Backpressure Index:        {backpressure_index:6.2f}%"
        )
        print(
            f" • Preprocessing to Inference Cost Density:    {model_cost_density:6.2f}x"
        )
        print(
            f" • Physical PCIe Gen4 Bus Saturation Level:    {pcie_bus_saturation:6.2f}%"
        )
        print("=" * 60)

        self.__class__.benchmarks.append(
            {
                "Test Name": self._testMethodName,
                "Detection Type": self.config.DETECTION_TYPE,
                "Device": self.config.DEVICE,
                "Smart Filtering": "Enabled" if sf_enabled else "Disabled",
                "Video": self.source,  # self.name?
                "Video FPS": f"{self.reader.input_fps:.2f}",
                "Video Original Duration (s)": f"{duration_s:.4f}",
                "Video Frames": self.reader.total_input_frames,
                # PIPELINE OVERVIEW
                "Target FPS": f"{self.reader.target_fps:.2f}",
                "Output Duration (s)": f"{out_duration_s:.4f}",
                "Output Frames": total_written_frames,
                "Pipeline Latency (s)": f"{total_pipeline_s:.2f}",
                "Pipeline FPS (Video frames)": f"{all_frame_fps:.2f}",
                "Pipeline FPS (Target frames)": f"{target_frame_fps:.2f}",
                "Real Pipeline Latency (s)": f"{total_real_pipeline_s:.2f}",
                "Real Pipeline FPS (Target frames)": f"{real_est_fps:.2f}",
                # MAIN CONSUMER LOOP TIMELINE (SEQUENTIAL OVERHEAD)
                "Avg Frame Reading (ms)": f"{avg_reader_ms:.2f}",
                # "Avg Resize (ms)": f"{avg_resize_ms:.2f}",
                # "Avg Writer Async Handoff (ms)": f"{avg_write_ms:.2f}",
                # BACKGROUND INGESTION HEALTH (ASYNC METRICS)
                "Avg Host Copy (ms)": f"{avg_copy_ms:.2f}",
                "Avg DMA Upload (ms)": f"{avg_h2d_ms:.2f}",
                "Avg Data Prep Overhead (ms)": f"{avg_prep_ms:.2f}",
                "Avg Queue Blocked (ms)": f"{avg_blocked_ms:.2f}",
                "Avg Queue Saturation (%)": f"{avg_saturation:.2f}",
                # Framework Overhead Leak Tracking (Measures loop time spent outside raw AI inference model blocks)
                "Loop Overhead %": f"{loop_overhead:.2f}%",
                # Inbound Stream Frame Drop Rate (Identifies if background threads are dropping packets due to I/O)
                "Inbound Frame Drop %": f"{frame_drop:.2f}%",
                # Serialization Pressure Index (Tracks what percentage of your frame time is lost to thread stalls)
                "Serialization Pressure %": f"{serial_pressure:.2f}%",
                # Queue Evacuation Velocity Coefficient (Values > 1.0 mean your consumer loop drains frames faster than ingestion)
                "Evacuation Velocity": f"{evac_vel:.2f}x",
                "Compute Backpressure Index": f"{backpressure_index:.2f}%",
                "Model Cost Density Ratio": f"{model_cost_density:.2f}x",
                "PCIe Bus Saturation %": f"{pcie_bus_saturation:.2f}%",
                # PIPELINE HEALTH & BEHAVIOR INSIGHTS
                "Pipeline Status": pipeline_status,
                "Peak Queue Fullness (%)": f"{peak_saturation:.2f}",
                "Targeted Pacing Delivery (%)": f"{target_match_ratio:.2f}",
                # DETECTION PIPELINE (Should B NULL)
                "SF/Detection Latency (s)": f"{det_latency_s:.2f}",
                "SF/Detection FPS": f"{det_est_fps:.2f}",
                "Avg SF (ms)": f"{avg_sf:.2f}",
                "Avg ROI (ms)": f"{avg_roi:.2f}",
                "Avg Obj. Detection (ms)": f"{avg_det:.2f}",
                "Total Breakdown Sum (ms)": f"{det_latency_ms:.2f}",
                "Avg Area Coverage %": f"{avg_cov:.2f}%",
                "Avg Crops/Frame": f"{avg_crops:.1f}",
                # "Frames w/o ROIs": self.no_roi_frame_cnt,
                "Crop Cap Rate (>20)": f"{cap_rate:.1f}%",
                "Objects Detected": num_objs,
                # TIming to display frame (read to after send to render queue)
                "Display Latency (s)": f"{self.elapsed_display_time:.2f}",
                "Display Frames": stat_frame_count,
                "Display FPS": f"{stat_fps:.2f}",
                "PCIe Bandwidth Throughput (GB/s)": f"{self.pcie_throughput_gbps:.2f}",
                "Avg Async Thread Pool Backlog": f"{sum(self.component_stats.get('thread_backlog', [0])) / len(self.component_stats.get('thread_backlog', [1])):.1f}",
                "VRAM Allocation Efficiency (%)": f"{self.vram_efficiency:.1f}%",
            }
        )

        print(
            f"\n[{self._testMethodName}] Pipeline Latency (s): {total_pipeline_s:.2f} sec"
        )
        print(
            f"\n[{self._testMethodName}] Pipeline FPS (Target frames): {target_frame_fps:.2f} ({total_written_frames} frames)"
        )
        print(
            f"\n[{self._testMethodName}] Pipeline FPS (Video frames): {all_frame_fps:.2f} ({self.reader.total_input_frames} frames)"
        )
        print(
            f"\n[{self._testMethodName}] SF/Detection FPS (Target frames): {det_est_fps:.2f} ({total_written_frames} frames)"
        )
        print(
            f"\n[{self._testMethodName}] Display FPS: {stat_fps:.2f} ({stat_frame_count} frames)"
        )

    def frame2video(
        self,
        device_frame,
        frameNum,
        metadata_or_bbs,
        class_list,
    ):
        scale_display_x = self.disp_w / 640
        scale_display_y = self.disp_h / 640

        if self.device_input == "cuda" and torch.is_tensor(device_frame):
            # 2. Fast Inline VRAM Downscaling into our static memory slot
            # Reshape to (Batch, Channel, Height, Width) seamlessly without duplicating data
            gpu_tensor = device_frame[None, :].permute(0, 3, 1, 2).float()

            resized_tensor = torch.nn.functional.interpolate(
                gpu_tensor,
                size=(self.disp_h, self.disp_w),
                mode="nearest",
            )

            # CONVERTS TO CPU EARLIER (WORKS but lowers fps)
            # hwc_byte_tensor = resized_tensor.squeeze(0).permute(1, 2, 0).byte()
            # cpu_tensor = hwc_byte_tensor.cpu()
            # bgr_contiguous = cpu_tensor.contiguous()

            # Using copy_() avoids creating any runtime allocations or memory-stride drift.
            self.static_gpu_byte_bchw.copy_(resized_tensor.byte())

            # Safely reshape the pre-allocated, locked GPU memory layout back to standard HWC array topology.
            # Since the underlying array layout is strictly contiguous, this permute is guaranteed
            # to be zero-copy on the GPU and free from memory race stalls.
            bgr_contiguous = self.static_gpu_byte_bchw.squeeze(0).permute(1, 2, 0)

            # 1. Grab the current free pinned slot tracking variables from your reader
            d2h_idx = self.reader.d2h_selector
            pinned_tensor_buf = self.reader.d2h_buffers[d2h_idx]

            # c_idx = self.canvas_selector
            # current_canvas = self.static_host_canvases[c_idx]

            # host_tensor_view = torch.as_tensor(current_canvas, device="cpu")

            # 3. Non-blocking asynchronous PCIe DMA Download directly into our page-locked canvas
            with torch.cuda.stream(self.reader.download_stream):
                # pinned_tensor = torch.from_numpy(current_canvas).cuda()
                pinned_tensor_buf.copy_(bgr_contiguous, non_blocking=True)

            # Synchronize only the side download stream layout channel
            self.reader.download_stream.synchronize()

            cpu_360p_frame = self.reader.d2h_numpys[d2h_idx]

            # 5. Non-Blocking Push straight to your background AsyncVideoWriter thread pool
            # We pass a direct .copy() slice so the hot loop can instantly reuse the pinned ring buffer
            display_frame = np.array(cpu_360p_frame, copy=True, order="C")

            if display_frame is not None and display_frame.shape[-1] == 3:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

            # --- Draw Detection Overlays ---
            if isinstance(metadata_or_bbs, dict):
                # Object Mode (YOLO Structs)
                display_frame = get_metadata_overlay(
                    display_frame,
                    metadata_or_bbs,
                    class_list,
                    (scale_display_x, scale_display_y),
                    (self.disp_w, self.disp_h),
                    is_bgr=True,
                )

            elif metadata_or_bbs is not None:
                # # Motion / Smart Filtering Overlay Path
                display_frame = get_bb_overlay(
                    display_frame,
                    metadata_or_bbs,
                    (scale_display_x, scale_display_y),
                    (self.disp_w, self.disp_h),
                )

            self.async_writer.write_frame(display_frame)
            # self.canvas_selector = 1 - self.canvas_selector
            self.reader.d2h_selector = 1 - self.reader.d2h_selector

        else:
            # Reusable baseline track for CPU execution mappings
            display_frame = cv2.resize(
                device_frame,
                (self.disp_w, self.disp_h),
                interpolation=cv2.INTER_NEAREST,
            )
            # if metrics["bbs"] is not None:
            #     for box in metrics["bbs"]:
            #         cv2.rectangle(cpu_resized, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # --- Draw Detection Overlays ---
            if isinstance(metadata_or_bbs, dict):
                # Object Mode (YOLO Structs)
                display_frame = get_metadata_overlay(
                    display_frame,
                    metadata_or_bbs,
                    class_list,
                    (scale_display_x, scale_display_y),
                    (self.disp_w, self.disp_h),
                    is_bgr=True,
                )

            elif metadata_or_bbs is not None:
                # # Motion / Smart Filtering Overlay Path
                display_frame = get_bb_overlay(
                    display_frame,
                    metadata_or_bbs,
                    (scale_display_x, scale_display_y),
                    (self.disp_w, self.disp_h),
                )
            self.async_writer.write_frame(display_frame)

    # PIPELINE FUNCTIONS --------------------------------------------

    def pipeline_fn(self, device_frame, overall_frame_num, stat_start_time):
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

        if self.config.sf_enabled:
            if self.device_input == "cuda":
                sf_start.record(self.inference_stream)
                bgs_input_frame = (
                    device_frame.byte()
                    if torch.is_tensor(device_frame)
                    else device_frame
                )
                inf_data = self.rbtd_full_gpu(bgs_input_frame)
                sf_end.record(self.inference_stream)
            else:
                t_start = time.perf_counter()
                inf_data = self.rbtd_full_cpu(device_frame)
                metrics["sf_time"] = (time.perf_counter() - t_start) * 1000.0
        else:
            inf_data = {}
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

        # If inline optimizations left the tensor in channels-first format (B, C, H, W),
        # restore it to standard standard layout standard (H, W, C) so get_detections
        # can decode bounding boxes accurately without corrupting weights.
        det_frame = device_frame
        # Cast float canvases back to standard unsigned byte tracking arrays
        # so get_detections can safely run its internal float normalizations.
        if torch.is_tensor(det_frame):
            det_frame = det_frame.byte()

            if det_frame.ndim == 4:
                det_frame = det_frame.squeeze(0).permute(1, 2, 0)
            elif det_frame.shape[0] == 3:
                det_frame = det_frame.permute(1, 2, 0)

        # --- 3. MODEL INFERENCE TIMING BLOCK ---
        if self.device_input == "cuda":
            det_start.record(self.inference_stream)
        else:
            t_start = time.perf_counter()

        metadata = {}
        if self.config.DETECTION_TYPE != "motion":
            # Pass directly or use a light NumPy view slice to protect the raw pointer
            # det_frame = (
            #     device_frame
            #     if not isinstance(device_frame, np.ndarray)
            #     else device_frame.view()
            # )

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
            # torch.cuda.synchronize()
            self.inference_stream.synchronize()

            # Extract hardware timings smoothly without stalling mid-run
            if self.config.sf_enabled:
                metrics["sf_time"] = sf_start.elapsed_time(sf_end)
                metrics["roi_time"] = roi_start.elapsed_time(roi_end)
                metrics["det_time"] = det_start.elapsed_time(det_end)
                metrics["bbs"] = bbs_full_res
                metrics["batch_density"] = (
                    len(clean_bbs) if clean_bbs is not None else 0
                )
            else:
                # Full-frame YOLO baseline tracks the elapsed time from t_start on page 20, line 1273
                metrics["sf_time"] = 0.0
                metrics["roi_time"] = 0.0
                metrics["det_time"] = det_start.elapsed_time(det_end)
                metrics["bbs"] = None
                metrics["batch_density"] = 1  # 1 Full frame block

        else:
            # CPU Path Execution: Must use standard wall-clock timing loops to avoid CUDA Event errors
            if self.config.sf_enabled:
                # Timings for sf_time and roi_time are already gathered on CPU on page 19 line 1221 and page 20 line 1246
                metrics["det_time"] = (time.perf_counter() - t_start) * 1000.0
                metrics["bbs"] = bbs_full_res
                metrics["batch_density"] = (
                    len(clean_bbs) if clean_bbs is not None else 0
                )
            else:
                metrics["sf_time"] = 0.0
                metrics["roi_time"] = 0.0
                metrics["det_time"] = (time.perf_counter() - t_start) * 1000.0
                metrics["bbs"] = None
                metrics["batch_density"] = 1

        data_to_draw = clean_bbs if self.config.DETECTION_TYPE == "motion" else metadata

        self.frame2video(
            det_frame,
            inf_data["frameNum"],
            data_to_draw,
            self.label_source,
        )

        self.update_frame(stat_start_time)
        return num_objs, metrics  # Skip full detection pass

    def run_realtime_inference(self, sf_enabled=True):
        # self.duration_target = 30
        self.status = "RUNNING"
        metrics = {}
        num_objs = 0
        self.is_cuda = self.device.lower() == "gpu" and torch.cuda.is_available()
        self.frame_count_target = 0
        self.next_process_idx = 0.0
        self.frame_in_clip_count = 0
        total_pipeline_time_ms = 0.0
        total_read_time = 0.0
        total_queue_saturation = 0.0
        # queue_saturation_history = [0.0]
        total_written_frames = 0
        total_sf_pipeline_time_ms = 0.0
        real_world_latency_ms = 0.0

        # queue_size = 4
        total_run_pipelinefn = 0
        # total_read_time = 0
        # total_resize_time = 0
        # total_written_frames = 0
        # total_disk_write_overhead = 0.0
        total_frame_loop_latency = 0.0
        total_active_processing_overhead = 0.0
        # total_queue_saturation = 0.0
        # total_queue_saturation = 0.0
        queue_saturation_history = []
        consecutive_slow_frames = 0
        # total_dropped_frames = 0

        coverage_percentages = []
        self.component_stats = {
            "sf": [],
            "roi": [],
            "det": [],
            "dma_upload": [],  # Track PCIe Transfer Latencies
            "queue_blocked": [],  # Track GIL Serialization stalls
            "batch_sizes": [],  # Track Smart Filtering density
            "thread_backlog": [],  # Track thread work pool backlog
        }
        self.crops_per_frame_list = []

        self.step_size = (
            float(self.input_fps) / float(self.target_fps)
            if hasattr(self, "target_fps")
            else 1.0
        )
        self.max_target_frames = (
            int(self.duration_target * float(self.target_fps))
            if hasattr(self, "target_fps")
            else float("inf")
        )

        # self.start()
        # time.sleep(0.1)

        # if self.source.startswith("rtsp"):
        #     short_name = "rtsp"
        # else:
        #     short_name = Path(self.source).stem
        # output_path= f"{self.config.SHARED_OUTPUT}/{short_name}_{self.device}.mp4"
        # disp_wh = (self.disp_w, self.disp_h)
        # async_writer = AsyncVideoWriter(
        #     self.output_path,
        #     cv2.VideoWriter_fourcc(*'avc1'),
        #     float(self.target_fps),
        #     (self.disp_w, self.disp_h)
        # )

        pipeline_start_time = time.perf_counter()
        last_loop_cycle_timestamp = time.perf_counter()

        # try:
        self.dynamic_limit = max(2, int(0.5 * self.target_fps))
        # last_frame_time = time.perf_counter()
        while self.active and self.frame_count_target < self.max_target_frames:
            try:
                # FRAME RETRIEVAL ---------------------------------------------
                try:
                    frame_start_time = time.perf_counter()
                    t_read = time.perf_counter()
                    # if self.active:
                    ret, frame_8k, frame_num = (
                        self.reader.read()
                    )  # Successfully reads the 8K frame

                    if not ret or frame_8k is None:
                        # Check if there are any remaining frames still buffered in flight inside the reader queue
                        # if hasattr(self.reader, 'frame_queue') and not self.reader.frame_queue.empty():
                        #     ret, frame_8k, frame_num = self.reader.read()
                        #     if not ret or frame_8k is None:
                        #         break
                        # else:
                        #     break
                        if self.reader is None or (
                            hasattr(self.reader, "stopped") and self.reader.stopped
                        ):
                            if self.device_input == "cuda":
                                torch.cuda.synchronize()
                            self.active = False
                            break
                        continue

                    reader_time = time.perf_counter() - t_read
                    total_read_time += reader_time

                    # Calculate the exact time gap between the last frame completion
                    # and the start of the next read. This accurately captures downstream GIL blocks.
                    cycle_gap = time.perf_counter() - last_loop_cycle_timestamp
                    # Subtract the actual read duration to isolate the thread stall time
                    true_serialization_stall = max(
                        0.0, (cycle_gap - (time.perf_counter() - t_read)) * 1000.0
                    )
                    self.component_stats["queue_blocked"].append(
                        true_serialization_stall
                    )

                except queue.Empty:
                    if getattr(self.reader, "reconnect_failed", False):
                        self.active = False
                        break
                    time.sleep(0.002)
                    continue

                # print(f"frame_num: {frame_num}")

                self.stat_start_time = time.perf_counter()
                self.frame_count += 1
                # is_target_frame = True

                # Real-Time Metric A: Track Queue Saturation Density Ratio
                # current_q_size = self.reader.frame_queue.qsize()
                saturation_ratio = (
                    self.reader.frame_queue.qsize() / self.reader.frame_queue.maxsize
                )
                total_queue_saturation += saturation_ratio
                queue_saturation_history.append(saturation_ratio)

                # RUN PIPELINE_FN ---------------------------------------------
                run_pipelinefn_start = time.perf_counter()
                self.frame_count_target += 1
                self.frame_in_clip_count += 1

                if self.device_input == "cuda":
                    # Instantiate a lightweight hardware fence event object
                    curr_event = torch.cuda.Event(enable_timing=False)
                    # curr_event.record()
                    # Record the exact milestone on the default stream right after fetching the frame data
                    curr_event.record(torch.cuda.default_stream())
                    self.inference_stream.wait_event(curr_event)

                    with torch.cuda.stream(self.inference_stream):
                        nob, metrics = self.pipeline_fn(
                            frame_8k,
                            frame_num,
                            # is_target_frame,
                            self.stat_start_time,
                        )
                    # torch.cuda.default_stream().synchronize()
                    # self.inference_stream.synchronize()

                    # Extract PCIe DMA upload latency from your hardware background reader
                    if hasattr(self.reader, "total_h2d_time"):
                        avg_h2d_frame_ms = (
                            self.reader.total_h2d_time
                            / max(1, total_written_frames + 1)
                        ) * 1000.0
                        self.component_stats["dma_upload"].append(avg_h2d_frame_ms)
                else:
                    nob, metrics = self.pipeline_fn(
                        frame_8k,
                        frame_num,
                        # is_target_frame,
                        self.stat_start_time,
                    )
                num_objs += nob

                total_written_frames += 1

                # Real-Time Metric B: Frame Processing Latency Check
                frame_loop_latency = (
                    time.perf_counter() - frame_start_time
                ) * 1000  # ms
                total_frame_loop_latency += frame_loop_latency

                # Isolate active processing time (excluding the queue blocking read)
                active_processing_overhead = frame_loop_latency - (reader_time * 1000)
                total_active_processing_overhead += active_processing_overhead

                # RUNTIME TERMINAL WARNING ALERTS
                if saturation_ratio >= 1.0 or active_processing_overhead > (
                    1000 / self.target_fps
                ):
                    consecutive_slow_frames += 1
                    if (
                        consecutive_slow_frames % self.target_fps == 0
                    ):  # Throttle terminal spam
                        print(
                            f"\033[93m⚠️ [PERF WARNING] Main loop starving! Waiting on stream ingestion... "
                            f"Read Wait: {reader_time:.1f}ms | Other Wait: {active_processing_overhead:.1f}ms | Queue Fullness: {saturation_ratio * 100:.0f}%\033[0m"
                        )
                else:
                    consecutive_slow_frames = max(0, consecutive_slow_frames - 1)

                if total_written_frames % (5 * self.target_fps) == 0:
                    print(
                        f"Captured {total_written_frames}/{self.max_target_frames} frames..."
                    )

                # WRAP-UP FRAME INSTANCE ---------------------------------------------
                # if self.device_input == "cuda" and torch.cuda.is_available():
                #     torch.cuda.default_stream().synchronize()

                total_run_pipelinefn += time.perf_counter() - run_pipelinefn_start

                # Explicitly delete frame variables to free their references
                del frame_8k

                last_loop_cycle_timestamp = time.perf_counter()

                # Force PyTorch's internal allocator to release cached segments back to the OS
                # if self.frame_count_target % (5 * self.target_fps) == 0:
                #     print(
                #         f"Captured {self.frame_count_target}/{self.max_target_frames} frames..."
                #     )
                #     torch.cuda.empty_cache()
                #     torch.cuda.ipc_collect()

            except torch.cuda.OutOfMemoryError:
                print("\n" + "!" * 70)
                print(
                    "[CRITICAL TEST CRASH] GPU MEMORY CEILING HIT INSIDE RUNNER LOOP!"
                )
                print(
                    "Freezing allocation history registers and writing diagnostic log..."
                )
                print("!" * 70)

                try:
                    import os

                    snapshot_filename = (
                        f"/tmp/test_vram_leak_profile_pid{os.getpid()}.pickle"
                    )
                    torch.cuda.memory._dump_snapshot(snapshot_filename)
                    print(
                        f"[PROFILER SUCCESSFUL] Snapshot profile written to: {snapshot_filename}"
                    )
                    print(
                        "--> Drag and drop this file directly into: https://pytorch.org"
                    )
                except Exception as dump_err:
                    print(f"Failed to record profile data snapshot: {dump_err}")

                # Force safe system unlinking of background workers to clean up OS handles
                self.active = False
                if hasattr(self, "reader") and self.reader is not None:
                    self.reader.stop()
                raise

            except Exception:
                traceback.print_exc()

        # POST FRAME PROCESSING ---------------------------------------------
        pipeline_end_time = time.perf_counter()

        # Resolve the final execution window. If the stream ran to completion,
        # use the operational end marker to avoid measuring downstream disk write bottlenecks.
        final_end_marker = getattr(self, "pipeline_completion_timestamp", None)
        if final_end_marker is None:
            final_end_marker = time.perf_counter()
            # self.pipeline_completion_timestamp = final_end_marker

        real_world_latency_ms = float((final_end_marker - pipeline_start_time) * 1000.0)

        total_sf_pipeline_time_ms = float(
            sum(self.component_stats.get("sf", [0.0]))
            + sum(self.component_stats.get("roi", [0.0]))
            + sum(self.component_stats.get("det", [0.0]))
        )
        total_pipeline_time_ms = float(
            (pipeline_end_time - pipeline_start_time) * 1000.0
        )

        # total_queue_saturation = sum(
        #     self.component_stats.get("batch_sizes", [0.0])
        # )  # Sample array profile proxy

        if metrics != {}:
            num_crops = len(metrics["bbs"]) if metrics["bbs"] is not None else 0
            self.crops_per_frame_list.append(num_crops)

            # Context-aware extraction of the newly proposed metrics
            density = metrics.get("batch_density", 0 if self.config.sf_enabled else 1)
            self.component_stats["batch_sizes"].append(density)

            self.component_stats["sf"].append(metrics["sf_time"])
            self.component_stats["roi"].append(metrics["roi_time"])
            self.component_stats["det"].append(metrics["det_time"])

            # Calculate coverage OUTSIDE the timed block to prevent interference
            if self.config.sf_enabled and metrics.get("bbs") is not None:
                cov = self.calculate_unique_coverage(metrics["bbs"])
                coverage_percentages.append(cov)
        self.async_writer.release()
        # Continuously sample the thread work pool backlog state mechanics
        if hasattr(self, "executor") and self.executor is not None:
            # Safely probe internal concurrent.futures work queue bounds
            self.component_stats["thread_backlog"].append(
                self.executor._work_queue.qsize()
            )
        else:
            self.component_stats["thread_backlog"].append(0)

        # Real-Time Metric B: Frame Processing Latency Check
        # total_frame_loop_latency = (pipeline_end_time - frame_start_time) * 1000  # ms

        # Isolate active processing time (excluding the queue blocking read)
        # total_active_processing_overhead = total_frame_loop_latency - total_read_time  #(reader_time * 1000)

        # RUNTIME TERMINAL WARNING ALERTS
        # if saturation_ratio >= 1.0 or active_processing_overhead > (1000 / self.target_fps):
        #     consecutive_slow_frames += 1
        #     if consecutive_slow_frames % self.target_fps == 0:  # Throttle terminal spam
        #         print(
        #             f"\033[93m⚠️ [PERF WARNING] Main loop starving! Waiting on stream ingestion... "
        #             f"Read Wait: {frame_loop_latency:.1f}ms | Queue Fullness: {saturation_ratio * 100:.0f}%\033[0m"
        #         )
        # else:
        #     consecutive_slow_frames = max(0, consecutive_slow_frames - 1)

        # CALCULATE PERFORMANCE METRICS ---------------------------------------------
        print(
            f"\nExecution Finished. Total Output Frames Written: {self.frame_count_target}"
        )

        # Render your metrics dashboard table cleanly
        if self.frame_count_target > 0:
            avg_frame_loop_latency = total_frame_loop_latency / self.frame_count_target
            self._finalize_benchmarks(
                num_objs,
                self.frame_count_target,  # Reads the updated integer directly!
                total_pipeline_time_ms,
                real_world_latency_ms,
                total_sf_pipeline_time_ms,
                avg_frame_loop_latency,
                coverage_percentages,
                total_read_time,
                total_queue_saturation,
                queue_saturation_history,
            )
        else:
            print(
                f"[SKIPPED SUMMARY] Only {self.frame_count_target} frames processed. ",
                flush=True,
            )
        self.stop()


# MAIN ----------------------------------------


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
    # Force all PyTorch extension handles to compile and load BEFORE threads spawn
    if torch.cuda.is_available():
        _ = torch.zeros(1).cuda()
        import torch._ops
        import torch.utils

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
