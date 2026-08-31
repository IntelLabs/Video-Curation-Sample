# ==============================================================================
# IMPORTS

import gc
import inspect
import io
import os
import queue
import sys
import threading
import time
import traceback
import types
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Retrieve repo packages
REPO_DIR = str(Path(__file__).parent.parent)
sys.path.insert(1, REPO_DIR)
from include.default_configs import TARGET_FPS
from include.handlers import (
    get_bb_overlay,
)
from include.utils import (
    AsyncVideoWriter,
    default_attr_keys,
    global_frame_prefetch_worker_v1,
    install_and_load_pip_package,
    release_native_linux_heap,
    scale_bbox,
    scale_bbox_xywh,
)

gdown = install_and_load_pip_package("gdown", attribute_name=None)
objgraph = install_and_load_pip_package("objgraph", attribute_name=None)
target_width, target_height = 7680, 4320  # 8K
# torch.set_grad_enabled(False)

# ==============================================================================
# LOGGING
import logging

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("libav").setLevel(logging.CRITICAL)
logging.getLogger("libav.hevc").setLevel(logging.CRITICAL)
main_app_logger = logging.getLogger(__name__)

# ==============================================================================
# FUNCTIONS


# Download tracking (video) data and associated ground truth
def download_eval_data(video_name_list, target_fps=TARGET_FPS):
    DATA_DIR = Path(__file__).parent / "eval_data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # DOWNLOAD GT ZIP
    VIDEO_GROUND_TRUTH_URL = (
        "https://drive.google.com/open?id=16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq"
    )
    VIDEO_GROUND_TRUTH_ZIP_PATH = str(DATA_DIR / "Anti-UAV-Tracking-V0GT.zip")
    if not Path(VIDEO_GROUND_TRUTH_ZIP_PATH).exists():
        main_app_logger.info("Downloading ground truth zip file...")
        gdown.download(
            url=VIDEO_GROUND_TRUTH_URL, output=VIDEO_GROUND_TRUTH_ZIP_PATH, quiet=False
        )

    # GET GT BBOXES
    # GT bbox: [x_min, y_min, w, h]
    gt_boxes_dict = {}
    with zipfile.ZipFile(VIDEO_GROUND_TRUTH_ZIP_PATH, "r") as gt_ref:
        for gt_info in gt_ref.infolist():
            matched_vid = next(
                (
                    v
                    for v in video_name_list
                    if gt_info.filename.endswith(f"{v}_gt.txt")
                ),
                None,
            )
            if matched_vid:
                gt_bytes = gt_ref.read(gt_info.filename)
                gt_stream = io.StringIO(gt_bytes.decode("utf-8"))
                delimiter = "," if b"," in gt_bytes else None
                gt_boxes_dict[matched_vid] = np.loadtxt(
                    gt_stream, delimiter=delimiter, dtype=np.int32
                ).tolist()

    # DOWNLOAD FRAME ZIP
    VIDEO_SEQUENCE_URL = (
        "https://drive.google.com/open?id=1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9"
    )
    VIDEO_SEQUENCE_ZIP_PATH = str(DATA_DIR / "Anti-UAV-Tracking-V0.zip")
    if not Path(VIDEO_SEQUENCE_ZIP_PATH).exists():
        main_app_logger.info("Downloading video sequence zip file...")
        gdown.download(
            url=VIDEO_SEQUENCE_URL, output=VIDEO_SEQUENCE_ZIP_PATH, quiet=False
        )

    # WRITE VIDEO AND GET VIDEO DETAILS
    ALL_VIDEO_DETAILS = {}
    # ALL_VIDEO_WRITERS = {}
    # frame_counters = {v: 0 for v in video_name_list}
    # Open Zip once, but process each video COMPLETELY one by one
    with zipfile.ZipFile(VIDEO_SEQUENCE_ZIP_PATH, "r") as zip_ref:
        # Pre-index all archive paths for ultra-fast matching
        all_files = zip_ref.infolist()

        for video_name in video_name_list:
            main_app_logger.info(f"Processing sequence: {video_name}...")

            # Filter and sort frames belonging strictly to the current video
            video_img_infos = [
                info
                for info in all_files
                if info.filename.startswith(f"Anti-UAV-Tracking-V0/{video_name}/")
                and info.filename.lower().endswith(".jpg")
            ]
            video_img_infos.sort(key=lambda x: x.filename)

            if not video_img_infos:
                continue

            video_writer = None
            frame_idx = 0

            # Setup dictionary tracking structure
            sample_outpath = DATA_DIR / video_img_infos[0].filename.replace(
                "Anti-UAV-Tracking-V0", "Anti-UAV-Tracking-V0-8K"
            )
            sample_outpath.parent.mkdir(parents=True, exist_ok=True)

            # BYPASS COMPUTE IF VIDEO EXISTS
            target_mp4_path = sample_outpath.parent / f"{video_name}.mp4"
            if target_mp4_path.exists():
                main_app_logger.info(
                    f"\t[SKIP] 8K target video {target_mp4_path.name} already compiled."
                )

                first_frame_info = video_img_infos[0]
                img_bytes = zip_ref.read(first_frame_info.filename)
                np_array = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                orig_height, orig_width, _ = img.shape if img is not None else (0, 0, 0)

                current_gt_boxes = gt_boxes_dict.get(video_name, [])
                num_invalid = sum(
                    1 for bbox in current_gt_boxes if bbox == [0, 0, 0, 0]
                )
                num_valid = len(current_gt_boxes) - num_invalid

                scaled_gt_boxes = []
                for bbox in current_gt_boxes:
                    if bbox == [0, 0, 0, 0]:
                        scaled_gt_boxes.append([0, 0, 0, 0])
                    else:
                        scaled_gt_boxes.append(
                            scale_bbox_xywh(
                                bbox,
                                orig_width,
                                orig_height,
                                targetW=target_width,
                                targetH=target_height,
                            )
                        )

                # Reconstruct the metadata payload so downstream tracking logic doesn't break
                ALL_VIDEO_DETAILS[video_name] = {
                    "frame_dir": str(sample_outpath.parent),
                    "orig_height": orig_height,  # Safe placeholder or pull from cache if variable
                    "orig_width": orig_width,
                    "target_height": target_height,
                    "target_width": target_width,
                    "sequence": [Path(info.filename).stem for info in video_img_infos],
                    "gt_bbox": scaled_gt_boxes,
                    "num_invalid_gt_bboxes": num_invalid,
                    "num_valid_gt_bboxes": num_valid,
                }
                num_frames = len(ALL_VIDEO_DETAILS[video_name]["sequence"])
                num_boxes = len(ALL_VIDEO_DETAILS[video_name]["gt_bbox"])
                num_invalid_boxes = ALL_VIDEO_DETAILS[video_name][
                    "num_invalid_gt_bboxes"
                ]
                main_app_logger.info(
                    f"\t{video_name}: {num_frames} frames, {num_boxes} gt boxes ({num_invalid_boxes} out-of-view)"
                )
                continue  # Instantly advance to the next video name in the zip reference

            # Initialize detail log
            ALL_VIDEO_DETAILS[video_name] = {
                "frame_dir": str(sample_outpath.parent),
                "orig_height": 0,
                "orig_width": 0,
                "target_height": target_height,
                "target_width": target_width,
                "sequence": [],
                "gt_bbox": [],
                "num_invalid_gt_bboxes": 0,
                "num_valid_gt_bboxes": 0,
            }

            for file_info in video_img_infos:
                frame_id = Path(file_info.filename).stem

                # In-memory decompression
                img_bytes = zip_ref.read(file_info.filename)
                np_array = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                if img is not None:
                    orig_height, orig_width, _ = img.shape

                    # Update dimensions once per video segment
                    if frame_idx == 0:
                        ALL_VIDEO_DETAILS[video_name]["orig_height"] = orig_height
                        ALL_VIDEO_DETAILS[video_name]["orig_width"] = orig_width

                    # Performance upscale to 8K
                    img_8K = cv2.resize(
                        img,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LINEAR,  # INTER_CUBIC,
                    )

                    # Initialize Video Writer sequentially if missing
                    if (
                        video_writer is None
                        and not (sample_outpath.parent / f"{video_name}.mp4").exists()
                    ):
                        video_writer = AsyncVideoWriter(
                            str(sample_outpath.parent / f"{video_name}.mp4"),
                            cv2.VideoWriter_fourcc(*"avc1"),
                            target_fps,
                            (target_width, target_height),
                        )

                    # Synchronous frame pipeline updates
                    if video_writer is not None:
                        video_writer.write_frame(img_8K)
                    ALL_VIDEO_DETAILS[video_name]["sequence"].append(frame_id)

                    # Bounding Box Sync
                    if video_name in gt_boxes_dict and frame_idx < len(
                        gt_boxes_dict[video_name]
                    ):
                        gt_bbox = gt_boxes_dict[video_name][frame_idx]

                        if gt_bbox == [0, 0, 0, 0]:
                            x, y, w, h = [0, 0, 0, 0]
                            ALL_VIDEO_DETAILS[video_name]["num_invalid_gt_bboxes"] += 1
                        else:
                            x, y, w, h = scale_bbox_xywh(
                                gt_bbox,
                                orig_width,
                                orig_height,
                                targetW=target_width,
                                targetH=target_height,
                            )
                            ALL_VIDEO_DETAILS[video_name]["num_valid_gt_bboxes"] += 1

                        ALL_VIDEO_DETAILS[video_name]["gt_bbox"].append([x, y, w, h])

                    frame_idx += 1

            # Close the writer IMMEDIATELY
            if video_writer is not None:
                video_writer.release()

            num_frames = len(ALL_VIDEO_DETAILS[video_name]["sequence"])
            num_boxes = len(ALL_VIDEO_DETAILS[video_name]["gt_bbox"])
            num_invalid_boxes = ALL_VIDEO_DETAILS[video_name]["num_invalid_gt_bboxes"]
            main_app_logger.info(
                f"\t{video_name}: {num_frames} frames, {num_boxes} gt boxes ({num_invalid_boxes} out-of-view)"
            )

    return ALL_VIDEO_DETAILS


# Print active CUDA tensors
def track_gpu_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensor_candidate = obj
            elif (
                hasattr(obj, "data") and torch.is_tensor(obj.data) and obj.data.is_cuda
            ):
                tensor_candidate = obj.data
            else:
                tensor_candidate = None

            if tensor_candidate is not None:  # and tensor_candidate.is_cuda:
                nelement = tensor_candidate.nelement()
                cand_shape = list(tensor_candidate.shape)
                cand_dtype = tensor_candidate.dtype
                tensor_bytes = tensor_candidate.element_size() * nelement
                # main_app_logger.info(f"Leaked Tensor -> Type: {type(obj)} | Size: {obj.size()} | Device: {obj.device}")
                main_app_logger.info(
                    f"  > Tensor | Device: {tensor_candidate.device} | Shape: {str(cand_shape):<18} | Dtype: {str(cand_dtype):<12} | Size: {tensor_bytes / 1024**2:6.2f} MB"
                )
        except Exception:
            pass
    del obj


# Save FPS comparison chart
def fps_comparison_chart(chart_path, results, fps_key="Pipeline FPS (Video frames)"):
    try:
        # names = [r["Test Name"] for r in results]
        # fps_values = [float(r[fps_key]) for r in results]
        names = []
        fps_values = []

        for r in results:
            if isinstance(r, dict) and "Test Name" in r:
                # Force casting to clean, unpinned Python native types
                names.append(str(r["Test Name"]))
                fps_values.append(float(r.get(fps_key, 0.0)))

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
        main_app_logger.info(f" Comparison chart saved to: {chart_path}")
    except Exception:
        main_app_logger.info("Skipping chart generation: error occurred.")


# ==============================================================================
#  CLASSES
class BaseTest:
    # PIPELINE FUNCTIONS --------------------------------------------
    @torch.inference_mode()
    def pipeline_fn(
        self,
        device_frame,
        overall_frame_num,
        stat_start_time,
        current_clip_id,
        gt_boxes=None,
        read_frame_only=False,
    ):
        current_clip_key = f"{self.name}_{current_clip_id:03d}.mp4"
        current_clip_path = f"{self.config.SHARED_OUTPUT}/{current_clip_key}"

        metadata = {}
        motion_detected = False
        metrics = {"sf_time": 0, "roi_time": 0, "det_time": 0, "bbs": None}

        # Given input frame, get metadata and metrics
        try:
            if read_frame_only:
                # Get frame, metadata and metrics
                _, det_frame = self.processor.format_bbs_and_frame_4_detection(
                    [], device_frame
                )

            else:
                # Run model on frame, get metadata and metrics
                # with torch.inference_mode():
                metrics, metadata, det_frame, motion_detected = self.processor.run(
                    device_frame,
                    overall_frame_num,
                    frame_in_clip_count=self.frame_in_clip_count,
                    gt_boxes=None,
                )

            self.total_objects_detected += len(metadata.keys())

            # Save results in video
            self.frame2video(
                det_frame,
                overall_frame_num,
                metadata,
                getattr(self, "label_source", None),
                stat_start_time,
                gt_boxes=gt_boxes,
            )

        except Exception as e_detection:
            traceback.print_exc()
            main_app_logger.info(f"[DETECTION ERROR] Exception: {e_detection}")
            # traceback.print_exc()
            # if self.active:
            #     traceback.print_exc()
            #     main_app_logger.info(f"[DETECTION ERROR] Exception: {e_detection}")
            #     self.active = False
        finally:
            # del inf_data, bbs_full_res, device_frame, det_frame
            if "det_frame" in locals():
                del det_frame
            if "device_frame" in locals():
                del device_frame
            # if "bbs_full_res" in locals():
            #     del bbs_full_res
            # if "inf_data" in locals():
            #     del inf_data

        # del inf_data, bbs_full_res, device_frame, det_frame

        return metadata, metrics, motion_detected  # Skip full detection pass

    def initialize_run_realtime_inference_v1(
        self, read_frame_only, num_prefetch_workers
    ):
        # self.duration_target = 30
        # self.status = "RUNNING"
        self.processor = self._setup_processor(read_frame_only)

        if hasattr(self, "processor") and hasattr(self.processor, "label_source"):
            self.label_source = self.processor.label_source
        # Setup empty list to track background futures natively
        # self._active_inference_futures = []

        if self.is_cuda and not hasattr(self, "sf_start") and self.config.TEST_MODE:
            self.sf_start, self.sf_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.roi_start, self.roi_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.det_start, self.det_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )

        if self.config.TEST_MODE:
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

        # Bind active execution permanently to sub-stream BEFORE entering loop (Bypasses TLS Overhead)
        # if self.device_input == "cuda" and torch.cuda.is_available():
        #     torch.cuda.set_stream(self.inference_stream)
        # if hasattr(self, "target_fps") and hasattr(self, "duration_target"):
        #     self.max_target_frames = int(self.duration_target * float(self.target_fps))
        # elif hasattr(self, "numFrames"):
        #     self.max_target_frames = self.numFrames
        # else:
        #     self.max_target_frames = float("inf")

        if (
            hasattr(self, "numFrames")
            and self.numFrames > 0
            and not getattr(self, "is_rtsp", False)
        ):
            # For local test files, compute total target frames from input video length
            self.max_target_frames = int(
                self.numFrames / getattr(self.reader, "step_size", 1.0)
            )
        elif hasattr(self, "target_fps") and hasattr(self, "duration_target"):
            self.max_target_frames = int(self.duration_target * float(self.target_fps))
        else:
            self.max_target_frames = float("inf")

        # self.dynamic_limit = max(2, int(0.5 * self.target_fps))

        # Initialize a thread-safe signaling handle at class setup (run_realtime_inference)
        self.queue_data_ready_event = threading.Event()

        try:
            for i in range(num_prefetch_workers):
                # prefetch_thread = mp.Process(
                prefetch_thread = threading.Thread(
                    target=lambda: global_frame_prefetch_worker_v1(self),
                    daemon=True,
                    # name=f"{self.name}_prefetch_{i}",
                )
                prefetch_thread.start()
                self.prefetch_threads.append(prefetch_thread)
        except Exception as e:
            main_app_logger.critical(
                f"[{self.name}] Failed to start prefetch workers: {e}", exc_info=True
            )
            traceback.print_exc()
            raise

    @torch.inference_mode()
    def run_realtime_inference_v1(
        self,
        sf_enabled=True,
        profiler=None,
        gt_enabled=False,
        read_frame_only=False,
    ):
        self.status = "RUNNING"

        metrics = {}
        all_metadata = {}
        num_objs = 0
        total_pipeline_time_ms = 0.0
        total_read_time = 0.0
        total_queue_saturation = 0.0
        total_written_frames = 0
        total_sf_pipeline_time_ms = 0.0
        real_world_latency_ms = 0.0
        total_run_pipelinefn = 0
        total_frame_loop_latency = 0.0
        total_active_processing_overhead = 0.0
        queue_saturation_history = []
        consecutive_slow_frames = 0
        coverage_percentages = []

        if gt_enabled and hasattr(self, "VIDEO_GT_DETAILS") and self.VIDEO_GT_DETAILS:
            # VIDEO_GT_DETAILS = download_eval_data([self.name])
            gt_boxes = self.VIDEO_GT_DETAILS[self.name]["gt_bbox"]
            gt_sequence = self.VIDEO_GT_DETAILS[self.name]["sequence"]

        self.initialize_run_realtime_inference(
            read_frame_only, self.active_workers_count
        )

        self.stat_start_time = time.perf_counter()
        pipeline_start_time = self.stat_start_time
        last_loop_cycle_timestamp = self.stat_start_time

        missing_frame_cnt = 0
        max_retries = int(self.target_fps)

        # Release the master startup event at the last possible moment.
        self.main_startup_event.set()
        while (
            self.active  # or not self.prefetch_queue.empty()
        ):  # and self.frame_count_target < self.max_target_frames:
            if self.frame_count_target >= self.max_target_frames:
                self.active = False
                self.reader.running_flag.value = False
                break

            stat_start_time = time.perf_counter()
            try:
                # FRAME RETRIEVAL ---------------------------------------------
                try:
                    frame_start_time = time.perf_counter()
                    safe_frame, frame_details, should_continue = (
                        self._get_frame_from_queue()
                    )

                    if not should_continue:
                        self.active = False  # Signal loop termination
                        break
                    if safe_frame is None:
                        # Allow retries while prefetch workers are still active
                        if getattr(self, "prefetch_active", True) and not getattr(
                            self.reader, "stopped", False
                        ):
                            time.sleep(0.002)
                            continue
                        missing_frame_cnt += 1
                        if missing_frame_cnt >= max_retries:
                            self.active = False
                            main_app_logger.info(
                                "Too many frames missing and workers done. Exiting..."
                            )
                            break
                        continue  # Skip to the next iteration if the frame is invalid

                    # try:
                    #     # ret, slot_idx = self.prefetch_queue.get(block=True, timeout=1.0)
                    #     # ret, slot_idx = self.prefetch_queue.get(block=False)
                    #     ret, slot_idx = self.prefetch_queue.get(block=True, timeout=1.0)
                    #     self.prefetch_queue.task_done()

                    # except queue.Empty:
                    #     with self.worker_tracking_lock:
                    #         if (
                    #             self.active_workers_count == 0
                    #             and self.prefetch_queue.empty()
                    #         ):
                    #             self.active = False
                    #             break
                    #     # If the queue is empty AND the prefetch workers are done, we can exit.
                    #     if not self.active:
                    #         break
                    #     continue

                    # # If self.stop() drops the queues, break out of the thread loop natively.
                    # except (OSError, ValueError, AssertionError):
                    #     main_app_logger.info(
                    #         "[PROCESS THREAD] Ingestion queue disconnected via stop() signal. Breaking loop."
                    #     )
                    #     self.active = False
                    #     break

                    # if ret is False or slot_idx == "END_OF_STREAM":
                    #     main_app_logger.info(
                    #         "[PROCESS THREAD] End of video stream detected. Breaking loop naturally."
                    #     )
                    #     self.active = False
                    #     break

                    # if not self.active:
                    #     break

                    # # if slot_idx == -1 or self.shm_buffer_pool[slot_idx] is None:
                    # #     continue
                    # if (
                    #     slot_idx == -1
                    #     or not hasattr(self, "shm_buffer_pool")
                    #     or self.shm_buffer_pool is None
                    # ):
                    #     continue
                    # if (
                    #     slot_idx >= len(self.shm_buffer_pool)
                    #     or self.shm_buffer_pool[slot_idx] is None
                    # ):
                    #     continue

                    # # Zero-Copy Reference Extraction straight out of the memory slot array
                    # (
                    #     raw_shm_frame,
                    #     current_event,
                    #     frame_num,
                    #     abs_frame_num,
                    #     true_read_latency_secs,
                    # ) = self.shm_buffer_pool[slot_idx]

                    # if frame_num == 0:
                    #     main_app_logger.info(
                    #         f"[VERIFY - CONSUMER] Main loop is officially processing Frame {frame_num}!",
                    #     )
                    # # self.shm_buffer_pool[slot_idx] = None

                    # reader_time = (
                    #     true_read_latency_secs  # time.perf_counter() - frame_start_time
                    # )
                    # total_read_time += reader_time

                    # if "cuda" in str(self.device_input):
                    #     safe_frame = self.gpu_input[slot_idx]
                    #     # safe_frame = cpu_tensor.to(self.device_input, non_blocking=True)
                    #     safe_frame.copy_(raw_shm_frame)
                    #     # self.gpu_input[slot_idx].zero_()
                    # else:
                    #     safe_frame = raw_shm_frame

                    # # Record the PyTorch CUDA event on the current stream
                    # if current_event is not None and isinstance(
                    #     current_event, torch.cuda.Event
                    # ):
                    #     # This tells the GPU that the consumer is done reading this slot
                    #     current_event.record(torch.cuda.current_stream())

                    # Originally 0-based but make 1-based
                    # frame_num += 1
                    # abs_frame_num += 1
                    total_read_time += frame_details["reader_time"]
                    missing_frame_cnt = 0

                    # Calculate the exact time gap between the last frame completion
                    # and the start of the next read. This accurately captures downstream GIL blocks.
                    cycle_gap = frame_start_time - last_loop_cycle_timestamp
                    # Subtract the actual read duration to isolate the thread stall time
                    true_serialization_stall = max(
                        0.0, (cycle_gap - frame_details["reader_time"]) * 1000.0
                    )
                    self.component_stats["queue_blocked"].append(
                        true_serialization_stall
                    )

                except queue.Empty:
                    if getattr(self.reader, "reconnect_failed", False):
                        self.active = False
                        break
                    if not self.active:
                        break  # Exit on error if shutdown has been initiated
                    time.sleep(0.001)
                    continue

                # FRAME PROCESSING ---------------------------------------------
                # stat_start_time = time.perf_counter()
                # stat_start_time = self.stat_start_time
                self.abs_frame_num = frame_details["abs_frame_num"]
                calculated_clip_id = (
                    frame_details["frame_num"] - 1
                ) // self.max_frames_per_clip
                self.frame_count += 1
                self.frame_count_target += 1
                self.frame_in_clip_count += 1

                # Real-Time Metric A: Track Queue Saturation Density Ratio
                # current_q_size = self.reader.frame_queue.qsize()
                saturation_ratio = (
                    len(self.reader.frame_queue) / self.reader.frame_queue.maxlen
                )
                total_queue_saturation += saturation_ratio
                queue_saturation_history.append(saturation_ratio)

                # RUN PIPELINE_FN ---------------------------------------------
                run_pipelinefn_start = time.perf_counter()
                # self.frame_count_target += 1
                # self.frame_in_clip_count += 1

                if gt_enabled:
                    target_boxes_array = self.get_frame_gt_boxes(
                        frame_details["abs_frame_num"], gt_sequence, gt_boxes
                    )

                metadata_or_bbs, metrics, motion_detected = self.pipeline_fn(
                    safe_frame,
                    frame_details["frame_num"],
                    stat_start_time,
                    calculated_clip_id,
                    gt_boxes=target_boxes_array if gt_enabled else None,
                    read_frame_only=read_frame_only,
                )

                num_objs += len(metadata_or_bbs.keys())
                # if metadata_or_bbs is not None and isinstance(metadata_or_bbs, dict):
                #     all_metadata.update(metadata_or_bbs)

                total_written_frames += 1

                # FRAME POST PROCESSING ---------------------------------------------
                # Track Execution Boundaries
                frame_end_time = time.perf_counter()
                total_run_pipelinefn += frame_end_time - run_pipelinefn_start
                last_loop_cycle_timestamp = frame_end_time

                # Calculate frame metrics

                # Real-Time Metric B: Frame Processing Latency Check
                frame_loop_latency = (frame_end_time - frame_start_time) * 1000  # ms
                total_frame_loop_latency += frame_loop_latency

                # Isolate active processing time (excluding the queue blocking read)
                active_processing_overhead = frame_loop_latency - (
                    frame_details["reader_time"] * 1000
                )
                total_active_processing_overhead += active_processing_overhead

                # RUNTIME TERMINAL WARNING ALERTS
                if saturation_ratio >= 1.0 or active_processing_overhead > (
                    1000 / self.target_fps
                ):
                    consecutive_slow_frames += 1
                    if (
                        consecutive_slow_frames % self.target_fps == 0
                    ):  # Throttle terminal spam
                        reader_time = frame_details["reader_time"]
                        main_app_logger.info(
                            f"\033[93m⚠️ [PERF WARNING] Main loop starving! Waiting on stream ingestion... "
                            f"Read Wait: {reader_time:.1f}ms | Other Wait: {active_processing_overhead:.1f}ms | Queue Fullness: {saturation_ratio * 100:.0f}%\033[0m"
                        )
                else:
                    consecutive_slow_frames = max(0, consecutive_slow_frames - 1)
                # if self.device_input == "cuda" and torch.cuda.is_available():
                #     torch.cuda.default_stream().synchronize()

                if gt_enabled and motion_detected:
                    self.update_eval_stats(metadata_or_bbs, target_boxes_array)

                if metrics != {}:
                    num_crops = len(metrics["bbs"]) if metrics["bbs"] is not None else 0
                    self.crops_per_frame_list.append(num_crops)

                    # Context-aware extraction of the newly proposed metrics
                    density = metrics.get(
                        "batch_density", 0 if self.config.sf_enabled else 1
                    )
                    self.component_stats["batch_sizes"].append(density)

                    self.component_stats["sf"].append(metrics["sf_time"])
                    # self.component_stats["roi"].append(metrics["roi_time"])
                    self.component_stats["det"].append(metrics["det_time"])

                    # Calculate coverage OUTSIDE the timed block to prevent interference
                    if self.config.sf_enabled and metrics.get("bbs") is not None:
                        cov = self.calculate_unique_coverage(metrics["bbs"])
                        if hasattr(cov, "item"):
                            coverage_percentages.append(float(cov.item()))
                        else:
                            coverage_percentages.append(float(cov))

                # Explicitly delete frame variables to free their references
                frame_8k = None
                del frame_8k, safe_frame
                if "metadata_or_bbs" in locals():
                    del metadata_or_bbs

                if total_written_frames % (10 * self.target_fps) == 0:
                    # main_app_logger.info(
                    #     f"Captured {total_written_frames}/{self.max_target_frames} frames..."
                    # )
                    # Querying the cross-process property once every 5 seconds reduces proxy overhead to 0%
                    # if self.device_input == "cuda" and hasattr(
                    #     self.reader, "total_h2d_time"
                    # ):
                    #     avg_h2d = (
                    #         self.reader.total_h2d_time
                    #         / max(1, total_written_frames + 1)
                    #     ) * 1000.0
                    #     main_app_logger.info(
                    #         f"Captured {total_written_frames}/{self.max_target_frames} frames... DMA Upload: {avg_h2d:.2f}ms"
                    #     )
                    # else:
                    main_app_logger.info(
                        f"Captured {total_written_frames}/{self.max_target_frames} frames..."
                    )

                if self.frame_count % 100 == 0:
                    gc.collect()

                # Force PyTorch's internal allocator to release cached segments back to the OS
                # if self.frame_count_target % (5 * self.target_fps) == 0:
                #     main_app_logger.info(
                #         f"Captured {self.frame_count_target}/{self.max_target_frames} frames..."
                #     )
                #     torch.cuda.empty_cache()
                #     torch.cuda.ipc_collect()

                # Force a microscopic micro-yield if executing on CPU.
                # This grants immediate execution priority back to the background cleaning thread,
                # allowing the garbage collector to evict data from RAM instantly!
                if self.device_input == "cpu":
                    time.sleep(
                        0
                    )  # .005)  # 1ms yield breaks core processor starvation lock

                # END -> frame processing
            except torch.cuda.OutOfMemoryError:
                main_app_logger.info("!" * 70)
                main_app_logger.info(
                    "[CRITICAL TEST CRASH] GPU MEMORY CEILING HIT INSIDE RUNNER LOOP!"
                )
                main_app_logger.info(
                    "Freezing allocation history registers and writing diagnostic log..."
                )
                main_app_logger.info("!" * 70)

                try:
                    snapshot_filename = (
                        f"/tmp/test_vram_leak_profile_pid{os.getpid()}.pickle"
                    )
                    torch.cuda.memory._dump_snapshot(snapshot_filename)
                    main_app_logger.info(
                        f"[PROFILER SUCCESSFUL] Snapshot profile written to: {snapshot_filename}"
                    )
                    main_app_logger.info(
                        "--> Drag and drop this file directly into: https://pytorch.org"
                    )
                except Exception as dump_err:
                    main_app_logger.info(
                        f"Failed to record profile data snapshot: {dump_err}"
                    )

                # Force safe system unlinking of background workers to clean up OS handles
                self.active = False
                if hasattr(self, "reader") and self.reader is not None:
                    self.reader.stop()
                raise

            except Exception as e:
                if not self.active or not getattr(self, "prefetch_active", True):
                    main_app_logger.info(
                        "[PROCESS THREAD] System shutdown detected during exception sweep. Exiting thread payload context."
                    )
                    break

                main_app_logger.info(
                    f"[CRITICAL PIPELINE ERROR] Crash on frame: {repr(e)}"
                )
                traceback.print_exc()
                raise e  # Let it break so you can see the exact line number!

        # END -> while self.active and self.frame_count_target < self.max_target_frames:

        # PIPELINE POST PROCESSING ---------------------------------------------
        pipeline_end_time = time.perf_counter()

        real_world_latency_ms = float(
            (pipeline_end_time - pipeline_start_time) * 1000.0
        )

        total_sf_pipeline_time_ms = float(
            sum(self.component_stats.get("sf", [0.0]))
            + sum(self.component_stats.get("roi", [0.0]))
            + sum(self.component_stats.get("det", [0.0]))
        )
        total_pipeline_time_ms = float(
            (pipeline_end_time - pipeline_start_time) * 1000.0
        )

        self.async_writer.release()

        # Continuously sample the thread work pool backlog state mechanics
        if hasattr(self, "executor") and self.executor is not None:
            # Safely probe internal concurrent.futures work queue bounds
            self.component_stats["thread_backlog"].append(
                self.executor._work_queue.qsize()
            )
        else:
            self.component_stats["thread_backlog"].append(0)

        # CALCULATE PERFORMANCE METRICS ---------------------------------------------
        main_app_logger.info(
            f"Execution Finished. Total Output Frames Written: {self.frame_count_target}"
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
                computed_eval_metrics=self.evaluator.compute_final_metrics()
                if hasattr(self, "evaluator")
                else None
                if gt_enabled
                else None,
            )
        else:
            main_app_logger.info(
                f"[SKIPPED SUMMARY] Only {self.frame_count_target} frames processed. "
            )

        # Force early hardware driver sweep before unbinding threads
        if self.device_input == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()

        if profiler is not None:
            profiler.stop()

        self.stop()

    def initialize_run_realtime_inference(self, read_frame_only, num_prefetch_workers):
        self.processor = self._setup_processor(read_frame_only)

        if hasattr(self, "processor") and hasattr(self.processor, "label_source"):
            self.label_source = self.processor.label_source

        if (
            hasattr(self, "numFrames")
            and self.numFrames > 0
            and not getattr(self, "is_rtsp", False)
        ):
            # For local test files, compute total target frames from input video length
            self.max_target_frames = int(
                self.numFrames / getattr(self.reader, "step_size", 1.0)
            )
        elif hasattr(self, "target_fps") and hasattr(self, "duration_target"):
            self.max_target_frames = int(self.duration_target * float(self.target_fps))
        else:
            self.max_target_frames = float("inf")
        # =========================================================================
        # 🚀 PRE-ALLOCATED METRICS SCRATCHPAD (Zero Dynamic Allocations in Loop)
        # =========================================================================
        cap = max(
            1000,
            self.max_target_frames if self.max_target_frames != float("inf") else 10000,
        )
        self._metric_sf_times = np.zeros(cap, dtype=np.float32)
        self._metric_det_times = np.zeros(cap, dtype=np.float32)
        self._metric_reader_times = np.zeros(cap, dtype=np.float32)
        self._metric_queue_blocked = np.zeros(cap, dtype=np.float32)
        self._metric_queue_saturations = np.zeros(cap, dtype=np.float32)
        self._metric_crops_count = np.zeros(cap, dtype=np.int32)
        self._metric_batch_densities = np.zeros(cap, dtype=np.int32)
        self._metric_coverage_tensors = []
        # ============================

        if self.is_cuda and not hasattr(self, "sf_start") and self.config.TEST_MODE:
            self.sf_start, self.sf_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.roi_start, self.roi_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self.det_start, self.det_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )

        if self.config.TEST_MODE:
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

        # Initialize a thread-safe signaling handle at class setup (run_realtime_inference)
        self.queue_data_ready_event = threading.Event()

        try:
            for i in range(num_prefetch_workers):
                # prefetch_thread = mp.Process(
                prefetch_thread = threading.Thread(
                    target=lambda: global_frame_prefetch_worker_v1(self),
                    daemon=True,
                    # name=f"{self.name}_prefetch_{i}",
                )
                prefetch_thread.start()
                self.prefetch_threads.append(prefetch_thread)
        except Exception as e:
            main_app_logger.critical(
                f"[{self.name}] Failed to start prefetch workers: {e}", exc_info=True
            )
            traceback.print_exc()
            raise

    @torch.inference_mode()
    def run_realtime_inference(
        self,
        sf_enabled=True,
        profiler=None,
        gt_enabled=False,
        read_frame_only=False,
    ):
        self.status = "RUNNING"
        self.initialize_run_realtime_inference(
            read_frame_only, self.active_workers_count
        )

        gt_boxes = None
        gt_sequence = None
        if gt_enabled and hasattr(self, "VIDEO_GT_DETAILS") and self.VIDEO_GT_DETAILS:
            gt_boxes = self.VIDEO_GT_DETAILS[self.name]["gt_bbox"]
            gt_sequence = self.VIDEO_GT_DETAILS[self.name]["sequence"]

        self.stat_start_time = time.perf_counter()
        pipeline_start_time = self.stat_start_time
        last_loop_cycle_timestamp = self.stat_start_time

        missing_frame_cnt = 0
        max_retries = int(self.target_fps)
        total_read_time = 0.0
        total_written_frames = 0
        total_queue_saturation = 0.0
        total_run_pipelinefn = 0.0
        total_frame_loop_latency = 0.0
        total_active_processing_overhead = 0.0
        consecutive_slow_frames = 0
        num_objs = 0
        queue_saturation_history = []
        coverage_percentages = []

        # Release the master startup event
        self.main_startup_event.set()

        while self.active:
            if self.frame_count_target >= self.max_target_frames:
                self.active = False
                if hasattr(self.reader, "running_flag"):
                    self.reader.running_flag.value = False
                break

            stat_start_time = time.perf_counter()
            try:
                # 1. FRAME RETRIEVAL ------------------------------------------
                frame_start_time = time.perf_counter()
                safe_frame, frame_details, should_continue = (
                    self._get_frame_from_queue()
                )

                if not should_continue:
                    self.active = False
                    break

                if safe_frame is None:
                    if getattr(self, "prefetch_active", True) and not getattr(
                        self.reader, "stopped", False
                    ):
                        time.sleep(0.001)
                        continue
                    missing_frame_cnt += 1
                    if missing_frame_cnt >= max_retries:
                        self.active = False
                        main_app_logger.info(
                            "Too many frames missing and workers done. Exiting..."
                        )
                        break
                    continue

                missing_frame_cnt = 0
                total_read_time += frame_details["reader_time"]

                # Downstream Serialization Pressure Tracking
                cycle_gap = frame_start_time - last_loop_cycle_timestamp
                true_serialization_stall = max(
                    0.0, (cycle_gap - frame_details["reader_time"]) * 1000.0
                )
                self.component_stats["queue_blocked"].append(true_serialization_stall)

                # Frame counters and clipping indices
                self.abs_frame_num = frame_details["abs_frame_num"]
                calculated_clip_id = (
                    frame_details["frame_num"] - 1
                ) // self.max_frames_per_clip
                self.frame_count += 1
                self.frame_count_target += 1
                self.frame_in_clip_count += 1

                # Queue Saturation Density Ratio Tracking
                if (
                    hasattr(self.reader, "frame_queue")
                    and self.reader.frame_queue.maxlen
                ):
                    saturation_ratio = (
                        len(self.reader.frame_queue) / self.reader.frame_queue.maxlen
                    )
                else:
                    saturation_ratio = 0.0
                total_queue_saturation += saturation_ratio
                queue_saturation_history.append(saturation_ratio)

                # 2. RUN PIPELINE_FN ------------------------------------------
                run_pipelinefn_start = time.perf_counter()

                target_boxes_array = None
                if gt_enabled:
                    target_boxes_array = self.get_frame_gt_boxes(
                        frame_details["abs_frame_num"], gt_sequence, gt_boxes
                    )

                metadata_or_bbs, metrics, motion_detected = self.pipeline_fn(
                    safe_frame,
                    frame_details["frame_num"],
                    stat_start_time,
                    calculated_clip_id,
                    gt_boxes=target_boxes_array if gt_enabled else None,
                    read_frame_only=read_frame_only,
                )

                num_objs += len(metadata_or_bbs.keys())
                total_written_frames += 1

                # 3. FRAME POST-PROCESSING & COMPONENT METRICS ----------------
                frame_end_time = time.perf_counter()
                total_run_pipelinefn += frame_end_time - run_pipelinefn_start
                last_loop_cycle_timestamp = frame_end_time

                frame_loop_latency = (frame_end_time - frame_start_time) * 1000.0
                total_frame_loop_latency += frame_loop_latency

                active_processing_overhead = frame_loop_latency - (
                    frame_details["reader_time"] * 1000.0
                )
                total_active_processing_overhead += active_processing_overhead

                if gt_enabled and motion_detected:
                    self.update_eval_stats(metadata_or_bbs, target_boxes_array)

                if metrics != {}:
                    num_crops = (
                        len(metrics["bbs"]) if metrics.get("bbs") is not None else 0
                    )
                    self.crops_per_frame_list.append(num_crops)

                    density = metrics.get(
                        "batch_density", 0 if self.config.sf_enabled else 1
                    )
                    self.component_stats["batch_sizes"].append(density)
                    self.component_stats["sf"].append(metrics.get("sf_time", 0.0))
                    self.component_stats["det"].append(metrics.get("det_time", 0.0))

                    # Asynchronous GPU coverage collection (NO per-frame GPU stall)
                    if self.config.sf_enabled and metrics.get("bbs") is not None:
                        # cov = self.calculate_unique_coverage(metrics["bbs"])
                        # if hasattr(cov, "item"):
                        #     coverage_percentages.append(float(cov.item()))
                        # else:
                        #     coverage_percentages.append(float(cov))
                        raw_bbs = metrics["bbs"]
                        coverage_percentages.append(
                            raw_bbs.clone() if torch.is_tensor(raw_bbs) else raw_bbs
                        )

                # Clean temporary loop variables
                del safe_frame
                if "metadata_or_bbs" in locals():
                    del metadata_or_bbs

                if total_written_frames % (10 * int(self.target_fps)) == 0:
                    main_app_logger.info(
                        f"Captured {total_written_frames}/{self.max_target_frames} frames..."
                    )

                # if self.frame_count % 100 == 0:
                #     gc.collect()

            except Exception as e:
                if not self.active or not getattr(self, "prefetch_active", True):
                    break
                main_app_logger.info(
                    f"[CRITICAL PIPELINE ERROR] Crash on frame: {repr(e)}"
                )
                traceback.print_exc()
                raise e

        # =====================================================================
        # POST-LOOP BENCHMARK CALCULATIONS (Single Batch Reduction at Teardown)
        # =====================================================================
        pipeline_end_time = time.perf_counter()
        total_pipeline_ms = float((pipeline_end_time - pipeline_start_time) * 1000.0)

        if hasattr(self, "async_writer") and self.async_writer is not None:
            self.async_writer.release()

        main_app_logger.info(
            f"Execution Finished. Total Output Frames Written: {total_written_frames}"
        )

        computed_eval_metrics = None
        if gt_enabled and hasattr(self, "evaluator"):
            computed_eval_metrics = self.evaluator.compute_final_metrics()

        if hasattr(self.__class__, "benchmarks"):
            print(f"SELF.__CLASS__.BENCHMARKS: {self.__class__.benchmarks}")
        elif hasattr(self, "benchmarks"):
            print(f"SELF.BENCHMARKS: {self.benchmarks}")

        self._finalize_benchmarks(
            num_objs=num_objs,
            total_written_frames=total_written_frames,
            total_pipeline_ms=total_pipeline_ms,
            real_world_latency_ms=total_pipeline_ms,
            total_sf_pipeline_ms=float(
                sum(self.component_stats.get("sf", [0.0]))
                + sum(self.component_stats.get("det", [0.0]))
            ),
            frame_loop_latency_ms=total_frame_loop_latency
            / max(1, total_written_frames),
            coverage_percentages=coverage_percentages,
            total_read_time=total_read_time,
            total_queue_saturation=total_queue_saturation,
            queue_saturation_history=queue_saturation_history,
            computed_eval_metrics=computed_eval_metrics,
        )

        if self.device_input == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

        # gc.collect()
        if profiler is not None:
            profiler.stop()

        self.stop()

    # CLEANUP --------------------------------------------

    def clean_up_tensors_and_arrays(self):
        main_app_logger.info("[CLEANUP] Safely stripping active runtime arrays ...")

        # 1. Force disable tracking states to stop graph allocation recursion
        # torch.set_grad_enabled(False)

        # if hasattr(torch, "jit") and hasattr(torch.jit, "_builtins"):
        #     if isinstance(torch.jit._builtins, dict):
        #         torch.jit._sbuiltins.clear()
        #     else:
        #         torch.jit._builtins = {}

        # 2. Collect references safely using strict type checking
        # This completely avoids pulling unmanaged proxy objects from the heap
        all_live_objects = gc.get_objects()

        target_tensors = []
        target_arrays = []

        for obj in all_live_objects:
            try:
                obj_type = type(obj)
                if isinstance(obj, types.FrameType):
                    frame_info = inspect.getframeinfo(obj)
                    if (
                        "openvino" in frame_info.filename
                        or "openvino.py" in frame_info.filename
                    ):
                        # Clear the frame's local variable dictionary to break circular references
                        obj.f_locals.clear()

                # Check for concrete types to prevent triggering proxy __getattr__ hooks
                if obj_type is torch.Tensor:
                    target_tensors.append(obj)
                elif obj_type is np.ndarray:
                    # Explicitly guard size checks to keep it completely stable
                    if obj.base is None and obj.ndim > 0:
                        target_arrays.append(obj)
            except Exception:
                pass

        # 3. Truncate discovered references in-place without triggering deletions
        reclaimed_tensors = 0
        for tensor in target_tensors:
            try:
                # Truncate raw storage footprint safely
                tensor.data = torch.empty(0, device=self.device_input)
                reclaimed_tensors += 1
            except Exception:
                pass

        reclaimed_arrays = 0
        for arr in target_arrays:
            try:
                # Shrink writeable numpy arrays down to 0 bytes safely
                if arr.flags.writeable:
                    arr.resize((0,), refcheck=False)
                    reclaimed_arrays += 1
            except (ValueError, SystemError):
                pass

        main_app_logger.info(
            f"[CLEANUP] Reclaimed {reclaimed_tensors} tensors and {reclaimed_arrays} arrays safely."
        )

        # Clean local registers immediately
        all_live_objects = None
        target_tensors = None
        target_arrays = None

        # Clear out low-level traceback registries if active
        # if hasattr(sys, "exc_info"):
        if hasattr(sys, "exc_clear"):
            sys.exc_clear()  # if hasattr(sys, "exc_clear") else None

        # Walk up the execution frame tree and clear out f_locals references
        try:
            frame_cursor = inspect.currentframe()
            while frame_cursor:
                # Empty out active scopes to force unlinking of trapped closures
                frame_cursor.f_locals.clear()
                frame_cursor = frame_cursor.f_back
        except Exception:
            pass
        finally:
            # Prevent the frame cursor property itself from pinning the block
            del frame_cursor
        gc.collect()

    def execute_teardown(self):
        if hasattr(self, "baseline_before_start"):
            setattr(self, "baseline_before_start", None)

        if hasattr(self, "evaluator") and self.evaluator is not None:
            try:
                # If your evaluator class tracks raw coordinate arrays in a list/dict,
                # force-clear them to allow Python to free up the unmanaged memory:
                if hasattr(self.evaluator, "history"):
                    self.evaluator.history.clear()
                if hasattr(self.evaluator, "all_predictions"):
                    self.evaluator.all_predictions.clear()
            except Exception:
                pass
            self.evaluator = None

        if hasattr(self, "component_stats") and self.component_stats:
            self.component_stats.clear()

        release_native_linux_heap()

        # main_app_logger.info(
        #     "[TEST HARNESS] Test Teardown complete. Releasing session cleanly.\n",
        #     ,
        # )

    # HELPER FUNCTIONS --------------------------------------------
    def _finalize_benchmarks_v1(
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
        computed_eval_metrics=None,
    ):
        """Aggregates metrics and adds them to the results list."""
        # num_objs = len(all_metadata.keys())
        sf_enabled = self.config.sf_enabled
        stat_frame_count = self.stat_frame_count
        stat_fps = self.stat_fps
        total_frames = self.abs_frame_num  # self.reader.total_input_frames

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
            total_frames / self.reader.input_fps if self.reader.input_fps > 0 else 0
        )
        out_duration_s = (
            total_written_frames / self.reader.target_fps
            if self.reader.target_fps > 0
            else 0
        )
        # main_app_logger.info(
        #     f"Expected duration: {self.duration_target:.2f} stream_duration_s: {duration_s:.2f} output_duration_s: {out_duration_s:.2f}"
        # )

        h2d_display_label = f"Reader {h2d_label}:"
        avg_reader_ms = (total_read_time / total_written_frames) * 1000
        # avg_resize_ms = (total_resize_time / total_written_frames)*1000
        # avg_write_ms = (total_disk_write_overhead / total_written_frames)*1000
        target_frame_fps = total_written_frames / total_pipeline_s
        all_frame_fps = total_frames / total_pipeline_s
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
            (1.0 - (total_written_frames / max(1, total_frames))) * 100.0,
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
        # avg_reader_ms = (total_read_time / total_frames) * 1000
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
        # avg_cov = (
        #     sum(coverage_percentages) / len(coverage_percentages)
        #     if coverage_percentages
        #     else (100.0 if not sf_enabled else 0)
        # )
        # peak_cov = max(coverage_percentages) if coverage_percentages else 0.0

        if coverage_percentages:
            if torch.is_tensor(coverage_percentages[0]):
                cov_stack = torch.stack(coverage_percentages)
                avg_cov = cov_stack.mean().item()
                peak_cov = cov_stack.max().item()
            else:
                avg_cov = sum(coverage_percentages) / len(coverage_percentages)
                peak_cov = max(coverage_percentages) if coverage_percentages else 0.0
        else:
            avg_cov = 100.0 if not sf_enabled else 0.0
            peak_cov = 0.0
        # If coverage_percentages contains un-synchronized GPU tensors, reduce them all at once!
        # if coverage_percentages and isinstance(coverage_percentages[0], torch.Tensor):
        #     # Stack all frame elements into a single contiguous GPU tensor block
        #     coverage_tensor_stack = torch.stack(coverage_percentages)

        #     # Calculate reductions fast on the device without stepping back onto the CPU
        #     avg_cov = torch.mean(coverage_tensor_stack).item()
        #     peak_cov = torch.max(coverage_tensor_stack).item()

        #     # Safe fallback: Convert the tracking list back into plain floats for standard reporting
        #     coverage_percentages = coverage_tensor_stack.cpu().tolist()
        # else:
        #     # Standard fallback if the script is running on a CPU target pass
        #     avg_cov = sum(coverage_percentages) / max(1, len(coverage_percentages))
        #     peak_cov = max(coverage_percentages) if coverage_percentages else 0.0

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

        main_app_logger.info(
            "=" * 60,
        )
        main_app_logger.info(
            f"     FULLY-OPTIMIZED ASYNC STAGE ({self.config.DEVICE}) PIPELINE BREAKDOWN  "
        )
        main_app_logger.info(
            "=" * 60,
        )
        main_app_logger.info(
            f"Total Output Frames Written:   {total_written_frames}",
        )
        main_app_logger.info(
            f"Total Pipeline Execution Time: {total_pipeline_s:.4f} seconds",
        )
        main_app_logger.info(
            f"Overall Processing Speed:      {target_frame_fps:.2f} FPS",
        )
        main_app_logger.info(
            "=" * 60,
        )
        main_app_logger.info(
            "MAIN CONSUMER LOOP TIMELINE (SEQUENTIAL OVERHEAD):",
        )
        main_app_logger.info(
            f" 1. Shared Memory Copy to Host:                {avg_copy_ms:6.2f} ms"
        )
        main_app_logger.info(
            f" 2. GIL / Downstream Queue Serialization Stalls: {avg_blocked_ms:6.2f} ms"
        )
        main_app_logger.info(
            f" 3. Pure Video Frame File-Ingestion Read:      {avg_reader_ms:6.2f} ms"
        )
        main_app_logger.info(
            "=" * 60,
        )
        main_app_logger.info(
            "BACKGROUND INGESTION HEALTH (ASYNC METRICS):",
        )
        main_app_logger.info(
            f" A. Inbound Stream Decode Speed:               {all_frame_fps:6.2f} FPS"
        )
        main_app_logger.info(
            f" B. {h2d_display_label:<42} {avg_h2d_ms:6.2f} ms",
        )
        main_app_logger.info(
            f" C. Consumer Queue Saturation Density:         {avg_saturation:6.1f}%"
        )
        main_app_logger.info(
            f" D. Peak PCIe Upload Throughput:               {self.pcie_throughput_gbps:6.2f} GB/s"
        )
        main_app_logger.info(
            f" E. Active Thread Pool Work Backlog:           {avg_backlog:6.1f} tasks"
        )
        main_app_logger.info(
            f" F. VRAM Hardware Memory Efficiency (Cache):   {self.vram_efficiency:6.1f}%"
        )
        main_app_logger.info(
            "=" * 60,
        )

        # EXPANDED PERFORMANCE INSIGHTS AND DIAGNOSTICS
        main_app_logger.info("PIPELINE HEALTH & BEHAVIOR INSIGHTS:")
        if avg_saturation > 80.0:
            pipeline_status = "CHOKED"
            main_app_logger.info(
                f" • Status: \033[91m🔴 {pipeline_status}\033[0m (Downstream logic cannot keep up with inbound stream speed)"
            )
        elif avg_saturation > 40.0:
            pipeline_status = "BALANCED"
            main_app_logger.info(
                f" • Status: \033[93m🟡 {pipeline_status}\033[0m (Queue buffer is actively pacing consumer workloads)"
            )
        else:
            pipeline_status = "IDLE/LIGHT"
            main_app_logger.info(
                f" • Status: \033[92m🟢 {pipeline_status}\033[0m (Consumer loop finishes ahead of background ingestion clock)"
            )

        main_app_logger.info(
            f" • Peak Queue Fullness Reached:               {peak_saturation:.1f}%"
        )
        # Calculate stream drop indicator
        main_app_logger.info(
            f" • Targeted Pacing Delivery Accuracy:        {target_match_ratio:.1f}%"
        )
        main_app_logger.info(
            f" • Downstream Serialization Pressure Index:   {serial_pressure}"
        )
        main_app_logger.info(
            f" • Core Queue Evacuation Velocity Rank:       {evac_vel}",
        )
        main_app_logger.info(
            f" • Hardware Compute Backpressure Index:        {backpressure_index:6.2f}%"
        )
        main_app_logger.info(
            f" • Preprocessing to Inference Cost Density:    {model_cost_density:6.2f}x"
        )
        main_app_logger.info(
            f" • Physical PCIe Gen4 Bus Saturation Level:    {pcie_bus_saturation:6.2f}%"
        )
        main_app_logger.info(
            "=" * 60,
        )

        result_dict = {
            "Test Name": self._testMethodName,
            "Detection Type": self.config.DETECTION_TYPE,
            "Device": self.config.DEVICE,
            "Smart Filtering": "Enabled" if sf_enabled else "Disabled",
            "Video": self.source,  # self.name?
            "Video FPS": f"{self.reader.input_fps:.2f}",
            "Video Original Duration (s)": f"{duration_s:.4f}",
            "Video Frames": total_frames,
            # TARGET OVERVIEW
            "Target FPS": f"{self.reader.target_fps:.2f}",
            "Output Duration (s)": f"{out_duration_s:.4f}",
            "Output Frames": total_written_frames,
        }

        if computed_eval_metrics:
            result_dict.update(
                {
                    # EVAL OVERVIEW
                    "Max_Recall_IoU_10": computed_eval_metrics["Max_Recall_IoU_10"],
                    "mAP_10": computed_eval_metrics["mAP_10"],
                    "mAP_50": computed_eval_metrics["mAP_50"],
                    "mAP_75": computed_eval_metrics["mAP_75"],
                    "mAP_10_95": computed_eval_metrics["mAP_10_95"],
                    "Precision_10": computed_eval_metrics["Precision_10"],
                    "Recall_10": computed_eval_metrics["Recall_10"],
                    "F1_Score_10": computed_eval_metrics["F1_Score_10"],
                    "Precision_50": computed_eval_metrics["Precision_50"],
                    "Recall_50": computed_eval_metrics["Recall_50"],
                    "F1_Score_50": computed_eval_metrics["F1_Score_50"],
                }
            )

        result_dict.update(
            {
                # PIPELINE OVERVIEW
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
                "Peak Area Coverage %": f"{peak_cov:.2f}%",
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
        self.__class__.benchmarks.append(result_dict)

        main_app_logger.info(
            f"[{self._testMethodName}] Pipeline Latency (s): {total_pipeline_s:.2f} sec"
        )
        main_app_logger.info(
            f"[{self._testMethodName}] Pipeline FPS (Target frames): {target_frame_fps:.2f} ({total_written_frames} frames)"
        )
        main_app_logger.info(
            f"[{self._testMethodName}] Pipeline FPS (Video frames): {all_frame_fps:.2f} ({total_frames} frames)"
        )
        main_app_logger.info(
            f"[{self._testMethodName}] SF/Detection FPS (Target frames): {det_est_fps:.2f} ({total_written_frames} frames)"
        )
        main_app_logger.info(
            f"[{self._testMethodName}] Display FPS: {stat_fps:.2f} ({stat_frame_count} frames)"
        )

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
        computed_eval_metrics=None,
    ):
        """Aggregates metrics and adds them to the results list."""
        sf_enabled = self.config.sf_enabled
        stat_frame_count = self.stat_frame_count
        stat_fps = self.stat_fps
        total_frames = self.abs_frame_num

        h2d_label = (
            "Pinned H2D Transfer (PCIe DMA)"
            if self.is_cuda
            else "Data Preparation Overhead"
        )

        total_pipeline_s = total_pipeline_ms / 1000.0
        total_real_pipeline_s = real_world_latency_ms / 1000.0
        real_est_fps = (
            self.frame_count_target / total_real_pipeline_s
            if total_real_pipeline_s > 0
            else 0.0
        )

        duration_s = (
            total_frames / self.reader.input_fps if self.reader.input_fps > 0 else 0.0
        )
        out_duration_s = (
            total_written_frames / self.reader.target_fps
            if self.reader.target_fps > 0
            else 0.0
        )

        h2d_display_label = f"Reader {h2d_label}:"
        avg_reader_ms = (
            (total_read_time / total_written_frames) * 1000.0
            if total_written_frames > 0
            else 0.0
        )
        target_frame_fps = (
            total_written_frames / total_pipeline_s if total_pipeline_s > 0 else 0.0
        )
        all_frame_fps = total_frames / total_pipeline_s if total_pipeline_s > 0 else 0.0
        avg_copy_ms = (
            (self.reader.total_shm_copy_time / total_written_frames) * 1000.0
            if total_written_frames > 0
            else 0.0
        )

        avg_blocked_ms = (
            sum(self.component_stats.get("queue_blocked", [0.0]))
            / max(1, len(self.component_stats.get("queue_blocked", [])))
            if self.component_stats.get("queue_blocked")
            else 0.0
        )

        avg_saturation = (
            (total_queue_saturation / total_written_frames) * 100.0
            if total_written_frames > 0
            else 0.0
        )
        peak_saturation = (
            max(queue_saturation_history) * 100.0 if queue_saturation_history else 0.0
        )
        target_match_ratio = (
            (total_written_frames / (total_pipeline_s * self.reader.target_fps)) * 100.0
            if (total_pipeline_s > 0 and self.reader.target_fps > 0)
            else 0.0
        )

        frame_drop = max(
            0.0,
            (1.0 - (total_written_frames / max(1, total_frames))) * 100.0,
        )
        serial_pressure = (avg_blocked_ms / max(0.1, frame_loop_latency_ms)) * 100.0
        evac_vel = target_frame_fps / max(0.1, all_frame_fps)
        backpressure_index = (avg_blocked_ms / max(0.1, frame_loop_latency_ms)) * 100.0

        if self.is_cuda:
            avg_h2d_ms = (
                (self.reader.total_h2d_time / total_written_frames) * 1000.0
                if total_written_frames > 0
                else 0.0
            )
            avg_prep_ms = 0.0
        else:
            avg_prep_ms = (
                (self.reader.total_h2d_time / total_written_frames) * 1000.0
                if total_written_frames > 0
                else 0.0
            )
            avg_h2d_ms = 0.0

        frame_size_gb = (self.reader.frame_height * self.reader.frame_width * 3) / (
            1024**3
        )

        avg_h2d_s = (
            (
                sum(self.component_stats["dma_upload"])
                / len(self.component_stats["dma_upload"])
            )
            / 1000.0
            if self.component_stats.get("dma_upload")
            else 0.0
        )
        if self.is_cuda and avg_h2d_s > 0.00001:
            self.pcie_throughput_gbps = frame_size_gb / avg_h2d_s
        else:
            self.pcie_throughput_gbps = 0.0

        pcie_bus_saturation = (self.pcie_throughput_gbps / 31.5) * 100.0

        if self.is_cuda and torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(0) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(0) / 1024**2
            self.vram_efficiency = (
                (peak_alloc / peak_reserved * 100.0) if peak_reserved > 0 else 100.0
            )
        else:
            self.vram_efficiency = 0.0

        avg_sf = (
            sum(self.component_stats["sf"]) / len(self.component_stats["sf"])
            if self.component_stats.get("sf")
            else 0.0
        )
        avg_roi = (
            sum(self.component_stats["roi"]) / len(self.component_stats["roi"])
            if self.component_stats.get("roi")
            else 0.0
        )
        avg_det = (
            sum(self.component_stats["det"]) / len(self.component_stats["det"])
            if self.component_stats.get("det")
            else 0.0
        )

        det_latency_ms = avg_sf + avg_roi + avg_det
        det_latency_s = det_latency_ms / 1000.0
        det_est_fps = (
            (total_written_frames / det_latency_s) if det_latency_s > 0 else 0.0
        )

        preprocessing_overhead = avg_sf + avg_roi
        model_cost_density = avg_det / max(0.1, preprocessing_overhead)

        loop_overhead = max(
            0.0, 100.0 - ((det_latency_ms / max(1.0, frame_loop_latency_ms)) * 100.0)
        )

        # Batch GPU Reduction for Coverage Tensors
        if coverage_percentages:
            cov_results = []
            for bbs in coverage_percentages:
                if bbs is not None and len(bbs) > 0:
                    cov_results.append(self.calculate_unique_coverage(bbs))
                else:
                    cov_results.append(0.0)

            avg_cov = sum(cov_results) / max(1, len(cov_results))
            peak_cov = max(cov_results)
        else:
            avg_cov = 100.0 if not sf_enabled else 0.0
            peak_cov = 0.0

        avg_crops = (
            sum(self.crops_per_frame_list) / len(self.crops_per_frame_list)
            if self.crops_per_frame_list
            else 0.0
        )

        capped_frames = sum(1 for c in self.crops_per_frame_list if c >= 20)
        cap_rate = (
            (capped_frames / len(self.crops_per_frame_list)) * 100.0
            if self.crops_per_frame_list
            else 0.0
        )

        backlog_list = self.component_stats.get("thread_backlog", [])
        avg_backlog = (
            sum(backlog_list) / len(backlog_list) if len(backlog_list) > 0 else 0.0
        )

        # Formatted Performance Breakdown Logging
        main_app_logger.info("=" * 60)
        main_app_logger.info(
            f"     FULLY-OPTIMIZED ASYNC STAGE ({self.config.DEVICE}) PIPELINE BREAKDOWN  "
        )
        main_app_logger.info("=" * 60)
        main_app_logger.info(f"Total Output Frames Written:   {total_written_frames}")
        main_app_logger.info(
            f"Total Pipeline Execution Time: {total_pipeline_s:.4f} seconds"
        )
        main_app_logger.info(
            f"Overall Processing Speed:      {target_frame_fps:.2f} FPS"
        )
        main_app_logger.info("=" * 60)
        main_app_logger.info("MAIN CONSUMER LOOP TIMELINE (SEQUENTIAL OVERHEAD):")
        main_app_logger.info(
            f" 1. Shared Memory Copy to Host:                  {avg_copy_ms:6.2f} ms"
        )
        main_app_logger.info(
            f" 2. GIL / Downstream Queue Serialization Stalls: {avg_blocked_ms:6.2f} ms"
        )
        main_app_logger.info(
            f" 3. Pure Video Frame File-Ingestion Read:        {avg_reader_ms:6.2f} ms"
        )
        main_app_logger.info("=" * 60)
        main_app_logger.info("BACKGROUND INGESTION HEALTH (ASYNC METRICS):")
        main_app_logger.info(
            f" A. Inbound Stream Decode Speed:                 {all_frame_fps:6.2f} FPS"
        )
        main_app_logger.info(f" B. {h2d_display_label:<42} {avg_h2d_ms:6.2f} ms")
        main_app_logger.info(
            f" C. Consumer Queue Saturation Density:           {avg_saturation:6.1f}%"
        )
        main_app_logger.info(
            f" D. Peak PCIe Upload Throughput:                 {self.pcie_throughput_gbps:6.2f} GB/s"
        )
        main_app_logger.info(
            f" E. Active Thread Pool Work Backlog:             {avg_backlog:6.1f} tasks"
        )
        main_app_logger.info(
            f" F. VRAM Hardware Memory Efficiency (Cache):     {self.vram_efficiency:6.1f}%"
        )
        main_app_logger.info("=" * 60)

        main_app_logger.info("PIPELINE HEALTH & BEHAVIOR INSIGHTS:")
        if avg_saturation > 80.0:
            pipeline_status = "CHOKED"
            main_app_logger.info(
                f" • Status: \033[91m🔴 {pipeline_status}\033[0m (Downstream logic cannot keep up with inbound stream speed)"
            )
        elif avg_saturation > 40.0:
            pipeline_status = "BALANCED"
            main_app_logger.info(
                f" • Status: \033[93m🟡 {pipeline_status}\033[0m (Queue buffer is actively pacing consumer workloads)"
            )
        else:
            pipeline_status = "IDLE/LIGHT"
            main_app_logger.info(
                f" • Status: \033[92m🟢 {pipeline_status}\033[0m (Consumer loop finishes ahead of background ingestion clock)"
            )

        main_app_logger.info(
            f" • Peak Queue Fullness Reached:               {peak_saturation:.1f}%"
        )
        main_app_logger.info(
            f" • Targeted Pacing Delivery Accuracy:        {target_match_ratio:.1f}%"
        )
        main_app_logger.info(
            f" • Downstream Serialization Pressure Index:   {serial_pressure:.4f}"
        )
        main_app_logger.info(
            f" • Core Queue Evacuation Velocity Rank:       {evac_vel:.4f}"
        )
        main_app_logger.info(
            f" • Hardware Compute Backpressure Index:        {backpressure_index:6.2f}%"
        )
        main_app_logger.info(
            f" • Preprocessing to Inference Cost Density:    {model_cost_density:6.2f}x"
        )
        main_app_logger.info(
            f" • Physical PCIe Gen4 Bus Saturation Level:    {pcie_bus_saturation:6.2f}%"
        )
        main_app_logger.info("=" * 60)

        result_dict = {
            "Test Name": self._testMethodName,
            "Detection Type": self.config.DETECTION_TYPE,
            "Device": self.config.DEVICE,
            "Smart Filtering": "Enabled" if sf_enabled else "Disabled",
            "Video": self.source,
            "Video FPS": f"{self.reader.input_fps:.2f}",
            "Video Original Duration (s)": f"{duration_s:.4f}",
            "Video Frames": total_frames,
            "Target FPS": f"{self.reader.target_fps:.2f}",
            "Output Duration (s)": f"{out_duration_s:.4f}",
            "Output Frames": total_written_frames,
        }

        if computed_eval_metrics:
            result_dict.update(
                {
                    "Max_Recall_IoU_10": computed_eval_metrics.get(
                        "Max_Recall_IoU_10", 0.0
                    ),
                    "mAP_10": computed_eval_metrics.get("mAP_10", 0.0),
                    "mAP_50": computed_eval_metrics.get("mAP_50", 0.0),
                    "mAP_75": computed_eval_metrics.get("mAP_75", 0.0),
                    "mAP_10_95": computed_eval_metrics.get("mAP_10_95", 0.0),
                    "Precision_10": computed_eval_metrics.get("Precision_10", 0.0),
                    "Recall_10": computed_eval_metrics.get("Recall_10", 0.0),
                    "F1_Score_10": computed_eval_metrics.get("F1_Score_10", 0.0),
                    "Precision_50": computed_eval_metrics.get("Precision_50", 0.0),
                    "Recall_50": computed_eval_metrics.get("Recall_50", 0.0),
                    "F1_Score_50": computed_eval_metrics.get("F1_Score_50", 0.0),
                }
            )

        result_dict.update(
            {
                "Pipeline Latency (s)": f"{total_pipeline_s:.2f}",
                "Pipeline FPS (Video frames)": f"{all_frame_fps:.2f}",
                "Pipeline FPS (Target frames)": f"{target_frame_fps:.2f}",
                "Real Pipeline Latency (s)": f"{total_real_pipeline_s:.2f}",
                "Real Pipeline FPS (Target frames)": f"{real_est_fps:.2f}",
                "Avg Frame Reading (ms)": f"{avg_reader_ms:.2f}",
                "Avg Host Copy (ms)": f"{avg_copy_ms:.2f}",
                "Avg DMA Upload (ms)": f"{avg_h2d_ms:.2f}",
                "Avg Data Prep Overhead (ms)": f"{avg_prep_ms:.2f}",
                "Avg Queue Blocked (ms)": f"{avg_blocked_ms:.2f}",
                "Avg Queue Saturation (%)": f"{avg_saturation:.2f}",
                "Loop Overhead %": f"{loop_overhead:.2f}%",
                "Inbound Frame Drop %": f"{frame_drop:.2f}%",
                "Serialization Pressure %": f"{serial_pressure:.2f}%",
                "Evacuation Velocity": f"{evac_vel:.2f}x",
                "Compute Backpressure Index": f"{backpressure_index:.2f}%",
                "Model Cost Density Ratio": f"{model_cost_density:.2f}x",
                "PCIe Bus Saturation %": f"{pcie_bus_saturation:.2f}%",
                "Pipeline Status": pipeline_status,
                "Peak Queue Fullness (%)": f"{peak_saturation:.2f}",
                "Targeted Pacing Delivery (%)": f"{target_match_ratio:.2f}",
                "SF/Detection Latency (s)": f"{det_latency_s:.2f}",
                "SF/Detection FPS": f"{det_est_fps:.2f}",
                "Avg SF (ms)": f"{avg_sf:.2f}",
                "Avg ROI (ms)": f"{avg_roi:.2f}",
                "Avg Obj. Detection (ms)": f"{avg_det:.2f}",
                "Total Breakdown Sum (ms)": f"{det_latency_ms:.2f}",
                "Avg Area Coverage %": f"{avg_cov:.2f}%",
                "Peak Area Coverage %": f"{peak_cov:.2f}%",
                "Avg Crops/Frame": f"{avg_crops:.1f}",
                "Crop Cap Rate (>20)": f"{cap_rate:.1f}%",
                "Objects Detected": num_objs,
                "Display Latency (s)": f"{self.elapsed_display_time:.2f}",
                "Display Frames": stat_frame_count,
                "Display FPS": f"{stat_fps:.2f}",
                "PCIe Bandwidth Throughput (GB/s)": f"{self.pcie_throughput_gbps:.2f}",
                "Avg Async Thread Pool Backlog": f"{sum(self.component_stats.get('thread_backlog', [0])) / max(1, len(self.component_stats.get('thread_backlog', [1]))):.1f}",
                "VRAM Allocation Efficiency (%)": f"{self.vram_efficiency:.1f}%",
            }
        )
        self.__class__.benchmarks.append(result_dict)

        main_app_logger.info(
            f"[{self._testMethodName}] Pipeline Latency (s): {total_pipeline_s:.2f} sec"
        )
        main_app_logger.info(
            f"[{self._testMethodName}] Pipeline FPS (Target frames): {target_frame_fps:.2f} ({total_written_frames} frames)"
        )
        main_app_logger.info(
            f"[{self._testMethodName}] Pipeline FPS (Video frames): {all_frame_fps:.2f} ({total_frames} frames)"
        )
        main_app_logger.info(
            f"[{self._testMethodName}] SF/Detection FPS (Target frames): {det_est_fps:.2f} ({total_written_frames} frames)"
        )
        main_app_logger.info(
            f"[{self._testMethodName}] Display FPS: {stat_fps:.2f} ({stat_frame_count} frames)"
        )

    def motion2metadata(self, merged, frame_count_target):
        metadata = {}
        if merged is not None and merged.size > 0:
            # merged = merged / self.scales_tensor.cpu().numpy().reshape(-1, 4)
            merged = merged.div(self.scales_tensor.view(-1, 4))

            # Calculate area for each box: (xmax - xmin) * (ymax - ymin)
            widths = merged[:, 2] - merged[:, 0]
            heights = merged[:, 3] - merged[:, 1]
            areas = widths * heights
            max_area = np.max(areas) if np.max(areas) > 0 else 1.0

            # Format motion results like detection results for evaluation
            for i, bb in enumerate(merged):
                disp_x, disp_y, disp_x2, disp_y2 = bb
                disp_w = disp_x2 - disp_x
                disp_h = disp_y2 - disp_y

                if disp_w > 2 and disp_h > 2:
                    obj_id = len(metadata)
                    # num_objs += 1
                    framenum_str = f"{frame_count_target:04d}_{obj_id:04d}"
                    score = areas[i] / max_area

                    metadata[framenum_str] = {
                        "frameId": int(frame_count_target),
                        "bbId": framenum_str,
                        "bbox": {
                            "x": int(disp_x),
                            "y": int(disp_y),
                            "height": int(disp_h),
                            "width": int(disp_w),
                            "object": "",
                            "object_det": {
                                "confidence": score,
                                "frameH": int(self.resize_h),
                                "frameW": int(self.resize_w),
                            },
                        },
                    }

        return metadata

    def calculate_unique_coverage_v1(self, merged_boxes, target_w=640, target_h=640):
        """
        Pure Loopless Vectorized Coverage Engine.
        Eliminates sequential Python loop slicing overhead via parallel hardware broadcast registers.
        """
        if merged_boxes is None or len(merged_boxes) == 0:
            return 0.0

        target_device = self.device_input if hasattr(self, "device_input") else "cpu"

        if isinstance(merged_boxes, np.ndarray):
            if merged_boxes.size == 0:
                return 0.0
            boxes_tensor = torch.as_tensor(
                merged_boxes, device=target_device, dtype=torch.float32
            )
        else:
            if merged_boxes.shape[0] == 0:
                return 0.0
            boxes_tensor = merged_boxes.to(device=target_device, dtype=torch.float32)

        # Vectorized scaling projection matrix
        scale = torch.tensor(
            [
                target_w / self.frame_width,
                target_h / self.frame_height,
                target_w / self.frame_width,
                target_h / self.frame_height,
            ],
            device=target_device,
            dtype=torch.float32,
        )

        coords = (boxes_tensor * scale).long()
        coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, target_w)
        coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, target_h)

        # Build parallel coordinate meshes on target hardware
        # grid_y, grid_x = torch.meshgrid(
        #     torch.arange(target_h, device=target_device),
        #     torch.arange(target_w, device=target_device),
        #     indexing="ij"
        # )

        # Extract coordinate tracking vectors
        x1 = coords[:, 0].unsqueeze(1).unsqueeze(2)
        y1 = coords[:, 1].unsqueeze(1).unsqueeze(2)
        x2 = coords[:, 2].unsqueeze(1).unsqueeze(2)
        y2 = coords[:, 3].unsqueeze(1).unsqueeze(2)

        # Direct 3D array tensor broadcast reduction sum pass
        # inside_mask = (grid_x >= x1_v) & (grid_x < x2_v) & (grid_y >= y1_v) & (grid_y < y2_v)
        inside_boxes = (
            (self._cached_grid_x >= x1)
            & (self._cached_grid_x < x2)
            & (self._cached_grid_y >= y1)
            & (self._cached_grid_y < y2)
        )
        unique_mask = torch.any(inside_boxes, dim=0)

        coverage_percentage = (
            torch.sum(unique_mask).float().div(target_w * target_h).mul(100.0)
        )
        return coverage_percentage.item()

    def calculate_unique_coverage(self, merged_boxes, target_w=640, target_h=640):
        if merged_boxes is None or len(merged_boxes) == 0:
            return None

        target_device = self.device_input if hasattr(self, "device_input") else "cpu"

        if isinstance(merged_boxes, np.ndarray):
            if merged_boxes.size == 0:
                return None
            boxes_tensor = torch.as_tensor(
                merged_boxes, device=target_device, dtype=torch.float32
            )
        else:
            if merged_boxes.shape[0] == 0:
                return None
            boxes_tensor = merged_boxes.to(device=target_device, dtype=torch.float32)

        # Scale coordinates to target space
        scale_x = target_w / self.frame_width
        scale_y = target_h / self.frame_height

        x1 = (boxes_tensor[:, 0] * scale_x).clamp(0, target_w).long().view(-1, 1, 1)
        y1 = (boxes_tensor[:, 1] * scale_y).clamp(0, target_h).long().view(-1, 1, 1)
        x2 = (boxes_tensor[:, 2] * scale_x).clamp(0, target_w).long().view(-1, 1, 1)
        y2 = (boxes_tensor[:, 3] * scale_y).clamp(0, target_h).long().view(-1, 1, 1)

        # 3D broadcast evaluation
        inside_boxes = (
            (self._cached_grid_x >= x1)
            & (self._cached_grid_x < x2)
            & (self._cached_grid_y >= y1)
            & (self._cached_grid_y < y2)
        )
        unique_mask = torch.any(inside_boxes, dim=0)

        # Return the 0-dim GPU tensor directly without calling .item()!
        return torch.sum(unique_mask).float().div(target_w * target_h).mul(100.0)

    def print_active_cpu_tensor_memory(self):
        """
        CPU EQUIVALENT TO print_active_gpu_tensor_memory:
        Scans the global Python garbage collection registries to locate, measure,
        and map every single live PyTorch tensor actively residing on host RAM.
        """

        main_app_logger.info(
            "=" * 60,
        )
        main_app_logger.info(
            "[RAM INVESTIGATOR] Scanning Active CPU Tensors with Sources:",
        )
        main_app_logger.info(
            f"{'ALLOCATED SIZE':<15} | {'TENSOR SHAPE':<25} | {'DATA TYPE':<15}"
        )
        main_app_logger.info(
            "-" * 80,
        )

        total_cpu_tensor_bytes = 0
        live_tensor_count = 0

        # Force a shallow sweep to catch immediately orphaned pointers first
        gc.collect()

        # Iterate over the entire active Python heap
        for obj in gc.get_objects():
            try:
                # Target only genuine PyTorch tensors residing on the host CPU
                if torch.is_tensor(obj) and not obj.is_cuda and obj.nelement() > 0:
                    live_tensor_count += 1

                    # Calculate true memory footprint based on element size and count
                    element_size = obj.element_size()
                    num_elements = obj.nelement()
                    tensor_bytes = num_elements * element_size
                    total_cpu_tensor_bytes += tensor_bytes

                    # Filter out tiny scalar metrics to keep logs focused on leaks (e.g., > 10 KB)
                    if tensor_bytes > 10240:
                        size_kb = tensor_bytes / 1024
                        shape_str = str(list(obj.shape))
                        dtype_str = str(obj.dtype).replace("torch.", "")

                        main_app_logger.info(
                            f"{size_kb:>10.1f} KiB  | {shape_str:<25} | {dtype_str:<15}"
                        )
            except Exception:
                # Guard against objects being destroyed concurrently during the scan loop
                pass

        total_cpu_tensor_mb = total_cpu_tensor_bytes / (1024 * 1024)
        main_app_logger.info(
            "-" * 80,
        )
        main_app_logger.info(
            f"Total Live CPU Tensor Count: {live_tensor_count}",
        )
        main_app_logger.info(
            f"Total Live CPU Tensor Memory: {total_cpu_tensor_mb:.2f} MB",
        )
        main_app_logger.info(
            "=" * 60,
        )

    # def print_active_gpu_tensor_memory(self):
    #     main_app_logger.info(
    #         "=" * 60,
    #     )
    #     main_app_logger.info(
    #         "\033[95m[VRAM INVESTIGATOR] Scanning Active GPU Tensors with Sources:\033[0m"
    #     )

    #     # Capture the exact C++ memory registry structure
    #     try:
    #         raw_snapshot = torch.cuda.memory._snapshot()
    #         segments = raw_snapshot.get("segments", [])
    #     except Exception:
    #         segments = []
    #         main_app_logger.info(
    #             "[WARN] Failed parsing native memory context snapshot.",
    #         )

    #     # Map raw block storage memory addresses straight to Python frames
    #     addr_to_source = {}
    #     for seg in segments:
    #         for block in seg.get("blocks", []):
    #             if block.get("state") == "active_allocated":
    #                 addr = block.get("address")
    #                 history = block.get("history", [])
    #                 if history:
    #                     # Inspect the deepest frame in the allocation stack
    #                     frame = history[-1]
    #                     filename = frame.get("filename", "Unknown")
    #                     lineno = frame.get("line", 0)
    #                     func_name = frame.get("name", "unknown_func")
    #                     addr_to_source[addr] = f"{filename}:{lineno} ({func_name})"

    #     # Scan the heap via GC and resolve actual backing storage layers
    #     leaked_tensors = []
    #     total_detected_bytes = 0

    #     for obj in gc.get_objects():
    #         try:
    #             if torch.is_tensor(obj) and obj.is_cuda:
    #                 t_bytes = obj.element_size() * obj.nelement()
    #                 total_detected_bytes += t_bytes
    #                 if t_bytes > 0:
    #                     leaked_tensors.append(obj)
    #         except Exception:
    #             pass

    #     obj = None  # Break tracking register references

    #     for i, tensor in enumerate(leaked_tensors):
    #         t_bytes = tensor.element_size() * tensor.nelement()

    #         # --- THE FIX: Extract the address of the underlying storage block ---
    #         try:
    #             if hasattr(tensor, "untyped_storage"):
    #                 storage_addr = tensor.untyped_storage().data_ptr()
    #             elif hasattr(tensor, "storage") and tensor.storage():
    #                 storage_addr = tensor.storage().data_ptr()
    #             else:
    #                 storage_addr = tensor.data_ptr()
    #         except Exception:
    #             storage_addr = tensor.data_ptr()

    #         # Extract absolute code trace locations from our snapshot dictionary
    #         source_loc = addr_to_source.get(
    #             storage_addr, "Unknown Native C++ Allocation / Model Context"
    #         )

    #         main_app_logger.info(
    #             f" > Tensor {i:3d} | Shape: {str(list(tensor.shape)):<18} | "
    #             f"Size: {t_bytes / 1024**2:6.2f} MB | Source: \033[93m{source_loc}\033[0m"
    #         )

    #     del leaked_tensors
    #     main_app_logger.info(
    #         f"[VRAM INVESTIGATOR] Total Live Tensor Memory: {total_detected_bytes / 1024**2:.2f} MB"
    #     )
    #     gc.collect()
    #     if (
    #         torch.cuda.is_available()
    #     ):  # self.device_input == "cuda" and torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #         torch.cuda.empty_cache()  # Flush the pool BEFORE the guard snapshots it

    # def print_active_shared_memory(self):
    #     main_app_logger.info(
    #         "=" * 60,
    #     )
    #     main_app_logger.info(
    #         "\033[96m[SHM INVESTIGATOR] Scanning Active OS Shared Memory Filesystem Tables:\033[0m"
    #     )
    #     try:
    #         shm_dir = Path("/dev/shm")
    #         if shm_dir.exists():
    #             # Extract and inventory all live POSIX memory segments allocated right now
    #             shm_files = [
    #                 f
    #                 for f in shm_dir.iterdir()
    #                 if f.is_file()
    #                 and not f.name.startswith("sem.")
    #                 and not f.name.startswith("psm")
    #             ]
    #             main_app_logger.info(
    #                 f" > Discovered Live OS-Mapped Memory Nodes: {len(shm_files)}"
    #             )

    #             for f_path in shm_files:
    #                 try:
    #                     f_stat = f_path.stat()
    #                     size_mb = f_stat.st_size / (1024 * 1024)

    #                     # Highlight the files using visual color anchors for scannability
    #                     main_app_logger.info(
    #                         f"   ⚠️  \033[93m[ALIVE SHM NODE]\033[0m File: {f_path.name:<25} | Size: {size_mb:7.2f} MB"
    #                     )
    #                 except Exception:
    #                     pass
    #         else:
    #             main_app_logger.info(
    #                 "   [ERROR] /dev/shm runtime directory is inaccessible on this host context."
    #             )
    #     except Exception as e:
    #         main_app_logger.info(
    #             f"   [WARN] Kernel inspection execution pass failed: {e}"
    #         )

    #     # gc.collect()
    #     # if torch.cuda.is_available():
    #     #     torch.cuda.synchronize()
    #     #     torch.cuda.empty_cache()  # Flush the pool BEFORE the guard snapshots it

    # def calculate_leaked_memory(
    #     self, device, video_name, start_allocated, start_reserved
    # ):
    #     _testMethodName = f"{video_name}_{device}"
    #     main_app_logger.info("=" * 60)
    #     max_allowed_leak = 1024 * 1024  # 1MB buffer allowance
    #     msg = "[LEAKAGE INVESTIGATOR] Scanning memory allocations:\n"
    #     if device == "gpu" and torch.cuda.is_available():
    #         # torch.cuda.synchronize()

    #         end_allocated = torch.cuda.memory_allocated(0)
    #         end_reserved = torch.cuda.memory_reserved(0)

    #         leak_allocated = end_allocated - start_allocated
    #         leak_reserved = end_reserved - start_reserved
    #         # max_allowed_leak = 1024 * 1024  # 1MB buffer allowance

    #         if leak_allocated > max_allowed_leak:
    #             msg += (
    #                 f"\n🔴 GPU Memory Leak Detected for {_testMethodName}!\n"
    #                 f"Check for dangling references or missing 'del' statements\n\n"
    #             )
    #             # else:
    #             #     msg = "\n"

    #         msg += (
    #             f"\tPre-Setup Allocation:  {start_allocated / 1024**2:.2f} MB\n"
    #             f"\tPost-Teardown Allocation:  {end_allocated / 1024**2:.2f} MB\n"
    #             f"\tNet Leaked VRAM: {leak_allocated / 1024**2:.2f} MB\n"
    #             f"\tNet Leaked Reserved Blocks: {leak_reserved / 1024**2:.2f} MB"
    #         )

    #         # main_app_logger.info(msg, )
    #     else:
    #         process = psutil.Process(os.getpid())
    #         end_rss = process.memory_info().rss

    #         # start_allocated and start_reserved must be populated with baseline RSS in each_test_setup
    #         leak_rss = end_rss - start_allocated

    #         if leak_rss > max_allowed_leak:
    #             msg += (
    #                 f"\n🔴 CPU Memory Leak Detected for {_testMethodName}!\n"
    #                 f"Check for dangling references or missing 'del' statements\n\n"
    #             )
    #         # else:
    #         #     msg = "\n"

    #         msg += (
    #             f" Pre-Setup Host RAM Allocation: {start_allocated / 1024**2:.2f} MB\n"
    #         )
    #         msg += f" Post-Teardown Host RAM Allocation: {end_rss / 1024**2:.2f} MB\n"
    #         msg += f" Net Leaked Host System Memory: {leak_rss / 1024**2:.2f} MB"
    #     main_app_logger.info(msg)
    #     main_app_logger.info("=" * 60)

    # def diagnostic_profiler(self, device, video_name, start_allocated, start_reserved):
    #     _testMethodName = f"{video_name}_{device}"
    #     self.calculate_leaked_memory(
    #         device, video_name, start_allocated, start_reserved
    #     )

    #     # main_app_logger.info("=" * 60, )
    #     # main_app_logger.info(
    #     #     f"\n\033[95m[DIAGNOSTICS] Starting Automated Leak Analysis for {_testMethodName}...\033[0m",
    #     #     ,
    #     # )

    #     # 1. Run objgraph to inspect the Python object reference trees before clearing containers
    #     # try:
    #     # main_app_logger.info(
    #     #     "\033[94m[DIAGNOSTICS] Python Object Registry Standings:\033[0m",
    #     # )
    #     # objgraph.show_most_common_types(limit=10)

    #     # Check if the metrics arrays are pinning references inside memory
    #     # for tracker_attr in ["all_preds", "all_targets"]:
    #     #     if hasattr(self, tracker_attr):
    #     #         tgt_list = getattr(self, tracker_attr)
    #     #         if len(tgt_list) > 0:
    #     #             graph_path = f"/tmp/backrefs_{tracker_attr}_{device}.png"
    #     #             main_app_logger.info(
    #     #                 f"\033[93m[WARN] '{tracker_attr}' contains {len(tgt_list)} entries. Generating reference graph to: {graph_path}\033[0m"
    #     #             )
    #     #             objgraph.show_backrefs(
    #     #                 [tgt_list], max_depth=3, filename=graph_path
    #     #             )
    #     # except ImportError:
    #     #     main_app_logger.info(
    #     #         "\033[91m[DIAGNOSTICS] 'objgraph' package missing. Skipping reference chain mapping. (pip install objgraph)\033[0m"
    #     #     )

    #     # 2. Dump PyTorch Memory Snapshot before clearing VRAM caches
    #     if device == "gpu" and torch.cuda.is_available():
    #         try:
    #             # snapshot_path = f"/tmp/vram_leak_profile_{self._testMethodName}.pickle"
    #             snapshot_path = self.output_path.replace(".mp4", "_vram_profile.html")
    #             torch.cuda.memory._dump_snapshot(snapshot_path)
    #             main_app_logger.info(
    #                 f"\033[92m[DIAGNOSTICS] VRAM Snapshot Trace generated successfully: {snapshot_path}\033[0m"
    #             )
    #             main_app_logger.info(
    #                 "\033[92m--> Upload this file to https://pytorch.org to inspect leak allocation stacks.\033[0m"
    #             )

    #             with open(snapshot_path, "rb") as f:
    #                 snapshot = pickle.load(f)

    #             # Print an HTML visualization path map of the allocations
    #             html_timeline = memory_viz.trace_plot(snapshot)
    #             html_path = snapshot_path.replace(".pickle", ".html")
    #             with open(html_path, "w", encoding="utf-8") as f:
    #                 f.write(html_timeline)
    #         except Exception as e:
    #             main_app_logger.info(
    #                 f"\033[91m[DIAGNOSTICS] Failed to generate PyTorch memory snapshot: {e}\033[0m"
    #             )

    # def assess_memory(self, device, video_name, start_allocated, start_reserved):
    #     gc.collect()

    #     surviving_arrays = [
    #         obj
    #         for obj in gc.get_objects()
    #         # if isinstance(obj, np.ndarray) and obj.size >= 1  #(1920 * 1080)
    #         if type(obj) is np.ndarray and obj.ndim > 0
    #     ]
    #     analyze_tracemalloc_snapshot()

    #     main_app_logger.info(
    #         f"[DIAGNOSTICS] Found {len(surviving_arrays)} uncollected large arrays alive in RAM filesystem."
    #     )

    #     for i, arr in enumerate(surviving_arrays):
    #         referrers = gc.get_referrers(arr)
    #         main_app_logger.info(
    #             f"  > Array {i} | Shape: {arr.shape} | Pinned by {len(referrers)} references:"
    #         )
    #         for ref in referrers:
    #             if isinstance(ref, dict):
    #                 main_app_logger.info(
    #                     f"    - Dict Keys holding this array: {list(ref.keys())[:4]}"
    #                 )
    #             else:
    #                 main_app_logger.info(
    #                     f"    - Variable holding object layout: {type(ref)}",
    #                 )
    #     # ───────────────────────────────────────

    #     # analyze_tracemalloc_snapshot()

    #     if device == "gpu":
    #         self.print_active_gpu_tensor_memory()
    #     else:
    #         # --- CPU RAM PATH METRICS ---
    #         # main_app_logger.info("=" * 60, )
    #         # main_app_logger.info("\n[RAM INVESTIGATOR] Scanning Host CPU Memory Standings:", )
    #         # process = psutil.Process(os.getpid())
    #         # current_rss = process.memory_info().rss / (1024 * 1024)  # Host RAM in MB
    #         # main_app_logger.info(f" > Current Process Resident Set Size (RSS): {current_rss:.2f} MB", )
    #         self.print_active_cpu_tensor_memory()

    #     self.print_active_shared_memory()

    #     # AUTOMATED DIAGNOSTIC PROFILING PHASE (TRIGGERED ON TEST TEARDOWN)
    #     self.diagnostic_profiler(device, video_name, start_allocated, start_reserved)

    def _print_gpu_mem(self):
        if self.device_input == "cuda" and torch.cuda.is_available():
            # Memory currently used by tensors
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            # Total memory reserved by PyTorch (the "Pool")
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            main_app_logger.info(f"\tAllocated: {allocated:0.2f} MB")
            main_app_logger.info(f"\tReserved:  {reserved:0.2f} MB")
        else:
            main_app_logger.info("\tCUDA not available.")

    def get_frame_gt_boxes(self, abs_frame_num, gt_sequence, gt_boxes):
        if abs_frame_num - 1 >= len(gt_sequence):
            return np.empty((0, 4), dtype=np.float32)

        gt_seq = int(gt_sequence[abs_frame_num - 1])
        if gt_seq == abs_frame_num:
            gt_b_xywh = gt_boxes[abs_frame_num - 1]
            gt_b = [
                gt_b_xywh[0],
                gt_b_xywh[1],
                min(gt_b_xywh[0] + gt_b_xywh[2], target_width - 1),
                min(gt_b_xywh[1] + gt_b_xywh[3], target_height - 1),
            ]
            # Force a 2D matrix shape layout of (1, 4) even for a single box
            target_boxes_array = np.array([gt_b], dtype=np.float32).reshape(-1, 4)
            # target_labels_array = np.array([0], dtype=np.int64)
        else:
            # gt_b = [0,0,0,0]
            target_boxes_array = np.empty((0, 4), dtype=np.float32)

        return target_boxes_array

    def update_eval_stats(self, metadata_or_bbs, target_boxes_array):
        """
        metadata_or_bbs: In 640 (resize) space
        """

        # gt_seq = int(gt_sequence[abs_frame_num - 1])
        # if gt_seq == abs_frame_num:
        #     gt_b_xywh = gt_boxes[abs_frame_num - 1]
        #     gt_b = [
        #         gt_b_xywh[0],
        #         gt_b_xywh[1],
        #         min(gt_b_xywh[0] + gt_b_xywh[2], target_width - 1),
        #         min(gt_b_xywh[1] + gt_b_xywh[3], target_height - 1),
        #     ]
        #     # Force a 2D matrix shape layout of (1, 4) even for a single box
        #     target_boxes_array = np.array([gt_b], dtype=np.float32).reshape(
        #         -1, 4
        #     )
        #     # target_labels_array = np.array([0], dtype=np.int64)
        # else:
        #     # gt_b = [0,0,0,0]
        #     target_boxes_array = np.empty((0, 4), dtype=np.float32)
        # target_labels_array = np.empty((0,), dtype=np.int64)

        bbs = []
        scores = []
        labels = []
        if isinstance(metadata_or_bbs, dict):
            for _, v in metadata_or_bbs.items():
                labels.append(0)
                scores.append(float(v["bbox"]["object_det"]["confidence"]))
                bbox = [
                    v["bbox"]["x"],
                    v["bbox"]["y"],
                    v["bbox"]["width"],
                    v["bbox"]["height"],
                ]
                # x, y, w, h = scale_bbox_xywh(
                #     bbox,
                #     self.resize_w,
                #     self.resize_h,
                #     targetW=target_width,
                #     targetH=target_height,
                # )
                # x2 = min(x + w, target_width - 1)
                # y2 = min(y + h, target_height - 1)

                x, y, x2, y2 = scale_bbox(
                    bbox,
                    self.resize_w,
                    self.resize_h,
                    targetW=target_width,
                    targetH=target_height,
                    in_format="xywh",
                    out_format="xyxy",
                )

                bbs.append([x, y, x2, y2])
        else:
            for bbox in metadata_or_bbs:
                labels.append(0)
                scores.append(0.9)
                x, y, x2, y2 = scale_bbox(
                    bbox,
                    self.resize_w,
                    self.resize_h,
                    targetW=target_width,
                    targetH=target_height,
                    in_format="xyxy",
                    out_format="xyxy",
                )
                bbs.append([x, y, x2, y2])

        # Convert to array, then explicitly reshape to enforce the 2D constraint
        pred_boxes_array = np.array(bbs, dtype=np.float32).reshape(-1, 4)
        pred_scores_array = np.array(scores, dtype=np.float32)
        # pred_labels_array = np.array(labels, dtype=np.int64)
        # preds = {
        #     "boxes": pred_boxes_array,  # Guaranteed to be (N, 4) even if N=0
        #     "scores": pred_scores_array,
        #     "labels": pred_labels_array,
        # }
        # ]
        # self.all_preds.append(preds)
        # self.all_targets.append(targets)
        # metric_engine.update(preds, targets)
        self.evaluator.update_frame(
            pred_boxes_array, pred_scores_array, target_boxes_array
        )

        del pred_boxes_array, pred_scores_array

    # COMPARE SNAPSHOTS  --------------------------------------------
    def capture_state_snapshot(self):
        """Captures a metadata-only registry mapping of all active keys and values."""
        snapshot = {}
        for attr, val in list(self.__dict__.items()):
            if val is None:
                type_str = "NoneType"
                details = None
            else:
                type_str = val.__class__.__name__

                # Extract detailed allocation size/state contexts depending on type
                if type_str == "Tensor":
                    details = f"shape={list(val.shape)}, device={val.device}, dtype={val.dtype}"
                elif type_str == "ndarray":
                    details = f"shape={list(val.shape)}, dtype={val.dtype}"
                elif type_str in ("list", "dict", "set", "tuple"):
                    try:
                        details = f"len={len(val)}"
                    except Exception:
                        details = "uncountable"
                elif type_str in ("Thread", "Process", "DummyProcess"):
                    try:
                        details = f"alive={val.is_alive()}"
                    except Exception:
                        details = "unknown"
                elif type_str in ("Lock", "_RLock"):
                    try:
                        details = f"locked={val.locked()}"
                    except Exception:
                        details = "unknown"
                elif type_str == "Queue":
                    try:
                        details = f"approx_qsize={val.qsize()}"
                    except Exception:
                        details = "uncountable"
                else:
                    # Capture values for strings, ints, bools, etc.
                    try:
                        details = (
                            str(val)[:50]
                            if type_str in ("str", "int", "bool", "float")
                            else hex(id(val))
                        )
                    except Exception:
                        details = "unresolved_pointer"

            snapshot[attr] = {"type": type_str, "details": details}
        return snapshot

    def print_lifecycle_delta(
        self,
        before_snapshot,
        after_snapshot,
        keys_to_skip=default_attr_keys,
        return_keys=False,
    ):
        """Compares snapshots and outputs a precise list of elements that must be evicted."""
        main_app_logger.info("=" * 80)
        main_app_logger.info(
            " RESOURCE LIFECYCLE LIFESPAN ANALYSIS (BEFORE START vs AFTER STOP)"
        )
        main_app_logger.info("=" * 80)

        # 1. New keys initialized during execution
        new_keys = [
            key
            for key in sorted(
                list(set(after_snapshot.keys()) - set(before_snapshot.keys()))
            )
            if key not in keys_to_skip
        ]

        main_app_logger.info(
            f"[+] NEW ARTIFACTS GENERATED DURING THE RUN ({len(new_keys)} keys found):"
        )
        if new_keys:
            for key in new_keys:
                main_app_logger.info(
                    f"  └─ {key} -> Type: {after_snapshot[key]['type']} ({after_snapshot[key]['details']})"
                )
                main_app_logger.info(
                    "     ⚠️ ACTION REQUIRED: Must be deleted or unlinked via delattr()."
                )
        else:
            main_app_logger.info(
                "  None! (No new top-level attributes were registered)."
            )

        # 2. Key values modified, grown, or mutated during execution
        mutated_keys = []
        static_keys = []
        common_keys = set(before_snapshot.keys()) & set(after_snapshot.keys())
        for key in sorted(list(common_keys)):
            if (key not in keys_to_skip) and (
                before_snapshot[key] != after_snapshot[key]
            ):
                if after_snapshot[key]["details"] not in [None, "len=0"]:
                    mutated_keys.append(key)
            elif (
                (key not in keys_to_skip)
                and (before_snapshot[key] == after_snapshot[key])
                and getattr(self, key).__class__.__name__ != "method"
            ):
                static_keys.append(key)

        main_app_logger.info(
            f"[Δ] RETAINED KEYS MUTATED OR GROWN DURING THE RUN ({len(mutated_keys)} keys found):"
        )
        if mutated_keys:
            for key in mutated_keys:
                b_meta = before_snapshot[key]
                a_meta = after_snapshot[key]
                main_app_logger.info(f"  └─ {key}")
                main_app_logger.info(
                    f"     ├── Before Start: {b_meta['type']} ({b_meta['details']})"
                )
                main_app_logger.info(
                    f"     └── After Stop:  {a_meta['type']} ({a_meta['details']})"
                )
                main_app_logger.info(
                    "     ⚠️ ACTION REQUIRED: Revert back to baseline state, call .clear(), or set to None."
                )
        else:
            main_app_logger.info(
                "  None! (All original tracking keys remained perfectly static)."
            )

        # main_app_logger.info(f"\n[Δ] RETAINED STATIC KEYS DURING THE RUN ({len(static_keys)} keys found):")
        # if static_keys:
        #     for key in static_keys:
        #         b_meta = before_snapshot[key]
        #         main_app_logger.info(f"  └─ {key}")
        #         main_app_logger.info(f"     ├── Before Start: {b_meta['type']} ({b_meta['details']})")
        #         main_app_logger.info(f"     ⚠️ ACTION REQUIRED: Possibly remove because it has not changed.")
        main_app_logger.info("=" * 80)

        if return_keys:
            # return new_keys, static_keys
            return new_keys, mutated_keys, static_keys

    def stop_blueprint_executor(self, new_keys, mutated_keys):
        """Dynamically creates a blueprint of all changes and liquidates them safely."""
        main_app_logger.info("=" * 80)
        main_app_logger.info(" DYNAMIC TEARDOWN COMPILER & RESOURCE EVICTION ENGINE")
        main_app_logger.info(
            "=" * 80,
        )

        # keys_to_skip = [
        #     "_is_stopped",
        #     "active",
        #     "status",
        #     "baseline_before_start",
        # ]

        # 1. Identify dynamically added attributes (Must be destroyed / unlinked)
        new_keys = set(
            new_keys
        )  # set(state_before_stop.keys()) - set(before_snapshot.keys())

        # 2. Identify baseline attributes that grew or mutated (Must be reset)
        mutated_keys = set(mutated_keys)
        # mutated_keys = set()
        # for key in (set(before_snapshot.keys()) & set(state_before_stop.keys())):
        #     if before_snapshot[key] != state_before_stop[key] and key not in keys_to_skip:
        #         mutated_keys.append(key) if isinstance(mutated_keys, list) else mutated_keys.add(key)

        # --- PHASE 1: TARGETED HARDWARE DEALLOCATIONS (ORDER SENSITIVE) ---
        # First, handle background threads and OS processes before severing structural arrays
        all_targeted_keys = (
            new_keys | mutated_keys
        )  # [key for key in list(new_keys | mutated_keys)]  # if key not in keys_to_skip]

        # A. Prioritize process and thread termination hooks
        for key in list(all_targeted_keys):
            val = getattr(self, key, None)
            if val is None:
                continue
            type_name = val.__class__.__name__

            if type_name in ("Thread", "Process", "DummyProcess"):
                main_app_logger.info(
                    f" [RECONCILING] Terminating unmanaged runtime worker: {key}"
                )
                # try:
                #     if hasattr(val, "is_alive") and val.is_alive():
                #         if hasattr(val, "terminate"): val.terminate()
                #         val.join(timeout=0.5)
                # except Exception: pass
                self.stop_thread(val)

            elif type_name == "ThreadPoolExecutor":
                main_app_logger.info(
                    f" [RECONCILING] Shutting down concurrent thread pool: {key}"
                )
                try:
                    val.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    pass

        # B. Decouple IPC Multi-processing Queues next (Non-blocking)
        for key in list(all_targeted_keys):
            val = getattr(self, key, None)
            if val is None:
                continue
            if val.__class__.__name__ == "Queue":
                main_app_logger.info(
                    f" [RECONCILING] Draining and closing IPC channel: {key}"
                )
                try:
                    while not val.empty():
                        val.get_nowait()
                except Exception:
                    pass
                try:
                    val.close()
                    val.cancel_join_thread()
                except Exception:
                    pass

        # C. Unmap Shared Memory nodes from /dev/shm
        # for key in list(all_targeted_keys):
        #     val = getattr(self, key, None)
        #     if val is None: continue
        #     type_name = val.__class__.__name__
        #     if "shm" in key.lower() or type_name in ("SharedMemory", "SharedMemoryManager"):
        #         main_app_logger.info(f" [RECONCILING] Unlinking OS Shared Memory layout node: {key}")
        #         if type_name == "list":
        #             for item in val:
        #                 if item.__class__.__name__ == "tuple":
        #                     for i in item:
        #                         if i.__class__.__name__ == "Event":
        #                             try:
        #                                 if hasattr(i, "_handle"):
        #                                     i._handle.close()
        #                             except Exception:
        #                                 pass
        #                             i = None
        #                         else:
        #                             i = None

        #                     try:
        #                         if hasattr(item, "close"):
        #                             item.close()
        #                             item.unlink()
        #                         else:
        #                             item = None
        #                     except Exception:
        #                         # pass
        #                         traceback.print_exc()
        #                 elif item is not None:
        #                     try:
        #                         item.close()
        #                         item.unlink()
        #                     except Exception:
        #                         # pass
        #                         traceback.print_exc()
        #         else:
        #             try:
        #                 val.close()
        #                 val.unlink()
        #             except Exception:
        #                 # pass
        #                 traceback.print_exc()

        # --- PHASE 2: FLUSHING AND UNLINKING OBJECT REFERENCES ---
        # Now that the background hardware loops are dead, scrub the attributes
        for key in sorted(list(all_targeted_keys)):
            val = getattr(self, key, None)
            if val is None:
                continue
            type_name = val.__class__.__name__

            # If it's a new attribute, wipe its internal space and erase it completely
            if key in new_keys:
                if type_name == "Tensor":
                    try:
                        val.data = torch.empty(0, device=val.device)
                    except Exception:
                        pass
                elif isinstance(val, dict):
                    val.clear()
                elif isinstance(val, list):
                    val.clear()
                elif type_name == "GpuMat":
                    try:
                        val.release()
                    except Exception:
                        pass

                # Permanently strip the attribute from the object namespace
                try:
                    delattr(self, key)
                except AttributeError:
                    pass

            # If it was a mutated pre-existing baseline key, reset it to its origin baseline
            elif key in mutated_keys:
                if isinstance(val, (list, dict, set)):
                    try:
                        val.clear()
                    except Exception:
                        pass
                # else:
                #     # Revert primitive counters or strings back to their exact original baseline state
                #     orig_type = before_snapshot[key]["type"]
                elif isinstance(val, None):  # orig_type == "NoneType":
                    setattr(self, key, None)
                elif isinstance(val, int):  # orig_type == "int":
                    setattr(self, key, 0)
                elif isinstance(val, bool):  # orig_type == "bool":
                    setattr(self, key, False)
                elif isinstance(val, str):  # orig_type == "str":
                    setattr(self, key, "")

        # --- PHASE 3: FINAL HOST HEAP FLUSH ---
        # import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        main_app_logger.info("=" * 80)
        main_app_logger.info(
            " [SUCCESS] Dynamic Blueprint Execution Complete. Clean state achieved."
        )
        main_app_logger.info("=" * 80)

    # DEBUG FUNCTIONS --------------------------------------------
    def debug_save_mask(self, frame_source, frame_num, rois=None, gt_boxes=None):
        debug_dir = self.result_dir / "debug_stages" / self._testMethodName / "mask"
        debug_dir.mkdir(parents=True, exist_ok=True)

        save_path = (
            debug_dir / f"frame_{frame_num:04d}_mask.jpg"
        )  # f"mask_{frame_num:04d}.jpg"

        # cv2.imwrite(
        #     # str(stage_debug_dir / f"frame_{f_num:04d}_stage6_threshold.jpg"),
        #     str(stage_debug_dir / f"frame_{self.frame_count_target:04d}_stagerois_merged.jpg"),
        #     display_frame,
        # )

        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        # #  Download or copy the data
        # if hasattr(frame_source, "download"):
        #     img_cpu = frame_source.download()
        # elif torch.is_tensor(frame_source):
        #     # .contiguous() fixes the horizontal "shredding"/static look
        #     temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
        #     img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
        # else:
        #     # For numpy arrays (like your pinned memory), ensure memory is linear
        #     img_cpu = np.ascontiguousarray(frame_source)

        # #  Fix Visibility (Normalization)
        # # If float, scale to 0-255. If uint8, leave as is to avoid "neon" colors.
        # if img_cpu.dtype != np.uint8:
        #     if img_cpu.max() <= 1.0:
        #         img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
        #     else:
        #         img_cpu = img_cpu.astype(np.uint8)

        #  Download/Copy the frame
        if torch.is_tensor(frame_source):
            # .contiguous() is CRITICAL here to fix the "shredded" look
            temp = frame_source.squeeze(0) if frame_source.ndim == 4 else frame_source
            # img_cpu = temp.permute(1, 2, 0).contiguous().cpu().numpy()
            img_cpu = temp.contiguous().cpu().numpy()
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
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

        h_img, w_img = img_cpu.shape[:2]
        scale_x = float(w_img) / self.frame_width
        scale_y = float(h_img) / self.frame_height
        if gt_boxes is not None:
            # Factor for converting 8K (original) bb dimensions to display dimensions
            # disp_w, disp_h = inf_data["mask"].shape[:2] if hasattr(inf_data["mask"], "shape") else inf_data["mask"].size()
            # scale_display_ox = disp_w / self.frame_width
            # scale_display_oy = disp_h / self.frame_height

            img_cpu = get_bb_overlay(
                img_cpu,
                gt_boxes,
                (scale_x, scale_y),
                (w_img, h_img),
                color=(0, 255, 0),  # green
            )

        # Draw 8K Boxes (Scaled down)
        if rois is not None and len(rois) > 0:
            boxes = rois.cpu().tolist() if torch.is_tensor(rois) else rois
            for box in boxes:
                x1, y1, x2, y2 = [
                    # int(box[0] * scale_x),
                    # int(box[1] * scale_y),
                    # int(box[2] * scale_x),
                    # int(box[3] * scale_y),
                    max(0, int(box[0] * scale_x)),
                    max(0, int(box[1] * scale_y)),
                    min(w_img - 1, int(box[2] * scale_x)),
                    min(h_img - 1, int(box[3] * scale_y)),
                ]
                cv2.rectangle(img_cpu, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save to disk
        cv2.imwrite(str(save_path), img_cpu)

    def debug_save_img_roi(self, frame_source, bbs_full_res, frame_num, gt_boxes=None):
        debug_dir = self.result_dir / "debug_stages" / self._testMethodName / "img_roi"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # out_filename = str(debug_dir / f"analysis_{frame_num:04d}.jpg")
        save_path = debug_dir / f"frame_{frame_num:04d}_analysis.jpg"

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

        if gt_boxes is not None:
            # Factor for converting 8K (original) bb dimensions to display dimensions
            # disp_w, disp_h = inf_data["mask"].shape[:2] if hasattr(inf_data["mask"], "shape") else inf_data["mask"].size()
            # scale_display_ox = disp_w / self.frame_width
            # scale_display_oy = disp_h / self.frame_height

            img_cpu = get_bb_overlay(
                img_cpu,
                gt_boxes,
                (scale_x, scale_y),
                (w_img, h_img),
                color=(0, 255, 0),  # green
            )

        if bbs_full_res is not None:
            boxes = (
                bbs_full_res.cpu().tolist()
                if torch.is_tensor(bbs_full_res)
                else bbs_full_res
            )
            # Annotate Box to image shape
            for box in boxes:
                x1, y1, x2, y2 = [
                    max(0, int(box[0] * scale_x)),
                    max(0, int(box[1] * scale_y)),
                    min(w_img - 1, int(box[2] * scale_x)),
                    min(h_img - 1, int(box[3] * scale_y)),
                ]
                img_cpu = cv2.rectangle(img_cpu, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(save_path), img_cpu)

    def debug_save_crops(self, cropped_batch, frame_num):
        """Saves the first 5 crops of a batch to the results directory."""
        debug_dir = self.result_dir / "debug_stages" / self._testMethodName / "crops"
        # debug_dir = self.result_dir / "debug_stages" / self._testMethodName
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Only save for the first self.config.DEBUG_FRAME_LIMIT frames to avoid disk bloat
        if frame_num > self.config.DEBUG_FRAME_LIMIT:
            return

        for i, crop in enumerate(cropped_batch[: self.config.DEBUG_FRAME_LIMIT]):
            # Convert GPU Tensor [C, H, W] -> NumPy [H, W, C]
            if torch.is_tensor(crop):
                # Reverse normalization (* 255) and permute to BGR
                # img = (crop.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
                temp = crop.squeeze(0) if crop.ndim == 4 else crop
                # Flip the color channels from RGB to BGR before any extraction utilities run
                # Flips only the first axis (channels) from [0, 1, 2] to [2, 1, 0]
                temp = torch.flip(temp, dims=[0])
                img = temp.permute(1, 2, 0).contiguous().cpu().numpy()
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = crop

            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                cv2.imwrite(str(debug_dir / f"frame_{frame_num}_crop_{i}.jpg"), img)

    def debug_save_img(self, frame_source, frame_num):
        debug_dir = self.result_dir / "debug_stages" / self._testMethodName / "img"
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
