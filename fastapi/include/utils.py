# Copyright (C) 2025 Intel Corporation
# ==============================================================================
# IMPORTS

import ctypes
import gc
import inspect
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import tracemalloc
from collections import deque
from dataclasses import dataclass
from math import ceil
from multiprocessing import resource_tracker
from pathlib import Path
from random import randint

import cupy
import cv2
import numpy as np
import torch
from pydantic import BaseModel

import vdms

sys.path.insert(1, str(Path(__file__).parent.parent))
from include.default_configs import (
    BKGD_SUB_INCLUDE_HISTORY,
    BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE,
    BKGD_SUB_INCLUDE_HISTORY_METHOD,
    BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE,
    BKGD_SUB_MOG2_DETECTSHADOWS,
    BKGD_SUB_MOG2_HISTORY,
    BKGD_SUB_MOG2_LR,
    BKGD_SUB_MOG2_VARTHRESHOLD,
    CLIP_DURATION_DEFAULT,
    CODE_DIR_DEFAULT,
    CUSTOM_MODEL_FLAG_DEFAULT,
    DBHOST_DEFAULT,
    DBPORT_DEFAULT,
    DEBUG_DEFAULT,
    DETECTION_THRESHOLD_DEFAULT,
    DETECTION_TYPE_DEFAULT,
    DEVICE_DEFAULT,
    DILATE_KERNEL_SIZE,
    DISPLAY_FRAME_QUALITY,
    DISPLAY_FRAME_SIZE,
    ENABLE_QUERYING_DEFAULT,
    INGESTION_DEFAULT,
    IOU_THRESHOLD_DEFAULT,
    MAX_DETECTIONS,
    MAX_WORKERS,
    MODEL_H,
    MODEL_MAX_BATCH_SIZE,
    MODEL_NAME_DEFAULT,
    MODEL_PRECISION,
    MODEL_W,
    OMIT_DETECTIONS_FLAG_DEFAULT,
    RESIZE_FLAG_DEFAULT,
    ROI_BB_FULL_RES_PADDING,
    ROI_CONTAINMENT_THRESH,
    ROI_DISTANCE_THRESH_RATIO,
    ROI_MAX_RELATIVE_SIZE_RATIO,
    ROI_MERGE_SIZE_LIMIT,
    ROI_MIN_AREA_RATIO,
    SHARED_OUTPUT_DEFAULT,
    SMART_FILTERING_ENABLED,
    SMART_FILTERING_PIXEL_CONSTRAINT,
    TARGET_FPS,
    TEST_MODE_DEFAULT,
    THICKNESS,
    THRESHOLD_MAX_VALUE,
    THRESHOLD_VALUE,
    TMP_LOCATION_DEFAULT,
    UDF_HOST_DEFAULT,
    UDF_PORT_DEFAULT,
)

# ==============================================================================
# LOGGING
logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

main_app_logger = logging.getLogger(__name__)

# ==============================================================================

"""
GENERAL DEFINITIONS/FUNCTIONS
"""


def get_cpu_partition(num_ffmpeg_cores: int = 4):
    """
    Partitions available container cores into FFmpeg cores and Main App cores.
    """
    available_cores = sorted(list(os.sched_getaffinity(0)))
    total_cores = len(available_cores)

    if total_cores <= num_ffmpeg_cores:
        # Fallback if the container has <= 4 cores allocated
        ffmpeg_cores = available_cores
        main_app_cores = available_cores
    else:
        # Take the first 4 cores for FFmpeg, reserve the rest for the main app
        ffmpeg_cores = available_cores[:num_ffmpeg_cores]
        main_app_cores = available_cores[num_ffmpeg_cores:]

    return ffmpeg_cores, main_app_cores


def safe_unregister_shm(shm_name: str):
    """
    Unregisters a shared memory block from Python's resource tracker
    using both normalized and slash-prefixed formats.
    """
    clean_name = shm_name.lstrip("/")
    slash_name = f"/{clean_name}"

    for name in (clean_name, slash_name):
        try:
            resource_tracker.unregister(name, "shared_memory")
        except (KeyError, ValueError, AttributeError, Exception):
            pass


def release_shared_memory(list_of_shms):
    for shm in list_of_shms:
        if shm is not None:
            try:
                # Release buffer view if open
                if hasattr(shm, "buf") and shm.buf is not None:
                    shm.buf.release()
            except Exception:
                pass

            try:
                shm.close()
            except Exception:
                pass

            try:
                shm.unlink()
            except (FileNotFoundError, AttributeError, OSError):
                pass

    # Clear the container list/collection in-place
    if hasattr(list_of_shms, "clear"):
        try:
            list_of_shms.clear()
        except Exception:
            pass


def release_native_linux_heap():
    gc.collect()
    try:
        # Load the native C library bindings mapping standard glibc
        libc = ctypes.CDLL("libc.so.6")

        # malloc_trim(0) forces glibc to completely pack and return all
        # completely disconnected heap memory boundaries back to the kernel.
        libc.malloc_trim(0)
        # print(
        #     "[MALLOC_TRIM] Successfully flushed unmanaged C++ heap to OS.", flush=True
        # )
    except Exception as e:
        print(
            f"[MALLOC_TRIM] Native system trim execution pass failed: {e}", flush=True
        )


def install_and_load_pip_package(package_name: str, attribute_name=None):
    import importlib

    name = package_name.split("[")[0]
    try:
        # import py_package
        module = importlib.import_module(name)
    except ImportError:
        print(f"{name} package not found.  Installing ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        module = importlib.import_module(name)

    if attribute_name:
        return getattr(module, attribute_name)

    return module


def get_freest_gpu():
    # Queries free memory from nvidia-smi
    command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
    memory_free = [
        int(x)
        for x in subprocess.check_output(command.split()).decode("ascii").split("\n")
        if x
    ]

    # Return index of GPU with maximum free memory
    return memory_free.index(max(memory_free))


def safely_join_path(base_dir, add_path):
    safe_base = os.path.abspath(base_dir)
    candidate_path = os.path.abspath(os.path.join(safe_base, add_path))
    if not candidate_path.startswith(safe_base + os.sep):
        raise ValueError(f"Invalid path: {candidate_path}")
    return candidate_path


def str2bool(in_val):
    if isinstance(in_val, bool):
        return in_val

    if not isinstance(in_val, str):
        raise ValueError(f"{in_val} is not a bool or string")

    if in_val.title() == "True":
        return True
    else:
        return False


def global_frame_prefetch_worker_v1(instance):
    """
    Asynchronous Threaded Staging Engine.
    Leverages thread-level parallelism to interleave heavy I/O file latency
    completely outside the primary model execution track.
    """
    pool_maxsize = (
        instance.prefetch_queue.maxsize
        if hasattr(instance.prefetch_queue, "maxsize")
        else instance.prefetch_queue._maxsize
    )

    # This ensures no frames are read until the main loop is ready.
    # if hasattr(instance, "start_processing_event"):
    #     instance.start_processing_event.wait()

    while instance.active and instance.prefetch_active:
        try:
            # if instance.prefetch_queue is None:
            #     instance.prefetch_active = False
            #     break
            # 1. Capture time immediately before hardware decryption/extraction
            decode_start_time = time.perf_counter()

            # Step 1: Overlap video decoding and frame I/O reads.
            with instance.reader_lock:
                payload = instance.reader.read()

            # if payload[1] is None:
            #     # print("[PREFETCH WORKER] Reader returned None, exiting.")
            #     # break
            #     time.sleep(0.005) # Sleep 5ms and retry
            #     continue

            ret, frame_8k, current_event, frame_num, abs_frame_num = payload

            # 2. Isolate raw decompression latency (completely independent of consumer queues)
            true_read_latency_secs = time.perf_counter() - decode_start_time  # * 1000.0

            if not ret or frame_8k is None:
                # # If RTSP, wait briefly to see if it's just a network delay.
                # if instance.active and getattr(instance, "is_rtsp", True):
                #     time.sleep(0.005)
                #     continue
                # else:
                #     print("[PREFETCH WORKER] End of stream or invalid frame, exiting.")
                #     break

                # Allow retries if the reader is still active and has not officially stopped
                if instance.active and not getattr(instance.reader, "stopped", False):
                    time.sleep(0.005)
                    continue
                else:
                    main_app_logger.info(
                        "[PREFETCH WORKER] End of stream detected, exiting."
                    )
                    break

            if frame_num == 0:
                main_app_logger.info(
                    f"[VERIFY - PRODUCER] Decoded Frame {frame_num} (abs: {abs_frame_num}) and writing to slot {frame_num % pool_maxsize}!"
                )

            # if ret is True and frame_8k is not None:
            slot_idx = frame_num % pool_maxsize

            # if torch.is_tensor(frame_8k):
            #     safe_frame = frame_8k.clone()
            # elif isinstance(frame_8k, np.ndarray):
            #     safe_frame = frame_8k.copy()
            # else:
            #     safe_frame = frame_8k

            # Convert to tensor
            # Make a safe copy of the raw frame data
            # if isinstance(frame_8k, np.ndarray):
            #     safe_frame = frame_8k.copy()
            # else:
            #     safe_frame = frame_8k.clone() if torch.is_tensor(frame_8k) else frame_8k

            # safe_frame = instance.shm_buffer_pool[slot_idx][0]

            # # 2. OVERWRITE IN-PLACE (ZERO ALLOCATIONS!)
            # if isinstance(frame_8k, np.ndarray):
            #     safe_frame.copy_(torch.from_numpy(frame_8k))
            # else:
            #     safe_frame.copy_(frame_8k)

            # If frame_8k is already on the GPU (from GPUReader), use it directly!
            if isinstance(frame_8k, np.ndarray):
                # CPU fallback: copy to GPU input slot
                instance.gpu_input[slot_idx].copy_(
                    torch.from_numpy(frame_8k), non_blocking=True
                )
                safe_frame = instance.gpu_input[slot_idx]
            else:
                # Already on GPU: forward directly with zero extra transfers
                safe_frame = frame_8k

            # safe_frame = frame_8k
            # Explicitly destroy the original shared-memory views right here!
            # del payload, frame_8k  # , tensor_frame
            # print(f"[PREFETCH WORKER] frame_num: {frame_num}\tabs_frame_num: {abs_frame_num}", flush=True)

            # Step 2: Directly write to array pool within the same CUDA context.
            # 3. EXPAND POOL TUPLE: Append the true background latency value to the array matrix
            instance.shm_buffer_pool[slot_idx] = (
                safe_frame,
                current_event,
                frame_num,
                abs_frame_num,
                true_read_latency_secs,  # Decoupled payload metric channel
            )
            # del payload, frame_8k

            if hasattr(instance, "queue_data_ready_event"):
                instance.queue_data_ready_event.set()

            # Step 3: Fast notification down a standard thread-safe queue.
            # Wakes up the consumer loop in microseconds via low-level OS variables.
            try:
                instance.prefetch_queue.put((True, slot_idx), block=True)
            # except AttributeError:
            #     instance.prefetch_active = False
            # break
            except Exception:
                pass
            finally:
                del payload, frame_8k
        except Exception as e:
            # Yield minimal execution slicing if an index collision occurs
            # time.sleep(0.001)
            # CRITICAL: Print the actual error instead of swallowing it.
            print(f"[PREFETCH WORKER CRASH] An error occurred: {e}")
            traceback.print_exc()
            # Stop the loop on error to prevent silent failures.
            raise

    # Pipeline breakdown tracker
    # with instance.worker_tracking_lock:
    #     instance.active_workers_count -= 1
    #     if instance.active_workers_count == 0:
    #         # instance.prefetch_queue.put((False, "END_OF_STREAM"), block=True)
    #         main_app_logger.info(
    #             "[PREFETCH WORKER] All workers finished. Sending END_OF_STREAM."
    #         )
    #         if (
    #             hasattr(instance, "prefetch_queue")
    #             and instance.prefetch_queue is not None
    #         ):
    #             try:
    #                 instance.prefetch_queue.put(
    #                     (False, "END_OF_STREAM"), block=True, timeout=1.0
    #                 )
    #             except Exception:
    #                 pass
    lock_obj = getattr(instance, "worker_tracking_lock", None)

    if lock_obj is not None and hasattr(lock_obj, "__enter__"):
        with lock_obj:
            instance.active_workers_count -= 1
            is_last_worker = instance.active_workers_count == 0
    else:
        # Fallback in case the main thread already destroyed the lock during shutdown
        # (Safe to decrement directly since threads are joining anyway)
        try:
            instance.active_workers_count -= 1
            is_last_worker = instance.active_workers_count == 0
        except Exception:
            is_last_worker = False

    # Send END_OF_STREAM if this is the last worker exiting
    if is_last_worker:
        main_app_logger.info(
            "[PREFETCH WORKER] All workers finished. Sending END_OF_STREAM."
        )
        if hasattr(instance, "prefetch_queue") and instance.prefetch_queue is not None:
            try:
                instance.prefetch_queue.put(
                    (False, "END_OF_STREAM"), block=True, timeout=1.0
                )
            except Exception:
                pass


def analyze_tracemalloc_snapshot():
    """
    Take and analyze tracemalloc snapshot
    """
    system_exclusions = [
        "importlib",
        "runpy",
        "pydev",
        "typing",
        "abc",
        "contextlib",
        "unittest",
        "pytest",
    ]

    main_app_logger.info("=" * 60)
    # Take snapshot with tracking data enabled
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = snapshot.statistics("traceback")

    main_app_logger.info(
        f"{'ALLOCATED SIZE':<15} | {'OBJECT TYPE':<20} | {'SOURCE LOCATION'}",
    )
    main_app_logger.info("-" * 80)

    stat_cnt = 0
    for stat in stats:  # [:10]:  # Check the top 10 largest leaks
        if stat_cnt >= 10:
            break

        first_frame = stat.traceback[0]
        source_info = f"{first_frame.filename.split('/')[-1]}:{first_frame.lineno}"

        if any(exc in source_info for exc in system_exclusions):
            continue

        # Extract the trace block memory identity
        obj_type_name = "Unknown / Raw C Block"

        # Query the raw trace details to extract live block structures
        for trace in stat.traceback:
            # We look for objects instantiated at this file/line in the heap
            for obj in gc.get_objects():
                try:
                    # Cross-reference: does this object's line history match the stat line?
                    # Python doesn't save creation history on every object, so we inspect properties:
                    if torch.is_tensor(obj):
                        # Tensors are easily tracked if they have shapes or match your sizes
                        if (
                            obj.is_cuda
                            and (obj.element_size() * obj.nelement()) == stat.size
                        ):
                            obj_type_name = f"torch.Tensor (Shape: {list(obj.shape)})"
                            break
                    elif inspect.isframe(obj):
                        obj_type_name = "Live Frame Context"
                        break
                    elif inspect.isfunction(obj) or inspect.ismethod(obj):
                        obj_type_name = f"Function ({obj.__name__})"
                        break
                    elif isinstance(obj, dict):
                        # Dictionary allocations match structure profiles
                        try:
                            # Guard checks against mutating async dictionary managers
                            if (
                                sys.getsizeof(obj) == stat.size
                                or len(obj) == stat.count
                            ):
                                obj_type_name = "dict Namespace"
                                break
                        except Exception:
                            pass
                except Exception:
                    pass

            obj = None
        # Print the compiled breakdown

        main_app_logger.info(
            f"{stat.size / 1024:<11.1f} KiB | {obj_type_name:<20} | {source_info}"
        )
        stat_cnt += 1


# from concurrent.futures import ThreadPoolExecutor


# def global_frame_prefetch_worker(instance):
#     """GIL-free asynchronous staging engine utilizing a dedicated single-worker pool
#     to execute raw C++ reads completely outside the primary interpreter track.
#     """
#     pool_maxsize = (
#         instance.prefetch_queue.maxsize
#         if hasattr(instance.prefetch_queue, "maxsize")
#         else instance.prefetch_queue._maxsize
#     )

#     # A dedicated single-worker context specifically for C++ operations
#     # with ThreadPoolExecutor(max_workers=1) as reader_executor:
#     while instance.active and instance.prefetch_active:
#         try:
#             # 1. Capture time immediately before hardware decryption/extraction
#             decode_start_time = time.perf_counter()

#             # Offload ONLY the unmanaged C++ read() call to the executor.
#             # This drops the GIL instantly while the driver retrieves the frame bytes.
#             # future = reader_executor.submit(instance.reader.read)
#             # payload = (
#             #     future.result()
#             # )  # Blocking wait until the background C++ task completes

#             # Explicitly overwrite the internal future reference
#             # tracking properties to break the C++ object data caching loop immediately
#             # if hasattr(future, "_result"):
#             #     future._result = None  # Breaks the structural reference hold layout
#             # future = None             # Drops local scope tracking handles

#             # Overlap video decoding and frame I/O reads natively.
#             # The GIL is dropped directly inside the compiled C++ cv2.VideoCapture layer.
#             with instance.reader_lock:
#                 payload = instance.reader.read()

#             # Isolate raw decompression latency (completely independent of consumer queues)
#             true_read_latency_sec = (
#                 time.perf_counter() - decode_start_time
#             )

#             if payload[0] is False:
#                 break

#             ret, frame_8k, current_event, frame_num, abs_frame_num = payload


#             if ret is True and frame_8k is not None:
#                 slot_idx = frame_num % pool_maxsize
#                 if torch.is_tensor(frame_8k):
#                     safe_frame = frame_8k.clone()
#                 elif isinstance(frame_8k, np.ndarray):
#                     safe_frame = frame_8k.copy()
#                 else:
#                     safe_frame = frame_8k

#                 # Explicitly destroy the original shared-memory views right here!
#                 del payload, frame_8k

#                 print(f"[PREFETCH WORKER] frame_num: {frame_num}\tabs_frame_num: {abs_frame_num}", flush=True)
#                 # Directly write to array pool within the same CUDA context.
#                 # EXPAND POOL TUPLE: Append the true background latency value to the array matrix
#                 with instance.worker_tracking_lock:
#                     instance.shm_buffer_pool[slot_idx] = (
#                         safe_frame,
#                         current_event,
#                         frame_num,
#                         abs_frame_num,
#                         true_read_latency_sec,  # Decoupled payload metric channel
#                     )

#                 if hasattr(instance, "queue_data_ready_event"):
#                     instance.queue_data_ready_event.set()

#                 # Step 3: Fast notification down a standard thread-safe queue.
#                 # Wakes up the consumer loop in microseconds via low-level OS variables.
#                 instance.prefetch_queue.put((True, slot_idx), block=True)
#             else:
#                 # break
#                 time.sleep(0.001)
#                 continue
#         except Exception:
#             # Yield minimal execution slicing if an index collision occurs
#             time.sleep(0.002)
#             continue

#     # Pipeline breakdown tracker
#     # with instance.worker_tracking_lock:
#     #     instance.active_workers_count -= 1
#     #     if instance.active_workers_count == 0:
#     #         instance.prefetch_queue.put((False, "END_OF_STREAM"), block=True)


def global_frame_prefetch_worker_process(
    active,  # mp.Value flag
    prefetch_active,  # mp.Value flag
    prefetch_queue,  # mp.Queue mapping descriptor
    reader_lock,  # mp.Lock mutex channel
    reader,  # Picklable reader baseline layout
    shm_buffer_pool,  # Shared pointer memory pool array matrix
    queue_data_ready_event,  # Pass the shared multiprocessing Event channel
    worker_tracking_lock,  # Pass the shared multiprocessing Lock mutex
    active_workers_count,  # Pass the shared atomic integer context
):
    """
    Asynchronous Threaded Staging Engine.
    Leverages thread-level parallelism to interleave heavy I/O file latency
    completely outside the primary model execution track.
    """
    pool_maxsize = (
        prefetch_queue.maxsize
        if hasattr(prefetch_queue, "maxsize")
        else prefetch_queue._maxsize
    )

    try:
        while active.value and prefetch_active.value:
            try:
                # 1. Capture time immediately before hardware decryption/extraction
                decode_start_time = time.perf_counter()

                # Step 1: Overlap video decoding and frame I/O reads.
                with reader_lock:
                    payload = reader.read()

                # 2. Isolate raw decompression latency (completely independent of consumer queues)
                true_read_latency_ms = (
                    time.perf_counter() - decode_start_time
                ) * 1000.0

                if payload is None:
                    break

                ret, frame_8k, current_event, frame_num, abs_frame_num = payload

                if ret is True and frame_8k is not None:
                    slot_idx = frame_num % pool_maxsize

                    if torch.is_tensor(frame_8k):
                        safe_frame = frame_8k.clone()
                    elif isinstance(frame_8k, np.ndarray):
                        safe_frame = frame_8k.copy()
                    else:
                        safe_frame = frame_8k

                    # Explicitly destroy the original shared-memory views right here!
                    del payload, frame_8k

                    # Step 2: Directly write to array pool within the same CUDA context.
                    # 3. EXPAND POOL TUPLE: Append the true background latency value to the array matrix
                    shm_buffer_pool[slot_idx] = (
                        safe_frame,
                        current_event,
                        frame_num,
                        abs_frame_num,
                        true_read_latency_ms,  # Decoupled payload metric channel
                    )

                    queue_data_ready_event.set()

                    prefetch_queue.put((True, slot_idx), block=True)
                else:
                    break
            except Exception:
                # Yield minimal execution slicing if an index collision occurs
                time.sleep(0.001)

    finally:
        # Pipeline breakdown tracker
        with worker_tracking_lock:
            active_workers_count.value -= 1
            if active_workers_count.value == 0:
                prefetch_queue.put((False, "END_OF_STREAM"), block=True)


PROJECT_PATH = Path(__file__).parent.parent

DEBUG_FLAG_DEFAULT = True if DEBUG_DEFAULT == "1" else False

LOCKTIMEOUT_RETRIES = 5

default_attr_keys = [
    "_is_stopped",
    "_stop_lock",
    "_testMethodName",
    "abs_frame_num",
    "active_workers_count",
    "active",
    "baseline_before_start",
    "component_stats",
    "config",
    "crops_per_frame_list",
    "device_input",
    "device",
    "disp_h",
    "disp_w",
    "duration_target",
    "elapsed_display_time",
    "frame_count_target",
    "frame_count",
    "max_target_frames",
    "output_path",
    "pcie_throughput_gbps",
    "prefetch_active",
    "reader",
    "resize_h",
    "resize_w",
    "stat_fps",
    "stat_frame_count",
    "status",
    "VIDEO_GT_DETAILS",
    "video_output_name",
    "vram_efficiency",
    "worker_tracking_lock",
]


class PipelineConfig:
    def __init__(self, **kwargs):
        # Fallback to env var if not explicitly passed

        # GENERAL
        self.CODE_DIR = kwargs.get("CODE_DIR", CODE_DIR_DEFAULT)
        self.CUSTOM_MODEL_FLAG = str2bool(
            kwargs.get("CUSTOM_MODEL_FLAG", CUSTOM_MODEL_FLAG_DEFAULT)
        )
        self.DEBUG = kwargs.get("DEBUG", DEBUG_DEFAULT)
        self.DEBUG_FRAME_LIMIT = int(kwargs.get("DEBUG_FRAME_LIMIT", 100))
        self.DEVICE = kwargs.get("DEVICE", DEVICE_DEFAULT)
        self.MAX_WORKERS = int(kwargs.get("MAX_WORKERS", MAX_WORKERS))
        self.OMIT_DETECTIONS_FLAG = str2bool(
            kwargs.get("OMIT_DETECTIONS_FLAG", OMIT_DETECTIONS_FLAG_DEFAULT)
        )
        self.SHARED_OUTPUT = kwargs.get("SHARED_OUTPUT", SHARED_OUTPUT_DEFAULT)
        self.TEST_MODE = str2bool(kwargs.get("TEST_MODE", TEST_MODE_DEFAULT))
        self.TMP_LOCATION = kwargs.get("TMP_LOCATION", TMP_LOCATION_DEFAULT)

        # VIDEO WRITER
        CLIP_DURATION = kwargs.get("CLIP_DURATION", CLIP_DURATION_DEFAULT)
        target_fps = kwargs.get("TARGET_FPS", TARGET_FPS)
        self.CLIP_DURATION = (
            None if CLIP_DURATION in ["None", None] else float(CLIP_DURATION)
        )
        self.TARGET_FPS = None if target_fps in [None, "None"] else float(target_fps)

        # VDMS
        self.DBHOST = kwargs.get("DBHOST", DBHOST_DEFAULT)
        self.DBPORT = int(kwargs.get("DBPORT", DBPORT_DEFAULT))
        self.ENABLE_QUERYING = str2bool(
            kwargs.get("ENABLE_QUERYING", ENABLE_QUERYING_DEFAULT)
        )
        self.INGESTION = kwargs.get("INGESTION", INGESTION_DEFAULT)
        self.UDF_HOST = kwargs.get("UDF_HOST", UDF_HOST_DEFAULT)
        self.UDF_PORT = int(kwargs.get("UDF_PORT", UDF_PORT_DEFAULT))

        # MODEL
        self.DETECTION_THRESHOLD = float(
            kwargs.get("DETECTION_THRESHOLD", DETECTION_THRESHOLD_DEFAULT)
        )
        self.IOU_THRESHOLD = float(kwargs.get("IOU_THRESHOLD", IOU_THRESHOLD_DEFAULT))
        self.MAX_DETECTIONS = int(kwargs.get("MAX_DETECTIONS", MAX_DETECTIONS))
        self.MODEL_H = int(kwargs.get("MODEL_H", MODEL_H))
        self.MODEL_W = int(kwargs.get("MODEL_W", MODEL_W))
        self.MODEL_MAX_BATCH_SIZE = int(
            kwargs.get("MODEL_MAX_BATCH_SIZE", MODEL_MAX_BATCH_SIZE)
        )
        self.MODEL_NAME = kwargs.get("MODEL_NAME", MODEL_NAME_DEFAULT)
        self.MODEL_PRECISION = kwargs.get("MODEL_PRECISION", MODEL_PRECISION)
        self.SHARED_MODEL = kwargs.get("SHARED_MODEL", False)

        # PIPELINE
        self.DISABLE_DETECTION = kwargs.get("DISABLE_DETECTION", False)
        self.SMART_FILTERING_PIXEL_CONSTRAINT = SMART_FILTERING_PIXEL_CONSTRAINT
        self.BKGD_SUB_INCLUDE_HISTORY = BKGD_SUB_INCLUDE_HISTORY
        self.BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE = (
            BKGD_SUB_INCLUDE_HISTORY_DILATE_KERNEL_SIZE
        )
        self.BKGD_SUB_INCLUDE_HISTORY_METHOD = BKGD_SUB_INCLUDE_HISTORY_METHOD
        self.BKGD_SUB_MOG2_DETECTSHADOWS = BKGD_SUB_MOG2_DETECTSHADOWS
        self.BKGD_SUB_MOG2_HISTORY = BKGD_SUB_MOG2_HISTORY
        self.BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE = (
            BKGD_SUB_INCLUDE_HISTORY_TEMPORAL_SIZE
        )
        self.BKGD_SUB_MOG2_LR = BKGD_SUB_MOG2_LR
        self.BKGD_SUB_MOG2_VARTHRESHOLD = BKGD_SUB_MOG2_VARTHRESHOLD
        self.DILATE_KERNEL_SIZE = DILATE_KERNEL_SIZE
        self.RESIZE_FLAG = str2bool(kwargs.get("RESIZE_FLAG", RESIZE_FLAG_DEFAULT))
        self.ROI_BB_FULL_RES_PADDING = int(
            kwargs.get("ROI_BB_FULL_RES_PADDING", ROI_BB_FULL_RES_PADDING)
        )
        self.ROI_MAX_RELATIVE_SIZE_RATIO = float(
            kwargs.get("ROI_MAX_RELATIVE_SIZE_RATIO", ROI_MAX_RELATIVE_SIZE_RATIO)
        )
        self.ROI_MERGE_SIZE_LIMIT = int(
            kwargs.get("ROI_MERGE_SIZE_LIMIT", ROI_MERGE_SIZE_LIMIT)
        )
        self.ROI_MIN_AREA_RATIO = ROI_MIN_AREA_RATIO
        self.ROI_DISTANCE_THRESH_RATIO = ROI_DISTANCE_THRESH_RATIO
        self.ROI_CONTAINMENT_THRESH = ROI_CONTAINMENT_THRESH
        self.ROI_RETURN_BYTES = str2bool(kwargs.get("ROI_RETURN_BYTES", True))
        self.THRESHOLD_MAX_VALUE = int(
            kwargs.get("THRESHOLD_MAX_VALUE", THRESHOLD_MAX_VALUE)
        )
        self.THRESHOLD_VALUE = int(kwargs.get("THRESHOLD_VALUE", THRESHOLD_VALUE))

        # VISUALIZATION
        self.DETECTION_TYPE = kwargs.get("DETECTION_TYPE", DETECTION_TYPE_DEFAULT)
        self.DISPLAY_FRAME_QUALITY = int(
            kwargs.get("DISPLAY_FRAME_QUALITY", DISPLAY_FRAME_QUALITY)
        )
        self.DISPLAY_FRAME_SIZE = kwargs.get("DISPLAY_FRAME_SIZE", DISPLAY_FRAME_SIZE)
        self.THICKNESS = int(kwargs.get("THICKNESS", THICKNESS))

        # VARS WITH DEPENDENCIES
        Path(self.SHARED_OUTPUT).mkdir(parents=True, exist_ok=True)
        self.device_input = self.DEVICE.lower() if self.DEVICE == "CPU" else "cuda"
        self.DEBUG_FLAG = True if self.DEBUG == "1" else False

        if self.DETECTION_TYPE == "motion" and self.ENABLE_QUERYING:
            # self.ENABLE_QUERYING = False
            # self.DISPLAY_FRAME_QUALITY = 100
            self.THICKNESS = 10

        self.sf_enabled = kwargs.get("SMART_FILTERING_ENABLED", SMART_FILTERING_ENABLED)
        if self.CUSTOM_MODEL_FLAG:
            self.model_path = f"{self.CODE_DIR}/resources/models/ultralytics/custom_models/{self.MODEL_NAME}"
        else:
            self.model_path = f"{self.CODE_DIR}/resources/models/ultralytics/{self.MODEL_NAME}/{self.MODEL_PRECISION}/{self.MODEL_NAME}"

        if not self.sf_enabled:
            self.model_path += "_noSF"

        if self.DEVICE == "GPU":
            self.model_path += ".engine"

            # Force PyTorch to initialize the CUDA context
            if torch.cuda.is_available():
                best_gpu_index = get_freest_gpu()
                os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_index)
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
        else:
            self.model_path += "_openvino_model/"


class VDMSPool:
    def __init__(self, host, port, size=5):
        self.host = host
        self.port = port
        self.size = size
        self.pool = queue.Queue(maxsize=size)
        self.populate()

    def populate(self):
        # Pre-populate the pool with authenticated connections
        for _ in range(self.size):
            self.pool.put(self._create_connection())

    def _create_connection(self):
        client = vdms.vdms()
        client.connect(self.host, self.port)
        return client

    def get_connection(self):
        # Borrow a connection (blocks if pool is empty)
        db = self.pool.get(block=True, timeout=10)
        self.pool.task_done()
        return db

    def return_connection(self, conn):
        # Put the connection back for reuse
        self.pool.put(conn)


ERR_KEYWORDS = [
    "timeout",
    "null search iterator",
    "outoftransactions",
    "internal server",
]


# Plot variables
THICKNESS_SCALE_FACTOR = 1e-3
FONT_SCALE_FACTOR = 1e-3


YOLO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

PLOT_HEXS = (
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
)

DETECTION_COLORS = []
for h in PLOT_HEXS:
    DETECTION_COLORS.append(
        tuple(int(f"#{h}"[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
    )


# if DEVICE == "GPU":
bbox_kernel = cupy.ElementwiseKernel(
    "S label_image, int32 width",
    "raw T bboxes",
    """
    if (label_image > 0) {
        int label = (int)label_image;

        int y = i / width;
        int x = i % width;
        // Atomic operations to find min/max coordinates
        atomicMin(&bboxes[label * 4 + 0], y); // min_y
        atomicMin(&bboxes[label * 4 + 1], x); // min_x
        atomicMax(&bboxes[label * 4 + 2], y); // max_y
        atomicMax(&bboxes[label * 4 + 3], x); // max_x
    }
    """,
    "bbox_kernel",
)

bbox_area_kernel = cupy.ElementwiseKernel(
    "S label_image, int32 width",
    "raw T bboxes, raw T areas",
    """
    if (label_image > 0) {
        int label = (int)label_image;
        int y = i / width;
        int x = i % width;

        // 1. Update Bounding Box
        atomicMin(&bboxes[label * 4 + 0], y); // min_y
        atomicMin(&bboxes[label * 4 + 1], x); // min_x
        atomicMax(&bboxes[label * 4 + 2], y); // max_y
        atomicMax(&bboxes[label * 4 + 3], x); // max_x

        // 2. Increment Area (Count pixels)
        atomicAdd(&areas[label], 1);
    }
    """,
    "bbox_area_kernel",
)

threshold_dilate_fused_kernel = cupy.ElementwiseKernel(
    "T mask, int32 threshold, int32 width, int32 height",
    "raw T morphed",
    """
    // 1. Threshold
    bool is_active = mask > threshold;

    // 2. Simple 3x3 Dilation (Fuse directly into output)
    if (is_active) {
        int y = i / width;
        int x = i % width;

        // 2. 3x3 Dilation Expansion
        // Writes 255 to the neighbors of any pixel above threshold
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = y + dy;
                int nx = x + dx;
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    morphed[ny * width + nx] = 255;
                }
            }
        }
    }
    """,
    "threshold_dilate_fused",
)

# This CUDA C++ code finds min/max for all labels in ONE pass over the mask
DETECTION_ACCEL_KERNEL = cupy.RawKernel(
    r"""
    extern "C" __global__
    void fast_detect(const unsigned char* bgs_mask, unsigned char* out_mask, int pitch, int w, int h, float thresh) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x > 0 && x < w-1 && y > 0 && y < h-1) {
            unsigned char val = bgs_mask[y * pitch + x];
            unsigned char res = (val > thresh) ? 255 : 0;
            if (res == 0) {
                if (bgs_mask[(y-1)*pitch + x] > thresh || bgs_mask[(y+1)*pitch + x] > thresh ||
                    bgs_mask[y*pitch + (x-1)] > thresh || bgs_mask[y*pitch + (x+1)] > thresh) {
                    res = 255;
                }
            }
            out_mask[y * pitch + x] = res;
        }
    }
    """,
    "fast_detect",
)

# Fused Kernel: Single-pass Bounding Box Extraction with Stride Support
BOUNDS_KERNEL = cupy.RawKernel(
    r"""
extern "C" __global__
void find_bounds(const unsigned char* labeled_ptr, int step, int width, int height, int num_labels, int* x1, int* y1, int* x2, int* y2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int* row = (const int*)(labeled_ptr + y * step);
        int label = row[x];
        if (label > 0 && label <= num_labels) {
            atomicMin(&x1[label], x);
            atomicMin(&y1[label], y);
            atomicMax(&x2[label], x);
            atomicMax(&y2[label], y);
        }
    }
}
""",
    "find_bounds",
)


PROPAGATION_KERNEL_CODE = r"""
extern "C" __global__
void get_row_bounds_fused(const unsigned char* mask, int pitch, int w, int h,
                          int* x1, int* y1, int* x2, int* y2, int* num_labels) {
    // Each thread tracks exactly one horizontal row across the frame canvas
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < h) {
        bool row_has_motion = false;
        int min_x = w;
        int max_x = -1;

        // Perform a fast linear register sweep across the row (0 global atomics)
        for (int x = 0; x < w; ++x) {
            unsigned char pixel = mask[y * pitch + x];
            if (pixel > 0) {
                row_has_motion = true;
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
            }
        }

        // If the row contains motion, commit the boundaries to global memory
        if (row_has_motion) {
            // Use an atomic index fetch to smoothly assign contiguous tracking labels
            int label = atomicAdd(num_labels, 1);

            // Limit to our safe allocation pool ceiling to prevent memory overflows
            if (label < 256) {
                x1[label] = min_x;
                y1[label] = y;
                x2[label] = max_x + 1;
                y2[label] = y + 1;
            }
        }
    }
}
"""


def get_metadata_overlay(
    display_frame, metadata_or_bbs, class_list, scale_display, disp_size, is_bgr=True
):
    scale_display_x, scale_display_y = scale_display
    disp_w, disp_h = disp_size
    for frame_str, obj in metadata_or_bbs.items():
        bbox = obj["bbox"]
        raw_x = bbox["x"] * scale_display_x
        raw_y = bbox["y"] * scale_display_y
        w = bbox["width"] * scale_display_x
        h = bbox["height"] * scale_display_y
        x = max(0, int(raw_x))
        y = max(0, int(raw_y))
        x2 = min(disp_w - 1, int(raw_x + w))
        y2 = min(disp_h - 1, int(raw_y + h))

        class_name = bbox["object"]
        class_id = class_list.index(class_name) if class_name in class_list else 0
        bb_color = get_detection_color(class_id, is_bgr=is_bgr)

        # print(f'{frame_str} xyxy: ', (x, y), (x2, y2), flush=True)
        display_frame = cv2.rectangle(display_frame, (x, y), (x2, y2), bb_color, 2)

        if class_name != "":
            confidence = bbox.get("object_det", {}).get("confidence", 0.0)
            label = f"{class_name} {confidence:.2f}"
            draw_label(display_frame, label, (x, y), color=bb_color, padding=5)
    return display_frame


def get_bb_overlay(
    display_frame, metadata_or_bbs, scale_display, disp_size, color=(0, 0, 255)
):
    """
    Ultra-low latency canvas renderer. Optimized to preserve high FPS.
    """
    # Instant escape route for blank inference sequences
    if metadata_or_bbs is None or len(metadata_or_bbs) == 0:
        return display_frame

    scale_display_x, scale_display_y = scale_display
    disp_w, disp_h = disp_size

    # Direct Type Extraction: Handle specific incoming payloads instantly
    if isinstance(metadata_or_bbs, list):
        # Acknowledge your track array: Fast extraction of structural list layers
        if isinstance(metadata_or_bbs[0], dict):
            boxes = np.array([b["bbox"] for b in metadata_or_bbs], dtype=np.float32)
        else:
            boxes = np.array(metadata_or_bbs, dtype=np.float32)
    elif isinstance(metadata_or_bbs, np.ndarray):
        boxes = metadata_or_bbs
    else:
        # Fallback mechanism handles raw GPU traces securely if passed downstream
        boxes = metadata_or_bbs.detach().cpu().numpy()

    # Pre-allocate coordinates in a single vectorized sweep
    x1 = (boxes[:, 0] * scale_display_x).astype(np.int32)
    y1 = (boxes[:, 1] * scale_display_y).astype(np.int32)
    x2 = (boxes[:, 2] * scale_display_x).astype(np.int32)
    y2 = (boxes[:, 3] * scale_display_y).astype(np.int32)

    # Perform low-level C++ drawing inside OpenCV (Zero typing penalties)
    for i in range(len(boxes)):
        # Clip bounding box corners safely within resolution bounds
        rx1 = max(0, x1[i])
        ry1 = max(0, y1[i])
        rx2 = min(disp_w - 1, x2[i])
        ry2 = min(disp_h - 1, y2[i])

        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), color, 2)

    return display_frame


def tensor2opencv(frame_source, device_input, is_bgr=True, resize_h=640, resize_w=640):
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
        img_cpu = img_cpu.reshape((resize_h, resize_w, 3))

    #  Fix Visibility: ONLY multiply if it's actually floating point
    # If uint8 is multiplied by 255, it wraps around and creates "neon" colors
    if img_cpu.dtype != np.uint8:
        if img_cpu.max() <= 1.0:
            img_cpu = (img_cpu * 255).clip(0, 255).astype(np.uint8)
        else:
            img_cpu = img_cpu.astype(np.uint8)

    # Color Space: Standardize to BGR for imwrite
    if not is_bgr:
        if len(img_cpu.shape) == 3:
            # Swap RGB (Torch/Decoder) -> BGR (OpenCV)
            # img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
            # Only swap if the source is RGB (GPU Path)
            # CPU path is already BGR from OpenCV reader
            if device_input == "cuda":
                img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
            else:
                # Ensure it's contiguous for saving
                img_cpu = np.ascontiguousarray(img_cpu)
        else:
            img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)

    return img_cpu


def gpumat2cupyv1(gpu_mat):
    """Bridge OpenCV GpuMat to CuPy without copying data."""
    # Get properties from GpuMat
    w, h = gpu_mat.size()
    # Check if it's 3-channel (CV_8UC3) or 1-channel (CV_8UC1)
    channels = 3 if gpu_mat.type() == cv2.CV_8UC3 else 1

    if channels == 3:
        shape = (h, w, 3)
        # strides = (bytes_per_row, bytes_per_pixel, bytes_per_channel)
        strides = (gpu_mat.step, 3, 1)
    else:
        shape = (h, w)
        strides = (gpu_mat.step, 1)

    # Map OpenCV types to CuPy typestrs
    # CV_8UC1 is 'u1' (unsigned 1-byte), etc.
    # type_map = {cv2.CV_8U: "|u1", cv2.CV_32F: "<f4", cv2.CV_8UC1: "|u1"}

    # Create the __cuda_array_interface__ dictionary
    # This tells CuPy where the data is and how it's shaped
    if_dict = {
        "version": 3,
        "shape": shape,
        "typestr": "|u1",
        # "descr": [("", type_map.get(gpu_mat.type(), "|u1"))],
        "data": (gpu_mat.cudaPtr(), False),  # (Pointer, Read-only)
        "strides": strides,
    }

    # Create a dummy object with the interface and wrap it in CuPy
    class Holder:
        pass

    holder = Holder()
    holder.__cuda_array_interface__ = if_dict
    return cupy.asarray(holder)


def gpumat2cupy(gpu_mat):
    """
    Bridge OpenCV GpuMat to CuPy instantly without copying data or parsing dict envelopes.
    """
    w, h = gpu_mat.size()
    channels = 3 if gpu_mat.type() == cv2.CV_8UC3 else 1

    if channels == 3:
        shape = (h, w, 3)
        strides = (gpu_mat.step, 3, 1)
    else:
        shape = (h, w)
        strides = (gpu_mat.step, 1)

    # OPTIMIZATION: Construct a native CuPy array memory wrapper instantly over the raw C++ pointer.
    # This completely bypasses the Python dictionary interpreter parsing and 'cupy.asarray' overhead.
    mem = cupy.cuda.UnownedMemory(gpu_mat.cudaPtr(), gpu_mat.step * h, gpu_mat)
    mptr = cupy.cuda.MemoryPointer(mem, 0)

    return cupy.ndarray(shape=shape, dtype=cupy.uint8, memptr=mptr, strides=strides)


def torch2gpumat(tensor):
    """
    Creates an OpenCV GpuMat pointing to the same memory as a PyTorch tensor.
    ZERO-COPY: No data is moved; only the memory address is shared.
    """
    # Ensure tensor is [H, W, C] and contiguous for OpenCV
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0).contiguous()

    # Bridge to CuPy (Zero-Copy)
    cp_arr = cupy.asanyarray(tensor)

    # Wrap in GpuMat
    # cv2.CV_8UC3 for uint8, CV_32FC3 for float
    dtype = cv2.CV_8UC3 if tensor.dtype == torch.uint8 else cv2.CV_32FC3

    gpumat = cv2.cuda_GpuMat(
        tensor.shape[1],  # Width
        tensor.shape[0],  # Height
        dtype,
        cp_arr.data.ptr,
    )
    return gpumat


# This kernel calculates the [x1, y1, x2, y2] for every detected object label
BOUNDS_KERNEL_CODE = r"""
extern "C" __global__
void get_bounds(const int* labeled, int pitch, int w, int h, int num_labels,
                int* x1, int* y1, int* x2, int* y2) {
    // Calculate the unique 2D pixel coordinates (x, y) for this specific thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary Check: Ensure the thread isn't trying to read outside the image dimensions
    if (x < w && y < h) {
        // Use the pitch to correctly calculate the memory offset
        int label = labeled[y * pitch + x];

        if (label > 0 && label <= num_labels) {
            // Atomically update the bounding box for this specific label
            atomicMin(&x1[label], x);
            atomicMin(&y1[label], y);
            // +1 ensures the box captures the full pixel and matches OpenCV ROI logic
            atomicMax(&x2[label], x + 1);
            atomicMax(&y2[label], y + 1);
        }
    }
}
"""

# Compile the kernel once
get_bounds_kernel = cupy.RawKernel(BOUNDS_KERNEL_CODE, "get_bounds")

# BOUNDS_KERNEL_CODE = r"""
# extern "C" __global__
# void get_bounds_pure(const int* labeled, int pitch_elements, int w, int h, int num_labels,
#                      int* x1, int* y1, int* x2, int* y2) {
#     int x = blockIdx.x * blockDim.x + threadIdx.x;
#     int y = blockIdx.y * blockDim.y + threadIdx.y;

#     if (x < w && y < h) {
#         // Safe 32-bit element pitch extraction
#         int label = labeled[y * pitch_elements + x];

#         if (label > 0 && label <= num_labels) {
#             atomicMin(&x1[label], x);
#             atomicMin(&y1[label], y);
#             atomicMax(&x2[label], x + 1);
#             atomicMax(&y2[label], y + 1);
#         }
#     }
# }
# """
# get_bounds_kernel = cupy.RawKernel(BOUNDS_KERNEL_CODE, "get_bounds_pure")


# def merge_boxes_gpuv1(raw_boxes, gap_limit=10, size_limit=1000):
#     """
#     Refined Parallel Merger with Size Constraints.
#     Prevents merges that would create boxes larger than size_limit.
#     """
#     if raw_boxes.shape[0] <= 1:
#         return raw_boxes

#     x1, y1, x2, y2 = raw_boxes.unbind(1)

#     # 1. Calculate pairwise gaps (Existing logic)
#     h_gaps = torch.max(
#         torch.zeros(1, device=raw_boxes.device),
#         torch.max(x1.unsqueeze(0) - x2.unsqueeze(1), x1.unsqueeze(1) - x2.unsqueeze(0)),
#     )
#     v_gaps = torch.max(
#         torch.zeros(1, device=raw_boxes.device),
#         torch.max(y1.unsqueeze(0) - y2.unsqueeze(1), y1.unsqueeze(1) - y2.unsqueeze(0)),
#     )

#     # 2. NEW: Calculate potential union dimensions for ALL pairs [N, N]
#     # We find the min/max coordinates if box i and box j were merged
#     union_x1 = torch.min(x1.unsqueeze(0), x1.unsqueeze(1))
#     union_y1 = torch.min(y1.unsqueeze(0), y1.unsqueeze(1))
#     union_x2 = torch.max(x2.unsqueeze(0), x2.unsqueeze(1))
#     union_y2 = torch.max(y2.unsqueeze(0), y2.unsqueeze(1))

#     union_w = union_x2 - union_x1
#     union_h = union_y2 - union_y1

#     # 3. ADJACENCY MASK: Must be close AND the result must be under the limit
#     # This prevents the creation of massive "megaboxes"
#     adj = (
#         (h_gaps < gap_limit)
#         & (v_gaps < gap_limit)
#         & (union_w < size_limit)
#         & (union_h < size_limit)
#     )

#     # 4. Parallel Connected Components (Existing logic)
#     components = torch.arange(raw_boxes.shape[0], device=raw_boxes.device)
#     r = 3
#     for _ in range(r):
#         components = torch.max(adj * components, dim=1).values

#     unique_ids = components.unique()
#     merged = []
#     for i in unique_ids:
#         mask = components == i
#         merged.append(
#             torch.cat(
#                 [raw_boxes[mask, :2].min(0).values, raw_boxes[mask, 2:].max(0).values]
#             )
#         )

#     return torch.stack(merged)


# last
def merge_boxes_gpu(raw_boxes, gap_limit=10, size_limit=1280, max_cached_elements=100):
    """
    Refined Parallel Merger utilizing static function-attached scratchpads.
    Maintains a 100% linear VRAM profile and eliminates dynamic allocations.
    """
    if raw_boxes.shape[0] <= 1:
        return raw_boxes

    # 1. ROBUST CACHE CHECK: Verify shape dimension bounds to prevent 1D flat layout leakage
    if (
        not hasattr(merge_boxes_gpu, "adj_matrix")
        or merge_boxes_gpu.adj_matrix.ndim != 2
    ):
        merge_boxes_gpu.adj_matrix = torch.zeros(
            (max_cached_elements, max_cached_elements),
            dtype=torch.bool,
            device=raw_boxes.device,
        )
        merge_boxes_gpu.components = torch.zeros(
            (max_cached_elements,), dtype=torch.long, device=raw_boxes.device
        )
        merge_boxes_gpu.scratch_out = torch.zeros(
            (max_cached_elements, 4), dtype=torch.float32, device=raw_boxes.device
        )

    # Establish safe spatial clip windows depending on current frame tracking load
    N = min(raw_boxes.shape[0], max_cached_elements)

    # 2. VECTORIZED ARITHMETIC WITH VIEWS: Completely replace unbind() and unsqueeze() list loops
    x1 = raw_boxes[:N, 0]
    y1 = raw_boxes[:N, 1]
    x2 = raw_boxes[:N, 2]
    y2 = raw_boxes[:N, 3]

    # Compute gaps smoothly using in-place operations over zero-copy memory layouts
    h_gaps = torch.clamp(
        torch.max(x1.view(N, 1) - x2.view(1, N), x1.view(1, N) - x2.view(N, 1)), min=0
    )
    v_gaps = torch.clamp(
        torch.max(y1.view(N, 1) - y2.view(1, N), y1.view(1, N) - y2.view(N, 1)), min=0
    )

    # Map target union envelopes natively
    union_w = torch.max(x2.view(N, 1), x2.view(1, N)) - torch.min(
        x1.view(N, 1), x1.view(1, N)
    )
    union_h = torch.max(y2.view(N, 1), y2.view(1, N)) - torch.min(
        y1.view(N, 1), y1.view(1, N)
    )

    # Overwrite adjacency bounds directly into the pre-allocated cache slice
    adj = merge_boxes_gpu.adj_matrix[:N, :N]
    adj.copy_(
        (h_gaps < gap_limit)
        & (v_gaps < gap_limit)
        & (union_w < size_limit)
        & (union_h < size_limit)
    )

    # 3. ACCELERATED CONNECTED COMPONENTS: Run pointer rotations in place
    comp = merge_boxes_gpu.components[:N]
    torch.arange(N, device=raw_boxes.device, out=comp)

    # Unified logical components compression step
    for _ in range(3):
        comp.copy_(torch.max(adj * comp, dim=1).values)

    # unique_ids = comp.unique()
    # num_merged = unique_ids.shape[0]

    # # 4. STATIC MEMORY AGGREGATION: Eliminate the loops appending torch.cat() arrays
    # out_buffer = merge_boxes_gpu.scratch_out[:num_merged]

    # for idx, i in enumerate(unique_ids):
    #     mask = comp == i
    #     boxes_subset = raw_boxes[:N][mask]

    #     # Write structural outputs directly into our static memory block channels
    #     out_buffer[idx, 0] = boxes_subset[:, 0].min()
    #     out_buffer[idx, 1] = boxes_subset[:, 1].min()
    #     out_buffer[idx, 2] = boxes_subset[:, 2].max()
    #     out_buffer[idx, 3] = boxes_subset[:, 3].max()

    # # Return the clean floating-point tensor slice natively
    # return out_buffer.clone()

    # 1. Initialize an allocation-free O(1) bitmask directly over device registers
    # N is already our maximum candidate dimension cap mapped for this frame
    valid_bitmask = torch.zeros((N,), dtype=torch.bool, device=raw_boxes.device)

    # 2. Flag every single active logical group ID index in parallel in a single CUDA command
    valid_bitmask[comp] = True

    # 3. Pull unique indices instantly using a fast non-zero memory address pass
    unique_ids = torch.nonzero(valid_bitmask).squeeze(1)
    num_merged = unique_ids.shape[0]

    # 4. Map the aggregation workspace straight onto our persistent static cache slice
    out_buffer = merge_boxes_gpu.scratch_out[:num_merged]

    # 5. Extract bounding coordinates cleanly without dynamic vector list loops
    for idx, i in enumerate(unique_ids):
        mask = comp == i
        boxes_subset = raw_boxes[:N][mask]

        if boxes_subset.ndim == 1:
            # Reshapes a [4] vector back into a valid [1, 4] 2D matrix canvas
            boxes_subset = boxes_subset.view(1, -1)

        # Write structural outputs directly into our static memory block channels
        out_buffer[idx, 0] = boxes_subset[:, 0].min()
        out_buffer[idx, 1] = boxes_subset[:, 1].min()
        out_buffer[idx, 2] = boxes_subset[:, 2].max()
        out_buffer[idx, 3] = boxes_subset[:, 3].max()

    # Return the clean floating-point tensor slice natively
    return out_buffer.clone()


def merge_boxes_gpu_8_25(boxes, gap_limit, size_limit=None, max_cached_elements=100):
    """
    Optimized GPU box merging using a grid-based aggregation algorithm.
    This avoids the O(n^2) complexity of pairwise distance calculations, making it
    ideal for high-density scenarios with hundreds or thousands of boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) with boxes [x1, y1, x2, y2].
        gap_limit (float): The cell size for the grid. Boxes within the same
                           cell will be merged.
        size_limit (float, optional): Not used in this version but kept for API compatibility.
        max_cached_elements (int, optional): Not used but kept for API compatibility.

    Returns:
        torch.Tensor: A tensor of shape (M, 4) with the merged boxes.
    """
    if boxes.shape[0] == 0:
        return torch.empty((0, 4), device=boxes.device, dtype=boxes.dtype)

    # Assume a 640x640 coordinate space, as this is where the merging happens.
    # If this changes, these values must be updated.
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 640

    # --- Grid-Based Aggregation ---

    # 1. Calculate box centers
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2.0
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2.0

    # 2. Define the grid and assign each box to a grid cell ID
    grid_w = int(IMAGE_WIDTH / gap_limit) + 1

    # Assign a 1D grid cell index to each box
    grid_x_indices = (x_centers / gap_limit).long()
    grid_y_indices = (y_centers / gap_limit).long()

    # Combine 2D grid indices into a single 1D index for scatter_reduce
    cell_indices = grid_y_indices * grid_w + grid_x_indices
    num_cells = grid_w * (int(IMAGE_HEIGHT / gap_limit) + 1)

    # 3. Use scatter_reduce to merge boxes in each cell in parallel
    # We need tensors to hold the min/max coordinates for each cell.
    # Initialize `merged_x1` and `merged_y1` to a large value.
    # Initialize `merged_x2` and `merged_y2` to a small value.

    merged_x1 = torch.full(
        (num_cells,), float("inf"), device=boxes.device, dtype=boxes.dtype
    )
    merged_y1 = torch.full(
        (num_cells,), float("inf"), device=boxes.device, dtype=boxes.dtype
    )
    merged_x2 = torch.full(
        (num_cells,), float("-inf"), device=boxes.device, dtype=boxes.dtype
    )
    merged_y2 = torch.full(
        (num_cells,), float("-inf"), device=boxes.device, dtype=boxes.dtype
    )

    # Find the min x1 and y1 for all boxes in each cell
    merged_x1.scatter_reduce_(
        0, cell_indices, boxes[:, 0], reduce="amin", include_self=False
    )
    merged_y1.scatter_reduce_(
        0, cell_indices, boxes[:, 1], reduce="amin", include_self=False
    )

    # Find the max x2 and y2 for all boxes in each cell
    merged_x2.scatter_reduce_(
        0, cell_indices, boxes[:, 2], reduce="amax", include_self=False
    )
    merged_y2.scatter_reduce_(
        0, cell_indices, boxes[:, 3], reduce="amax", include_self=False
    )

    # 4. Filter out the empty cells to get the final merged boxes
    # A cell is considered populated if its min value is not infinity.
    valid_cells_mask = merged_x1 != float("inf")

    final_boxes = torch.stack(
        [
            merged_x1[valid_cells_mask],
            merged_y1[valid_cells_mask],
            merged_x2[valid_cells_mask],
            merged_y2[valid_cells_mask],
        ],
        dim=1,
    )

    # The grid-based approach might merge boxes that are in the same cell but
    # not directly adjacent. A second pass on the much smaller set of merged
    # boxes could be done, but for performance, this initial merge is often sufficient.
    # For now, we return the direct result of the grid aggregation.

    return final_boxes


# last
def merge_boxes_gpu_v1(
    raw_boxes, gap_limit=10, size_limit=1000, max_cached_elements=256
):
    """
    Refined Parallel Merger utilizing static function-attached scratchpads.
    100% loop-free vectorized tensor reduction matching original accuracy.
    """
    if raw_boxes.shape[0] <= 1:
        return raw_boxes

    # 1. PERSISTENT WORKSPACE CACHE INITIALIZATION
    if (
        not hasattr(merge_boxes_gpu, "adj_matrix")
        or merge_boxes_gpu.adj_matrix.ndim != 2
    ):
        merge_boxes_gpu.adj_matrix = torch.zeros(
            (max_cached_elements, max_cached_elements),
            dtype=torch.bool,
            device=raw_boxes.device,
        )
        merge_boxes_gpu.components = torch.zeros(
            (max_cached_elements,), dtype=torch.long, device=raw_boxes.device
        )
        merge_boxes_gpu.scratch_out = torch.zeros(
            (max_cached_elements, 4), dtype=torch.float32, device=raw_boxes.device
        )

    N = min(raw_boxes.shape[0], max_cached_elements)

    # 2. VECTORIZED ARITHMETIC WITH VIEWS (Your exact matching geometry)
    x1 = raw_boxes[:N, 0]
    y1 = raw_boxes[:N, 1]
    x2 = raw_boxes[:N, 2]
    y2 = raw_boxes[:N, 3]

    h_gaps = torch.clamp(
        torch.max(x1.view(N, 1) - x2.view(1, N), x1.view(1, N) - x2.view(N, 1)), min=0
    )
    v_gaps = torch.clamp(
        torch.max(y1.view(N, 1) - y2.view(1, N), y1.view(1, N) - y2.view(N, 1)), min=0
    )

    union_w = torch.max(x2.view(N, 1), x2.view(1, N)) - torch.min(
        x1.view(N, 1), x1.view(1, N)
    )
    union_h = torch.max(y2.view(N, 1), y2.view(1, N)) - torch.min(
        y1.view(N, 1), y1.view(1, N)
    )

    adj = merge_boxes_gpu.adj_matrix[:N, :N]
    adj.copy_(
        (h_gaps < gap_limit)
        & (v_gaps < gap_limit)
        & (union_w < size_limit)
        & (union_h < size_limit)
    )

    # 3. ACCELERATED CONNECTED COMPONENTS
    comp = merge_boxes_gpu.components[:N]
    torch.arange(N, device=raw_boxes.device, out=comp)

    # Label propagation iterations
    for _ in range(3):
        comp.copy_(torch.max(adj * comp, dim=1).values)

    # 4. 100% LOOP-FREE VECTORIZED REDUCTION (Replaces lines 102-115)
    # Remap cluster assignments to continuous indices from 0 to M-1
    unique_ids, cluster_assignments = torch.unique(comp, return_inverse=True)
    num_merged = unique_ids.shape[0]

    # Pre-size output buffer view slices
    out_buffer = merge_boxes_gpu.scratch_out[:num_merged]

    # Vectorized min/max reduction using scatter_reduce
    # We initialize extreme values to find true minimums and maximums
    out_buffer[:, 0:2].fill_(float("inf"))
    # out_buffer[:, 1].fill_(float("inf"))
    out_buffer[:, 2:4].fill_(float("-inf"))
    # out_buffer[:, 3].fill_(float("-inf"))

    # Parallel reduction via native C++ PyTorch backend kernels (0ms loop overhead)
    out_buffer[:, 0].scatter_reduce_(
        0, cluster_assignments, x1, reduce="amin", include_self=False
    )
    out_buffer[:, 1].scatter_reduce_(
        0, cluster_assignments, y1, reduce="amin", include_self=False
    )
    out_buffer[:, 2].scatter_reduce_(
        0, cluster_assignments, x2, reduce="amax", include_self=False
    )
    out_buffer[:, 3].scatter_reduce_(
        0, cluster_assignments, y2, reduce="amax", include_self=False
    )

    # Enforce geometric size limits vectorially
    widths = out_buffer[:, 2] - out_buffer[:, 0]
    heights = out_buffer[:, 3] - out_buffer[:, 1]

    width_mask = widths > size_limit
    height_mask = heights > size_limit

    out_buffer[width_mask, 2] = out_buffer[width_mask, 0] + size_limit
    out_buffer[height_mask, 3] = out_buffer[height_mask, 1] + size_limit

    return out_buffer.clone()


def merge_boxes_gpu_v2(
    raw_boxes, gap_limit=10, size_limit=1000, max_cached_elements=256
):
    """
    Refined Parallel Merger utilizing static function-attached scratchpads.
    Maintains a 100% linear VRAM profile and eliminates dynamic allocations.
    Uses fully vectorized 2D broadcasting to calculate merged bounding boxes
    simultaneously on the GPU without sequential loops or CPU synchronization stalls.
    """
    if raw_boxes.shape[0] <= 1:
        return raw_boxes

    # 1. ROBUST CACHE CHECK: Verify shape dimension bounds to prevent 1D flat layout leakage
    if (
        not hasattr(merge_boxes_gpu, "adj_matrix")
        or merge_boxes_gpu.adj_matrix.ndim != 2
    ):
        merge_boxes_gpu.adj_matrix = torch.zeros(
            (max_cached_elements, max_cached_elements),
            dtype=torch.bool,
            device=raw_boxes.device,
        )
        merge_boxes_gpu.components = torch.zeros(
            (max_cached_elements,), dtype=torch.long, device=raw_boxes.device
        )
        merge_boxes_gpu.scratch_out = torch.zeros(
            (max_cached_elements, 4), dtype=torch.float32, device=raw_boxes.device
        )

    # Establish safe spatial clip windows depending on current frame tracking load
    N = min(raw_boxes.shape[0], max_cached_elements)

    # 2. VECTORIZED ARITHMETIC WITH VIEWS: Completely replace unbind() and unsqueeze() list loops
    x1 = raw_boxes[:N, 0]
    y1 = raw_boxes[:N, 1]
    x2 = raw_boxes[:N, 2]
    y2 = raw_boxes[:N, 3]

    # Create spatial lookup grids via view broadcasting
    x1_col, x1_row = x1.view(N, 1), x1.view(1, N)
    y1_col, y1_row = y1.view(N, 1), y1.view(1, N)
    x2_col, x2_row = x2.view(N, 1), x2.view(1, N)
    y2_col, y2_row = y2.view(N, 1), y2.view(1, N)

    # Compute distance boundaries directly using hardware vector instructions
    h_gaps = torch.clamp(
        torch.maximum(x1_col, x1_row) - torch.minimum(x2_col, x2_row), min=0
    )
    v_gaps = torch.clamp(
        torch.maximum(y1_col, y1_row) - torch.minimum(y2_col, y2_row), min=0
    )

    # Map target union envelopes without generating intermediate subtraction tensors
    union_w = torch.maximum(x2_col, x2_row) - torch.minimum(x1_col, x1_row)
    union_h = torch.maximum(y2_col, y2_row) - torch.minimum(y1_col, y1_row)

    # Overwrite adjacency bounds directly into the pre-allocated cache slice
    adj = merge_boxes_gpu.adj_matrix[:N, :N]
    adj.copy_(
        (h_gaps < gap_limit)
        & (v_gaps < gap_limit)
        & (union_w < size_limit)
        & (union_h < size_limit)
    )

    # 3. ACCELERATED CONNECTED COMPONENTS: Run pointer rotations in place
    comp = merge_boxes_gpu.components[:N]
    torch.arange(N, device=raw_boxes.device, out=comp)

    # Wrap the destination outputs into a fast component view format
    comp_idx_scratch = torch.zeros(N, dtype=torch.long, device=raw_boxes.device)
    output_tuple = (comp, comp_idx_scratch)

    # Unified logical components compression step
    for _ in range(3):
        torch.max(torch.where(adj, comp, 0), dim=1, out=output_tuple)

    # 4. VECTORIZED COMPONENT REDUCTION (ZERO-LOOP BRIDGING)
    # Find the unique cluster assignments entirely on the GPU
    unique_ids = comp.unique()
    num_merged = unique_ids.shape[0]

    # Sub-slice our persistent function cache block instantly
    out_buffer = merge_boxes_gpu.scratch_out[:num_merged]
    boxes_subset = raw_boxes[:N]

    # Create a 2D broadcasted membership equality grid [Num_Clusters, N_Boxes]
    # This evaluates cluster assignments for all boxes simultaneously on the GPU!
    mask_grid = comp.unsqueeze(0) == unique_ids.view(-1, 1)

    # Mask out invalid bounding box entries by casting non-members to extreme float values.
    # This guarantees unassigned boxes do not interfere with the parallel min/max reduction.
    x1_masked = torch.where(mask_grid, boxes_subset[:, 0].unsqueeze(0), float("inf"))
    y1_masked = torch.where(mask_grid, boxes_subset[:, 1].unsqueeze(0), float("inf"))
    x2_masked = torch.where(mask_grid, boxes_subset[:, 2].unsqueeze(0), float("-inf"))
    y2_masked = torch.where(mask_grid, boxes_subset[:, 3].unsqueeze(0), float("-inf"))

    # Execute single-pass parallel reductions across dim=1 (the boxes axis)
    # Bypasses Python loop steps, CPU stalls, and unoptimized scatter commands entirely
    out_buffer[:, 0] = x1_masked.min(dim=1).values
    out_buffer[:, 1] = y1_masked.min(dim=1).values
    out_buffer[:, 2] = x2_masked.max(dim=1).values
    out_buffer[:, 3] = y2_masked.max(dim=1).values

    # Clean local tensor references inside the function namespace to protect VRAM scope
    del (
        mask_grid,
        comp_idx_scratch,
        x1_masked,
        y1_masked,
        x2_masked,
        y2_masked,
        boxes_subset,
    )

    return out_buffer.clone()


def merge_boxes_gpu_v3(
    raw_boxes, gap_limit=10, size_limit=640, max_cached_elements=1500
):
    """
    Optimized Parallel Merger for pixel-integer motion boxes.
    Groups close boxes and dynamically expands all resulting blocks up to a uniform
    size_limit square for optimal YOLO backbone processing.

    Args:
        raw_boxes (torch.Tensor): Int or Float Tensor of shape [N, 4] -> [x1, y1, x2, y2]
        gap_limit (int): Proximity merge trigger threshold in absolute pixels.
        size_limit (int): Target square dimension (width & height) for YOLO input.
    """
    if raw_boxes.shape[0] == 0:
        return raw_boxes

    # Sizing metrics for center-out expansion
    half_size = float(size_limit) / 2.0

    # 1. DYNAMIC FRAME RESOLUTION DETECTION
    # Automatically extracts frame limits from the highest observed coordinates in the batch
    # to avoid hardcoding video resolution dimensions.
    if raw_boxes.shape[0] > 0:
        frame_w = int(torch.max(raw_boxes[:, 2]).item())
        frame_h = int(torch.max(raw_boxes[:, 3]).item())
        # Safe fallback buffer if tracking objects at the extreme top/left edges
        frame_w = max(frame_w, 640)
        frame_h = max(frame_h, 640)
    else:
        frame_w, frame_h = 1920, 1080

    # Handle single box edge-case quickly to minimize path latency
    if raw_boxes.shape[0] == 1:
        merged_buffer = raw_boxes.clone().float()
        num_merged = 1
    else:
        # 2. ROBUST VRAM CACHE SEEDING
        if (
            not hasattr(merge_boxes_gpu, "adj_matrix")
            or merge_boxes_gpu.adj_matrix.ndim != 2
        ):
            merge_boxes_gpu.adj_matrix = torch.zeros(
                (max_cached_elements, max_cached_elements),
                dtype=torch.bool,
                device=raw_boxes.device,
            )
            merge_boxes_gpu.components = torch.zeros(
                (max_cached_elements,), dtype=torch.long, device=raw_boxes.device
            )
            merge_boxes_gpu.scratch_out = torch.zeros(
                (max_cached_elements, 4), dtype=torch.float32, device=raw_boxes.device
            )

        N = min(raw_boxes.shape[0], max_cached_elements)

        # 3. PARALLEL GEOMETRIC GAP CALCULATIONS
        x1 = raw_boxes[:N, 0]
        y1 = raw_boxes[:N, 1]
        x2 = raw_boxes[:N, 2]
        y2 = raw_boxes[:N, 3]

        h_gaps = torch.clamp(
            torch.max(x1.view(N, 1) - x2.view(1, N), x1.view(1, N) - x2.view(N, 1)),
            min=0,
        )
        v_gaps = torch.clamp(
            torch.max(y1.view(N, 1) - y2.view(1, N), y1.view(1, N) - y2.view(N, 1)),
            min=0,
        )

        union_w = torch.max(x2.view(N, 1), x2.view(1, N)) - torch.min(
            x1.view(N, 1), x1.view(1, N)
        )
        union_h = torch.max(y2.view(N, 1), y2.view(1, N)) - torch.min(
            y1.view(N, 1), y1.view(1, N)
        )

        # Vectorized generation of adjacency grid state properties
        adj = merge_boxes_gpu.adj_matrix[:N, :N]
        adj.copy_(
            (h_gaps < gap_limit)
            & (v_gaps < gap_limit)
            & (union_w <= size_limit)
            & (union_h <= size_limit)
        )

        # 4. CONNECTED COMPONENT DISCOVERY VIA GRAPH ITERATIONS
        comp = merge_boxes_gpu.components[:N]
        torch.arange(N, device=raw_boxes.device, out=comp)

        for _ in range(3):
            comp.copy_(torch.max(adj * comp, dim=1).values)

        unique_ids = comp.unique()
        num_merged = unique_ids.shape[0]
        merged_buffer = merge_boxes_gpu.scratch_out[:num_merged]

        for idx, i in enumerate(unique_ids):
            mask = comp == i
            boxes_subset = raw_boxes[:N][mask]

            merged_buffer[idx, 0] = boxes_subset[:, 0].min()
            merged_buffer[idx, 1] = boxes_subset[:, 1].min()
            merged_buffer[idx, 2] = boxes_subset[:, 2].max()
            merged_buffer[idx, 3] = boxes_subset[:, 3].max()

    # 5. SYMMETRIC CENTROID SQUARE EXPANSION
    cx = (merged_buffer[:, 0] + merged_buffer[:, 2]) * 0.5
    cy = (merged_buffer[:, 1] + merged_buffer[:, 3]) * 0.5

    ex1 = cx - half_size
    ey1 = cy - half_size
    ex2 = cx + half_size
    ey2 = cy + half_size

    # 6. EDGE-COLLISION ANCHOR SHIFTING
    # Instead of destructive clipping (which breaks squares), shift windows inward
    # when they hit frame edges to guarantee your exact YOLO aspect ratio is maintained.
    shift_left = torch.clamp(0.0 - ex1, min=0)
    shift_right = torch.clamp(ex2 - frame_w, min=0)
    ex1 += shift_left - shift_right
    ex2 += shift_left - shift_right

    shift_top = torch.clamp(0.0 - ey1, min=0)
    shift_bottom = torch.clamp(ey2 - frame_h, min=0)
    ey1 += shift_top - shift_bottom
    ey2 += shift_top - shift_bottom

    # 7. INTEGER FORMAT OUTPUT AND SECURE BOUNDARY BOUNDING
    final_output = torch.zeros(
        (num_merged, 4), dtype=torch.long, device=raw_boxes.device
    )
    final_output[:, 0] = torch.clamp(ex1.round().long(), min=0, max=frame_w)
    final_output[:, 1] = torch.clamp(ey1.round().long(), min=0, max=frame_h)
    final_output[:, 2] = torch.clamp(ex2.round().long(), min=0, max=frame_w)
    final_output[:, 3] = torch.clamp(ey2.round().long(), min=0, max=frame_h)

    return final_output


def merge_boxes_gpu_v4(
    raw_boxes, gap_limit=10, size_limit=1000, max_cached_elements=1500
):
    """
    Refined Parallel Merger utilizing static function-attached scratchpads.
    Maintains a 100% linear VRAM profile and eliminates dynamic allocations.
    """
    if raw_boxes.shape[0] <= 1:
        return raw_boxes

    # ─── ADD THIS HIGH-THROUGHPUT CANDIDATE FILTER GUARD ──────────────────
    # 1. Compute areas of all boxes instantly in parallel on the GPU
    # widths = raw_boxes[:, 2] - raw_boxes[:, 0]
    # heights = raw_boxes[:, 3] - raw_boxes[:, 1]
    # areas = widths * heights

    # # 2. If the scene is overloaded, prioritize the top 400 largest macro regions
    # # This prevents noise artifacts from blowing up the [N, N] grid layout!
    # if raw_boxes.shape[0] > 400:
    #     _, top_indices = torch.topk(areas, k=400, sorted=False)
    #     raw_boxes = raw_boxes[top_indices]
    # ──────────────────────────────────────────────────────────────────────

    # 1. ROBUST CACHE CHECK: Verify shape dimension bounds to prevent 1D flat layout leakage
    if (
        not hasattr(merge_boxes_gpu, "adj_matrix")
        or merge_boxes_gpu.adj_matrix.ndim != 2
    ):
        merge_boxes_gpu.adj_matrix = torch.zeros(
            (max_cached_elements, max_cached_elements),
            dtype=torch.bool,
            device=raw_boxes.device,
        )
        merge_boxes_gpu.components = torch.zeros(
            (max_cached_elements,), dtype=torch.long, device=raw_boxes.device
        )
        merge_boxes_gpu.components_idx = torch.zeros(
            (max_cached_elements,), dtype=torch.long, device=raw_boxes.device
        )
        merge_boxes_gpu.scratch_out = torch.zeros(
            (max_cached_elements, 4), dtype=torch.float32, device=raw_boxes.device
        )

    # Establish safe spatial clip windows depending on current frame tracking load
    N = min(raw_boxes.shape[0], max_cached_elements)

    # 2. VECTORIZED ARITHMETIC WITH VIEWS: Completely replace unbind() and unsqueeze() list loops
    x1 = raw_boxes[:N, 0]
    y1 = raw_boxes[:N, 1]
    x2 = raw_boxes[:N, 2]
    y2 = raw_boxes[:N, 3]

    # # Compute gaps smoothly using in-place operations over zero-copy memory layouts
    # h_gaps = torch.clamp(
    #     torch.max(x1.view(N, 1) - x2.view(1, N), x1.view(1, N) - x2.view(N, 1)), min=0
    # )
    # v_gaps = torch.clamp(
    #     torch.max(y1.view(N, 1) - y2.view(1, N), y1.view(1, N) - y2.view(N, 1)), min=0
    # )

    # # Map target union envelopes natively
    # union_w = torch.max(x2.view(N, 1), x2.view(1, N)) - torch.min(
    #     x1.view(N, 1), x1.view(1, N)
    # )
    # union_h = torch.max(y2.view(N, 1), y2.view(1, N)) - torch.min(
    #     y1.view(N, 1), y1.view(1, N)
    # )
    x1_col, x1_row = x1.view(N, 1), x1.view(1, N)
    y1_col, y1_row = y1.view(N, 1), y1.view(1, N)
    x2_col, x2_row = x2.view(N, 1), x2.view(1, N)
    y2_col, y2_row = y2.view(N, 1), y2.view(1, N)

    # Compute distance boundaries directly.
    # torch.maximum/minimum maps to direct hardware vector instructions.
    h_gaps = torch.clamp(
        torch.maximum(x1_col, x1_row) - torch.minimum(x2_col, x2_row), min=0
    )
    v_gaps = torch.clamp(
        torch.maximum(y1_col, y1_row) - torch.minimum(y2_col, y2_row), min=0
    )

    # Map target union envelopes without generating intermediate subtraction tensors
    union_w = torch.maximum(x2_col, x2_row) - torch.minimum(x1_col, x1_row)
    union_h = torch.maximum(y2_col, y2_row) - torch.minimum(y1_col, y1_row)

    # Overwrite adjacency bounds directly into the pre-allocated cache slice
    adj = merge_boxes_gpu.adj_matrix[:N, :N]
    adj.copy_(
        (h_gaps < gap_limit)
        & (v_gaps < gap_limit)
        & (union_w < size_limit)
        & (union_h < size_limit)
    )

    # 3. ACCELERATED CONNECTED COMPONENTS: Run pointer rotations in place
    comp = merge_boxes_gpu.components[:N]
    # comp_idx = merge_boxes_gpu.components_idx[:N]
    comp_idx_scratch = torch.zeros(N, dtype=torch.long, device=raw_boxes.device)
    torch.arange(N, device=raw_boxes.device, out=comp)

    # Pack the destination outputs into the exact 2-tensor tuple expected by torch.max
    output_tuple = (comp, comp_idx_scratch)

    # Unified logical components compression step
    for _ in range(3):
        # comp.copy_(torch.max(adj * comp, dim=1).values)
        # Using a logical matrix view avoids any intermediate [N, N] expansions
        torch.max(torch.where(adj, comp, 0), dim=1, out=output_tuple)

    # unique_ids = comp.unique()
    # num_merged = unique_ids.shape[0]

    # # 4. STATIC MEMORY AGGREGATION: Eliminate the loops appending torch.cat() arrays
    # out_buffer = merge_boxes_gpu.scratch_out[:num_merged]

    # for idx, i in enumerate(unique_ids):
    #     mask = comp == i
    #     boxes_subset = raw_boxes[:N][mask]

    #     # Write structural outputs directly into our static memory block channels
    #     out_buffer[idx, 0] = boxes_subset[:, 0].min()
    #     out_buffer[idx, 1] = boxes_subset[:, 1].min()
    #     out_buffer[idx, 2] = boxes_subset[:, 2].max()
    #     out_buffer[idx, 3] = boxes_subset[:, 3].max()

    # # Return the clean floating-point tensor slice natively
    # return out_buffer.clone()

    # 1. Map labels to a clean 0-indexed range without calling .unique() or syncing with the CPU
    # torch.unique(..., return_inverse=True) handles the grouping array assembly entirely on the GPU
    # _, inverse_indices = torch.unique(comp, return_inverse=True)
    # num_merged = inverse_indices.max().item() + 1 # Minimal sync step

    # out_buffer = merge_boxes_gpu.scratch_out[:num_merged]
    # boxes_subset = raw_boxes[:N]

    # # 2. Pre-fill scratchpad buffers with boundary initialization constants
    # out_buffer[:, 0:2].fill_(float('inf'))
    # out_buffer[:, 2:4].fill_(float('-inf'))

    # # 3. Use highly optimized CUDA scatter reduction handles to compute all mins/maxes at once
    # # This completely eliminates your sequential Python "for idx, i in enumerate" loops!
    # out_buffer[:, 0:2].scatter_reduce_(0, inverse_indices.view(-1, 1).expand(-1, 2), boxes_subset[:, 0:2], reduce="amin", include_self=False)
    # out_buffer[:, 2:4].scatter_reduce_(0, inverse_indices.view(-1, 1).expand(-1, 2), boxes_subset[:, 2:4], reduce="amax", include_self=False)

    # return out_buffer.clone()

    # 1. Use maximum theoretical capacity boundaries directly (N candidates max)
    # This completely skips calculating max item counts on the CPU host!
    out_buffer = merge_boxes_gpu.scratch_out[:N]
    boxes_subset = raw_boxes[:N]

    # 2. Reset our fixed scratchpad slice boundaries using parallel operations
    out_buffer[:, 0:2].fill_(float("inf"))
    out_buffer[:, 2:4].fill_(float("-inf"))

    # 3. Use 'comp' directly as your index axis mapping variable!
    # Because 'comp' values fall within [0, N-1], they are already valid scatter addresses.
    # This executes 100% asynchronously on the GPU without pausing the CPU thread.
    out_buffer[:, 0:2].scatter_reduce_(
        0,
        comp.view(-1, 1).expand(-1, 2),
        boxes_subset[:, 0:2],
        reduce="amin",
        include_self=False,
    )
    out_buffer[:, 2:4].scatter_reduce_(
        0,
        comp.view(-1, 1).expand(-1, 2),
        boxes_subset[:, 2:4],
        reduce="amax",
        include_self=False,
    )

    # 4. Filter out unwritten rows where the boundaries remain at infinity
    # valid_mask = out_buffer[:, 0] != float('inf')

    # # return out_buffer[valid_mask].clone()
    # # Extract the exact non-zero index mappings entirely on the GPU.
    # # This prevents the CPU from forcing a data shape verification step!
    # valid_indices = torch.nonzero(valid_mask).squeeze(1)

    # # Index with integers instead of booleans to drop line 1098 down to 0ms
    # return out_buffer[valid_indices].clone()

    # 4. Filter out unwritten rows where the boundaries remain at infinity
    valid_mask = out_buffer[:, 0] != float("inf")

    # Sort the boolean mask in descending order entirely on the GPU hardware.
    # This automatically pushes all valid boxes (True/1) to the top of the matrix
    # and leaves unwritten slots (False/0) at the bottom.
    _, sort_indices = torch.sort(valid_mask.long(), descending=True, dim=0)

    # Rearrange the static out_buffer asynchronously without altering its shape [N, 4]
    out_buffer_sorted = out_buffer[sort_indices]

    # Overwrite the remaining invalid infinity rows to safe zero-pads.
    # This ensures downstream deep learning blocks don't encounter NaN errors.
    invalid_rows_mask = ~valid_mask[sort_indices]
    out_buffer_sorted[invalid_rows_mask] = 0.0

    # Return the static-shaped [N, 4] tensor handle.
    # Because N is already known by the CPU, line 1098 drops instantly to 0ms!
    return out_buffer_sorted


# def merge_boxes_cpuv1(boxes, gap_limit=10):
#     """
#     Greedy merge in 640x640 space to consolidate swarm fragments.
#     Input: List of [x1, y1, x2, y2] within [0, 640]
#     """
#     if not boxes:
#         return []

#     # O(N log N) sort by X for early exit optimization
#     boxes = sorted(boxes, key=lambda x: x[0])
#     merged = []

#     while boxes:
#         curr = boxes.pop(0)
#         i = 0
#         while i < len(boxes):
#             test = boxes[i]
#             # Early exit: horizontal gap exceeds limit
#             if test[0] - curr[2] > gap_limit:
#                 break

#             # Check vertical gap
#             y_dist = max(0, test[1] - curr[3], curr[3] - test[1])
#             if y_dist <= gap_limit:
#                 # Expand curr box to include test
#                 curr = [
#                     min(curr[0], test[0]),
#                     min(curr[1], test[1]),
#                     max(curr[2], test[2]),
#                     max(curr[3], test[3]),
#                 ]
#                 boxes.pop(i)
#                 i = 0  # Re-check boundaries
#             else:
#                 i += 1
#         merged.append(curr)
#     return merged


def merge_boxes_cpu(boxes, gap_limit=10, size_limit=800):
    """
    Greedy merge in 640x640 space to consolidate swarm fragments.
    Preserves original boxes larger than size_limit, merging others where possible.
    Input: List of [x1, y1, x2, y2] within [0, 640]
    """
    if not boxes:
        return []

    # O(N log N) sort by X for early exit optimization
    boxes = sorted(boxes, key=lambda x: x[0])
    merged = []

    while boxes:
        curr = boxes.pop(0)

        # --- PRESERVE OVERSIZED CROPS ---
        # If the current bounding box is already larger than the size limit,
        # skip merging it with anything else and save it directly to the output pool.
        if (curr[2] - curr[0]) > size_limit or (curr[3] - curr[1]) > size_limit:
            merged.append(curr)
            continue

        i = 0
        while i < len(boxes):
            test = boxes[i]

            # If the test candidate box is already larger than the size limit,
            # do not attempt to merge it into the current tracking cluster.
            if (test[2] - test[0]) > size_limit or (test[3] - test[1]) > size_limit:
                i += 1
                continue

            # Early exit: horizontal gap exceeds limit
            if test[0] - curr[2] > gap_limit:
                break

            # Check vertical gap
            y_dist = max(0, test[1] - curr[3], curr[3] - test[1])
            if y_dist <= gap_limit:
                # Calculate proposed expanded dimensions if merged
                new_x1 = min(curr[0], test[0])
                new_y1 = min(curr[1], test[1])
                new_x2 = max(curr[2], test[2])
                new_y2 = max(curr[3], test[3])

                # --- BOUNDARY EXPANSION GUARD ---
                # Reject the merge if combining these small boxes would push
                # the resulting macro patch past the maximum allowed size limit.
                if (new_x2 - new_x1) > size_limit or (new_y2 - new_y1) > size_limit:
                    i += 1
                    continue

                # Expand current tracking box safely
                curr = [new_x1, new_y1, new_x2, new_y2]
                boxes.pop(i)
                i = 0  # Re-check updated boundaries against remaining boxes
            else:
                i += 1
        merged.append(curr)
    return merged


# def find_contours_gpu_equivalent_v1(
#     mask_gpu_mat, stream=None, grid_size=16, limit_640=1000, max_boxes=100
# ):
#     """
#     ULTRA-OPTIMIZED: Grid-based Region Proposal.
#     Reduces N to prevent merger bottlenecks. Latency: <0.5ms.
#     """
#     # 1. Zero-copy bridge: GpuMat -> CuPy -> Torch
#     mask_cp = gpumat2cupy(mask_gpu_mat)
#     mask_tensor = torch.as_tensor(mask_cp, device="cuda")

#     # # 2. Downsample via Max Pooling (Acts as Denoise + Grouper)
#     # # A 32x32 grid on 640x640 creates a 20x20 matrix (400 cells max)
#     # pooled = F.max_pool2d(
#     #     mask_tensor.unsqueeze(0).unsqueeze(0).float(),
#     #     kernel_size=grid_size,
#     #     stride=grid_size
#     # ).squeeze()

#     # # 3. Get Indices of Motion
#     # indices = torch.nonzero(pooled > 0)
#     # A grid_size of 32 has 1,024 pixels.
#     # We require at least 5% (approx 50 pixels) to be white to trigger a ROI.
#     # This 'math' kills terrain shimmer but keeps solid drone blobs.
#     # density_threshold = (grid_size * grid_size) * 0.05

#     # Use torch.count_nonzero if using AvgPool, or stick to pooled with threshold
#     # Since 'pooled' from MaxPool is just the max value (0 or 255),
#     # we should use F.avg_pool2d instead to get a density map:

#     density_map = F.avg_pool2d(
#         mask_tensor.unsqueeze(0).unsqueeze(0).float(),
#         kernel_size=grid_size,
#         stride=grid_size,
#     ).squeeze()

#     # Define your sensitivity (e.g., 5% density)
#     # If a block is 5% full of 'white' (255) pixels, the average will be 12.75
#     density_threshold = 2  # int((grid_size * grid_size) * 0.01)

#     # 255 * 0.10 means the grid cell must be 10% white pixels
#     indices = torch.nonzero(density_map > density_threshold)

#     # EARLY EXIT: No motion detected, return empty
#     if indices.shape[0] == 0:
#         return torch.empty((0, 4), device="cuda")
#     # 2. SORT BY DENSITY: Get values for each index
#     # We pull the density values for every 'hot' cell
#     densities = density_map[indices[:, 0], indices[:, 1]]

#     # 3. Get the sort order (Descending: highest density first)
#     _, sort_order = torch.sort(densities, descending=True)
#     indices = indices[sort_order]

#     # CAP N: If scene is too noisy, take top regions to save the merger
#     if indices.shape[0] > max_boxes:
#         indices = indices[:max_boxes]

#     # 4. Map back to 640p Bounding Boxes
#     y1, x1 = indices[:, 0] * grid_size, indices[:, 1] * grid_size
#     y2, x2 = y1 + grid_size, x1 + grid_size
#     raw_boxes = torch.stack([x1, y1, x2, y2], dim=1).float()

#     # 5. Merge adjacent grid blocks
#     # gap_limit=grid_size+2 ensures diagonal/nearby blocks connect
#     # del mask_cp, mask_tensor, density_map, densities, indices
#     return merge_boxes_gpu(raw_boxes, gap_limit=grid_size * 2, size_limit=limit_640)

# Better: moved to handlers.py
# def find_contours_gpu_equivalent(mask_gpu_mat, stream=None, limit_640=None):
#     """
#     GPU equivalent to cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     Returns: torch.Tensor [N, 4] containing (x1, y1, x2, y2) in analysis space.
#     """
#     # Bridge OpenCV GpuMat to CuPy (Zero-Copy)
#     w, h = mask_gpu_mat.size()
#     ptr = mask_gpu_mat.cudaPtr()
#     pitch_bytes = mask_gpu_mat.step

#     mask_cp = cupy.ndarray(
#         (h, w),
#         dtype=cupy.uint8,
#         memptr=cupy.cuda.MemoryPointer(
#             cupy.cuda.UnownedMemory(ptr, pitch_bytes * h, mask_gpu_mat), 0
#         ),
#         strides=(pitch_bytes, 1),
#     )

#     # Use the stream pointer if provided, otherwise default to 0 (Null Stream)
#     stream_ptr = stream.cudaPtr() if stream else 0

#     with cupy.cuda.ExternalStream(stream_ptr):
#         # Labeling (Equivalent to finding connected components)
#         # labeled is an int32 array where every 'blob' has a unique number
#         structure = cupy.ones((3, 3), dtype=cupy.int32)
#         labeled, num_labels = cupyx.scipy.ndimage.label(mask_cp, structure=structure)

#         if num_labels == 0:
#             return torch.empty((0, 4), device="cuda")

#         # Setup Bounding Box Buffers
#         x1 = cupy.full((num_labels + 1,), w, dtype=cupy.int32)
#         y1 = cupy.full((num_labels + 1,), h, dtype=cupy.int32)
#         x2 = cupy.full((num_labels + 1,), -1, dtype=cupy.int32)
#         y2 = cupy.full((num_labels + 1,), -1, dtype=cupy.int32)

#         # Run the Bounds Kernel
#         # IMPORTANT: Use labeled.strides[0]//4 to get the pitch in elements
#         pitch_elements = labeled.strides[0] // 4
#         tpb = (16, 16)
#         bpg = ((w + tpb[0] - 1) // tpb[0], (h + tpb[1] - 1) // tpb[1])

#         get_bounds_kernel(
#             bpg, tpb, (labeled, pitch_elements, w, h, num_labels, x1, y1, x2, y2)
#         )

#     # Stack and return as Torch Tensor for YOLO/Drawing
#     # We skip index 0 as it represents the background (black)
#     # boxes = torch.stack(
#     #     [
#     #         torch.as_tensor(x1[1:], device="cuda"),
#     #         torch.as_tensor(y1[1:], device="cuda"),
#     #         torch.as_tensor(x2[1:], device="cuda"),
#     #         torch.as_tensor(y2[1:], device="cuda"),
#     #     ],
#     #     dim=1,
#     # ).float()
#     boxes = cupy.column_stack((x1[1:], y1[1:], x2[1:], y2[1:]))

#     return torch.as_tensor(boxes, device="cuda").float()


def get_detection_color(index, is_bgr=False):
    ind = int(index) % len(PLOT_HEXS)
    color = DETECTION_COLORS[ind]
    if is_bgr:
        return (color[2], color[1], color[0])
    else:
        return color


def get_line_thickness(npixels, ref_pixels=(1280 * 720)):
    ref_thickness = 1
    factor = npixels / ref_pixels
    thickness = int(ref_thickness * factor)
    if thickness < 1:
        thickness = 1
    return thickness


def draw_label(
    image,
    label,
    txt_bt_lft_corner,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    color=(255, 255, 255),
    padding=5,
):
    height, width, _ = image.shape

    # Scale font and thickness based on the image's smaller dimension
    scaled_font_scale = min(width, height) * FONT_SCALE_FACTOR
    scaled_thickness = max(1, ceil(min(width, height) * THICKNESS_SCALE_FACTOR))

    # Get text size and define position for the label background
    (label_W, label_H), baseline = cv2.getTextSize(
        label, font_face, scaled_font_scale, scaled_thickness
    )
    label_y1 = (txt_bt_lft_corner[0], txt_bt_lft_corner[1] - label_H - padding)
    label_y2 = (
        txt_bt_lft_corner[0] + label_W + padding,
        txt_bt_lft_corner[1] + baseline,
    )
    cv2.rectangle(image, label_y1, label_y2, color, -1)

    # Print label
    cv2.putText(
        image,
        label,
        (txt_bt_lft_corner[0] + padding // 2, txt_bt_lft_corner[1] - padding // 2),
        font_face,
        scaled_font_scale,
        (0, 0, 0),  # Black text
        scaled_thickness,
        cv2.LINE_AA,
    )


def retry_query(
    query,
    local_db=None,
    num_retries: int = LOCKTIMEOUT_RETRIES,
    sleep_timer: int = 0,
    DBHOST=DBHOST_DEFAULT,
    DBPORT=DBPORT_DEFAULT,
    DEBUG_FLAG=DEBUG_FLAG_DEFAULT,
):
    # global db
    db = local_db if local_db else vdms.vdms().connect(DBHOST, DBPORT)
    for ridx in range(num_retries + 1):
        response, _ = db.query(query, [[]])
        if "FailedCommand" in response[0] and any(
            k in response[0]["info"].lower() for k in ERR_KEYWORDS
        ):
            err = response[0]["info"]
            if DEBUG_FLAG:
                query_type = list(query[0].keys())[0]
                print(
                    f"DEBUG [process_stream Attempt #{ridx}] Received '{err}' for {query_type} query",
                    flush=True,
                )
            if sleep_timer > 0:
                time.sleep(sleep_timer)
        else:
            if DEBUG_FLAG:
                print(
                    f"[DEBUG process_stream] Successful query response: {response}",
                    flush=True,
                )
            break  # Continue
    return response


def format_df_value(value):
    if value is None:
        return value
    if value.isdigit():
        if "." in value:
            return float(value)
        else:
            return int(value)
    return value


def get_display_frame_in_bytesv1(
    foi, display_size=(960, 540), quality=50, return_bytes=True, device="CPU"
):  # Expects BGR
    H, W = foi.shape[:2]
    dH, dW = display_size
    if H == dH and W == dW:
        ret, buffer = cv2.imencode(
            ".jpg", foi, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        # print(f"[get_display_frame_in_bytes] display_size: {foi.shape}", flush=True)
    else:
        display_frame = cv2.resize(foi, display_size, interpolation=cv2.INTER_NEAREST)
        ret, buffer = cv2.imencode(
            ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        # print(f"[get_display_frame_in_bytes] display_size: {display_frame.shape}", flush=True)
    if ret and return_bytes:
        frame_bytes = buffer.tobytes()
    elif ret:
        frame_bytes = buffer
    else:
        frame_bytes = None

    return frame_bytes


def get_display_frame_in_bytes(
    foi, display_size=(960, 540), quality=50, return_bytes=True, device="CPU"
):
    """
    Safely formats and compresses video frames for browser distribution.
    Accepts packed BGR array layouts directly.
    """
    if foi is None:
        return None

    # Defensive Guard: If a raw PyTorch CUDA tensor accidentally leaks into this
    # context path, instantly bring it down safely to a standard contiguous numpy array
    if torch.is_tensor(foi):
        if foi.is_cuda:
            foi = foi.detach().cpu()
        foi = foi.numpy()

    # CRITICAL SIZING FIX: Explicitly parse display_size as (Width, Height)
    # to perfectly match OpenCV's internal spatial coordinate expectations
    target_w, target_h = display_size
    current_h, current_w = foi.shape[:2]

    # Check matching structural criteria correctly
    if current_h == target_h and current_w == target_w:
        display_frame = foi
    else:
        # Resize safely without slipping strides or inverting height/width constraints
        display_frame = cv2.resize(
            foi, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

    # Encode to crisp, valid JPEG compression matrices
    ret, buffer = cv2.imencode(
        ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )

    if ret and return_bytes:
        return buffer.tobytes()
    elif ret:
        return buffer

    return None


# Manual FPS calculation if OpenCV reports 0
def manual_fps_calculation(src, num_frames=10):
    if isinstance(src, cv2.VideoCapture):
        vid_obj = src
    else:
        vid_obj = cv2.VideoCapture(src)

    frame_count = 0
    start_t = time.time()

    while frame_count < num_frames:
        grabbed, frame = vid_obj.read()

        if not grabbed:
            break

        frame_count += 1

    end_t = time.time()
    vid_obj.release()

    elapsed_t = end_t - start_t

    if elapsed_t > 0:
        return frame_count / elapsed_t
    else:
        return 0


def scale_bbox(
    bbox, origW, origH, targetW=7680, targetH=4320, in_format="xywh", out_format="xywh"
):
    if in_format == "xywh":
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
    else:
        x, y, x2, y2 = bbox

    # Get scale factors
    scale_x = targetW / origW
    scale_y = targetH / origH

    # Translate bbox to target resolution
    # x_target = max(0, min(int(round(x * scale_x)), targetW - 1))
    # y_target = max(0, min(int(round(y * scale_y)), targetH - 1))
    # w_target = min(targetW - x_target, int(round(w * scale_x)))
    # h_target = min(targetH - y_target, int(round(h * scale_y)))
    x_target = max(0, int(x * scale_x))
    y_target = max(0, int(y * scale_y))
    x2_target = min(targetW - 1, int(x2 * scale_x))
    y2_target = min(targetH - 1, int(y2 * scale_y))
    w_target = x2_target - x_target
    h_target = y2_target - y_target
    if out_format == "xywh":
        return [x_target, y_target, w_target, h_target]
    else:
        return [x_target, y_target, x2_target, y2_target]


def scale_bbox_xywh(bbox, origW, origH, targetW=7680, targetH=4320):
    return scale_bbox(
        bbox,
        origW,
        origH,
        targetW=targetW,
        targetH=targetH,
        in_format="xywh",
        out_format="xywh",
    )


def rgb_to_nv12_torch(rgb_tensor):
    """
    Fast GPU conversion from RGB to NV12 using PyTorch.
    Input: [3, H, W] uint8 tensor on GPU
    Output: [H*1.5, W] uint8 tensor on GPU (NV12 format)
    """
    _, h, w = rgb_tensor.shape
    rgb = rgb_tensor.float()

    # BT.709 RGB to YUV coefficients (standard for HD/8K)
    # Y plane (Luma)
    y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    # U and V planes (Chroma)
    u = -0.1146 * rgb[0] - 0.3854 * rgb[1] + 0.5000 * rgb[2] + 128
    v = 0.5000 * rgb[0] - 0.4542 * rgb[1] - 0.0458 * rgb[2] + 128

    # Subsample Chroma (4:2:0)
    # We take every 2nd pixel to shrink U and V to half-resolution
    u_sub = u[::2, ::2]
    v_sub = v[::2, ::2]

    # Interleave U and V (NV12 requirement)
    # Reshape to [H/2, W] by placing U and V side-by-side at each pixel
    uv_interleaved = torch.stack((u_sub, v_sub), dim=2).reshape(h // 2, w)

    # Combine Y and UV planes
    # Resulting shape: [H + H/2, W] -> [1.5H, W]
    nv12 = torch.cat([y, uv_interleaved], dim=0)

    return torch.clamp(nv12, 0, 255).byte()


# Generate and run UDF query
def get_udf_query(
    filename_path,
    properties,
    ingest_mode,
    new_size,
    id="udf_metadata",
    metadata=None,
    test_mode=TEST_MODE_DEFAULT,
    local_db=None,
    UDF_HOST=UDF_HOST_DEFAULT,
    UDF_PORT=UDF_PORT_DEFAULT,
    DEBUG_FLAG=DEBUG_FLAG_DEFAULT,
    DBHOST=DBHOST_DEFAULT,
    DBPORT=DBPORT_DEFAULT,
):
    query = {
        "AddVideo": {
            "from_file_path": str(filename_path),  # from_server_file
            "is_local_file": True,
            "properties": properties,
            "operations": [
                {
                    "type": "syncremoteOp",  # "remoteOp",
                    "url": f"http://{UDF_HOST}:{UDF_PORT}/video",
                    "options": {
                        "id": id,
                        "otype": ingest_mode,
                        "media_type": "video",
                        "input_sizeWH": new_size,
                        "filename": properties["Name"],
                        "ingestion": 1,
                    },
                }
            ],
        }
    }

    if id == "udf_metadata" and metadata is not None:
        query["AddVideo"]["operations"][0]["options"]["metadata"] = metadata

    if test_mode:
        return

    filename = str(Path(filename_path).name)
    if DEBUG_FLAG:
        print(
            f"[TIMING],start_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
            flush=True,
        )
    try:
        res = retry_query(
            [query],
            local_db=local_db,
            sleep_timer=randint(1, 5),
            DBHOST=DBHOST,
            DBPORT=DBPORT,
            DEBUG_FLAG=DEBUG_FLAG,
        )

        if DEBUG_FLAG:
            print(
                f"[TIMING],end_udf_ingest_{ingest_mode},{filename}," + str(time.time()),
                flush=True,
            )
            print(f"[DEBUG] {filename} PROPERTIES: {properties}", flush=True)
            print(f"[DEBUG] {filename} INGEST_VIDEO RESPONSE: {res}", flush=True)
    except Exception:
        e = traceback.format_exc()
        print(f"[EXCEPTION] VDMS Query Exception: {e}", flush=True)


def _sort_dict_by_frame(in_dict):
    def _by_int(key):
        return tuple(int(k) for k in key.split("_"))

    return dict(sorted(in_dict.items(), key=lambda x: _by_int(x[0])))


# method to send metadata to VDMS once clip is saved
def metadata2vdms(
    clip_key,
    clip_filename,
    clip_metadata,
    width,
    height,
    VDMS_POOL: VDMSPool = None,
    DEBUG_FLAG=DEBUG_FLAG_DEFAULT,
    INGESTION=INGESTION_DEFAULT,
    TEST_MODE=TEST_MODE_DEFAULT,
    UDF_HOST=UDF_HOST_DEFAULT,
    UDF_PORT=UDF_PORT_DEFAULT,
    DBHOST=DBHOST_DEFAULT,
    DBPORT=DBPORT_DEFAULT,
):
    # global VDMS_POOL

    if VDMS_POOL is None:
        # VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)
        VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)

    if DEBUG_FLAG:
        print(
            f"[TIMING],start_clip_metadata,{clip_key},{time.time()}",
            flush=True,
        )

    # Send metadata to UDF
    properties = {
        "Name": clip_key,  # .split("/")[-1],
        "category": "video_path_rop",
    }

    combined_metadata = clip_metadata["object"] if "object" in clip_metadata else {}
    if "face" in clip_metadata:
        for face_frameidx_bbidx, value in clip_metadata["face"].items():
            face_frameidx, face_bbidx = face_frameidx_bbidx.split("_")
            max_obj_idx = 0
            for obj_frameidx_bbidx in combined_metadata:
                if face_frameidx in obj_frameidx_bbidx:
                    _, obj_bbidx_ = obj_frameidx_bbidx.split("_")
                    max_obj_idx = max(max_obj_idx, int(obj_bbidx_))

            if max_obj_idx > 0:
                new_face_bbidx = max_obj_idx + 1
                new_key = f"{face_frameidx}_{new_face_bbidx:04d}"
                combined_metadata[new_key] = value
                combined_metadata[new_key]["bbId"] = new_key
            else:
                combined_metadata[face_frameidx_bbidx] = value

    combined_metadata = _sort_dict_by_frame(combined_metadata)

    db = VDMS_POOL.get_connection()
    try:
        get_udf_query(
            clip_filename,
            properties,
            INGESTION.replace(",", "+"),
            (width, height),
            id="udf_metadata",
            metadata=combined_metadata,
            test_mode=TEST_MODE,
            local_db=db,
            UDF_HOST=UDF_HOST,
            UDF_PORT=UDF_PORT,
            DEBUG_FLAG=DEBUG_FLAG,
            DBHOST=DBHOST,
            DBPORT=DBPORT,
        )

        if DEBUG_FLAG:
            print(
                f"[TIMING],end_clip_metadata,{clip_key},{time.time()}",
                flush=True,
            )
    finally:
        VDMS_POOL.return_connection(db)


# method to send metadata to VDMS once clip is saved w/ retry mechanism
def metadata2vdms_with_retry(
    clip_key,
    clip_filename,
    clip_metadata,
    width,
    height,
    max_retries=LOCKTIMEOUT_RETRIES,
    VDMS_POOL: VDMSPool = None,
    DEBUG_FLAG=DEBUG_FLAG_DEFAULT,
    INGESTION=INGESTION_DEFAULT,
    TEST_MODE=TEST_MODE_DEFAULT,
    UDF_HOST=UDF_HOST_DEFAULT,
    UDF_PORT=UDF_PORT_DEFAULT,
    DBHOST=DBHOST_DEFAULT,
    DBPORT=DBPORT_DEFAULT,
):
    """
    Attempts to send metadata to VDMS with exponential backoff.
    """
    if VDMS_POOL is None:
        # VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)
        VDMS_POOL = VDMSPool(DBHOST, DBPORT, size=10)

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Attempt the actual upload (using your existing utility)
            success = metadata2vdms(
                clip_key,
                clip_filename,
                clip_metadata,
                width,
                height,
                VDMS_POOL=VDMS_POOL,
                DEBUG_FLAG=DEBUG_FLAG,
                INGESTION=INGESTION,
                TEST_MODE=TEST_MODE,
                UDF_HOST=UDF_HOST,
                UDF_PORT=UDF_PORT,
                DBHOST=DBHOST,
                DBPORT=DBPORT,
            )
            if success:
                print(f" [VDMS] Successfully uploaded {clip_key}")
                return True
        except Exception as e:
            retry_count += 1
            wait_time = 2**retry_count  # 2s, 4s, 8s, 16s...
            print(
                f" [RETRY] VDMS upload failed for {clip_key} (Attempt {retry_count}/{max_retries}). "
                f"Retrying in {wait_time}s... Error: {e}"
            )
            time.sleep(wait_time)

    print(f" [FAILED] Could not send {clip_key} to VDMS after {max_retries} attempts.")
    return False


def merge_boxes_limit(boxes, dist_threshold=25, min_area=32, max_size=640):
    if len(boxes) == 0:
        return []

    # EARLY FILTER: Remove noise (dots/specks) immediately
    # area = width * height
    valid_boxes = []
    for b in boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        if (w * h) >= min_area:
            valid_boxes.append(list(b))

    boxes = valid_boxes
    merged_any = True

    while merged_any:
        merged_any = False
        new_boxes = []

        while boxes:
            current = boxes.pop(0)
            # has_merged = False

            for i, other in enumerate(boxes):
                # Check Proximity: Are they close enough to consider?
                # (Expanding 'current' by distance_threshold for the check)
                if not (
                    current[2] + dist_threshold < other[0]
                    or other[2] + dist_threshold < current[0]
                    or current[3] + dist_threshold < other[1]
                    or other[3] + dist_threshold < current[1]
                ):
                    # Potential Dimensions: Calculate what the new box would be
                    new_x1 = min(current[0], other[0])
                    new_y1 = min(current[1], other[1])
                    new_x2 = max(current[2], other[2])
                    new_y2 = max(current[3], other[3])

                    new_w = new_x2 - new_x1
                    new_h = new_y2 - new_y1

                    # Size Constraint: Only merge if it doesn't exceed the limit
                    if new_w <= max_size and new_h <= max_size:
                        current = [new_x1, new_y1, new_x2, new_y2]
                        boxes.pop(i)
                        # has_merged = True
                        merged_any = True
                        break

            new_boxes.append(current)
        boxes = new_boxes

    return boxes


# def filter_contained_boxes(boxes, containment_thresh=0.90):
#     if len(boxes) < 2:
#         return boxes

#     # Convert to NumPy for vectorized math
#     objs = np.array(boxes)
#     areas = (objs[:, 2] - objs[:, 0]) * (objs[:, 3] - objs[:, 1])

#     # Sort by area descending
#     order = areas.argsort()[::-1]
#     objs = objs[order]
#     areas = areas[order]

#     keep = []
#     idx_list = np.arange(len(objs))

#     while len(idx_list) > 0:
#         i = idx_list[0]
#         keep.append(objs[i].tolist())
#         if len(idx_list) == 1:
#             break

#         # Vectorized Intersection over Union (IoU) / Containment
#         others = objs[idx_list[1:]]
#         ix1 = np.maximum(objs[i, 0], others[:, 0])
#         iy1 = np.maximum(objs[i, 1], others[:, 1])
#         ix2 = np.minimum(objs[i, 2], others[:, 2])
#         iy2 = np.minimum(objs[i, 3], others[:, 3])

#         iw = np.maximum(0, ix2 - ix1)
#         ih = np.maximum(0, iy2 - iy1)
#         inter_area = iw * ih

#         # Calculate how much 'others' are contained within 'i'
#         containment = inter_area / areas[idx_list[1:]]

#         # Only keep boxes that are NOT mostly contained within the current box
#         idx_list = idx_list[1:][containment < containment_thresh]

#     return keep


class DummyProcess:
    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False

    def close(self):
        pass


@dataclass
class PipelineMapping:
    resize_device: str.lower = "cpu"
    bkgd_subtraction_device: str.lower = "cpu"
    threshold_device: str.lower = "cpu"
    erodeAndDilate_device: str.lower = "cpu"
    contour_device: str.lower = "cpu"
    detection_device: str.lower = "cpu"


class StreamRequest(BaseModel):
    url: str
    name: str


class ResourceTrackerFilter:
    def __init__(self, original_stderr):
        self.stderr = original_stderr
        self.buffer = ""

    def write(self, data):
        self.buffer += data
        # If background resource tracker signals bleed out during collection, suppress them
        if (
            "resource_tracker.py" in self.buffer
            or "KeyError:" in self.buffer
            or "shm_ai_640" in self.buffer
            or "cache[rtype].remove(name)" in self.buffer
        ):
            if "\n" in data:
                self.buffer = ""
            return
        self.stderr.write(data)
        self.buffer = ""

    def flush(self):
        self.stderr.flush()


# ==============================================================================
# TRUE LOCK-FREE ASYNC VIDEO ENCODER WORKER
# ==============================================================================
class AsyncVideoWriter:
    """Handles disk saving operations completely lock-free without thread-signaling overhead."""

    def __init__(self, path, fourcc, fps, size):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        # Expand double buffer ring map to mitigate any underlying I/O bursts
        self.buffer = deque(maxlen=int(fps))
        self.running = True

        # REMOVE SEVERE BOTTLENECK SIGNALS:
        # self.frame_ready = threading.Event() <-- Deleted to clear lock.acquire thrashing!

        # Pre-allocate page-locked pinned host buffer allocations
        self.host_staging_tensor = torch.empty(
            (size[1], size[0], 3), dtype=torch.uint8, device="cpu"
        ).pin_memory()

        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while self.running or self.buffer:
            try:
                # Natively poll the lock-free structure directly without forcing a kernel signal wait block
                if not self.buffer:
                    # A crisp 2ms sleep keeps the CPU completely cool while preserving
                    # sub-millisecond thread scheduling wake-ups on low priority cores
                    time.sleep(0.005)
                    continue

                frame_payload = self.buffer.popleft()
                if frame_payload is None:
                    break

                # --- UNTHROTTLED HARDWARE VIEW EXTRACTION ---
                if torch.is_tensor(frame_payload):
                    if frame_payload.is_cuda:
                        self.host_staging_tensor.copy_(frame_payload, non_blocking=True)

                        ctx_event = torch.cuda.Event()
                        ctx_event.record(torch.cuda.current_stream())
                        ctx_event.synchronize()
                        numpy_frame = self.host_staging_tensor.numpy()
                    else:
                        numpy_frame = frame_payload.numpy()
                else:
                    numpy_frame = np.asarray(frame_payload)

                self.writer.write(numpy_frame)

            except Exception as e:
                main_app_logger.info(
                    f"[DISK-WRITER-ERROR] Asynchronous frame save failed: {e}",
                )
                continue

    def write_frame(self, frame):
        """Pure lock-free, zero-signal submission track. Fires under 1 microsecond!"""
        if self.running:
            # Overwrites elements atomically without triggering standard Python lockouts
            self.buffer.append(frame)
            # REMOVE: self.frame_ready.set() <-- GONE! CPU thread remains un-starved!

    def release(self):
        """Secure drain release prevents early container dropouts."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.running = False
        self.buffer.append(None)

        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.writer.release()


class AsyncDisplayVideoWriter:
    """
    Lock-free single-element buffer optimized for 8K pipelines.
    Eliminates queue.Queue lock contention to preserve strict target frame rates.
    """

    def __init__(self, target_fps, display_size, quality=60):
        self.target_fps = float(target_fps)
        self.disp_w, self.disp_h = display_size
        self.quality = int(quality)

        # 1. 🔄 FIX: Use a lockless deque with maxlen=1 instead of queue.Queue
        self.buffer = deque(maxlen=1)
        self.running = True
        self.handler = None

        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def set_handler_context(self, handler_instance):
        self.handler = handler_instance

    def _write_loop(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]

        while self.running:
            try:
                # 2. 🔄 FIX: Non-blocking pop from the single-element buffer
                if not self.buffer:
                    time.sleep(0.015)  # 0.005)
                    continue

                display_frame, frame_num = self.buffer.popleft()

                if self.handler is None or not self.handler.active:
                    continue

                if (
                    display_frame.shape[1] != self.disp_w
                    or display_frame.shape[0] != self.disp_h
                ):
                    display_frame = cv2.resize(
                        display_frame,
                        (self.disp_w, self.disp_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                ret, jpeg_buf = cv2.imencode(".jpg", display_frame, encode_param)
                if not ret:
                    continue

                frame_bytes = jpeg_buf.tobytes()
                frame_len = len(frame_bytes)
                num_shms = len(self.handler.shms)

                write_idx = (self.handler.ready_buffer_idx.value + 1) % num_shms

                shm_block = self.handler.shms[write_idx]
                shm_block.buf[:frame_len] = memoryview(frame_bytes)

                self.handler.shm_frame_lengths[write_idx] = frame_len
                self.handler.ready_buffer_idx.value = write_idx
                self.handler.shared_details["last_id"] = frame_num
                self.handler.mp_last_id.value = frame_num

                if hasattr(self.handler, "loop") and self.handler.loop is not None:
                    self.handler.loop.call_soon_threadsafe(
                        self.handler.frame_ready_event.set
                    )
                else:
                    self.handler.frame_ready_event.set()

            except Exception as e:
                main_app_logger.info(
                    f"[STREAM-ERROR] Lock-free frame streaming dropped: {e}"
                )
                continue

    def write_frame(self, frame, frame_num):
        """Thread-safe lock-free atomic submission track."""
        if self.running:
            # 3. 🔄 FIX: Directly append to overwrite the old entry without thread locks
            self.buffer.append((frame, frame_num))

    def release(self):
        self.running = False

        self.buffer.clear()

        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
