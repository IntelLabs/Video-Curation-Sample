import os
import warnings
from multiprocessing import resource_tracker

import torch

# ==============================================================================
# SUPPRESS WARNINGS

# warnings.filterwarnings(
#     "ignore", category=FutureWarning, message=".*reduce_op` is deprecated.*"
# )
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
warnings.filterwarnings("ignore", message=".*reduce_op.*")
warnings.filterwarnings("ignore", message=".*expandable_segments.*")

# ==============================================================================
# MONKEY PATCH


def _patch_resource_tracker_for_shm():
    """
    Prevents Python's resource_tracker from tracking shared memory segments.
    Eliminates BPO-39959 / Issue #82300 KeyError crashes.
    """
    _orig_register = resource_tracker.register
    _orig_unregister = resource_tracker.unregister

    def _safe_register(name, rtype):
        if rtype == "shared_memory":
            return  # Do not register shared memory with tracker daemon
        return _orig_register(name, rtype)

    def _safe_unregister(name, rtype):
        if rtype == "shared_memory":
            return  # Do not send unregister signals for shared memory
        return _orig_unregister(name, rtype)

    resource_tracker.register = _safe_register
    resource_tracker.unregister = _safe_unregister


_patch_resource_tracker_for_shm()


# Cache the hardware availability state globally.
# This prevents internal framework loops from triggering driver/NVML checks per frame.
_cuda_available = torch.cuda.is_available()


def _patched_is_available():
    return _cuda_available


torch.cuda.is_available = _patched_is_available

# ==============================================================================
# ENVIRONMENT

os.environ["OMP_NUM_THREADS"] = "4"  # "4"  #"2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Suppress low-delay reference block warnings from OpenCV/PyAV/FFmpeg
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

try:
    torch.set_num_interop_threads(1)
    torch.set_num_threads(2)
except RuntimeError:
    # Safe graceful fallback if a process-level fork duplicated context maps
    pass

# Force disable tracking states to stop graph allocation recursion
torch.set_grad_enabled(False)
