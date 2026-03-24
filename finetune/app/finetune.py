#################################################################################
# SCRIPT IS TEST FOR FINETUNING YOLO MODEL
# This script uses ultralytics for training
# Next step is to modify without Ultralytics FW (possible?)
#
# NOTE: To keep original classes, they must be included in dataset; (2nd attempt)
#       Otherwise only new classes will be detected (1st attempt)
#################################################################################

import gc
import os
import time
from pathlib import Path

from include.utils import (
    DETECTION_THRESHOLD,
    IOU_THRESHOLD,
    convert_SynDroneVision_2_Train_Structure,
    get_logger,
)
from torch.cuda import empty_cache
from ultralytics import YOLO

# WORKSPACE = Path(__file__).parent
TRAIN_MODEL = True
WORKSPACE = Path("/workspace/app")
SYSTEM_DATA_DIR = Path(
    # "/data1/dataset"
    "/workspace/dataset"
)  # Path(__file__).parent  # Where to store data
LOCAL_DATA_DIR = Path(__file__).parent / "data"  # Where data info yamls are stored
ORIGINAL_DATA_DIR = SYSTEM_DATA_DIR / "SynDroneVision"
DATA_DIR = SYSTEM_DATA_DIR / "SynDroneVision_yolo"
YAML_PATH = DATA_DIR / "drones.yaml"
RESULT_DIR = WORKSPACE / "SynDroneVision-Results"
PROJECT_NAME = str(RESULT_DIR / "finetune_revised_3.9")

TRAIN_RUN_NAME = "train_output"
DEVICES = [4, 5]  # [0, 1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in DEVICES])
BATCH_SIZE = 16  # 8  # 16
NUM_EPOCHS = 100  # 60
IMGZ_SHAPE = 1280  # 1024  # 640  #Image shape: 2560x1489 too large, using 1280
LEARNING_RATE = 0.001  # 0.001
OPTIMIZER_NAME = "AdamW"
RECT_FLAG = False  # True  # Enables minimum padding strategy; cannot use with multi-gpu training
WARMUP_EPOCHS = (
    3  # Set to 0 to prevent the learning rate from starting too low [Default: 3]
)
PATIENCE = 20  # 5  # Automatically stops training if no improvement after P epochs [Default: 100]
MULTI_SCALE = 0  # .75  #True  # Change imgsz by up to a factor of 0.5 during training to be more accurate with multiple imgsz during inference
SCALE = 0.8  # Default:0.5  This tells YOLO to zoom in significantly on your 2560px images during training, effectively creating "crops" on the fly that keep the drone closer to its original size

VAL_RUN_NAME = "val_output"
NUM_WORKER_THREADS = 2  # 4

DETECT_RUN_NAME = "predict_output"
TEST_VIDEO = "./inputs/anduril_swarm.mp4"
# TEST_VIDEO = "inputs/pexels-joseph-redfield-8459631 (1080p).mp4"

Path(PROJECT_NAME).mkdir(parents=True, exist_ok=True)

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d")
# log_filename = LOGS_DIR / f"{operation}_{timestamp}.log"
log_filename = Path(PROJECT_NAME) / f"finetune_test_{timestamp}.log"
logger = get_logger(log_filename)
logger.info(
    f"🚀 Logging initialized. Writing to screen and {log_filename.relative_to(WORKSPACE)}"
)

# Convert Data Structure to Structure of Training
if not (DATA_DIR / "train/images").exists():
    convert_SynDroneVision_2_Train_Structure(ORIGINAL_DATA_DIR, DATA_DIR)


# Generate dataset configuration for dataset
if not YAML_PATH.exists():
    data_info_content = f"""
# Dataset root directory
path: {DATA_DIR}        # dataset root dir
train: train                  # train images relative path
val: val                      # validation images relative path
test: test                    # test images relative path (optional)


# Num. of classes
nc: 1

# Classes
names: ["drone"]
    """

    with open(YAML_PATH, "w") as f:
        f.write(data_info_content)


# Create Results directory
if not RESULT_DIR.exists():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


""" TRAIN """
if TRAIN_MODEL:
    model = YOLO(RESULT_DIR / "yolo11n.pt")
    start_train = time.time()
    results = model.train(
        batch=BATCH_SIZE,
        data=YAML_PATH,
        epochs=NUM_EPOCHS,
        imgsz=IMGZ_SHAPE,
        lr0=LEARNING_RATE,
        optimizer=OPTIMIZER_NAME,
        project=PROJECT_NAME,
        name=TRAIN_RUN_NAME,
        device=DEVICES,
        patience=PATIENCE,
        multi_scale=MULTI_SCALE,
        workers=NUM_WORKER_THREADS,
        rect=RECT_FLAG,
        warmup_epochs=WARMUP_EPOCHS,
        close_mosaic=10,  # Turn off mosaic earlier to stabilize
        scale=SCALE,  # set scale=0.8 or higher (the default is usually 0.5)
    )
    train_time = time.time() - start_train
    empty_cache()  # Frees memory no longer used
    gc.collect()  # Forces garbage collector
else:
    if (
        Path(f"{PROJECT_NAME}/{TRAIN_RUN_NAME}/results.csv").exists()
        and not Path(f"{PROJECT_NAME}/{TRAIN_RUN_NAME}/results.png").exists()
    ):
        from ultralytics.utils.plotting import plot_results

        plot_results(file=f"{PROJECT_NAME}/{TRAIN_RUN_NAME}/results.csv")


""" VALIDATION """
# Check latest directory in case of multiple training runs
idx_run = 0
original_TRAIN_RUN_NAME = TRAIN_RUN_NAME
for train_runs in Path(PROJECT_NAME).glob(f"{original_TRAIN_RUN_NAME}*"):
    name_idx_str = train_runs.name.replace(original_TRAIN_RUN_NAME, "")
    if name_idx_str != "" and int(name_idx_str) > idx_run:
        TRAIN_RUN_NAME = train_runs.name
        idx_run = int(name_idx_str)

model = YOLO(f"{PROJECT_NAME}/{TRAIN_RUN_NAME}/weights/best.pt")
start_val = time.time()
val_result = model.val(
    batch=BATCH_SIZE,
    data=YAML_PATH,
    imgsz=IMGZ_SHAPE,
    conf=DETECTION_THRESHOLD,
    iou=IOU_THRESHOLD,
    split="test",
    project=PROJECT_NAME,
    name=VAL_RUN_NAME,
    workers=NUM_WORKER_THREADS,
    device=",".join([f"cuda:{d}" for d in DEVICES]),
)
val_time = time.time() - start_val
empty_cache()  # Frees memory no longer used
gc.collect()  # Forces garbage collector


""" DETECT """
start_detect = time.time()
result = model.predict(
    source=TEST_VIDEO,
    conf=DETECTION_THRESHOLD,
    iou=IOU_THRESHOLD,
    show=False,
    imgsz=IMGZ_SHAPE,
    save=True,
    project=PROJECT_NAME,
    name=DETECT_RUN_NAME,
    exist_ok=False,  # overwrite if folder exists
    device=DEVICES[0],
)
detect_time = time.time() - start_detect
empty_cache()  # Frees memory no longer used
gc.collect()  # Forces garbage collector


""" SUMMARY """
info_file = Path(PROJECT_NAME) / "Summary.txt"
with open(info_file, "w") as f:
    if TRAIN_MODEL:
        print(
            f"Training took {train_time:0.3f} secs for bs {BATCH_SIZE} and {NUM_EPOCHS} epochs",
            file=f,
        )

    print(
        f"\n\nValidation took {val_time:0.3f} secs",
        file=f,
    )
    print("mAP50-95:", val_result.box.map, file=f)
    print("mAP50:", val_result.box.map50, file=f)
    print("mAP75:", val_result.box.map75, file=f)
    print("mAP:", val_result.box.maps, file=f)

    print(f"\n\nDetection took {detect_time:0.3f} secs", file=f)
