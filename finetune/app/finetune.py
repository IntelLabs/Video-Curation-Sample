#################################################################################
# SCRIPT IS TEST FOR FINETUNING YOLO MODEL
# This script uses ultralytics for training
# Next step is to modify without Ultralytics FW (possible?)
#
# NOTE: To keep original classes, they must be included in dataset; (2nd attempt)
#       Otherwise only new classes will be detected (1st attempt)
#################################################################################

import argparse
import gc
import os
import time
from datetime import datetime
from pathlib import Path

from include.train_args import (
    BATCH_SIZE,
    CLOSE_MOSAIC,
    IMGZ_SHAPE,
    LEARNING_RATE,
    MULTI_SCALE,
    NUM_EPOCHS,
    NUM_WORKER_THREADS,
    OPTIMIZER_NAME,
    PATIENCE,
    RECT_FLAG,
    SCALE,
    WARMUP_EPOCHS,
)
from include.utils import (
    DETECTION_THRESHOLD,
    IOU_THRESHOLD,
    convert_Dataset_2_Train_Structure,
    get_logger,
)
from torch.cuda import empty_cache
from ultralytics import YOLO

WORKSPACE = Path(__file__).parent
RESULT_DIR = (
    WORKSPACE / "SynDroneVision-Results"
)  # Default location where to store results
SYSTEM_DATA_DIR = Path(
    "/datasets"
)  # Default location where dataset directories are stored


def csv_to_int_list(value_string):
    """
    Converts a comma-separated string to a list of integers.
    Raises argparse.ArgumentError if any value is not a valid integer.
    """
    try:
        # Split the string by commas and convert each part to an integer
        return [int(item) for item in value_string.split(",")]
    except ValueError as err:
        # Raise an ArgumentTypeError to provide a clear error message to the user
        raise argparse.ArgumentTypeError(
            f"Invalid value: '{value_string}'. Values must be comma-separated integers."
        ) from err


def get_input_args():
    parser = argparse.ArgumentParser()

    """ GENERAL """
    parser.add_argument(
        "-r",
        "--result-dir",
        type=Path,
        default=RESULT_DIR,
        help=f"Directory to store any results. [Default: {RESULT_DIR.relative_to(WORKSPACE)}]",
    )
    parser.add_argument(
        "--devices",
        type=str,
        dest="devices_str",
        default="0,1",
        help="A comma-separated list of GPU devices to use. This value will be CUDA_VISIBLE_DEVICES.  [Default: 0,1]",
    )

    """ TRAINING """
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        dest="local_data_dir",
        default=SYSTEM_DATA_DIR,
        help=f"Parent directory of dataset directories. [Default: {SYSTEM_DATA_DIR}]",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SynDroneVision",
        help="Name of dataset; A directory with this name should be in --data-dir (-d). [Default: SynDroneVision]",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        dest="labels_str",
        default="drone",
        help="A comma-separated list of labels (classes) for model.  [Default: drone]",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default="drones",
        help="Name of file (<yaml-name>.yaml) with data specifications [Default: drones]",
    )
    parser.add_argument(
        "--no-train",
        action="store_false",
        dest="train_model",
        help="Skip finetune stage",
    )

    """ INFERENCE """
    parser.add_argument(
        "--test-video",
        type=str,
        help="Test video path for prediction. If not provided, inference is disabled",
    )

    args = parser.parse_args()

    # Define additional variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices_str
    args.devices = csv_to_int_list(args.devices_str)
    args.labels = args.labels_str.split(",")

    # File/Directory definition
    timestamp = datetime.now().strftime("%Y%m%d")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.project_name = args.result_dir / f"finetune_{timestamp}"
    args.log_filename = args.project_name / f"finetune_{timestamp}.log"
    args.info_file = args.project_name / f"Summary_{timestamp}.txt"
    args.project_name.mkdir(parents=True, exist_ok=True)
    args.project_name = str(args.project_name)

    # Prepare dataset for finetuning/validation/inference
    def prepare_dataset():
        args.original_data_dir = args.local_data_dir / args.dataset_name
        if all(
            not (args.original_data_dir / f"{stage}/images").exists()
            for stage in ["train", "val", "test"]
        ):
            args.data_dir = args.local_data_dir / f"{args.dataset_name}_yolo"
            if all(
                not (args.data_dir / f"{stage}/images").exists()
                for stage in ["train", "val", "test"]
            ):
                convert_Dataset_2_Train_Structure(args.original_data_dir, args.data_dir)
        else:
            args.data_dir = args.original_data_dir

        args.yaml_path = args.data_dir / f"{args.yaml_name}.yaml"

        # Generate dataset configuration for dataset
        if not args.yaml_path.exists():
            num_labels = len(args.labels)
            data_info_content = f"""
        # Dataset root directory
        path: {args.data_dir}        # dataset root dir
        train: train                  # train images relative path
        val: val                      # validation images relative path
        test: test                    # test images relative path (optional)


        # Num. of classes
        nc: {num_labels}

        # Classes
        names: {args.labels}
            """

            with open(args.yaml_path, "w") as f:
                f.write(data_info_content)

    prepare_dataset()

    return args


def main(args):
    TRAIN_RUN_NAME = "train_output"
    VAL_RUN_NAME = "val_output"
    DETECT_RUN_NAME = "predict_output"

    logger = get_logger(args.log_filename)
    logger.info(
        f"🚀 Logging initialized. Writing to screen and {args.log_filename.relative_to(WORKSPACE)}"
    )

    """ TRAIN """
    if args.train_model:
        logger.info("Running training ...")
        model = YOLO(str(args.result_dir / "yolo11n.pt"))
        start_train = time.time()
        _ = model.train(
            batch=BATCH_SIZE,
            data=args.yaml_path,
            epochs=NUM_EPOCHS,
            imgsz=IMGZ_SHAPE,
            lr0=LEARNING_RATE,
            optimizer=OPTIMIZER_NAME,
            project=args.project_name,
            name=TRAIN_RUN_NAME,
            device=args.devices,
            patience=PATIENCE,
            multi_scale=MULTI_SCALE,
            workers=NUM_WORKER_THREADS,
            rect=RECT_FLAG,
            warmup_epochs=WARMUP_EPOCHS,
            close_mosaic=CLOSE_MOSAIC,  # Turn off mosaic earlier to stabilize
            scale=SCALE,  # set scale=0.8 or higher (the default is usually 0.5)
        )
        train_time = time.time() - start_train
        logger.info(f"Training took {train_time:0.3f} secs\n")
        empty_cache()  # Frees memory no longer used
        gc.collect()  # Forces garbage collector
    else:
        if (
            Path(f"{args.project_name}/{TRAIN_RUN_NAME}/results.csv").exists()
            and not Path(f"{args.project_name}/{TRAIN_RUN_NAME}/results.png").exists()
        ):
            from ultralytics.utils.plotting import plot_results

            plot_results(file=f"{args.project_name}/{TRAIN_RUN_NAME}/results.csv")

    """ VALIDATION """
    # Check latest directory in case of multiple training runs
    idx_run = 0
    original_TRAIN_RUN_NAME = TRAIN_RUN_NAME
    for train_runs in Path(args.project_name).glob(f"{original_TRAIN_RUN_NAME}*"):
        name_idx_str = train_runs.name.replace(original_TRAIN_RUN_NAME, "")
        if name_idx_str != "" and int(name_idx_str) > idx_run:
            TRAIN_RUN_NAME = train_runs.name
            idx_run = int(name_idx_str)

    logger.info("Running validation ...")
    model = YOLO(f"{args.project_name}/{TRAIN_RUN_NAME}/weights/best.pt")
    start_val = time.time()
    val_result = model.val(
        batch=BATCH_SIZE,
        data=args.yaml_path,
        imgsz=IMGZ_SHAPE,
        conf=DETECTION_THRESHOLD,
        iou=IOU_THRESHOLD,
        split="test",
        project=args.project_name,
        name=VAL_RUN_NAME,
        workers=NUM_WORKER_THREADS,
        device=",".join([f"cuda:{d}" for d in args.devices]),
    )
    val_time = time.time() - start_val
    logger.info(f"Validation took {val_time:0.3f} secs\n")
    empty_cache()  # Frees memory no longer used
    gc.collect()  # Forces garbage collector

    """ INFERENCE """
    if args.test_video:
        logger.info("Running inference ...")
        start_detect = time.time()
        _ = model.predict(
            source=args.test_video,
            conf=DETECTION_THRESHOLD,
            iou=IOU_THRESHOLD,
            show=False,
            imgsz=IMGZ_SHAPE,
            save=True,
            project=args.project_name,
            name=DETECT_RUN_NAME,
            exist_ok=False,  # overwrite if folder exists
            device=args.devices[0],
        )
        detect_time = time.time() - start_detect
        logger.info(f"Inference took {detect_time:0.3f} secs\n")
        empty_cache()  # Frees memory no longer used
        gc.collect()  # Forces garbage collector

    """ SUMMARY """
    with open(args.info_file, "w") as f:
        """ TRAINING SUMMARY """
        if args.train_model:
            print(
                f"Training took {train_time:0.3f} secs for bs {BATCH_SIZE} and {NUM_EPOCHS} epochs",
                file=f,
            )

        """ VALIDATION SUMMARY """
        print(
            f"\n\nValidation took {val_time:0.3f} secs",
            file=f,
        )
        print("mAP50-95:", val_result.box.map, file=f)
        print("mAP50:", val_result.box.map50, file=f)
        print("mAP75:", val_result.box.map75, file=f)
        print("mAP:", val_result.box.maps, file=f)

        """ INFERENCE SUMMARY """
        if args.test_video:
            print(f"\n\nDetection took {detect_time:0.3f} secs", file=f)


if __name__ == "__main__":
    in_params = get_input_args()
    main(in_params)
