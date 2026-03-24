import logging
import shutil
import sys
from pathlib import Path

from colorlog import ColoredFormatter

# Model Variables
DETECTION_THRESHOLD = 0.25
DYNAMIC_FLAG = True
HALF_FLAG = True
IOU_THRESHOLD = 0.5  # 0.7
MAX_DETECTIONS = 300
MODEL_W, MODEL_H = (640, 640)

from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

    def isatty(self):
        return False


def get_logger(log_filename):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 2. Configure specific library levels
    # (This controls how much detail you want from each)
    ULTRALYTICS_LOGGER.propagate = True
    ultra_logger = logging.getLogger("ultralytics")
    ultra_logger.propagate = True
    ultra_logger.setLevel(logging.INFO)
    logging.getLogger("openvino").setLevel(logging.INFO)

    # Define the format (added date to file, kept succinct for screen)
    # We color the name of the logger (e.g., kiss, stdout, main) differently
    console_formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s | %(name_log_color)s%(name)-12s%(reset)s | %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={
            "name": {
                "stdout": "purple",  # Prints will be purple
                "stderr": "red",  # Stderr will be red
                "ultralytics": "blue",  # ultralytics logs will be blue
                "openvino": "cyan",  # GEPA logs will be cyan
                "main": "white",  # Main program is white
            }
        },
        style="%",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    # console_formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")

    # Create StreamHandler (Screen)
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(console_formatter)

    # Create FileHandler (File)
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setFormatter(file_formatter)

    # 5. Add handlers to ROOT instead of specific loggers
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    # 3. Redirect STDOUT and STDERR
    # Assigning to 'main' logger is fine, but root is often easier
    sys.stdout = LoggerWriter(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger("stderr"), logging.ERROR)

    return root_logger


def copy_file(src: Path, dst: Path):
    if not dst.exists():
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            raise ValueError(f"Error occurred during copy: {e}")
    else:
        # pass
        raise FileExistsError(f"File exists: {dst}")


def convert_SynDroneVision_2_Train_Structure(ORIGINAL_DATA_DIR, DATA_DIR):
    for stage in ["train", "val", "test"]:
        (DATA_DIR / f"{stage}/images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / f"{stage}/labels").mkdir(parents=True, exist_ok=True)

        # Copy files to new folder
        for src_file in (ORIGINAL_DATA_DIR / f"images/{stage}").rglob("*.png"):
            src_label_str = str(src_file.with_suffix(".txt"))
            src_label_file = Path(src_label_str.replace("/images/", "/labels/"))

            # if src_label_file.exists():
            dest = DATA_DIR / f"{stage}/images/{src_file.parent.name}_{src_file.name}"
            copy_file(src_file, dest)

            if src_label_file.exists():
                dest_label = (
                    DATA_DIR
                    / f"{stage}/labels/{src_label_file.parent.name}_{src_label_file.name}"
                )
                copy_file(src_label_file, dest_label)
            elif Path(str(src_label_file).replace(".txt", "")).exists():
                src_label_file = Path(str(src_label_file).replace(".txt", ""))
                dest_label = (
                    DATA_DIR
                    / f"{stage}/labels/{src_label_file.parent.name}_{src_label_file.name}.txt"
                )
                copy_file(src_label_file, dest_label)
