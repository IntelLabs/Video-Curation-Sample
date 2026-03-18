import os
from pathlib import Path

from ultralytics import YOLO


def str2bool(in_val):
    if isinstance(in_val, bool):
        return in_val

    if not isinstance(in_val, str):
        raise ValueError(f"{in_val} is not a bool or string")

    if in_val.title() == "True":
        return True
    else:
        return False


model_precision_object = "FP16"
half_flag = True
dynamic_flag = True
CUSTOM_MODEL_FLAG = str2bool(os.getenv("CUSTOM_MODEL_FLAG", False))
DEVICE = os.environ.get("DEVICE", "CPU")
MODEL_NAME = os.environ.get("MODEL_NAME", "yolo11n")
if DEVICE == "GPU":
    EXPORT_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", 1))
    run_platform_name = "engine"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[!] USING GPU & TENSORRT")
else:
    EXPORT_BATCH_SIZE = int(os.environ.get("CPU_BATCH_SIZE", 1))
    run_platform_name = "openvino"
    print("[!] USING CPU & OPENVINO")


def get_model(model_dir, run_platform, device_input, batch=1, force_export=False):
    final_model_path = f"{model_dir}/{MODEL_NAME}.pt"
    pt_detection_model = YOLO(final_model_path, verbose=False, task="detect")
    if run_platform == "openvino":
        final_model_path = f"{model_dir}/{MODEL_NAME}_openvino_model/"
        if not Path(final_model_path).exists() or force_export:
            pt_detection_model.export(
                format="openvino",
                half=half_flag,
                dynamic=dynamic_flag,
                device=device_input,
                batch=batch,
            )

        object_detection_model = YOLO(
            final_model_path,
            verbose=False,
            task="detect",
        )

        # det_ov_model = core.read_model(final_model_path+"yolo11n.xml")
        # ov_config = {hints.performance_mode: hints.PerformanceMode.LATENCY}
        # if device == "GPU":
        #     ov_config["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "YES"
        # compiled_model = core.compile_model(det_ov_model, device, ov_config)
        # object_detection_model.predictor.model.ov_compiled_model = compiled_model

    elif run_platform == "engine":
        final_model_path = f"{model_dir}/{MODEL_NAME}.engine"
        if not Path(final_model_path).exists() or force_export:
            pt_detection_model.export(
                format="engine",
                half=half_flag,
                imgsz=[7680, 4320],  # Max dimensions (8K-[W,H]-[7680,4320])
                dynamic=dynamic_flag,
                device=device_input,
                simplify=True,
                batch=batch,
            )

        object_detection_model = YOLO(
            final_model_path,
            verbose=False,
            task="detect",
        )

    elif run_platform == "onnx":
        from torch import cuda
        from ultralytics.utils.checks import check_requirements

        check_requirements(
            "onnxruntime-gpu"
            if cuda.is_available() and device_input != "cpu"
            else "onnxruntime"
        )

        final_model_path = f"{model_dir}/{MODEL_NAME}.onnx"
        if not Path(final_model_path).exists() or force_export:
            pt_detection_model.export(
                format="onnx",
                half=half_flag,
                dynamic=dynamic_flag,
                device=device_input,
                simplify=True,
                batch=batch,
            )

        object_detection_model = YOLO(final_model_path, verbose=False, task="detect")

    elif run_platform == "pytorch":
        object_detection_model = pt_detection_model
        if DEVICE == "GPU":
            object_detection_model.to("cuda")
        else:
            object_detection_model.to(device_input)

    else:
        raise ValueError(f"[!] Model for {run_platform} is not implemented.")

    return object_detection_model, final_model_path


if __name__ == "__main__":
    if CUSTOM_MODEL_FLAG:
        dir_path = "/home/resources/models/ultralytics/custom_models"
    else:
        dir_path = (
            f"/home/resources/models/ultralytics/{MODEL_NAME}/{model_precision_object}"
        )

    ydir = Path(dir_path)

    device_input = DEVICE.lower() if DEVICE == "CPU" else "cuda"
    _, _ = get_model(
        ydir,
        run_platform_name,
        device_input,
        batch=EXPORT_BATCH_SIZE,
        force_export=False,
    )

    os.remove(f"{ydir}/{MODEL_NAME}.pt")
