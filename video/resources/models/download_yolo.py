import os
from pathlib import Path

from ultralytics import YOLO

model_precision_object = "FP16"
model_name = "yolo11"
half_flag = True
dynamic_flag = True
DEVICE = os.environ.get("DEVICE", "CPU")
if DEVICE == "GPU":
    batch_size = 1
else:
    # batch_size = 8
    batch_size = 1


def get_model(model_dir, run_platform, device_input, batch=1):
    final_model_path = f"{model_dir}/{model_name}n.pt"
    pt_detection_model = YOLO(final_model_path, verbose=False, task="detect")
    if run_platform == "openvino":
        pt_detection_model.export(
            format="openvino",
            half=half_flag,
            dynamic=dynamic_flag,
            device=device_input,
            batch=batch,
        )

        final_model_path = f"{model_dir}/{model_name}n_openvino_model/"
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
        pt_detection_model.export(
            format="engine",
            half=half_flag,
            dynamic=dynamic_flag,
            simplify=True,
            batch=batch,
        )
        # pt_detection_model.export(format='engine')  # Rohit

        final_model_path = f"{model_dir}/{model_name}n.engine"
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

        final_model_path = f"{model_dir}/{model_name}n.onnx"
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
        if device == "GPU":
            object_detection_model.to("cuda")
        else:
            object_detection_model.to(device_input)

    else:
        raise ValueError(f"[!] Model for {run_platform} is not implemented.")

    return object_detection_model, final_model_path


if __name__ == "__main__":
    ydir = Path(
        f"/home/resources/models/ultralytics/{model_name}/{model_precision_object}"
    )
    device = os.environ.get("DEVICE", "CPU")
    if device == "GPU":
        run_platform = "engine"
        print("[!] USING GPU & TENSORRT")
    else:
        run_platform = "openvino"
        print("[!] USING CPU & OPENVINO")

    device_input = device.lower() if device == "CPU" else 0
    _, _ = get_model(ydir, run_platform, device_input, batch=batch_size)

    os.remove(f"{ydir}/{model_name}n.pt")
