import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tensorrt as trt

sys.path.insert(1, str(Path(__file__).parent.parent))
from include.default_configs import ENABLE_QUERYING_DEFAULT
from include.utils import PipelineConfig, get_freest_gpu, str2bool
from torch import cuda
from ultralytics import YOLO
from ultralytics.utils.checks import check_requirements


# OBJECT DETECTION
def build_engine(
    onnx_path,
    engine_path,
    profile=[(1, 3, 32, 32), (8, 3, 640, 640), (100, 3, 640, 640)],
    metadata=None,
):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    tensor_name = "images"

    min_prof, opt_prof, max_pro = profile
    prof = builder.create_optimization_profile()
    prof.set_shape(tensor_name, min_prof, opt_prof, max_pro)
    config.add_optimization_profile(prof)

    config.set_flag(trt.BuilderFlag.FP16)
    # Ensure sufficient workspace for 8K operations
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 << 30)  # 2GB

    # Sometimes TensorRT fails because it restricts which software libraries (like cuDNN or cuBLAS) it can use.
    # Force it to search all available implementations
    tactic_sources = (
        1 << int(trt.TacticSource.CUBLAS)
        | 1 << int(trt.TacticSource.CUDNN)
        | 1 << int(trt.TacticSource.CUBLAS_LT)
    )
    config.set_tactic_sources(tactic_sources)

    print("Building engine... this will take some time.")
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        if metadata:
            meta_string = json.dumps(metadata)
            meta_bytes = meta_string.encode("utf-8")
            # Write a 4-byte little-endian signed integer indicating metadata length
            f.write(len(meta_bytes).to_bytes(4, byteorder="little", signed=True))
            # Write the raw JSON string bytes
            f.write(meta_bytes)

        f.write(serialized_engine)


def get_model(
    model_dir,
    model_name,
    run_platform,
    device_input,
    batch=100,
    force_export=False,
    sf_enabled=False,
    half_flag=True,
    dynamic_flag=True,
    model_h=640,
    model_w=640,
):
    final_model_path = f"{model_dir}/{model_name}.pt"
    pt_detection_model = YOLO(final_model_path, verbose=False, task="detect")
    label_source = []
    for k, v in pt_detection_model.names.items():
        label_source.append(v)

    if run_platform == "openvino":
        final_model_path = f"{model_dir}/{model_name}_openvino_model/"
        if not Path(final_model_path).exists() or force_export:
            pt_detection_model.export(
                format="openvino",
                half=half_flag,
                dynamic=dynamic_flag,
                device=device_input,
                # batch=batch,
                data={"names": pt_detection_model.names},
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
        final_model_path = f"{model_dir}/{model_name}.engine"
        onnx_model_path = f"{model_dir}/{model_name}.onnx"
        profile = [
            (1, 3, 32, 32),
            (8, 3, model_h, model_w),
            (batch, 3, model_h, model_w),
        ]
        if not sf_enabled:
            # Copy base model
            shutil.copy(
                f"{model_dir}/{model_name}.pt", f"{model_dir}/{model_name}_noSF.pt"
            )
            pt_detection_model = YOLO(
                f"{model_dir}/{model_name}_noSF.pt", verbose=False, task="detect"
            )
            final_model_path = f"{model_dir}/{model_name}_noSF.engine"
            onnx_model_path = f"{model_dir}/{model_name}_noSF.onnx"
            # profile = [(1, 3, 4320, 7680), (1, 3, 4320, 7680), (1, 3, 4320, 7680)]
            profile = [(1, 3, 32, 32), (1, 3, 4320, 7680), (1, 3, 4320, 7680)]

        if not Path(final_model_path).exists() or force_export:
            # pt_detection_model.export(
            #     format="engine",
            #     half=half_flag,
            #     imgsz=[640, 640],
            #     # imgsz=[7680, 4320],  # Max dimensions (8K-[W,H]-[7680,4320])
            #     dynamic=dynamic_flag,
            #     device=device_input,
            #     simplify=True,
            #     batch=batch,
            # )

            # Export to onnx
            check_requirements(
                "onnxruntime-gpu"
                if cuda.is_available() and device_input != "cpu"
                else "onnxruntime"
            )
            # onnx_model_path = f"{model_dir}/{model_name}.onnx"
            if not Path(onnx_model_path).exists() or force_export:
                pt_detection_model.export(
                    format="onnx",
                    half=half_flag,
                    dynamic=True,
                    device=device_input,
                    simplify=True,
                    data={"names": pt_detection_model.names},
                )

            if hasattr(pt_detection_model.model, "stride"):
                max_stride = int(pt_detection_model.model.stride.max().item())
            else:
                max_stride = 32  # Safe default fallback value for standard YOLO layouts

            build_engine(
                onnx_model_path,
                final_model_path,
                profile=profile,
                metadata={
                    "stride": max_stride,
                    "task": "detect",
                    "names": pt_detection_model.names,
                },
            )

        object_detection_model = YOLO(
            final_model_path,
            verbose=False,
            task="detect",
        )

    elif run_platform == "onnx":
        check_requirements(
            "onnxruntime-gpu"
            if cuda.is_available() and device_input != "cpu"
            else "onnxruntime"
        )

        final_model_path = f"{model_dir}/{model_name}.onnx"
        if not Path(final_model_path).exists() or force_export:
            pt_detection_model.export(
                format="onnx",
                half=half_flag,
                dynamic=dynamic_flag,
                device=device_input,
                simplify=True,
                batch=batch,
                data={"names": pt_detection_model.names},
            )

        object_detection_model = YOLO(final_model_path, verbose=False, task="detect")

    elif run_platform == "pytorch":
        object_detection_model = pt_detection_model
        if device_input != "cpu":
            object_detection_model.to("cuda")
        else:
            object_detection_model.to(device_input)

    else:
        raise ValueError(f"[!] Model for {run_platform} is not implemented.")

    return object_detection_model, final_model_path, label_source


# FACE DETECTION
def get_models_from_list(lst_path):
    """Reads model names from the .lst file, ignoring comments."""
    if not os.path.exists(lst_path):
        return []
    with open(lst_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def verify_and_download(
    models_lst="/home/resources/models/models.lst",
    output_dir="/home/resources/models",
    precisions="FP16",
):
    # Check which models are missing
    model_names = get_models_from_list(models_lst)
    missing_models = []

    for model in model_names:
        # Check standard OpenVINO subfolders: 'public' and 'intel'
        public_path = os.path.join(output_dir, "public", model)
        intel_path = os.path.join(output_dir, "intel", model)

        if not (os.path.exists(public_path) or os.path.exists(intel_path)):
            missing_models.append(model)

    # Only run downloader if models are missing
    if not missing_models:
        print("All models already exist in the output directory. Skipping download.")
        return

    print(f"Missing models: {missing_models}. Starting download...")

    command = [
        "omz_downloader",
        "--list",
        models_lst,
        "-o",
        output_dir,
        "--precisions",
        precisions,
    ]

    try:
        subprocess.run(command, check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Download failed with exit code {e.returncode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        dest="output_path",
        help="Path to print json of model classes",
    )
    args = parser.parse_args()

    config = PipelineConfig(
        CODE_DIR=os.getenv("CODE_DIR", "/home"),
        CUSTOM_MODEL_FLAG=str2bool(os.getenv("CUSTOM_MODEL_FLAG", False)),
        DBHOST=os.getenv("DBHOST", "vdms-service"),
        DEBUG=os.getenv("DEBUG", "0"),
        DEVICE=os.getenv("DEVICE", "CPU"),
        ENABLE_QUERYING=os.getenv("ENABLE_QUERYING", ENABLE_QUERYING_DEFAULT),
        INGESTION=os.getenv("INGESTION", "object"),
        MODEL_NAME=os.getenv("MODEL_NAME", "yolo11n"),
        OMIT_DETECTIONS_FLAG=str2bool(os.getenv("OMIT_DETECTIONS_FLAG", False)),
        # RESIZE_FLAG=str2bool(os.getenv("RESIZE_FLAG", False)),
        SHARED_MODEL=os.getenv("SHARED_MODEL", False),
        SHARED_OUTPUT=os.getenv("SHARED_OUTPUT", "/var/www/mp4"),
        TEST_MODE=str2bool(os.getenv("TEST_MODE", False)),
        TMP_LOCATION=os.getenv("TMP_LOCATION", "/var/www/cache"),
        UDF_HOST=os.getenv("UDF_HOST", "udf-service"),
        UDF_PORT=5011,
    )
    if config.CUSTOM_MODEL_FLAG:
        dir_path = f"{config.CODE_DIR}/resources/models/ultralytics/custom_models"
    else:
        dir_path = f"{config.CODE_DIR}/resources/models/ultralytics/{config.MODEL_NAME}/{config.MODEL_PRECISION}"

    if config.DEVICE == "GPU":
        best_gpu_index = get_freest_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_index)
        # Force PyTorch to initialize the CUDA context
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            # print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # EXPORT_BATCH_SIZE = 64  # 32, 50  # int(os.environ.get("GPU_BATCH_SIZE", 1))
        run_platform_name = "engine"
        print("[!] USING GPU & TENSORRT")
    else:
        # EXPORT_BATCH_SIZE = int(os.environ.get("CPU_BATCH_SIZE", 1))
        run_platform_name = "openvino"
        print("[!] USING CPU & OPENVINO")

    # Download models if it doesn't exist
    if "object" in config.INGESTION:
        _, _, classes = get_model(
            Path(dir_path),
            config.MODEL_NAME,
            run_platform_name,
            config.device_input,
            batch=config.MODEL_MAX_BATCH_SIZE,
            force_export=False,
            sf_enabled=True,
            half_flag=True,
            dynamic_flag=True,
        )
        if args.output_path is not None:
            import json

            with open(args.output_path, "w") as f:
                json.dump({"classes": classes}, f, indent=4)

    if "face" in config.INGESTION:
        verify_and_download(
            models_lst=f"{config.CODE_DIR}/resources/models/models.lst",
            output_dir=f"{config.CODE_DIR}/resources/models",
            precisions=config.MODEL_PRECISION,
        )
