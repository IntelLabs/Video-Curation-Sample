# Fine-Tune YOLO Model

This quide provides details on how to fine-tune a YOLO model using Ultralytics on GPU ONLY.
For simplicity, the use-case for this guide is Drone Detection.
Therefore, the goal is to finetune the YOLO11n model to detect only one class (`drone`).


## Drone Detection Dataset
To prepare for fine-tuning, first identify a large dataset for the use-case.
If detecting only one class, be sure the dataset contains negative images (without object of interest).
For a well rounded dataset, be sure it contains train, validation, AND test sets to follow the expected [Ultralytics YOLO Format](https://docs.ultralytics.com/datasets/detect/).
The provided script checks if the original dataset contains `train`, `validation`, AND `test` directories.
If it does not, it proceeds with converting the dataset into this format assuming the original dataset has `images` and `labels` directory with sub-directories `train`, `validation`, AND `test`.
If your dataset does not follow this format, please modify `prepare_dataset` in `finetune.py`.

In this guide, the [SynDroneVision dataset](https://zenodo.org/records/13360116) is used.
Please see their [paper](https://ieeexplore.ieee.org/document/10943801) for more details.
The original dataset is saved in `SynDroneVision` directory and since it is not in the proper YOLO format, the script converts the data structure and saves new structure in directory `SynDroneVision_yolo`.  ***NOTE:*** In other cases, if data structure is modified, the new directory prefixes the original directory name with `_yolo`.


## Training Configurations
The configurations used for training on 2x NVIDIA A100 80GB PCIe are specified in `include/train_args.py`.
Feel free to modify these parameters based on your hardware limitations such as VRAM of GPU.


## Finetune Script
THe finetune script is used to run training, validation, and also test on a provided video (optional).
To make deployment easy, we provide a Dockerfile which has the ideal environment and allow the script to run with deployment.
The script has a few adjustible arguments, so feel free to modify the call in next section as needed.
```bash
# Runs default arguments
python finetune.py
```

THe following arguments are available:

| Argument                                        | Default                  | Description |
| ----------------------------------------------- | ------------------------ | ----------- |
| -r RESULT_DIR,<br>--result-dir RESULT_DIR       | `SynDroneVision-Results` | Directory to store any results. |
| --devices DEVICES_STR                           | `0,1`                    | A comma-separated list of GPU devices to use. This value will be CUDA_VISIBLE_DEVICES. |
| -d LOCAL_DATA_DIR,<br>--data-dir LOCAL_DATA_DIR | `/datasets`              | Parent directory of dataset directories. |
| --dataset-name DATASET_NAME                     | `SynDroneVision`         | Name of dataset; A directory with this name should be in --data-dir (-d). |
| -l LABELS_STR,<br>--labels LABELS_STR           | `drone`                  | A comma-separated list of labels (classes) for model. |
| --yaml-name YAML_NAME                           | `drones`                 | Name of file (YAML_NAME.yaml) with data specifications. Be sure the `path` in this file correlates with `LOCAL_DATA_DIR` value. |
| --no-train                                      |                          | Skip finetune stage |
| --test-video TEST_VIDEO                         |                          | Test video path for prediction. If not provided, inference is disabled |
<br>


## Deployment
Python 3.10.12 on Ubuntu 22 was used for testing.
To manually setup your environment, use the provided `requirements.txt` and `requirements.GPU.txt` files.
Be sure your system contains the required NVIDIA packages to use GPUs for training.

To avoid modifying your system for training, you can use the provided Dockerfile to deploy a container.
For easy access of host data, the `inputs` directory containing any input videos, the `finetune/app` directory containing this code, and the parent directory where datasets are stored (i.e. `/data1/datasets`) are mounted to the container.
Please see below for instructions for deploying container via `docker` and `docker compose`.
***NOTE:*** `REPO_DIR` is the path of this repo's main directory. Also be sure to update values in `.env` if using docker compose.
- **Docker:** For this option, be sure to build container first.  You can start the container and finetune script via run command.
  ```bash
  REPO_DIR=`pwd`
  LOCAL_DATA_DIR=/path/to/your/actual/data/directory

  # Build container
  cd finetune
  docker build -f Dockerfile -t lcc_finetune:latest .

  # Run finetune script default values
  docker run -it --ipc=host --gpus all \
  --name finetune_container \
  -v ${REPO_DIR}/inputs:/watch_dir \
  -v ${REPO_DIR}/finetune/app:/home \
  -v ${LOCAL_DATA_DIR}:/datasets \
  lcc_finetune:latest /bin/bash -c "python finetune.py"
  ```
- **Docker Compose:** If different arguments are needed, please update `command:` in `finetune/docker-compose.yml` prior to deployment.
  ```bash
  REPO_DIR=`pwd`

  cd finetune
  docker compose up
  ```

Once all stages are completed, stop and/or remove running container.
<!-- docker compose down --remove-orphans -->
Keep note of the latest model, as this model will be copied to different location for inclusion in full application.

