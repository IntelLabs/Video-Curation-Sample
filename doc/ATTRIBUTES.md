# Media Attributions

This repository utilizes external datasets and media for training/demo purposes.
<br>


## 1. SynDroneVision Dataset
* **Source:** [Zenodo (Record 13360116)](https://zenodo.org/records/13360116)
* **License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en)
* **Citation:** See DOI [10.1109/WACV61041.2025.00742](https://doi.org/10.1109/WACV61041.2025.00742) (Lenhard et al., WACV 2025).
* **Usage:** External data for fine-tuning sample model for drone detection.  Model generated using this data is available at [Video Curation Sample Models: Drone Detection](https://github.com/IntelLabs/video-curation-sample-models/tree/main/Drone_Detection).
<br>


## 2. Sample Drone Detection Model
* **Source:** [Video Curation Sample Models: Drone Detection](https://github.com/IntelLabs/video-curation-sample-models/tree/main/Drone_Detection)
* **License:** [AGPL 3.0](https://github.com/IntelLabs/video-curation-sample-models/tree/main/Drone_Detection/LICENSE)
* **Citation:** N/A
* **Usage:** Model fine-tuned using Ultralytics Yolo11n model and SynDroneVision dataset.
<br>


## 3. [Optional] Sample Drone Video (`anduril_swarm.mp4`)
* **Source:** [droneforge/yolov11-UAV-finetune](https://github.com/droneforge/yolov11-UAV-finetune/blob/main/anduril_swarm.mp4)
* **Usage:** Open sourced sample video for pipeline demonstration.
<br>


## 4. [Optional] DUT Anti-UAV Dataset
* **Source:** [wangdongdut/DUT-Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV)
* **License:** [Apache License 2.0](https://github.com/wangdongdut/DUT-Anti-UAV/blob/master/LICENSE)
* **Citation:** See DOI [10.48550/arXiv.2205.10851](https://doi.org/10.48550/arXiv.2205.10851) (Zhao et al., IEEE T-ITS 2022).
* **Usage:** External sample video dataset ONLY used for pipeline evaluation via command: `python tests/test_eval.py --type object`. Test script ([test_eval.py](../fastapi/tests/test_eval.py)) evaluates videos 9-20 of the dataset since the camera seems stationary.
