# Dataset and Third-Party Media Attributions

This repository utilizes external datasets and media for training/demo purposes, without storing the actual files.

---

## 1. SynDroneVision Dataset
* **Source:** [Zenodo (Record 13360116)](https://zenodo.org/records/13360116)
* **License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en)
* **Citation:** See DOI [10.1109/WACV61041.2025.00742](https://doi.org/10.1109/WACV61041.2025.00742) (Lenhard et al., WACV 2025).
* **Usage:** External data for fine-tuning model for drone detection.

---

## 2. Sample Drone Video (`anduril_swarm.mp4`)
* **Source:** [droneforge/yolov11-UAV-finetune](https://github.com/droneforge/yolov11-UAV-finetune/blob/main/anduril_swarm.mp4)
* **Usage:** External sample video for pipeline demonstration.

---

## 3. [Optional] DUT Anti-UAV Dataset
* **Source:** [wangdongdut/DUT-Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV)
* **License:** [Apache License 2.0](https://github.com/wangdongdut/DUT-Anti-UAV/blob/master/LICENSE)
* **Citation:** See DOI [10.48550/arXiv.2205.10851](https://doi.org/10.48550/arXiv.2205.10851) (Zhao et al., IEEE T-ITS 2022).
* **Usage:** External sample video dataset ONLY used for pipeline evaluation via command: `python tests/test_eval.py --type object`. Test script ([test_eval.py](../fastapi/tests/test_eval.py)) evaluates videos 9-20 of the dataset since the camera seems stationary.
