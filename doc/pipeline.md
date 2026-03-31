# Detection using Fine-Tuned YOLO Model

This guide assumes a YOLO model or fine-tuned YOLO model is available for detection.


## Prepare Model for Application

Place your model in the appropriate directory for the application.

| Directory                                            | Description |
| ---------------------------------------------------- | ----------- |
| `fastapi/resources/models/ultralytics/custom_models` | Place any custom models in this directory. Each model file should have a unique name which will be used to deploy (and/or export) the model via the application start script. The PT model (`${UNIQUE_MODEL_NAME}.pt`) is exported to OpenVINO (`${UNIQUE_MODEL_NAME}_openvino_model/`) or TensorRT (`${UNIQUE_MODEL_NAME}.engine`), dependent on device used.  |
| `fastapi/resources/models/ultralytics/${MODEL_NAME}/FP16` | Ultralytics YOLO models are typically placed in this directory where MODEL_NAME is the short name for the model (i.e. `yolo11n`). The PT model (`${MODEL_NAME}.pt`) is exported to OpenVINO (`${MODEL_NAME}_openvino_model/`) or TensorRT (`${MODEL_NAME}.engine`), dependent on device used. |

Please note the model labels are retrieved from the model directly, so the model must contain these details.


## High Resolution Object Detection Pipeline
Resource limitation....

PLACE IMAGE OF PIPELINE


### Smart Filtering Pipeline
The Smart Filtering is a portion of the pipeline which filters high resolution videos for region of interest and the ROIs are only used in the detection phase instead of the entire frame.
This helps ......

The current implementation of the Smart Filtering pipeline is optimized for the test use-case, drone detection.
Drones are typically small in the video frames so if your use-case of interest has different objects, it may be beneficial to test the pipeline results on an existing test video for your use-case.

For this case, we provide [``]() which annotates bbs of objects identified by the Smart Filtering pipeline onto each frame of the video for visual inspection.
If you are not satisfied with the results, feel free to modify/optimize the pipeline further.

IDENTIFY OPTIMIZATION POINTS FOR EACH COMPONENT

Once satisfied with annotated results, you can proceed with running the full pipeline.


## Pipeline Deployment

DEPLLOYMENT
./stop.sh –p
./start_app.sh –e GPU –o –m <unique name>


VISUALIZATION/QUERY
# View live detection
GOTO: http://<HOST>:30077/

# View Page to Query
GOTO: http://<HOST>:30007/


STOPPING
./stop.sh –p


