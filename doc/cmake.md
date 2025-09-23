
## CMake Options:

Use the following definitions to customize the building process:
- **DEBUG**: Flag to enable debug messages
- **DEVICE**: Specify the device: `CPU` or `GPU`
- **DOCKER_TAR**: Flag to load docker images instead of building from Dockerfiles
<!-- - **IN_SOURCE**: Specify the input video source: `videos` and/or `stream` (comma-delimited). -->
- **INGESTION**: Specify the ingestion mode: `face` and/or `object`. Use comma as the deliminator to specify more than 1 ingestion mode.
- **MODEL_NAME**: Specify the custom YOLO model name which is expected to be in `video/resources/models/ultralytics/custom_models/<model name>.pt`. If not provided, the Ultralytics YOLO11n (`yolo11n`) is used.  If custom model is used, be sure the object list in `frontend/setting.py` matches the classes of the model.
<!-- - **NCURATIONS**: Specify the number of curation processes running in the background. -->
<!-- - **NSTREAMS**: Specify the number of video streams -->
- **OMIT_DETECTIONS_FLAG**: Flag to omit printing detections to screen. By default, object detections are printed.
<!-- - **PLATFORM**: Specify the target platform: `Xeon` -->
- **RESIZE_FLAG**: Specify `True` to resize videos to model input size or `False` (default) to use video resolution.
<!-- - **REGISTRY**: Name of private registry to push image. If registry secret is available, update `imagePullSecrets` field in [frontend.yaml.m4](../deployment/kubernetes/frontend.yaml.m4), [video_stream.yaml.m4](../deployment/kubernetes/video_stream.yaml.m4), and/or [video.yaml.m4](../deployment/kubernetes/video.yaml.m4) for Kubernetes. `docker login` may be necessary. -->
<br>


## Examples:
### Use videos
This sample provides a list of ten sample videos from Pexel.
If interested in using sample videos, run the provided script to download Pexel videos and accept the terms and conditions of the data set license.
```bash
cd inputs
./download.sh
```

#### Deploy Service
Then use the start script to deploy. An example for CPU is:
```bash
cd ..
./start_app.sh -e CPU
```
<br>

### Stream from RTSP Camera
This application accepts video stream from RTSP cameras.
The URL for the camera should be set in [camera_config.yaml](/inputs/camera_config.yaml).

#### Simulate Camera Stream
In cases where a camera is not available, Ffmpeg and MediaMTX can be used to simulate a camera feed from a source such as MP4 file.

First install MediaMTX ([see MediaMTX](https://github.com/bluenviron/mediamtx)) or use their docker container to start an RTSP server:
```bash
docker run --rm -it --network=host --name rstp_server bluenviron/mediamtx:latest
```

Use Ffmpeg to send video to the running server.
For example, to stream video `TEST_VIDEO` in a continuous loop to `URL` (specified in `camera_config.yaml`) at 30 frames per second, run the following:
```bash
FPS=30
GENERAL_OPTS="-threads 0 -flags +global_header -hide_banner -loglevel error -nostats -tune zerolatency -flush_packets 0"
TEST_VIDEO="<path_to_MP4_video>"
URL="rtsp://<HOSTNAME>:8554/rtsp1"  # User-defined

ffmpeg -re -stream_loop -1 -i ${TEST_VIDEO} ${GENERAL_OPTS} \
-vcodec libx264 -preset fast -crf 23 -bufsize 8M -pix_fmt yuv420p \
-filter:v fps=fps=${FPS} -f rtsp -rtsp_transport tcp ${URL}
```

#### Deploy Service
Then use the start script to deploy. An example for GPU and resizing videos to lower resolution (640x640) is:
```bash
./start_app.sh -e GPU -z
```
<br>

<!--
## Make Commands:

- **build**: Build the sample (docker) images.
- **update**: Distribute the sample images to worker nodes.
- **dist**: Create the sample distribution package.
- **start/stop_docker_compose**: Start/stop the sample orchestrated by docker-compose.
- **start/stop_docker_swarm**: Start/stop the sample orchestrated by docker swarm.
- **start/stop_kubernetes**: Start/stop the sample orchestrated by Kubernetes.

## See Also:

- [Sample Distribution](dist.md)
 -->
