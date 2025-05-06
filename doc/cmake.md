
## CMake Options:

Use the following definitions to customize the building process:
- **PLATFORM**: Specify the target platform: `Xeon`
- **NCURATIONS**: Specify the number of curation processes running in the background.
- **DEVICE**: Specify the device: `CPU` or `GPU`
- **NSTREAMS**: Specify the number of video streams
- **INGESTION**: Specify the ingestion mode: `face` and/or `object`. Use comma as the deliminator to specify more than 1 ingestion mode.
- **IN_SOURCE**: Specify the input video source: `videos` and/or `stream`. Use comma as the deliminator to specify more than 1 source.
- **STREAM_URL**: Specify the URL for streaming source. This is useful only if `IN_SOURCE` is `stream` and input is a URL and not a video stream.
<!-- - **REGISTRY**: Name of private registry to push image. If registry secret is available, update `imagePullSecrets` field in [frontend.yaml.m4](../deployment/kubernetes/frontend.yaml.m4), [video_stream.yaml.m4](../deployment/kubernetes/video_stream.yaml.m4), and/or [video.yaml.m4](../deployment/kubernetes/video.yaml.m4) for Kubernetes. `docker login` may be necessary. -->
<br>

<!-- ***Optimizations for sharing host with other applications:*** -->
<!-- The following optimizations are helpful if running other applications on the same host.
- [Assigning CPU resources](https://kubernetes.io/docs/tasks/configure-pod-container/assign-cpu-resource/) is helpful in this case. In this sample, we specify a CPU request for the ingest container by including the resources:requests field in the container resource manifest. Remove the following from [frontend.yaml.m4](../deployment/kubernetes/frontend.yaml.m4) under configurations for ingest container to disable this feature or modify as needed.
    ```JSON
    resources:
        requests:
            cpu: "1"
    ``` -->
- **NCPU**: Use `NCPU` in your cmake command to specify number of CPU cores for Ingestion. The ingest pool will run on randomly selected CPUs. Similar to `taskset` on Linux.
<br>

## Examples:
### Use videos
This sample uses a list of ten video from Pexel.  Please accept the license when prompted.  Use the following command to build the sample:
```bash
mkdir build
cd build
cmake ..
make
```

### Stream from webcam
Build the sample:
```bash
mkdir build
cd build
cmake -DIN_SOURCE=stream ..
make
```
Start the sample using preferred method, then use FFMPeg to start your webcam locally and send via UDP to the host machine (`<hostname>`) and udp port 30009. A sample command is the following:
```bash
ffmpeg -re -f dshow -rtbufsize 100M -i video="HP HD Camera" -vcodec libx264 -crf 28 -threads 1 -s 640x360 -f mpegts -flush_packets 0 udp://<hostname>:30009?pkt_size=18800
```

### Stream video using URL
Use the following command to stream the [face-demographics-walking Sample video](https://github.com/intel-iot-devkit/sample-videos) from [Intel's IoT Libraries & Code Samples](https://github.com/intel-iot-devkit):
```
cd build
cmake -DSTREAM_URL="https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4" -DIN_SOURCE=stream ..
make
```


## Make Commands:

- **build**: Build the sample (docker) images.
- **update**: Distribute the sample images to worker nodes.
- **dist**: Create the sample distribution package.
- **start/stop_docker_compose**: Start/stop the sample orchestrated by docker-compose.
- **start/stop_docker_swarm**: Start/stop the sample orchestrated by docker swarm.
- **start/stop_kubernetes**: Start/stop the sample orchestrated by Kubernetes.

## See Also:

- [Sample Distribution](dist.md)

