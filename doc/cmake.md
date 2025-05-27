
## CMake Options:

Use the following definitions to customize the building process:
- **PLATFORM**: Specify the target platform: `Xeon`
- **NCURATIONS**: Specify the number of curation processes running in the background.
- **DEVICE**: Specify the device: `CPU` or `GPU`
- **NSTREAMS**: Specify the number of video streams
- **INGESTION**: Specify the ingestion mode: `face` and/or `object`. Use comma as the deliminator to specify more than 1 ingestion mode.
- **IN_SOURCE**: Specify the input video source: `videos` and/or `stream`. Use comma as the deliminator to specify more than 1 source.
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

### Stream from webcam or URL
Build the sample:
```bash
mkdir build
cd build
cmake -DIN_SOURCE=stream ..
make
```
Start the sample using preferred method, then use FFMPeg to start your webcam locally (or send a MP4 URL) and send via UDP to the host machine (`<hostname>`) and udp port (`<stream_port>`). Below is a sample command to stream using the internal camera on an HP laptop:
```bash
ffmpeg -re -f dshow -rtbufsize 100M -i video="HP HD Camera" -c copy -f mpegts -flush_packets 0 udp://<hostname>:<stream_port>?pkt_size=18800
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

