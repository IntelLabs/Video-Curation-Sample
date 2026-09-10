![GitHub License](https://img.shields.io/github/license/IntelLabs/Video-Curation-Sample)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/Video-Curation-Sample/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/Video-Curation-Sample)

This sample implements a pipeline focusing on real-time processing of high-resolution (8K) video for object detection.
The application can process high-resolution video in real-time using a Smart Filtering Pipeline to significantly reduce pixel processing (compute) by *automatically* identifying the regions of interest which is then forwarded to the detection model.

Please see [High Resolution Object Detection Pipeline](./doc/pipeline.md) for more details.


### License Obligations
- FFmpeg is an open source project licensed under LGPL and GPL. See https://www.ffmpeg.org/legal.html. You are solely responsible for determining if your use of FFmpeg requires any additional licenses. Intel is not responsible for obtaining any such licenses, nor liable for any licensing fees due, in connection with your use of FFmpeg.


### Datasets & Attributions
- This project utilizes third-party open datasets. Please see our [Data Attributions](docs/DATASETS.md) for full licensing, copyright details, and citation parameters.


## Install Prerequisites:

- **Time Zone**: Check that the timezone setting of your host machine is correctly configured. Timezone is used during build.
<!-- If you plan to run the sample on a cluster of machines managed by Docker Swarm or Kubernetes, please make sure to synchronize time among the manager/master node and worker nodes. -->

- **Build Tools**: Install ```cmake``` and ```m4``` if they are not available on your system.

- **Docker Engine**:
  - Install [docker engine](https://docs.docker.com/install) and verify you have Docker Compose V2 2.18.0+ setup.
  - Setup docker proxy as follows if you are behind a firewall:
    ```bash
    sudo mkdir -p /etc/systemd/system/docker.service.d
    printf "[Service]\nEnvironment=\"HTTPS_PROXY=$https_proxy\" \"NO_PROXY=$no_proxy\"\n" | sudo tee /etc/systemd/system/docker.service.d/proxy.conf
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```


## Deploy High Resolution Drone Detection using Smart Filtering
All components for this application are dockerize.
Scripts are provided to make deployment easier.

### Start
[Optional] To make sure there aren't any running containers for this application, run the following which stops the application and prunes containers:
```bash
./stop.sh –p
```

To start the application, run the following:
```bash
./start_app.sh –m drone_detection
```


### View Live Detections
Launch your browser and browse to ```https://<hostname>:30077```. The sample UI is similar to the following:

<center><IMG src="doc/sample-ui.gif" height="270px"></IMG></center>

***NOTE:*** If you see a browser warning of self-signed certificate, please accept it to proceed to the sample UI.


### Shutdown
To shutdown this application, run the following:
```bash
./stop.sh
```

Or add the necessary flag to stop the application and prune containers:
```bash
./stop.sh –p
```
