define(`PROFILE_DEFAULT', `')
define(`PROFILE_GPU', `runtime: nvidia
        deploy:
            resources:
                reservations:
                    devices:
                        - driver: nvidia
                          capabilities: [gpu]')
    video-service:
        image: defn(`REGISTRY_PREFIX')lcc_video:stream
        environment:
            YOLO_CONFIG_DIR: "/tmp"
            RETENTION_MINS: "60"
            CLEANUP_INTERVAL: "10m"
            DBHOST: "vdms-service"
            UDF_HOST: "udf-service"
            `MODEL_NAME': "defn(`MODEL_NAME')"
            `CUSTOM_MODEL_FLAG': "defn(`CUSTOM_MODEL_FLAG')"
            `RESIZE_FLAG': "defn(`RESIZE_FLAG')"
            `OMIT_DETECTIONS_FLAG': "defn(`OMIT_DETECTIONS_FLAG')"
            CPU_BATCH_SIZE: 1
            GPU_BATCH_SIZE: 1
            `DEBUG': "defn(`DEBUG')"
            `DEVICE': "defn(`DEVICE')"
            `INGESTION': "defn(`INGESTION')"
            WATCH_DIR: "/watch_dir"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "fastapi-service,localhost,127.0.0.1,vdms-service,udf-service,${no_proxy}"
            NO_PROXY: "fastapi-service,localhost,127.0.0.1,vdms-service,udf-service,${NO_PROXY}"
        secrets:
            - source: self_crt
              target: /var/run/secrets/self.crt
              uid: ${USER_ID}
              gid: ${GROUP_ID}
              mode: 0444
            - source: self_key
              target: /var/run/secrets/self.key
              uid: ${USER_ID}
              gid: ${GROUP_ID}
              mode: 0440
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - app-content:/var/www
            - ../../inputs:/watch_dir:ro
            - ../../inputs/camera_config.yaml:/home/camera_config.yaml:ro
        networks:
            - appnet
        restart: always
        depends_on:
            - fastapi-service
            - udf-service
            - vdms-service

    fastapi-service:
        shm_size: '2gb'  # Give it plenty of space for video frames
        image: defn(`REGISTRY_PREFIX')lcc_fastapi:stream
        environment:
            YOLO_CONFIG_DIR: "/tmp"
            DBHOST: "vdms-service"
            UDF_HOST: "udf-service"
            `MODEL_NAME': "defn(`MODEL_NAME')"
            `CUSTOM_MODEL_FLAG': "defn(`CUSTOM_MODEL_FLAG')"
            `RESIZE_FLAG': "defn(`RESIZE_FLAG')"
            `OMIT_DETECTIONS_FLAG': "defn(`OMIT_DETECTIONS_FLAG')"
            CPU_BATCH_SIZE: 1
            GPU_BATCH_SIZE: 1
            `DEBUG': "defn(`DEBUG')"
            `DEVICE': "defn(`DEVICE')"
            `INGESTION': "defn(`INGESTION')"
            WATCH_DIR: "/watch_dir"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "video-service,localhost,127.0.0.1,vdms-service,udf-service,${no_proxy}"
            NO_PROXY: "video-service,localhost,127.0.0.1,vdms-service,udf-service,${NO_PROXY}"
        ports:
            - target: 80
              published: 30077
              protocol: tcp
              mode: host
        secrets:
            - source: self_crt
              target: /var/run/secrets/self.crt
              uid: ${USER_ID}
              gid: ${GROUP_ID}
              mode: 0444
            - source: self_key
              target: /var/run/secrets/self.key
              uid: ${USER_ID}
              gid: ${GROUP_ID}
              mode: 0440
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - app-content:/var/www
            - ../../inputs:/watch_dir:ro
        networks:
            - appnet
        restart: always
        depends_on:
            - udf-service
            - vdms-service
        ifdef(`GPU', PROFILE_GPU, PROFILE_DEFAULT)
