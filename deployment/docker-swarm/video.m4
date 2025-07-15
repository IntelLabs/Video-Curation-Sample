define(`PROFILE_DEFAULT', `depends_on:
            - udf-service
            - vdms-service')
define(`PROFILE_GPU', `depends_on:
            - udf-service
            - vdms-service
        runtime: nvidia
        deploy:
            resources:
                reservations:
                    devices:
                        - driver: nvidia
                          capabilities: [gpu]')
    video-service:
        image: defn(`REGISTRY_PREFIX')lcc_video:stream
        environment:
            RETENTION_MINS: "60"
            CLEANUP_INTERVAL: "10m"
            KKHOST: "kafka-service:9092"
            SHOST: "http://stream-service:8080"
            ZKHOST: "zookeeper-service:2181"
            DBHOST: "vdms-service"
            `RESIZE_FLAG': "defn(`RESIZE_FLAG')"
            CPU_BATCH_SIZE: 1
            GPU_BATCH_SIZE: 1
            `DEBUG': "defn(`DEBUG')"
            `DEVICE': "defn(`DEVICE')"
            `IN_SOURCE': "defn(`IN_SOURCE')"
            `INGESTION': "defn(`INGESTION')"
            `NCURATIONS': "defn(`NCURATIONS')"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "stream-service,vdms-service,${no_proxy}"
            NO_PROXY: "stream-service,vdms-service,${NO_PROXY}"
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - app-content:/var/www
        networks:
            - appnet
        restart: always
        ifdef(`GPU', PROFILE_GPU, PROFILE_DEFAULT)
