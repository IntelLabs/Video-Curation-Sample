
    video-service:
        image: defn(`REGISTRY_PREFIX')lcc_video:stream
        environment:
            RETENTION_MINS: "60"
            CLEANUP_INTERVAL: "10m"
            KKHOST: "kafka-service:9092"
            ZKHOST: "zookeeper-service:2181"
            `INGESTION': "defn(`INGESTION')"
            `IN_SOURCE': "defn(`IN_SOURCE')"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "${no_proxy}"
            NO_PROXY: "${NO_PROXY}"
        volumes:
            - /etc/localtime:/etc/localtime:ro
        networks:
            - appnet
        restart: always
