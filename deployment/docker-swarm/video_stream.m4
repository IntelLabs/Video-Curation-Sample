
    video-service:
        image: defn(`REGISTRY_PREFIX')lcc_video:stream
        environment:
            RETENTION_MINS: "60"
            CLEANUP_INTERVAL: "10m"
            KKHOST: "kafka-service:9092"
            SHOST: "http://stream-service:8080"
            ZKHOST: "zookeeper-service:2181"
            `INGESTION': "defn(`INGESTION')"
            `IN_SOURCE': "defn(`IN_SOURCE')"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "stream-service,${no_proxy}"
            NO_PROXY: "stream-service,${NO_PROXY}"
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - stream-content:/var/www/streams
        networks:
            - appnet
        restart: always
