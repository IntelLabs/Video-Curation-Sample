
    ingest:
        image: defn(`REGISTRY_PREFIX')lcc_ingest:stream
        environment:
            KKHOST: "kafka-service:9092"
            VDHOST: "http://video-service:8080"
            DBHOST: "vdms-service"
            ZKHOST: "zookeeper-service:2181"
            `IN_SOURCE': "defn(`IN_SOURCE')"
            `NCPU': "defn(`NCPU')"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "video-service,${no_proxy}"
            NO_PROXY: "video-service,${NO_PROXY}"
        volumes:
            - /etc/localtime:/etc/localtime:ro
        networks:
            - appnet
        restart: always
        deploy:
            replicas: defn(`NCURATIONS')
