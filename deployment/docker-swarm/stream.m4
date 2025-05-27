
    stream-service:
        image: defn(`REGISTRY_PREFIX')lcc_stream:stream
        ports:
            - "30009-30109:8088/tcp"
            - "30009-30109:8088/udp"
        environment:
            KKHOST: "kafka-service:9092"
            VDHOST: "http://video-service:8080"
            DBHOST: "vdms-service"
            ZKHOST: "zookeeper-service:2181"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "video-service,${no_proxy}"
            NO_PROXY: "video-service,${NO_PROXY}"
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - app-content:/var/www
        networks:
            - appnet
        restart: always
        deploy:
            replicas: defn(`NSTREAMS')
