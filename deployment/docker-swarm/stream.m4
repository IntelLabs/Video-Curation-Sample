
    stream-service:
        image: defn(`REGISTRY_PREFIX')lcc_stream:stream
        ports:
            - target: 8088
              published: 30009
              protocol: udp
              mode: host
        environment:
            KKHOST: "kafka-service:9092"
            VDHOST: "http://video-service:8080"
            DBHOST: "vdms-service"
            ZKHOST: "zookeeper-service:2181"
            `STREAM_URL': "defn(`STREAM_URL')"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "video-service,${no_proxy}"
            NO_PROXY: "video-service,${NO_PROXY}"
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - stream-content:/var/www/mp4
        networks:
            - appnet
        restart: always
