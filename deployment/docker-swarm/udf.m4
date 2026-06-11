    udf-service:
        image: defn(`REGISTRY_PREFIX')lcc_udf:stream
        expose:
            - "5011"
        environment:
            RETENTION_MINS: "60"
            CLEANUP_INTERVAL: "10m"
            DBHOST: "vdms-service"
            UDF_PORT: 5011
            `RESIZE_FLAG': "defn(`RESIZE_FLAG')"
            CPU_BATCH_SIZE: 1
            GPU_BATCH_SIZE: 1
            `DEBUG': "defn(`DEBUG')"
            `DEVICE': "defn(`DEVICE')"
            `INGESTION': "defn(`INGESTION')"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "fastapi-service,video-service,vdms-service,${no_proxy}"
            NO_PROXY: "fastapi-service,video-service,vdms-service,${NO_PROXY}"
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - app-content:/var/www:ro
        networks:
            - appnet
        restart: always
        depends_on:
            - vdms-service
        deploy:
            replicas: 1
    udf-bkgd-service:
        image: defn(`REGISTRY_PREFIX')lcc_udf:stream
        expose:
            - "5012"
        environment:
            RETENTION_MINS: "60"
            CLEANUP_INTERVAL: "10m"
            DBHOST: "vdms-service"
            UDF_PORT: 5012
            `RESIZE_FLAG': "defn(`RESIZE_FLAG')"
            CPU_BATCH_SIZE: 1
            GPU_BATCH_SIZE: 1
            `DEBUG': "defn(`DEBUG')"
            `DEVICE': "defn(`DEVICE')"
            `INGESTION': "defn(`INGESTION')"
            http_proxy: "${http_proxy}"
            HTTP_PROXY: "${HTTP_PROXY}"
            https_proxy: "${https_proxy}"
            HTTPS_PROXY: "${HTTPS_PROXY}"
            no_proxy: "fastapi-service,video-service,vdms-service,${no_proxy}"
            NO_PROXY: "fastapi-service,video-service,vdms-service,${NO_PROXY}"
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - app-content:/var/www:ro
        networks:
            - appnet
        restart: always
        depends_on:
            - vdms-service
        deploy:
            replicas: 2
