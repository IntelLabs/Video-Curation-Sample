
    vdms-service:
        image: defn(`REGISTRY_PREFIX')lcc_vdms:stream
        ports:
            - target: 55555
              published: 55555
              protocol: tcp
              mode: host
        volumes:
            - /etc/localtime:/etc/localtime:ro
            - app-content:/var/www
            - vdms-content:/mnt:rw
        networks:
            - appnet
        restart: always
        healthcheck:
            disable: true
        environment:
            OVERRIDE_db_root_path: "/mnt/db"
            OVERRIDE_print_high_level_timing: "true"
            no_proxy: "udf-bkgd-service,udf-service,${no_proxy}"
            NO_PROXY: "udf-bkgd-service,udf-service,${NO_PROXY}"
