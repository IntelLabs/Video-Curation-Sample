
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
        networks:
            - appnet
        restart: always
        healthcheck:
            disable: true
        environment:
            OVERRIDE_print_high_level_timing: "true"
