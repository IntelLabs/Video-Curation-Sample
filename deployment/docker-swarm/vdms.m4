
    vdms-service:
        image: intellabs/vdms:latest
        ports:
            - target: 55555
              published: 55555
              protocol: tcp
              mode: host
        volumes:
            - /etc/localtime:/etc/localtime:ro
        networks:
            - appnet
        restart: always

