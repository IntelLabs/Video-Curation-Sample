
    frontend-service:
        image: defn(`REGISTRY_PREFIX')lcc_frontend:stream
        ports:
            - target: 8443
              published: 30007
              protocol: tcp
              mode: host
        environment:
            DBHOST: "vdms-service"
            VDHOST: "http://video-service:8080"
            no_proxy: "vdms-service,video-service,${no_proxy}"
            NO_PROXY: "vdms-service,video-service,${NO_PROXY}"
            `DEBUG': "defn(`DEBUG')"
        secrets:
            - source: self_crt
              target: /var/run/secrets/self.crt
              uid: ${USER_ID}
              gid: ${GROUP_ID}
              mode: 0444
            - source: self_key
              target: /var/run/secrets/self.key
              uid: ${USER_ID}
              gid: ${GROUP_ID}
              mode: 0440
        volumes:
            - /etc/localtime:/etc/localtime:ro
        networks:
            - appnet
        restart: always
        deploy:
            placement:
                constraints:
                    - node.role==manager
        depends_on:
            - vdms-service
            - video-service
