#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
yml="$DIR/docker-compose.yml"
DEVICE="$8"

case "$1" in
docker_compose)
    dcv="$(docker compose version | cut -f4 -d' ' | cut -f1 -d',')"
    mdcv="$(printf '%s\n' $dcv v2.18 | sort -r -V | head -n 1)"
    if test "$mdcv" = "v2.18"; then
        echo ""
        echo "docker compose >=2.18 is required."
        echo "Please upgrade docker compose at https://docs.docker.com/compose/install."
        echo ""
        exit 0
    fi

    echo "Cleanup $(hostname)..."
    docker container prune -f; echo
    docker volume prune -f; echo
    docker network prune -f; echo

    shift
    . "$DIR/build.sh"
    # if [ "$DEVICE" == "GPU" ]; then
    #     #TODO: Test running separately then running others
    #     DOCKER_BUILDKIT=0 docker compose -f "$yml" -p lcc --compatibility up video-service
    #     docker compose -f "$yml" -p lcc --compatibility up
    # else
    docker compose -f "$yml" -p lcc --compatibility up
    # fi
    ;;
*)
    shift
    . "$DIR/build.sh"
    docker stack deploy -c "$yml" lcc  # TODO: Incorporate device
    ;;
esac
