#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
yml="$DIR/docker-compose.yml"

case "$1" in
docker_compose)
    dcv="$(docker-compose --version | cut -f3 -d' ' | cut -f1 -d',')"
    mdcv="$(printf '%s\n' $dcv 1.20 | sort -r -V | head -n 1)"
    if test "$mdcv" = "1.20"; then
        echo ""
        echo "docker-compose >=1.20 is required."
        echo "Please upgrade docker-compose at https://docs.docker.com/compose/install."
        echo ""
        exit 0
    fi

    echo "Cleanup $(hostname)..."
    docker container prune -f; echo
    docker volume prune -f; echo
    docker network prune -f; echo

    shift
    . "$DIR/build.sh"
    docker-compose -f "$yml" -p lcc --compatibility up
    ;;
*)
    shift
    . "$DIR/build.sh"
    docker stack deploy -c "$yml" lcc
    ;;
esac
