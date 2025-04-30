#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
yml="$DIR/docker-compose.yml"

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

    if [ "$DEVICE" == "GPU" ]; then
        docker compose -f "$yml" -p lcc --profile $DEVICE --compatibility down -v
    else
        docker compose -f "$yml" -p lcc --compatibility down -v
    fi
    ;;
*)
    docker stack services lcc
    echo "Shutting down stack lcc..."
    while test -z "$(docker stack rm lcc 2>&1 | grep 'Nothing found in stack')"; do
        sleep 2
    done
    ;;
esac

docker container prune -f; echo
docker volume prune -f; echo
docker network prune -f; echo
