#!/bin/bash -e
#######################################################################################################################
# This script stops the Curation application
#######################################################################################################################
# DEFAULT VARIABLES
EXP_TYPE=compose
DOCKER_PRUNE="0"

DIR=$(dirname $(readlink -f "$0"))
BUILD_DIR=$DIR/build

LONG_LIST=(
    "type"
    "prune"
)

OPTS=$(getopt \
    --longoptions "$(printf "%s:," "${LONG_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "hpt:" \
    -- "$@"
)

eval set -- $OPTS

#######################################################################################################################
# GET SCRIPT OPTIONS
script_usage()
{
    cat <<EOF
    This script stops the Video Curation Streaming Application

    Usage: $0 [ options ]

    Options:
        -h                  optional    Print this help message
        -t or --type        optional    Deployment method (compose, k8) [Default: compose]
        -p or --prune       optional    Flag to prune docker builder

EOF
}

while true; do
    case "$1" in
        -h) script_usage; exit 0 ;;
        -t | --type) shift; EXP_TYPE="$1"; shift ;;
        -p | --prune) shift; DOCKER_PRUNE="1" ;;
        --) shift; break ;;
        *) script_usage; exit 0 ;;
    esac
done

#######################################################################################################################
# STOP APP
cd $BUILD_DIR

if [ $EXP_TYPE == "compose" ]; then
    make stop_docker_compose

elif [ $EXP_TYPE == "k8" ]; then
    make stop_kubernetes

else
    echo "INVALID TYPE: ${EXP_TYPE}"

fi

if [ $DOCKER_PRUNE == "1" ]; then
    DOCKER_BUILDKIT=1 docker builder prune -f || true
    docker container prune -f || true
    docker volume prune -f || true
    docker network prune -f || true
fi

cd $DIR
