#!/bin/bash -e
#######################################################################################################################
# This script stops the Curation application
#######################################################################################################################
# DEFAULT VARIABLES
EXP_TYPE=compose

DIR=$(dirname $(readlink -f "$0"))
BUILD_DIR=$DIR/build

LONG_LIST=(
    "type"
)

OPTS=$(getopt \
    --longoptions "$(printf "%s:," "${LONG_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "ht:" \
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

EOF
}

while true; do
    case "$1" in
        -h) script_usage; exit 0 ;;
        -t | --type) shift; EXP_TYPE="$1"; shift ;;
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

cd $DIR
