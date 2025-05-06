#!/bin/bash -e
#######################################################################################################################
# This script runs the Curation application
#######################################################################################################################
# DEFAULT VARIABLES
INGESTION="object,face"
EXP_TYPE=compose
REGISTRY=None
NCPU=0
NCURATIONS=1
NSTREAMS=1
IN_SOURCE=stream
SOURCE="-DSTREAM_URL="rtp://127.0.0.1:8088" -DIN_SOURCE=${IN_SOURCE}"
DEBUG="0"
DEVICE="CPU"

DIR=$(dirname $(readlink -f "$0"))
BUILD_DIR=$DIR/build

LONG_LIST=(
    "ingestion"
    "type"
    "registry"
    "ncurations"
    "nstreams"
    "ncpu"
    "source"
    "debug"
    "device"
)

OPTS=$(getopt \
    --longoptions "$(printf "%s:," "${LONG_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "hdi:t:r:n:v:c:s:e:" \
    -- "$@"
)

eval set -- $OPTS

if [ -d "$BUILD_DIR" ]; then
    rm -rf $BUILD_DIR
fi

mkdir -p $BUILD_DIR

#######################################################################################################################
# GET SCRIPT OPTIONS
script_usage()
{
    cat <<EOF
    This script runs the Video Curation Streaming Application

    Usage: $0 [ options ]

    Options:
        -h                  optional    Print this help message
        -d or --debug       optional    '1' to enable debug messages, otherwise '0'
        -e or --device      optional    Device for inference (CPU, GPU) [Default: CPU]
        -i or --ingestion   optional    Ingestion type (object, face) [Default: "object,face"]
        -n or --ncurations  optional    Number of ingestion containers [Default: 1]
        -r or --registry    optional    Registry [Default: None]
        -s or --source      optional    Input source type (videos, stream) [Default: stream]
        -t or --type        optional    Deployment method (compose) [Default: compose]
        -v or --nstreams    optional    Number of video streams [Default: 1]

EOF
}

while true; do
    case "$1" in
        -h) script_usage; exit 0 ;;
        -c | --ncpu) shift; NCPU=$1; shift ;;
        -d | --debug) shift; DEBUG="1" ;;
        -e | --device) shift; DEVICE=$1; shift ;;
        -i | --ingestion) shift; INGESTION=$1; shift ;;
        -n | --ncurations) shift; NCURATIONS=$1; shift ;;
        -r | --registry) shift; REGISTRY="$1"; shift ;;
        -s | --source)
            shift;
            IN_SOURCE="$1";
            if [ $IN_SOURCE == "stream" ]; then
                SOURCE="-DSTREAM_URL="rtp://127.0.0.1:8088" -DIN_SOURCE=${IN_SOURCE}";

            elif [ $IN_SOURCE == "videos" ]; then
                SOURCE="-DIN_SOURCE=${IN_SOURCE}";

            else
                echo "INVALID SOURCE TYPE (-s): ${IN_SOURCE}";
                script_usage;
                exit 0;

            fi

            shift;
            ;;
        -t | --type) shift; EXP_TYPE="$1"; shift ;;
        -v | --nstreams) shift; NSTREAMS=$1; shift ;;
        --) shift; break ;;
        *) script_usage; exit 0 ;;
    esac
done

#######################################################################################################################
# BUILD AND START APP
cd $BUILD_DIR

if [ $REGISTRY == "None" ]; then
    cmake -DINGESTION=$INGESTION $SOURCE -DNCPU=$NCPU -DNCURATIONS=$NCURATIONS -DNSTREAMS=$NSTREAMS -DDEBUG=$DEBUG -DDEVICE=$DEVICE ..
else
    cmake -DREGISTRY=$REGISTRY -DINGESTION=$INGESTION $SOURCE -DNCPU=$NCPU -DNCURATIONS=$NCURATIONS -DNSTREAMS=$NSTREAMS -DDEBUG=$DEBUG -DDEVICE=$DEVICE ..
fi

make

if [ $EXP_TYPE == "compose" ]; then
    make start_docker_compose

# elif [ $EXP_TYPE == "k8" ]; then
#     if [ $REGISTRY == "None" ]; then
#         make update
#     fi

#     make start_kubernetes

else
    echo "INVALID TYPE: ${EXP_TYPE}"

fi

cd $DIR
