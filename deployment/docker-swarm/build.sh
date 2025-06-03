#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
PLATFORM="${1:-Xeon}"
NCURATIONS="$2"
INGESTION="$3"
IN_SOURCE="$4"
NCPU="$5"
REGISTRY="$6"
NSTREAMS="$7"
DEVICE="$8"
DEBUG="$9"
DOCKER_TAR="${10}"
DOCKER_TAR_DIR="${11}"

echo "Generating templates with PLATFORM=${PLATFORM},NCURATIONS=${NCURATIONS},NSTREAMS=${NSTREAMS},INGESTION=${INGESTION},DEVICE=${DEVICE},IN_SOURCE=${IN_SOURCE},NCPU=${NCPU},HOSTIP=${HOSTIP},DEBUG=${DEBUG},DOCKER_TAR=${DOCKER_TAR},DOCKER_TAR_DIR=${DOCKER_TAR_DIR}"

if test -f "${DIR}/docker-compose.yml.m4"; then
    echo "Generating docker-compose.yml"
    m4 -D${DEVICE} -Din_${IN_SOURCE} -DREGISTRY_PREFIX=$REGISTRY -DINGESTION="$INGESTION" -DDEVICE="$DEVICE" -DDEBUG="$DEBUG" -DNCURATIONS="${NCURATIONS}" -DNSTREAMS="${NSTREAMS}" -DIN_SOURCE="${IN_SOURCE}" -DNCPU="${NCPU}" -I "${DIR}" "${DIR}/docker-compose.yml.m4" > "${DIR}/docker-compose.yml"
fi

