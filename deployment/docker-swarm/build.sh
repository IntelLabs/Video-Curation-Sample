#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
PLATFORM="${1:-Xeon}"
NCURATIONS="$2"
INGESTION="$3"
IN_SOURCE="$4"
STREAM_URL="$5"
NCPU="$6"
REGISTRY="$7"
NSTREAMS="$8"
DEVICE="$9"
DEBUG="${10}"

echo "Generating templates with PLATFORM=${PLATFORM},NCURATIONS=${NCURATIONS},NSTREAMS=${NSTREAMS},INGESTION=${INGESTION},DEVICE=${DEVICE},IN_SOURCE=${IN_SOURCE},STREAM_URL=${STREAM_URL},NCPU=${NCPU},HOSTIP=${HOSTIP},DEBUG=${DEBUG}"

if test -f "${DIR}/docker-compose.yml.m4"; then
    echo "Generating docker-compose.yml"
    m4 -D${DEVICE} -Din_${IN_SOURCE} -DREGISTRY_PREFIX=$REGISTRY -DINGESTION="$INGESTION" -DDEVICE="$DEVICE" -DDEBUG="$DEBUG" -DNCURATIONS="${NCURATIONS}" -DNSTREAMS="${NSTREAMS}" -DIN_SOURCE="${IN_SOURCE}" -DSTREAM_URL="${STREAM_URL}" -DNCPU="${NCPU}" -I "${DIR}" "${DIR}/docker-compose.yml.m4" > "${DIR}/docker-compose.yml"
fi

