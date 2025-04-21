#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
PLATFORM="${1:-Xeon}"
NCURATIONS=${2:-1}
INGESTION="$3"
IN_SOURCE="$4"
STREAM_URL="$5"
NCPU=$6
REGISTRY="$7"
HOSTIP=$(ip route get 8.8.8.8 | awk '/ src /{split(substr($0,index($0," src ")),f);print f[2];exit}')

echo "Generating templates with PLATFORM=${PLATFORM},NCURATIONS=${NCURATIONS},INGESTION=${INGESTION},IN_SOURCE=${IN_SOURCE},STREAM_URL=${STREAM_URL},NCPU=${NCPU},HOSTIP=${HOSTIP}"
find "${DIR}" -maxdepth 1 -mindepth 1 -name "*.yaml" -exec rm -rf "{}" \;
for template in $(find "${DIR}" -maxdepth 1 -mindepth 1 -name "*.yaml.m4" -print); do
    yaml=${template/.m4/}
    m4 -DREGISTRY_PREFIX=${REGISTRY} -DPLATFORM=${PLATFORM} -DNCURATIONS=${NCURATIONS} -DINGESTION="${INGESTION}" -DIN_SOURCE=${IN_SOURCE} -DSTREAM_URL=${STREAM_URL} -DNCPU=${NCPU} -DUSERID=$(id -u) -DGROUPID=$(id -g) -DHOSTIP=${HOSTIP} -I "${DIR}" "${template}" > "${yaml}"
    # m4 -DREGISTRY_PREFIX=$REGISTRY -DINGESTION="$INGESTION" -DNCURATIONS=$NCURATIONS -DIN_SOURCE=$IN_SOURCE -DSTREAM_URL=$STREAM_URL -I "${DIR}" "${template}" > "${yaml}"
done
