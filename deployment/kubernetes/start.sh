#!/bin/bash -e

DIR=$(dirname $(readlink -f "$0"))
PLATFORM="${1:-Xeon}"
NCURATIONS="$2"
INGESTION="$3"
IN_SOURCE="$4"
STREAM_URL="$5"
NCPU="$6"
REGISTRY="$7"

shift
. "$DIR/build.sh"

function create_secret {
    kubectl create secret generic self-signed-certificate "--from-file=${DIR}/../certificate/self.crt" "--from-file=${DIR}/../certificate/self.key"
    kubectl create configmap proxy-config \
        --from-literal=http_proxy="$http_proxy" \
        --from-literal=HTTP_PROXY="$http_proxy" \
        --from-literal=https_proxy="$https_proxy" \
        --from-literal=HTTPS_PROXY="$http_proxy" \
        --from-literal=no_proxy="$no_proxy" \
        --from-literal=NO_PROXY="$no_proxy"
}

# create secrets
"$DIR/../certificate/self-sign.sh"
create_secret 2>/dev/null || (kubectl delete secret self-signed-certificate; create_secret)

# Choose video yaml based on IN_SOURCE
if [ "$IN_SOURCE" == "stream" ]; then
    skip_name="video.yaml"
else
    skip_name="video_stream.yaml"
fi

for yaml in $(find "$DIR" -maxdepth 1 -name "*.yaml" -print); do
    if [[ $yaml != *"$skip_name"* ]]; then
        kubectl apply -f "$yaml"
    fi
done

