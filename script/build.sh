#!/bin/bash -e

if test -z "${DIR}"; then
    echo "This script should not be called directly."
    exit -1
fi

PLATFORM="${1:-Xeon}"
FRAMEWORK="gst"
IN_SOURCE="$4"
REGISTRY="$6"
DEVICE="$8"
DEBUG="$9"
DOCKER_TAR="${10}"
DOCKER_TAR_DIR="${11}"
RESIZE_FLAG="${12}"
USER="docker"
GROUP="docker"

build_docker() {
    docker_file="$1"
    shift
    image_name="$1:stream"
    shift
    if test -f "$docker_file.m4"; then
        m4 -D${DEVICE} -I "$(dirname $docker_file)" "$docker_file.m4" > "$docker_file"
    fi

    if [[ "$DOCKER_TAR" != "1" ]] || [[ "${image_name}" == *"_certificate"* ]]; then
        if [[ "$DOCKER_TAR" == "1" ]] && [[ "${image_name}" == *"_certificate"* ]]; then
            echo "Loading zookeeper and kafka docker image"
            docker load -i ${DOCKER_TAR_DIR}/zookeeper.tar
            docker load -i ${DOCKER_TAR_DIR}/kafka.tar

            echo "Loading certificate image"
            docker load -i ${DOCKER_TAR_DIR}/lcc_certificate_stream.tar
        fi

        (cd "$DIR"; docker build --network host --file="$docker_file" "$@" -t "$image_name" "$DIR" $(env | cut -f1 -d= | grep -E '_(proxy|REPO|VER)$' | sed 's/^/--build-arg /') --build-arg USER=${USER} --build-arg GROUP=${GROUP} --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg DEVICE=${DEVICE} --build-arg IN_SOURCE=${IN_SOURCE} --build-arg DEBUG=${DEBUG} --build-arg RESIZE_FLAG=${RESIZE_FLAG})
    else
        tar_name="${image_name/:/_}"
        if [[ "${tar_name}" == *"_video_"* ]]; then
            tar_name=${tar_name}.${DEVICE}
        fi
        echo "Loading ${tar_name}.tar docker image"
        docker load -i "${DOCKER_TAR_DIR}/${tar_name}.tar"
    fi

    # if REGISTRY is specified, push image to the private registry
    if [ "$REGISTRY" != " " ]; then
        docker tag "$image_name" "$REGISTRY$image_name"
        docker push "$REGISTRY$image_name"
    fi
}

# build image(s) in order (to satisfy dependencies)
for dep in '.5.*' '.4.*' '.3.*' '.2.*' '.1.*' '.0.*' ''; do
    dirs=("$DIR/$PLATFORM/$FRAMEWORK" "$DIR/$PLATFORM" "$DIR")
    for dockerfile in $(find "${dirs[@]}" -maxdepth 1 -name "Dockerfile$dep" -print 2>/dev/null); do
        image=$(head -n 1 "$dockerfile" | grep '# ' | cut -d' ' -f2)
        if test -z "$image"; then image="$IMAGE"; fi
        build_docker "$dockerfile" "$image"
    done
done

docker rmi $(docker images -f "dangling=true" -q) || true
