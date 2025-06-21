cmake_minimum_required (VERSION 2.8)

Project(library_curation NONE)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_HOME_DIRECTORY}/dist")

set(EMPTY " ")
if (NOT DEFINED REGISTRY)
    set(REGISTRY "${EMPTY}")
elseif (NOT ${REGISTRY} MATCHES "/$")
    set(REGISTRY "${REGISTRY}/")
endif()

if (NOT DEFINED PLATFORM)
    set(PLATFORM "Xeon")
endif()

if (NOT DEFINED NSTREAMS)
    set(NSTREAMS 1)
endif()

if (NOT DEFINED NCURATIONS)
    set(NCURATIONS 1)
endif()

if (NOT DEFINED INGESTION)
    set(INGESTION "object,face")
endif()

if (NOT DEFINED DEVICE)
    set(DEVICE "CPU")
endif()

if(DEVICE MATCHES "CPU")
    set(NVIDIA_VISIBLE_DEVICES "all")
else()
    set(NVIDIA_VISIBLE_DEVICES "none")
endif()

if (NOT DEFINED DEBUG)
    set(DEBUG "0")
endif()

if (NOT DEFINED DOCKER_TAR)
    set(DOCKER_TAR "0")
endif()

set(DOCKER_TAR_DIR ${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars)

if(DOCKER_TAR MATCHES "1")
    set(TAR_FILES
        "${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars/lcc_frontend_stream.tar"
        "${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars/lcc_video_stream.${DEVICE}.tar"
        "${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars/zookeeper.tar"
        "${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars/lcc_vdms_stream.tar"
        "${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars/kafka.tar"
        "${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars/lcc_certificate_stream.tar"
    )

    # Check if all files exist
    set(ALL_TARS_EXIST TRUE)
    foreach(TAR_FILE ${TAR_FILES})
        if(NOT EXISTS "${TAR_FILE}")
            set(ALL_TARS_EXIST FALSE)
        endif()
    endforeach()

    if(NOT ALL_TARS_EXIST)
        message(WARNING "All required tar files do not exist.")
        set(DOCKER_TAR "0")
        set(DOCKER_TAR_DIR "${EMPTY}")
    endif()

endif()

if(NOT DEFINED INGEST_METHOD)
    set(INGEST_METHOD "manual")
endif()

if (NOT DEFINED IN_SOURCE)
    set(IN_SOURCE "videos") # Use videos
endif()

if (NOT DEFINED NCPU)
    set(NCPU 0)
endif()


if(REGISTRY MATCHES "${EMPTY}")
    add_custom_target(update  ${CMAKE_HOME_DIRECTORY}/script/update-image.sh)
endif()
add_custom_target(dist  ${CMAKE_HOME_DIRECTORY}/script/mk-dist.sh)

file(GLOB dirs "deployment" "*")
list(REMOVE_DUPLICATES dirs)
foreach (dir ${dirs})
    if(EXISTS ${dir}/CMakeLists.txt)
        if(${dir} MATCHES "stream" AND IN_SOURCE MATCHES "stream")
            add_subdirectory(${dir})
        elseif(NOT ${dir} MATCHES "stream")
            add_subdirectory(${dir})
        endif()
    endif()
endforeach()

# legal message
execute_process(COMMAND printf "\n\nThis script will build third party components licensed under various open source licenses into your container images. The terms under which those components may be used and distributed can be found with the license document that is provided with those components. Please familiarize yourself with those terms to ensure your distribution of those components complies with the terms of those licenses.\n\n")

execute_process(COMMAND printf "PLATFORM=${PLATFORM}, NSTREAMS=${NSTREAMS}, NCURATIONS=${NCURATIONS}, INGESTION=${INGESTION}\n")
execute_process(COMMAND printf "IN_SOURCE=${IN_SOURCE}, NCPU=${NCPU}, DEVICE=${DEVICE}, DEBUG=${DEBUG}\n")
execute_process(COMMAND printf "DOCKER_TAR=${DOCKER_TAR}, DOCKER_TAR_DIR=${DOCKER_TAR_DIR}, INGEST_METHOD=${INGEST_METHOD},REGISTRY=${REGISTRY}\n\n")

