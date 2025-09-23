cmake_minimum_required (VERSION 2.8)

Project(library_curation NONE)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_HOME_DIRECTORY}/dist")

set(EMPTY " ")
if (NOT DEFINED REGISTRY)
    set(REGISTRY "${EMPTY}")
elseif (NOT ${REGISTRY} MATCHES "/$")
    set(REGISTRY "${REGISTRY}/")
endif()

if(MODEL_NAME STREQUAL "")
    set(MODEL_NAME "yolo11n")
endif()

if (MODEL_NAME STREQUAL "yolo11n")
    set(CUSTOM_MODEL_FLAG "False")
else()
    set(CUSTOM_MODEL_FLAG "True")
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

if (NOT DEFINED IN_SOURCE)
    set(IN_SOURCE "videos") # Use videos
endif()

if (NOT DEFINED RESIZE_FLAG)
    set(RESIZE_FLAG "False")
endif()

if(NOT (RESIZE_FLAG STREQUAL "True" OR RESIZE_FLAG STREQUAL "False"))
  set(RESIZE_FLAG "False")
endif()

if (NOT DEFINED OMIT_DETECTIONS_FLAG)
    set(OMIT_DETECTIONS_FLAG "False")
endif()

if(NOT (OMIT_DETECTIONS_FLAG STREQUAL "True" OR OMIT_DETECTIONS_FLAG STREQUAL "False"))
  set(OMIT_DETECTIONS_FLAG "False")
endif()

if (NOT DEFINED NCPU)
    set(NCPU 0)
endif()

if (NOT DEFINED DOCKER_TAR)
    set(DOCKER_TAR "0")
endif()

if (RESIZE_FLAG STREQUAL "True")
    set(DOCKER_TAR_DIR ${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars/resize)
else()
    set(DOCKER_TAR_DIR ${CMAKE_HOME_DIRECTORY}/deployment/DockerImageTars/full)
endif()

if(DOCKER_TAR MATCHES "1")
    set(TAR_FILES
        "${DOCKER_TAR_DIR}/lcc_frontend_stream.tar"
        "${DOCKER_TAR_DIR}/lcc_video_stream.${DEVICE}.tar"
        "${DOCKER_TAR_DIR}/lcc_vdms_stream.tar"
        "${DOCKER_TAR_DIR}/lcc_certificate_stream.tar"
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


file(GLOB dirs "deployment" "*")
list(REMOVE_DUPLICATES dirs)
foreach (dir ${dirs})
    if(EXISTS ${dir}/CMakeLists.txt)
        add_subdirectory(${dir})
    endif()
endforeach()

# legal message
execute_process(COMMAND printf "\n\nThis script will build third party components licensed under various open source licenses into your container images. The terms under which those components may be used and distributed can be found with the license document that is provided with those components. Please familiarize yourself with those terms to ensure your distribution of those components complies with the terms of those licenses.\n\n")
execute_process(COMMAND printf "INGESTION=${INGESTION}, DEVICE=${DEVICE}, DEBUG=${DEBUG}, RESIZE_FLAG=${RESIZE_FLAG}, OMIT_DETECTIONS_FLAG=${OMIT_DETECTIONS_FLAG}\n")
execute_process(COMMAND printf "DOCKER_TAR=${DOCKER_TAR}, DOCKER_TAR_DIR=${DOCKER_TAR_DIR}, MODEL_NAME=${MODEL_NAME}, CUSTOM_MODEL_FLAG=${CUSTOM_MODEL_FLAG}\n")
execute_process(COMMAND printf "PLATFORM=${PLATFORM}, NSTREAMS=${NSTREAMS}, NCURATIONS=${NCURATIONS}, IN_SOURCE=${IN_SOURCE}, NCPU=${NCPU}, REGISTRY=${REGISTRY}\n\n")

