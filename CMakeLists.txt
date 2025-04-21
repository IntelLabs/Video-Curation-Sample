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

if (NOT DEFINED IN_SOURCE)
    set(IN_SOURCE "videos") # Use videos
    set(STREAM_URL "${EMPTY}")
endif()

if (NOT DEFINED NCPU)
    set(NCPU 0)
endif()

if (NOT IN_SOURCE MATCHES "stream")
    set(STREAM_URL "${EMPTY}") # Use videos
endif()

if (NOT DEFINED STREAM_URL)
    set(STREAM_URL "${EMPTY}") # Use videos
    if (IN_SOURCE MATCHES "stream")
        set(IN_SOURCE "videos")
    endif()
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
execute_process(COMMAND printf "IN_SOURCE=${IN_SOURCE}, STREAM_URL=${STREAM_URL}, NCPU=${NCPU}\n")
execute_process(COMMAND printf "REGISTRY=${REGISTRY}\n\n")

