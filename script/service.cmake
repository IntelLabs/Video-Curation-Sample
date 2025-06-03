if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/build.sh")
    add_custom_target(build_${service} ALL "${CMAKE_CURRENT_SOURCE_DIR}/build.sh" "${PLATFORM}" "${NCURATIONS}" "${INGESTION}" "${IN_SOURCE}" "${NCPU}" "${REGISTRY}" "${NSTREAMS}" "${DEVICE}" "${DEBUG}" "${DOCKER_TAR}" "${DOCKER_TAR_DIR}")
endif()
