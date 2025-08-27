add_custom_target(start_${service} "${CMAKE_CURRENT_SOURCE_DIR}/start.sh" "${service}" "${PLATFORM}" "${NCURATIONS}" "${INGESTION}" "${IN_SOURCE}" "${NCPU}" "${REGISTRY}" "${NSTREAMS}" "${DEVICE}" "${DEBUG}" "${DOCKER_TAR}" "${DOCKER_TAR_DIR}" "${RESIZE_FLAG}" "${MODEL_NAME}" "${CUSTOM_MODEL_FLAG}" )
add_custom_target(stop_${service} "${CMAKE_CURRENT_SOURCE_DIR}/stop.sh" "${service}")
