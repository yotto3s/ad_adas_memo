#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

. "${SCRIPT_DIR}/docker_config.sh"

docker exec -it -u devuser -w "${ROOT_DIR}" "${CONTAINER_NAME}" bash -l
