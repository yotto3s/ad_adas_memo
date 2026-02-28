#!/usr/bin/bash

set -eux
set -o pipefail

SCRIPT_DIR=$(dirname "$0")

. "${SCRIPT_DIR}/docker_config.sh"

docker pull "${BASE_IMAGE}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.dev" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${DEV_IMAGE}" \
    "${SCRIPT_DIR}"
