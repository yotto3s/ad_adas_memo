#!/usr/bin/bash

set -eu
set -o pipefail

SCRIPT_FULL_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_FULL_PATH}")
PROJECT_DIR="${SCRIPT_DIR}/../../"

. "${SCRIPT_DIR}/docker_config.sh"

# Mount host .gitconfig if it exists
GITCONFIG_MOUNT=()
if [ -f "${HOME}/.gitconfig" ]; then
    GITCONFIG_MOUNT=(-v "${HOME}/.gitconfig:/home/devuser/.gitconfig:ro")
fi

# Mount host DBus session socket if available
DBUS_SOCKET_PATH="/run/user/$(id -u)/bus"
DBUS_MOUNT=()
DBUS_ENV=()
if [ -S "${DBUS_SOCKET_PATH}" ]; then
    DBUS_MOUNT=(-v "${DBUS_SOCKET_PATH}:${DBUS_SOCKET_PATH}")
    DBUS_ENV=(-e "DBUS_SESSION_BUS_ADDRESS=unix:path=${DBUS_SOCKET_PATH}")
fi

docker run -d \
    --name "${CONTAINER_NAME}" \
    -v "${PROJECT_DIR}:/workspace/ad-adas-memo" \
    -v "${HOME}/.claude:/home/devuser/.claude" \
    -v "${HOME}/.claude.json:/home/devuser/.claude.json" \
    -v "${HOME}/.credentials.json:/home/devuser/.credentials.json" \
    -v "${HOME}/.ssh:/home/devuser/.ssh:ro" \
    ${GITCONFIG_MOUNT[@]+"${GITCONFIG_MOUNT[@]}"} \
    ${DBUS_MOUNT[@]+"${DBUS_MOUNT[@]}"} \
    ${DBUS_ENV[@]+"${DBUS_ENV[@]}"} \
    -e TARGET_UID="$(id -u)" \
    -e TARGET_GID="$(id -g)" \
    -e "TERM=xterm-256color" \
    ${ANTHROPIC_API_KEY:+-e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} \
    ${CLAUDE_CODE_OAUTH_TOKEN:+-e CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN"} \
    -w /workspace/ad-adas-memo \
    "${DEV_IMAGE}" \
    sleep infinity
