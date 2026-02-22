#!/bin/bash
set -e

TARGET_UID=${TARGET_UID:-1000}
TARGET_GID=${TARGET_GID:-1000}
USERNAME=devuser

# Adjust GID if different from default
if [ "$(id -g "$USERNAME")" != "$TARGET_GID" ]; then
    groupmod -g "$TARGET_GID" "$USERNAME"
fi

# Adjust UID if different from default
if [ "$(id -u "$USERNAME")" != "$TARGET_UID" ]; then
    usermod -u "$TARGET_UID" -o "$USERNAME"
fi

# Fix home directory ownership (skip read-only mounts)
find "/home/$USERNAME" -maxdepth 1 -mindepth 1 ! -name .ssh ! -name .gitconfig -exec chown -R "$TARGET_UID:$TARGET_GID" {} +
chown "$TARGET_UID:$TARGET_GID" "/home/$USERNAME"

# Drop to user and exec the command
exec gosu "$USERNAME" "$@"
