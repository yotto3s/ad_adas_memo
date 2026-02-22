#!/usr/bin/env bash
# Run clang-tidy on all C++ source files.
#   --fix          apply suggested fixes
#   [build-dir]    path to build directory (default: build/clang-debug)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCANUM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

FIX_ARGS=()
BUILD_DIR="${ARCANUM_DIR}/build/clang-debug"

# Parse arguments
for arg in "$@"; do
  case "${arg}" in
    --fix)
      FIX_ARGS=(--fix)
      ;;
    *)
      BUILD_DIR="${ARCANUM_DIR}/${arg}"
      ;;
  esac
done

# Verify compile_commands.json exists
if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "Error: ${BUILD_DIR}/compile_commands.json not found." >&2
  echo "Run 'cmake --preset clang-debug' first." >&2
  exit 1
fi

# Filter GCC-specific flags that clang-tidy does not understand.
FILTERED_DIR="$(mktemp -d)"
trap 'rm -rf "${FILTERED_DIR}"' EXIT

sed -e 's/-fno-canonical-system-headers//g' \
    -e 's/-fstack-usage//g' \
    "${BUILD_DIR}/compile_commands.json" > "${FILTERED_DIR}/compile_commands.json"

# Find .cpp files in src/ and run clang-tidy in parallel
find "${ARCANUM_DIR}/src" -name '*.cpp' -print0 \
  | xargs -0 -P"$(nproc)" -I{} \
    clang-tidy \
      -p "${FILTERED_DIR}" \
      --warnings-as-errors='*' \
      "${FIX_ARGS[@]}" \
      {}
