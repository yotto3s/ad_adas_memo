#!/usr/bin/env bash
# Run clang-format on all C++ source files.
#   --check   dry-run mode for CI (exits non-zero on diff)
#   (default) auto-fix mode for local development
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCANUM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Determine mode
CHECK_ARGS=()
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ARGS=(--dry-run --Werror)
  shift
fi

# Find all .cpp and .h files in src/ and tests/
find "${ARCANUM_DIR}/src" "${ARCANUM_DIR}/tests" \
  \( -name '*.cpp' -o -name '*.h' \) \
  -print0 \
  | xargs -0 clang-format "${CHECK_ARGS[@]}" -i
