# CI Pipeline Implementation Plan

**Goal:** Replace the single `build-and-test` CI job with a multi-stage pipeline covering format checking, static analysis (clang-tidy), a five-variant build matrix with ccache, sanitizers, code coverage with Codecov upload, and two new helper scripts.

**Architecture:** The pipeline has three stages: Stage 1 runs `format-check` and `lint` in parallel with no dependencies; Stage 2 runs a five-variant `build` matrix (clang-debug, clang-release, asan, ubsan, coverage) that depends on Stage 1 passing and uploads build artifacts; Stage 3 runs a matching `test` matrix that downloads artifacts and runs ctest, lit tests, and (for coverage) Codecov upload. All jobs run in the `ghcr.io/yotto3s/arcanum-base:latest` container on `ubuntu-24.04`. Supporting changes include adding ccache/lcov to the Docker image, adding ccache launcher and coverage presets to CMakePresets.json, and creating two helper shell scripts for clang-format and clang-tidy.

**Tech Stack:** GitHub Actions, CMake presets, clang-format-21, clang-tidy-21, ccache, lcov, Codecov, Docker

**Strategy:** Team-driven

---

### Task 1: Add ccache and lcov to Dockerfile.base

**Files:**
- Modify: `arcanum/docker/Dockerfile.base` (line 6-28, the first `apt-get install` block)

**Agent role:** junior-engineer

**Step 1: Add ccache and lcov packages to the apt-get install list**

In `arcanum/docker/Dockerfile.base`, add `ccache` and `lcov` to the first `apt-get install` block (lines 6-28). Insert them alphabetically into the existing package list:

```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    ccache \
    cmake \
    wget \
    flex \
    bison \
    gcc \
    g++ \
    gdb \
    git \
    lcov \
    lsb-release \
    software-properties-common \
    gnupg \
    libzstd-dev \
    libz-dev \
    python3 \
    python3-pip \
    opam \
    pkg-config \
    sudo \
    z3 \
    libgmp-dev \
    && apt-get clean
```

**Step 2: Verify the Dockerfile syntax**

Run: `docker run --rm -i hadolint/hadolint < arcanum/docker/Dockerfile.base 2>&1 || true`

If hadolint is not available, visually confirm the file has no syntax issues.

---

### Task 2: Add ccache launcher and coverage presets to CMakePresets.json

**Files:**
- Modify: `arcanum/CMakePresets.json`

**Agent role:** junior-engineer

**Step 1: Add ccache launcher variables to the base hidden preset**

Add two new cache variables to the existing `"base"` hidden configure preset's `"cacheVariables"` object:

```json
{
  "name": "base",
  "hidden": true,
  "binaryDir": "${sourceDir}/build/${presetName}",
  "cacheVariables": {
    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
    "CMAKE_PREFIX_PATH": "/usr/lib/llvm-21",
    "CMAKE_C_COMPILER_LAUNCHER": "ccache",
    "CMAKE_CXX_COMPILER_LAUNCHER": "ccache"
  }
}
```

**Step 2: Add the coverage configure preset**

Add a new configure preset named `"coverage"` after the existing `"ubsan"` preset:

```json
{
  "name": "coverage",
  "displayName": "Coverage",
  "description": "Clang-21 with code coverage instrumentation",
  "inherits": "base",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Debug",
    "CMAKE_C_COMPILER": "clang-21",
    "CMAKE_CXX_COMPILER": "clang++-21",
    "CMAKE_C_FLAGS": "--coverage",
    "CMAKE_CXX_FLAGS": "--coverage",
    "CMAKE_EXE_LINKER_FLAGS": "--coverage"
  }
}
```

**Step 3: Add the coverage build preset**

Add to `"buildPresets"` array:

```json
{ "name": "coverage", "configurePreset": "coverage" }
```

**Step 4: Add asan, ubsan, and coverage test presets**

Add three new entries to the `"testPresets"` array:

```json
{
  "name": "asan",
  "configurePreset": "asan",
  "output": { "outputOnFailure": true }
},
{
  "name": "ubsan",
  "configurePreset": "ubsan",
  "output": { "outputOnFailure": true }
},
{
  "name": "coverage",
  "configurePreset": "coverage",
  "output": { "outputOnFailure": true }
}
```

**Step 5: Validate JSON syntax**

Run: `python3 -c "import json; json.load(open('arcanum/CMakePresets.json'))"`

---

### Task 3: Create run-clang-format.sh helper script

**Files:**
- Create: `arcanum/scripts/run-clang-format.sh`

**Agent role:** junior-engineer

**Step 1: Create the scripts directory**

Run: `mkdir -p arcanum/scripts`

**Step 2: Write the run-clang-format.sh script**

Create `arcanum/scripts/run-clang-format.sh`:

```bash
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
```

**Step 3: Make executable and verify syntax**

Run: `chmod +x arcanum/scripts/run-clang-format.sh && bash -n arcanum/scripts/run-clang-format.sh`

---

### Task 4: Create run-clang-tidy.sh helper script

**Files:**
- Create: `arcanum/scripts/run-clang-tidy.sh`

**Agent role:** junior-engineer

**Step 1: Write the run-clang-tidy.sh script**

Create `arcanum/scripts/run-clang-tidy.sh`:

```bash
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
```

**Step 2: Make executable and verify syntax**

Run: `chmod +x arcanum/scripts/run-clang-tidy.sh && bash -n arcanum/scripts/run-clang-tidy.sh`

---

### Task 5: Rewrite ci.yml with multi-stage pipeline

**Files:**
- Modify: `.github/workflows/ci.yml` (replace entire contents)

**Agent role:** senior-engineer

**Step 1: Replace the entire ci.yml**

Replace the full contents of `.github/workflows/ci.yml` with:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  # ── Stage 1: Quality gates (no build required) ─────────────────────
  format-check:
    runs-on: ubuntu-24.04
    name: Format Check
    container:
      image: ghcr.io/yotto3s/arcanum-base:latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: clang-format (check mode)
        run: ./arcanum/scripts/run-clang-format.sh --check

  lint:
    runs-on: ubuntu-24.04
    name: Lint (clang-tidy)
    container:
      image: ghcr.io/yotto3s/arcanum-base:latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure (clang-debug for compile_commands.json)
        run: |
          cd arcanum
          cmake --preset clang-debug

      - name: clang-tidy
        run: ./arcanum/scripts/run-clang-tidy.sh build/clang-debug

  # ── Stage 2: Build matrix ──────────────────────────────────────────
  build:
    runs-on: ubuntu-24.04
    name: Build (${{ matrix.preset }})
    needs: [format-check, lint]
    container:
      image: ghcr.io/yotto3s/arcanum-base:latest
    strategy:
      fail-fast: false
      matrix:
        preset: [clang-debug, clang-release, asan, ubsan, coverage]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Restore ccache
        uses: actions/cache@v4
        with:
          path: ~/.cache/ccache
          key: ccache-${{ matrix.preset }}-${{ hashFiles('arcanum/src/**', 'arcanum/CMakeLists.txt') }}
          restore-keys: |
            ccache-${{ matrix.preset }}-

      - name: Configure
        run: |
          cd arcanum
          cmake --preset ${{ matrix.preset }}

      - name: Build
        run: |
          cd arcanum
          cmake --build build/${{ matrix.preset }}

      - name: Upload build artifact
        uses: actions/upload-artifact@v4
        with:
          name: build-${{ matrix.preset }}
          path: arcanum/build/${{ matrix.preset }}
          retention-days: 1

  # ── Stage 3: Test matrix ───────────────────────────────────────────
  test:
    runs-on: ubuntu-24.04
    name: Test (${{ matrix.preset }})
    needs: [build]
    container:
      image: ghcr.io/yotto3s/arcanum-base:latest
    strategy:
      fail-fast: false
      matrix:
        preset: [clang-debug, clang-release, asan, ubsan, coverage]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Download build artifact
        uses: actions/download-artifact@v4
        with:
          name: build-${{ matrix.preset }}
          path: arcanum/build/${{ matrix.preset }}

      - name: Fix artifact permissions
        run: |
          chmod -R +x arcanum/build/${{ matrix.preset }}/bin/ || true

      - name: Run unit tests
        run: |
          cd arcanum
          ctest --preset ${{ matrix.preset }}

      - name: Run lit tests
        run: |
          cd arcanum
          cmake --build build/${{ matrix.preset }} --target check-arcanum-lit

      - name: Create llvm-gcov wrapper
        if: matrix.preset == 'coverage'
        run: |
          cat > arcanum/llvm-gcov.sh << 'GCOV_EOF'
          #!/usr/bin/env bash
          exec llvm-cov-21 gcov "$@"
          GCOV_EOF
          chmod +x arcanum/llvm-gcov.sh

      - name: Generate coverage report
        if: matrix.preset == 'coverage'
        run: |
          cd arcanum
          lcov --capture \
            --directory build/coverage \
            --output-file build/coverage/coverage.info \
            --gcov-tool "$(pwd)/llvm-gcov.sh" \
            --ignore-errors mismatch
          lcov --remove build/coverage/coverage.info \
            '/usr/*' \
            '*/tests/*' \
            '*/build/*' \
            --output-file build/coverage/coverage.info

      - name: Upload to Codecov
        if: matrix.preset == 'coverage'
        uses: codecov/codecov-action@v4
        with:
          files: arcanum/build/coverage/coverage.info
          fail_ci_if_error: false
        env:
          CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
```

**Step 2: Validate YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`

---

## Execution: Team-Driven

**Fragments:** 3

### Team Composition

| Name | Type | Scope |
|------|------|-------|
| ci-reviewer-1 | code-reviewer | All fragments |
| ci-junior-engineer-1 | junior-engineer | Fragment 1 (Tasks 1, 2) |
| ci-junior-engineer-2 | junior-engineer | Fragment 2 (Tasks 3, 4) |
| ci-senior-engineer-1 | senior-engineer | Fragment 3 (Task 5) |

### Fragment 1: Docker and CMakePresets changes
- **Tasks:** Task 1, Task 2
- **File scope:** `arcanum/docker/Dockerfile.base`, `arcanum/CMakePresets.json`
- **Agent:** ci-junior-engineer-1
- **Dependencies:** none

### Fragment 2: Helper scripts
- **Tasks:** Task 3, Task 4
- **File scope:** `arcanum/scripts/run-clang-format.sh`, `arcanum/scripts/run-clang-tidy.sh`
- **Agent:** ci-junior-engineer-2
- **Dependencies:** none

### Fragment 3: CI workflow rewrite
- **Tasks:** Task 5
- **File scope:** `.github/workflows/ci.yml`
- **Agent:** ci-senior-engineer-1
- **Dependencies:** Fragments 1 and 2 must complete first (workflow references scripts and presets)

### Review Protocol

Each fragment gets a two-stage review by ci-reviewer-1:
1. **Spec compliance** — Does the output match the design doc?
2. **Code quality** — Formatting, error handling, no regressions

Both stages must PASS before a fragment is merge-ready.
