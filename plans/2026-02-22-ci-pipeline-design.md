# CI Pipeline Design

## Goal

Replace the single `build-and-test` CI job with a multi-stage pipeline covering format checking, static analysis, multi-config builds, sanitizers, and code coverage.

## Pipeline Structure

```
Stage 1 (parallel):  format-check, lint
Stage 2 (parallel):  build matrix (needs stage 1) → upload artifacts
Stage 3 (parallel):  test matrix (needs stage 2) → download artifacts, run tests
```

### Jobs

| Job | Matrix | Depends on | Key actions |
|-----|--------|------------|-------------|
| `format-check` | — | — | `find` + `clang-format-21 --dry-run --Werror` on `arcanum/src/` and `arcanum/tests/` |
| `lint` | — | — | Configure `clang-debug` preset, run `clang-tidy-21 -p build/clang-debug` on `arcanum/src/*.cpp` |
| `build` | clang-debug, clang-release, asan, ubsan, coverage | format-check, lint | `cmake --preset` + `cmake --build`, upload `build/<preset>` as artifact |
| `test` | clang-debug, clang-release, asan, ubsan, coverage | build | Download artifact, `ctest --preset`, lit tests, Codecov upload (coverage variant only) |

All jobs run on `ubuntu-24.04` in the `ghcr.io/yotto3s/arcanum-base:latest` container.

### Triggers

Same as current: push to `main`, pull requests, manual dispatch.

## File Changes

### 1. `arcanum/CMakePresets.json`

Changes:

- Add `CMAKE_C_COMPILER_LAUNCHER: ccache` and `CMAKE_CXX_COMPILER_LAUNCHER: ccache` to the `base` hidden preset

**New configure presets:**
- `coverage` — Clang-21, Debug, `--coverage` in C/CXX/linker flags

**Build presets:**
- `coverage`

**Test presets:**
- `asan` — references `asan` configure preset
- `ubsan` — references `ubsan` configure preset
- `coverage` — references `coverage` configure preset

### 2. `arcanum/scripts/run-clang-format.sh`

New helper script adapted from polang. Features:
- `--check` flag for CI (dry-run + Werror), no flag for local auto-fix
- Targets all `.cpp` and `.h` files in `src/` and `tests/`
- Uses `clang-format` (resolved via alternatives to clang-format-21)

### 3. `arcanum/scripts/run-clang-tidy.sh`

New helper script adapted from polang. Features:
- Accepts optional build directory argument (defaults to `build/clang-debug`)
- `--fix` flag for local auto-fix
- Filters GCC-specific flags from `compile_commands.json`
- Runs in parallel via `xargs -P$(nproc)`
- `--warnings-as-errors='*'` for strict CI enforcement
- Targets `.cpp` files in `src/`

### 4. `.github/workflows/ci.yml`

Replace the single `build-and-test` job with:

**`format-check` job:**
- Checkout code
- Run `./arcanum/scripts/run-clang-format.sh --check`

**`lint` job:**
- Checkout code
- `cd arcanum && cmake --preset clang-debug` (to generate `compile_commands.json`)
- Run `./arcanum/scripts/run-clang-tidy.sh build/clang-debug`

**`build` job (matrix: [clang-debug, clang-release, asan, ubsan, coverage]):**
- Checkout code
- Restore ccache from `actions/cache@v4`
- `cmake --preset ${{ matrix.preset }}`
- `cmake --build build/${{ matrix.preset }}`
- Upload `arcanum/build/${{ matrix.preset }}` as artifact named `build-${{ matrix.preset }}`

**`test` job (matrix: [clang-debug, clang-release, asan, ubsan, coverage]):**
- Checkout code
- Download artifact `build-${{ matrix.preset }}`
- `ctest --preset ${{ matrix.preset }}`
- `cmake --build build/${{ matrix.preset }} --target check-arcanum-lit`
- If coverage variant: generate lcov report + `codecov/codecov-action@v4`

### 5. `arcanum/docker/Dockerfile.base`

Add `ccache` and `lcov` to the `apt-get install` list. This requires rebuilding and pushing the base Docker image.

### 6. No other files changed

`.clang-format`, `.clang-tidy`, `CMakeLists.txt` remain unchanged.

## ccache

Speed up CI builds by caching compiled objects across runs.

- Add `ccache` to `Dockerfile.base` apt packages
- Add `CMAKE_C_COMPILER_LAUNCHER: ccache` and `CMAKE_CXX_COMPILER_LAUNCHER: ccache` to the `base` hidden preset in `CMakePresets.json`
- In the `build` job: use `actions/cache@v4` to persist `~/.cache/ccache` between runs
- Cache key: `ccache-${{ matrix.preset }}-${{ hashFiles('arcanum/src/**', 'arcanum/CMakeLists.txt') }}`
- Restore key fallback: `ccache-${{ matrix.preset }}-` (partial match for older caches)

## Coverage Details

- Compiler flags: `--coverage` (gcov-compatible)
- After tests: use `lcov` to collect coverage data from `build/coverage`
- Upload via `codecov/codecov-action@v4`
- Requires `CODECOV_TOKEN` repository secret

## Sanitizer Environment

Already configured in CMakePresets.json:
- ASAN: `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`
- UBSAN: `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`
