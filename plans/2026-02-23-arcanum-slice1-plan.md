# Arcanum Slice 1 Implementation Plan -- Minimal End-to-End Formal Verification Pipeline

**Goal:** Wire all 8 pipeline stages end-to-end for the smallest useful C++ subset (`int32_t`, `bool`, basic arithmetic, `if`/`else`, `requires`/`ensures`/`\result`), proving overflow safety and postcondition correctness.

**Architecture:** An 8-stage pipeline: Clang Frontend parses C++ into an AST, SubsetEnforcer validates allowed constructs, ContractParser extracts `//@ ` annotations, Lowering translates to a custom Arc MLIR dialect, PassManager runs the MLIR verifier, WhyMLEmitter emits `.mlw` files, Why3Runner invokes `why3 prove -P z3`, and ReportGenerator formats results. Docker provides the LLVM 21 + MLIR + Why3 + Z3 toolchain.

**Tech Stack:** C++23, CMake 3.20+, LLVM/Clang/MLIR 21, MLIR TableGen, GoogleTest (FetchContent), LLVM lit + FileCheck, Why3, Z3, Docker (Ubuntu 24.04), GitHub Actions

**Strategy:** Team-driven

---

### Task 1: Docker Environment

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/docker/Dockerfile.base`
- Create: `/home/yotto/ad-adas-memo/arcanum/docker/Dockerfile.dev`
- Create: `/home/yotto/ad-adas-memo/arcanum/docker/docker_config.sh`
- Create: `/home/yotto/ad-adas-memo/arcanum/docker/docker_build.sh`
- Create: `/home/yotto/ad-adas-memo/arcanum/docker/docker_run.sh`
- Create: `/home/yotto/ad-adas-memo/arcanum/docker/entrypoint.sh`

**Agent role:** senior-engineer

**Step 1: Create `docker_config.sh` with image names and paths**

```bash
#!/usr/bin/bash
export BASE_IMAGE=ghcr.io/yotto3s/arcanum-base:latest
export DEV_IMAGE=arcanum-dev
export CONTAINER_NAME="arcanum-dev"
export ROOT_DIR=/workspace/ad-adas-memo
export BUILD_DIR=${ROOT_DIR}/arcanum/build
```

**Step 2: Create `Dockerfile.base` with LLVM 21, MLIR, Why3, and Z3**

```dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essential development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    flex \
    bison \
    gcc \
    g++ \
    gdb \
    git \
    sudo \
    lsb-release \
    software-properties-common \
    gnupg \
    libzstd-dev \
    libz-dev \
    python3 \
    python3-pip \
    opam \
    z3 \
    && apt-get clean

# Install LLVM 21 via llvm.sh
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh 21 all && \
    rm llvm.sh

# Install MLIR development packages
RUN apt-get update && apt-get install -y \
    libmlir-21-dev \
    mlir-21-tools \
    && apt-get clean

# Configure LLVM library path for dynamic linker
RUN echo "/usr/lib/llvm-21/lib" > /etc/ld.so.conf.d/llvm-21.conf && ldconfig

# Register clang alternatives
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100 \
    --slave /usr/bin/clang++ clang++ /usr/bin/clang++-21 \
    --slave /usr/bin/clang-format clang-format /usr/bin/clang-format-21 \
    --slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-21 \
    --slave /usr/bin/clangd clangd /usr/bin/clangd-21

# Install Why3 via opam
RUN opam init --disable-sandboxing --yes && \
    eval $(opam env) && \
    opam install why3 --yes && \
    cp $(opam var bin)/why3 /usr/local/bin/why3

# Configure Why3 to use Z3
RUN why3 config detect

# Remove default ubuntu user
RUN touch /var/mail/ubuntu && chown ubuntu /var/mail/ubuntu && userdel -r ubuntu

# Create default devuser (UID/GID 1000) with sudo privileges
RUN groupadd --gid 1000 devuser \
    && useradd --uid 1000 --gid 1000 -m devuser \
    && echo "devuser ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/devuser \
    && chmod 0440 /etc/sudoers.d/devuser

# Set the working directory
WORKDIR /workspace
RUN chown 1000:1000 /workspace

CMD ["/bin/bash"]
```

**Step 3: Create `Dockerfile.dev` extending base with dev tools**

```dockerfile
ARG BASE_IMAGE=ghcr.io/yotto3s/arcanum-base:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Entrypoint dependency + essential tools
RUN apt-get update && apt-get install -y \
    gosu \
    curl \
    jq \
    libnotify-bin \
    ripgrep \
    strace \
    valgrind \
    nano \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Claude Code (official standalone installer)
RUN curl -fsSL https://claude.ai/install.sh | bash -s latest \
    && cp -L /root/.local/bin/claude /usr/local/bin/claude \
    && rm -rf /root/.local/bin/claude /root/.claude

# gh CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y gh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
CMD ["/bin/bash"]
```

**Step 4: Create `entrypoint.sh` with UID/GID matching**

```bash
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
```

**Step 5: Create `docker_build.sh`**

```bash
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
```

**Step 6: Create `docker_run.sh` with project mount and credential forwarding**

```bash
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
```

**Step 7: Make shell scripts executable and commit**

Run:
```bash
chmod +x /home/yotto/ad-adas-memo/arcanum/docker/docker_config.sh
chmod +x /home/yotto/ad-adas-memo/arcanum/docker/docker_build.sh
chmod +x /home/yotto/ad-adas-memo/arcanum/docker/docker_run.sh
chmod +x /home/yotto/ad-adas-memo/arcanum/docker/entrypoint.sh
```

**Step 8: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/docker/Dockerfile.base arcanum/docker/Dockerfile.dev arcanum/docker/docker_config.sh arcanum/docker/docker_build.sh arcanum/docker/docker_run.sh arcanum/docker/entrypoint.sh
git commit -m "feat(arcanum): add Docker environment with LLVM 21, MLIR, Why3, Z3"
```

**Step 9: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Dockerfile.base installs LLVM 21, MLIR dev packages, Why3, Z3 | All packages listed, ldconfig configured | |
| Dockerfile.dev extends base with gosu, Claude Code, gh CLI | All dev tools present | |
| entrypoint.sh handles UID/GID matching | gosu exec pattern present | |
| docker_build.sh pulls base and builds dev | Correct build flow | |
| docker_run.sh mounts project, .claude, .ssh, git config | All volumes correct | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 2: Project Scaffold -- CMake, Presets, Code Style, .gitignore

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/CMakeLists.txt`
- Create: `/home/yotto/ad-adas-memo/arcanum/CMakePresets.json`
- Create: `/home/yotto/ad-adas-memo/arcanum/.clang-format`
- Create: `/home/yotto/ad-adas-memo/arcanum/.clang-tidy`
- Create: `/home/yotto/ad-adas-memo/arcanum/.gitignore`
- Create: `/home/yotto/ad-adas-memo/arcanum/README.md`

**Agent role:** senior-engineer

**Step 1: Create `/home/yotto/ad-adas-memo/arcanum/.clang-format`**

```yaml
BasedOnStyle: LLVM
DerivePointerAlignment: false
PointerAlignment: Left
```

**Step 2: Create `/home/yotto/ad-adas-memo/arcanum/.clang-tidy`**

```yaml
---
Checks: >
  -*,
  bugprone-*,
  performance-*,
  modernize-*,
  readability-*,
  -bugprone-easily-swappable-parameters,
  -modernize-use-trailing-return-type,
  -readability-identifier-length,
  -readability-function-cognitive-complexity,
  -readability-convert-member-functions-to-static,
  -performance-enum-size

HeaderFilterRegex: '^.*/arcanum/src/.*\.h$'

WarningsAsErrors: ''

CheckOptions:
  - key: readability-braces-around-statements.ShortStatementLines
    value: '0'
  - key: readability-implicit-bool-conversion.AllowIntegerConditions
    value: false
  - key: readability-implicit-bool-conversion.AllowPointerConditions
    value: false
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.StructCase
    value: CamelCase
  - key: readability-identifier-naming.EnumCase
    value: CamelCase
  - key: readability-identifier-naming.EnumConstantCase
    value: CamelCase
  - key: readability-identifier-naming.TypeAliasCase
    value: CamelCase
  - key: readability-identifier-naming.TypedefCase
    value: CamelCase
  - key: readability-identifier-naming.TemplateParameterCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: camelBack
  - key: readability-identifier-naming.MethodCase
    value: camelBack
  - key: readability-identifier-naming.ParameterCase
    value: camelBack
  - key: readability-identifier-naming.LocalVariableCase
    value: camelBack
  - key: readability-identifier-naming.LocalConstantCase
    value: camelBack
  - key: readability-identifier-naming.MemberCase
    value: camelBack
  - key: readability-identifier-naming.PrivateMemberCase
    value: camelBack
  - key: readability-identifier-naming.ProtectedMemberCase
    value: camelBack
  - key: readability-identifier-naming.GlobalConstantCase
    value: UPPER_CASE
  - key: readability-identifier-naming.StaticConstantCase
    value: UPPER_CASE
  - key: readability-identifier-naming.ConstexprVariableCase
    value: UPPER_CASE
  - key: readability-identifier-naming.NamespaceCase
    value: lower_case
  - key: readability-magic-numbers.IgnoredIntegerValues
    value: '0;1;2;-1'
  - key: bugprone-unused-return-value.CheckedFunctions
    value: ''
```

**Step 3: Create `/home/yotto/ad-adas-memo/arcanum/.gitignore`**

```
# Build directory
build*/

# Compiled objects
*.o
*.a
*.so

# Editor/IDE
*.swp
*.swo
*~
.vscode/
.idea/

# compile_commands.json (generated by CMake)
compile_commands.json

# Temporary WhyML files
*.mlw
```

**Step 4: Create `/home/yotto/ad-adas-memo/arcanum/CMakeLists.txt` -- root CMake with MLIR/Clang/GoogleTest**

```cmake
cmake_minimum_required(VERSION 3.20)
project(Arcanum LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Common warning flags
add_compile_options(-Wall -Wextra -Werror)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Find MLIR (which includes LLVM) ---
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Store LLVM/MLIR include directories for use by targets that need them.
# We DON'T add these globally to avoid LLVM's cxxabi.h conflicting with
# system headers when building GoogleTest.
set(ARCANUM_LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
set(ARCANUM_MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS})

# Project include directories (for TableGen outputs and dialect headers)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# --- Find Clang for LibTooling ---
find_package(Clang REQUIRED CONFIG)
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
set(ARCANUM_CLANG_INCLUDE_DIRS ${CLANG_INCLUDE_DIRS})

# --- TableGen for Arc dialect ---
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/dialect)

# Generate dialect declarations/definitions from ArcOps.td
set(LLVM_TARGET_DEFINITIONS src/dialect/ArcOps.td)
mlir_tablegen(src/dialect/ArcOps.h.inc -gen-op-decls)
mlir_tablegen(src/dialect/ArcOps.cpp.inc -gen-op-defs)
mlir_tablegen(src/dialect/ArcDialect.h.inc -gen-dialect-decls)
mlir_tablegen(src/dialect/ArcDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ArcOpsIncGen)

# Generate type declarations/definitions from ArcTypes.td
set(LLVM_TARGET_DEFINITIONS src/dialect/ArcTypes.td)
mlir_tablegen(src/dialect/ArcTypes.h.inc -gen-typedef-decls)
mlir_tablegen(src/dialect/ArcTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(ArcTypesIncGen)

# --- Arc dialect library ---
add_mlir_dialect_library(ArcDialect
  src/dialect/ArcDialect.cpp
  src/dialect/ArcOps.cpp
  src/dialect/ArcTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/src/dialect

  DEPENDS
  ArcOpsIncGen
  ArcTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
)
target_include_directories(ArcDialect PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)

# --- Frontend library (SubsetEnforcer + ContractParser) ---
add_library(ArcanumFrontend
  src/frontend/SubsetEnforcer.cpp
  src/frontend/ContractParser.cpp
)
target_include_directories(ArcanumFrontend PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(ArcanumFrontend PUBLIC
  clangAST
  clangBasic
  clangFrontend
  clangTooling
  clangASTMatchers
)

# --- Lowering library ---
add_library(ArcanumLowering
  src/dialect/Lowering.cpp
)
target_include_directories(ArcanumLowering PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(ArcanumLowering PUBLIC
  ArcDialect
  ArcanumFrontend
  MLIRIR
)

# --- Passes library ---
add_library(ArcanumPasses
  src/passes/Passes.cpp
)
target_include_directories(ArcanumPasses PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(ArcanumPasses PUBLIC
  ArcDialect
  MLIRPass
  MLIRIR
)

# --- Backend library (WhyMLEmitter + Why3Runner) ---
add_library(ArcanumBackend
  src/backend/WhyMLEmitter.cpp
  src/backend/Why3Runner.cpp
)
target_include_directories(ArcanumBackend PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(ArcanumBackend PUBLIC
  ArcDialect
  MLIRIR
  LLVMSupport
)

# --- Report library ---
add_library(ArcanumReport
  src/report/ReportGenerator.cpp
)
target_include_directories(ArcanumReport PUBLIC
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(ArcanumReport PUBLIC
  LLVMSupport
)

# --- CLI executable ---
add_executable(arcanum src/main.cpp)
target_include_directories(arcanum PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(arcanum PRIVATE
  ArcanumFrontend
  ArcanumLowering
  ArcanumPasses
  ArcanumBackend
  ArcanumReport
  ArcDialect
  clangTooling
  clangFrontend
  clangSerialization
  clangDriver
  clangParse
  clangSema
  clangAnalysis
  clangEdit
  clangAST
  clangLex
  clangBasic
  LLVMSupport
  MLIRIR
  MLIRPass
)

# --- Testing ---
enable_testing()
add_subdirectory(tests)
```

**Step 5: Create `/home/yotto/ad-adas-memo/arcanum/CMakePresets.json`**

```json
{
  "version": 3,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 20,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "base",
      "hidden": true,
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_PREFIX_PATH": "/usr/lib/llvm-21"
      }
    },
    {
      "name": "default",
      "displayName": "Default (Debug)",
      "description": "Quick local development build with system default compiler",
      "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug"
      }
    },
    {
      "name": "clang-debug",
      "displayName": "Clang Debug",
      "description": "Clang-21 compiler, Debug mode",
      "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_C_COMPILER": "clang-21",
        "CMAKE_CXX_COMPILER": "clang++-21"
      }
    },
    {
      "name": "clang-release",
      "displayName": "Clang Release",
      "description": "Clang-21 compiler, Release mode",
      "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_C_COMPILER": "clang-21",
        "CMAKE_CXX_COMPILER": "clang++-21"
      }
    },
    {
      "name": "asan",
      "displayName": "AddressSanitizer",
      "description": "Clang-21 with AddressSanitizer",
      "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_C_COMPILER": "clang-21",
        "CMAKE_CXX_COMPILER": "clang++-21",
        "CMAKE_C_FLAGS": "-fsanitize=address -fno-omit-frame-pointer -g",
        "CMAKE_CXX_FLAGS": "-fsanitize=address -fno-omit-frame-pointer -g",
        "CMAKE_EXE_LINKER_FLAGS": "-fsanitize=address"
      },
      "environment": {
        "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1"
      }
    },
    {
      "name": "ubsan",
      "displayName": "UndefinedBehaviorSanitizer",
      "description": "Clang-21 with UndefinedBehaviorSanitizer",
      "inherits": "base",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_C_COMPILER": "clang-21",
        "CMAKE_CXX_COMPILER": "clang++-21",
        "CMAKE_C_FLAGS": "-fsanitize=undefined -fno-omit-frame-pointer -g",
        "CMAKE_CXX_FLAGS": "-fsanitize=undefined -fno-omit-frame-pointer -g",
        "CMAKE_EXE_LINKER_FLAGS": "-fsanitize=undefined"
      },
      "environment": {
        "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"
      }
    }
  ],
  "buildPresets": [
    { "name": "default", "configurePreset": "default" },
    { "name": "clang-debug", "configurePreset": "clang-debug" },
    { "name": "clang-release", "configurePreset": "clang-release" },
    { "name": "asan", "configurePreset": "asan" },
    { "name": "ubsan", "configurePreset": "ubsan" }
  ],
  "testPresets": [
    {
      "name": "default",
      "configurePreset": "default",
      "output": { "outputOnFailure": true }
    },
    {
      "name": "clang-debug",
      "configurePreset": "clang-debug",
      "output": { "outputOnFailure": true }
    },
    {
      "name": "clang-release",
      "configurePreset": "clang-release",
      "output": { "outputOnFailure": true }
    }
  ]
}
```

**Step 6: Create `/home/yotto/ad-adas-memo/arcanum/README.md`**

```markdown
# Arcanum

Formal verification tool for a safe C++ subset targeting safety-critical automotive software (ISO 26262).

## Prerequisites

Build inside the Docker environment:

```bash
cd arcanum/docker
./docker_build.sh
./docker_run.sh
docker exec -it arcanum-dev bash
```

## Build

```bash
cd /workspace/ad-adas-memo/arcanum
cmake --preset default
cmake --build build/default
```

## Test

```bash
ctest --preset default
```

## Usage (Slice 1)

```bash
./build/default/bin/arcanum --mode=verify input.cpp
```
```

**Step 7: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/CMakeLists.txt arcanum/CMakePresets.json arcanum/.clang-format arcanum/.clang-tidy arcanum/.gitignore arcanum/README.md
git commit -m "feat(arcanum): add CMake scaffold with MLIR/Clang TableGen, presets, code style"
```

**Step 8: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| CMakeLists.txt finds MLIR, Clang, sets up TableGen, GoogleTest | All targets defined, scoped includes | |
| CMakePresets.json has default, clang-debug, clang-release, asan, ubsan | All presets with LLVM-21 prefix path | |
| LLVM/MLIR include dirs scoped per-target (not global) | Avoids GoogleTest header conflicts | |
| .clang-format and .clang-tidy follow LLVM conventions | CamelCase types, camelBack functions/vars | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 3: CLI Skeleton + Clang Frontend (Stages 0-1)

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/src/main.cpp`

**Agent role:** senior-engineer

**Step 1: Create `src/main.cpp` -- CLI argument parsing and Clang frontend wiring**

```cpp
#include "frontend/ContractParser.h"
#include "frontend/SubsetEnforcer.h"
#include "dialect/Lowering.h"
#include "passes/Passes.h"
#include "backend/WhyMLEmitter.h"
#include "backend/Why3Runner.h"
#include "report/ReportGenerator.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"

#include <string>

using namespace llvm;
using namespace clang::tooling;

static cl::OptionCategory arcanumCategory("Arcanum options");

static cl::opt<std::string> mode(
    "mode",
    cl::desc("Operating mode (verify)"),
    cl::init("verify"),
    cl::cat(arcanumCategory));

static cl::opt<std::string> why3Path(
    "why3-path",
    cl::desc("Path to why3 binary"),
    cl::init("why3"),
    cl::cat(arcanumCategory));

static cl::opt<int> timeout(
    "timeout",
    cl::desc("Per-obligation timeout in seconds"),
    cl::init(30),
    cl::cat(arcanumCategory));

int main(int argc, const char** argv) {
  auto expectedParser =
      CommonOptionsParser::create(argc, argv, arcanumCategory);
  if (!expectedParser) {
    llvm::errs() << expectedParser.takeError();
    return 5;
  }
  CommonOptionsParser& optionsParser = expectedParser.get();

  if (mode != "verify") {
    llvm::errs() << "error: unsupported mode '" << mode
                 << "' (only 'verify' is supported in Slice 1)\n";
    return 5;
  }

  const auto& sourceFiles = optionsParser.getSourcePathList();
  if (sourceFiles.empty()) {
    llvm::errs() << "error: no input files\n";
    return 5;
  }

  // Validate input files exist
  for (const auto& file : sourceFiles) {
    if (!llvm::sys::fs::exists(file)) {
      llvm::errs() << "error: file not found: " << file << "\n";
      return 5;
    }
  }

  // Stage 1: Clang Frontend — parse source into AST
  ClangTool tool(optionsParser.getCompilations(),
                 optionsParser.getSourcePathList());
  tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-fparse-all-comments", ArgumentInsertPosition::BEGIN));

  // Stages 2-8 will be wired here
  // For now, the AST is captured in the FrontendAction and passed forward.

  arcanum::ArcanumFrontendAction action;
  auto result = tool.run(newFrontendActionFactory(&action).get());
  if (result != 0) {
    return 4; // Parse error
  }

  // Stage 2: Subset Enforcer
  auto enforceResult = arcanum::enforceSubset(action.getASTContext());
  if (!enforceResult.passed) {
    for (const auto& diag : enforceResult.diagnostics) {
      llvm::errs() << diag << "\n";
    }
    return 3;
  }

  // Stage 3: Contract Parser
  auto contracts = arcanum::parseContracts(action.getASTContext());

  // Stage 4: Arc MLIR Lowering
  mlir::MLIRContext mlirContext;
  auto arcModule =
      arcanum::lowerToArc(mlirContext, action.getASTContext(), contracts);
  if (!arcModule) {
    llvm::errs() << "error: lowering to Arc MLIR failed\n";
    return 5;
  }

  // Stage 5: MLIR Pass Manager
  if (arcanum::runPasses(*arcModule).failed()) {
    llvm::errs() << "error: MLIR verification failed\n";
    return 5;
  }

  // Stage 6: WhyML Emitter
  auto whymlResult = arcanum::emitWhyML(*arcModule);
  if (!whymlResult) {
    llvm::errs() << "error: WhyML emission failed\n";
    return 5;
  }

  // Stage 7: Why3 Runner
  auto obligations =
      arcanum::runWhy3(whymlResult->filePath, why3Path, timeout);

  // Stage 8: Report Generator
  auto report =
      arcanum::generateReport(obligations, whymlResult->locationMap);
  llvm::outs() << report.text << "\n";

  return report.allPassed ? 0 : 1;
}
```

Note: The actual `ArcanumFrontendAction` class and the exact wiring will be refined as part of Tasks 4 and 5. This skeleton establishes the pipeline orchestration. The `ArcanumFrontendAction` will be a custom `ASTFrontendAction` that stores the `ASTContext*` for the pipeline.

**Step 2: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/src/main.cpp
git commit -m "feat(arcanum): add CLI skeleton with 8-stage pipeline orchestration"
```

**Step 3: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| CLI uses `cl::opt` for --mode, --why3-path, --timeout | LLVM command-line parsing | |
| Pipeline stages 1-8 called sequentially | Correct data flow between stages | |
| Exit codes match spec (0=pass, 1=fail, 3=subset, 4=parse, 5=tool error) | All paths return correct code | |
| `-fparse-all-comments` enabled for annotation retention | Argument adjuster added | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 4: Stage 2 -- Subset Enforcer

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/src/frontend/SubsetEnforcer.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/frontend/SubsetEnforcer.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/unit/SubsetEnforcerTest.cpp`

**Agent role:** senior-engineer

**Step 1: Write `SubsetEnforcer.h` -- interface**

```cpp
#ifndef ARCANUM_FRONTEND_SUBSETENFORCER_H
#define ARCANUM_FRONTEND_SUBSETENFORCER_H

#include "clang/AST/ASTContext.h"
#include <string>
#include <vector>

namespace arcanum {

struct SubsetResult {
  bool passed = true;
  std::vector<std::string> diagnostics;
};

/// Walk the Clang AST and reject any constructs outside the Slice 1 subset.
/// Allowed: int32_t, bool, non-template non-recursive functions with single
/// return, variable declarations, assignments, if/else, return,
/// +, -, *, /, %, comparisons, &&, ||, !.
SubsetResult enforceSubset(clang::ASTContext& context);

} // namespace arcanum

#endif // ARCANUM_FRONTEND_SUBSETENFORCER_H
```

**Step 2: Write the failing test `SubsetEnforcerTest.cpp`**

```cpp
#include "frontend/SubsetEnforcer.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>
#include <string>

namespace arcanum {
namespace {

/// Helper: parse source string into ASTContext and run enforceSubset.
SubsetResult checkSubset(const std::string& code) {
  // Use Clang tooling to parse the code
  std::unique_ptr<clang::ASTUnit> ast = clang::tooling::buildASTFromCode(
      code, "test.cpp", std::make_shared<clang::PCHContainerOperations>());
  EXPECT_NE(ast, nullptr);
  return enforceSubset(ast->getASTContext());
}

TEST(SubsetEnforcerTest, AcceptsInt32ArithmeticFunction) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t add(int32_t a, int32_t b) { return a + b; }
  )");
  EXPECT_TRUE(result.passed);
  EXPECT_TRUE(result.diagnostics.empty());
}

TEST(SubsetEnforcerTest, AcceptsBoolFunction) {
  auto result = checkSubset(R"(
    #include <cstdint>
    bool isPositive(int32_t x) { return x > 0; }
  )");
  EXPECT_TRUE(result.passed);
}

TEST(SubsetEnforcerTest, AcceptsIfElse) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t abs(int32_t x) {
      if (x < 0) {
        return -x;
      } else {
        return x;
      }
    }
  )");
  EXPECT_TRUE(result.passed);
}

TEST(SubsetEnforcerTest, RejectsVirtualFunction) {
  auto result = checkSubset(R"(
    class Base {
    public:
      virtual int foo() { return 0; }
    };
  )");
  EXPECT_FALSE(result.passed);
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_NE(result.diagnostics[0].find("virtual"), std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsRawPointer) {
  auto result = checkSubset(R"(
    void foo(int* p) {}
  )");
  EXPECT_FALSE(result.passed);
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_NE(result.diagnostics[0].find("pointer"), std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsNewExpression) {
  auto result = checkSubset(R"(
    void foo() { int* p = new int(42); }
  )");
  EXPECT_FALSE(result.passed);
}

TEST(SubsetEnforcerTest, RejectsThrow) {
  auto result = checkSubset(R"(
    void foo() { throw 42; }
  )");
  EXPECT_FALSE(result.passed);
}

TEST(SubsetEnforcerTest, RejectsDoubleType) {
  auto result = checkSubset(R"(
    double foo() { return 3.14; }
  )");
  EXPECT_FALSE(result.passed);
}

TEST(SubsetEnforcerTest, AcceptsAllArithmeticOps) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t compute(int32_t a, int32_t b) {
      int32_t sum = a + b;
      int32_t diff = a - b;
      int32_t prod = a * b;
      int32_t quot = a / b;
      int32_t rem = a % b;
      return sum;
    }
  )");
  EXPECT_TRUE(result.passed);
}

TEST(SubsetEnforcerTest, AcceptsAllComparisonAndLogicalOps) {
  auto result = checkSubset(R"(
    #include <cstdint>
    bool check(int32_t a, int32_t b) {
      bool r1 = a < b;
      bool r2 = a <= b;
      bool r3 = a > b;
      bool r4 = a >= b;
      bool r5 = a == b;
      bool r6 = a != b;
      bool r7 = r1 && r2;
      bool r8 = r3 || r4;
      bool r9 = !r5;
      return r9;
    }
  )");
  EXPECT_TRUE(result.passed);
}

} // namespace
} // namespace arcanum
```

**Step 3: Run the test to verify it fails**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --preset default && cmake --build build/default --target SubsetEnforcerTest && ctest --test-dir build/default -R SubsetEnforcerTest -V`
Expected: Build failure (SubsetEnforcer.cpp not yet implemented)

**Step 4: Implement `SubsetEnforcer.cpp`**

```cpp
#include "frontend/SubsetEnforcer.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"

#include <string>

namespace arcanum {
namespace {

class SubsetVisitor : public clang::RecursiveASTVisitor<SubsetVisitor> {
public:
  explicit SubsetVisitor(clang::ASTContext& ctx, SubsetResult& result)
      : ctx_(ctx), result_(result) {}

  bool VisitFunctionDecl(clang::FunctionDecl* decl) {
    if (!decl->hasBody()) {
      return true; // Skip declarations without bodies
    }
    // Skip compiler-generated functions
    if (decl->isImplicit()) {
      return true;
    }
    // Reject virtual functions
    if (auto* method = llvm::dyn_cast<clang::CXXMethodDecl>(decl)) {
      if (method->isVirtual()) {
        addDiagnostic(decl->getLocation(), "virtual functions are not allowed");
        return true;
      }
    }
    // Reject templates
    if (decl->isTemplated()) {
      addDiagnostic(decl->getLocation(),
                    "template functions are not allowed in Slice 1");
      return true;
    }
    // Check return type
    checkType(decl->getReturnType(), decl->getLocation());
    // Check parameter types
    for (const auto* param : decl->parameters()) {
      checkType(param->getType(), param->getLocation());
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl* decl) {
    if (decl->isImplicit()) {
      return true;
    }
    checkType(decl->getType(), decl->getLocation());
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr* expr) {
    addDiagnostic(expr->getBeginLoc(),
                  "dynamic allocation (new) is not allowed");
    return true;
  }

  bool VisitCXXDeleteExpr(clang::CXXDeleteExpr* expr) {
    addDiagnostic(expr->getBeginLoc(),
                  "dynamic deallocation (delete) is not allowed");
    return true;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr* expr) {
    addDiagnostic(expr->getBeginLoc(), "throw expressions are not allowed");
    return true;
  }

  bool VisitCXXTryStmt(clang::CXXTryStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(), "try/catch is not allowed");
    return true;
  }

  bool VisitGotoStmt(clang::GotoStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(), "goto is not allowed");
    return true;
  }

private:
  void checkType(clang::QualType type, clang::SourceLocation loc) {
    type = type.getCanonicalType();
    // Allow void (for functions returning void, though Slice 1 expects
    // int32_t/bool)
    if (type->isVoidType()) {
      return;
    }
    // Allow bool
    if (type->isBooleanType()) {
      return;
    }
    // Allow int32_t (which is a typedef for int on most platforms, but check
    // for 32-bit signed integer)
    if (const auto* bt = type->getAs<clang::BuiltinType>()) {
      if (bt->getKind() == clang::BuiltinType::Int) {
        return; // int32_t maps to int on most platforms
      }
    }
    // Check for typedef to int32_t specifically
    if (type->isIntegerType()) {
      auto width = ctx_.getTypeSize(type);
      if (width == 32 && type->isSignedIntegerType()) {
        return;
      }
    }
    // Reject everything else
    if (type->isPointerType()) {
      addDiagnostic(loc, "raw pointer types are not allowed");
    } else if (type->isFloatingType()) {
      addDiagnostic(loc, "floating-point types are not allowed in Slice 1");
    } else {
      addDiagnostic(loc,
                    "type '" + type.getAsString() +
                        "' is not allowed in Slice 1 (only int32_t and bool)");
    }
  }

  void addDiagnostic(clang::SourceLocation loc, const std::string& msg) {
    result_.passed = false;
    auto& sm = ctx_.getSourceManager();
    if (loc.isValid()) {
      auto presumed = sm.getPresumedLoc(loc);
      if (presumed.isValid()) {
        result_.diagnostics.push_back(
            std::string(presumed.getFilename()) + ":" +
            std::to_string(presumed.getLine()) + ": error: " + msg);
        return;
      }
    }
    result_.diagnostics.push_back("error: " + msg);
  }

  clang::ASTContext& ctx_;
  SubsetResult& result_;
};

} // namespace

SubsetResult enforceSubset(clang::ASTContext& context) {
  SubsetResult result;
  SubsetVisitor visitor(context, result);
  visitor.TraverseDecl(context.getTranslationUnitDecl());
  return result;
}

} // namespace arcanum
```

**Step 5: Run the tests to verify they pass**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target SubsetEnforcerTest && ctest --test-dir build/default -R SubsetEnforcerTest -V`
Expected: All tests PASS

**Step 6: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/src/frontend/SubsetEnforcer.h arcanum/src/frontend/SubsetEnforcer.cpp arcanum/tests/unit/SubsetEnforcerTest.cpp
git commit -m "feat(arcanum): add SubsetEnforcer for Slice 1 C++ subset validation"
```

**Step 7: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Accepts int32_t, bool, arithmetic, if/else, comparisons, logical ops | Tests pass for all allowed constructs | |
| Rejects virtual, raw pointers, new, throw, goto, float/double | Tests pass for all rejected constructs | |
| Diagnostics include source location and descriptive message | Format: "file:line: error: message" | |
| Unit tests cover both accepted and rejected constructs | Minimum 10 test cases | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 5: Stage 3 -- Contract Parser

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/src/frontend/ContractParser.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/frontend/ContractParser.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/unit/ContractParserTest.cpp`

**Agent role:** senior-engineer

**Step 1: Write `ContractParser.h` -- interface with ContractInfo and expression AST**

```cpp
#ifndef ARCANUM_FRONTEND_CONTRACTPARSER_H
#define ARCANUM_FRONTEND_CONTRACTPARSER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace arcanum {

/// Simple expression AST node for contract expressions.
enum class ContractExprKind {
  IntLiteral,     // 42
  BoolLiteral,    // true, false
  ParamRef,       // parameter name
  ResultRef,      // \result
  BinaryOp,       // a + b, a && b, a < b, etc.
  UnaryOp,        // !a, -a
};

struct ContractExpr;
using ContractExprPtr = std::shared_ptr<ContractExpr>;

enum class BinaryOpKind {
  Add, Sub, Mul, Div, Rem,
  Lt, Le, Gt, Ge, Eq, Ne,
  And, Or,
};

enum class UnaryOpKind {
  Not, Neg,
};

struct ContractExpr {
  ContractExprKind kind;

  // For IntLiteral
  int64_t intValue = 0;

  // For BoolLiteral
  bool boolValue = false;

  // For ParamRef
  std::string paramName;

  // For BinaryOp
  BinaryOpKind binaryOp = BinaryOpKind::Add;
  ContractExprPtr left;
  ContractExprPtr right;

  // For UnaryOp
  UnaryOpKind unaryOp = UnaryOpKind::Not;
  ContractExprPtr operand;

  static ContractExprPtr makeIntLiteral(int64_t val);
  static ContractExprPtr makeBoolLiteral(bool val);
  static ContractExprPtr makeParamRef(const std::string& name);
  static ContractExprPtr makeResultRef();
  static ContractExprPtr makeBinaryOp(BinaryOpKind op, ContractExprPtr lhs,
                                      ContractExprPtr rhs);
  static ContractExprPtr makeUnaryOp(UnaryOpKind op, ContractExprPtr operand);
};

struct ContractInfo {
  std::vector<ContractExprPtr> requires;
  std::vector<ContractExprPtr> ensures;
};

/// Parse //@ requires: and //@ ensures: annotations from raw comments,
/// associating them with the FunctionDecl they immediately precede.
std::map<const clang::FunctionDecl*, ContractInfo>
parseContracts(clang::ASTContext& context);

} // namespace arcanum

#endif // ARCANUM_FRONTEND_CONTRACTPARSER_H
```

**Step 2: Write the failing test `ContractParserTest.cpp`**

```cpp
#include "frontend/ContractParser.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

std::map<const clang::FunctionDecl*, ContractInfo>
parseFromSource(const std::string& code,
                std::unique_ptr<clang::ASTUnit>& astOut) {
  astOut = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp",
      "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  EXPECT_NE(astOut, nullptr);
  return parseContracts(astOut->getASTContext());
}

TEST(ContractParserTest, ParsesSimpleRequires) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )", ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.requires.size(), 1u);
  EXPECT_EQ(it->second.ensures.size(), 0u);
}

TEST(ContractParserTest, ParsesMultipleRequires) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    //@ requires: a <= 1000
    int32_t foo(int32_t a) { return a; }
  )", ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.requires.size(), 2u);
}

TEST(ContractParserTest, ParsesEnsuresWithResult) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t foo(int32_t a) { return a; }
  )", ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.ensures.size(), 1u);
}

TEST(ContractParserTest, ParsesRequiresAndEnsures) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) { return a + b; }
  )", ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.requires.size(), 2u);
  EXPECT_EQ(it->second.ensures.size(), 1u);
}

TEST(ContractParserTest, NoContractsReturnsEmptyMap) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    int32_t foo(int32_t a) { return a; }
  )", ast);

  EXPECT_TRUE(contracts.empty());
}

TEST(ContractParserTest, ParsesBinaryComparisonExpr) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )", ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.requires.size(), 1u);
  auto& expr = it->second.requires[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Ge);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ParamRef);
  EXPECT_EQ(expr->left->paramName, "a");
  EXPECT_EQ(expr->right->kind, ContractExprKind::IntLiteral);
  EXPECT_EQ(expr->right->intValue, 0);
}

TEST(ContractParserTest, ParsesAndExpression) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    int32_t foo(int32_t a) { return a; }
  )", ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.requires.size(), 1u);
  auto& expr = it->second.requires[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::And);
}

TEST(ContractParserTest, ParsesResultRef) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t foo(int32_t a) { return a; }
  )", ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.ensures.size(), 1u);
  auto& expr = it->second.ensures[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ResultRef);
}

} // namespace
} // namespace arcanum
```

**Step 3: Run the test to verify it fails**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target ContractParserTest && ctest --test-dir build/default -R ContractParserTest -V`
Expected: Build failure (ContractParser.cpp not yet implemented)

**Step 4: Implement `ContractParser.cpp`**

```cpp
#include "frontend/ContractParser.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Comment.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/ADT/StringRef.h"

#include <cctype>
#include <optional>
#include <sstream>

namespace arcanum {

ContractExprPtr ContractExpr::makeIntLiteral(int64_t val) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::IntLiteral;
  e->intValue = val;
  return e;
}

ContractExprPtr ContractExpr::makeBoolLiteral(bool val) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::BoolLiteral;
  e->boolValue = val;
  return e;
}

ContractExprPtr ContractExpr::makeParamRef(const std::string& name) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::ParamRef;
  e->paramName = name;
  return e;
}

ContractExprPtr ContractExpr::makeResultRef() {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::ResultRef;
  return e;
}

ContractExprPtr ContractExpr::makeBinaryOp(BinaryOpKind op,
                                           ContractExprPtr lhs,
                                           ContractExprPtr rhs) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::BinaryOp;
  e->binaryOp = op;
  e->left = std::move(lhs);
  e->right = std::move(rhs);
  return e;
}

ContractExprPtr ContractExpr::makeUnaryOp(UnaryOpKind op,
                                          ContractExprPtr operand) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::UnaryOp;
  e->unaryOp = op;
  e->operand = std::move(operand);
  return e;
}

namespace {

/// Simple recursive-descent parser for contract expressions.
class ExprParser {
public:
  explicit ExprParser(llvm::StringRef text) : text_(text), pos_(0) {}

  ContractExprPtr parse() {
    auto expr = parseOr();
    skipWhitespace();
    return expr;
  }

private:
  void skipWhitespace() {
    while (pos_ < text_.size() && std::isspace(text_[pos_])) {
      ++pos_;
    }
  }

  bool matchString(llvm::StringRef s) {
    skipWhitespace();
    if (text_.substr(pos_).starts_with(s)) {
      pos_ += s.size();
      return true;
    }
    return false;
  }

  bool peekString(llvm::StringRef s) {
    skipWhitespace();
    return text_.substr(pos_).starts_with(s);
  }

  ContractExprPtr parseOr() {
    auto lhs = parseAnd();
    while (matchString("||")) {
      auto rhs = parseAnd();
      lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Or, lhs, rhs);
    }
    return lhs;
  }

  ContractExprPtr parseAnd() {
    auto lhs = parseComparison();
    while (matchString("&&")) {
      auto rhs = parseComparison();
      lhs = ContractExpr::makeBinaryOp(BinaryOpKind::And, lhs, rhs);
    }
    return lhs;
  }

  ContractExprPtr parseComparison() {
    auto lhs = parseAddSub();
    skipWhitespace();
    if (matchString("<=")) {
      auto rhs = parseAddSub();
      return ContractExpr::makeBinaryOp(BinaryOpKind::Le, lhs, rhs);
    }
    if (matchString(">=")) {
      auto rhs = parseAddSub();
      return ContractExpr::makeBinaryOp(BinaryOpKind::Ge, lhs, rhs);
    }
    if (matchString("==")) {
      auto rhs = parseAddSub();
      return ContractExpr::makeBinaryOp(BinaryOpKind::Eq, lhs, rhs);
    }
    if (matchString("!=")) {
      auto rhs = parseAddSub();
      return ContractExpr::makeBinaryOp(BinaryOpKind::Ne, lhs, rhs);
    }
    if (matchString("<")) {
      auto rhs = parseAddSub();
      return ContractExpr::makeBinaryOp(BinaryOpKind::Lt, lhs, rhs);
    }
    if (matchString(">")) {
      auto rhs = parseAddSub();
      return ContractExpr::makeBinaryOp(BinaryOpKind::Gt, lhs, rhs);
    }
    return lhs;
  }

  ContractExprPtr parseAddSub() {
    auto lhs = parseMulDiv();
    skipWhitespace();
    while (true) {
      if (matchString("+")) {
        auto rhs = parseMulDiv();
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Add, lhs, rhs);
      } else if (matchString("-")) {
        auto rhs = parseMulDiv();
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Sub, lhs, rhs);
      } else {
        break;
      }
    }
    return lhs;
  }

  ContractExprPtr parseMulDiv() {
    auto lhs = parseUnary();
    skipWhitespace();
    while (true) {
      if (matchString("*")) {
        auto rhs = parseUnary();
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Mul, lhs, rhs);
      } else if (matchString("/")) {
        auto rhs = parseUnary();
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Div, lhs, rhs);
      } else if (matchString("%")) {
        auto rhs = parseUnary();
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Rem, lhs, rhs);
      } else {
        break;
      }
    }
    return lhs;
  }

  ContractExprPtr parseUnary() {
    skipWhitespace();
    if (matchString("!")) {
      auto operand = parseUnary();
      return ContractExpr::makeUnaryOp(UnaryOpKind::Not, operand);
    }
    if (matchString("-")) {
      auto operand = parsePrimary();
      return ContractExpr::makeUnaryOp(UnaryOpKind::Neg, operand);
    }
    return parsePrimary();
  }

  ContractExprPtr parsePrimary() {
    skipWhitespace();
    if (pos_ >= text_.size()) {
      return ContractExpr::makeIntLiteral(0); // Error fallback
    }
    // Parenthesized expression
    if (text_[pos_] == '(') {
      ++pos_;
      auto expr = parseOr();
      skipWhitespace();
      if (pos_ < text_.size() && text_[pos_] == ')') {
        ++pos_;
      }
      return expr;
    }
    // \result
    if (matchString("\\result")) {
      return ContractExpr::makeResultRef();
    }
    // true/false
    if (matchString("true")) {
      return ContractExpr::makeBoolLiteral(true);
    }
    if (matchString("false")) {
      return ContractExpr::makeBoolLiteral(false);
    }
    // Integer literal
    if (std::isdigit(text_[pos_])) {
      size_t start = pos_;
      while (pos_ < text_.size() && std::isdigit(text_[pos_])) {
        ++pos_;
      }
      int64_t val = std::stoll(text_.substr(start, pos_ - start).str());
      return ContractExpr::makeIntLiteral(val);
    }
    // Identifier (parameter name)
    if (std::isalpha(text_[pos_]) || text_[pos_] == '_') {
      size_t start = pos_;
      while (pos_ < text_.size() &&
             (std::isalnum(text_[pos_]) || text_[pos_] == '_')) {
        ++pos_;
      }
      return ContractExpr::makeParamRef(
          text_.substr(start, pos_ - start).str());
    }
    // Fallback
    return ContractExpr::makeIntLiteral(0);
  }

  llvm::StringRef text_;
  size_t pos_;
};

/// Extract //@ lines from a raw comment block and return them.
std::vector<std::string> extractAnnotationLines(llvm::StringRef commentText) {
  std::vector<std::string> lines;
  llvm::SmallVector<llvm::StringRef, 8> splitLines;
  commentText.split(splitLines, '\n');

  for (auto& line : splitLines) {
    auto trimmed = line.trim();
    // Remove leading // if present
    if (trimmed.starts_with("//")) {
      trimmed = trimmed.drop_front(2).ltrim();
    }
    // Check for @ prefix
    if (trimmed.starts_with("@")) {
      trimmed = trimmed.drop_front(1).ltrim();
      lines.push_back(trimmed.str());
    }
  }
  return lines;
}

} // namespace

std::map<const clang::FunctionDecl*, ContractInfo>
parseContracts(clang::ASTContext& context) {
  std::map<const clang::FunctionDecl*, ContractInfo> result;
  auto& sm = context.getSourceManager();

  // Iterate over all function declarations
  for (auto* decl : context.getTranslationUnitDecl()->decls()) {
    auto* funcDecl = llvm::dyn_cast<clang::FunctionDecl>(decl);
    if (!funcDecl || !funcDecl->hasBody()) {
      continue;
    }

    // Get the raw comment associated with this declaration
    const auto* rawComment = context.getRawCommentForDeclNoCache(funcDecl);
    if (!rawComment) {
      continue;
    }

    auto commentText = rawComment->getRawText(sm);
    auto annotationLines = extractAnnotationLines(commentText);

    if (annotationLines.empty()) {
      continue;
    }

    ContractInfo info;
    for (const auto& line : annotationLines) {
      llvm::StringRef lineRef(line);
      if (lineRef.starts_with("requires:")) {
        auto exprText = lineRef.drop_front(9).trim();
        ExprParser parser(exprText);
        if (auto expr = parser.parse()) {
          info.requires.push_back(std::move(expr));
        }
      } else if (lineRef.starts_with("ensures:")) {
        auto exprText = lineRef.drop_front(8).trim();
        ExprParser parser(exprText);
        if (auto expr = parser.parse()) {
          info.ensures.push_back(std::move(expr));
        }
      }
    }

    if (!info.requires.empty() || !info.ensures.empty()) {
      result[funcDecl] = std::move(info);
    }
  }

  return result;
}

} // namespace arcanum
```

**Step 5: Run the tests to verify they pass**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target ContractParserTest && ctest --test-dir build/default -R ContractParserTest -V`
Expected: All tests PASS

**Step 6: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/src/frontend/ContractParser.h arcanum/src/frontend/ContractParser.cpp arcanum/tests/unit/ContractParserTest.cpp
git commit -m "feat(arcanum): add ContractParser for //@ requires/ensures annotations"
```

**Step 7: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Parses //@ requires: and //@ ensures: prefixes | Correctly splits annotation lines | |
| Builds expression AST with correct precedence | && binds tighter than \|\|, comparisons above arithmetic | |
| Supports \result, integer literals, parameter names | All token types parsed | |
| Associates contracts with correct FunctionDecl | Comment-to-decl mapping via RawCommentList | |
| Unit tests cover all expression forms | Binary ops, unary ops, \result, compound expressions | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 6: Arc MLIR Dialect Definition (TableGen) + Lowering + Pass Manager (Stages 4-5)

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcDialect.td`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcOps.td`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcTypes.td`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcDialect.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcDialect.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcOps.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcOps.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcTypes.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/ArcTypes.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/Lowering.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/dialect/Lowering.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/passes/Passes.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/passes/Passes.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/unit/ArcDialectTest.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/unit/LoweringTest.cpp`

**Agent role:** senior-engineer

**Step 1: Write `ArcDialect.td` -- dialect definition**

```tablegen
//===- ArcDialect.td - Arc dialect definition ----------*- tablegen -*-===//
//
// Defines the Arc dialect for Arcanum formal verification.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_DIALECT_TD
#define ARC_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Arc dialect definition
//===----------------------------------------------------------------------===//

def Arc_Dialect : Dialect {
  let name = "arc";
  let summary = "Arc dialect for formal verification of C++ programs";
  let description = [{
    The Arc dialect provides operations and types for representing
    verified C++ programs. It captures verification-relevant semantics
    (contracts, overflow checking, etc.) in a small set of MLIR operations
    that translate nearly 1:1 to WhyML.
  }];
  let cppNamespace = "::arcanum::arc";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base Arc operation definition
//===----------------------------------------------------------------------===//

class Arc_Op<string mnemonic, list<Trait> traits = []> :
    Op<Arc_Dialect, mnemonic, traits>;

#endif // ARC_DIALECT_TD
```

**Step 2: Write `ArcTypes.td` -- type definitions (Slice 1: i32 and bool only)**

```tablegen
//===- ArcTypes.td - Arc type definitions -----------------*- tablegen -*-===//
//
// Defines the types for the Arc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_TYPES_TD
#define ARC_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "dialect/ArcDialect.td"

//===----------------------------------------------------------------------===//
// Arc type definitions
//===----------------------------------------------------------------------===//

class Arc_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Arc_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Arc_I32Type : Arc_Type<"I32", "i32"> {
  let summary = "32-bit signed integer type";
  let description = [{
    Represents int32_t in the Arc dialect. Arithmetic on this type
    generates overflow proof obligations in trap mode.
  }];
}

def Arc_BoolType : Arc_Type<"Bool", "bool"> {
  let summary = "Boolean type";
  let description = [{
    Represents bool in the Arc dialect.
  }];
}

#endif // ARC_TYPES_TD
```

**Step 3: Write `ArcOps.td` -- operation definitions for Slice 1**

```tablegen
//===- ArcOps.td - Arc operation definitions ---------------*- tablegen -*-===//
//
// Defines the operations for the Arc dialect (Slice 1).
//
//===----------------------------------------------------------------------===//

#ifndef ARC_OPS_TD
#define ARC_OPS_TD

include "dialect/ArcDialect.td"
include "dialect/ArcTypes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Function operation
//===----------------------------------------------------------------------===//

def Arc_FuncOp : Arc_Op<"func", [
    Symbol,
    IsolatedFromAbove
]> {
  let summary = "Function with verification contracts";
  let description = [{
    Represents a verified function with optional requires/ensures attributes.
    The body is a single region.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$requires_attr,
    OptionalAttr<StrAttr>:$ensures_attr
  );
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Constant operations
//===----------------------------------------------------------------------===//

def Arc_ConstantOp : Arc_Op<"constant", [Pure]> {
  let summary = "Integer or boolean constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Arithmetic operations (overflow-checked in trap mode)
//===----------------------------------------------------------------------===//

def Arc_AddOp : Arc_Op<"add", [Pure]> {
  let summary = "Checked integer addition";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Arc_SubOp : Arc_Op<"sub", [Pure]> {
  let summary = "Checked integer subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Arc_MulOp : Arc_Op<"mul", [Pure]> {
  let summary = "Checked integer multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Arc_DivOp : Arc_Op<"div", [Pure]> {
  let summary = "Checked integer division";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Arc_RemOp : Arc_Op<"rem", [Pure]> {
  let summary = "Checked integer remainder";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Comparison operation
//===----------------------------------------------------------------------===//

def Arc_CmpOp : Arc_Op<"cmp", [Pure]> {
  let summary = "Integer comparison";
  let arguments = (ins StrAttr:$predicate, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Logical operations
//===----------------------------------------------------------------------===//

def Arc_AndOp : Arc_Op<"and", [Pure]> {
  let summary = "Logical AND";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Arc_OrOp : Arc_Op<"or", [Pure]> {
  let summary = "Logical OR";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Arc_NotOp : Arc_Op<"not", [Pure]> {
  let summary = "Logical NOT";
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Variable operations
//===----------------------------------------------------------------------===//

def Arc_VarOp : Arc_Op<"var"> {
  let summary = "Local variable declaration";
  let arguments = (ins StrAttr:$name, AnyType:$init);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name `=` $init attr-dict `:` type($result)";
}

def Arc_AssignOp : Arc_Op<"assign"> {
  let summary = "Variable assignment";
  let arguments = (ins AnyType:$target, AnyType:$value);
  let assemblyFormat = "$target `=` $value attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

def Arc_ReturnOp : Arc_Op<"return", [Terminator]> {
  let summary = "Function return";
  let arguments = (ins Optional<AnyType>:$value);
  let assemblyFormat = "($value^ `:` type($value))? attr-dict";
}

def Arc_IfOp : Arc_Op<"if"> {
  let summary = "Conditional branch";
  let arguments = (ins AnyType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
  let results = (outs Optional<AnyType>:$result);
  let hasCustomAssemblyFormat = 1;
}

#endif // ARC_OPS_TD
```

**Step 4: Write `ArcDialect.h`**

```cpp
#ifndef ARCANUM_DIALECT_ARCDIALECT_H
#define ARCANUM_DIALECT_ARCDIALECT_H

#include "mlir/IR/Dialect.h"

#include "dialect/ArcDialect.h.inc"

#endif // ARCANUM_DIALECT_ARCDIALECT_H
```

**Step 5: Write `ArcDialect.cpp`**

```cpp
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

using namespace arcanum::arc;

#include "dialect/ArcDialect.cpp.inc"

void ArcDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "dialect/ArcOps.cpp.inc"
      >();
}
```

**Step 6: Write `ArcTypes.h`**

```cpp
#ifndef ARCANUM_DIALECT_ARCTYPES_H
#define ARCANUM_DIALECT_ARCTYPES_H

#include "mlir/IR/Types.h"
#include "dialect/ArcDialect.h"

#define GET_TYPEDEF_CLASSES
#include "dialect/ArcTypes.h.inc"

#endif // ARCANUM_DIALECT_ARCTYPES_H
```

**Step 7: Write `ArcTypes.cpp`**

```cpp
#include "dialect/ArcTypes.h"
#include "dialect/ArcDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace arcanum::arc;

#define GET_TYPEDEF_CLASSES
#include "dialect/ArcTypes.cpp.inc"

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "dialect/ArcTypes.h.inc"
      >();
}
```

**Step 8: Write `ArcOps.h`**

```cpp
#ifndef ARCANUM_DIALECT_ARCOPS_H
#define ARCANUM_DIALECT_ARCOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "dialect/ArcDialect.h"
#include "dialect/ArcTypes.h"

#define GET_OP_CLASSES
#include "dialect/ArcOps.h.inc"

#endif // ARCANUM_DIALECT_ARCOPS_H
```

**Step 9: Write `ArcOps.cpp`**

```cpp
#include "dialect/ArcOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace arcanum::arc;

#define GET_OP_CLASSES
#include "dialect/ArcOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
  // Minimal custom parsing for Slice 1 — will be refined
  // For now, delegate to simple attribute-based format
  return mlir::failure(); // TODO: implement full parser
}

void FuncOp::print(mlir::OpAsmPrinter& printer) {
  printer << " @" << getSymName();
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"sym_name", "function_type"});
  printer << " ";
  printer.printRegion(getBody());
}

//===----------------------------------------------------------------------===//
// ConstantOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser& parser,
                                    mlir::OperationState& result) {
  return mlir::failure(); // TODO: implement full parser
}

void ConstantOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getValue();
  printer << " : " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// IfOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult IfOp::parse(mlir::OpAsmParser& parser,
                              mlir::OperationState& result) {
  return mlir::failure(); // TODO: implement full parser
}

void IfOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getCondition();
  printer << " ";
  printer.printRegion(getThenRegion());
  if (!getElseRegion().empty()) {
    printer << " else ";
    printer.printRegion(getElseRegion());
  }
}
```

**Step 10: Write `Lowering.h`**

```cpp
#ifndef ARCANUM_DIALECT_LOWERING_H
#define ARCANUM_DIALECT_LOWERING_H

#include "frontend/ContractParser.h"
#include "clang/AST/ASTContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <map>
#include <memory>

namespace arcanum {

/// Lower a Clang ASTContext + parsed contracts into an Arc MLIR ModuleOp.
mlir::OwningOpRef<mlir::ModuleOp> lowerToArc(
    mlir::MLIRContext& context,
    clang::ASTContext& astContext,
    const std::map<const clang::FunctionDecl*, ContractInfo>& contracts);

} // namespace arcanum

#endif // ARCANUM_DIALECT_LOWERING_H
```

**Step 11: Write `Lowering.cpp` -- Clang AST to Arc MLIR translation**

```cpp
#include "dialect/Lowering.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include <string>

namespace arcanum {
namespace {

class ArcLowering {
public:
  ArcLowering(mlir::MLIRContext& ctx, clang::ASTContext& astCtx,
              const std::map<const clang::FunctionDecl*, ContractInfo>& contracts)
      : mlirCtx_(ctx), astCtx_(astCtx), contracts_(contracts),
        builder_(&ctx) {
    ctx.getOrLoadDialect<arc::ArcDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> lower() {
    module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    builder_.setInsertionPointToEnd(module_->getBody());

    for (auto* decl : astCtx_.getTranslationUnitDecl()->decls()) {
      if (auto* funcDecl = llvm::dyn_cast<clang::FunctionDecl>(decl)) {
        if (funcDecl->hasBody()) {
          lowerFunction(funcDecl);
        }
      }
    }

    return std::move(module_);
  }

private:
  mlir::Location getLoc(clang::SourceLocation clangLoc) {
    if (clangLoc.isValid()) {
      auto& sm = astCtx_.getSourceManager();
      auto presumed = sm.getPresumedLoc(clangLoc);
      if (presumed.isValid()) {
        return mlir::FileLineColLoc::get(
            builder_.getStringAttr(presumed.getFilename()),
            presumed.getLine(), presumed.getColumn());
      }
    }
    return builder_.getUnknownLoc();
  }

  mlir::Type getArcType(clang::QualType type) {
    auto canonical = type.getCanonicalType();
    if (canonical->isBooleanType()) {
      return arc::BoolType::get(&mlirCtx_);
    }
    // Default to i32 for integer types in Slice 1
    return arc::I32Type::get(&mlirCtx_);
  }

  void lowerFunction(clang::FunctionDecl* funcDecl) {
    auto loc = getLoc(funcDecl->getLocation());
    auto name = funcDecl->getNameAsString();

    // Build function type
    llvm::SmallVector<mlir::Type> paramTypes;
    for (const auto* param : funcDecl->parameters()) {
      paramTypes.push_back(getArcType(param->getType()));
    }
    mlir::Type resultType = getArcType(funcDecl->getReturnType());
    auto funcType = builder_.getFunctionType(paramTypes, {resultType});

    // Get contract strings if present
    mlir::StringAttr requiresAttr, ensuresAttr;
    auto it = contracts_.find(funcDecl);
    if (it != contracts_.end()) {
      // Serialize contract expressions as string attributes for Slice 1.
      // Future slices will use structured MLIR attributes.
      std::string reqStr, ensStr;
      for (size_t i = 0; i < it->second.requires.size(); ++i) {
        if (i > 0) reqStr += " && ";
        reqStr += serializeExpr(it->second.requires[i]);
      }
      for (size_t i = 0; i < it->second.ensures.size(); ++i) {
        if (i > 0) ensStr += " && ";
        ensStr += serializeExpr(it->second.ensures[i]);
      }
      if (!reqStr.empty()) {
        requiresAttr = builder_.getStringAttr(reqStr);
      }
      if (!ensStr.empty()) {
        ensuresAttr = builder_.getStringAttr(ensStr);
      }
    }

    // Create arc.func
    auto funcOp = builder_.create<arc::FuncOp>(
        loc, name, mlir::TypeAttr::get(funcType), requiresAttr, ensuresAttr);

    // Create entry block with parameters
    auto& entryBlock = funcOp.getBody().emplaceBlock();
    for (size_t i = 0; i < paramTypes.size(); ++i) {
      entryBlock.addArgument(paramTypes[i], loc);
    }

    // Map Clang params to MLIR block args
    llvm::DenseMap<const clang::ValueDecl*, mlir::Value> valueMap;
    for (size_t i = 0; i < funcDecl->getNumParams(); ++i) {
      valueMap[funcDecl->getParamDecl(i)] = entryBlock.getArgument(i);
    }

    // Lower function body
    auto savedIp = builder_.saveInsertionPoint();
    builder_.setInsertionPointToEnd(&entryBlock);
    lowerStmt(funcDecl->getBody(), valueMap);
    builder_.restoreInsertionPoint(savedIp);
  }

  void lowerStmt(const clang::Stmt* stmt,
                 llvm::DenseMap<const clang::ValueDecl*, mlir::Value>& valueMap) {
    if (auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
      for (const auto* child : compound->body()) {
        lowerStmt(child, valueMap);
      }
    } else if (auto* ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
      auto retVal = lowerExpr(ret->getRetValue(), valueMap);
      builder_.create<arc::ReturnOp>(getLoc(ret->getReturnLoc()), retVal);
    } else if (auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
      for (const auto* d : declStmt->decls()) {
        if (auto* varDecl = llvm::dyn_cast<clang::VarDecl>(d)) {
          if (varDecl->hasInit()) {
            auto initVal = lowerExpr(varDecl->getInit(), valueMap);
            auto loc = getLoc(varDecl->getLocation());
            auto varOp = builder_.create<arc::VarOp>(
                loc, getArcType(varDecl->getType()),
                varDecl->getNameAsString(), initVal);
            valueMap[varDecl] = varOp.getResult();
          }
        }
      }
    } else if (auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
      auto cond = lowerExpr(ifStmt->getCond(), valueMap);
      auto loc = getLoc(ifStmt->getIfLoc());
      auto ifOp = builder_.create<arc::IfOp>(loc, mlir::TypeRange{}, cond);

      // Then region
      auto& thenBlock = ifOp.getThenRegion().emplaceBlock();
      auto savedIp = builder_.saveInsertionPoint();
      builder_.setInsertionPointToEnd(&thenBlock);
      lowerStmt(ifStmt->getThen(), valueMap);
      builder_.restoreInsertionPoint(savedIp);

      // Else region
      if (ifStmt->getElse()) {
        auto& elseBlock = ifOp.getElseRegion().emplaceBlock();
        auto savedIp2 = builder_.saveInsertionPoint();
        builder_.setInsertionPointToEnd(&elseBlock);
        lowerStmt(ifStmt->getElse(), valueMap);
        builder_.restoreInsertionPoint(savedIp2);
      }
    }
    // TODO: handle assignment expressions in Slice 1
  }

  mlir::Value lowerExpr(const clang::Expr* expr,
                        llvm::DenseMap<const clang::ValueDecl*, mlir::Value>& valueMap) {
    expr = expr->IgnoreParenImpCasts();
    auto loc = getLoc(expr->getBeginLoc());

    if (auto* intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
      auto val = intLit->getValue().getSExtValue();
      return builder_.create<arc::ConstantOp>(
          loc, arc::I32Type::get(&mlirCtx_),
          builder_.getI32IntegerAttr(val));
    }

    if (auto* boolLit = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
      return builder_.create<arc::ConstantOp>(
          loc, arc::BoolType::get(&mlirCtx_),
          builder_.getBoolAttr(boolLit->getValue()));
    }

    if (auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
      auto it = valueMap.find(declRef->getDecl());
      if (it != valueMap.end()) {
        return it->second;
      }
      // Fallback: return a zero constant
      return builder_.create<arc::ConstantOp>(
          loc, arc::I32Type::get(&mlirCtx_), builder_.getI32IntegerAttr(0));
    }

    if (auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
      auto lhs = lowerExpr(binOp->getLHS(), valueMap);
      auto rhs = lowerExpr(binOp->getRHS(), valueMap);

      switch (binOp->getOpcode()) {
      case clang::BO_Add:
        return builder_.create<arc::AddOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Sub:
        return builder_.create<arc::SubOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Mul:
        return builder_.create<arc::MulOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Div:
        return builder_.create<arc::DivOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Rem:
        return builder_.create<arc::RemOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_LT:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("lt"), lhs, rhs);
      case clang::BO_LE:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("le"), lhs, rhs);
      case clang::BO_GT:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("gt"), lhs, rhs);
      case clang::BO_GE:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("ge"), lhs, rhs);
      case clang::BO_EQ:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("eq"), lhs, rhs);
      case clang::BO_NE:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("ne"), lhs, rhs);
      case clang::BO_LAnd:
        return builder_.create<arc::AndOp>(
            loc, arc::BoolType::get(&mlirCtx_), lhs, rhs);
      case clang::BO_LOr:
        return builder_.create<arc::OrOp>(
            loc, arc::BoolType::get(&mlirCtx_), lhs, rhs);
      default:
        break;
      }
    }

    if (auto* unaryOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
      auto operand = lowerExpr(unaryOp->getSubExpr(), valueMap);
      switch (unaryOp->getOpcode()) {
      case clang::UO_LNot:
        return builder_.create<arc::NotOp>(
            loc, arc::BoolType::get(&mlirCtx_), operand);
      case clang::UO_Minus: {
        auto zero = builder_.create<arc::ConstantOp>(
            loc, arc::I32Type::get(&mlirCtx_), builder_.getI32IntegerAttr(0));
        return builder_.create<arc::SubOp>(loc, operand.getType(), zero, operand);
      }
      default:
        break;
      }
    }

    // Fallback: return zero constant
    return builder_.create<arc::ConstantOp>(
        loc, arc::I32Type::get(&mlirCtx_), builder_.getI32IntegerAttr(0));
  }

  std::string serializeExpr(const ContractExprPtr& expr) {
    switch (expr->kind) {
    case ContractExprKind::IntLiteral:
      return std::to_string(expr->intValue);
    case ContractExprKind::BoolLiteral:
      return expr->boolValue ? "true" : "false";
    case ContractExprKind::ParamRef:
      return expr->paramName;
    case ContractExprKind::ResultRef:
      return "\\result";
    case ContractExprKind::BinaryOp: {
      std::string op;
      switch (expr->binaryOp) {
      case BinaryOpKind::Add: op = "+"; break;
      case BinaryOpKind::Sub: op = "-"; break;
      case BinaryOpKind::Mul: op = "*"; break;
      case BinaryOpKind::Div: op = "/"; break;
      case BinaryOpKind::Rem: op = "%"; break;
      case BinaryOpKind::Lt: op = "<"; break;
      case BinaryOpKind::Le: op = "<="; break;
      case BinaryOpKind::Gt: op = ">"; break;
      case BinaryOpKind::Ge: op = ">="; break;
      case BinaryOpKind::Eq: op = "=="; break;
      case BinaryOpKind::Ne: op = "!="; break;
      case BinaryOpKind::And: op = "&&"; break;
      case BinaryOpKind::Or: op = "||"; break;
      }
      return "(" + serializeExpr(expr->left) + " " + op + " " +
             serializeExpr(expr->right) + ")";
    }
    case ContractExprKind::UnaryOp: {
      std::string op = expr->unaryOp == UnaryOpKind::Not ? "!" : "-";
      return op + serializeExpr(expr->operand);
    }
    }
    return "";
  }

  mlir::MLIRContext& mlirCtx_;
  clang::ASTContext& astCtx_;
  const std::map<const clang::FunctionDecl*, ContractInfo>& contracts_;
  mlir::OpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> lowerToArc(
    mlir::MLIRContext& context,
    clang::ASTContext& astContext,
    const std::map<const clang::FunctionDecl*, ContractInfo>& contracts) {
  ArcLowering lowering(context, astContext, contracts);
  return lowering.lower();
}

} // namespace arcanum
```

**Step 12: Write `Passes.h`**

```cpp
#ifndef ARCANUM_PASSES_PASSES_H
#define ARCANUM_PASSES_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace arcanum {

/// Run MLIR passes on the Arc module.
/// Slice 1: identity pass-through + MLIR verifier only.
mlir::LogicalResult runPasses(mlir::ModuleOp module);

} // namespace arcanum

#endif // ARCANUM_PASSES_PASSES_H
```

**Step 13: Write `Passes.cpp`**

```cpp
#include "passes/Passes.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

namespace arcanum {

mlir::LogicalResult runPasses(mlir::ModuleOp module) {
  // Slice 1: identity pass-through.
  // Just run the MLIR verifier to catch malformed IR.
  mlir::PassManager pm(module->getContext());

  // The PassManager's built-in verifier runs automatically after each pass.
  // Since we have no passes, just verify the module manually.
  if (mlir::verify(module).failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

} // namespace arcanum
```

**Step 14: Write the failing test `ArcDialectTest.cpp`**

```cpp
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

class ArcDialectTest : public ::testing::Test {
protected:
  void SetUp() override {
    context_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&context_);
  }

  mlir::MLIRContext context_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

TEST_F(ArcDialectTest, I32TypeCreation) {
  auto type = arc::I32Type::get(&context_);
  EXPECT_TRUE(type);
}

TEST_F(ArcDialectTest, BoolTypeCreation) {
  auto type = arc::BoolType::get(&context_);
  EXPECT_TRUE(type);
}

TEST_F(ArcDialectTest, ConstantOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto constOp = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type,
      builder_->getI32IntegerAttr(42));

  EXPECT_TRUE(constOp);
  module->destroy();
}

TEST_F(ArcDialectTest, AddOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto addOp = builder_->create<arc::AddOp>(
      builder_->getUnknownLoc(), i32Type, lhs, rhs);

  EXPECT_TRUE(addOp);
  module->destroy();
}

} // namespace
} // namespace arcanum
```

**Step 15: Write the failing test `LoweringTest.cpp`**

```cpp
#include "dialect/Lowering.h"
#include "frontend/ContractParser.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"

#include "clang/Tooling/Tooling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(LoweringTest, LowersSimpleAddFunction) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) {
      return a + b;
    }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());

  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  // Check that we have at least one arc.func operation
  bool foundFunc = false;
  module->walk([&](arc::FuncOp funcOp) {
    EXPECT_EQ(funcOp.getSymName(), "safe_add");
    foundFunc = true;
  });
  EXPECT_TRUE(foundFunc);
}

TEST(LoweringTest, LowersIfElseFunction) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    int32_t myAbs(int32_t x) {
      if (x < 0) {
        return -x;
      } else {
        return x;
      }
    }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundIf = false;
  module->walk([&](arc::IfOp ifOp) {
    foundIf = true;
  });
  EXPECT_TRUE(foundIf);
}

} // namespace
} // namespace arcanum
```

**Step 16: Run the tests to verify they compile and pass**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target ArcDialectTest LoweringTest && ctest --test-dir build/default -R "ArcDialectTest|LoweringTest" -V`
Expected: All tests PASS

**Step 17: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/src/dialect/ arcanum/src/passes/ arcanum/tests/unit/ArcDialectTest.cpp arcanum/tests/unit/LoweringTest.cpp
git commit -m "feat(arcanum): add Arc MLIR dialect, Clang-to-MLIR lowering, and pass manager"
```

**Step 18: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| TableGen compiles ArcDialect.td, ArcOps.td, ArcTypes.td | mlir_tablegen targets generate .h.inc/.cpp.inc | |
| Arc dialect types: I32Type, BoolType | Created and registered in dialect | |
| Arc dialect ops: func, constant, add/sub/mul/div/rem, cmp, and/or/not, var, assign, return, if | All Slice 1 ops defined | |
| Lowering handles: functions, params, arithmetic, comparisons, if/else, return | Maps Clang AST nodes to Arc ops | |
| Contracts attached as string attributes on arc.func | requires_attr and ensures_attr populated | |
| Pass manager runs MLIR verifier | Identity pass + verify | |
| Unit tests for dialect types/ops and lowering | Tests pass | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 7: Stage 6 -- WhyML Emitter

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/src/backend/WhyMLEmitter.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/backend/WhyMLEmitter.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/unit/WhyMLEmitterTest.cpp`

**Agent role:** senior-engineer

**Step 1: Write `WhyMLEmitter.h`**

```cpp
#ifndef ARCANUM_BACKEND_WHYMLEMITTER_H
#define ARCANUM_BACKEND_WHYMLEMITTER_H

#include "mlir/IR/BuiltinOps.h"

#include <map>
#include <optional>
#include <string>

namespace arcanum {

/// Source location mapping: maps WhyML construct identifiers back to
/// original C++ source locations.
struct LocationEntry {
  std::string functionName;
  std::string fileName;
  unsigned line = 0;
};

struct WhyMLResult {
  std::string whymlText;        // The generated WhyML source
  std::string filePath;         // Path to temporary .mlw file
  std::map<std::string, LocationEntry> locationMap;
};

/// Emit WhyML from an Arc MLIR module. Writes the .mlw file to a temp path.
std::optional<WhyMLResult> emitWhyML(mlir::ModuleOp module);

} // namespace arcanum

#endif // ARCANUM_BACKEND_WHYMLEMITTER_H
```

**Step 2: Write the failing test `WhyMLEmitterTest.cpp`**

```cpp
#include "backend/WhyMLEmitter.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"
#include "dialect/Lowering.h"
#include "frontend/ContractParser.h"

#include "clang/Tooling/Tooling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(WhyMLEmitterTest, EmitsSafeAddModule) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) {
      return a + b;
    }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // Check WhyML text contains expected constructs
  EXPECT_NE(result->whymlText.find("module"), std::string::npos);
  EXPECT_NE(result->whymlText.find("use int.Int"), std::string::npos);
  EXPECT_NE(result->whymlText.find("safe_add"), std::string::npos);
  EXPECT_NE(result->whymlText.find("requires"), std::string::npos);
  EXPECT_NE(result->whymlText.find("ensures"), std::string::npos);
  EXPECT_NE(result->whymlText.find("end"), std::string::npos);
}

TEST(WhyMLEmitterTest, EmitsOverflowAssertions) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) {
      return a + b;
    }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // Check overflow bounds are emitted
  EXPECT_NE(result->whymlText.find("-2147483648"), std::string::npos);
  EXPECT_NE(result->whymlText.find("2147483647"), std::string::npos);
}

TEST(WhyMLEmitterTest, LocationMapPopulated) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t foo(int32_t a) { return a; }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->locationMap.empty());
}

} // namespace
} // namespace arcanum
```

**Step 3: Run the test to verify it fails**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target WhyMLEmitterTest && ctest --test-dir build/default -R WhyMLEmitterTest -V`
Expected: Build failure (WhyMLEmitter.cpp not yet implemented)

**Step 4: Implement `WhyMLEmitter.cpp`**

```cpp
#include "backend/WhyMLEmitter.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <sstream>

namespace arcanum {
namespace {

/// Convert a contract expression string from Arc format to WhyML format.
/// Transforms: \result -> result, && -> /\, || -> \/, etc.
std::string contractToWhyML(llvm::StringRef contract) {
  std::string result;
  size_t i = 0;
  while (i < contract.size()) {
    if (contract.substr(i).starts_with("\\result")) {
      result += "result";
      i += 7;
    } else if (contract.substr(i).starts_with("&&")) {
      result += "/\\";
      i += 2;
    } else if (contract.substr(i).starts_with("||")) {
      result += "\\/";
      i += 2;
    } else if (contract.substr(i).starts_with("==")) {
      result += "=";
      i += 2;
    } else if (contract.substr(i).starts_with("!=")) {
      result += "<>";
      i += 2;
    } else {
      result += contract[i];
      ++i;
    }
  }
  return result;
}

/// Convert a CamelCase or snake_case function name to a WhyML module name
/// (first letter capitalized).
std::string toModuleName(llvm::StringRef funcName) {
  std::string result = funcName.str();
  if (!result.empty()) {
    result[0] = std::toupper(result[0]);
    // Convert snake_case to CamelCase for module name
    std::string camel;
    bool nextUpper = true;
    for (char c : result) {
      if (c == '_') {
        nextUpper = true;
      } else if (nextUpper) {
        camel += std::toupper(c);
        nextUpper = false;
      } else {
        camel += c;
      }
    }
    return camel;
  }
  return "Module";
}

class WhyMLWriter {
public:
  explicit WhyMLWriter(mlir::ModuleOp module) : module_(module) {}

  std::optional<WhyMLResult> emit() {
    WhyMLResult result;

    module_.walk([&](arc::FuncOp funcOp) {
      emitFunction(funcOp, result);
    });

    if (result.whymlText.empty()) {
      return std::nullopt;
    }

    // Write to temp file
    llvm::SmallString<128> tmpPath;
    std::error_code ec;
    ec = llvm::sys::fs::createTemporaryFile("arcanum", "mlw", tmpPath);
    if (ec) {
      return std::nullopt;
    }

    std::ofstream out(tmpPath.c_str());
    out << result.whymlText;
    out.close();

    result.filePath = tmpPath.str().str();
    return result;
  }

private:
  void emitFunction(arc::FuncOp funcOp, WhyMLResult& result) {
    std::ostringstream out;
    auto funcName = funcOp.getSymName().str();
    auto moduleName = toModuleName(funcName);

    out << "module " << moduleName << "\n";
    out << "  use int.Int\n\n";

    // Function signature
    auto funcType = funcOp.getFunctionType();
    out << "  let " << funcName << " ";

    // Parameters
    auto& entryBlock = funcOp.getBody().front();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      // Try to get parameter name from source location or use default
      out << "(arg" << i << ": int) ";
    }
    out << ": int\n";

    // Requires clauses
    if (auto reqAttr = funcOp.getRequiresAttrAttr()) {
      auto reqStr = reqAttr.getValue();
      // Split on && at top level for separate requires clauses
      out << "    requires { " << contractToWhyML(reqStr) << " }\n";
    }

    // Ensures clauses
    if (auto ensAttr = funcOp.getEnsuresAttrAttr()) {
      auto ensStr = ensAttr.getValue();
      out << "    ensures  { " << contractToWhyML(ensStr) << " }\n";
    }

    // Function body - walk the ops and emit WhyML
    out << "  =\n";
    emitBody(funcOp, out);

    out << "\nend\n\n";

    result.whymlText += out.str();

    // Populate location map
    auto loc = funcOp.getLoc();
    LocationEntry entry;
    entry.functionName = funcName;
    if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) {
      entry.fileName = fileLoc.getFilename().str();
      entry.line = fileLoc.getLine();
    }
    result.locationMap[funcName] = entry;
  }

  void emitBody(arc::FuncOp funcOp, std::ostringstream& out) {
    auto& entryBlock = funcOp.getBody().front();
    // Map MLIR values to WhyML variable names
    llvm::DenseMap<mlir::Value, std::string> nameMap;

    // Map block arguments to parameter names
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      nameMap[entryBlock.getArgument(i)] = "arg" + std::to_string(i);
    }

    for (auto& op : entryBlock.getOperations()) {
      emitOp(op, out, nameMap);
    }
  }

  void emitOp(mlir::Operation& op, std::ostringstream& out,
              llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    if (auto constOp = llvm::dyn_cast<arc::ConstantOp>(&op)) {
      auto attr = constOp.getValue();
      std::string valStr;
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
        valStr = std::to_string(intAttr.getInt());
      } else if (auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(attr)) {
        valStr = boolAttr.getValue() ? "true" : "false";
      }
      nameMap[constOp.getResult()] = valStr;
    } else if (auto addOp = llvm::dyn_cast<arc::AddOp>(&op)) {
      auto lhs = getExpr(addOp.getLhs(), nameMap);
      auto rhs = getExpr(addOp.getRhs(), nameMap);
      auto expr = "(" + lhs + " + " + rhs + ")";

      // Emit overflow assertion for trap mode
      out << "    (* overflow check for addition *)\n";
      out << "    assert { -2147483648 <= " << lhs << " + " << rhs
          << " <= 2147483647 };\n";

      nameMap[addOp.getResult()] = expr;
    } else if (auto subOp = llvm::dyn_cast<arc::SubOp>(&op)) {
      auto lhs = getExpr(subOp.getLhs(), nameMap);
      auto rhs = getExpr(subOp.getRhs(), nameMap);
      auto expr = "(" + lhs + " - " + rhs + ")";
      out << "    assert { -2147483648 <= " << lhs << " - " << rhs
          << " <= 2147483647 };\n";
      nameMap[subOp.getResult()] = expr;
    } else if (auto mulOp = llvm::dyn_cast<arc::MulOp>(&op)) {
      auto lhs = getExpr(mulOp.getLhs(), nameMap);
      auto rhs = getExpr(mulOp.getRhs(), nameMap);
      auto expr = "(" + lhs + " * " + rhs + ")";
      out << "    assert { -2147483648 <= " << lhs << " * " << rhs
          << " <= 2147483647 };\n";
      nameMap[mulOp.getResult()] = expr;
    } else if (auto divOp = llvm::dyn_cast<arc::DivOp>(&op)) {
      auto lhs = getExpr(divOp.getLhs(), nameMap);
      auto rhs = getExpr(divOp.getRhs(), nameMap);
      auto expr = "(div " + lhs + " " + rhs + ")";
      out << "    assert { " << rhs << " <> 0 };\n";
      nameMap[divOp.getResult()] = expr;
    } else if (auto remOp = llvm::dyn_cast<arc::RemOp>(&op)) {
      auto lhs = getExpr(remOp.getLhs(), nameMap);
      auto rhs = getExpr(remOp.getRhs(), nameMap);
      auto expr = "(mod " + lhs + " " + rhs + ")";
      out << "    assert { " << rhs << " <> 0 };\n";
      nameMap[remOp.getResult()] = expr;
    } else if (auto cmpOp = llvm::dyn_cast<arc::CmpOp>(&op)) {
      auto lhs = getExpr(cmpOp.getLhs(), nameMap);
      auto rhs = getExpr(cmpOp.getRhs(), nameMap);
      auto pred = cmpOp.getPredicate().str();
      std::string whymlOp;
      if (pred == "lt") whymlOp = "<";
      else if (pred == "le") whymlOp = "<=";
      else if (pred == "gt") whymlOp = ">";
      else if (pred == "ge") whymlOp = ">=";
      else if (pred == "eq") whymlOp = "=";
      else if (pred == "ne") whymlOp = "<>";
      nameMap[cmpOp.getResult()] = "(" + lhs + " " + whymlOp + " " + rhs + ")";
    } else if (auto andOp = llvm::dyn_cast<arc::AndOp>(&op)) {
      auto lhs = getExpr(andOp.getLhs(), nameMap);
      auto rhs = getExpr(andOp.getRhs(), nameMap);
      nameMap[andOp.getResult()] = "(" + lhs + " /\\ " + rhs + ")";
    } else if (auto orOp = llvm::dyn_cast<arc::OrOp>(&op)) {
      auto lhs = getExpr(orOp.getLhs(), nameMap);
      auto rhs = getExpr(orOp.getRhs(), nameMap);
      nameMap[orOp.getResult()] = "(" + lhs + " \\/ " + rhs + ")";
    } else if (auto notOp = llvm::dyn_cast<arc::NotOp>(&op)) {
      auto operand = getExpr(notOp.getOperand(), nameMap);
      nameMap[notOp.getResult()] = "(not " + operand + ")";
    } else if (auto retOp = llvm::dyn_cast<arc::ReturnOp>(&op)) {
      if (retOp.getValue()) {
        auto val = getExpr(retOp.getValue(), nameMap);
        out << "    " << val << "\n";
      }
    } else if (auto varOp = llvm::dyn_cast<arc::VarOp>(&op)) {
      auto init = getExpr(varOp.getInit(), nameMap);
      auto name = varOp.getName().str();
      out << "    let " << name << " = " << init << " in\n";
      nameMap[varOp.getResult()] = name;
    }
    // Note: IfOp handling is more complex and will emit if-then-else in WhyML
  }

  std::string getExpr(mlir::Value val,
                      llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto it = nameMap.find(val);
    if (it != nameMap.end()) {
      return it->second;
    }
    return "?unknown?";
  }

  mlir::ModuleOp module_;
};

} // namespace

std::optional<WhyMLResult> emitWhyML(mlir::ModuleOp module) {
  WhyMLWriter writer(module);
  return writer.emit();
}

} // namespace arcanum
```

**Step 5: Run the tests to verify they pass**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target WhyMLEmitterTest && ctest --test-dir build/default -R WhyMLEmitterTest -V`
Expected: All tests PASS

**Step 6: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/src/backend/WhyMLEmitter.h arcanum/src/backend/WhyMLEmitter.cpp arcanum/tests/unit/WhyMLEmitterTest.cpp
git commit -m "feat(arcanum): add WhyML emitter translating Arc MLIR to .mlw files"
```

**Step 7: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Emits valid WhyML module with `use int.Int` | Module structure correct | |
| Translates requires/ensures contracts to WhyML syntax | \result -> result, && -> /\\ | |
| Emits overflow assertions for arithmetic ops | -2147483648 <= expr <= 2147483647 | |
| Writes to temp .mlw file | File path in result | |
| Location map maps function names to source locations | Map populated | |
| Unit tests check WhyML structure and overflow assertions | All tests pass | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 8: Stage 7 -- Why3 Runner

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/src/backend/Why3Runner.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/backend/Why3Runner.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/unit/Why3RunnerTest.cpp`

**Agent role:** senior-engineer

**Step 1: Write `Why3Runner.h`**

```cpp
#ifndef ARCANUM_BACKEND_WHY3RUNNER_H
#define ARCANUM_BACKEND_WHY3RUNNER_H

#include <chrono>
#include <string>
#include <vector>

namespace arcanum {

enum class ObligationStatus {
  Valid,
  Unknown,
  Timeout,
  Failure,
};

struct ObligationResult {
  std::string name;
  ObligationStatus status = ObligationStatus::Unknown;
  std::chrono::milliseconds duration{0};
};

/// Run Why3 on a .mlw file with the given solver and timeout.
/// Returns per-obligation results parsed from Why3 stdout.
std::vector<ObligationResult> runWhy3(const std::string& mlwPath,
                                      const std::string& why3Binary = "why3",
                                      int timeoutSeconds = 30);

/// Parse Why3 stdout output into obligation results (exposed for testing).
std::vector<ObligationResult> parseWhy3Output(const std::string& output);

} // namespace arcanum

#endif // ARCANUM_BACKEND_WHY3RUNNER_H
```

**Step 2: Write the failing test `Why3RunnerTest.cpp`**

```cpp
#include "backend/Why3Runner.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(Why3RunnerTest, ParsesValidObligation) {
  std::string output = R"(
File "test.mlw", line 5, characters 10-30:
    Goal safe_add'vc. Valid (0.01s, 0 steps).
)";
  auto results = parseWhy3Output(output);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Valid);
}

TEST(Why3RunnerTest, ParsesTimeoutObligation) {
  std::string output = R"(
File "test.mlw", line 5, characters 10-30:
    Goal safe_add'vc. Timeout.
)";
  auto results = parseWhy3Output(output);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Timeout);
}

TEST(Why3RunnerTest, ParsesUnknownObligation) {
  std::string output = R"(
File "test.mlw", line 5, characters 10-30:
    Goal safe_add'vc. Unknown ("unknown").
)";
  auto results = parseWhy3Output(output);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Unknown);
}

TEST(Why3RunnerTest, ParsesMultipleObligations) {
  std::string output = R"(
File "test.mlw", line 3, characters 10-30:
    Goal overflow_check'vc. Valid (0.01s, 0 steps).
File "test.mlw", line 5, characters 10-30:
    Goal postcondition'vc. Valid (0.02s, 0 steps).
)";
  auto results = parseWhy3Output(output);
  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].status, ObligationStatus::Valid);
  EXPECT_EQ(results[1].status, ObligationStatus::Valid);
}

TEST(Why3RunnerTest, ParsesEmptyOutput) {
  auto results = parseWhy3Output("");
  EXPECT_TRUE(results.empty());
}

} // namespace
} // namespace arcanum
```

**Step 3: Run the test to verify it fails**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target Why3RunnerTest && ctest --test-dir build/default -R Why3RunnerTest -V`
Expected: Build failure (Why3Runner.cpp not yet implemented)

**Step 4: Implement `Why3Runner.cpp`**

```cpp
#include "backend/Why3Runner.h"

#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdio>
#include <memory>
#include <regex>
#include <sstream>

namespace arcanum {

std::vector<ObligationResult> parseWhy3Output(const std::string& output) {
  std::vector<ObligationResult> results;

  // Match lines like: "    Goal <name>. Valid (0.01s, 0 steps)."
  // or: "    Goal <name>. Timeout."
  // or: "    Goal <name>. Unknown ("reason")."
  std::regex goalRegex(
      R"(Goal\s+(\S+)\.\s+(Valid|Timeout|Unknown)(?:\s+\(([^)]*)\))?)");

  std::istringstream stream(output);
  std::string line;
  while (std::getline(stream, line)) {
    std::smatch match;
    if (std::regex_search(line, match, goalRegex)) {
      ObligationResult result;
      result.name = match[1].str();

      auto statusStr = match[2].str();
      if (statusStr == "Valid") {
        result.status = ObligationStatus::Valid;
      } else if (statusStr == "Timeout") {
        result.status = ObligationStatus::Timeout;
      } else if (statusStr == "Unknown") {
        result.status = ObligationStatus::Unknown;
      } else {
        result.status = ObligationStatus::Failure;
      }

      // Parse duration if present (e.g., "0.01s, 0 steps")
      if (match.size() > 3 && match[3].matched) {
        std::regex durationRegex(R"(([\d.]+)s)");
        std::smatch durMatch;
        auto detailStr = match[3].str();
        if (std::regex_search(detailStr, durMatch, durationRegex)) {
          double seconds = std::stod(durMatch[1].str());
          result.duration = std::chrono::milliseconds(
              static_cast<int>(seconds * 1000));
        }
      }

      results.push_back(std::move(result));
    }
  }

  return results;
}

std::vector<ObligationResult> runWhy3(const std::string& mlwPath,
                                      const std::string& why3Binary,
                                      int timeoutSeconds) {
  // Find the why3 binary
  auto why3 = llvm::sys::findProgramByName(why3Binary);
  if (!why3) {
    ObligationResult err;
    err.name = "why3_not_found";
    err.status = ObligationStatus::Failure;
    return {err};
  }

  // Build command: why3 prove -P z3 --timelimit=<timeout> <file.mlw>
  std::string cmd = why3.get() + " prove -P z3 --timelimit=" +
                    std::to_string(timeoutSeconds) + " " + mlwPath +
                    " 2>&1";

  // Execute and capture output
  std::array<char, 4096> buffer;
  std::string output;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                 pclose);
  if (!pipe) {
    ObligationResult err;
    err.name = "execution_error";
    err.status = ObligationStatus::Failure;
    return {err};
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    output += buffer.data();
  }

  return parseWhy3Output(output);
}

} // namespace arcanum
```

**Step 5: Run the tests to verify they pass**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target Why3RunnerTest && ctest --test-dir build/default -R Why3RunnerTest -V`
Expected: All tests PASS (output parsing tests do not require Why3 at runtime)

**Step 6: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/src/backend/Why3Runner.h arcanum/src/backend/Why3Runner.cpp arcanum/tests/unit/Why3RunnerTest.cpp
git commit -m "feat(arcanum): add Why3 runner with subprocess invocation and output parsing"
```

**Step 7: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| parseWhy3Output handles Valid, Timeout, Unknown | Regex parsing correct | |
| Duration extracted from output | Millisecond precision | |
| runWhy3 validates why3 binary in PATH | Error handling for missing binary | |
| Command builds `why3 prove -P z3 --timelimit=N file.mlw` | Correct arguments | |
| Unit tests cover all obligation statuses and edge cases | 5+ test cases | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 9: Stage 8 -- Report Generator

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/src/report/ReportGenerator.h`
- Create: `/home/yotto/ad-adas-memo/arcanum/src/report/ReportGenerator.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/unit/ReportGeneratorTest.cpp`

**Agent role:** junior-engineer

**Step 1: Write `ReportGenerator.h`**

```cpp
#ifndef ARCANUM_REPORT_REPORTGENERATOR_H
#define ARCANUM_REPORT_REPORTGENERATOR_H

#include "backend/Why3Runner.h"
#include "backend/WhyMLEmitter.h"

#include <string>
#include <vector>
#include <map>

namespace arcanum {

struct Report {
  std::string text;
  bool allPassed = true;
  int passCount = 0;
  int failCount = 0;
  int timeoutCount = 0;
};

/// Generate a human-readable verification report.
Report generateReport(
    const std::vector<ObligationResult>& obligations,
    const std::map<std::string, LocationEntry>& locationMap);

} // namespace arcanum

#endif // ARCANUM_REPORT_REPORTGENERATOR_H
```

**Step 2: Write the failing test `ReportGeneratorTest.cpp`**

```cpp
#include "report/ReportGenerator.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(ReportGeneratorTest, AllPassedReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back(
      {"overflow_check'vc", ObligationStatus::Valid,
       std::chrono::milliseconds(100)});
  obligations.push_back(
      {"postcondition'vc", ObligationStatus::Valid,
       std::chrono::milliseconds(200)});

  std::map<std::string, LocationEntry> locMap;
  locMap["safe_add"] = {"safe_add", "input.cpp", 6};

  auto report = generateReport(obligations, locMap);

  EXPECT_TRUE(report.allPassed);
  EXPECT_EQ(report.passCount, 1);
  EXPECT_EQ(report.failCount, 0);
  EXPECT_EQ(report.timeoutCount, 0);
  EXPECT_NE(report.text.find("[PASS]"), std::string::npos);
  EXPECT_NE(report.text.find("2/2"), std::string::npos);
  EXPECT_NE(report.text.find("Summary"), std::string::npos);
}

TEST(ReportGeneratorTest, FailedReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back(
      {"overflow_check'vc", ObligationStatus::Valid,
       std::chrono::milliseconds(100)});
  obligations.push_back(
      {"postcondition'vc", ObligationStatus::Unknown,
       std::chrono::milliseconds(200)});

  std::map<std::string, LocationEntry> locMap;
  locMap["bad_func"] = {"bad_func", "input.cpp", 10};

  auto report = generateReport(obligations, locMap);

  EXPECT_FALSE(report.allPassed);
  EXPECT_NE(report.text.find("[FAIL]"), std::string::npos);
}

TEST(ReportGeneratorTest, TimeoutReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back(
      {"invariant'vc", ObligationStatus::Timeout,
       std::chrono::milliseconds(30000)});

  std::map<std::string, LocationEntry> locMap;

  auto report = generateReport(obligations, locMap);

  EXPECT_FALSE(report.allPassed);
  EXPECT_EQ(report.timeoutCount, 1);
}

TEST(ReportGeneratorTest, EmptyObligations) {
  std::vector<ObligationResult> obligations;
  std::map<std::string, LocationEntry> locMap;

  auto report = generateReport(obligations, locMap);

  EXPECT_TRUE(report.allPassed);
  EXPECT_NE(report.text.find("Summary"), std::string::npos);
}

} // namespace
} // namespace arcanum
```

**Step 3: Run the test to verify it fails**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target ReportGeneratorTest && ctest --test-dir build/default -R ReportGeneratorTest -V`
Expected: Build failure (ReportGenerator.cpp not yet implemented)

**Step 4: Implement `ReportGenerator.cpp`**

```cpp
#include "report/ReportGenerator.h"

#include <chrono>
#include <sstream>

namespace arcanum {

Report generateReport(
    const std::vector<ObligationResult>& obligations,
    const std::map<std::string, LocationEntry>& locationMap) {
  Report report;

  int validCount = 0;
  int totalCount = static_cast<int>(obligations.size());
  bool hasUnknown = false;
  bool hasTimeout = false;
  auto totalDuration = std::chrono::milliseconds(0);

  for (const auto& ob : obligations) {
    totalDuration += ob.duration;
    switch (ob.status) {
    case ObligationStatus::Valid:
      ++validCount;
      break;
    case ObligationStatus::Unknown:
    case ObligationStatus::Failure:
      hasUnknown = true;
      break;
    case ObligationStatus::Timeout:
      hasTimeout = true;
      break;
    }
  }

  std::ostringstream out;

  double totalSeconds =
      static_cast<double>(totalDuration.count()) / 1000.0;

  // Per-function report line
  if (totalCount > 0) {
    // Try to get the first function from the location map
    std::string funcLine;
    if (!locationMap.empty()) {
      auto& entry = locationMap.begin()->second;
      funcLine = entry.fileName + ":" + entry.functionName;
    } else {
      funcLine = "unknown";
    }

    if (validCount == totalCount) {
      out << "[PASS]  " << funcLine << "    " << validCount << "/"
          << totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.passCount = 1;
    } else if (hasTimeout && !hasUnknown) {
      out << "[TIMEOUT]  " << funcLine << "    " << validCount << "/"
          << totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.timeoutCount = 1;
      report.allPassed = false;
    } else {
      out << "[FAIL]  " << funcLine << "    " << validCount << "/"
          << totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.failCount = 1;
      report.allPassed = false;
    }
  }

  // Summary line
  out << "\nSummary: " << report.passCount << " passed, " << report.failCount
      << " failed, " << report.timeoutCount << " timeout\n";

  report.text = out.str();
  return report;
}

} // namespace arcanum
```

**Step 5: Run the tests to verify they pass**

Run: `cd /home/yotto/ad-adas-memo/arcanum && cmake --build build/default --target ReportGeneratorTest && ctest --test-dir build/default -R ReportGeneratorTest -V`
Expected: All tests PASS

**Step 6: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/src/report/ReportGenerator.h arcanum/src/report/ReportGenerator.cpp arcanum/tests/unit/ReportGeneratorTest.cpp
git commit -m "feat(arcanum): add ReportGenerator for terminal verification output"
```

**Step 7: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Formats [PASS]/[FAIL]/[TIMEOUT] lines with obligation counts | Matches spec format | |
| Includes summary line with pass/fail/timeout counts | Summary present | |
| Handles empty obligations gracefully | No crash, valid output | |
| Report.allPassed reflects actual results | true only when all Valid | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 10: Test Infrastructure + Integration Tests + CI

**Files:**
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/CMakeLists.txt`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/lit/lit.cfg.py`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/lit/lit.site.cfg.py.in`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/lit/CMakeLists.txt`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/lit/subset-check/reject-virtual.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/lit/subset-check/reject-raw-ptr.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/lit/verify/pass-simple-add.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/lit/verify/fail-overflow.cpp`
- Create: `/home/yotto/ad-adas-memo/arcanum/tests/unit/CMakeLists.txt` (or embed in tests/CMakeLists.txt)
- Create: `/home/yotto/ad-adas-memo/arcanum/.github/workflows/ci.yml`

**Agent role:** senior-engineer

**Step 1: Create `tests/CMakeLists.txt` -- GoogleTest + lit**

```cmake
# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Suppress Clang warnings in GoogleTest code
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(gtest PRIVATE -Wno-covered-switch-default)
  target_compile_options(gmock PRIVATE -Wno-covered-switch-default)
endif()

include(GoogleTest)

# --- SubsetEnforcerTest ---
add_executable(SubsetEnforcerTest unit/SubsetEnforcerTest.cpp)
target_include_directories(SubsetEnforcerTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_link_libraries(SubsetEnforcerTest PRIVATE
  ArcanumFrontend
  gtest_main
  clangTooling
)
gtest_discover_tests(SubsetEnforcerTest)

# --- ContractParserTest ---
add_executable(ContractParserTest unit/ContractParserTest.cpp)
target_include_directories(ContractParserTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_link_libraries(ContractParserTest PRIVATE
  ArcanumFrontend
  gtest_main
  clangTooling
)
gtest_discover_tests(ContractParserTest)

# --- ArcDialectTest ---
add_executable(ArcDialectTest unit/ArcDialectTest.cpp)
target_include_directories(ArcDialectTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_link_libraries(ArcDialectTest PRIVATE
  ArcDialect
  gtest_main
  MLIRIR
)
gtest_discover_tests(ArcDialectTest)

# --- LoweringTest ---
add_executable(LoweringTest unit/LoweringTest.cpp)
target_include_directories(LoweringTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_link_libraries(LoweringTest PRIVATE
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
)
gtest_discover_tests(LoweringTest)

# --- WhyMLEmitterTest ---
add_executable(WhyMLEmitterTest unit/WhyMLEmitterTest.cpp)
target_include_directories(WhyMLEmitterTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_link_libraries(WhyMLEmitterTest PRIVATE
  ArcanumBackend
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
  LLVMSupport
)
gtest_discover_tests(WhyMLEmitterTest)

# --- Why3RunnerTest ---
add_executable(Why3RunnerTest unit/Why3RunnerTest.cpp)
target_include_directories(Why3RunnerTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${ARCANUM_LLVM_INCLUDE_DIRS}
)
target_link_libraries(Why3RunnerTest PRIVATE
  ArcanumBackend
  gtest_main
  LLVMSupport
)
gtest_discover_tests(Why3RunnerTest)

# --- ReportGeneratorTest ---
add_executable(ReportGeneratorTest unit/ReportGeneratorTest.cpp)
target_include_directories(ReportGeneratorTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(ReportGeneratorTest PRIVATE
  ArcanumReport
  gtest_main
  LLVMSupport
)
gtest_discover_tests(ReportGeneratorTest)

# --- Lit tests ---
add_subdirectory(lit)
```

**Step 2: Create `tests/lit/lit.site.cfg.py.in`**

```python
# -*- Python -*-

import os
import sys

config.arcanum_obj_root = "@ARCANUM_OBJ_ROOT@"
config.arcanum_src_root = "@ARCANUM_SRC_ROOT@"
config.arcanum_path = "@ARCANUM_PATH@"
config.filecheck_path = "@FILECHECK_PATH@"
config.not_path = "@NOT_PATH@"

config.test_source_root = os.path.dirname(__file__)

# Load the main lit.cfg.py
lit_config.load_config(config, os.path.join(config.test_source_root, "lit.cfg.py"))
```

**Step 3: Create `tests/lit/lit.cfg.py`**

```python
# -*- Python -*-

import os
import lit.formats
import lit.util

config.name = "Arcanum"
config.suffixes = [".cpp"]
config.test_format = lit.formats.ShTest(not lit.util.which("bash"))
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.arcanum_obj_root

config.substitutions.append(("%arcanum", config.arcanum_path))
config.substitutions.append(("%FileCheck", config.filecheck_path + " --match-full-lines"))
config.substitutions.append(("%not", config.not_path))

if lit.util.which("bash"):
    config.available_features.add("shell")

config.environment["PATH"] = os.pathsep.join(
    [os.path.dirname(config.arcanum_path), config.environment.get("PATH", "")]
)
```

**Step 4: Create `tests/lit/CMakeLists.txt`**

```cmake
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LIT_COMMAND "${Python3_EXECUTABLE}" "/usr/lib/llvm-21/build/utils/lit/lit.py"
    CACHE STRING "Command to run lit")

set(LLVM_TOOLS_DIR "/usr/lib/llvm-21/bin" CACHE PATH "Path to LLVM tools")
set(FILECHECK_PATH "${LLVM_TOOLS_DIR}/FileCheck" CACHE FILEPATH "Path to FileCheck")
set(NOT_PATH "${LLVM_TOOLS_DIR}/not" CACHE FILEPATH "Path to not tool")

set(ARCANUM_OBJ_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(ARCANUM_SRC_ROOT "${CMAKE_SOURCE_DIR}")
set(ARCANUM_PATH "${CMAKE_BINARY_DIR}/bin/arcanum")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
    "${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py"
    COPYONLY
)

# Copy test files to build directory
set(TEST_SUBDIRS subset-check verify)
foreach(SUBDIR ${TEST_SUBDIRS})
    file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/*.cpp")
    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME)
        configure_file(
            "${TEST_FILE}"
            "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${TEST_NAME}"
            COPYONLY
        )
    endforeach()
endforeach()

add_custom_target(check-arcanum-lit
    COMMAND ${LIT_COMMAND} -v "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS arcanum
    COMMENT "Running Arcanum lit tests"
    USES_TERMINAL
)

add_test(
    NAME arcanum-lit-tests
    COMMAND ${LIT_COMMAND} -v "${CMAKE_CURRENT_BINARY_DIR}"
)
set_tests_properties(arcanum-lit-tests PROPERTIES
    DEPENDS "arcanum"
)
```

**Step 5: Create lit test `subset-check/reject-virtual.cpp`**

```cpp
// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: error: virtual functions are not allowed

class Base {
public:
  virtual int foo() { return 0; }
};
```

**Step 6: Create lit test `subset-check/reject-raw-ptr.cpp`**

```cpp
// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: error: raw pointer types are not allowed

void foo(int* p) {}
```

**Step 7: Create lit test `verify/pass-simple-add.cpp`**

```cpp
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}safe_add{{.*}}obligations proven

#include <cstdint>

//@ requires: a >= 0 && a <= 1000
//@ requires: b >= 0 && b <= 1000
//@ ensures: \result >= 0 && \result <= 2000
int32_t safe_add(int32_t a, int32_t b) {
    return a + b;
}
```

**Step 8: Create lit test `verify/fail-overflow.cpp`**

```cpp
// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: [FAIL]{{.*}}obligations

#include <cstdint>

//@ ensures: \result >= 0
int32_t unchecked_add(int32_t a, int32_t b) {
    return a + b;
}
```

**Step 9: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository_owner }}/arcanum-base

jobs:
  build-and-test:
    runs-on: ubuntu-24.04
    container:
      image: ghcr.io/${{ github.repository_owner }}/arcanum-base:latest
      credentials:
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
      options: --user root

    name: Build and Test

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure
        run: |
          cd arcanum
          cmake --preset default

      - name: Build
        run: |
          cd arcanum
          cmake --build build/default

      - name: Run unit tests
        run: |
          cd arcanum
          ctest --preset default

      - name: Run lit tests
        run: |
          cd arcanum
          cmake --build build/default --target check-arcanum-lit
```

**Step 10: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/tests/CMakeLists.txt arcanum/tests/lit/ arcanum/.github/workflows/ci.yml
git commit -m "feat(arcanum): add test infrastructure, lit integration tests, and CI workflow"
```

**Step 11: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| tests/CMakeLists.txt links all unit tests to correct libraries | GoogleTest targets build | |
| lit.cfg.py configures %arcanum, %FileCheck, %not substitutions | Correct paths | |
| reject-virtual.cpp and reject-raw-ptr.cpp check subset enforcement | FileCheck patterns match error output | |
| pass-simple-add.cpp end-to-end test passes | [PASS] with obligation count | |
| CI workflow builds inside Docker container | cmake configure + build + ctest + lit | |
| All ctest tests pass | Unit tests + lit tests | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 11: Move Spec Documents to `docs/` and Final README

**Files:**
- Move: `/home/yotto/ad-adas-memo/arcanum/arcanum-tool-spec.md` to `/home/yotto/ad-adas-memo/arcanum/docs/arcanum-tool-spec.md`
- Move: `/home/yotto/ad-adas-memo/arcanum/arcanum-safe-cpp-subset.md` to `/home/yotto/ad-adas-memo/arcanum/docs/arcanum-safe-cpp-subset.md`

**Agent role:** junior-engineer

**Step 1: Move spec documents to `docs/` directory**

```bash
mkdir -p /home/yotto/ad-adas-memo/arcanum/docs
git mv /home/yotto/ad-adas-memo/arcanum/arcanum-tool-spec.md /home/yotto/ad-adas-memo/arcanum/docs/arcanum-tool-spec.md
git mv /home/yotto/ad-adas-memo/arcanum/arcanum-safe-cpp-subset.md /home/yotto/ad-adas-memo/arcanum/docs/arcanum-safe-cpp-subset.md
```

**Step 2: Commit**

```bash
cd /home/yotto/ad-adas-memo
git add arcanum/docs/
git commit -m "refactor(arcanum): move spec documents to docs/ directory"
```

**Step 3: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Both spec files moved to arcanum/docs/ | Files exist at new paths | |
| No stale copies remain at old paths | Old paths deleted by git mv | |
| Project structure matches design doc | docs/ directory present | |

Reviewer: `arcanum-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

## Execution: Team-Driven

> **For Claude:** REQUIRED SUB-SKILL: Use [oneteam:skill] `team-management` skill to orchestrate
> execution starting from Phase 2 (Team Setup).

**Fragments:** 3 (max 4)

### Team Composition

| Name | Type | Scope |
|------|------|-------|
| arcanum-reviewer-1 | code-reviewer | All fragments |
| arcanum-senior-engineer-1 | senior-engineer | Fragment 1 (Tasks 1, 2, 3) |
| arcanum-senior-engineer-2 | senior-engineer | Fragment 2 (Tasks 6, 7, 8) |
| arcanum-senior-engineer-3 | senior-engineer | Fragment 3 (Tasks 4, 5, 9, 10, 11) |

### Fragment 1: Infrastructure + Scaffold + CLI

- **Tasks:** Task 1 (Docker), Task 2 (CMake scaffold), Task 3 (CLI + Clang Frontend)
- **File scope:** `arcanum/docker/`, `arcanum/CMakeLists.txt`, `arcanum/CMakePresets.json`, `arcanum/.clang-format`, `arcanum/.clang-tidy`, `arcanum/.gitignore`, `arcanum/README.md`, `arcanum/src/main.cpp`
- **Agent role:** senior-engineer
- **Inter-fragment dependencies:** None -- this is the foundation all other fragments depend on.

#### Fragment 1: Post-Completion Review

| Stage | Reviewer | Criteria | Status |
|-------|----------|----------|--------|
| 1. Spec compliance | arcanum-reviewer-1 | Docker builds LLVM 21 + MLIR + Why3 + Z3; CMake finds all packages; CLI parses args and orchestrates pipeline | |
| 2. Code quality | arcanum-reviewer-1 | Shell scripts are executable, Dockerfiles follow best practices, CMake uses scoped includes, LLVM naming conventions | |

Both stages must PASS before fragment is merge-ready.

### Fragment 2: Arc Dialect + Backend (WhyML + Why3 + Report)

- **Tasks:** Task 6 (Arc dialect + Lowering + Passes), Task 7 (WhyML Emitter), Task 8 (Why3 Runner), Task 9 (Report Generator)
- **File scope:** `arcanum/src/dialect/`, `arcanum/src/passes/`, `arcanum/src/backend/`, `arcanum/src/report/`, `arcanum/tests/unit/ArcDialectTest.cpp`, `arcanum/tests/unit/LoweringTest.cpp`, `arcanum/tests/unit/WhyMLEmitterTest.cpp`, `arcanum/tests/unit/Why3RunnerTest.cpp`, `arcanum/tests/unit/ReportGeneratorTest.cpp`
- **Agent role:** senior-engineer
- **Inter-fragment dependencies:** Fragment 1 must complete Tasks 1-2 (Docker + CMake scaffold) before this fragment can build and test. Fragment 3 must complete Tasks 4-5 (SubsetEnforcer + ContractParser) before the Lowering can be fully integrated.

#### Fragment 2: Post-Completion Review

| Stage | Reviewer | Criteria | Status |
|-------|----------|----------|--------|
| 1. Spec compliance | arcanum-reviewer-1 | TableGen compiles, Arc ops match design, WhyML output matches spec examples, Why3 output parsing correct, report format matches spec | |
| 2. Code quality | arcanum-reviewer-1 | MLIR patterns match polang reference, overflow assertions present, unit tests comprehensive, no regressions | |

Both stages must PASS before fragment is merge-ready.

### Fragment 3: Frontend Stages + Test Infrastructure + Integration

- **Tasks:** Task 4 (SubsetEnforcer), Task 5 (ContractParser), Task 10 (Test infra + lit + CI), Task 11 (Move docs)
- **File scope:** `arcanum/src/frontend/`, `arcanum/tests/unit/SubsetEnforcerTest.cpp`, `arcanum/tests/unit/ContractParserTest.cpp`, `arcanum/tests/CMakeLists.txt`, `arcanum/tests/lit/`, `arcanum/.github/workflows/`, `arcanum/docs/`
- **Agent role:** senior-engineer
- **Inter-fragment dependencies:** Fragment 1 must complete Tasks 1-2 (Docker + CMake scaffold) before this fragment can build and test. Fragment 2 must complete all tasks before lit end-to-end tests can pass.

#### Fragment 3: Post-Completion Review

| Stage | Reviewer | Criteria | Status |
|-------|----------|----------|--------|
| 1. Spec compliance | arcanum-reviewer-1 | SubsetEnforcer rejects all forbidden constructs, ContractParser handles all annotation forms, lit tests pass end-to-end, CI workflow runs in Docker | |
| 2. Code quality | arcanum-reviewer-1 | RecursiveASTVisitor patterns correct, expression parser handles precedence, GoogleTest + lit setup follows polang patterns, no regressions | |

Both stages must PASS before fragment is merge-ready.

Fragment groupings are designed for parallel execution with worktree isolation.