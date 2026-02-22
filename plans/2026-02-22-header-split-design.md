# Arcanum Header/Source Split Design

## Summary

Reorganize arcanum from a colocated `src/` layout to an LLVM-style split with `include/arcanum/` for headers, `lib/` for implementations, and `tools/` for executables. Add CMake install and package export so external projects can `find_package(Arcanum)` and link against the Arc MLIR dialect.

## Motivation

The MLIR dialect (`ArcDialect`, `ArcOps`, `ArcTypes`) is reusable by other projects that want to generate or consume Arc IR. Currently, all headers are colocated with sources in `src/`, making it impossible to install or consume the dialect externally.

## Directory Layout

```
arcanum/
├── include/arcanum/
│   ├── dialect/
│   │   ├── ArcDialect.h
│   │   ├── ArcDialect.td
│   │   ├── ArcOps.h
│   │   ├── ArcOps.td
│   │   ├── ArcTypes.h
│   │   ├── ArcTypes.td
│   │   └── Lowering.h
│   ├── passes/
│   │   └── Passes.h
│   ├── frontend/
│   │   ├── SubsetEnforcer.h
│   │   └── ContractParser.h
│   ├── backend/
│   │   ├── WhyMLEmitter.h
│   │   └── Why3Runner.h
│   ├── report/
│   │   └── ReportGenerator.h
│   ├── DiagnosticTracker.h
│   └── ExitCodes.h
├── lib/
│   ├── dialect/
│   │   ├── ArcDialect.cpp
│   │   ├── ArcOps.cpp
│   │   ├── ArcTypes.cpp
│   │   └── Lowering.cpp
│   ├── passes/
│   │   └── Passes.cpp
│   ├── frontend/
│   │   ├── SubsetEnforcer.cpp
│   │   └── ContractParser.cpp
│   ├── backend/
│   │   ├── WhyMLEmitter.cpp
│   │   └── Why3Runner.cpp
│   └── report/
│       └── ReportGenerator.cpp
├── tools/arcanum/
│   └── main.cpp
├── tests/
│   ├── (unit tests)
│   └── lit/
├── cmake/
│   └── ArcanumConfig.cmake.in
├── CMakeLists.txt
└── CMakePresets.json
```

Key changes:
- `src/` splits into `include/arcanum/` (headers + .td) and `lib/` (implementations)
- `src/main.cpp` moves to `tools/arcanum/main.cpp`
- New `cmake/ArcanumConfig.cmake.in` for package export

## Include Path Convention

All `#include` directives change from `"subdir/File.h"` to `"arcanum/subdir/File.h"`:

| Before | After |
|--------|-------|
| `#include "dialect/ArcDialect.h"` | `#include "arcanum/dialect/ArcDialect.h"` |
| `#include "dialect/ArcOps.h.inc"` | `#include "arcanum/dialect/ArcOps.h.inc"` |
| `#include "frontend/ContractParser.h"` | `#include "arcanum/frontend/ContractParser.h"` |
| `#include "backend/Why3Runner.h"` | `#include "arcanum/backend/Why3Runner.h"` |
| `#include "DiagnosticTracker.h"` | `#include "arcanum/DiagnosticTracker.h"` |

CMake include directories:
- Source tree: `${PROJECT_SOURCE_DIR}/include`
- Build tree: `${CMAKE_CURRENT_BINARY_DIR}/include` (for generated .inc files)

## TableGen

`.td` files live in `include/arcanum/dialect/`. Generated `.inc` files output to `${CMAKE_CURRENT_BINARY_DIR}/include/arcanum/dialect/`:

```cmake
set(LLVM_TARGET_DEFINITIONS include/arcanum/dialect/ArcOps.td)
mlir_tablegen(include/arcanum/dialect/ArcOps.h.inc -gen-op-decls)
mlir_tablegen(include/arcanum/dialect/ArcOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/arcanum/dialect/ArcDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/arcanum/dialect/ArcDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS include/arcanum/dialect/ArcTypes.td)
mlir_tablegen(include/arcanum/dialect/ArcTypes.h.inc -gen-typedef-decls)
mlir_tablegen(include/arcanum/dialect/ArcTypes.cpp.inc -gen-typedef-defs)
```

## CMake Install & Export

### Install targets

```cmake
# Public headers
install(DIRECTORY include/arcanum
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.td"
)

# Generated .inc files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/arcanum/dialect
  DESTINATION include/arcanum
  FILES_MATCHING PATTERN "*.inc"
)

# Library targets
install(TARGETS ArcDialect ArcanumPasses ArcanumLowering
        ArcanumFrontend ArcanumBackend ArcanumReport
  EXPORT ArcanumTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# CMake package export
install(EXPORT ArcanumTargets
  NAMESPACE Arcanum::
  DESTINATION lib/cmake/Arcanum
)

# Package config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/ArcanumConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ArcanumConfig.cmake
  INSTALL_DESTINATION lib/cmake/Arcanum
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ArcanumConfig.cmake
  DESTINATION lib/cmake/Arcanum
)
```

### External project usage

```cmake
find_package(Arcanum REQUIRED)
target_link_libraries(my_tool Arcanum::ArcDialect Arcanum::ArcanumPasses)
```

```cpp
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
```

## Migration Strategy

Mechanical — no logic changes, only file moves and path updates:

1. Create new directory structure (`include/arcanum/...`, `lib/...`, `tools/arcanum/`, `cmake/`)
2. Move files with `git mv` for history preservation:
   - `.h` and `.td`: `src/<subdir>/` → `include/arcanum/<subdir>/`
   - `.cpp`: `src/<subdir>/` → `lib/<subdir>/`
   - `src/main.cpp` → `tools/arcanum/main.cpp`
   - Root headers: `src/DiagnosticTracker.h`, `src/ExitCodes.h` → `include/arcanum/`
3. Update all `#include` directives — prefix with `arcanum/`
4. Update `CMakeLists.txt` — source paths, include dirs, TableGen, install/export
5. Update test includes
6. Add `cmake/ArcanumConfig.cmake.in`
7. Remove empty `src/` directory
8. Verify: cmake configure + build + all tests pass
