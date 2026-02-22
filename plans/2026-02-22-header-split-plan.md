# LLVM-Style Header/Source Split Implementation Plan

**Goal:** Reorganize arcanum from a colocated `src/` layout to an LLVM-style split with `include/arcanum/` for headers, `lib/` for implementations, and `tools/` for executables, adding CMake install and package export.

**Architecture:** All files in `src/` split by extension: `.h` and `.td` files move to `include/arcanum/<subdir>/`, `.cpp` files move to `lib/<subdir>/` (except `main.cpp` which moves to `tools/arcanum/`). All `#include` directives gain an `arcanum/` prefix. CMake build system is updated for new paths, TableGen output locations, and install/export rules.

**Tech Stack:** CMake 3.20+, MLIR TableGen, Clang/LLVM 21, GoogleTest, lit

**Strategy:** Subagent-driven

---

### Task 1: Create directory structure and move all files with git mv

**Files:**
- Create directories: `arcanum/include/arcanum/dialect/`, `arcanum/include/arcanum/passes/`, `arcanum/include/arcanum/frontend/`, `arcanum/include/arcanum/backend/`, `arcanum/include/arcanum/report/`, `arcanum/lib/dialect/`, `arcanum/lib/passes/`, `arcanum/lib/frontend/`, `arcanum/lib/backend/`, `arcanum/lib/report/`, `arcanum/tools/arcanum/`, `arcanum/cmake/`
- Move: all 26 files from `arcanum/src/` to their new locations
- Delete: `arcanum/src/` (empty after moves)

**Agent role:** junior-engineer

**Step 1: Create the new directory structure**

```bash
cd /workspace/ad-adas-memo/arcanum && \
mkdir -p include/arcanum/dialect \
         include/arcanum/passes \
         include/arcanum/frontend \
         include/arcanum/backend \
         include/arcanum/report \
         lib/dialect \
         lib/passes \
         lib/frontend \
         lib/backend \
         lib/report \
         tools/arcanum \
         cmake
```

**Step 2: Move header files (.h) and TableGen files (.td) to include/arcanum/**

```bash
cd /workspace/ad-adas-memo/arcanum && \
git mv src/dialect/ArcDialect.h include/arcanum/dialect/ArcDialect.h && \
git mv src/dialect/ArcDialect.td include/arcanum/dialect/ArcDialect.td && \
git mv src/dialect/ArcOps.h include/arcanum/dialect/ArcOps.h && \
git mv src/dialect/ArcOps.td include/arcanum/dialect/ArcOps.td && \
git mv src/dialect/ArcTypes.h include/arcanum/dialect/ArcTypes.h && \
git mv src/dialect/ArcTypes.td include/arcanum/dialect/ArcTypes.td && \
git mv src/dialect/Lowering.h include/arcanum/dialect/Lowering.h && \
git mv src/passes/Passes.h include/arcanum/passes/Passes.h && \
git mv src/frontend/SubsetEnforcer.h include/arcanum/frontend/SubsetEnforcer.h && \
git mv src/frontend/ContractParser.h include/arcanum/frontend/ContractParser.h && \
git mv src/backend/WhyMLEmitter.h include/arcanum/backend/WhyMLEmitter.h && \
git mv src/backend/Why3Runner.h include/arcanum/backend/Why3Runner.h && \
git mv src/report/ReportGenerator.h include/arcanum/report/ReportGenerator.h && \
git mv src/DiagnosticTracker.h include/arcanum/DiagnosticTracker.h && \
git mv src/ExitCodes.h include/arcanum/ExitCodes.h
```

**Step 3: Move implementation files (.cpp) to lib/**

```bash
cd /workspace/ad-adas-memo/arcanum && \
git mv src/dialect/ArcDialect.cpp lib/dialect/ArcDialect.cpp && \
git mv src/dialect/ArcOps.cpp lib/dialect/ArcOps.cpp && \
git mv src/dialect/ArcTypes.cpp lib/dialect/ArcTypes.cpp && \
git mv src/dialect/Lowering.cpp lib/dialect/Lowering.cpp && \
git mv src/passes/Passes.cpp lib/passes/Passes.cpp && \
git mv src/frontend/SubsetEnforcer.cpp lib/frontend/SubsetEnforcer.cpp && \
git mv src/frontend/ContractParser.cpp lib/frontend/ContractParser.cpp && \
git mv src/backend/WhyMLEmitter.cpp lib/backend/WhyMLEmitter.cpp && \
git mv src/backend/Why3Runner.cpp lib/backend/Why3Runner.cpp && \
git mv src/report/ReportGenerator.cpp lib/report/ReportGenerator.cpp
```

**Step 4: Move main.cpp to tools/arcanum/**

```bash
cd /workspace/ad-adas-memo/arcanum && \
git mv src/main.cpp tools/arcanum/main.cpp
```

**Step 5: Remove the now-empty src/ directory tree**

```bash
cd /workspace/ad-adas-memo/arcanum && \
rmdir src/dialect src/passes src/frontend src/backend src/report src
```

Expected: all directories should be empty and removable. If any are not empty, it means a file was missed.

**Step 6: Verify the move with git status**

```bash
cd /workspace/ad-adas-memo/arcanum && git status
```

Expected: all 26 files show as renamed (src/* -> include/arcanum/*, lib/*, tools/arcanum/*). No untracked files except new empty directories (cmake/).

**Step 7: Commit the file moves**

```bash
cd /workspace/ad-adas-memo/arcanum && \
git add -A && \
git commit -m "refactor: move files to LLVM-style include/lib/tools layout

Move headers and .td files to include/arcanum/, implementation files to
lib/, and main.cpp to tools/arcanum/. This is a pure file move with no
content changes -- includes and CMake will be updated in follow-up commits."
```

---

### Task 2: Update #include directives in all header files (include/arcanum/)

**Files:**
- Modify: `arcanum/include/arcanum/dialect/ArcDialect.h`
- Modify: `arcanum/include/arcanum/dialect/ArcOps.h`
- Modify: `arcanum/include/arcanum/dialect/ArcTypes.h`
- Modify: `arcanum/include/arcanum/dialect/Lowering.h`
- Modify: `arcanum/include/arcanum/report/ReportGenerator.h`

**Agent role:** junior-engineer

In each file, change internal includes from `"subdir/File.h"` to `"arcanum/subdir/File.h"`:

- `ArcDialect.h`: `"dialect/ArcDialect.h.inc"` → `"arcanum/dialect/ArcDialect.h.inc"`
- `ArcOps.h`: `"dialect/ArcDialect.h"` → `"arcanum/dialect/ArcDialect.h"`, `"dialect/ArcTypes.h"` → `"arcanum/dialect/ArcTypes.h"`, `"dialect/ArcOps.h.inc"` → `"arcanum/dialect/ArcOps.h.inc"`
- `ArcTypes.h`: `"dialect/ArcDialect.h"` → `"arcanum/dialect/ArcDialect.h"`, `"dialect/ArcTypes.h.inc"` → `"arcanum/dialect/ArcTypes.h.inc"`
- `Lowering.h`: `"frontend/ContractParser.h"` → `"arcanum/frontend/ContractParser.h"`
- `ReportGenerator.h`: `"backend/Why3Runner.h"` → `"arcanum/backend/Why3Runner.h"`, `"backend/WhyMLEmitter.h"` → `"arcanum/backend/WhyMLEmitter.h"`

Note: `DiagnosticTracker.h`, `ExitCodes.h`, `SubsetEnforcer.h`, `ContractParser.h`, `WhyMLEmitter.h`, `Why3Runner.h`, and `Passes.h` have NO internal includes to update.

Commit message: `refactor: update #include paths in public headers to arcanum/ prefix`

---

### Task 3: Update #include directives in library source files (lib/) and TableGen .td files

**Files:**
- Modify: `arcanum/lib/dialect/ArcDialect.cpp`
- Modify: `arcanum/lib/dialect/ArcOps.cpp`
- Modify: `arcanum/lib/dialect/ArcTypes.cpp`
- Modify: `arcanum/lib/dialect/Lowering.cpp`
- Modify: `arcanum/lib/passes/Passes.cpp`
- Modify: `arcanum/lib/frontend/SubsetEnforcer.cpp`
- Modify: `arcanum/lib/frontend/ContractParser.cpp`
- Modify: `arcanum/lib/backend/WhyMLEmitter.cpp`
- Modify: `arcanum/lib/backend/Why3Runner.cpp`
- Modify: `arcanum/lib/report/ReportGenerator.cpp`
- Modify: `arcanum/include/arcanum/dialect/ArcOps.td`
- Modify: `arcanum/include/arcanum/dialect/ArcTypes.td`

**Agent role:** junior-engineer

In each `.cpp` file, prefix all internal includes with `arcanum/`:
- `"dialect/ArcDialect.h"` → `"arcanum/dialect/ArcDialect.h"`
- `"dialect/ArcOps.h"` → `"arcanum/dialect/ArcOps.h"`
- `"dialect/ArcTypes.h"` → `"arcanum/dialect/ArcTypes.h"`
- `"dialect/Lowering.h"` → `"arcanum/dialect/Lowering.h"`
- `"dialect/ArcDialect.cpp.inc"` → `"arcanum/dialect/ArcDialect.cpp.inc"`
- `"dialect/ArcOps.cpp.inc"` → `"arcanum/dialect/ArcOps.cpp.inc"`
- `"dialect/ArcTypes.cpp.inc"` → `"arcanum/dialect/ArcTypes.cpp.inc"`
- `"DiagnosticTracker.h"` → `"arcanum/DiagnosticTracker.h"`
- `"passes/Passes.h"` → `"arcanum/passes/Passes.h"`
- `"frontend/SubsetEnforcer.h"` → `"arcanum/frontend/SubsetEnforcer.h"`
- `"frontend/ContractParser.h"` → `"arcanum/frontend/ContractParser.h"`
- `"backend/WhyMLEmitter.h"` → `"arcanum/backend/WhyMLEmitter.h"`
- `"backend/Why3Runner.h"` → `"arcanum/backend/Why3Runner.h"`
- `"report/ReportGenerator.h"` → `"arcanum/report/ReportGenerator.h"`

In `.td` files, prefix internal TableGen includes:
- `ArcOps.td`: `include "dialect/ArcDialect.td"` → `include "arcanum/dialect/ArcDialect.td"`, `include "dialect/ArcTypes.td"` → `include "arcanum/dialect/ArcTypes.td"`
- `ArcTypes.td`: `include "dialect/ArcDialect.td"` → `include "arcanum/dialect/ArcDialect.td"`

Commit message: `refactor: update #include paths in lib/ sources and .td files to arcanum/ prefix`

---

### Task 4: Update #include directives in tools/arcanum/main.cpp

**Files:**
- Modify: `arcanum/tools/arcanum/main.cpp`

**Agent role:** junior-engineer

Change the first 9 includes:
```cpp
// Before:                              // After:
#include "DiagnosticTracker.h"          #include "arcanum/DiagnosticTracker.h"
#include "ExitCodes.h"                  #include "arcanum/ExitCodes.h"
#include "backend/Why3Runner.h"         #include "arcanum/backend/Why3Runner.h"
#include "backend/WhyMLEmitter.h"       #include "arcanum/backend/WhyMLEmitter.h"
#include "dialect/Lowering.h"           #include "arcanum/dialect/Lowering.h"
#include "frontend/ContractParser.h"    #include "arcanum/frontend/ContractParser.h"
#include "frontend/SubsetEnforcer.h"    #include "arcanum/frontend/SubsetEnforcer.h"
#include "passes/Passes.h"             #include "arcanum/passes/Passes.h"
#include "report/ReportGenerator.h"     #include "arcanum/report/ReportGenerator.h"
```

Commit message: `refactor: update #include paths in tools/arcanum/main.cpp to arcanum/ prefix`

---

### Task 5: Update #include directives in test files

**Files:**
- Modify: `arcanum/tests/unit/SubsetEnforcerTest.cpp`
- Modify: `arcanum/tests/unit/ContractParserTest.cpp`
- Modify: `arcanum/tests/unit/ArcDialectTest.cpp`
- Modify: `arcanum/tests/unit/LoweringTest.cpp`
- Modify: `arcanum/tests/unit/WhyMLEmitterTest.cpp`
- Modify: `arcanum/tests/unit/Why3RunnerTest.cpp`
- Modify: `arcanum/tests/unit/ReportGeneratorTest.cpp`

**Agent role:** junior-engineer

Same pattern: prefix all internal `#include` directives with `arcanum/`. Each test file includes 1-6 internal headers that need updating.

Commit message: `refactor: update #include paths in unit tests to arcanum/ prefix`

---

### Task 6: Rewrite CMakeLists.txt for new layout, TableGen paths, and install/export

**Files:**
- Modify: `arcanum/CMakeLists.txt`
- Modify: `arcanum/tests/CMakeLists.txt`

**Agent role:** senior-engineer

Key changes to root CMakeLists.txt:
1. **Include directories**: `${PROJECT_SOURCE_DIR}/src` → `${PROJECT_SOURCE_DIR}/include`, `${PROJECT_BINARY_DIR}/src` → `${PROJECT_BINARY_DIR}/include`
2. **TableGen output dir**: `file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/arcanum/dialect)`
3. **TableGen paths**: `src/dialect/ArcOps.td` → `include/arcanum/dialect/ArcOps.td`, output paths similarly prefixed
4. **Library source paths**: `src/dialect/*.cpp` → `lib/dialect/*.cpp` (all 6 libraries)
5. **ADDITIONAL_HEADER_DIRS**: `${PROJECT_SOURCE_DIR}/src/dialect` → `${PROJECT_SOURCE_DIR}/include/arcanum/dialect`
6. **Executable source**: `src/main.cpp` → `tools/arcanum/main.cpp`
7. **New install/export block**: install headers, .inc files, library targets, export ArcanumTargets with Arcanum:: namespace, configure_package_config_file

Key changes to tests/CMakeLists.txt:
- All `${PROJECT_SOURCE_DIR}/src` → `${PROJECT_SOURCE_DIR}/include`
- All `${PROJECT_BINARY_DIR}/src` → `${PROJECT_BINARY_DIR}/include`

Commit message: `refactor: update CMakeLists.txt for include/lib/tools layout with install/export`

---

### Task 7: Add cmake/ArcanumConfig.cmake.in

**Files:**
- Create: `arcanum/cmake/ArcanumConfig.cmake.in`

**Agent role:** junior-engineer

Content:
```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

find_dependency(MLIR REQUIRED)
find_dependency(Clang REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/ArcanumTargets.cmake")

check_required_components(Arcanum)
```

Commit message: `feat: add CMake package config template for find_package(Arcanum)`

---

### Task 8: Build and verify all tests pass

**Files:** none (verification only)

**Agent role:** senior-engineer

1. Clean old build: `rm -rf /workspace/ad-adas-memo/arcanum/build`
2. Configure: `cmake --preset default`
3. Build: `cmake --build build/default -j$(nproc)`
4. Unit tests: `ctest --test-dir build/default --output-on-failure`
5. Lit tests: `cmake --build build/default --target check-arcanum-lit`
6. Verify no stale `src/` references in CMake files
7. Verify header tree structure matches design

---

## Execution: Subagent-Driven

**Task Order:** Sequential, dependency-respecting.

1. Task 1: Create directory structure and git-mv all files — no dependencies
2. Task 2: Update #include directives in header files — depends on Task 1
3. Task 3: Update #include directives in lib/ sources and .td files — depends on Task 1
4. Task 4: Update #include directives in tools/arcanum/main.cpp — depends on Task 1
5. Task 5: Update #include directives in test files — depends on Task 1
6. Task 6: Rewrite CMakeLists.txt for new layout — depends on Task 1
7. Task 7: Add cmake/ArcanumConfig.cmake.in — depends on Task 1
8. Task 8: Build and verify all tests pass — depends on Tasks 1-7

Tasks 2-7 are independent of each other and all depend only on Task 1. Task 8 must run last as it validates all preceding work.
