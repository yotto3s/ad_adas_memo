# Arcanum Slice 1: Implementation Design

## Summary

Slice 1 wires all 8 pipeline stages end-to-end for the smallest useful C++ subset: `int32_t`, `bool`, basic arithmetic, `if`/`else`, `requires`/`ensures`/`\result` annotations. It proves overflow safety and postcondition correctness.

This document covers the Docker environment, project scaffold, stage-by-stage implementation plan, and testing strategy.

## Approach

**Stage-by-Stage Wiring**: Build each of the 8 pipeline stages sequentially in data-flow order. Each stage gets unit tests before moving to the next. End-to-end lit tests added once all stages connect.

## Docker Environment

Two-layer Dockerfile approach (modeled after yotto3s/polang):

### Dockerfile.base (Ubuntu 24.04)

- LLVM 21 via `llvm.sh 21 all` (Clang 21, MLIR 21 dev packages)
- `libmlir-21-dev`, `mlir-21-tools` for MLIR development headers and tools
- Z3 via apt (`z3`)
- Why3 via opam (`opam install why3`)
- Build essentials: cmake, gcc, g++, python3 (for lit)
- `update-alternatives` for clang-21/clang++-21 defaults
- LD config for `/usr/lib/llvm-21/lib`

### Dockerfile.dev (extends base)

- Development tools: gosu, Claude Code, gh CLI, ripgrep, valgrind, gdb, nano
- Entrypoint with UID/GID matching for host-container file ownership

### Helper Scripts (`arcanum/docker/`)

| Script | Purpose |
|--------|---------|
| `docker_config.sh` | Image names, container name, paths |
| `docker_build.sh` | Pull base + build dev image |
| `docker_run.sh` | Mount project, .claude, .ssh, git config, credentials |
| `entrypoint.sh` | UID/GID matching |

## Project Structure

```
arcanum/
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.dev
│   ├── docker_config.sh
│   ├── docker_build.sh
│   ├── docker_run.sh
│   └── entrypoint.sh
├── docs/
│   ├── arcanum-tool-spec.md
│   └── arcanum-safe-cpp-subset.md
├── src/
│   ├── main.cpp
│   ├── frontend/
│   │   ├── SubsetEnforcer.h
│   │   ├── SubsetEnforcer.cpp
│   │   ├── ContractParser.h
│   │   └── ContractParser.cpp
│   ├── dialect/
│   │   ├── ArcDialect.td
│   │   ├── ArcDialect.h
│   │   ├── ArcDialect.cpp
│   │   ├── ArcOps.td
│   │   ├── ArcOps.h
│   │   ├── ArcOps.cpp
│   │   ├── ArcTypes.td
│   │   ├── ArcTypes.h
│   │   ├── ArcTypes.cpp
│   │   ├── Lowering.h
│   │   └── Lowering.cpp
│   ├── passes/
│   │   ├── Passes.h
│   │   └── Passes.cpp
│   ├── backend/
│   │   ├── WhyMLEmitter.h
│   │   ├── WhyMLEmitter.cpp
│   │   ├── Why3Runner.h
│   │   └── Why3Runner.cpp
│   └── report/
│       ├── ReportGenerator.h
│       └── ReportGenerator.cpp
├── include/arcanum/
├── tests/
│   ├── unit/
│   │   ├── SubsetEnforcerTest.cpp
│   │   ├── ContractParserTest.cpp
│   │   ├── ArcDialectTest.cpp
│   │   ├── LoweringTest.cpp
│   │   ├── WhyMLEmitterTest.cpp
│   │   ├── Why3RunnerTest.cpp
│   │   └── ReportGeneratorTest.cpp
│   └── lit/
│       ├── lit.cfg.py
│       ├── subset-check/
│       │   ├── reject-virtual.cpp
│       │   └── reject-raw-ptr.cpp
│       └── verify/
│           ├── pass-simple-add.cpp
│           └── fail-overflow.cpp
├── cmake/modules/
├── CMakeLists.txt
├── CMakePresets.json
├── .clang-format
├── .clang-tidy
├── .gitignore
└── README.md
```

## CMake Configuration

### Root CMakeLists.txt

- `cmake_minimum_required(VERSION 3.20)`
- Project: `Arcanum`, languages C/CXX, C++23 standard
- `find_package(MLIR REQUIRED CONFIG)` (pulls in LLVM)
- `find_package(Clang REQUIRED CONFIG)` for LibTooling
- TableGen targets for Arc dialect (`ArcDialect.td`, `ArcOps.td`, `ArcTypes.td`)
- GoogleTest via `FetchContent`
- LLVM lit via `add_lit_testsuite` or custom target
- Warning flags: `-Wall -Wextra -Werror`
- LLVM/MLIR include dirs scoped per-target (not global) to avoid header conflicts with GoogleTest

### CMakePresets.json

Base preset sets `CMAKE_PREFIX_PATH=/usr/lib/llvm-21` and `CMAKE_EXPORT_COMPILE_COMMANDS=ON`.

| Preset | Compiler | Build Type |
|--------|----------|------------|
| `default` | System default | Debug |
| `clang-debug` | clang-21 | Debug |
| `clang-release` | clang-21 | Release |
| `asan` | clang-21 | Debug + AddressSanitizer |
| `ubsan` | clang-21 | Debug + UBSanitizer |

### Code Style

- `.clang-format`: LLVM base style
- `.clang-tidy`: Adapted from polang; LLVM naming conventions (CamelCase types, camelBack functions/variables)

## Pipeline Stages (Slice 1)

### Stage 0: CLI Skeleton (`src/main.cpp`)

- LLVM `cl::opt` for argument parsing
- Input: source file path, `--mode=verify` (only mode)
- Validates input file exists
- Orchestrates the pipeline: calls each stage sequentially, passes outputs forward
- Exit code: 0 if all pass, 1 otherwise

### Stage 1: Clang Frontend

**Files:** Integrated into CLI (uses `clang::tooling::ClangTool`)

- Custom `FrontendAction` that retains the `ASTContext`
- Enables `-fparse-all-comments` for `//@ ` annotation retention
- Fails with Clang diagnostics on syntax errors

### Stage 2: Subset Enforcer (`frontend/SubsetEnforcer`)

**Input:** `clang::ASTContext`
**Output:** `vector<Diagnostic>`, bool pass/fail

`clang::RecursiveASTVisitor` that walks the AST.

Allowed constructs (Slice 1):
- Types: `int32_t`, `bool`
- Functions: non-template, non-recursive, single file, single return
- Statements: variable declarations, assignments, `if`/`else`, `return`
- Expressions: `+`, `-`, `*`, `/`, `%`, comparisons, `&&`, `||`, `!`

Everything else is rejected with a source-located diagnostic.

### Stage 3: Contract Parser (`frontend/ContractParser`)

**Input:** `clang::ASTContext`
**Output:** `map<FunctionDecl*, ContractInfo>`

- Scans raw comment list for `//@ requires:` and `//@ ensures:` prefixes
- Associates contracts with the `FunctionDecl` they immediately precede
- Parses contract expressions into a simple expression AST
- Supported: comparisons, `&&`, `||`, `!`, `\result`, integer literals, parameter names

### Stage 4: Arc MLIR Lowering (`dialect/Lowering`)

**Input:** `clang::ASTContext` + `map<FunctionDecl*, ContractInfo>`
**Output:** `mlir::ModuleOp` (Arc dialect)

Walks annotated Clang AST, emits Arc dialect operations:
- `arc.func` with `requires`/`ensures` attributes
- `arc.add`, `arc.sub`, `arc.mul`, `arc.div`, `arc.rem` for arithmetic
- `arc.cmp` for comparisons
- `arc.if` / `arc.else` for conditionals
- `arc.return` for return
- `arc.var` / `arc.assign` for locals
- `mlir::FileLineColLoc` location metadata from Clang source locations

### Stage 5: MLIR Pass Manager (`passes/Passes`)

**Input:** `mlir::ModuleOp`
**Output:** `mlir::ModuleOp` (verified)

- Identity pass-through for Slice 1 (no optimization passes)
- Runs MLIR verifier to catch malformed IR
- Sets up `PassManager` infrastructure for future slices

### Stage 6: WhyML Emitter (`backend/WhyMLEmitter`)

**Input:** `mlir::ModuleOp` (Arc dialect)
**Output:** WhyML source text, written to temp `.mlw` file

Translation rules:
- `arc.func` -> `let <name> (<params>) : int` with `requires`/`ensures` clauses
- Arithmetic ops -> WhyML arithmetic + overflow assertions (`-2147483648 <= result <= 2147483647`)
- Comparisons/logical ops -> WhyML equivalents
- `if`/`else` -> `if then else`
- Wraps in `module <FuncName> ... end`, adds `use int.Int`

### Stage 7: Why3 Runner (`backend/Why3Runner`)

**Input:** `.mlw` file path
**Output:** `vector<ObligationResult>`

- Validates `why3` in `$PATH` (or `--why3-path` flag)
- Spawns `why3 prove -P z3 <file.mlw>`
- Parses stdout for per-obligation results: `Valid` / `Unknown` / `Timeout`
- Configurable timeout (default 30s per obligation)

### Stage 8: Report Generator (`report/ReportGenerator`)

**Input:** `vector<ObligationResult>` + source location mapping
**Output:** Formatted terminal text

Format:
```
[PASS]  input.cpp:safe_add    2/2 obligations proven (0.3s)

Summary: 1 passed, 0 failed, 0 timeout
```

## Testing Strategy

### Unit Tests (GoogleTest)

| Test File | Coverage |
|-----------|----------|
| `SubsetEnforcerTest.cpp` | Accepted/rejected constructs |
| `ContractParserTest.cpp` | Valid/malformed annotation parsing |
| `ArcDialectTest.cpp` | Dialect ops creation, verification, round-trip |
| `LoweringTest.cpp` | Clang AST to Arc MLIR for known inputs |
| `WhyMLEmitterTest.cpp` | Arc MLIR to WhyML text comparison |
| `Why3RunnerTest.cpp` | Output parsing (mock Why3 stdout) |
| `ReportGeneratorTest.cpp` | Formatting verification |

GoogleTest fetched via CMake `FetchContent`.

### Integration Tests (LLVM lit + FileCheck)

**`subset-check/`** - validates the subset enforcer rejects forbidden constructs:
```
// RUN: arcanum --mode=verify %s 2>&1 | FileCheck %s
// CHECK: error: virtual functions are not allowed
```

**`verify/`** - end-to-end verification:
```
// RUN: arcanum --mode=verify %s | FileCheck %s
// CHECK: [PASS] {{.*}}safe_add{{.*}}2/2 obligations proven
```

### CI (GitHub Actions)

Docker-based workflow:
1. Build the dev Docker image
2. `cmake --preset default && cmake --build build/default`
3. `ctest --preset default`
4. Run lit tests

## Slice 1 End-to-End Example

**Input** (`safe_add.cpp`):
```cpp
#include <cstdint>

//@ requires: a >= 0 && a <= 1000
//@ requires: b >= 0 && b <= 1000
//@ ensures: \result >= 0 && \result <= 2000
int32_t safe_add(int32_t a, int32_t b) {
    return a + b;
}
```

**Arc MLIR** (intermediate):
```mlir
arc.func @safe_add(%a: !arc.i32, %b: !arc.i32) -> !arc.i32
    attrs {
      requires = arc.and(arc.cmp(gte, %a, 0), arc.cmp(lte, %a, 1000),
                         arc.cmp(gte, %b, 0), arc.cmp(lte, %b, 1000)),
      ensures = arc.and(arc.cmp(gte, %result, 0), arc.cmp(lte, %result, 2000))
    }
{
  %result = arc.add %a, %b : !arc.i32
  arc.return %result
}
```

**WhyML** (emitted):
```whyml
module SafeAdd
  use int.Int

  let safe_add (a: int) (b: int) : int
    requires { 0 <= a <= 1000 }
    requires { 0 <= b <= 1000 }
    ensures  { 0 <= result <= 2000 }
  = a + b
end
```

**Output**:
```
[PASS]  safe_add.cpp:safe_add    2/2 obligations proven (0.3s)

Summary: 1 passed, 0 failed, 0 timeout
```

## Implementation Order

| Step | What | Deliverable |
|------|------|-------------|
| 1 | Docker environment | Working `docker_build.sh` + `docker_run.sh`, LLVM 21 + MLIR + Why3 + Z3 verified |
| 2 | CMake scaffold + CLI skeleton | `arcanum` binary that parses args and exits |
| 3 | Stage 1: Clang frontend | Parses .cpp, produces ASTContext |
| 4 | Stage 2: Subset enforcer | Rejects forbidden constructs with diagnostics |
| 5 | Stage 3: Contract parser | Extracts requires/ensures from //@ comments |
| 6 | Arc dialect definition (TableGen) | ArcDialect.td, ArcOps.td, ArcTypes.td compile |
| 7 | Stage 4: Arc MLIR lowering | Clang AST + contracts -> Arc MLIR |
| 8 | Stage 5: MLIR pass manager | Identity pass-through + verifier |
| 9 | Stage 6: WhyML emitter | Arc MLIR -> .mlw file |
| 10 | Stage 7: Why3 runner | Invoke Why3, parse results |
| 11 | Stage 8: Report generator | Format and print results |
| 12 | End-to-end integration | lit tests pass, safe_add example works |
| 13 | CI workflow | GitHub Actions runs all tests in Docker |
