# Arcanum Implementation Design

## Summary

Arcanum is a formal verification tool for a safe C++ subset targeting safety-critical automotive software (ISO 26262). It proves mathematical correctness properties of C++ programs using modular, per-function deductive verification with SMT solvers.

This document defines the implementation design: technology choices, project structure, pipeline architecture, and the incremental delivery plan.

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Implementation language | C++23 | Natural fit for Clang LibTooling and MLIR APIs; std::expected, std::format, deducing this |
| Build system | CMake | Standard for LLVM/Clang/MLIR ecosystem |
| LLVM/Clang/MLIR version | 21.x (stable) | Latest stable release with mature MLIR support |
| Why3 integration | External process | Simplest integration; Why3 CLI invoked at runtime. Avoids OCaml build dependency. Spec's WhyML-centric design makes this a natural fit |
| SMT solver | Z3 (default), CVC5/Alt-Ergo via Why3 | Why3 handles multi-prover dispatch |
| Unit testing | GoogleTest (FetchContent) | Industry standard for C++ |
| Integration testing | LLVM lit + FileCheck | Standard for compiler/tool testing; test .cpp inputs against expected outputs |
| Target platform | Linux only | Simplifies build and CI |
| Repository | ad-adas-memo (this repo), under `arcanum/` | Spec and implementation co-located |

## Implementation Strategy: Vertical Slice

Rather than building each pipeline stage to completion sequentially, we wire up all 8 stages end-to-end for a minimal subset first (Slice 1), then widen incrementally.

**Why vertical slice:**
- Validates the full architecture early, before heavy investment in any one stage
- Every new feature is testable end-to-end immediately
- Catches interface mismatches between stages before they become expensive
- The spec's "near 1:1 Arc-to-WhyML translation" claim is validated from the start

## Project Structure

```
ad-adas-memo/
├── arcanum/
│   ├── docs/                           # Specification documents
│   │   ├── arcanum-tool-spec.md
│   │   └── arcanum-safe-cpp-subset.md
│   ├── src/
│   │   ├── main.cpp                    # CLI entry point
│   │   ├── frontend/
│   │   │   ├── SubsetEnforcer.h/cpp    # Clang AST walker for subset checks
│   │   │   └── ContractParser.h/cpp    # //@ annotation parser
│   │   ├── dialect/
│   │   │   ├── ArcDialect.td           # TableGen dialect definition
│   │   │   ├── ArcDialect.h/cpp        # MLIR dialect registration
│   │   │   ├── ArcOps.td               # TableGen op definitions
│   │   │   ├── ArcOps.h/cpp            # Generated + custom op code
│   │   │   ├── ArcTypes.td             # TableGen type definitions
│   │   │   ├── ArcTypes.h/cpp          # Dialect type code
│   │   │   └── Lowering.h/cpp          # Clang AST -> Arc MLIR lowering
│   │   ├── passes/
│   │   │   └── Passes.h/cpp            # MLIR optimization passes
│   │   ├── backend/
│   │   │   ├── WhyMLEmitter.h/cpp      # Arc MLIR -> WhyML text
│   │   │   └── Why3Runner.h/cpp        # Why3 CLI invocation + result parsing
│   │   └── report/
│   │       └── ReportGenerator.h/cpp   # Terminal + JSON output formatting
│   ├── include/
│   │   └── arcanum/                    # Public headers if needed
│   ├── tests/
│   │   ├── unit/                       # GoogleTest
│   │   │   ├── SubsetEnforcerTest.cpp
│   │   │   ├── ContractParserTest.cpp
│   │   │   ├── ArcDialectTest.cpp
│   │   │   └── WhyMLEmitterTest.cpp
│   │   └── lit/                        # LLVM lit integration tests
│   │       ├── lit.cfg.py
│   │       ├── subset-check/
│   │       │   ├── reject-virtual.cpp
│   │       │   └── reject-raw-ptr.cpp
│   │       └── verify/
│   │           ├── pass-simple-add.cpp
│   │           └── fail-overflow.cpp
│   ├── cmake/
│   │   └── modules/                    # CMake helper modules
│   ├── CMakeLists.txt                  # Arcanum CMake root
│   └── README.md                       # Build instructions
├── plans/                              # Design documents
└── README.md                           # Repo-level README
```

## Pipeline Architecture

The tool is an 8-stage pipeline. Each stage has a well-defined input and output, and is testable in isolation.

```
CLI (main.cpp)
  │
  ▼
Stage 1: Clang Frontend (LibTooling)
  Input:  Source file path
  Output: clang::ASTContext
  Impl:   clang::tooling::ClangTool with custom FrontendAction
  │
  ▼
Stage 2: Subset Enforcer
  Input:  clang::ASTContext
  Output: vector<Diagnostic>, bool pass/fail
  Impl:   clang::RecursiveASTVisitor that rejects disallowed constructs
  │
  ▼
Stage 3: Contract Parser
  Input:  clang::ASTContext
  Output: map<FunctionDecl*, ContractInfo>
  Impl:   Scans ASTContext RawComment list for //@ prefix, parses expressions
  │
  ▼
Stage 4: Arc MLIR Lowering
  Input:  clang::ASTContext + ContractInfo map
  Output: mlir::ModuleOp (Arc dialect)
  Impl:   Walks annotated AST, creates Arc dialect operations
  │
  ▼
Stage 5: MLIR Pass Manager
  Input:  mlir::ModuleOp
  Output: mlir::ModuleOp (optimized)
  Impl:   Standard MLIR pass infrastructure. Slice 1: identity pass-through
  │
  ▼
Stage 6: WhyML Emitter
  Input:  mlir::ModuleOp (Arc dialect)
  Output: string (WhyML source text), written to temp .mlw file
  Impl:   Walks Arc ops, emits corresponding WhyML constructs
  │
  ▼
Stage 7: Why3 Runner
  Input:  .mlw file path
  Output: vector<ObligationResult>
  Impl:   Spawns why3 prove -P z3, parses stdout for PASS/FAIL/TIMEOUT
  │
  ▼
Stage 8: Report Generator
  Input:  vector<ObligationResult> + source location mapping
  Output: Formatted text (terminal or JSON)
  Impl:   Maps results back to source locations via MLIR location metadata
```

## Dependency Management

### Compile-time Dependencies

| Dependency | Version | Acquisition |
|-----------|---------|-------------|
| LLVM | 21.x | `find_package(LLVM 21 REQUIRED CONFIG)` — system install |
| Clang | 21.x | `find_package(Clang REQUIRED CONFIG)` — bundled with LLVM |
| MLIR | 21.x | `find_package(MLIR REQUIRED CONFIG)` — bundled with LLVM |
| GoogleTest | latest | CMake `FetchContent` |

### Runtime Dependencies

| Dependency | Version | Notes |
|-----------|---------|-------|
| Why3 | latest | Searched in `$PATH` or via `--why3-path` flag |
| Z3 | 4.x+ | Invoked by Why3, not directly by Arcanum |

Why3 and Z3 are runtime-only. The `Why3Runner` validates their presence at startup and fails with a clear error message if not found.

### Arc MLIR Dialect (TableGen)

The Arc dialect uses MLIR TableGen for op and type definitions. TableGen generates C++ boilerplate for:
- Op class definitions with builders and verifiers
- Type storage and parsing
- Canonicalization patterns

## Slice 1: Minimal End-to-End Feature Set

Slice 1 is the first milestone. It wires up all 8 pipeline stages for the smallest useful subset.

### Supported C++

- **Types:** `int32_t`, `bool`
- **Functions:** Non-template, non-recursive, single file, single return
- **Statements:** Variable declarations, assignments, `if`/`else`, `return`
- **Expressions:** `+`, `-`, `*`, `/`, `%`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `&&`, `||`, `!`

### Supported Annotations

- `//@ requires: <expr>` — precondition
- `//@ ensures: <expr>` — postcondition
- `\result` — return value reference in ensures

### Supported Modes

- `--mode=verify` only

### What Slice 1 Proves

- **Overflow safety** (trap mode): Every arithmetic operation's result fits in `int32_t`
- **Postcondition correctness:** `ensures` clause holds for all inputs satisfying `requires`

### Example: Slice 1 End-to-End

```cpp
//@ requires: a >= 0 && a <= 1000
//@ requires: b >= 0 && b <= 1000
//@ ensures: \result >= 0 && \result <= 2000
int32_t safe_add(int32_t a, int32_t b) {
    return a + b;
}
```

Expected output:
```
[PASS]  input.cpp:safe_add    2/2 obligations proven (0.3s)

Summary: 1 passed, 0 failed, 0 timeout
```

### Arc Dialect for this Example

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

### WhyML for this Example

```whyml
module SafeAdd
  use int.Int
  use int.ComputerDivision

  let safe_add (a: int) (b: int) : int
    requires { 0 <= a <= 1000 }
    requires { 0 <= b <= 1000 }
    ensures  { 0 <= result <= 2000 }
  = a + b
end
```

## Incremental Widening Plan

After Slice 1, features are added incrementally. Each slice extends the pipeline without redesigning it.

| Slice | New Features | Primary Stages Affected |
|-------|-------------|------------------------|
| 1 | `int32_t`, `bool`, `if/else`, arithmetic, `requires`/`ensures`, `\result` | All (initial wiring) |
| 2 | All integer types (`i8`-`i64`, `u8`-`u64`), `static_cast`, overflow modes | Dialect types, WhyML emitter, Subset Enforcer |
| 3 | `for`/`while` loops, `loop_invariant`, `loop_variant`, `break`/`continue` | Lowering, Dialect ops, WhyML emitter |
| 4 | Function calls (non-recursive), modular verification with callee contracts | Lowering, WhyML emitter |
| 5 | `std::array<T,N>`, `std::span<T>`, bounds checking, `\forall`/`\exists` | Dialect types/ops, WhyML emitter, Contract Parser |
| 6 | `struct`/`class`, field access, constructors | Dialect types, Lowering |
| 7 | `std::expected<T,E>`, `std::optional<T>` | Dialect types, Lowering |
| 8 | `//@ predicate`, `//@ logic`, ghost state | Contract Parser, Dialect ops, WhyML emitter |
| 9 | `//@ pure` functions, contract chaining | WhyML emitter |
| 10 | Templates (per-instantiation), `enum class`, `std::variant`/`std::visit` | Lowering, Subset Enforcer |
| 11 | Totality mode (`//@ total`, `//@ decreases`, exhaustiveness) | New mode, Dialect, WhyML |
| 12 | Floating-point (real arithmetic + `fp_safe`) | Dialect types, WhyML emitter |
| 13 | MLIR optimization passes | Passes |
| 14 | Multi-file verification (header contracts, `//@ trusted`) | Frontend, CLI |
| 15 | Full CLI, JSON output, trust report, `.arcanum.yaml` config | CLI, Report Generator |
