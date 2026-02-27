# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Arcanum is a formal verification tool for safety-critical automotive C++ software (ISO 26262, ASIL C/D). It proves absence of runtime errors (buffer overflow, integer overflow, division by zero, NaN) and user-specified functional properties via contract annotations (`//@` comments). The pipeline: Clang AST → Subset Enforcer → Contract Parser → Arc MLIR Dialect → MLIR Passes → WhyML Emitter → Why3 + SMT solvers → Report. Currently in Slice 1 (initial implementation phase).

## Domain

Automotive ADAS — formal verification, MLIR compiler infrastructure, SMT-based theorem proving, ISO 26262 functional safety.

## Build

All source lives under `arcanum/`. The project requires LLVM 21 with MLIR, Clang 21, GoogleTest, Why3, and SMT solvers (Z3, CVC5, Alt-Ergo). The Docker dev image (`arcanum/docker/`) has everything pre-installed.

```bash
cd arcanum
cmake --preset default          # Configure (debug, system compiler, ccache)
cmake --build build/default     # Build
```

Other presets: `clang-debug`, `clang-release`, `asan`, `ubsan`, `coverage`.

## Test

```bash
ctest --preset default --output-on-failure          # Unit tests (GoogleTest)
cmake --build build/default --target check-arcanum-lit  # LIT tests (dialect/MLIR)
```

## Format

```bash
cd arcanum
./scripts/run-clang-format.sh           # Fix in place
./scripts/run-clang-format.sh --check   # CI check mode
```

Style: LLVM base, left pointer alignment (`int* p` not `int *p`).

## Lint (clang-tidy)

TableGen headers must be generated before running clang-tidy:

```bash
cd arcanum
cmake --preset clang-debug
cmake --build build/clang-debug --target ArcOpsIncGen ArcTypesIncGen
./scripts/run-clang-tidy.sh build/clang-debug       # Check
./scripts/run-clang-tidy.sh --fix build/clang-debug  # Auto-fix
```
