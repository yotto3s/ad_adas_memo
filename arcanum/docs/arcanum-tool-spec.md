# Arcanum — Tool Specification

## 1. Overview

Arcanum is a standalone formal verification tool for a restricted subset of C++ designed for safety-critical automotive software. It proves mathematical correctness properties of C++ programs — including absence of runtime errors, functional correctness, and totality — using modular, per-function deductive verification with SMT solvers.

The tool is inspired by Frama-C (for C) and extends its approach to a defined subset of C++ with contracts, ghost state, and totality checking.

### 1.1 Goals

- Prove absence of runtime errors (buffer overflow, integer overflow, division by zero, NaN propagation)
- Prove user-specified functional properties (pre/postconditions, invariants)
- Prove function totality (termination + coverage) when annotated
- Support incremental adoption in existing automotive C++ codebases
- Produce clear, actionable diagnostics when verification fails

### 1.2 Non-Goals (Version 1)

- Whole-program analysis
- Concurrent program verification (beyond restricted single-writer patterns)
- IEEE 754 strict floating-point verification
- Automatic invariant inference
- Code generation or compilation

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      arcanum CLI                         │
│                                                          │
│  Input: C++ source files (.cpp/.hpp) with annotations    │
│  Output: Verification report (PASS / FAIL / TIMEOUT)     │
└─────────────────────┬────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  1. Clang Frontend    │
          │     (LibTooling)      │
          │                       │
          │  - Parse C++ source   │
          │  - Resolve templates  │
          │  - Type checking      │
          └───────────┬───────────┘
                      │ Clang AST
          ┌───────────▼───────────┐
          │  2. Subset Enforcer   │
          │                       │
          │  - Walk AST           │
          │  - Reject disallowed  │
          │    constructs         │
          │  - Report violations  │
          └───────────┬───────────┘
                      │ Validated AST
          ┌───────────▼───────────┐
          │  3. Contract Parser   │
          │                       │
          │  - Parse //@ comments │
          │  - Build spec AST     │
          │  - Attach specs to    │
          │    function/loop nodes│
          └───────────┬───────────┘
                      │ Annotated AST
          ┌───────────▼───────────┐
          │  4. Arc Dialect       │
          │     Lowering          │
          │                       │
          │  - Translate Clang    │
          │    AST to Arc MLIR    │
          │  - Desugar all C++    │
          │    into ~30 ops       │
          │  - Attach contracts   │
          │    as MLIR attributes │
          └───────────┬───────────┘
                      │ Arc MLIR dialect
          ┌───────────▼───────────┐
          │  5. MLIR Passes       │
          │                       │
          │  - Contract propagation│
          │  - Overflow check     │
          │    simplification     │
          │  - Loop normalization │
          │  - Dead spec removal  │
          └───────────┬───────────┘
                      │ Optimized Arc MLIR
          ┌───────────▼───────────┐
          │  6. WhyML Emitter     │
          │                       │
          │  - Near 1:1 translation│
          │    from Arc ops to    │
          │    WhyML constructs   │
          └───────────┬───────────┘
                      │ WhyML (.mlw)
          ┌───────────▼───────────┐
          │  7. Why3 Backend      │
          │                       │
          │  - VC generation      │
          │  - SMT dispatch       │
          │    (Z3/CVC5/Alt-Ergo) │
          │  - Counterexample     │
          │    extraction         │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  8. Report Generator  │
          │                       │
          │  - Map results back   │
          │    to source via MLIR │
          │    location tracking  │
          │  - Format diagnostics │
          └───────────────────────┘
```

### 2.1 Design Decisions

**Clang as frontend:** Clang provides a production-quality C++ parser, template instantiation, and type checking. The tool uses LibTooling to walk the Clang AST rather than implementing a custom parser.

**Arc MLIR dialect as intermediate representation:** Rather than translating directly from Clang AST to WhyML, the tool lowers into a custom MLIR dialect (`arc`). This provides a clean abstraction boundary — the dialect has ~30 operations with well-defined verification semantics, insulating the WhyML emitter from C++ complexity. MLIR's pass infrastructure enables optimization of verification obligations before emitting WhyML. MLIR's location tracking provides precise source mapping for error reporting.

**WhyML as verification backend:** The Arc dialect is translated to WhyML (the input language of Why3), reusing Why3's mature weakest precondition calculus and multi-prover support. The translation from Arc to WhyML is nearly 1:1.

**Multiple backend support:** The Arc dialect enables future backends without modifying the frontend:

```
                    ┌→ WhyML → Why3 → SMT solvers (primary)
Arc MLIR dialect ───┼→ SMT-LIB → Z3 directly (future: simple cases)
                    ├→ Boogie → Z3/CVC5 (future: alternative backend)
                    └→ Pretty printer → annotated C++ report
```

**Modular verification:** Each function is verified independently against its contract. Callees are represented only by their contracts, not their implementations. This scales to large codebases and enables parallel verification.

### 2.2 Template Verification

Templates are verified **per-instantiation**. The verifier works on the fully instantiated Clang AST where all type parameters and non-type parameters are concrete. Contracts are written on the template definition and apply to every instantiation:

```cpp
template<typename T, size_t N>
//@ requires: N > 0
//@ ensures: \forall size_t i; 0 <= i < N ==> \result >= data[i]
T find_max(const std::array<T, N>& data) { ... }

// Each instantiation is verified separately:
auto m1 = find_max<int, 10>(int_arr);     // verified with T=int, N=10
auto m2 = find_max<float, 256>(float_arr); // verified with T=float, N=256
```

**Implications:**
- If a template is instantiated with 5 different type/size combinations, it is verified 5 times
- If a template is never instantiated, it is never verified
- Contracts may use operations on `T` (like `>=`) — these are resolved concretely per instantiation
- Generic verification using Concepts as type-level contracts is a future direction

### 2.3 Multi-File Verification and Contract Visibility

Contracts live in **header files**, attached to function declarations. This enables modular verification across translation units.

```cpp
// sensor.hpp — declaration with contract (authoritative specification)
//@ requires: raw >= 0 && raw <= 4095
//@ ensures: \result >= 0 && \result <= 1000
//@ assigns: \nothing
int scale_reading(int raw);
```

```cpp
// sensor.cpp — definition verified against header contract
#include "sensor.hpp"
int scale_reading(int raw) {
    return raw * 1000 / 4095;
}
```

```cpp
// control.cpp — call site verified against header contract
#include "sensor.hpp"
void process() {
    int raw = read_adc();
    //@ assert: raw >= 0 && raw <= 4095
    int scaled = scale_reading(raw);  // precondition checked here
}
```

**Rules:**
- Contracts on declarations (in headers) are the **authoritative specification**
- If a definition also has contracts, they must be identical to the declaration's contracts (the tool checks consistency)
- When verifying a call site, only the callee's declaration contract is used (modular verification)
- Predicates, logic functions, and ghost declarations can also be defined in headers and shared across files
- Contract-only headers (with `//@ trusted`) can be used for external libraries and hardware abstraction layers without access to the source

### 2.4 The Arc MLIR Dialect

The Arc dialect is the core intermediate representation of Arcanum. It captures the verification-relevant semantics of the Safe C++ Subset in a small, well-defined set of MLIR operations. All C++ surface complexity is resolved during lowering from Clang AST to Arc MLIR; the WhyML emitter only needs to handle the dialect's operations.

#### 2.4.1 Dialect Types

```
!arc.i8, !arc.i16, !arc.i32, !arc.i64       // signed fixed-width integers
!arc.u8, !arc.u16, !arc.u32, !arc.u64       // unsigned fixed-width integers
!arc.f32, !arc.f64                            // floating-point
!arc.bool                                     // boolean
!arc.index                                    // size/index type
!arc.span<T>               // dynamic-extent view (carries logical size)
!arc.span<T, N>            // static-extent view
!arc.array<T, N>           // fixed-size array
!arc.struct<name, {fields}>// named record type
!arc.enum<name, {variants}>// enum class
!arc.expected<T, E>        // result type
!arc.optional<T>           // optional type
!arc.ghost<T>              // ghost-only type (erased at runtime)
```

#### 2.4.2 Core Operations

**Arithmetic (verification-aware):**

| Operation | Description | Proof Obligation |
|-----------|-------------|------------------|
| `arc.add`, `arc.sub`, `arc.mul` | Overflow-checked (trap mode) | Result fits in type |
| `arc.add_wrap`, `arc.sub_wrap`, `arc.mul_wrap` | Wrapping arithmetic | None |
| `arc.add_sat`, `arc.sub_sat`, `arc.mul_sat` | Saturating arithmetic | None |
| `arc.div`, `arc.rem` | Division | Divisor ≠ 0 |
| `arc.shl`, `arc.shr` | Shift | Shift amount in [0, bit_width) |
| `arc.cast` | Explicit type conversion | Target range (in trap mode) |

**Memory access:**

| Operation | Description | Proof Obligation |
|-----------|-------------|------------------|
| `arc.span_get` | Read from span | Index < size |
| `arc.span_set` | Write to span | Index < size |
| `arc.array_get` | Read from array | Index < N (often static) |
| `arc.array_set` | Write to array | Index < N (often static) |
| `arc.struct_get` | Read struct field | None |
| `arc.struct_set` | Write struct field | None |

**Control flow:**

| Operation | Description |
|-----------|-------------|
| `arc.func` | Function definition with contract attributes |
| `arc.call` | Function call (generates callee precondition obligation) |
| `arc.return` | Function return |
| `arc.if` / `arc.else` | Conditional |
| `arc.for` | For loop with invariant and variant attributes |
| `arc.while` | While loop with invariant and variant attributes |
| `arc.switch` | Switch with exhaustiveness metadata |
| `arc.yield` | Yield value from loop body (for iter_args) |

**Specification:**

| Operation | Description |
|-----------|-------------|
| `arc.requires` | Precondition |
| `arc.ensures` | Postcondition |
| `arc.assigns` | Frame condition |
| `arc.invariant` | Loop invariant |
| `arc.variant` | Loop decreasing measure |
| `arc.assert` | Inline assertion (must be proven) |
| `arc.assume` | Inline assumption (taken as axiom) |
| `arc.old` | Value at function entry |
| `arc.at` | Value at labeled program point |
| `arc.label` | Named program point |
| `arc.forall` | Bounded universal quantifier |
| `arc.exists` | Bounded existential quantifier |

**Specification functions:**

| Operation | Description |
|-----------|-------------|
| `arc.predicate` | Named boolean specification expression |
| `arc.logic` | Specification-only function (may be recursive) |
| `arc.pure_call` | Call to a pure function (uses contract, not body) |
| `arc.ghost_read` | Read ghost variable |
| `arc.ghost_write` | Write ghost variable |

#### 2.4.3 Example: Arc Dialect Textual Form

```mlir
// Predicate definition
arc.predicate @valid_adc(%x: !arc.i32) -> !arc.bool {
  %lo = arc.cmp gte %x, %c0 : !arc.i32
  %hi = arc.cmp lte %x, %c4095 : !arc.i32
  %r = arc.and %lo, %hi : !arc.bool
  arc.return %r
}

// Verified function
arc.func @scale_reading(%raw: !arc.i32) -> !arc.i32
    attrs {
      requires = arc.pred_call @valid_adc(%raw),
      ensures = arc.and(arc.cmp gte %result, %c0,
                        arc.cmp lte %result, %c1000),
      assigns = arc.nothing,
      total = true
    }
{
  %c1000 = arc.const 1000 : !arc.i32
  %c4095 = arc.const 4095 : !arc.i32
  %mul = arc.mul %raw, %c1000 : !arc.i32       // overflow obligation
  %div = arc.div %mul, %c4095 : !arc.i32       // div-by-zero obligation
  arc.return %div
}

// Function with loop
arc.func @find_max(%data: !arc.span<!arc.i32>, %n: !arc.index) -> !arc.i32
    attrs {
      requires = arc.and(arc.cmp gt %n, %c0,
                         arc.cmp lte %n, arc.span_size(%data)),
      ensures = arc.forall %i : !arc.index,
                  arc.range(%i, %c0, %n),
                  arc.cmp gte %result, arc.span_get(%data, %i)
    }
{
  %init = arc.span_get %data, %c0 : !arc.i32
  %result = arc.for %i = %c1 to %n
      iter_args(%max = %init)
      attrs {
        invariant = arc.forall %j : !arc.index,
                      arc.range(%j, %c0, %i),
                      arc.cmp gte %max, arc.span_get(%data, %j),
        variant = arc.sub %n, %i
      }
  {
    %val = arc.span_get %data, %i : !arc.i32
    %cmp = arc.cmp gt %val, %max : !arc.bool
    %new = arc.select %cmp, %val, %max : !arc.i32
    arc.yield %new
  }
  arc.return %result
}
```

#### 2.4.4 Translation to WhyML

The Arc-to-WhyML translation is nearly mechanical:

| Arc Dialect | WhyML |
|------------|-------|
| `arc.func @name(...) attrs {requires=P, ensures=Q}` | `let name (...) requires {P} ensures {Q} = ...` |
| `arc.for %i = lo to hi attrs {invariant=I}` | `for i = lo to hi do invariant {I} ... done` |
| `arc.forall %i, range(%i, lo, hi), P(i)` | `forall i: int. lo <= i < hi -> P(i)` |
| `arc.span_get %a, %i` | `a[i]` (Why3 array/map access) |
| `arc.mul %a, %b` (trap mode) | `a * b` + separate VC: `min_int <= a*b <= max_int` |
| `arc.predicate @name(...)` | `predicate name (...) = ...` |
| `arc.logic @name(...)` | `function name (...) = ...` |
| `arc.ghost_write @var %val` | `ghost var := val` |
| `arc.old(%expr)` | `old expr` |
| `arc.at(%expr, L)` | `expr at L` |

#### 2.4.5 MLIR Passes

The following optimization passes run on the Arc dialect before WhyML emission:

**Contract propagation:** Inline simple predicates at call sites to give the SMT solver more direct information.

**Overflow check simplification:** Prove trivial overflow checks (e.g., `uint8 + uint8` always fits in `uint16`) at the MLIR level and remove the corresponding proof obligations.

**Loop normalization:** Canonicalize loop forms (e.g., normalize `while` with counter to `for`) to enable automatic variant inference.

**Dead specification removal:** Remove ghost state and specification constructs that are not referenced by any proof obligation.

**Constant propagation:** Evaluate `arc.const` expressions and simplify arithmetic on known values.

---

## 3. Operating Modes

The tool supports three operating modes, enabling incremental adoption.

### 3.1 Mode: `subset-check`

**Purpose:** Enforce the Safe C++ Subset without requiring any annotations.

**Usage:**
```bash
arcanum --mode=subset-check src/sensor.cpp
```

**Behavior:**
- Parses the source file
- Rejects any constructs outside the allowed subset (see Safe C++ Subset document)
- Reports violations with source locations and suggested replacements

**Output example:**
```
[SUBSET] src/sensor.cpp:15  virtual function not allowed
         Suggestion: use template or std::variant for dispatch

[SUBSET] src/sensor.cpp:23  raw pointer parameter `int* data`
         Suggestion: use std::span<int> or std::array reference

[SUBSET] src/sensor.cpp:41  implicit narrowing conversion (double → int)
         Suggestion: use static_cast<int>(expr)

Summary: 3 subset violations found
```

### 3.2 Mode: `verify` (Default)

**Purpose:** Full verification of annotated functions.

**Usage:**
```bash
arcanum src/sensor.cpp
arcanum --timeout=15 src/sensor.cpp src/filter.cpp
```

**Behavior:**
- Enforces subset compliance
- Parses contract annotations
- Generates and discharges proof obligations
- Reports results per function

**Output example:**
```
[PASS]    src/sensor.cpp:scale_percent        all 4 obligations proven (0.8s)
[FAIL]    src/sensor.cpp:clamp_value          postcondition violated (0.2s)
          Counterexample: input = -32769, result = -32769
          Expected: \result >= -32768 && \result <= 32767
[TIMEOUT] src/filter.cpp:moving_average       loop invariant unproven (10.0s)
          Obligation: .arcanum/filter_moving_average_inv_2.smt2
[PASS]    src/filter.cpp:low_pass             all 3 obligations proven (1.2s)

Summary: 2 passed, 1 failed, 1 timeout (12.2s total)
```

### 3.3 Mode: `totality`

**Purpose:** Verify totality of functions marked `//@ total`.

**Usage:**
```bash
arcanum --mode=totality src/sensor.cpp
```

**Behavior:**
- Everything in `verify` mode, plus:
- For functions annotated `//@ total`, additionally proves:
  - All loops terminate (variant is non-negative and strictly decreasing)
  - All callees' preconditions are satisfied
  - Postcondition holds for all inputs in the declared domain
  - All `switch` / `std::visit` on `enum class` / `std::variant` are exhaustive

**Output example:**
```
[PASS]    src/sensor.cpp:scale_percent        total: proven (1.1s)
[FAIL]    src/sensor.cpp:process_reading      not total: missing loop_variant
          Line 34: while (remaining > 0) — no termination measure provided
[FAIL]    src/sensor.cpp:get_priority         not total: non-exhaustive switch
          Missing case: SensorState::Shutdown
```

---

## 4. Contract Annotation Language

Annotations are written in structured comments beginning with `//@`. The annotation language is inspired by ACSL (Frama-C) and adapted for C++.

### 4.1 Function Contracts

```cpp
//@ requires: <precondition>
//@ ensures:  <postcondition>
//@ assigns:  <frame condition>
//@ total
//@ pure
```

**`requires`** — Boolean expression over function parameters. Must hold on entry. Multiple `requires` clauses are conjoined (AND).

**`ensures`** — Boolean expression over parameters, `\result`, and `\old(expr)`. Must hold on exit. Multiple `ensures` clauses are conjoined.

**`assigns`** — List of locations that may be modified. `\nothing` means the function is pure. Used for frame reasoning: anything not listed is guaranteed unchanged.

**`total`** — Asserts the function is total on its declared domain (see Section 5).

**`pure`** — Marks the function as side-effect free (`assigns: \nothing` is implied). Required for using the function in contract expressions of other functions. Must have an `ensures` clause that fully characterizes its return value (see Section 4.6.3).

### 4.2 Loop Annotations

```cpp
//@ loop_invariant: <boolean expression>
//@ loop_assigns:   <modified variables>
//@ loop_variant:   <decreasing expression>
```

**`loop_invariant`** — Holds before every iteration (including before the first). Multiple invariants are conjoined.

**`loop_assigns`** — Variables modified by the loop body. Enables frame reasoning for the loop.

**`loop_variant`** — Non-negative integer expression that strictly decreases on each iteration. Required for totality proofs. For simple bounded loops (`for (i = 0; i < n; i++)`), the tool attempts automatic inference.

### 4.3 Inline Assertions

```cpp
//@ assert: <boolean expression>     // must be proven
//@ assume: <boolean expression>     // assumed true (axiom)
```

**`assert`** — The tool must prove this holds at this program point.

**`assume`** — Taken as truth without proof. Use for environmental assumptions (e.g., hardware guarantees). Each `assume` generates a warning in the report listing all unverified assumptions.

### 4.4 Special Expressions

| Expression | Meaning | Context |
|------------|---------|---------|
| `\result` | Return value of the function | `ensures` only |
| `\old(expr)` | Value of `expr` at function entry | `ensures` only |
| `\at(expr, Label)` | Value of `expr` at the named label | Any annotation |
| `\forall T i; lo <= i < hi ==> P(i)` | Universal quantification (bounded) | Any annotation |
| `\exists T i; lo <= i < hi && P(i)` | Existential quantification (bounded) | Any annotation |
| `\nothing` | Empty set of locations | `assigns` only |
| `data[lo..hi]` | Array slice (range of elements) | `assigns`, quantifiers |
| `\valid(span, index)` | Index is within span bounds | Any annotation |
| `\separated(a, b)` | Memory regions do not overlap | Any annotation |

**Labels and `\at`:**

Use `//@ label: Name` to mark a program point, then reference it with `\at(expr, Name)`:

```cpp
void iterative_refine(std::span<float> data, size_t n) {
    for (int round = 0; round < 10; round++) {
        //@ label: RoundStart
        for (size_t i = 0; i < n; i++) {
            data[i] = refine(data[i]);
        }
        //@ assert: \forall size_t i; 0 <= i < n ==>
        //@     error(data[i]) <= error(\at(data[i], RoundStart))
    }
}
```

`\old(expr)` is equivalent to `\at(expr, Pre)` where `Pre` is an implicit label at function entry. Two built-in labels are always available:

| Label | Meaning |
|-------|---------|
| `Pre` | Function entry (same as `\old`) |
| `Post` | Function exit (for use in `assigns` clauses) |

### 4.5 Ghost State

Ghost variables and ghost assignments exist only in the specification world and are erased during compilation. They are used to model logical state that has no direct runtime representation.

**Ghost variable declaration:**
```cpp
//@ ghost: Type name = initial_value;
```

**Ghost assignment:**
```cpp
//@ ghost: name = new_value;
```

**Ghost in contracts:**
```cpp
//@ ghost: enum class ConnectionState { Idle, Connecting, Connected, Error };
//@ ghost: ConnectionState conn_state = ConnectionState::Idle;

//@ requires: conn_state == ConnectionState::Idle
//@ ensures:  conn_state == ConnectionState::Connecting
//@ assigns:  conn_state
void begin_connect() {
    send_syn_packet();
    //@ ghost: conn_state = ConnectionState::Connecting;
}

//@ requires: conn_state == ConnectionState::Connected
//@ ensures:  \result >= 0 && \result <= 65535
int read_data() {
    return receive_packet();
}
```

Ghost state is subject to the same verification as real state — assignments must be consistent with postconditions, and ghost values are tracked through the program symbolically.

### 4.6 Specification Functions

The contract language supports three kinds of reusable specification constructs, each with different capabilities and verification cost. All three can be used in `requires`, `ensures`, `assert`, `assume`, and `loop_invariant` expressions.

#### 4.6.1 Predicates

Predicates are named boolean expressions in the specification language. They are expanded inline during verification — no runtime cost, no recursion.

```cpp
//@ predicate valid_adc(int x) = x >= 0 && x <= 4095;

//@ predicate sorted(std::span<const int> a, size_t lo, size_t hi) =
//@     \forall size_t i, j; lo <= i < j < hi ==> a[i] <= a[j];

//@ predicate in_range(int x, int lo, int hi) = lo <= x && x <= hi;
```

**Usage in contracts:**
```cpp
//@ requires: valid_adc(raw)
//@ ensures:  in_range(\result, 0, 1000)
int scale_reading(int raw) { ... }
```

Predicates are the simplest and most efficient specification construct. Use them for properties that can be expressed as a single boolean expression, possibly with quantifiers.

#### 4.6.2 Logic Functions

Logic functions exist only in the specification world — they are never compiled and have no runtime representation. Unlike predicates, logic functions can return non-boolean types and can be recursive, making them suitable for defining mathematical models of data structures.

```cpp
//@ logic int sum_of(std::span<const int> a, size_t lo, size_t hi) =
//@     lo >= hi ? 0 : a[lo] + sum_of(a, lo + 1, hi);

//@ logic int max_of(std::span<const int> a, size_t lo, size_t hi) =
//@     lo + 1 >= hi ? a[lo] :
//@     (a[lo] > max_of(a, lo + 1, hi) ? a[lo] : max_of(a, lo + 1, hi));

//@ logic int factorial(int n) =
//@     n <= 0 ? 1 : n * factorial(n - 1);
```

**Usage in contracts:**
```cpp
//@ requires: n > 0 && n <= data.size()
//@ ensures: sum_of(data, 0, n) == \old(sum_of(data, 0, n))
//@ ensures: sorted(data, 0, n)
void sort(std::span<int> data, size_t n) { ... }
```

Logic functions are translated directly to SMT function definitions. Recursive logic functions are encoded using SMT recursive function support or unrolled to a bounded depth.

**Rules for logic functions:**
- No side effects (purely mathematical)
- Can reference other logic functions and predicates
- Cannot reference C++ functions or runtime state
- Recursive definitions must have a base case
- Operate on mathematical integers (no overflow) within the specification world

#### 4.6.3 Pure C++ Functions in Contracts

C++ functions annotated with `//@ pure` can be used directly in contract expressions. The verifier does not inline the function body — instead, it reasons about the function through its contract. This enables using real, executable C++ code as specification, while keeping verification modular.

**Declaring a pure function:**
```cpp
//@ total
//@ pure
//@ ensures: \result == (\forall size_t i; 0 <= i + 1 < n ==> data[i] <= data[i + 1])
bool is_sorted(std::span<const int> data, size_t n) {
    for (size_t i = 0; i + 1 < n; i++) {
        if (data[i] > data[i + 1]) return false;
    }
    return true;
}
```

**Using it in another function's contract:**
```cpp
//@ ensures: is_sorted(data, n)
void sort(std::span<int> data, size_t n) { ... }
```

**How the verifier handles this:**

The verifier chains contracts. When it encounters `is_sorted(data, n)` in a contract, it looks up `is_sorted`'s own `ensures` clause and substitutes. So proving `is_sorted(data, n)` in `sort`'s postcondition is equivalent to proving:

```
\forall size_t i; 0 <= i + 1 < n ==> data[i] <= data[i + 1]
```

The function `is_sorted` itself is verified separately — its body is proven to satisfy its contract. Once proven, the contract is used as a trusted specification everywhere else.

**Requirements for pure functions:**
- Must be annotated `//@ pure` — no side effects (`assigns: \nothing` is implied)
- Should be annotated `//@ total` — ensures the function is well-defined on its domain
- Must have an `ensures` clause that fully characterizes its return value
- The function body is verified once against its own contract
- At all usage sites in other contracts, only the contract is used (not the body)

**Why require `ensures` to fully characterize the return value?**

Without a complete `ensures`, the verifier has no information about what the function returns:

```cpp
// ✗ Bad: contract doesn't define what the function returns
//@ pure
bool is_valid(int x) { return x >= 0 && x <= 100; }

//@ ensures: is_valid(\result)  // Verifier knows nothing about is_valid
int compute() { ... }

// ✓ Good: contract fully specifies return value
//@ pure
//@ ensures: \result == (x >= 0 && x <= 100)
bool is_valid(int x) { return x >= 0 && x <= 100; }

//@ ensures: is_valid(\result)  // Verifier expands: \result >= 0 && \result <= 100
int compute() { ... }
```

#### 4.6.4 Comparison of Specification Constructs

| Property | Predicate | Logic Function | Pure C++ Function |
|----------|-----------|----------------|-------------------|
| Syntax | `//@ predicate` | `//@ logic` | `//@ pure` on C++ function |
| Return type | `bool` only | Any type | Any type |
| Recursion | No | Yes | Yes (with `//@ total`) |
| Compiled | No | No | Yes (real executable code) |
| Verification cost | Low (inline expansion) | Medium (SMT function) | Medium (contract chaining) |
| Can call C++ | No | No | Yes (other pure functions) |
| Use in contracts | ✓ | ✓ | ✓ |
| Use in runtime code | No | No | Yes |

#### 4.6.5 Combining All Three

A realistic example using all specification constructs together:

```cpp
// Predicate: simple value range check
//@ predicate valid_reading(int x) = x >= 0 && x <= 4095;

// Logic function: recursive mathematical model
//@ logic int count_valid(std::span<const int> a, size_t lo, size_t hi) =
//@     lo >= hi ? 0 :
//@     (valid_reading(a[lo]) ? 1 : 0) + count_valid(a, lo + 1, hi);

// Pure C++ function: executable validation
//@ total
//@ pure
//@ ensures: \result == (\forall size_t i; 0 <= i < n ==> valid_reading(data[i]))
bool all_valid(std::span<const int> data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!valid_reading(data[i])) return false;
    }
    return true;
}

// Main function using all three
//@ total
//@ requires: n > 0 && n <= data.size()
//@ requires: all_valid(data, n)
//@ ensures:  in_range(\result, 0, 4095)
//@ ensures:  count_valid(data, 0, n) == n
int find_max_reading(std::span<const int> data, size_t n) {
    int max = data[0];
    //@ loop_invariant: 1 <= i <= n
    //@ loop_invariant: valid_reading(max)
    //@ loop_invariant: \forall size_t j; 0 <= j < i ==> max >= data[j]
    //@ loop_variant: n - i
    for (size_t i = 1; i < n; i++) {
        if (data[i] > max) max = data[i];
    }
    return max;
}
```

### 4.7 Trusted Boundaries

The `trusted` annotation marks a function whose body is not verified. Its contract is assumed to hold. This is used for C interop, hardware access, OS calls, and any code outside the verified subset.

```cpp
//@ trusted
//@ requires: addr >= 0x4000'0000 && addr <= 0x4000'FFFF
//@ ensures:  \result >= 0 && \result <= 0xFFFF
uint16_t read_hw_register(uint32_t addr);

//@ trusted
//@ requires: buf.size() >= len
//@ ensures:  \result >= 0
//@ assigns:  buf[0..len-1]
int hal_read_dma(std::span<uint8_t> buf, size_t len);
```

**Rules for trusted functions:**
- The body (if present) is ignored by the verifier
- The contract is assumed correct without proof
- All call sites are verified against the contract normally
- The verification report lists all trusted functions and their assumed contracts

---

## 5. Totality Verification

A function annotated `//@ total` must satisfy all of the following:

### 5.1 Requirements for Totality

| Property | Description | How Proven |
|----------|-------------|------------|
| Termination | All loops and recursion terminate | `loop_variant` (user-provided or auto-inferred) |
| Runtime safety | No undefined behavior on any input in domain | WP + SMT (division by zero, overflow, bounds) |
| Postcondition | Postcondition holds for all inputs in domain | WP + SMT |
| Callee preconditions | All called functions' preconditions are met | WP + SMT at each call site |
| Exhaustiveness | All `switch` on `enum class` and `std::visit` on `std::variant` cover every case | Static check + SMT for value ranges |

### 5.2 Automatic Termination Inference

For simple loop patterns, the tool infers termination without user annotation:

```cpp
// Auto-inferred: variant is (N - i), i increases by 1 each iteration
for (size_t i = 0; i < N; i++) { ... }

// Auto-inferred: variant is i, i decreases by 1 each iteration
for (size_t i = n; i > 0; i--) { ... }

// Auto-inferred: range-based for always terminates over finite containers
for (auto& x : array) { ... }
```

**Patterns recognized for auto-inference:**
- `for (T i = start; i < end; i++)` where `end - start` is non-negative
- `for (T i = start; i > end; i--)` where `start - end` is non-negative
- Range-based `for` over `std::array` or `std::span`

All other loops require an explicit `loop_variant`.

### 5.3 Conditional Totality

A function with a precondition is total **on its declared domain**:

```cpp
//@ total
//@ requires: b != 0
//@ ensures: \result == a / b
int safe_div(int a, int b) {
    return a / b;
}
```

This function is total: for all `(a, b)` where `b != 0`, it terminates and produces the correct result. The verifier checks that callers satisfy `b != 0` at every call site.

A function with `//@ requires: true` (or no `requires` clause) must handle all possible inputs.

### 5.4 Recursion

Recursive functions require a decreasing measure:

```cpp
//@ total
//@ requires: n >= 0
//@ ensures: \result >= 1
//@ decreases: n
int factorial(int n) {
    if (n == 0) return 1;
    //@ assert: n - 1 >= 0 && n - 1 < n
    return n * factorial(n - 1);
}
```

The `decreases` annotation specifies an expression that is non-negative and strictly decreases on every recursive call. Mutual recursion requires a shared decreasing measure.

---

## 6. Floating-Point Handling

### 6.1 Default Mode: Real Arithmetic

By default, `float` and `double` are modeled as mathematical real numbers. Rounding, denormals, NaN, and infinity are ignored.

This is sound for properties that hold with a margin larger than floating-point error (e.g., "result is between 0 and 1000" when the actual range is [0, 999.7]).

```cpp
//@ requires: x >= 0.0f && x <= 1.0f
//@ ensures: \result >= 0.0f && \result <= 100.0f
float to_percent(float x) {
    return x * 100.0f;  // Verified under real arithmetic
}
```

### 6.2 NaN and Infinity Safety

The `fp_safe` annotation adds proof obligations that no floating-point operation produces NaN or infinity:

```cpp
//@ fp_safe: true
//@ requires: alpha >= 0.0f && alpha <= 1.0f
//@ requires: -1000.0f <= input && input <= 1000.0f
//@ requires: -1000.0f <= prev && prev <= 1000.0f
//@ ensures: -1000.0f <= \result && \result <= 1000.0f
float low_pass(float alpha, float input, float prev) {
    return alpha * input + (1.0f - alpha) * prev;
}
```

With `fp_safe`, the verifier additionally proves:
- No division by zero in floating-point operations
- No operation produces NaN (e.g., `0.0f / 0.0f`, `sqrt(-1.0f)`)
- No operation produces infinity (overflow to ±∞)

### 6.3 Strict IEEE 754 Mode (Future)

Reserved for version 2. Will model exact IEEE 754 rounding and enable precise floating-point verification:

```cpp
//@ fp_mode: strict
//@ ensures: -1000.002f <= \result && \result <= 1000.002f
float low_pass_precise(float alpha, float input, float prev);
```

### 6.4 Recommended Practice: Fixed-Point for ASIL D

For highest-assurance code, fixed-point arithmetic avoids floating-point complexity entirely:

```cpp
// Q12 format: value = raw / 4096.0
using Q12 = int32_t;

//@ predicate valid_q12(Q12 x) = x >= -2'097'152 && x <= 2'097'151;

//@ total
//@ requires: valid_q12(a) && valid_q12(b)
//@ ensures: valid_q12(\result)
Q12 q12_add(Q12 a, Q12 b) {
    int64_t sum = static_cast<int64_t>(a) + static_cast<int64_t>(b);
    if (sum > 2'097'151) return 2'097'151;    // saturate
    if (sum < -2'097'152) return -2'097'152;  // saturate
    return static_cast<Q12>(sum);
}
```

This is verified entirely with integer/bitvector SMT theories — fast and precise.

---

## 7. Integer Overflow Handling

### 9.1 Overflow Modes

The tool supports three modes for signed integer overflow, configured at the project level with per-function overrides.

**`trap` (default):** Overflow is an error. The verifier generates a proof obligation at every arithmetic operation that the result fits in the target type. The user must prove absence of overflow.

```cpp
// With trap mode, the verifier requires proof of no overflow:
//@ requires: a >= 0 && a <= 1000
//@ requires: b >= 0 && b <= 1000
//@ ensures: \result >= 0 && \result <= 2000
int add(int a, int b) {
    return a + b;  // verifier proves: 0+0=0 >= INT_MIN ✓, 1000+1000=2000 <= INT_MAX ✓
}
```

**`wrap`:** Two's complement wrapping. The verifier models signed integers as bitvectors. Overflow wraps silently. Useful for CRC, hash, and bitwise algorithms.

```cpp
//@ overflow: wrap
uint32_t compute_crc(std::span<const uint8_t> data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    // Wrapping arithmetic is intentional here
    ...
}
```

**`saturate`:** Overflow clamps to type min/max. The verifier models all signed arithmetic with implicit clamping. Useful for control algorithms.

```cpp
//@ overflow: saturate
int16_t pid_output(int16_t error, int16_t integral, int16_t derivative) {
    // Arithmetic saturates at INT16_MIN / INT16_MAX
    return kp * error + ki * integral + kd * derivative;
}
```

### 9.2 Per-Function Override

The overflow mode can be overridden on individual functions:

```cpp
//@ overflow: wrap
uint32_t hash(std::span<const uint8_t> data) { ... }

//@ overflow: saturate
int16_t clamp_output(int32_t value) { ... }
```

### 9.3 Unsigned Integers

Unsigned integer overflow is always wrapping, as defined by the C++ standard. No configuration needed.

---

## 8. Project Configuration

Project-wide settings are stored in `.arcanum.yaml` at the project root:

```yaml
# .arcanum.yaml
overflow:
  signed: trap          # trap (default), wrap, or saturate
  unsigned: wrap        # always wrap per C++ standard (not configurable)

solver:
  name: z3              # z3 (default), cvc5, alt-ergo
  timeout: 10           # per-obligation timeout in seconds
  parallel: 0           # 0 = use all available cores

std: c++20              # c++20 (default) or c++23

floating_point:
  mode: real            # real (default); strict reserved for future
  fp_safe: false        # global default for fp_safe (can override per-function)

output:
  format: terminal      # terminal (default) or json
  dump_smt: false       # write SMT queries to .arcanum/ directory
  trust_report: true    # always generate trust assumption report
```

The CLI flags override the configuration file. The configuration file overrides built-in defaults.

---

## 9. Verification Report

### 9.1 Result Statuses

| Status | Meaning | User Action |
|--------|---------|-------------|
| `PASS` | All proof obligations discharged | None |
| `FAIL` | Counterexample found — property is violated | Fix code or fix specification |
| `TIMEOUT` | Solver could not determine result within time limit | Add lemmas, simplify spec, or increase timeout |

### 9.2 Output Formats

**Terminal output** (default):
```
[PASS]    src/sensor.cpp:scale_percent        4/4 proven (0.8s)
[FAIL]    src/sensor.cpp:clamp_value          postcondition violated
          Counterexample: input = -32769, result = -32769
          Expected: -32768 <= \result <= 32767
[TIMEOUT] src/filter.cpp:moving_average       invariant unproven (10.0s)
          Dump: .arcanum/filter_moving_average_inv_2.smt2

Summary: 2 passed, 1 failed, 1 timeout
         Total: 8/10 obligations proven (12.2s)
```

**JSON output** (`--format=json`):
```json
{
  "version": "1.0",
  "timestamp": "2026-02-22T10:30:00Z",
  "results": [
    {
      "function": "scale_percent",
      "file": "src/sensor.cpp",
      "line": 8,
      "status": "pass",
      "obligations": 4,
      "proven": 4,
      "time_seconds": 0.8
    },
    {
      "function": "clamp_value",
      "file": "src/sensor.cpp",
      "line": 20,
      "status": "fail",
      "obligations": 3,
      "proven": 2,
      "failed": [
        {
          "kind": "postcondition",
          "line": 21,
          "property": "\\result >= -32768 && \\result <= 32767",
          "counterexample": {
            "input": -32769,
            "result": -32769
          }
        }
      ],
      "time_seconds": 0.2
    }
  ],
  "trusted_functions": [
    {
      "function": "read_hw_register",
      "file": "src/hal.hpp",
      "line": 12,
      "assumed_contract": "requires: addr in [0x40000000, 0x4000FFFF]; ensures: result in [0, 0xFFFF]"
    }
  ],
  "assumptions": [
    {
      "file": "src/sensor.cpp",
      "line": 55,
      "expression": "adc_resolution == 12"
    }
  ],
  "summary": {
    "functions_verified": 3,
    "pass": 2,
    "fail": 1,
    "timeout": 0,
    "total_obligations": 10,
    "total_proven": 9,
    "total_time_seconds": 12.2
  }
}
```

### 9.3 Trust Report

The tool always outputs a summary of all verification assumptions:

```
=== Trust Report ===

Trusted functions (contract assumed, body not verified):
  1. read_hw_register    src/hal.hpp:12
  2. hal_read_dma        src/hal.hpp:25

Assume statements (user axioms, not proven):
  1. adc_resolution == 12    src/sensor.cpp:55

These assumptions are NOT verified. The correctness of the
verification depends on these assumptions being true.
```

This is essential for ISO 26262 audits — the assessor needs to know exactly what was proven and what was assumed.

---

## 10. Command-Line Interface

### 10.1 Basic Usage

```bash
# Verify a single file
arcanum src/sensor.cpp

# Verify multiple files
arcanum src/sensor.cpp src/filter.cpp src/control.cpp

# Subset check only (no contracts needed)
arcanum --mode=subset-check src/sensor.cpp

# Totality mode
arcanum --mode=totality src/sensor.cpp
```

### 10.2 Options

```
arcanum [OPTIONS] <files...>

Modes:
  --mode=verify          Full verification (default)
  --mode=subset-check    Only check subset compliance
  --mode=totality        Verify mode + totality for //@ total functions

Solver:
  --timeout=<seconds>    Per-obligation SMT timeout (default: 10)
  --solver=<name>        SMT solver: z3 (default), cvc5, alt-ergo
  --solver-path=<path>   Path to solver binary
  --parallel=<n>         Verify n obligations in parallel (default: nproc)

Output:
  --format=terminal      Human-readable output (default)
  --format=json          Machine-readable JSON
  --report=<path>        Write report to file
  --dump-smt             Write all SMT queries to .arcanum/ directory
  --dump-arc             Dump Arc MLIR dialect (textual form) to stdout
  --dump-whyml           Dump generated WhyML to stdout
  --verbose              Show per-obligation details for passing functions

Include:
  --include=<path>       Additional include path for headers
  --std=<standard>       C++ standard: c++20 (default), c++23
  --defines=<KEY=VAL>    Preprocessor definitions

Trust:
  --trust-report         Generate trust assumption report
  --warn-assumes         Warn on every //@ assume statement
```

### 10.3 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All verifications passed |
| 1 | One or more verifications failed (counterexample) |
| 2 | One or more timeouts (no failures) |
| 3 | Subset violation found |
| 4 | Parse or annotation error |
| 5 | Tool error (solver not found, internal error) |

---

## 11. Implementation Plan

### Phase 1: Subset Enforcer

**Deliverable:** `arcanum --mode=subset-check`

**Tasks:**
1. Set up Clang LibTooling project structure
2. Implement AST visitor that walks all nodes
3. Implement rejection rules for each eliminated feature
4. Generate diagnostic messages with source locations and fix suggestions
5. Test against automotive code samples

**Estimated effort:** 4–6 weeks

### Phase 2: Contract Parser and Predicates

**Deliverable:** Parse `//@ ...` annotations and attach to AST, including predicate definitions

**Tasks:**
1. Implement lexer/parser for the contract annotation language
2. Support: `requires`, `ensures`, `assigns`, `assert`, `assume`
3. Support: `loop_invariant`, `loop_assigns`, `loop_variant`
4. Support: `\result`, `\old`, `\at`, `\forall`, `\exists`
5. Support: `//@ predicate` definitions with inline expansion
6. Support: `//@ label` for named program points
7. Type-check contract expressions against the C++ context
8. Attach parsed contracts to corresponding AST nodes

**Estimated effort:** 5–7 weeks

### Phase 3: Arc MLIR Dialect

**Deliverable:** Define the Arc dialect and implement lowering from Clang AST

**Tasks:**
1. Define Arc dialect types (`!arc.i32`, `!arc.span<T>`, `!arc.array<T,N>`, etc.)
2. Define Arc dialect operations (arithmetic, memory access, control flow, specification)
3. Implement Clang AST → Arc MLIR lowering for program constructs
4. Implement contract attachment as MLIR attributes on `arc.func` and `arc.for`
5. Implement `arc.predicate` and `arc.logic` for specification functions
6. Implement `arc.ghost_read` / `arc.ghost_write` for ghost state
7. Carry source location metadata through MLIR's location tracking
8. Implement textual dump (`--dump-arc`) for debugging

**Estimated effort:** 8–10 weeks

### Phase 4: MLIR Passes and WhyML Emission

**Deliverable:** Generate WhyML from Arc MLIR and discharge with Z3 via Why3

**Tasks:**
1. Implement basic MLIR passes: constant propagation, overflow check simplification
2. Implement loop normalization pass for automatic variant inference
3. Implement Arc → WhyML translation for all operations
4. Implement pure function contract chaining: substitute `ensures` clause at `arc.pure_call` sites
5. Interface with Why3 for VC generation and SMT dispatch
6. Map Why3/SMT results back to source locations via MLIR location metadata
7. Extract counterexamples from failing obligations

**Estimated effort:** 8–12 weeks

### Phase 5: Ghost State and Logic Functions

**Deliverable:** Support ghost variables, ghost assignments, and recursive specification functions

**Tasks:**
1. Track ghost state through the Arc dialect and WhyML emission
2. Generate verification conditions for ghost state transitions
3. Translate `arc.logic` operations to WhyML recursive functions
4. Implement bounded unrolling for recursive logic functions
5. Validate logic function definitions (base case, well-foundedness)

**Estimated effort:** 4–6 weeks

### Phase 6: Totality Checking

**Deliverable:** `arcanum --mode=totality`

**Tasks:**
1. Implement loop variant checking (user-provided) via `arc.variant` ops
2. Implement automatic termination inference as an MLIR pass
3. Implement exhaustiveness checking for `arc.switch` on enum/variant
4. Implement `decreases` annotation for recursive functions
5. Validate `//@ pure` function totality (required for use in contracts)
6. Integrate all checks under the `//@ total` annotation

**Estimated effort:** 4–6 weeks

### Phase 7: Floating-Point

**Deliverable:** Real arithmetic mode + `fp_safe`

**Tasks:**
1. Model `!arc.f32` / `!arc.f64` as SMT `Real` sort in WhyML emission
2. Translate floating-point Arc operations to real arithmetic
3. Implement `fp_safe` check: prove absence of NaN/infinity-producing operations
4. Document limitations of real arithmetic approximation

**Estimated effort:** 3–4 weeks

### Phase 8: Advanced MLIR Passes

**Deliverable:** Optimization passes that reduce solver workload

**Tasks:**
1. Contract propagation pass (inline simple predicates)
2. Dead specification elimination (remove unreferenced ghost state)
3. Overflow check simplification (prove trivial checks at MLIR level)
4. Benchmarking: measure proof time reduction from passes

**Estimated effort:** 4–6 weeks

### Phase 9: Bazel Integration

**Deliverable:** Bazel build rule for verification

**Tasks:**
1. Create `arcanum_verify()` Bazel rule
2. Integrate with existing C++ compilation rules
3. Cache verification results (re-verify only changed files)
4. Support as a CI gate (fail build on verification failure)

**Estimated effort:** 2–3 weeks

---

## 12. Relationship to Standards

| Standard | How Arcanum Supports It |
|----------|-------------------------------|
| ISO 26262 Part 6, Table 7 | Formal verification (method 1c) for ASIL C and D |
| ISO 26262 Part 6, Table 9 | Formal verification of software units |
| MISRA C++ 2023 | Subset rules are a strict superset of MISRA restrictions |
| AUTOSAR C++14 | Compatible with AUTOSAR coding guidelines |
| C++26 Contracts | Annotation syntax aligned with C++ Contracts direction |

### 12.1 ISO 26262 Compliance Artifacts

The tool produces verification evidence suitable for ISO 26262 work products:

- **Verification report (JSON):** Documents which functions were verified, which properties were proven, and which assumptions were made.
- **Trust report:** Lists all unverified boundaries and assumed axioms, supporting the safety case argument.
- **Proof obligation archive:** SMT-LIB files can be archived and re-checked for audit reproducibility.

---

## 13. Future Directions

- **Strict IEEE 754 floating-point mode** using SMT FP theory or interval arithmetic
- **Concurrency verification** for AUTOSAR runnable model with restricted shared state
- **Automatic invariant inference** using abstract interpretation as a preprocessing step
- **Interactive proof mode** for obligations that SMT solvers cannot discharge automatically
- **LSP / IDE integration** for real-time verification feedback in editors
- **Recursive predicates** extending `//@ predicate` to support recursion (currently only `//@ logic` supports recursion)
- **Whole-program analysis mode** for verifying global properties across translation units
- **Pure function inlining** as an optional strategy for cases where contract chaining is insufficient
- **Lemma annotations** (`//@ lemma`) for user-provided intermediate proof steps to help the SMT solver
