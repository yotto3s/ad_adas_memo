# Arcanum — Safe C++ Subset for Automotive Formal Verification

## Overview

This document defines the restricted subset of C++ supported by the Arcanum verification tool. The subset is designed for safety-critical automotive software, eliminating C++ features that are difficult or infeasible to formally verify while retaining enough expressiveness for practical embedded development.

The goal is to enable a standalone formal verification tool — similar to Frama-C for C — that can mathematically prove correctness properties of programs written in this subset.

### Design Principles

1. **Every function call is statically resolvable** — the verifier always knows exactly which code executes.
2. **No hidden control flow** — no exceptions, no implicit conversions, no virtual dispatch.
3. **Memory safety by construction** — no raw pointers, no manual allocation, no aliasing ambiguity.
4. **All behavior is defined** — no undefined behavior exists in the subset.
5. **Verification tractability over expressiveness** — features are included only if they can be formally modeled.

---

## Eliminated Features

The following C++ features are eliminated due to their high or very high verification difficulty.

### Exceptions (Difficulty: High)

**Eliminated:** `throw`, `try`, `catch`, `noexcept` specifications, stack unwinding.

**Rationale:** Exceptions create hidden control flow paths at every function call site and interact with destructors in complex ways. Modeling exception propagation requires tracking all possible throw sites and their types through the entire call graph.

**Replacement:** Use `std::expected<T, E>` (C++23) or a custom `Result<T, E>` type for recoverable errors. Use `[[noreturn]]` for truly unrecoverable failures.

```cpp
// ✗ Eliminated
int parse(const char* s) {
    if (!valid(s)) throw std::invalid_argument("bad input");
    return do_parse(s);
}

// ✓ Allowed
std::expected<int, ErrorCode> parse(std::span<const char> s) {
    if (!valid(s)) return std::unexpected(ErrorCode::InvalidInput);
    return do_parse(s);
}
```

### Raw Pointers and Pointer Arithmetic (Difficulty: Very High)

**Eliminated:** `T*` as a general-purpose type, `->` on raw pointers, pointer arithmetic (`p + n`, `p - q`, `p[n]` on pointers), `NULL`, `nullptr` as a value assigned to pointer variables.

**Rationale:** Pointer aliasing is the single hardest problem in C/C++ verification. Determining whether two pointers refer to the same memory requires separation logic or a complex memory model. Pointer arithmetic allows arbitrary memory access and makes bounds verification extremely difficult.

**Replacement:** Use references (`T&`, `const T&`), `std::array<T, N>`, `std::span<T, N>` (static extent), and `std::span<T>` (dynamic extent with contracts).

```cpp
// ✗ Eliminated
void process(int* data, int size) {
    for (int i = 0; i < size; i++)
        data[i] *= 2;
}

// ✓ Allowed
template<std::size_t N>
void process(std::array<int, N>& data) {
    for (std::size_t i = 0; i < N; i++)
        data[i] *= 2;
}

// ✓ Also allowed (dynamic extent with contract)
//@ requires: data.size() > 0
void process(std::span<int> data) {
    for (std::size_t i = 0; i < data.size(); i++)
        data[i] *= 2;
}
```

### Dynamic Memory Allocation (Difficulty: High)

**Eliminated:** `new`, `delete`, `new[]`, `delete[]`, `malloc`, `free`, `std::allocator` in general-purpose use, heap-allocated containers (`std::vector`, `std::map`, `std::string`, etc.).

**Rationale:** Dynamic allocation introduces the possibility of allocation failure, memory leaks, use-after-free, and double-free. Verifying heap properties requires sophisticated memory models. It is also prohibited by most automotive coding standards for hard real-time systems.

**Replacement:** Use stack allocation, `std::array`, and statically-sized containers. Where dynamic-like behavior is needed, use arena/pool allocators with static capacity.

```cpp
// ✗ Eliminated
auto data = new int[size];
std::vector<int> readings;
readings.push_back(42);

// ✓ Allowed
std::array<int, 256> readings{};
std::size_t count = 0;

//@ requires: count < readings.size()
void add_reading(int value) {
    readings[count] = value;
    count++;
}
```

### Virtual Dispatch and Runtime Polymorphism (Difficulty: High)

**Eliminated:** `virtual` functions, `override`, abstract classes, `dynamic_cast`, RTTI (`typeid`).

**Rationale:** Virtual dispatch makes it impossible to determine the call target statically. The verifier would need to track the dynamic type of every object through the program, including through containers and function boundaries. This dramatically increases verification complexity.

**Replacement:** Use templates (static polymorphism), `std::variant` with `std::visit`, or `constexpr if` for compile-time dispatch.

```cpp
// ✗ Eliminated
class Sensor {
public:
    virtual int read() = 0;
};
class TempSensor : public Sensor {
public:
    int read() override { return adc_value * 100 / 4095; }
};

// ✓ Allowed — templates (static dispatch)
template<typename SensorT>
int read_scaled(SensorT& sensor) {
    return sensor.read();  // resolved at compile time
}

// ✓ Allowed — variant-based dispatch
using AnySensor = std::variant<TempSensor, PressureSensor>;

int read_sensor(AnySensor& sensor) {
    return std::visit([](auto& s) { return s.read(); }, sensor);
}
```

### `reinterpret_cast` and Type Punning (Difficulty: High)

**Eliminated:** `reinterpret_cast`, C-style casts `(T*)expr`, `union` for type punning, `memcpy`-based type reinterpretation.

**Rationale:** Type punning breaks the type system entirely. The verifier would need to reason about raw memory bytes and their reinterpretation under different types. This makes the formal memory model dramatically more complex.

**Replacement:** Use `std::bit_cast<T>` (C++20) for well-defined, type-safe reinterpretation. Use `static_cast` for conversions within the type hierarchy.

```cpp
// ✗ Eliminated
float f = 3.14f;
int bits = *reinterpret_cast<int*>(&f);
int bits2 = *(int*)&f;

// ✓ Allowed
float f = 3.14f;
int bits = std::bit_cast<int>(f);
```

### `goto` and `setjmp`/`longjmp` (Difficulty: Medium-High)

**Eliminated:** `goto`, `setjmp`, `longjmp`.

**Rationale:** `goto` creates arbitrary control flow that breaks structured reasoning. `setjmp`/`longjmp` is essentially C-style exceptions — non-local jumps that bypass destructors and structured control flow.

**Replacement:** Use structured control flow (`if`, `for`, `while`, `break`, `continue`, early `return`).

### Shared Mutable Concurrency (Difficulty: Very High)

**Eliminated:** `std::thread`, `std::mutex`, `std::condition_variable`, raw shared mutable state across threads.

**Rationale:** Verifying concurrent programs with shared mutable state requires reasoning about all possible interleavings, which causes state space explosion. Even simple concurrent programs can have subtle data races that are extremely hard to prove absent.

**Replacement:** Use a restricted concurrency model:
- **Single-writer, multiple-reader** data patterns
- **Message passing** between tasks
- **`std::atomic`** for simple shared counters/flags (allowed, with usage contracts)
- **AUTOSAR-style runnable model** where the OS scheduler guarantees non-preemption within a runnable

The concurrency model should match the target RTOS execution model (e.g., AUTOSAR OS task model) where timing and scheduling provide isolation guarantees.

### Implicit Conversions (Difficulty: Medium, but High Bug Risk)

**Eliminated:** Implicit narrowing conversions, implicit conversions between unrelated types, implicit `bool` conversions for non-boolean types.

**Rationale:** While not the hardest to verify, implicit conversions are a major source of subtle bugs (e.g., signed/unsigned mixing, floating-point truncation). Eliminating them reduces both verification complexity and real-world bugs.

**Replacement:** Use explicit `static_cast` and define conversion functions.

```cpp
// ✗ Eliminated
int x = 3.14;          // implicit narrowing
if (some_integer) {}   // implicit bool conversion

// ✓ Allowed
int x = static_cast<int>(3.14);
if (some_integer != 0) {}
```

### Preprocessor Macros (Difficulty: Medium)

**Eliminated:** `#define` for function-like macros, conditional compilation that changes program semantics. Allowed: `#include`, `#pragma once`, simple constant macros, and include guards.

**Rationale:** Macros operate on text before parsing, bypassing the type system and making it impossible to reason about the code the verifier sees without running the preprocessor first. Clang handles preprocessing, so this is more about code hygiene than verifier difficulty.

**Replacement:** Use `constexpr` variables, `consteval` functions, templates, and `if constexpr`.

---

## Allowed Features

### Core Language

| Feature | Status | Notes |
|---------|--------|-------|
| `struct` / `class` (non-virtual) | ✓ Allowed | Plain data types with methods |
| Constructors / Destructors | ✓ Allowed | RAII is valuable; simplified without exceptions |
| References (`T&`, `const T&`) | ✓ Allowed | Must not dangle; verifier checks lifetime |
| `const` / `constexpr` / `consteval` | ✓ Allowed | Encouraged; compile-time computation is pre-verified |
| `static_assert` | ✓ Allowed | Compile-time invariant checking |
| `enum class` | ✓ Allowed | Finite value sets, easy to verify exhaustively |
| `if` / `for` / `while` / `switch` | ✓ Allowed | Standard structured control flow |
| `break` / `continue` / `return` | ✓ Allowed | Structured exits |
| Range-based `for` | ✓ Allowed | Desugars to iterators, bounds known |
| Templates | ✓ Allowed | Resolved at compile time; verified per-instantiation |
| `constexpr if` | ✓ Allowed | Compile-time branching |
| Concepts (C++20) | ✓ Allowed | Constrains templates, improves diagnostics |
| `static_cast` | ✓ Allowed | Explicit, well-defined conversions only |
| `std::bit_cast` (C++20) | ✓ Allowed | Well-defined type reinterpretation |
| Operator overloading | ✓ Allowed | Resolved to function calls by Clang |
| Lambdas (value capture only) | ✓ Allowed | Reference capture creates aliasing concerns |
| Namespaces | ✓ Allowed | No semantic impact |
| `auto` type deduction | ✓ Allowed | Resolved at compile time |
| Structured bindings | ✓ Allowed | Syntactic sugar, no semantic complexity |

### Standard Library (Allowed Subset)

| Feature | Status | Notes |
|---------|--------|-------|
| `std::array<T, N>` | ✓ Allowed | Fixed-size, bounds-verifiable |
| `std::span<T, N>` (static extent) | ✓ Allowed | Non-owning view, size in type |
| `std::span<T>` (dynamic extent) | ✓ With contracts | Size checked via preconditions |
| `std::expected<T, E>` | ✓ Allowed | Error handling replacement for exceptions |
| `std::optional<T>` | ✓ Allowed | Nullable values without pointers |
| `std::variant<Ts...>` | ✓ Allowed | Type-safe union, verifiable via `visit` |
| `std::tuple` / `std::pair` | ✓ Allowed | Simple composite types |
| `std::atomic<T>` | ✓ With contracts | For simple shared flags/counters only |
| `<cstdint>` fixed-width types | ✓ Allowed | `int32_t`, `uint16_t`, etc. |
| `<algorithm>` (non-allocating) | ✓ Allowed | `std::sort`, `std::find`, etc. on arrays/spans |
| `<numeric>` | ✓ Allowed | `std::accumulate`, etc. |
| `<cmath>` | ✓ Allowed | Mathematical functions |
| `std::bitset<N>` | ✓ Allowed | Fixed-size bit manipulation |

### Arithmetic Behavior

All arithmetic in this subset has **defined behavior**. The default overflow mode is **trap**, meaning the verifier must prove that overflow never occurs. This can be configured per-project or per-function.

**Overflow modes (configured in `.arcanum.yaml`):**

- **`trap` (default):** Signed integer overflow is an error. The verifier generates a proof obligation at every arithmetic operation that the result fits in the target type. This is the strictest and safest mode.
- **`wrap`:** Two's complement wrapping. The verifier models signed integers as bitvectors. Overflow wraps silently. Useful for CRC, hash, and bitwise algorithms.
- **`saturate`:** Overflow clamps to type min/max. Useful for control algorithms where clamping is the desired behavior.

**Per-function override:**
```cpp
//@ overflow: wrap
uint32_t compute_crc(std::span<const uint8_t> data) {
    // CRC computation intentionally uses wrapping arithmetic
}
```

**Other arithmetic rules:**
- **Unsigned integer overflow:** wrapping (as per standard C++).
- **Division by zero:** verifier must prove divisor ≠ 0 at every division site.
- **Shift amounts:** verifier must prove shift is within `[0, bit_width)`.

---

## Contract Annotations

The subset includes a specification language for formal verification, inspired by ACSL (Frama-C) and C++26 Contracts.

### Function Contracts

```cpp
//@ requires: n > 0 && n <= data.size()
//@ ensures:  \result >= 0 && \result <= 4095
//@ assigns:  \nothing
int find_max(std::span<const int> data, std::size_t n) {
    int max = data[0];
    //@ loop_invariant: i >= 1 && i <= n
    //@ loop_invariant: max >= 0 && max <= 4095
    //@ loop_invariant: \forall std::size_t j; 0 <= j < i ==> max >= data[j]
    //@ loop_assigns: i, max
    //@ loop_variant: n - i
    for (std::size_t i = 1; i < n; i++) {
        if (data[i] > max) max = data[i];
    }
    return max;
}
```

### Contract Keywords

| Annotation | Meaning |
|------------|---------|
| `requires` | Precondition — must hold on function entry |
| `ensures` | Postcondition — must hold on function exit |
| `assigns` | Frame condition — which variables may be modified |
| `total` | Function is total on its declared domain (terminates, no errors, postcondition holds) |
| `pure` | Function has no side effects; can be used in other functions' contracts |
| `loop_invariant` | Holds at every loop iteration |
| `loop_assigns` | Variables modified by the loop body |
| `loop_variant` | Decreasing expression proving termination |
| `assert` | Must hold at this program point |
| `assume` | Assumed true (axiom, not proven) |
| `trusted` | Function body is not verified; contract is assumed correct |
| `label` | Named program point for use with `\at(expr, Label)` |
| `predicate` | Named boolean specification expression |
| `logic` | Specification-only function (may be recursive, not compiled) |
| `overflow` | Per-function override of arithmetic overflow mode |
| `fp_safe` | Prove absence of NaN and infinity in floating-point operations |

### Trusted Boundary (C Interop)

For interfacing with existing C code, hardware registers, or OS APIs:

```cpp
//@ trusted
//@ requires: reg_addr >= 0x4000'0000 && reg_addr <= 0x4000'FFFF
//@ ensures:  \result >= 0 && \result <= 0xFFFF
uint16_t read_hw_register(uint32_t reg_addr);
```

The `trusted` annotation marks the boundary between verified and unverified code. The verifier assumes the contract holds and verifies all callers against it.

---

## Verification Guarantees

Programs written in this subset and verified by the tool are guaranteed to be free of:

- Buffer overflows / out-of-bounds access
- Integer overflow (or overflow is well-defined)
- Division by zero
- Use of uninitialized variables
- Null dereference (no nullable pointers exist)
- Use-after-free / double-free (no manual memory management)
- Data races (restricted concurrency model)
- Deadlocks (with restricted concurrency)
- Violation of user-specified functional properties (contracts)

---

## Relationship to Existing Standards

| Standard | Relationship |
|----------|-------------|
| MISRA C++ 2023 | This subset is stricter than MISRA. All MISRA-banned features are also eliminated here. |
| AUTOSAR C++14 | Compatible; this subset adds formal verification on top of AUTOSAR restrictions. |
| ISO 26262 | Supports ASIL A–D. Formal verification satisfies Table 7 (1c) of Part 6. |
| C++26 Contracts | Contract syntax is aligned with the direction of the C++ standard. |
| CERT C++ | This subset eliminates most categories of vulnerabilities addressed by CERT rules. |

---

## Target Toolchain Architecture

```
Source (.cpp, restricted subset + contracts)
    │
    ▼
┌──────────────────────┐
│   Clang Frontend     │  Parse C++, resolve templates,
│   (LibTooling)       │  reject non-subset features
└──────────┬───────────┘
           │ Clang AST (subset only)
           ▼
┌──────────────────────┐
│   Subset Enforcer    │  Verify all constructs are within
│                      │  the allowed subset
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Contract Parser    │  Parse //@ annotations,
│                      │  attach to AST nodes
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Arc Dialect        │  Lower Clang AST to Arc MLIR
│   Lowering           │  (~30 verification-oriented ops)
└──────────┬───────────┘
           │ Arc MLIR
           ▼
┌──────────────────────┐
│   MLIR Passes        │  Contract propagation, overflow
│                      │  simplification, loop normalization
└──────────┬───────────┘
           │ Optimized Arc MLIR
           ▼
┌──────────────────────┐
│   WhyML Emitter      │  Near 1:1 translation from Arc
│                      │  dialect to WhyML
└──────────┬───────────┘
           │ WhyML
           ▼
┌──────────────────────┐
│   Why3 Backend       │  VC generation + SMT solvers
│   (Z3 / CVC5)       │  (Z3, CVC5, Alt-Ergo)
└──────────┬───────────┘
           │
           ▼
    Proved ✓ / Counterexample ✗
```

---

## Future Considerations

- **Recursion:** Currently allowed but requires termination proofs (`loop_variant`-style decreasing measures). May restrict to bounded recursion or eliminate entirely.
- **Floating-point verification:** SMT solvers have limited floating-point reasoning. May need interval arithmetic or specialized decision procedures.
- **Concurrency model formalization:** The restricted concurrency model needs precise formal semantics tied to the target RTOS.
- **Inheritance (non-virtual):** May allow simple `struct` inheritance for composition. No virtual dispatch, no slicing.
- **Integration with Bazel:** The verification tool should integrate as a Bazel rule for seamless build-time verification.
