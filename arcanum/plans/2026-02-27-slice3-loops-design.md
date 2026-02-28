# Slice 3 Design: Loop Verification

## Summary

Slice 3 adds loop support to Arcanum's verification pipeline. After this slice, the tool can verify iterative programs with `for`, `while`, and `do-while` loops, including loop contracts (`loop_invariant`, `loop_variant`, `loop_assigns`) and early exit via `break`/`continue`.

All loops are translated to recursive WhyML functions, providing a uniform verification model where loop invariants become preconditions and loop variants become termination measures.

## Scope

### In Scope

| Feature | Description |
|---------|-------------|
| `for` loops | Standard counted and general C-style |
| `while` loops | Condition-driven iteration |
| `do-while` loops | Body-first condition loops |
| `break` / `continue` | Early loop exit and skip-to-next |
| `//@ loop_invariant` | Boolean expression that holds before every iteration |
| `//@ loop_variant` | Non-negative integer expression that strictly decreases (termination) |
| `//@ loop_assigns` | Comma-separated list of variables modified by the loop body |
| `//@ label` | Optional label for loop ops (future: labeled break/continue) |
| Auto-inference | Automatic `loop_variant` for standard counted for-loops |
| Nested loops | Naturally supported via nested recursive functions |

### Out of Scope (Deferred)

| Feature | Blocked By | Target Slice |
|---------|-----------|-------------|
| Range-based `for` | `std::array`/`std::span` types | Slice 5 |
| Function calls in loop bodies | Modular verification / callee contracts | Slice 4 |
| `\forall`/`\exists` in invariants | Quantifier parsing in Contract Parser | Slice 5 |
| Native WhyML loop emission | Optimization (not needed for correctness) | Future (optional) |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| WhyML emission strategy | All loops -> recursive WhyML functions | Uniform model, one code path. break = return, continue = skip to recursive call. Simplest correct implementation. |
| Arc dialect loop representation | Single `arc.loop` op | One op covers for/while/do-while via `condition_first` flag and optional regions. No need for separate arc.for/arc.while/arc.do_while since all loops emit the same way. |
| MLIR pass | `LoopContractPass` (lightweight) | Auto-infers variant for counted for-loops, auto-computes assigns from arc.assign analysis. Much simpler than full loop classification/normalization. |
| Mutable variable handling | Keep `arc.var`/`arc.assign` in loop body | Emitter converts modified variables to recursive function parameters. No SSA conversion needed at the dialect level. |
| do-while | `condition_first = false` flag on `arc.loop` | Body executes before first condition check. Emitter places body before the if-condition in the recursive function. |
| Loop labels | Optional on `arc.loop` | C++ break/continue always target innermost loop. Labels reserved for future extensions. |

## Architecture

```
Clang AST
    |  Lowering (structural translation)
    v
arc.loop           Mutable state, four optional regions, arc.break/arc.continue
    |  LoopContractPass (lightweight MLIR pass)
    v
arc.loop           Enriched with auto-inferred variant and computed assigns
    |  WhyML Emitter
    v
WhyML              All loops -> recursive functions
```

## Arc Dialect Changes

### `arc.loop` Operation

Single operation covers all loop types. Direct structural mapping from C++ AST using existing `arc.var`/`arc.assign` ops.

**For loop:**

```mlir
// for (int32_t i = 0; i < n; i++) { sum = sum + i; }
arc.loop { condition_first = true,
           invariant = "sum >= 0 && i >= 0 && i <= n",
           variant = "n - i",
           assigns = "i, sum" }
init {
  %i = arc.var "i" = %c0 : !arc.int<32, true>
  arc.yield
}
cond {
  %cond = arc.cmp lt, %i, %n : !arc.bool
  arc.condition %cond
}
update {
  %next = arc.add %i, %c1 : !arc.int<32, true>
  arc.assign %i, %next
  arc.yield
}
body {
  %val = arc.add %sum, %i : !arc.int<32, true>
  arc.assign %sum, %val
  arc.yield
}
```

**While loop:**

```mlir
// while (x > 0) { x = x / 2; }
arc.loop { condition_first = true,
           invariant = "x >= 0",
           variant = "x",
           assigns = "x" }
cond {
  %cond = arc.cmp gt, %x, %c0 : !arc.bool
  arc.condition %cond
}
body {
  %half = arc.div %x, %c2 : !arc.int<32, true>
  arc.assign %x, %half
  arc.yield
}
```

**Do-while loop:**

```mlir
// do { x = x / 2; } while (x > 0);
arc.loop { condition_first = false,
           invariant = "x >= 0",
           variant = "x",
           assigns = "x" }
cond {
  %cond = arc.cmp gt, %x, %c0 : !arc.bool
  arc.condition %cond
}
body {
  %half = arc.div %x, %c2 : !arc.int<32, true>
  arc.assign %x, %half
  arc.yield
}
```

**Loop with break:**

```mlir
// for (...) { if (cond) break; ... }
arc.loop { condition_first = true, assigns = "i, result" }
init {
  %i = arc.var "i" = %c0 : !arc.int<32, true>
  arc.yield
}
cond {
  %cond = arc.cmp lt, %i, %n : !arc.bool
  arc.condition %cond
}
update {
  %next = arc.add %i, %c1 : !arc.int<32, true>
  arc.assign %i, %next
  arc.yield
}
body {
  %found = arc.cmp eq, %i, %target : !arc.bool
  arc.if %found {
    arc.assign %result, %i
    arc.break
  }
  arc.yield
}
```

### `arc.loop` Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `condition_first` | `bool` | `true` for `for`/`while`; `false` for `do-while` |
| `invariant` | `string` (optional) | Loop invariant expression (multiple conjoined with `&&`) |
| `variant` | `string` (optional) | Termination measure (auto-inferred for counted for-loops) |
| `assigns` | `string` (optional) | Comma-separated modified variable names (auto-computed if absent) |
| `label` | `string` (optional) | Loop label for future extensions |

### `arc.loop` Regions

| Region | Present For | Terminator | Description |
|--------|------------|------------|-------------|
| `init` | `for` only | `arc.yield` | Loop variable initialization |
| `cond` | all loops | `arc.condition` | Loop condition evaluation |
| `update` | `for` only | `arc.yield` | Loop variable update (i++) |
| `body` | all loops | `arc.yield` | Loop body (may contain arc.break/arc.continue) |

### Supporting Operations

| Op | Description |
|----|-------------|
| `arc.break` | Early loop exit. Valid only inside `arc.loop` body. |
| `arc.continue` | Skip to next iteration. Valid only inside `arc.loop` body. |
| `arc.condition` | Condition region terminator. Takes `!arc.bool` operand. |
| `arc.yield` | Region terminator for init, update, and body regions. |

## Pipeline Changes by Stage

### Stage 1: Clang Frontend

No changes. Already parses all C++ constructs into AST.

### Stage 2: Subset Enforcer

- Un-reject `for`, `while`, `do-while` loops (currently rejected at SubsetEnforcer.cpp lines 86-113)
- Allow `break` and `continue` statements
- New constraint: loops should have `loop_invariant` annotation (warning if missing)
- New constraint: `while`/`do-while` require explicit `loop_variant` (counted `for` can auto-infer)
- New constraint: `break`/`continue` only valid inside loops
- Validate `//@ label` uniqueness within a function

### Stage 3: Contract Parser

New annotations:

| Annotation | Parsing | Notes |
|------------|---------|-------|
| `//@ loop_invariant: <expr>` | Same as `requires`/`ensures` (boolean expression) | Multiple lines conjoined with `&&` |
| `//@ loop_variant: <expr>` | Arithmetic expression (non-negative integer) | Single expression |
| `//@ loop_assigns: var1, var2` | Comma-separated identifier list | Simple tokenization |
| `//@ label: <identifier>` | Single identifier | Must precede a loop |

Placement: loop annotations must appear immediately before the loop statement.

### Stage 4: Lowering (Clang AST -> arc.loop)

- Lower `ForStmt` -> `arc.loop` with init/cond/update/body regions, `condition_first = true`
- Lower `WhileStmt` -> `arc.loop` with cond/body regions, `condition_first = true`
- Lower `DoStmt` -> `arc.loop` with cond/body regions, `condition_first = false`
- Lower `BreakStmt` -> `arc.break`
- Lower `ContinueStmt` -> `arc.continue`
- Attach `loop_invariant`/`loop_variant`/`loop_assigns`/`label` as string attributes on `arc.loop`
- Uses existing `arc.var`/`arc.assign` for mutable variable handling

### Stage 5: MLIR Passes

New: `LoopContractPass` (lightweight)

1. **Auto-infer `variant`** for counted for-loops:
   - Pattern-match: init sets var to lo, cond compares var < hi, update increments var by 1 -> variant = `hi - var`
   - Pattern-match: init sets var to hi, cond compares var > lo, update decrements var by 1 -> variant = `var - lo`
   - If pattern not recognized and no user-provided variant -> emit diagnostic
2. **Auto-compute `assigns`** if not user-provided:
   - Walk loop body and collect all `arc.assign` target variable names
   - Set `assigns` attribute with the computed list
3. **Validate** loop contracts:
   - `invariant` should be present (warn if missing)
   - `variant` must be present after auto-inference (error if missing for while/do-while)

### Stage 6: WhyML Emitter

All loops emit as **recursive WhyML functions**.

#### Emission model

For each `arc.loop`:
1. Collect modified variables from `assigns` attribute -> these become function parameters
2. Generate a recursive function:
   - Parameters: all modified variables (from `assigns`)
   - `requires` clause: loop invariant
   - `variant` clause: loop variant
   - Body: condition check (if-then-else), loop body, recursive call
3. Emit the initial call with starting values

#### For-loop emission pattern

```
C++:  for (int32_t i = 0; i < n; i++) { sum = sum + i; }

WhyML:
  let rec loop_name (i: int) (sum: int) : int
    requires { invariant(i, sum) }
    variant  { n - i }
  = if i < n then
      let sum' = sum + i in       (* body *)
      loop_name (i + 1) sum'      (* update + recurse *)
    else
      sum                         (* loop exit: return final value *)

  (* initial call *)
  loop_name 0 0
```

#### While-loop emission pattern

```
C++:  while (x > 0) { x = x / 2; }

WhyML:
  let rec loop_name (x: int) : int
    requires { invariant(x) }
    variant  { x }
  = if x > 0 then                (* condition *)
      let x' = div x 2 in        (* body *)
      loop_name x'                (* recurse *)
    else
      x                           (* loop exit *)

  loop_name x_init
```

#### Do-while emission pattern

```
C++:  do { x = x / 10; count++; } while (x > 0);

WhyML:
  let rec loop_name (x: int) (count: int) : (int, int)
    requires { invariant(x, count) }
    variant  { x }
  = let x' = div x 10 in         (* body first *)
    let count' = count + 1 in
    if x' > 0 then                (* then condition *)
      loop_name x' count'         (* recurse *)
    else
      (x', count')                (* loop exit *)

  loop_name x_init 0
```

#### Break emission pattern

```
break -> return from recursive function (the current values)

WhyML:
  if break_condition then
    current_values                (* break: just return *)
  else
    ... continue body ...
    loop_name updated_values      (* recurse *)
```

#### Continue emission pattern

```
continue -> skip remaining body, jump to recursive call

WhyML:
  if continue_condition then
    loop_name current_values      (* continue: recurse with current state *)
  else
    ... rest of body ...
    loop_name updated_values      (* normal iteration *)
```

#### Multiple return values

When a loop modifies multiple variables, the recursive function returns a tuple:

```whyml
let rec loop_name (i: int) (sum: int) (count: int) : (int, int, int)
  ...
= ...
  loop_name (i + 1) sum' count'
  ...

let (final_i, final_sum, final_count) = loop_name 0 0 0 in
```

### Stage 7: Why3 Runner

No changes.

### Stage 8: Report Generator

No changes.

## End-to-End Examples

### Example 1: Simple accumulation (for loop)

**C++ input:**

```cpp
//@ requires: n >= 0 && n <= 1000
//@ ensures: \result >= 0
int32_t sum_to_n(int32_t n) {
    int32_t sum = 0;
    //@ loop_invariant: sum >= 0 && sum <= i * 1000
    //@ loop_invariant: i >= 0 && i <= n
    //@ loop_assigns: i, sum
    for (int32_t i = 0; i < n; i++) {
        sum = sum + i;
    }
    return sum;
}
```

**Arc MLIR (arc.loop):**

```mlir
arc.func @sum_to_n(%n: !arc.int<32,true>) -> !arc.int<32,true>
    requires = "n >= 0 && n <= 1000"
    ensures  = "\\result >= 0"
{
  %sum = arc.var "sum" = %c0 : !arc.int<32,true>
  arc.loop { condition_first = true,
             invariant = "sum >= 0 && sum <= i * 1000 && i >= 0 && i <= n",
             variant = "n - i",
             assigns = "i, sum" }
  init {
    %i = arc.var "i" = %c0 : !arc.int<32,true>
    arc.yield
  }
  cond {
    %cond = arc.cmp lt, %i, %n : !arc.bool
    arc.condition %cond
  }
  update {
    %i_next = arc.add %i, %c1 : !arc.int<32,true>
    arc.assign %i, %i_next
    arc.yield
  }
  body {
    %new_sum = arc.add %sum, %i : !arc.int<32,true>
    arc.assign %sum, %new_sum
    arc.yield
  }
  arc.return %sum
}
```

**WhyML output:**

```whyml
module SumToN
  use int.Int

  let sum_to_n (n: int) : int
    requires { 0 <= n /\ n <= 1000 }
    ensures  { result >= 0 }
  =
    let rec loop_sum (i: int) (sum: int) : int
      requires { sum >= 0 /\ sum <= i * 1000 /\ 0 <= i /\ i <= n }
      variant  { n - i }
    = if i < n then
        let sum' = sum + i in
        loop_sum (i + 1) sum'
      else
        sum
    in
    loop_sum 0 0
end
```

**Expected verification output:**

```
[PASS]  input.cpp:sum_to_n    3/3 obligations proven (0.5s)
        - loop invariant preserved
        - loop invariant on entry
        - postcondition

Summary: 1 passed, 0 failed, 0 timeout
```

### Example 2: While loop (halving)

**C++ input:**

```cpp
//@ requires: x > 0
//@ ensures: \result == 0
int32_t halve_to_zero(int32_t x) {
    //@ loop_invariant: x >= 0
    //@ loop_variant: x
    //@ loop_assigns: x
    while (x > 0) {
        x = x / 2;
    }
    return x;
}
```

**WhyML output:**

```whyml
module HalveToZero
  use int.Int
  use int.ComputerDivision

  let halve_to_zero (x_param: int) : int
    requires { x_param > 0 }
    ensures  { result = 0 }
  =
    let rec loop_halve (x: int) : int
      requires { x >= 0 }
      variant  { x }
    = if x > 0 then
        let x' = div x 2 in
        loop_halve x'
      else
        x
    in
    loop_halve x_param
end
```

### Example 3: Loop with break

**C++ input:**

```cpp
//@ requires: n > 0 && n <= 100
//@ ensures: \result >= 0
int32_t find_first_even(int32_t n) {
    int32_t result = -1;
    //@ loop_invariant: i >= 0 && i <= n
    //@ loop_assigns: i, result
    for (int32_t i = 0; i < n; i++) {
        if (i % 2 == 0) {
            result = i;
            break;
        }
    }
    return result;
}
```

**WhyML output:**

```whyml
module FindFirstEven
  use int.Int
  use int.ComputerDivision

  let find_first_even (n: int) : int
    requires { n > 0 /\ n <= 100 }
    ensures  { 0 <= result }
  =
    let rec loop_find (i: int) (result: int) : int
      requires { 0 <= i /\ i <= n }
      variant  { n - i }
    = if i < n then
        if mod i 2 = 0 then
          i                              (* break: return value *)
        else
          loop_find (i + 1) result       (* next iteration *)
      else
        result
    in
    loop_find 0 (-1)
end
```

### Example 4: Do-while loop

**C++ input:**

```cpp
//@ requires: x > 0 && x <= 1000
//@ ensures: \result >= 1
int32_t count_digits(int32_t x) {
    int32_t count = 0;
    //@ loop_invariant: count >= 0 && x >= 0
    //@ loop_variant: x
    //@ loop_assigns: x, count
    do {
        x = x / 10;
        count = count + 1;
    } while (x > 0);
    return count;
}
```

**WhyML output:**

```whyml
module CountDigits
  use int.Int
  use int.ComputerDivision

  let count_digits (x_param: int) : int
    requires { x_param > 0 /\ x_param <= 1000 }
    ensures  { result >= 1 }
  =
    let rec loop_count (x: int) (count: int) : (int, int)
      requires { count >= 0 /\ x >= 0 }
      variant  { x }
    = let x' = div x 10 in
      let count' = count + 1 in
      if x' > 0 then
        loop_count x' count'
      else
        (x', count')
    in
    let (_, final_count) = loop_count x_param 0 in
    final_count
end
```

### Example 5: Nested loops

**C++ input:**

```cpp
//@ requires: n >= 0 && n <= 100
//@ ensures: \result >= 0
int32_t sum_triangle(int32_t n) {
    int32_t sum = 0;
    //@ loop_invariant: i >= 0 && i <= n && sum >= 0
    //@ loop_assigns: i, sum
    for (int32_t i = 0; i < n; i++) {
        //@ loop_invariant: j >= 0 && j <= i
        //@ loop_assigns: j, sum
        for (int32_t j = 0; j <= i; j++) {
            sum = sum + 1;
        }
    }
    return sum;
}
```

**WhyML output:**

```whyml
module SumTriangle
  use int.Int

  let sum_triangle (n: int) : int
    requires { 0 <= n /\ n <= 100 }
    ensures  { result >= 0 }
  =
    let rec loop_outer (i: int) (sum: int) : int
      requires { 0 <= i /\ i <= n /\ sum >= 0 }
      variant  { n - i }
    = if i < n then
        let rec loop_inner (j: int) (sum_inner: int) : int
          requires { 0 <= j /\ j <= i }
          variant  { i + 1 - j }
        = if j <= i then
            loop_inner (j + 1) (sum_inner + 1)
          else
            sum_inner
        in
        let sum' = loop_inner 0 sum in
        loop_outer (i + 1) sum'
      else
        sum
    in
    loop_outer 0 0
end
```

### Example 6: While loop with continue

**C++ input:**

```cpp
//@ requires: n > 0 && n <= 100
//@ ensures: \result >= 0
int32_t sum_odd(int32_t n) {
    int32_t sum = 0;
    int32_t i = 0;
    //@ loop_invariant: i >= 0 && i <= n && sum >= 0
    //@ loop_variant: n - i
    //@ loop_assigns: i, sum
    while (i < n) {
        i = i + 1;
        if (i % 2 == 0) {
            continue;
        }
        sum = sum + i;
    }
    return sum;
}
```

**WhyML output:**

```whyml
module SumOdd
  use int.Int
  use int.ComputerDivision

  let sum_odd (n: int) : int
    requires { n > 0 /\ n <= 100 }
    ensures  { result >= 0 }
  =
    let rec loop_sum (i: int) (sum: int) : (int, int)
      requires { 0 <= i /\ i <= n /\ sum >= 0 }
      variant  { n - i }
    = if i < n then
        let i' = i + 1 in
        if mod i' 2 = 0 then
          loop_sum i' sum                (* continue: skip rest *)
        else
          let sum' = sum + i' in
          loop_sum i' sum'               (* normal iteration *)
      else
        (i, sum)
    in
    let (_, final_sum) = loop_sum 0 0 in
    final_sum
end
```

## Updated Incremental Widening Plan

| Slice | Features | Primary Stages Affected |
|-------|----------|------------------------|
| 1 | `int32_t`, `bool`, `if/else`, arithmetic, `requires`/`ensures`, `\result` | All (initial wiring) |
| 2 | All integer types (`i8`-`i64`, `u8`-`u64`), `static_cast`, overflow modes | Dialect types, WhyML emitter, Subset Enforcer |
| **3** | **`for`/`while`/`do-while`, `break`/`continue`, `loop_invariant`/`loop_variant`/`loop_assigns`, labels, auto-inference, LoopContractPass, recursive WhyML emission** | **Lowering, Dialect ops, MLIR Passes, WhyML emitter, Contract Parser, Subset Enforcer** |
| 4 | Function calls (non-recursive), modular verification with callee contracts | Lowering, WhyML emitter |
| 5 | `std::array<T,N>`, `std::span<T>`, bounds checking, `\forall`/`\exists`, range-based `for` | Dialect types/ops, WhyML emitter, Contract Parser |
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
