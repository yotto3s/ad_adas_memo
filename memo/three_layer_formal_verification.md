# Three-Layer Formal Verification for Automotive Software: Detailed Description

## Overview

Ensuring the safety of automotive software — particularly for AD/ADAS functions — requires more than just testing. A mathematically rigorous approach decomposes the problem into three distinct layers of proof, each addressing a fundamentally different question:

| Layer | Question | Discipline |
|---|---|---|
| **Layer 1** | Does the algorithm guarantee safety in the physical world? | Control theory + formal methods |
| **Layer 2** | Does the code faithfully implement the algorithm? | Software verification |
| **Layer 3** | Is the code free from implementation-level defects? | Static analysis + abstract interpretation |

Each layer produces a formal guarantee, and these guarantees chain together to form an end-to-end safety argument from physics to executable code. The following sections describe each layer in detail.

---

## Layer 1: Proving That the Algorithm Is Correct

### 1.1 What This Layer Addresses

Layer 1 answers the question: **"If the control algorithm is executed perfectly, will the vehicle always remain safe?"**

This is a question about the mathematical relationship between the control logic and the physical world. It has nothing to do with code, compilers, or processors. Instead, it operates on an idealized model of the algorithm and a mathematical model of the vehicle and its environment.

Examples of safety properties proved at this layer:

- The vehicle's lateral position never exceeds the lane boundaries (no lane departure).
- The distance to the preceding vehicle never falls below a safe following distance (no rear-end collision).
- During a lane change maneuver, the ego vehicle never occupies the same space as another vehicle.
- The vehicle always comes to a complete stop before reaching a stop line when the traffic signal is red.

### 1.2 Prerequisites and Inputs

Before any proof can begin, the following artifacts must be prepared:

**1.2.1 Formal Safety Property (φ)**

The safety property must be stated as a precise mathematical formula, not as a natural-language requirement. For example, the requirement "the vehicle shall stay in its lane" becomes:

```
∀t ≥ 0 : y_min ≤ y(t) ≤ y_max
```

where `y(t)` is the vehicle's lateral position and `[y_min, y_max]` defines the lane boundaries. In temporal logic, this might be written as:

```
□ (y_min ≤ y ≤ y_max)
```

where `□` is the "always" operator. For more complex properties, Signal Temporal Logic (STL) allows quantitative reasoning:

```
□[0,T] (|y(t) - y_center| < w/2)
```

stating that for all time within horizon T, the deviation from lane center is less than half the lane width.

**1.2.2 Plant Model (Physical Dynamics)**

A mathematical model of the vehicle's physical behavior is required. This is typically expressed as a system of ordinary differential equations (ODEs). For lateral dynamics, a common model is the bicycle model:

```
ẏ   = v · sin(ψ)
ψ̇   = v / L · tan(δ)
```

where `y` is lateral position, `ψ` is heading angle, `v` is longitudinal velocity, `L` is wheelbase, and `δ` is steering angle. More detailed models may include tire slip, suspension dynamics, and road surface conditions.

The fidelity of this model is a critical concern: the proof is only as strong as the model's accuracy. Assumptions about the model's validity range (e.g., "valid for lateral acceleration ≤ 0.3g") become explicit preconditions of the proof.

**1.2.3 Algorithm Description (Abstract Control Law)**

The control algorithm is described at a mathematical level — not as code, but as a control law or decision policy. For example, a simple lane-keeping controller might be:

```
δ = -K_p · e_y - K_d · ė_y
```

where `e_y` is lateral error and `K_p`, `K_d` are proportional and derivative gains. For more complex algorithms, the description might include mode switches (e.g., "if lateral error exceeds threshold, switch to emergency lane centering"), which creates a hybrid system with both continuous dynamics and discrete transitions.

**1.2.4 Environment Assumptions**

The proof must make explicit what is assumed about the environment: road curvature bounds, friction coefficient ranges, behavior of other traffic participants, sensor measurement bounds, communication delays, etc. These assumptions define the operational design domain (ODD) for which the safety guarantee holds.

### 1.3 Proof Methods

**1.3.1 Hybrid Systems Verification with Differential Dynamic Logic (dL)**

This is currently the most complete framework for proving safety of systems that combine discrete control decisions with continuous physical dynamics.

The algorithm and plant model are expressed as a *hybrid program* in dL. A hybrid program interleaves discrete assignments and tests (the control logic) with continuous evolution along differential equations (the physics). The safety property is expressed as a dL formula of the form:

```
φ_init → [α*] φ_safe
```

meaning: "If the initial condition φ_init holds, then after any number of iterations of hybrid program α, the safety property φ_safe still holds."

The proof proceeds by finding a *loop invariant* — a property that (a) is implied by the initial condition, (b) is preserved by each iteration of the control loop, and (c) implies the safety property. For the continuous dynamics part, *differential invariants* are used: these are formulas that can be shown to hold throughout the evolution of a differential equation without solving the equation explicitly.

**Tool**: KeYmaera X (https://keymaerax.org/)

KeYmaera X is the primary theorem prover for dL. It combines automated proof search tactics with interactive proof guidance. Its soundness-critical core is only ~2000 lines of code, which makes the trusted computing base very small. Case studies include:

- Adaptive cruise control: proving safe following distance is maintained
- Highway driving: proving collision freedom on an infinite time horizon, including with neural network controllers
- Intersection management: proving vehicles do not enter the intersection when the light is red

KeYmaera X also implements *ModelPlex*, which automatically derives runtime monitor conditions from offline proofs (used in Layer 2, Approach C).

**1.3.2 Reachability Analysis**

Instead of proving a general invariant, reachability analysis directly computes the set of all states the system can reach from a given set of initial states. If the computed reachable set does not intersect the unsafe region, safety is established.

For linear systems, reachable sets can be computed exactly using zonotopes, polytopes, or ellipsoids. For nonlinear systems, over-approximations are computed using techniques such as:

- **Taylor model flow-pipe construction**: Computes tight enclosures of solutions of nonlinear ODEs
- **Abstraction to linear dynamics**: Linearizes the system within small regions and aggregates the results
- **Hamilton-Jacobi reachability**: Solves a PDE (Hamilton-Jacobi-Isaacs equation) to compute backward reachable sets, which represent all states from which the system can be driven into the unsafe set by worst-case disturbances

For the lane-keeping example: the analysis computes the set of all possible lateral positions the vehicle can occupy given the controller, vehicle dynamics, and bounded disturbances. If this set is contained within [y_min, y_max] for all time, lane keeping is proven.

**Tools**: CORA (TU Munich), SpaceEx, Flow*, JuliaReach, hj_reachability

**1.3.3 Barrier Certificates**

A barrier certificate `B(x)` is a scalar function of the system state that acts as a "wall" in state space separating safe trajectories from the unsafe region. The requirements are:

1. `B(x) ≤ 0` for all initial states `x ∈ X_0`
2. `B(x) > 0` for all unsafe states `x ∈ X_unsafe`
3. `dB/dt ≤ 0` along all system trajectories (the barrier is never crossed)

If such a function exists, safety is guaranteed without computing reachable sets. For polynomial systems, barrier certificates can be searched for using Sum of Squares (SOS) programming, which reduces to semidefinite programming (SDP) — a convex optimization problem that can be solved efficiently.

**Control Barrier Functions (CBFs)** extend this concept to control systems. A CBF not only proves safety but also synthesizes a safe controller: at each time step, the control input is chosen to satisfy the barrier condition, typically via a quadratic program (QP) that is solved in real time.

**Tools**: SOSTOOLS, DSOS/SDSOS (polynomial optimization), dReal (SMT solver for nonlinear arithmetic)

**1.3.4 RSS (Responsibility-Sensitive Safety)**

RSS takes a different approach: instead of verifying a specific algorithm, it defines a *safety envelope* — a set of formal rules that any algorithm must satisfy. The rules are derived from traffic regulations and common-sense driving principles, formalized as mathematical inequalities involving distances, velocities, reaction times, and braking capabilities.

For example, the longitudinal safe distance rule states:

```
d_min = v_r · ρ + (v_r²) / (2·a_min_brake) - (v_f²) / (2·a_max_brake)
```

where `v_r` is the rear vehicle's velocity, `v_f` is the front vehicle's velocity, `ρ` is the response time, and `a_min_brake` / `a_max_brake` are worst-case braking decelerations. If the ego vehicle always maintains at least `d_min` distance, longitudinal collision freedom is guaranteed.

RSS is particularly relevant for Layer 1 because it reduces the algorithm-level safety question to a parameter-level question: "Does the algorithm's output always satisfy the RSS constraints?"

### 1.4 Deliverables from Layer 1

1. **Formal proof of safety** — a machine-checked proof that the algorithm satisfies the safety property
2. **Explicit assumption list** — every assumption about the physical model, environment, sensor accuracy, timing, etc.
3. **Validity envelope** — the set of conditions under which the proof holds (e.g., vehicle speed ≤ 130 km/h, road curvature ≤ 0.01 m⁻¹, sensor latency ≤ 50 ms)
4. **Derived constraints on the implementation** — requirements that must be met by the software (e.g., control loop period ≤ 10 ms, steering angle command precision ≤ 0.1°), which become the input specifications for Layer 2

---

## Layer 2: Proving That the Software Implements the Algorithm Correctly

### 2.1 What This Layer Addresses

Layer 2 answers the question: **"Does the executable code faithfully realize the behavior specified by the algorithm?"**

Even if the algorithm is provably safe (Layer 1), the implementation could deviate due to:

- Logical errors in translating the algorithm to code (e.g., wrong sign, missing case)
- Numerical issues (e.g., floating-point rounding changes the behavior of a comparison)
- Timing discrepancies (e.g., the code does not execute within the assumed control period)
- Concurrency issues (e.g., race conditions in reading sensor data)
- Missing or incorrect handling of edge cases (e.g., sensor returns NaN)

Layer 2 bridges the gap between the mathematical world of Layer 1 and the software world of Layer 3.

### 2.2 Prerequisites and Inputs

**2.2.1 Software-Level Formal Specification**

The abstract algorithm description from Layer 1 must be translated into a form that can be compared against code. This typically takes the form of *function contracts*:

- **Preconditions**: What must be true when a function is called (e.g., "lateral_error is in [-0.5, 0.5] meters")
- **Postconditions**: What must be true when a function returns (e.g., "steering_angle is in [-5.0, 5.0] degrees")
- **Loop invariants**: What must be true at every iteration of a loop
- **Data invariants**: What must always be true about data structures (e.g., "the trajectory buffer always contains at least 3 points")

These specifications encode the constraints derived from Layer 1. For example, if Layer 1 proved that the algorithm is safe provided `|δ| ≤ δ_max`, then the function that computes steering commands must have `|return_value| ≤ δ_max` as a postcondition.

**2.2.2 Source Code**

The actual C/C++ (or Ada, Rust, etc.) source code of the control software module.

**2.2.3 Traceability Matrix**

A mapping from Layer 1 assumptions and constraints to specific code locations and specifications. This ensures no assumption is "lost in translation."

### 2.3 Proof Methods

**2.3.1 Approach A: Deductive Verification (Hoare Logic-Based)**

This is the most direct approach. The code is annotated with formal specifications (preconditions, postconditions, loop invariants) expressed in a specification language. A verification condition generator (VCGen) then produces a set of mathematical proof obligations. If all proof obligations are discharged (typically by SMT solvers), the code is proven correct with respect to its specification.

**For C code — Frama-C with the WP (Weakest Precondition) plugin:**

The specification is written in ACSL (ANSI/ISO C Specification Language), embedded as comments in the C source:

```c
/*@ requires -0.5 <= lateral_error <= 0.5;
    requires -1.0 <= lateral_error_rate <= 1.0;
    requires Kp > 0 && Kd > 0;
    ensures -5.0 <= \result <= 5.0;
    assigns \nothing;
*/
double compute_steering(double lateral_error, double lateral_error_rate,
                        double Kp, double Kd) {
    double command = -Kp * lateral_error - Kd * lateral_error_rate;
    if (command > 5.0) command = 5.0;
    if (command < -5.0) command = -5.0;
    return command;
}
```

WP generates proof obligations from these annotations and attempts to discharge them using SMT solvers (Alt-Ergo, Z3, CVC5). If the proof succeeds, it is mathematically guaranteed that the function's output always satisfies the postcondition whenever the precondition holds.

For floating-point arithmetic, special care is needed. Frama-C can reason about IEEE 754 floating-point semantics, but the proofs become significantly more complex. The gap between real-number arithmetic (used in Layer 1) and floating-point arithmetic (used in the implementation) must be explicitly bounded.

**For Ada/SPARK — SPARK Pro:**

SPARK is a subset of Ada specifically designed for formal verification. Contracts are part of the language syntax:

```ada
function Compute_Steering (Lateral_Error      : Float;
                           Lateral_Error_Rate : Float;
                           Kp, Kd             : Float) return Float
  with Pre  => Lateral_Error in -0.5 .. 0.5
           and Lateral_Error_Rate in -1.0 .. 1.0
           and Kp > 0.0 and Kd > 0.0,
       Post => Compute_Steering'Result in -5.0 .. 5.0;
```

The SPARK toolchain (GNATprove) performs both flow analysis (checking for uninitialized variables, data flow errors) and proof (checking contracts). SPARK's advantage is that the language restrictions make verification more tractable, and the tools are mature enough for industrial use.

**2.3.2 Approach B: Refinement-Based Development**

Instead of writing code and then proving it correct, refinement-based methods start from an abstract specification and systematically derive an implementation through a series of mathematically justified refinement steps. Each step is proven to preserve the properties of the previous level.

**B-Method / Event-B:**

The process begins with an abstract machine that specifies the algorithm at a high level (corresponding closely to the Layer 1 description). This is then refined through multiple intermediate levels, each adding implementation detail:

```
Abstract Machine (= algorithm specification)
    ↓ Refinement 1 (introduce data structures)
    ↓ Refinement 2 (introduce loops, concrete algorithms)
    ↓ Refinement 3 (introduce implementation-level details)
Implementation (= code)
```

At each refinement step, proof obligations are generated and discharged to verify that the refinement is correct (i.e., the concrete level simulates the abstract level). The final implementation can be automatically translated to code.

This approach has been used successfully in railway signaling (e.g., Paris Métro Line 14) and has relevance for automotive systems where the highest assurance is required.

**2.3.3 Approach C: Certified Runtime Monitors (ModelPlex)**

When full deductive verification of the entire codebase is impractical, an alternative is to verify a *runtime monitor* that checks at each control cycle whether the system's actual behavior is consistent with the assumptions of the Layer 1 proof.

KeYmaera X's ModelPlex technology works as follows:

1. From the Layer 1 dL proof, ModelPlex automatically derives a *monitor condition* — an arithmetic formula over the system's observable variables.
2. This monitor condition is proven (as part of the Layer 1 proof) to have the property: "If the monitor condition is satisfied at each control step, then the offline safety proof applies to the actual system execution."
3. The monitor condition is implemented as a simple runtime check (just arithmetic comparisons and basic operations).
4. If the monitor detects a violation, the system switches to a pre-verified safe fallback controller.

The key advantage is that only the monitor code (which is much simpler than the full control software) needs to be verified at Layer 2, rather than the entire control algorithm implementation. The monitor condition itself is typically a conjunction of inequalities, making it straightforward to implement and verify.

**2.3.4 Approach D: Translation Validation**

When code is auto-generated from models (e.g., Simulink/Stateflow → C via Embedded Coder), translation validation verifies that the generated code is semantically equivalent to the model. This is done by:

1. Extracting a formal model from the generated code
2. Extracting a formal model from the Simulink model
3. Proving equivalence (or a refinement relation) between the two

This is complementary to Tool Qualification (ISO 26262 Part 8, Clause 11): tool qualification argues that the code generator is unlikely to introduce errors based on process evidence, while translation validation provides a mathematical proof for each specific generated output.

### 2.4 Handling the Real-to-Float Gap

A subtle but critical concern in Layer 2 is the gap between real-number arithmetic (used in Layer 1 proofs) and floating-point arithmetic (used in actual code). A control law that is safe over the reals may become unsafe when implemented in IEEE 754 floating-point due to rounding errors.

Approaches to handle this include:

- **Absorb the gap into Layer 1**: Prove the algorithm safe with a margin that accounts for worst-case floating-point error. This requires bounding the accumulated rounding error through the entire computation.
- **Prove floating-point properties in Layer 2**: Use tools that reason about IEEE 754 semantics (Frama-C with its floating-point model, Gappa for bounding floating-point errors) to show that the floating-point implementation stays within the bounds assumed by Layer 1.
- **Use interval arithmetic**: Implement the control law using interval arithmetic, which tracks rounding errors explicitly and always produces a sound enclosure of the real result.

### 2.5 Deliverables from Layer 2

1. **Proof of functional correctness** — each function/module satisfies its contract
2. **Traceability report** — mapping from Layer 1 constraints to code-level specifications to proof results
3. **Floating-point error analysis** — bounding the deviation between real and floating-point computations
4. **Input ranges for Layer 3** — the preconditions that Layer 3 must show are sufficient to prevent runtime errors

---

## Layer 3: Proving the Absence of Runtime Errors

### 3.1 What This Layer Addresses

Layer 3 answers the question: **"Regardless of what the algorithm is supposed to do, will the code ever crash, produce undefined behavior, or corrupt memory?"**

This layer is concerned with *implementation robustness*: the code must be free from defects that could cause it to behave in ways not predicted by any model. In C and C++ (the dominant languages in automotive embedded software), undefined behavior is particularly dangerous because the compiler is free to do anything — including silently generating code that appears to work in testing but fails catastrophically in the field.

### 3.2 Target Defect Classes

The specific defect classes targeted by Layer 3 analysis include:

**3.2.1 Arithmetic Errors**
- **Integer overflow/underflow**: A 16-bit signed integer wrapping from 32767 to -32768 can invert a control command. In C, signed integer overflow is undefined behavior.
- **Division by zero**: If a divisor can ever be zero (e.g., dividing by a time delta that could be zero on the first cycle), the behavior is undefined.
- **Floating-point exceptions**: Overflow to infinity, operations producing NaN, loss of significance in subtraction of nearly equal values.

**3.2.2 Memory Errors**
- **Buffer overflow / out-of-bounds array access**: Writing past the end of an array can corrupt adjacent data, including safety-critical variables. This is also a major security vulnerability.
- **Null pointer dereference**: Accessing memory through a pointer that has not been initialized or has been freed.
- **Dangling pointer**: Using a pointer after the memory it points to has been freed or gone out of scope.
- **Use of uninitialized variables**: Reading a variable before it has been assigned a value; the result is indeterminate.
- **Stack overflow**: Exceeding the available stack space, typically due to deep recursion or large local variables.

**3.2.3 Concurrency Errors (for multi-threaded systems)**
- **Data races**: Two threads accessing the same variable without synchronization, with at least one write. In C11/C++11, this is undefined behavior.
- **Deadlocks**: Two or more threads waiting for each other to release resources, resulting in permanent blocking.
- **Priority inversion**: A high-priority task is blocked by a low-priority task holding a shared resource.

**3.2.4 Type and Conversion Errors**
- **Invalid type casts**: Casting a pointer to an incompatible type, violating strict aliasing rules.
- **Implicit narrowing conversions**: Assigning a 32-bit value to a 16-bit variable, silently truncating the value.

### 3.3 Proof Methods

**3.3.1 Abstract Interpretation**

Abstract interpretation is the most mature and industrially deployed technique for Layer 3 proofs. It works by computing an over-approximation of all possible values that every variable can take at every program point. If the over-approximation shows that no variable can ever take a value that would trigger a runtime error, the absence of that error class is mathematically proven.

The term "over-approximation" is key: abstract interpretation may report that an error is *possible* when it is not (false alarm / false positive), but it will never fail to report an error that can actually occur (no false negatives). This property is called *soundness*.

**Astrée:**

Astrée is designed specifically for safety-critical embedded C code. It is parametric, meaning the user can trade analysis speed for precision. Key features:

- Sound handling of IEEE 754 floating-point arithmetic, including rounding modes
- Precise analysis of finite state machines and digital filters (common in automotive control software)
- Support for concurrent code with priority-based scheduling (OSEK/AUTOSAR-style)
- Modular analysis with abstract domains (intervals, octagons, polyhedra, decision trees)

Industrial results: Airbus used Astrée to prove zero runtime errors in the A380 fly-by-wire software. Bosch adopted Astrée group-wide after a pilot project at Bosch Automotive Steering. ESA used Astrée to verify the docking software of the Jules Verne ATV.

Astrée can typically analyze 100,000+ lines of embedded C code and produce results with a false alarm rate of 10-20% of checked operations. Remaining alarms must be manually reviewed or addressed by code modifications.

**Polyspace (MathWorks):**

Polyspace Bug Finder performs pattern-based static analysis (similar to a linter on steroids), while Polyspace Code Prover performs abstract interpretation-based verification. Code Prover can prove code operations as:

- **Green** (proven safe): The operation cannot fail for any execution path
- **Red** (proven unsafe): The operation will always fail
- **Orange** (unproven): The tool cannot determine safety (may be a real bug or a false alarm)
- **Gray** (dead code): The operation is unreachable

The goal is to make every operation green. Orange operations require investigation — either the code needs to be fixed, or the tool's configuration needs to be refined to provide more precise analysis.

Polyspace integrates well with MATLAB/Simulink workflows, which makes it particularly natural for automotive projects using Model-Based Development.

**TrustInSoft Analyzer:**

Based on the Frama-C framework, TrustInSoft Analyzer performs "exhaustive" analysis of C code. It has been qualified as an ISO 26262 tool by TÜV SÜD. Its approach emphasizes the ability to provide a mathematical guarantee of the absence of undefined behaviors, with fine-grained control over the analysis context (input ranges, memory layout, etc.).

**3.3.2 Bounded Model Checking for Software**

Tools like CBMC (C Bounded Model Checker) can check C programs for runtime errors by:

1. Unrolling all loops up to a given bound
2. Converting the program to a logical formula (bit-precise, including floating-point)
3. Checking satisfiability using an SMT solver

If the formula is unsatisfiable, no error exists within the given bounds. If satisfiable, the solver provides a concrete counterexample (a specific input that triggers the error).

CBMC is useful for finding deep bugs and for verifying specific functions, but it does not scale to entire codebases as well as abstract interpretation.

**3.3.3 MISRA Compliance Checking**

MISRA C and MISRA C++ are coding guidelines developed specifically for safety-critical embedded systems. They restrict the use of language features that are prone to errors or undefined behavior (e.g., pointer arithmetic, implicit type conversions, recursion).

MISRA compliance is not a proof, but it eliminates many categories of potential defects upfront, making the subsequent abstract interpretation analysis more effective (fewer false alarms, faster convergence).

Tools: Parasoft C/C++test, PC-lint Plus, QA-MISRA, Polyspace Bug Finder, Astrée/RuleChecker

### 3.4 Handling Compiler and Hardware Concerns

Layer 3 typically operates at the source code level. However, the compiler could introduce errors during optimization, and the hardware could have errata. To close this gap:

- **CompCert**: A formally verified C compiler that is proven to produce machine code that faithfully implements the source program's semantics. Using CompCert eliminates the need to trust the compiler. CompCert is qualified for use in ISO 26262 projects.
- **Compiler qualification**: For non-verified compilers (e.g., GCC, LLVM), Tool Qualification per ISO 26262 Part 8 is used, combined with compiler validation test suites.
- **Target-specific analysis**: Tools like AbsInt's aiT and StackAnalyzer perform worst-case execution time (WCET) analysis and stack usage analysis at the binary level, accounting for processor-specific behavior.

### 3.5 Deliverables from Layer 3

1. **Runtime error freedom proof** — a report showing that all code operations are proven safe (or a list of remaining unresolved alarms with justification)
2. **MISRA compliance report** — documenting compliance with coding guidelines, including justified deviations
3. **Verification context** — the input ranges and environmental assumptions under which the proof holds (must be consistent with Layer 2 preconditions)
4. **Tool qualification evidence** — documentation required by ISO 26262 for the tools used

---

## Connecting the Three Layers: The Chain of Guarantees

### The Logical Structure

The end-to-end safety argument has the following logical structure:

```
Theorem: The vehicle does not depart from its lane during operation.

Proof:
  (1) By Layer 1 proof: IF the control law is executed with inputs within
      the specified ranges and at the specified frequency,
      THEN the vehicle's lateral position stays within lane boundaries.
      [Assumptions: sensor error ≤ ±0.1m, control period ≤ 10ms,
       speed ≤ 130 km/h, road curvature ≤ 0.01 m⁻¹]

  (2) By Layer 2 proof: IF the compute_steering() function is called with
      inputs satisfying its preconditions,
      THEN its output satisfies the postconditions, which correspond
      exactly to the control law constraints assumed in (1).
      [Preconditions: lateral_error ∈ [-0.5, 0.5], etc.]

  (3) By Layer 3 proof: IF compute_steering() is called with inputs
      in the ranges specified by its preconditions,
      THEN no undefined behavior occurs during execution, and the
      function terminates normally (returning a valid result as
      specified in Layer 2 postconditions).
      [Verified: no overflow, no division by zero, no out-of-bounds
       access for all inputs in the precondition range]

  Chain: (1) + (2) + (3) → The implementation maintains lane safety
  under the stated operational assumptions. ∎
```

### Assumption Propagation

The most critical aspect of the three-layer approach is ensuring that assumptions flow correctly between layers:

```
Layer 1 assumes:                    Layer 2 must ensure:
─────────────────                   ────────────────────
Sensor error ≤ ±0.1m          →    Input validation checks this bound
Control period ≤ 10ms          →    WCET analysis confirms execution
                                    within budget
|δ| ≤ δ_max in control law    →    Function postcondition guarantees
                                    |output| ≤ δ_max
Real arithmetic                →    Floating-point error bounded within
                                    Layer 1 safety margin

Layer 2 assumes:                    Layer 3 must ensure:
─────────────────                   ────────────────────
Input ranges per precondition  →    No undefined behavior for these
                                    input ranges
Function returns normally      →    No crash, no infinite loop, no
                                    stack overflow
Output satisfies postcondition →    Arithmetic is correctly computed
                                    (no overflow corrupts result)
```

If any link in this chain breaks — for example, if Layer 3 cannot prove the absence of integer overflow for the input ranges specified by Layer 2 — the entire safety argument fails. This typically means either the code must be fixed (add bounds checks, use wider integer types) or the Layer 1 proof must be strengthened (prove safety with a larger margin).

### Gap Analysis Checklist

Before claiming end-to-end safety, verify that the following gaps are addressed:

- [ ] Every Layer 1 assumption about sensor inputs maps to a Layer 2 precondition with bounds checking
- [ ] Every Layer 1 assumption about timing maps to a verified WCET budget
- [ ] Every Layer 1 continuous-time model maps to a Layer 2 discrete-time implementation with bounded discretization error
- [ ] Every Layer 1 real-arithmetic computation maps to a Layer 2 floating-point implementation with bounded rounding error
- [ ] Every Layer 2 precondition is covered by Layer 3's analysis scope (input ranges)
- [ ] Every Layer 2 postcondition is consistent with the Layer 3 proven output ranges
- [ ] Compiler correctness is addressed (verified compiler, or compiler qualification + binary-level checks)
- [ ] Hardware errata and fault tolerance mechanisms are addressed (outside the scope of these three layers, but required for the full ISO 26262 safety case)

---

## Practical Considerations

### Incremental Adoption

It is not necessary (or practical) to apply all three layers with full rigor from day one. A realistic adoption path:

1. **Start with Layer 3**: Abstract interpretation tools (Astrée, Polyspace) can be deployed on existing codebases with relatively low effort. This immediately eliminates a class of dangerous defects.
2. **Add Layer 2 for critical functions**: Apply deductive verification (Frama-C, SPARK) to the most safety-critical functions — the ones that compute control commands, perform safety checks, or implement the safety monitor.
3. **Introduce Layer 1 for new algorithms**: When designing new control algorithms (e.g., a new lane-keeping strategy), use KeYmaera X or reachability analysis to prove safety properties before implementation begins.

### Cost-Benefit by ASIL Level

| ASIL | Recommended Approach |
|---|---|
| **D** | All three layers with full formal proof. Consider SPARK Ada or formally verified C for safety-critical modules. |
| **C** | Layer 3 mandatory. Layer 2 for critical paths. Layer 1 for novel algorithms. |
| **B** | Layer 3 with abstract interpretation. Layer 2 with testing + partial formal methods. Layer 1 with simulation + formal elements. |
| **A** | Layer 3 with static analysis. Layer 2 with testing. Layer 1 with simulation. |

### Model-Based Development (Simulink)

For projects using Simulink/Stateflow → Embedded Coder:

- **Layer 1**: Can be performed on the Simulink model using tools like Simulink Design Verifier, or by exporting the model to dL for KeYmaera X
- **Layer 2**: Partially addressed by Tool Qualification of Embedded Coder (ISO 26262 Part 8). For higher assurance, apply translation validation or verify the generated code directly.
- **Layer 3**: Must still be performed on the generated C code. Polyspace integrates natively with the MathWorks toolchain.
