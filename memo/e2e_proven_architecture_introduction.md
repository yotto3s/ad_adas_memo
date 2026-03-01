# End-to-End Proven Software Architecture for Autonomous Driving: Introduction

## 1. Problem Statement

### 1.1 The Testing Gap

Autonomous driving software must satisfy ASIL D — the highest automotive safety integrity level. For the hardware side, ISO 26262 Part 5 sets a clear numerical target: the Probabilistic Metric for Hardware Failures (PMHF) must be below 10⁻⁸ per hour. For the software side, no equivalent numerical target exists — software faults are systematic, not random, so the standard manages them through process rigor rather than a failure rate. But the system-level expectation is the same: the risk of a dangerous failure must be astronomically low. Testing alone cannot demonstrate this. Even billions of driven kilometers leave statistical uncertainty far above what ASIL D demands. The gap between what testing can show and what the standard requires must be filled by mathematical proof.

The challenge is that no single formal method covers the entire AD stack. The perception layer runs neural networks with billions of parameters — not amenable to theorem proving. The control layer runs deterministic algorithms — provable, but only if the proof covers physics, code, and runtime errors together. A practical architecture must combine these realities into a coherent safety argument.

### 1.2 The Theoretical Gap in Software Functional Safety

ISO 26262 Part 6 (software development) is fundamentally **process-oriented**. It prescribes activities — requirements analysis, architectural design, unit testing, integration testing, code review — and recommends methods at each ASIL level. But it does not define what "correct software" means mathematically. The result is that software functional safety design relies heavily on designers' judgment and experience rather than on a rigorous theoretical foundation.

Compare this with hardware functional safety. As noted in Section 1.1, ISO 26262 Part 5 provides a clear probabilistic framework: failure rates in FIT, diagnostic coverage percentages, and PMHF as a single comparable number. Hardware safety is a quantitative discipline.

Software safety has no equivalent metric. As Section 1.1 also noted, software faults are systematic — a bug is either present or absent, it does not occur randomly. ISO 26262 acknowledges this distinction but does not provide an alternative mathematical framework for software. In practice, the safety argument for software reduces to: "we followed the prescribed process, we achieved the required test coverage, we performed the required reviews." This is a process argument, not a property argument.

The theoretical foundations for a property-based approach do exist:

| Foundation | What It Provides |
|---|---|
| **Hoare logic** (1969) | Mathematical framework for specifying and proving program correctness: preconditions, postconditions, invariants |
| **Abstract interpretation** (Cousot & Cousot, 1977) | Sound over-approximation of program behavior — can prove the absence of entire classes of runtime errors |
| **Hybrid systems theory** (Platzer, 2010s) | Connects software correctness to physical safety — proves that control algorithms keep the vehicle safe in the physical world |
| **Contract-based design** (Benveniste et al., 2018) | Compositional framework — system-level safety follows from component-level contracts |

These are not new ideas. But they have not been connected to the ISO 26262 framework in a way that practitioners can apply systematically. The three-layer verification approach described in this memo bridges that gap: it maps each theoretical foundation to a specific verification layer and shows how the layers compose into an end-to-end safety argument that is mathematical, not procedural.

### 1.3 The Practical Gap — and Why It Is Closing

Even where the theory was understood, applying it at production scale was historically impractical. Tools like Frama-C, KeYmaera X, and Astrée require specialized expertise — writing ACSL annotations, constructing loop invariants, formulating differential dynamic logic proofs. The number of engineers with these skills has always been small relative to the size of automotive codebases.

AI coding agents are changing this equation.

AI agents can now operate formal verification tools effectively — writing annotations, constructing proof obligations, and iterating with SMT solvers until proofs discharge. The tools themselves remain the trust anchor: they mechanically check every proof, regardless of who (or what) wrote it. An incorrect annotation simply fails to verify. This means the correctness guarantee does not depend on the AI being perfect — it depends on the tool being sound, which is a much smaller and well-established trust base.

At the same time, the engineering workflow is already shifting. Many engineers now spend most of their time reviewing code that AI agents wrote rather than writing code from scratch. Formal verification fits naturally into this new workflow:

```
Traditional workflow:
  Engineer writes code --> Engineer reviews code --> Engineer tests code

Current workflow:
  AI writes code --> Engineer reviews code --> Engineer tests code

Proven workflow:
  AI writes code + proof --> Tool verifies proof --> Engineer reviews design
```

In the proven workflow, the tool mechanically guarantees the absence of bugs. Engineers no longer need to search for bugs in code review. Instead, they can focus on what they are uniquely qualified to judge: whether the architecture is right, whether the specifications capture the true intent, and whether the code is maintainable. This is a shift from **bug detection** to **design quality** — a better use of human expertise.

### 1.4 The Hardware Fault Gap

Even if every line of software is formally proven correct, the safety argument has a hidden assumption: the hardware executes the software faithfully. In reality, it does not — cosmic rays flip bits in registers, electromagnetic interference corrupts bus transfers, transistor aging causes signals to stick. These are random hardware faults, and they can silently corrupt the execution of proven software, undermining the entire verification chain.

The brute-force solution is to run everything on ASIL-D hardware — lockstep cores, ECC memory, full redundancy. This is expensive in silicon area, power consumption, and unit cost. For most of the software stack, it is overkill: if a bit flip causes a minor perturbation in a noise-tolerant algorithm, the output remains within specification. The expensive ASIL-D hardware protection should be concentrated where it is actually needed.

ISO 26262 provides ASIL decomposition as the mechanism for this: split an ASIL-D requirement into an ASIL-B intended function (normal path) and an ASIL-B(D) safety mechanism (anomaly path). The normal path runs on cheaper hardware; the anomaly path runs on ASIL-D hardware and monitors the normal path's integrity. This asymmetric decomposition is well understood in principle.

In practice, however, the anomaly path design suffers from the same problem as software functional safety (Section 1.2): **it is not systematically formulated**. Engineers design anomaly paths ad-hoc, based on experience and intuition. Which checks to duplicate on the ASIL-D core? Which fault patterns are already caught by existing spec checks? Which are structurally undetectable? These questions are answered differently by different engineers on different projects, with no shared theoretical framework.

This architecture addresses the gap by providing a systematic derivation method: the anomaly path is derived from the normal path's software architecture artifacts — input definitions, output specifications, check algorithms, and state transition designs — at design phase, before implementation begins. The derivation identifies exactly which fault patterns escape the normal path's self-checks, and generates the minimum monitoring logic needed on the ASIL-D core.

### 1.5 This Document

This memo introduces an architecture that addresses all four gaps. It is called the **end-to-end proven architecture** because the safety guarantee chains from physics (vehicle dynamics) through algorithms (control logic) through software (C/C++ implementation) to runtime behavior (no crashes, no undefined behavior) — with every link in the chain either formally proven or bounded by a proven monitor, and the execution environment itself monitored for hardware faults through a systematically derived anomaly path.

---

## 2. Solution Overview: Proven Core + Monitored Envelope

The architecture partitions the AD/ADAS software into two concentric zones:

```
+--------------------------------------------------+
|            Zone 2: Monitored Envelope             |
|                                                   |
|   Perception (neural networks, sensor fusion)     |
|   Planning (behavior planning, motion planning)   |
|   Prediction (world model, intent estimation)     |
|                                                   |
|   +-----------------------------------------+     |
|   |        Zone 1: Proven Core              |     |
|   |                                         |     |
|   |   Control algorithms (verified)         |     |
|   |   Safety monitors (verified)            |     |
|   |   Fallback controllers (verified)       |     |
|   |   ODD boundary monitors (verified)      |     |
|   +-----------------------------------------+     |
|                                                   |
+--------------------------------------------------+
                        |
                    Actuators
```

**Zone 1 (Proven Core)**: Every safety-relevant property is mathematically proven. Control algorithms, safety monitors, and fallback controllers live here. Their correctness is established through formal verification — theorem proving, deductive verification, and abstract interpretation.

**Zone 2 (Monitored Envelope)**: Components where formal proof of the primary function is infeasible. Neural network perception, complex motion planning, and world prediction live here. Safety is not proven for these components directly. Instead, **proven monitors in Zone 1 constrain Zone 2's outputs at runtime**.

The key insight: *you do not need to prove the perception neural network is correct. You need to prove that the safety monitor will catch it when it is wrong, and that the fallback response is safe.*

This separation means the most complex and rapidly evolving parts of the stack (ML models, planning heuristics) can be developed and updated without invalidating the safety proof. Only Zone 1 changes require re-verification.

---

## 3. Three-Layer Formal Verification

Zone 1 components are verified through three layers of proof, each answering a different question:

```
+------------------------------------------------------------------+
|  Layer 1: Algorithm Correctness                                  |
|  "Does this algorithm guarantee safety in the physical world?"   |
|  Method: Theorem proving (KeYmaera X), reachability analysis     |
|  Output: Mathematical proof of safety property                   |
+------------------------------------------------------------------+
                              |
                    assumptions & constraints
                              |
                              v
+------------------------------------------------------------------+
|  Layer 2: Implementation Correctness                             |
|  "Does the code faithfully implement the algorithm?"             |
|  Method: Deductive verification (Frama-C, SPARK)                 |
|  Output: Proof that code satisfies algorithm's contracts         |
+------------------------------------------------------------------+
                              |
                      preconditions & ranges
                              |
                              v
+------------------------------------------------------------------+
|  Layer 3: Runtime Error Freedom                                  |
|  "Is the code free from crashes and undefined behavior?"         |
|  Method: Abstract interpretation (Astree, Polyspace)             |
|  Output: Proof of no runtime errors for all valid inputs         |
+------------------------------------------------------------------+
```

**Layer 1** proves the algorithm is safe under stated assumptions — e.g., "if sensor error is within ±0.1m and control runs at 100Hz, the vehicle never departs the lane." The proof is about mathematics and physics, not about code.

**Layer 2** proves the code implements the algorithm correctly — e.g., "the `compute_steering()` function's output always matches the control law within bounded floating-point error." This bridges the gap between the mathematical world and the software world.

**Layer 3** proves the code has no runtime errors — no integer overflow, no buffer overflow, no division by zero, no undefined behavior. This ensures the code behaves as the compiler and verification tools assume.

These three layers chain together:

```
Layer 1 proves:  algorithm safe       IF  assumptions hold
Layer 2 proves:  code = algorithm     IF  inputs in range
Layer 3 proves:  code runs correctly  IF  inputs in range
────────────────────────────────────────────────────────────
Combined:        implementation safe  IF  operational assumptions hold
```

If any link breaks, the entire chain fails. The architecture makes every link explicit and proven.

> **Detail**: See `three_layer_formal_verification.md` for the full technical description of each layer, including tool chains, proof methods, and worked examples.

---

## 4. Contract-Based Composition

The architecture uses **assume-guarantee contracts** at every component boundary. Each component declares:

- **Assumes**: What it requires from its inputs (e.g., "sensor data arrives within 50ms, position error ≤ 0.1m")
- **Guarantees**: What it promises about its outputs (e.g., "steering command within ±5°, computed within 2ms")

```
  Component A              Component B              Component C
+-------------+         +-------------+         +-------------+
| Assumes: ... |  --->  | Assumes: ... |  --->  | Assumes: ... |
| Guarantees: G_A |     | Guarantees: G_B |     | Guarantees: G_C |
+-------------+         +-------------+         +-------------+

Contract rule: G_A must satisfy the assumptions of B.
               G_B must satisfy the assumptions of C.

End-to-end: if all contracts are discharged,
            system-level safety follows by transitivity.
```

This enables **modular development**: teams can work on individual components independently, as long as they satisfy their contracts. It also enables **incremental adoption**: new components can be integrated by verifying only the new component's contract against its neighbors, rather than re-verifying the entire system.

When a component is updated (e.g., a new perception model), only its contract needs to be re-validated. If the contract interface is unchanged, downstream proofs remain valid.

---

## 5. Safety Envelope Enforcement

The mechanism that makes Zone 2 safe is the **safety envelope enforcement pattern**:

```
                    +-------------------+
                    |    Zone 2         |
                    |  (Perception,     |
  Sensor data ----> |   Planning)       |----> Candidate output
                    +-------------------+
                                              |
                                              v
                                     +----------------+
                                     | Safety Monitor |  <-- Zone 1
                                     | (Proven)       |      (Verified)
                                     +----------------+
                                        |          |
                                   PASS |          | FAIL
                                        v          v
                                   Use candidate   Use verified
                                   output          fallback
                                        |          |
                                        v          v
                                     +----------------+
                                     |   Actuators    |
                                     +----------------+
```

Zone 2 produces candidate outputs (trajectories, control commands). The safety monitor — a Zone 1 component, formally verified — checks whether the candidate satisfies the safety contracts. If it does, the candidate is passed through. If not, a pre-verified fallback is substituted.

The safety monitor itself is derived from the Layer 1 proof. The KeYmaera X tool can automatically extract monitor conditions from offline safety proofs (ModelPlex). These conditions are simple arithmetic checks — easy to implement, easy to verify at Layers 2 and 3.

> **Detail**: See `e2e_proven_architecture_research.md`, Section 2 (Safety Envelope Enforcement Pattern) for the full treatment, including ModelPlex derivation, Control Barrier Functions, and RSS.

---

## 6. Anomaly Path: Hardware Fault Coverage

Sections 3–5 address **software correctness** — proving the code does what it should. But code runs on hardware, and hardware fails. A cosmic ray flips a bit in a register; electromagnetic interference corrupts a bus transfer; a transistor degrades and a signal sticks at zero. If a hardware fault silently corrupts the execution of a proven software component, the entire verification chain is undermined.

The architecture addresses this through **asymmetric ASIL decomposition**:

```
+---------------------------+     +---------------------------+
|   Normal Path (ASIL-B)    |     |   Anomaly Path (ASIL-D)   |
|                           |     |                           |
|   Runs the algorithm      |     |   Does NOT re-execute     |
|   Runs spec checks on     |     |   the algorithm           |
|   inputs and outputs      |     |                           |
|   (plausibility, range,   |     |   Verifies that the       |
|    consistency checks)    |     |   normal path's checks    |
|                           |     |   have not been disabled   |
|   Produces correct output |     |   by HW faults            |
|   as long as no HW faults |     |                           |
+---------------------------+     +---------------------------+
        ASIL-B core                     ASIL-D core
       (no lockstep)                   (lockstep etc.)
```

The key insight is that the anomaly path does not need to re-run the full computation. It only needs to verify that the normal path's **self-checks** (range checks, plausibility checks, state consistency checks) are still functioning correctly. This is a much smaller task, requiring far less computational overhead.

This decomposition follows ISO 26262: ASIL D = ASIL B(D) + ASIL B(D), where the normal path carries the intended function and the anomaly path carries the safety mechanism. The anomaly path can be **derived systematically from the normal path's software architecture artifacts at design phase** — input definitions, output specifications, check algorithms, and state transition designs — without waiting for the normal path implementation.

The relationship to the rest of the architecture:

- **Three-layer verification** (Section 3) proves the software is correct assuming no HW faults
- **Safety envelope enforcement** (Section 5) catches wrong outputs from Zone 2 components
- **Anomaly path** catches HW faults that could silently corrupt Zone 1's execution
- Together, they cover both systematic failures (software bugs) and random failures (hardware faults)

> **Detail**: See `deriving_anomarly_path_en.md` for the full treatment, including the HW fault model (SEU, SET, EMI), fault pattern analysis, and the systematic derivation method.

---

## 7. Graceful Degradation

The architecture defines a multi-level fallback hierarchy. Each level has its own safety proof, and transitions between levels are themselves verified:

```
Level 0: Full Autonomy
  |  (safety envelope violation OR anomaly path fault detected)
  v
Level 1: Degraded Autonomy (simplified planner, reduced ODD)
  |  (further degradation trigger)
  v
Level 2: Minimal Risk Condition (controlled stop, safe pullover)
  |  (critical failure)
  v
Level 3: Emergency Stop (immediate braking)
```

Each level uses fewer and simpler components, making formal verification progressively easier. Level 0 uses the full Zone 1 + Zone 2 stack. Levels 2–3 use only Zone 1 components and are fully formally proven.

The transition logic is modeled as a timed automaton and verified to ensure: no unprotected states exist, the system can always degrade further, and transitions are deterministic.

> **Detail**: See `e2e_proven_architecture_research.md`, Section 5 (Graceful Degradation and Fallback Hierarchy).

---

## 8. Relationship to Standards

| Standard | Role in This Architecture |
|---|---|
| **ISO 26262** | ASIL decomposition across zones. Zone 1 at ASIL D, Zone 2 at ASIL B with ASIL D monitors. Three-layer verification maps to Part 6 (software). |
| **ISO 21448 (SOTIF)** | Addresses insufficiencies in Zone 2 (perception, planning). Safety envelope enforcement is a SOTIF mitigation strategy. |
| **UN R157** | Type approval for ALKS (Level 3 highway). ODD formal specification and runtime monitoring directly support R157 compliance. |
| **ISO 21434** | Cybersecurity. Attack detection integrated into the safety monitoring framework. |

The architecture does not replace these standards — it provides the technical mechanisms to satisfy their most demanding requirements.

---

## 9. Research Frontiers

Several areas remain active research topics:

- **Perception contracts** — How to formally specify and validate what the perception system guarantees to downstream components
- **Neural network verification** — Proving safety-relevant properties of NN sub-components for bounded input regions
- **Quantitative proof composition** — Combining deterministic proofs (Zone 1) with probabilistic evidence (Zone 2) into a single safety metric
- **OTA re-verification** — Determining which proofs must be redone after a software update, and doing so incrementally
- **Cybersecurity-safety integration** — Mapping cyber threats to safety contract violations

> **Detail**: See `e2e_proven_architecture_research.md` for the full research agenda with 10 prioritized topics.

---

## 10. Document Map

This memo is the entry point. The following documents provide detailed treatment:

| Document | Content |
|---|---|
| `three_layer_formal_verification.md` | Full technical description of Layers 1, 2, 3 with tools, methods, and examples |
| `e2e_proven_architecture_research.md` | 10-topic research agenda expanding every aspect of the architecture |
| `deriving_anomarly_path_en.md` | Deriving ASIL-D anomaly handling from normal-path specifications |
| `plans/2026-02-27-e2e-proved-architecture-design.md` | Complete architecture design document |
