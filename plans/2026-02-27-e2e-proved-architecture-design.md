# End-to-End Proved Software Architecture for Autonomous Driving

## 1. Introduction & Vision

### 1.1 The Problem

Autonomous driving software is among the most complex safety-critical systems ever built. It spans sensor processing, machine learning, planning, control, and actuation -- each introducing distinct failure modes. Traditional testing-based verification cannot provide the mathematical guarantees required for ASIL D safety integrity levels, and no single formal method covers the entire stack.

### 1.2 The Solution: Proved Core + Monitored Envelope

This document describes an architecture that achieves end-to-end safety assurance by partitioning the AD/ADAS software into two concentric zones:

**Zone 1: The Proved Core** -- Components where every safety-relevant property is mathematically proved. This includes control algorithms, safety monitors, and the software implementation layer. Correctness is established through formal verification: theorem proving, abstract interpretation, and deductive verification.

**Zone 2: The Monitored Envelope** -- Components where formal proof of the primary function is infeasible (neural network perception, complex motion planning, world prediction), but safety is ensured by **proved monitors that live in Zone 1** and constrain the envelope's outputs at runtime.

The architectural insight is: *you do not need to prove the perception neural network is correct -- you need to prove that the safety monitor will catch it when it is wrong, and that the fallback response is safe.*

```
+----------------------------------------------+
|           Zone 2: Monitored Envelope         |
|  +----------------------------------------+  |
|  |  Perception (NN, sensor fusion, etc.)  |  |
|  |  Planning (behavior + motion)          |  |
|  |  World model / prediction              |  |
|  +-------------------+--------------------+  |
|                       | outputs constrained  |
|                       v                      |
|  +----------------------------------------+  |
|  |      Zone 1: The Proved Core           |  |
|  |                                        |  |
|  |  Verified safety monitors              |  |
|  |  Verified control algorithms           |  |
|  |  Verified SW implementation            |  |
|  |  Hardware fault tolerance (SW level)   |  |
|  +----------------------------------------+  |
+----------------------------------------------+
```

### 1.3 Relationship to Existing Work

This design extends and integrates three existing research documents:

| Document | Covers | Extended by this design |
|---|---|---|
| `three_layer_formal_verification.md` | Algorithm correctness (L1), code alignment (L2), runtime error freedom (L3) | Becomes Zone 1 Sections 4.1-4.2; extended with perception/planning/composition |
| `deriving_anomarly_path_en.md` | Hardware fault models, anomaly path derivation, ASIL decomposition | Becomes Zone 1 Section 4.3; integrated into system-level safety argument |
| Arcanum (safe C++ subset, tool spec, implementation design) | Formal verification tooling for C++ | Becomes the implementation vehicle for Zone 1 verification (Section 4.2) |

---

## 2. Contract-Based Composition

### 2.1 The Unifying Principle

Every boundary between components, layers, and zones is defined by a formal **assume-guarantee contract** `(A, G)`:

- **A (Assumes)**: What this component requires from its environment to function correctly
- **G (Guarantees)**: What this component promises to deliver if its assumptions are met

A system is safe if and only if: for every component, the guarantees of its neighbors satisfy its assumptions. The end-to-end safety argument is the transitive closure of all contracts being discharged.

### 2.2 Contract Discharge Methods

Different components require different levels of evidence to discharge their contracts:

| Zone | Discharge Method | Assurance Level | Example |
|---|---|---|---|
| Zone 1 | Mathematical proof (theorem prover, abstract interpretation) | Deterministic guarantee | "Steering command is always within [-5.0, 5.0] degrees" |
| Zone 2 | Statistical evidence + runtime monitoring | Probabilistic bound + runtime enforcement | "Pedestrian detection recall >= 99.9% (measured) AND safety monitor rejects unsafe outputs (proved)" |
| Foundation | Supplier qualification + validation testing | Contractual guarantee | "RTOS scheduling meets WCET budgets (qualified by supplier)" |

### 2.3 Contract Formalism

Contracts are expressed in a hierarchy of formalisms depending on the layer:

- **Physical/algorithm layer**: Signal Temporal Logic (STL), differential dynamic logic (dL)
- **Software layer**: Function contracts in ACSL/Arcanum annotation syntax (requires/ensures/assigns)
- **System layer**: Assume-guarantee pairs in structured natural language + formal backing
- **Perception layer**: Probabilistic contracts (e.g., "P(detection | object within 50m) >= 0.999")

### 2.4 Key References

- SPEEDS methodology (FP7 project): Contract-based design for embedded systems
- OCRA tool (Fondazione Bruno Kessler): Contract-based verification for system architectures
- ISO 21839: Systems of systems -- definition and vocabulary
- Benveniste et al., "Contracts for System Design" (Foundations and Trends in EDA, 2018)

---

## 3. Foundation Assumptions

These layers are outside the direct scope of this architecture (handled by suppliers and platform providers), but the proved core's safety argument depends on their guarantees. Each is treated as an external contract: we specify what we assume, how we validate the assumption, and what breaks if it is violated.

### 3.1 Hardware

**What we assume:**
- Random hardware fault rates (SEU, SET) are within specified FIT budgets per component
- Hardware safety mechanisms (ECC for memory, lockstep for CPU cores) achieve specified diagnostic coverage
- Supplier provides FMEDA data with SPFM/LFM/PMHF values meeting ASIL D targets

**How we validate:**
- Review and audit supplier FMEDA data
- Our anomaly path architecture (Section 4.3) handles residual faults not covered by hardware mechanisms
- Fault injection testing on target hardware to validate coverage claims

**What breaks if violated:**
- If hardware fault rates exceed assumptions, the anomaly path's dual-point fault argument (Section 4.3) may not meet ASIL D PMHF targets
- Mitigation: design anomaly path with margin; include hardware monitoring (temperature, voltage) as early warning

### 3.2 OS/RTOS

**What we assume:**
- Task scheduling meets timing contracts (tasks execute within their period and WCET budget)
- Freedom from interference (FFI) between partitions: a fault in one partition cannot corrupt another
- Memory protection prevents unauthorized access across partition boundaries
- System services (timers, interrupts) behave according to specification

**How we validate:**
- Use a safety-qualified RTOS (e.g., QNX Safety, ETAS RTA-OS, Vector MICROSAR Safe)
- WCET budgets validated by supplier-provided analysis or independent measurement
- FFI validation through dedicated test campaigns per ISO 26262 Part 6

**What breaks if violated:**
- Timing violation: control loop misses deadline, Layer 1 safety proof assumption (control period <= T) is violated
- FFI violation: data corruption propagates between components, all proof assumptions about data integrity fail
- Mitigation: runtime timing monitors (watchdog), memory protection hardware checks

### 3.3 Compiler

**What we assume:**
- The compiled binary faithfully implements the semantics of the verified source code
- Compiler optimizations do not introduce behaviors not present in the source

**How we validate:**
- **Ideal**: Use CompCert (formally verified C compiler) for all Zone 1 safety-critical code
- **Practical**: Use qualified GCC/Clang (ISO 26262 Part 8 Tool Qualification) + compiler validation test suites
- **Future**: Translation validation at binary level (compare binary behavior against source semantics)

**What breaks if violated:**
- Compiler introduces a miscompilation in safety-critical code: the Layer 2 proof (source code correctness) does not transfer to the executable
- Mitigation: binary-level analysis tools (AbsInt aiT for WCET, StackAnalyzer for stack usage) provide partial binary-level assurance

### 3.4 Communication

**What we assume:**
- End-to-end protection (AUTOSAR E2E) detects data corruption during transmission with specified diagnostic coverage
- Network scheduling (CAN priorities, Ethernet TSN) guarantees message delivery within specified latency bounds
- No message loss beyond the specified rate

**How we validate:**
- Qualified AUTOSAR E2E library
- Network scheduling analysis (worst-case response time analysis for CAN; TSN gate control list verification)
- Network testing under load

**What breaks if violated:**
- Message corruption undetected: control algorithm receives corrupted input, Layer 1 assumption about input accuracy violated
- Message delay exceeds bound: control loop operates on stale data
- Mitigation: runtime detection through timeout monitors and sequence counters

---

## 4. Zone 1: The Proved Core

This zone contains all components for which mathematical proof of correctness is both required and achievable. Every property claimed here must be backed by a machine-checked proof or a sound static analysis result.

### 4.1 Control Algorithm Verification

**Full treatment**: See `three_layer_formal_verification.md`, Layer 1.

**Summary**: The control algorithm (e.g., lateral control, longitudinal control, trajectory tracking) is modeled as a hybrid system (discrete control decisions + continuous physical dynamics) and proved safe using:

- **Differential dynamic logic (dL)** with KeYmaera X: proves safety properties (lane keeping, collision avoidance) hold for all time under stated assumptions
- **Reachability analysis** (CORA, JuliaReach): computes the set of all reachable states and verifies it stays within safe bounds
- **Control Barrier Functions (CBFs)**: scalar functions that provably separate safe and unsafe states; can also synthesize safe controllers
- **RSS (Responsibility-Sensitive Safety)**: formal safety envelope derived from traffic rules and physics

**Contract interface (guarantees upward)**:
- `G_alg`: If inputs satisfy sensor accuracy bounds and timing constraints, the algorithm's output keeps the vehicle safe (within lane, collision-free, etc.)
- `G_constraints`: Derived constraints on the software implementation (output ranges, loop timing, precision requirements)

**Contract interface (assumes downward)**:
- `A_sensor`: Sensor measurements are within specified error bounds
- `A_timing`: Control loop executes within specified period
- `A_precision`: Computation is performed with at most specified floating-point error

### 4.2 Software Implementation Verification

**Full treatment**: See `three_layer_formal_verification.md`, Layers 2 and 3; Arcanum documentation.

**Summary**: The software implementing the control algorithm is verified at two levels:

**Layer 2 -- Code-Algorithm Alignment**: Proves the source code faithfully implements the algorithm. Using Arcanum (or Frama-C for C code), each function is annotated with contracts (requires/ensures) derived from Layer 1 constraints. The Weakest Precondition calculus generates proof obligations discharged by SMT solvers.

**Layer 3 -- Runtime Error Freedom**: Proves the code has no undefined behavior (no buffer overflow, integer overflow, division by zero, null dereference, etc.) using abstract interpretation (Astree, Polyspace) or bounded model checking (CBMC).

**Floating-point verification**: The gap between real arithmetic (Layer 1) and IEEE 754 floating-point (implementation) is explicitly bounded. Methods:
- Absorb rounding error into Layer 1 safety margin
- Use Gappa or Fluctuat to bound accumulated floating-point error through computations
- Arcanum's `fp_safe` annotation proves absence of NaN/infinity

**Contract interface**:
- `G_impl`: If called with inputs in precondition range, the function returns correct results with no undefined behavior
- `A_impl`: Inputs are within declared precondition ranges (verified by Layer 3 for all callers)

### 4.3 Hardware Fault Handling at Software Level

**Full treatment**: See `deriving_anomarly_path_en.md`.

**Summary**: The ASIL-D anomaly path architecture handles random hardware faults (SEU, SET, EMI-induced bit flips) through a two-tier monitoring scheme:

- **Normal path (ASIL-B core)**: Executes the algorithm + spec checks on I/O. Produces output directly to actuator.
- **Anomaly path (ASIL-D core)**: Does NOT re-execute the algorithm. Monitors the integrity of the normal path's check mechanisms:
  - Constant CRC verification (detects threshold corruption)
  - Signature verification (detects control flow corruption / check skip)
  - Encoded result verification (detects result flag bit flip)
  - State transition verification (detects state variable corruption)

The anomaly path is derivable at the SW architecture design phase from: input interface definitions, output spec definitions, spec check algorithm definitions, and state transition design.

**Contract interface**:
- `G_hw`: Random hardware faults affecting the normal path are detected with diagnostic coverage >= DC_target within FTTI
- `A_hw`: Hardware fault rates are within specified FIT budgets (from Section 3.1)

### 4.4 Safety Monitors

This is the critical bridge component that makes the Monitored Envelope safe. Safety monitors live in Zone 1 (formally verified) but constrain Zone 2 outputs.

#### 4.4.1 ModelPlex Runtime Monitors

Derived automatically from KeYmaera X offline safety proofs:

1. The Layer 1 dL proof proves: "If the algorithm is followed, the system is safe"
2. ModelPlex extracts a **monitor condition** -- an arithmetic formula over observable variables
3. The monitor condition is proved (as part of the dL proof) to have the property: "If the monitor condition holds at each control step, the offline safety proof applies to the actual execution"
4. The monitor condition is implemented as runtime arithmetic checks
5. The monitor implementation is verified through Layers 2 and 3 (it is simple enough)

If the monitor detects violation: transition to verified fallback controller.

#### 4.4.2 RSS Enforcement Layer

RSS (Responsibility-Sensitive Safety) defines formal safety distances:
- Longitudinal: minimum following distance based on velocities, reaction time, and braking capabilities
- Lateral: minimum side distance during lane changes
- Intersection: right-of-way rules formalized as timing constraints

The RSS enforcement layer computes safety distances in real-time and overrides the planner output if an RSS violation is imminent. The enforcement logic itself is formally verified.

#### 4.4.3 Perception Sanity Monitors

Verified runtime checks on perception pipeline output:
- **Object count bounds**: Number of detected objects is within physically plausible range
- **Velocity consistency**: Object velocities are consistent between frames (no teleportation)
- **Spatial consistency**: Object positions are consistent with road geometry
- **Temporal coherence**: Track continuity, no phantom objects appearing/disappearing erratically
- **Confidence calibration**: Perception confidence scores are above minimum threshold

These monitors are simple enough (arithmetic comparisons, sliding window statistics) to be formally verified through Layers 2 and 3.

#### 4.4.4 ODD Boundary Monitor

Detects when the system is leaving its Operational Design Domain:
- Weather conditions (rain intensity, visibility distance)
- Road type (highway vs. urban vs. construction zone)
- Sensor health (degradation detection)
- Localization quality (GPS accuracy, map matching confidence)

When ODD boundary is crossed: initiate graceful degradation (Section 6.3).

**Contract interface for all monitors**:
- `G_monitor`: If Zone 2 output violates safety conditions, the monitor detects it and triggers fallback within FTTI
- `A_monitor`: Monitor inputs (system state, perception output) are available with specified latency and accuracy

---

## 5. Zone 2: The Monitored Envelope

Zone 2 contains components where formal proof of the primary function is infeasible, but safety is achieved through the combination of:
1. **Best-effort correctness**: Statistical validation, testing, simulation
2. **Formal contracts**: Precisely specified input/output requirements
3. **Zone 1 monitors**: Proved runtime checks that catch violations and trigger safe fallback

### 5.1 Sensor Modeling & Perception Assurance

#### 5.1.1 The Challenge

The perception pipeline (camera, lidar, radar processing, neural network inference, object detection, classification, tracking, sensor fusion) is the least provable part of the AD stack. Neural networks are opaque nonlinear functions with billions of parameters. Their behavior is learned from data, not specified by design.

Yet Layer 1 safety proofs depend on assumptions about sensor/perception accuracy. The perception layer must provide grounded guarantees for these assumptions.

#### 5.1.2 Sensor Specification Contracts

Each sensor must have a formal contract specifying its capabilities and limitations:

```
Sensor Contract: Front Camera
  Assumes:
    - Illumination within [100, 100000] lux
    - No direct lens contamination
    - Object size >= 0.3m width
  Guarantees:
    - Detection recall >= 99.9% for objects within [5m, 80m]
    - Position error <= 0.5m lateral, 1.0m longitudinal
    - Latency <= 50ms from image capture to object list output
    - False positive rate <= 0.1 per frame
```

These contracts must be validated through:
- Sensor characterization testing across environmental conditions
- Statistical analysis with confidence intervals
- Degradation models (how performance degrades as conditions approach contract boundaries)

#### 5.1.3 Neural Network Verification

**What is provable today (limited but growing)**:
- **Local robustness**: For a given input, no perturbation within epsilon-ball changes the classification. Tools: Marabou, alpha-beta-CROWN, ERAN, VeriNet
- **Output bounds**: For inputs within a specified region, the output is within specified bounds. Useful for verifying that NN output ranges match downstream contract assumptions
- **Monotonicity properties**: Proving that certain input changes produce monotonic output changes (e.g., closer object produces larger bounding box)

**What is NOT provable today**:
- Global correctness ("this NN always detects pedestrians")
- Performance under arbitrary distribution shift
- Behavior on truly novel inputs

**Research direction: Property-specific verification**: Rather than proving global NN correctness, prove only the specific properties that the safety monitors need. Example: "If a pedestrian exists within 30m and meets minimum size criteria, the NN output bounding box overlaps the pedestrian's true position with IoU >= 0.5." This is a much weaker (and more provable) property than "the NN correctly detects all pedestrians."

#### 5.1.4 Uncertainty Quantification

**Conformal prediction**: Provides distribution-free prediction intervals with guaranteed coverage. For a trained model and calibration dataset:
- Output prediction sets (e.g., "the object is one of: pedestrian, cyclist") with guaranteed coverage probability
- Output prediction intervals (e.g., "object distance is 25m +/- 2m") with guaranteed coverage probability
- No assumptions about the model or data distribution required (only exchangeability)

This provides the probabilistic contracts needed for the sensor/perception layer.

**Research needed**: Applying conformal prediction to real-time perception with temporal dependencies (sequential data violates exchangeability; adaptive conformal prediction methods needed).

#### 5.1.5 Sensor Fusion Verification

Multi-sensor fusion must preserve or improve individual sensor guarantees:
- **Proved fusion bounds**: If sensor A has position error <= 0.5m and sensor B has position error <= 0.3m, prove the fused estimate has error <= X
- **Redundancy analysis**: Prove that the system remains safe with any single sensor failed
- Methods: Bayesian fusion with verified interval bounds, Dempster-Shafer theory for multi-source evidence combination

#### 5.1.6 Perception Degradation Detection

Runtime monitors that detect when perception quality drops below the assumed contract:
- Internal confidence metrics (NN softmax entropy, detection score distributions)
- Cross-sensor consistency checks (camera detects object but lidar does not)
- Environmental condition monitoring (rain sensors, visibility estimation)
- When degradation detected: tighten safety margins or trigger degraded mode

### 5.2 Planning & Decision-Making Verification

#### 5.2.1 Behavior Planning

The behavior planner (lane change decisions, stop/go decisions, yield decisions) is typically a state machine, rule-based system, or decision tree -- all formally verifiable.

**Verification approaches**:
- **Model checking** (UPPAAL, NuSMV): Exhaustively verify all states and transitions. Prove properties like "the system never enters an unsafe behavior state," "every behavior transition has a valid triggering condition"
- **Theorem proving**: For more complex behavior logic with parameterized rules
- **Temporal logic specification**: Behavior requirements expressed in LTL/CTL, verified against the behavior model

**Contract interface**:
- `G_behavior`: The behavior planner outputs only physically feasible and rule-compliant high-level decisions
- `A_behavior`: World model provides accurate road topology, traffic rules, and other vehicle intentions

#### 5.2.2 Motion Planning with Safety Filters

The motion planner (trajectory optimization, sampling-based planning) is typically too complex to verify directly. Instead, apply a **verified safety filter**:

```
Motion Planner (Zone 2, unverified)
    | produces candidate trajectory
    v
Safety Filter (Zone 1, verified)
    | checks: is trajectory collision-free AND physically feasible?
    +-- YES --> execute trajectory
    +-- NO  --> substitute verified fallback trajectory
```

**Safety filter methods**:
- **Control Barrier Functions (CBFs)**: Define a barrier function B(x) where B(x) <= 0 is the safe set. The safety filter modifies the planned control input minimally to ensure dB/dt <= -alpha*B(x), keeping the system in the safe set. Implemented as a QP solved at each control step.
- **Reachability-based trajectory validation**: Compute the reachable set of the planned trajectory (accounting for tracking error and disturbances). If reachable set intersects obstacles or road boundaries, reject.
- **RSS check**: Verify the planned trajectory maintains RSS-compliant distances from all other traffic participants at every time step.

**Research needed**:
- Computational efficiency of safety filters for real-time use (QP solve time, reachability computation time)
- Combining multiple safety filter criteria (RSS + CBF + road boundaries) without over-constraining the planner
- Formal proof that the safety filter preserves liveness (the system can always make progress, not just stay safe by stopping)

#### 5.2.3 Verified Fallback Trajectories

Pre-computed emergency maneuvers that are formally proved safe:
- **Emergency stop**: Braking profile that brings the vehicle to a complete stop within a known distance. Proved safe using Layer 1 methods (reachability analysis of braking dynamics).
- **Lane keeping at reduced speed**: Simple lane-centering controller at low speed. Proved safe using dL/KeYmaera X.
- **Controlled stop in safe location**: Trajectory to nearest safe stop location (shoulder, parking area). Pre-planned for the current road segment.

Each fallback trajectory has its own Layer 1 safety proof and Layer 2/3 implementation proof. These are the last line of defense.

#### 5.2.4 Reinforcement Learning (if applicable)

If RL is used for any planning or decision-making component:
- **Constrained RL**: Train with formal safety constraints (CBF constraints during training ensure the learned policy respects safety bounds)
- **Policy verification**: After training, verify the NN policy using the same techniques as perception NN verification (Section 5.1.3) -- prove specific safety properties for bounded input regions
- **Shielding**: Runtime shield (a verified safety filter) overrides the RL policy when it would violate safety constraints

### 5.3 World Model & Prediction

#### 5.3.1 Reachability-Based Prediction

Instead of predicting other vehicles' exact future trajectories (impossible to prove correct), compute **reachable sets** of possible future positions:
- For each detected vehicle, compute the set of all positions it could reach within the prediction horizon, given bounded acceleration/deceleration and steering
- Conservative but provably sound: the actual position is guaranteed to be within the computed set
- Tools: CORA (TU Munich), CommonRoad

**Contract interface**:
- `G_prediction`: The predicted reachable set contains the actual future position of each traffic participant (soundness)
- `A_prediction`: Object detection provides position and velocity within specified error bounds (from perception contract)

#### 5.3.2 ODD Formal Specification

The Operational Design Domain must be a formal mathematical object, not a natural-language description:

```
ODD Contract:
  Parameters:
    - Road type: highway, limited access (no pedestrians, no oncoming traffic)
    - Speed range: [0, 130] km/h
    - Weather: visibility >= 100m, no ice (friction coefficient >= 0.5)
    - Localization: position error <= 0.3m, heading error <= 1 degree
    - Map: HD map available, map age <= 30 days
    - Sensors: all primary sensors operational (camera, lidar, radar)
    - Connectivity: V2X not required
```

Each ODD parameter maps to an assumption in one or more Layer 1 proofs. The ODD boundary monitor (Section 4.4.4) checks these parameters at runtime.

**Research needed**: Formal ODD specification language that is both machine-readable (for automated checking) and human-readable (for regulatory submission). Potential alignment with ASAM OpenODD.

#### 5.3.3 Map and Localization Contracts

HD map accuracy and freshness are assumptions of the planning and control layers:
- **Map accuracy contract**: Lane boundary positions accurate to +/- 0.1m, curvature accurate to +/- 0.001 m^-1
- **Localization contract**: Vehicle position accurate to +/- 0.3m (from GNSS + IMU + map matching)
- **Map freshness**: Map data reflects current road geometry (construction zones detected within X hours)
- **Validation**: Continuous comparison of perceived environment against map; map-perception discrepancy triggers degraded mode

---

## 6. The Bridge: How Zone 1 Constrains Zone 2

This section describes the architectural patterns that connect the proved core to the monitored envelope, making the overall system safe despite unproved components.

### 6.1 Safety Envelope Enforcement Pattern

The fundamental safety pattern:

```
Zone 2 Component (e.g., NN-based planner)
    | produces candidate output
    v
Zone 1 Safety Filter (formally verified)
    | checks: does output satisfy safety contract?
    +-- YES --> forward output to actuator/downstream
    +-- NO  --> substitute verified fallback output
               + escalate degradation level
```

Properties that must be formally proved about the safety filter:

| Property | Meaning | Verification Method |
|---|---|---|
| **Soundness** | If the filter accepts an output, the safety property holds | Proved as part of Layer 1 proof (KeYmaera X / CBF) |
| **Completeness** | The filter checks ALL conditions required by the safety proof | Traceability from Layer 1 proof assumptions to filter checks |
| **Implementation correctness** | The filter code correctly implements the check formula | Layer 2 verification (Arcanum / Frama-C) |
| **Runtime error freedom** | The filter code has no undefined behavior | Layer 3 verification (Astree / CBMC) |
| **Timeliness** | Filter + fallback execute within WCET budget | WCET analysis (aiT or measurement-based) |
| **Availability** | Filter does not reject safe outputs excessively | False rejection rate analysis (testing + statistical bounds) |

### 6.2 Runtime Monitor Derivation from Offline Proofs

Systematic process linking Layer 1 proofs to runtime enforcement:

**Step 1**: Prove safety property offline using KeYmaera X
- Input: hybrid program model + safety property (dL formula)
- Output: machine-checked proof + proof certificate

**Step 2**: ModelPlex extracts monitor condition from the proof
- Input: dL proof
- Output: arithmetic formula phi_monitor over observable state variables
- Proved property: "If phi_monitor holds at each control step, the offline safety proof applies to the actual execution"

**Step 3**: Implement monitor condition as runtime check
- The monitor formula is a conjunction of arithmetic inequalities
- Implementation is straightforward: comparisons and basic arithmetic
- Code complexity is low (typically tens of lines, not thousands)

**Step 4**: Verify the monitor implementation
- Layer 2: Arcanum/Frama-C proves implementation matches the formula
- Layer 3: Astree/Polyspace proves no runtime errors

**Step 5**: Connect monitor to fallback
- If monitor detects violation: trigger fallback trajectory (Section 5.2.3)
- Fallback itself is verified through the same Layer 1-2-3 pipeline

**Research needed -- extensions beyond current ModelPlex**:
- **Multi-agent monitors**: Monitor conditions for scenarios with multiple interacting vehicles (current ModelPlex handles single ego vehicle well; multi-agent requires compositional reasoning)
- **Perception-aware monitors**: Monitor conditions that account for perception uncertainty (current ModelPlex assumes perfect state observation; need to incorporate observation error bounds)
- **Temporal monitors**: Properties spanning multiple time steps (current ModelPlex checks per-step conditions; some safety properties require history, e.g., "the vehicle has been decelerating for at least 2 seconds before entering the intersection")
- **Adaptive monitors**: Monitor conditions that adapt to the current ODD parameters (e.g., tighter safety margins in rain)

### 6.3 Graceful Degradation and Fallback Hierarchy

The system operates in a hierarchy of modes, each with its own safety proof:

```
Level 0: Full Autonomy
  - Zone 2 (NN perception + complex planner) active
  - Zone 1 monitors: GREEN
  - Safety proof: Zone 1 monitors + Zone 2 statistical evidence
      |
      | Monitor detects anomaly OR confidence drops
      v
Level 1: Degraded Autonomy
  - Simplified perception (reduced feature set, conservative assumptions)
  - Verified simple planner (formally proved, limited ODD)
  - Reduced speed, reduced ODD
  - Safety proof: Full formal proof of simplified system
      |
      | Cannot maintain degraded mode
      v
Level 2: Minimal Risk Condition (MRC)
  - Controlled stop in safe location (shoulder, parking area)
  - Pre-planned MRC trajectory (formally verified offline)
  - Safety proof: Reachability analysis of MRC trajectory
      |
      | Cannot reach safe location
      v
Level 3: Emergency Stop
  - Immediate controlled braking to standstill
  - Safety proof: Braking dynamics reachability analysis
```

**What must be formally proved**:
- Each fallback level achieves its specified safe state
- Transitions between levels are safe (no gap in protection during transition)
- Transition triggers are well-defined and monitored (no ambiguous state)
- The system can always reach at least Level 3 (emergency stop is always available)

**Research needed**:
- Formal specification and verification of transition logic (when to escalate, when to de-escalate)
- Dwell time analysis (minimum time in each level before allowing de-escalation)
- Hysteresis design to prevent oscillation between levels
- Concurrent failure analysis (what if multiple monitors trigger simultaneously?)

---

## 7. System-Level Safety Argument

### 7.1 Composing Proofs into a Safety Case

Individual layer proofs must compose into a coherent system-level safety argument. We use **Goal Structuring Notation (GSN)** to structure the argument:

```
G1: "Vehicle is safe within the defined ODD"
|
+-- S1: "Argued over proved core + monitored envelope architecture"
|   |
|   +-- G2: "Control algorithm maintains safety properties"
|   |   Evidence: KeYmaera X proof (Layer 1)
|   |
|   +-- G3: "Software correctly implements the algorithm"
|   |   Evidence: Arcanum/Frama-C proof (Layer 2)
|   |
|   +-- G4: "Software has no runtime errors"
|   |   Evidence: Astree analysis (Layer 3)
|   |
|   +-- G5: "Hardware faults are detected and handled"
|   |   Evidence: Anomaly path DC analysis + implementation proof
|   |
|   +-- G6: "Safety monitors detect envelope violations"
|   |   Evidence: ModelPlex derivation + monitor implementation proof
|   |
|   +-- G7: "Fallback responses are safe"
|   |   Evidence: Fallback trajectory proofs (Layer 1-2-3)
|   |
|   +-- G8: "Perception meets contract requirements"
|       Evidence: Statistical validation + runtime monitoring
|
+-- C1: "System operates within defined ODD"
|   Validated by: ODD boundary monitor (Zone 1, proved)
|
+-- C2: "Hardware fault rates within specified budgets"
|   Validated by: Supplier FMEDA + anomaly path architecture
|
+-- C3: "Platform (OS, compiler) behaves correctly"
    Validated by: Supplier qualification + validation testing
```

### 7.2 Quantitative Composition of Deterministic and Probabilistic Evidence

**The key challenge**: Zone 1 provides deterministic guarantees ("always safe IF assumptions hold"), while Zone 2 provides probabilistic evidence ("perception meets contract with probability >= p"). How to combine these into a quantitative safety argument?

**Approach**:
1. Decompose the end-to-end failure into mutually exclusive scenarios:
   - Scenario A: Zone 2 output is correct AND Zone 1 processes it correctly --> Safe (deterministically, by Zone 1 proofs)
   - Scenario B: Zone 2 output is incorrect BUT Zone 1 monitor detects it AND fallback is safe --> Safe (deterministically, by monitor + fallback proofs)
   - Scenario C: Zone 2 output is incorrect AND Zone 1 monitor fails to detect it --> UNSAFE (residual risk)

2. P(unsafe) = P(Zone 2 incorrect) * P(monitor miss | Zone 2 incorrect)

3. P(Zone 2 incorrect) is bounded by perception contract validation (statistical evidence)
4. P(monitor miss | Zone 2 incorrect) depends on monitor coverage:
   - For model-based monitors (ModelPlex): if the error violates the dL model assumptions, the monitor detects it with probability 1 (deterministic). Only errors consistent with the model but outside the safety envelope are missed.
   - This is where the monitor design matters: the monitor must cover the failure modes that matter, not all possible failures.

**Research needed**: Rigorous mathematical framework for combining formal proofs with statistical evidence in safety cases. Current safety standards (ISO 26262, ISO/PAS 21448 SOTIF) handle them separately; a unified quantitative framework is an open problem.

### 7.3 ASIL Decomposition Across Zones

Applying ASIL decomposition (ISO 26262 Part 9) to the two-zone architecture:

- **Zone 2 components (perception, planning)**: ASIL B for the primary function
- **Zone 1 monitors constraining Zone 2**: ASIL D for the safety mechanism
- **Zone 1 control/safety core**: ASIL D (or ASIL D decomposed per anomaly path memo: ASIL B(D) normal path + ASIL B(D) anomaly path)

The ASIL decomposition requires independence between the decomposed elements. Independence between Zone 1 and Zone 2 must be demonstrated:
- **Functional independence**: Zone 1 monitors do not share code with Zone 2 components
- **Data independence**: Zone 1 monitors have independent inputs (or verified input integrity)
- **Execution independence**: Zone 1 and Zone 2 execute on separate cores or partitions with FFI

### 7.4 Gap Analysis and Residual Risk

Systematic checklist for the complete safety argument:

**Zone 1 internal gaps**:
- [ ] Every Layer 1 assumption maps to a Layer 2 precondition with bounds checking
- [ ] Every Layer 1 timing assumption maps to a verified WCET budget
- [ ] Every Layer 1 real-arithmetic computation maps to a bounded floating-point implementation
- [ ] Every Layer 2 precondition is covered by Layer 3 analysis
- [ ] Anomaly path covers all identified fault patterns with sufficient DC
- [ ] Safety monitor implementation is verified through Layers 2 and 3

**Zone 1 <--> Zone 2 gaps**:
- [ ] Every Zone 2 output consumed by Zone 1 has a defined contract
- [ ] Zone 1 monitors cover all safety-relevant failure modes of Zone 2
- [ ] Fallback trajectories are available for every monitor trigger scenario
- [ ] Degradation transitions are formally specified and verified

**Foundation gaps**:
- [ ] Every Foundation assumption (HW, OS, compiler, comm) is validated
- [ ] Runtime mechanisms exist to detect Foundation assumption violations
- [ ] Residual risk from undetected Foundation failures is quantified and acceptable

---

## 8. Cross-Cutting Concerns

### 8.1 Cybersecurity-Safety Integration

A cybersecurity breach can violate safety assumptions (spoofed sensor data, corrupted control commands, manipulated calibration parameters). ISO 21434 (Road vehicles -- Cybersecurity engineering) and UN R155/R156 require cybersecurity risk management.

**Integration points with this architecture**:

| Safety assumption | Cybersecurity threat | Mitigation |
|---|---|---|
| Sensor data within error bounds | Sensor spoofing (GPS, radar, camera) | Cross-sensor consistency check (Zone 1 monitor), authenticated sensor data |
| Control commands within safe range | Command injection via compromised ECU | Authenticated inter-ECU communication, safety monitor checks output bounds |
| Calibration parameters correct | Parameter manipulation via diagnostic port | Authenticated calibration access, runtime parameter integrity check (CRC) |
| Software integrity | Malicious software update | Secure boot, signed firmware, OTA authentication |

**Research needed**:
- Formal verification of cryptographic protocol implementations used for authentication
- Threat Analysis and Risk Assessment (TARA) systematically mapped to safety contracts
- Attack detection as a type of safety monitor (anomalous patterns in input data or system behavior trigger the same fallback mechanisms as safety monitors)

### 8.2 OTA Update and Re-Verification

When software is updated over-the-air, which proofs must be re-done?

**Modular re-verification principle**: If the contract at a component boundary is unchanged, only the updated component needs re-verification. Components that depend on it via contracts do not need re-verification.

**Practical workflow**:
1. Developer modifies a component
2. Contract change analysis: did any external contract (requires/ensures at module boundary) change?
3. If NO: re-verify only the modified component (Layers 2+3). All other proofs remain valid.
4. If YES: re-verify the modified component + all components whose assumptions depended on the changed guarantee. Layer 1 proofs may need update if algorithm behavior changed.

**Incremental proof strategies**:
- SMT solvers can cache proof obligations: unchanged obligations reuse previous proof
- Arcanum/Frama-C support incremental verification (only re-check modified functions)
- Layer 3 (Astree) supports incremental analysis for changed modules

**Research needed**:
- Automated contract change impact analysis tooling (given a code diff, determine which contracts changed and which proofs are affected)
- Efficient regression proof for Layer 1 (KeYmaera X proofs are typically monolithic; incremental proof techniques for dL are under-explored)
- Formal model of verification artifacts as a dependency graph with invalidation propagation

### 8.3 Multi-Core Timing Interference

Modern automotive SoCs run multiple functions on shared multi-core processors. Interference through shared resources (caches, memory bus, interconnect) can cause WCET violations, breaking timing contracts.

**SW-level mitigations**:
- **Temporal partitioning**: Time-triggered scheduling where each core has exclusive access to shared resources during its time slot
- **Cache partitioning**: Software-controlled cache coloring to prevent cache line conflicts between partitions
- **Memory bandwidth regulation**: Monitoring and limiting memory access rates per core (e.g., MemGuard pattern)
- **Interference-aware WCET analysis**: WCET analysis that accounts for worst-case interference from other cores

**Contract extension for interference**:
- Current timing contracts: "Function executes within WCET budget"
- Extended contracts: "Function executes within WCET budget assuming interference budget <= X memory accesses per millisecond from other cores"
- This makes the interference assumption explicit and verifiable

**Research needed**: Formal methods for multi-core interference analysis that integrate with the contract framework; current practice relies heavily on measurement-based estimation.

---

## 9. Research Roadmap

### 9.1 Priority Matrix

Prioritized by impact on the end-to-end safety argument and feasibility for a motor company:

| Priority | Research Topic | Section | Why | Builds On | Estimated Maturity |
|---|---|---|---|---|---|
| **P1** | Safety envelope enforcement pattern | 6.1 | Core architecture that makes everything else work | Existing Layer 1 + ModelPlex literature | Medium -- pattern exists, needs productization |
| **P1** | Perception contract specification | 5.1.2 | Without sensor/perception contracts, the chain of guarantees is ungrounded | Sensor characterization data | Low -- no industry standard yet |
| **P1** | ODD formal specification | 5.3.2 | Defines the boundary of the entire safety guarantee | ASAM OpenODD (emerging) | Low -- active standardization |
| **P2** | Verified safety filter for planning | 5.2.2 | Enables complex planners with formal safety guarantee | CBF theory, reachability tools | Medium -- theory mature, real-time implementation challenging |
| **P2** | Runtime monitor derivation extensions | 6.2 | Extends ModelPlex to multi-agent, perception-aware scenarios | KeYmaera X, existing Layer 1 work | Low-Medium -- research frontier |
| **P2** | Graceful degradation verification | 6.3 | Required for Level 3+ autonomy (no driver fallback) | Formal methods for state machines | Medium -- individual levels provable, transitions need work |
| **P2** | Quantitative proof composition | 7.2 | Needed for regulatory acceptance of mixed-evidence safety cases | Safety case theory, probability theory | Low -- open research problem |
| **P3** | NN verification for specific properties | 5.1.3 | Strengthens perception assurance | Marabou, alpha-beta-CROWN | Low-Medium -- tools exist, scalability limits |
| **P3** | Cybersecurity-safety integration | 8.1 | Regulatory trend (UN R155/R156) | ISO 21434, existing security tools | Medium -- frameworks exist, formal integration lacking |
| **P3** | Conformal prediction for perception | 5.1.4 | Provides calibrated uncertainty with guarantees | Statistics literature | Medium -- theory mature, automotive application new |
| **P4** | OTA re-verification tooling | 8.2 | Lifecycle cost reduction | Incremental verification research | Low -- tooling doesn't exist |
| **P4** | Multi-core interference contracts | 8.3 | Important for high-performance SoCs | WCET research community | Low -- measurement-based practice, formal methods early |
| **P4** | Sensor fusion verification | 5.1.5 | Strengthens multi-sensor guarantees | Bayesian/interval analysis | Low -- limited prior work |

### 9.2 Recommended Investigation Order

**Phase 1 (Foundation)**: Establish the core architectural pattern
1. Formalize the safety envelope enforcement pattern (Section 6.1) for one concrete use case (e.g., adaptive cruise control)
2. Define perception contracts for primary sensors on the target vehicle
3. Draft formal ODD specification for the initial target ODD

**Phase 2 (Expansion)**: Extend to full planning pipeline
4. Implement and verify a CBF-based safety filter for the motion planner
5. Extend ModelPlex monitors to multi-vehicle highway scenarios
6. Verify the graceful degradation state machine and transitions

**Phase 3 (Integration)**: System-level composition
7. Build the GSN safety case connecting all individual proofs
8. Develop the quantitative framework for combining deterministic and probabilistic evidence
9. Integrate cybersecurity threat analysis into safety contracts

**Phase 4 (Lifecycle)**: Sustaining verified systems
10. Build incremental re-verification tooling for OTA updates
11. Develop multi-core interference contract extensions
12. Extend NN verification to perception-specific safety properties

### 9.3 What Exists vs. What Must Be Built

| Topic | What Exists | What Must Be Built / Researched |
|---|---|---|
| Control algorithm verification | KeYmaera X, CORA, CBFs well established | Application to specific vehicle dynamics and ODD |
| Code verification | Frama-C mature; Arcanum under development | Arcanum tool completion; workflow integration |
| Anomaly path | Theory developed (existing memo) | Implementation and DC validation |
| Safety monitors | ModelPlex theory exists | Extensions for multi-agent, perception-aware monitors; production implementation |
| Perception contracts | No standard exists | Formal contract language and validation methodology |
| NN verification | Tools exist (Marabou etc.) but limited scale | Scalable property-specific verification for automotive NNs |
| Safety filters for planning | CBF theory mature | Real-time verified implementation; liveness proofs |
| ODD specification | ASAM OpenODD in draft | Formal machine-readable specification with runtime checking |
| Proof composition | GSN practice exists | Quantitative framework for mixed deterministic/probabilistic evidence |
| OTA re-verification | Incremental analysis in tools | Automated dependency tracking and selective re-proof |

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| ASIL | Automotive Safety Integrity Level (A-D, ISO 26262) |
| CBF | Control Barrier Function |
| dL | Differential dynamic logic |
| DC | Diagnostic Coverage |
| E2E | End-to-End protection (AUTOSAR) |
| FFI | Freedom From Interference |
| FIT | Failures In Time (per 10^9 hours) |
| FMEDA | Failure Mode Effects and Diagnostic Analysis |
| FTTI | Fault Tolerant Time Interval |
| GSN | Goal Structuring Notation |
| MRC | Minimal Risk Condition |
| ODD | Operational Design Domain |
| PMHF | Probabilistic Metric for random Hardware Failures |
| RSS | Responsibility-Sensitive Safety |
| SEU | Single Event Upset |
| SET | Single Event Transient |
| SOTIF | Safety Of The Intended Functionality (ISO/PAS 21448) |
| STL | Signal Temporal Logic |
| WCET | Worst-Case Execution Time |

## Appendix B: Key Tool References

| Tool | Purpose | Layer/Section |
|---|---|---|
| KeYmaera X | Hybrid system theorem prover (dL) | Layer 1, Section 4.1 |
| CORA | Reachability analysis for continuous/hybrid systems | Layer 1, Section 5.3.1 |
| Frama-C (WP) | Deductive verification for C | Layer 2, Section 4.2 |
| Arcanum | Deductive verification for safe C++ subset | Layer 2, Section 4.2 |
| SPARK/GNATprove | Deductive verification for Ada/SPARK | Layer 2, Section 4.2 |
| Astree | Abstract interpretation for C | Layer 3, Section 4.2 |
| Polyspace | Abstract interpretation for C/C++ | Layer 3, Section 4.2 |
| CBMC | Bounded model checking for C | Layer 3, Section 4.2 |
| CompCert | Formally verified C compiler | Foundation, Section 3.3 |
| Marabou | Neural network verification | Section 5.1.3 |
| alpha-beta-CROWN | Neural network verification (competition winner) | Section 5.1.3 |
| UPPAAL | Model checker for timed automata | Section 5.2.1 |
| CommonRoad | Motion planning benchmarks and reachability | Section 5.3.1 |
| aiT (AbsInt) | WCET analysis at binary level | Foundation, Section 3.3 |
