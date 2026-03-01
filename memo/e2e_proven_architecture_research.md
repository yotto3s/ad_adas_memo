# End-to-End Proven Software Architecture: Research Agenda

## Overview

This document identifies the research areas that must be investigated to realize an end-to-end proven software architecture for autonomous driving. It extends the existing work on three-layer formal verification, anomaly path derivation, and the Arcanum verification tool.

The architecture follows a **proven core + monitored envelope** pattern:
- **Zone 1 (Proven Core)**: Control algorithms, safety monitors, and software implementation -- all formally verified
- **Zone 2 (Monitored Envelope)**: Perception (neural networks), planning, world modeling -- safety ensured by proven monitors in Zone 1

The unifying principle is **contract-based composition**: assume-guarantee contracts at every component boundary, with the end-to-end safety argument as the transitive closure of all contracts being discharged.

**Full design document**: `plans/2026-02-27-e2e-proven-architecture-design.md`

---

## 1. Perception Contract Specification

### Problem

Layer 1 safety proofs depend on assumptions about sensor/perception accuracy (e.g., "sensor measurements within +/-0.1m"). These assumptions are currently ungrounded -- no formal specification or validation methodology exists. Without perception contracts, the chain of guarantees collapses at the first link.

### What to Investigate

- **Sensor specification contracts**: Formal assume-guarantee contracts for each sensor (camera, lidar, radar). Assumptions cover operating conditions (illumination, temperature, contamination). Guarantees cover detection performance (recall, precision, position error, latency).

- **Statistical validation methodology**: How to validate perception contracts with quantifiable confidence. Clopper-Pearson exact binomial confidence intervals for performance claims. Sample size requirements (e.g., to validate recall >= 99.9% at 95% confidence with zero misses: ~3000 positive examples). Stratified testing across ODD parameter space.

- **Degradation models**: How sensor performance degrades as conditions approach contract boundaries. Graceful vs. cliff-edge degradation characterization. Formal degradation functions.

- **Conformal prediction**: Distribution-free prediction intervals with guaranteed coverage. Addresses the challenge of probabilistic contracts. Key issue: sequential/temporal data in automotive perception violates exchangeability assumption -- adaptive conformal prediction methods needed.

- **Sensor fusion contract composition**: How individual sensor contracts compose under fusion. Bayesian fusion with verified interval bounds. Redundancy analysis (system safety with any single sensor failed).

### Key References

- Benveniste et al., "Contracts for System Design" (Foundations and Trends in EDA, 2018)
- IEEE 2846 (Assumptions for Autonomous Driving)
- ISO/PAS 21448 (SOTIF), Clause 10
- Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift" (2021)
- Czarnecki & Salay, "Towards a Framework to Manage Perceptual Uncertainty" (2018)

---

## 2. Safety Envelope Enforcement Pattern

### Problem

The safety envelope enforcement pattern is the core mechanism that makes the proven core + monitored envelope approach work. Zone 2 produces candidate outputs; Zone 1 filters check them against formal safety contracts and substitute verified fallbacks when violated. This pattern must be precisely defined and its properties formally established.

### What to Investigate

- **Pattern formal definition**: Six properties that must be proven about the safety filter -- soundness (if accepted, safety holds), completeness (all conditions checked), implementation correctness (code matches formula), runtime error freedom (no undefined behavior), timeliness (WCET budget), availability (low false rejection rate).

- **ModelPlex runtime monitor derivation**: The 5-step pipeline from KeYmaera X offline proof to verified runtime check. How monitor conditions are automatically extracted from dL proofs. Concrete examples (adaptive cruise control monitor derivation).

- **Control Barrier Function (CBF) enforcement**: CBFs as safety filters that not only monitor but synthesize safe control. QP-based filtering at each control step. Advantages (synthesizes safe control directly) and challenges (CBF construction for complex dynamics, real-time QP solving).

- **RSS enforcement layer**: RSS as a specific safety envelope instantiation. Formal safety distance formulas. Conservatism tradeoff (may reject aggressive but safe maneuvers).

- **Research frontiers beyond current ModelPlex**:
  - Multi-agent monitors (multiple interacting vehicles)
  - Perception-aware monitors (incorporating observation error bounds)
  - Temporal monitors (properties spanning multiple time steps, STL)
  - Adaptive monitors (parameters adjust to ODD conditions)

### Key References

- Mitsch & Platzer, "ModelPlex: Verified Runtime Validation" (FMSD, 2016)
- Ames et al., "Control Barrier Functions: Theory and Applications" (ECC, 2019)
- Shalev-Shwartz et al., "On a Formal Model of Safe and Scalable Self-driving Cars" (2017)
- Donze & Maler, "Robust Satisfaction of Temporal Logic over Real-Valued Signals" (FORMATS, 2010)

---

## 3. ODD Formal Specification

### Problem

The Operational Design Domain defines the boundary of the entire safety guarantee. Every Layer 1 proof has assumptions about operating conditions. Currently, ODD specifications are natural-language descriptions, making automated runtime checking and proof assumption traceability impossible.

### What to Investigate

- **Current state of ODD specification**: SAE J3016 (conceptual), BSI PAS 1883:2020 (taxonomy), ASAM OpenODD (machine-readable format, in development), UN R157 (ALKS type approval requirements).

- **Formal ODD parameter language**: Typed parameters (continuous, discrete, boolean) with domains. ODD as a conjunction of parameter constraints. Hierarchical ODD (base system ODD + feature-specific extensions). Temporal ODD constraints ("visibility >= 100m for the past 10 seconds").

- **Mapping ODD parameters to Layer 1 proof assumptions**: Traceability table from each proof assumption to a measurable ODD parameter. Challenge: some assumptions are indirect (e.g., "other vehicles behave rationally" cannot be directly measured).

- **Runtime ODD boundary monitoring**: For each ODD parameter, a runtime monitoring mechanism (rain sensor, visibility estimation, sensor self-diagnostics, localization quality). Hysteresis and debouncing for boundary transitions. The monitor itself is a Zone 1 component (verified through Layers 2+3).

- **Regulatory alignment**: UN R157 formal ODD requirements for type approval. ISO 34503 (ODD taxonomy standard, in development). Machine-readable format for regulatory submission.

### Key References

- SAE J3016, BSI PAS 1883:2020, ASAM OpenODD
- UN R157 (Automated Lane Keeping Systems)
- ISO 34503 (in development)
- Czarnecki, "Operational Design Domain for Automated Driving Systems" (2018)

---

## 4. Verified Safety Filter for Motion Planning

### Problem

Motion planners (trajectory optimization, sampling-based planning, RL) are too complex to verify directly. A verified safety filter pattern allows using unverified planners with formal safety guarantees: candidate trajectory is checked, and if unsafe, a verified fallback is substituted.

### What to Investigate

- **CBF theory for safety filtering**: Full mathematical treatment -- barrier function definition, safety condition (Lie derivative constraint), QP formulation for minimal intervention filtering. Concrete collision avoidance example.

- **Reachability-based trajectory validation**: Compute reachable set of planned trajectory, verify non-intersection with obstacles. Comparison with CBF (open-loop trajectory check vs. closed-loop per-step enforcement). Tools: CORA, JuliaReach, CommonRoad.

- **Formal verification of the filter itself**: The safety filter goes through all three verification layers. Special challenge: QP solver verification -- must use a simple enough solver (e.g., explicit active-set for small QPs).

- **Liveness proofs**: Safety alone is insufficient -- the system must make progress. Proving "for any safe state, there exists a control input that satisfies the CBF constraint AND makes progress toward the goal." This is an active research frontier.

- **Real-time computational efficiency**: QP solve time within control cycle (10ms). Scalability as number of obstacles/constraints grows. Pre-computation strategies.

### Key References

- Ames et al., "Control Barrier Function Based Quadratic Programs" (IEEE TAC, 2017)
- Althoff, "An Introduction to CORA" (ARCH, 2015)
- Wabersich & Zeilinger, "A Predictive Safety Filter for Learning-Based Control" (Automatica, 2023)

---

## 5. Graceful Degradation and Fallback Hierarchy

### Problem

For Level 3+ autonomy, the system cannot simply shut down on fault detection -- the driver may not be available. A multi-level degradation hierarchy is needed, each level with its own safety proof, and transitions between levels must themselves be verified.

### What to Investigate

- **Four-level hierarchy design**: Level 0 (full autonomy) -> Level 1 (degraded autonomy, simplified planner) -> Level 2 (Minimal Risk Condition, controlled stop) -> Level 3 (emergency stop). For each: active components, safety proof method, ODD assumed.

- **Per-level safety proofs**: Level 0 uses mixed deterministic/probabilistic argument. Level 1 must be fully formally proven (simplified system is small enough). Levels 2-3 use offline reachability analysis.

- **Transition logic verification**: Model transitions as a timed automaton. Verify with UPPAAL/NuSMV: no unprotected states, always able to degrade, deterministic transitions. LTL/CTL specifications for timing properties.

- **Dwell time and hysteresis**: Minimum time in each level before de-escalation. Different thresholds for entering vs. leaving degraded levels. Formal proof of no Zeno behavior (infinite switches in finite time).

- **Concurrent failure analysis**: Multiple monitors triggering simultaneously. Worst case: perception failure + hardware fault + ODD violation. Degradation logic must handle compound events.

### Key References

- ISO 26262 Part 3 (safe state definition)
- E-Gas monitoring concept
- Alur & Dill, "A Theory of Timed Automata" (TCS, 1994)
- UPPAAL model checking documentation

---

## 6. Runtime Monitor Extensions

### Problem

Current ModelPlex monitors handle single ego vehicle with perfect state observation. For full AD, monitors must be extended to multi-agent scenarios, imperfect perception, temporal properties, and varying conditions.

### What to Investigate

- **Multi-agent monitors**: Monitor conditions for multiple interacting vehicles. Compositional reasoning -- per-pair monitors with scalable composition. Challenge: quadratic growth in number of pairs.

- **Perception-aware monitors**: Incorporate observation error bounds into monitor conditions. Instead of checking B(x) >= 0, check B(x - e) >= 0 for all ||e|| <= e_max. For linear monitors, this reduces to tightening thresholds. For nonlinear, requires interval arithmetic.

- **Temporal monitors**: Properties spanning multiple time steps (e.g., "vehicle has been decelerating for >= 2 seconds before entering intersection"). Extend to STL (Signal Temporal Logic) monitors over sliding windows. Tools: Breach, rtamt.

- **Adaptive monitors**: Monitor parameters adjust to ODD conditions (tighter margins in rain). Parameterized monitor conditions phi_monitor(x, theta). Offline proof must cover entire ODD parameter space.

- **Perception sanity monitors**: Simple runtime checks on perception output -- object count bounds, velocity consistency, spatial consistency, temporal coherence. Simple enough for full Layer 2+3 verification.

### Key References

- Mitsch & Platzer, "ModelPlex" (FMSD, 2016)
- Deshmukh et al., "Robust Online Monitoring of Signal Temporal Logic" (FMSD, 2017)
- Fan et al., "DryVR: Data-Driven Verification" (CAV, 2017)
- Heffernan et al., runtime monitoring for ISO 26262 (IET Software, 2014)

---

## 7. Quantitative Proof Composition

### Problem

Zone 1 provides deterministic guarantees ("always safe IF assumptions hold"). Zone 2 provides probabilistic evidence ("perception meets contract with probability >= p"). No existing framework combines both into a quantitative system-level safety argument.

### What to Investigate

- **Scenario decomposition**: P(unsafe) = P(Zone 2 incorrect) * P(monitor miss | Zone 2 incorrect). The deterministic proofs (Zone 1) eliminate certain scenarios entirely, leaving only the probabilistic residual.

- **GSN (Goal Structuring Notation) safety case**: Structured argument connecting all individual proofs. G1 (vehicle safe in ODD) decomposed into sub-goals, each backed by formal proof or statistical evidence. Machine-checkable safety cases.

- **P(unsafe) computation**: Mathematical formulation with concrete numerical example (e.g., ACC scenario). Bounding P(monitor miss | perception error) -- which perception failure modes are caught by model-based monitors vs. which can escape.

- **Gaps in ISO 26262 and ISO 21448**: Neither standard provides a method for computing P(unsafe) when some components have deterministic proofs and others have statistical bounds. Potential PMSF (Probabilistic Metric for Software Failures) as an extension of PMHF.

- **ASIL decomposition across zones**: Zone 2 components at ASIL B + Zone 1 monitors at ASIL D. Independence requirements (functional, data, execution) and how to demonstrate them in the architecture.

### Key References

- Kelly, "The Goal Structuring Notation" (2004)
- ISO 26262:2018 Parts 5, 9
- ISO/PAS 21448:2022
- UL 4600 (Standard for Safety for the Evaluation of Autonomous Products)
- Koopman, "How Safe is Safe Enough?" (SAE, 2019)

---

## 8. Cybersecurity-Safety Integration

### Problem

A cybersecurity breach can violate safety assumptions: spoofed sensor data violates perception contracts, injected commands violate output constraints, manipulated calibration invalidates Layer 1 proofs. ISO 21434 (cybersecurity) and ISO 26262 (safety) are treated separately -- their integration is an open problem.

### What to Investigate

- **Threat-to-safety-contract mapping**: For each cybersecurity threat (sensor spoofing, command injection, parameter manipulation, malicious update), identify which safety contracts are violated and which Layer 1 proofs are invalidated.

- **TARA-to-safety-contract methodology**: Systematic mapping from Threat Analysis and Risk Assessment (ISO 21434) to safety contract violations. For each threat: identify violated contract, determine safety impact, define cybersecurity mitigation as a safety mechanism with its own contract.

- **Attack detection as safety monitoring**: Cross-sensor consistency checks detect both sensor failures AND sensor spoofing. Runtime parameter integrity checks (CRC) detect both hardware faults AND parameter manipulation. Unifying the monitoring framework simplifies the architecture.

- **Formal verification of crypto implementations**: Authenticated communication (AUTOSAR SecOC) protects against spoofing. The crypto implementation itself must be verified. State of the art: HACL*, Vale, Jasmin (verified crypto primitives).

- **Supply chain security**: Third-party libraries and open-source dependencies as attack surface. Software Bill of Materials (SBOM) integration with safety assurance.

### Key References

- ISO 21434:2021
- UN R155 (Cyber security), R156 (Software update)
- Miller & Valasek, "Remote Exploitation of an Unaltered Passenger Vehicle" (2015)
- Protzenko et al., "Verified Low-Level Programming Embedded in F*" (ICFP, 2017) -- HACL*

---

## 9. OTA Re-Verification and Incremental Proof

### Problem

When software is updated over-the-air, which proofs must be re-done? Full re-verification of everything is prohibitively expensive. Modular and incremental strategies are needed.

### What to Investigate

- **Modular re-verification principle**: If a component's external contract is unchanged after modification, downstream components do not need re-verification. Only the modified component goes through Layers 2+3.

- **Contract change impact analysis**: Given a code diff, determine which contracts changed and which proofs are affected. Verification artifacts as a dependency graph with invalidation propagation. No existing tool performs this automatically.

- **Incremental strategies per layer**:
  - Layer 3 (Astree): Supports incremental analysis for changed modules
  - Layer 2 (Frama-C/Arcanum): SMT solver caching; unchanged VCs reuse previous proofs
  - Layer 1 (KeYmaera X): dL proofs are typically monolithic; incremental techniques under-explored

- **Challenges in dL proof incrementality**: KeYmaera X proofs cannot be partially re-verified. Research direction: modular dL proofs -- decompose hybrid systems into components with contracts, re-verify only the changed component.

- **OTA workflow integration**: CI/CD pipeline with contract change analysis, selective re-verification, re-verification report generation, signed deployment.

### Key References

- de Moura & Bjorner, "Z3: An Efficient SMT Solver" (TACAS, 2008)
- Muller et al., "A Component-Based Approach to Hybrid Systems Safety Verification" (IFM, 2017)
- Cuoq et al., "Frama-C: A Software Analysis Perspective" (SEFM, 2012)
- UN R156 (Software Update Management System)

---

## 10. Neural Network Verification for Safety Properties

### Problem

Neural networks are the least verifiable components in the AD stack. Global correctness proofs are infeasible. However, specific safety-relevant properties CAN be verified for bounded input regions, and this is a rapidly advancing research area.

### What to Investigate

- **What is verifiable today**: Local robustness (perturbation within epsilon-ball preserves classification), output bounds (for bounded inputs, output is within specified range), monotonicity properties. Tools: Marabou (SMT-based), alpha-beta-CROWN (branch-and-bound, VNN-COMP winner), ERAN (abstract interpretation), VeriNet (LP relaxation).

- **What is NOT verifiable**: Global correctness, performance under arbitrary distribution shift, behavior on novel inputs. The input space is too large and "correct" is undefined for novel inputs.

- **Property-specific verification direction**: Rather than proving global NN correctness, prove only properties the safety monitors need. Example: "If a pedestrian exists within 30m meeting minimum size criteria, the NN output bounding box overlaps with IoU >= 0.5." This is weaker but provable.

- **Conformal prediction as complement**: When formal verification is infeasible (large networks), conformal prediction provides distribution-free uncertainty quantification. Formal verification for critical properties on small networks + conformal prediction for uncertainty on large production networks.

- **Scalability gap**: Current tools handle thousands to tens of thousands of neurons. Automotive perception networks have millions to billions of parameters. Approaches: verify safety-critical sub-network only, abstraction, composition, training for verifiability (Lipschitz-bounded architectures).

### Key References

- Katz et al., "The Marabou Framework" (CAV, 2019)
- Wang et al., "Beta-CROWN: Efficient Bound Propagation" (NeurIPS, 2021)
- Singh et al., "An Abstract Domain for Certifying Neural Networks" (POPL, 2019)
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2023)
- VNN-COMP benchmark results (annual)

---

## Priority and Phasing

| Priority | Topics | Rationale |
|---|---|---|
| **P1** | 1. Perception contracts, 2. Safety envelope, 3. ODD specification | Core architecture -- everything else depends on these patterns and contracts |
| **P2** | 4. Safety filter for planning, 5. Graceful degradation, 6. Runtime monitor extensions, 7. Quantitative proof composition | Expansion -- makes the architecture practical for full AD |
| **P3** | 8. Cybersecurity-safety integration | Cross-cutting -- regulatory trend (UN R155/R156) |
| **P4** | 9. OTA re-verification, 10. NN verification | Lifecycle + long-term research -- important but not blocking initial architecture |

## Connection to Existing Memos

| Existing Memo | Research Topics That Extend It |
|---|---|
| `three_layer_formal_verification.md` | Topics 2, 4, 6, 7, 9 (all relate to extending or composing Layer 1-3 proofs) |
| `deriving_anomarly_path_en.md` | Topics 5, 8 (degradation architecture, shared monitoring for HW faults and cyber attacks) |
| Arcanum docs | Topics 4, 9 (safety filter implementation verification, incremental verification tooling) |
| *New: this document* | All 10 topics connect to each other via the contract-based composition framework |
