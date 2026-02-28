# End-to-End Proved Architecture Research Memos Implementation Plan

**Goal:** Produce 10 standalone research memos that expand each major topic from the end-to-end proved software architecture design document into a detailed, self-contained research document, matching the depth and style of the existing memos in the repository.

**Architecture:** Each memo takes one section (or group of closely related sections) from the design document at `/home/yotto/ad-adas-memo/.worktrees/memo/plans/2026-02-27-e2e-proved-architecture-design.md` and expands it into a full research document of 300-550 lines. Each memo follows the conventions established by the existing memos: a clear problem statement, detailed technical exposition with formulas and examples, state-of-the-art analysis with tool and paper references, open research challenges, and explicit connections to the other memos in the project. The priority ordering follows Section 9 of the design document.

**Tech Stack:** Markdown research documents; no code implementation. References to formal methods tools (KeYmaera X, Marabou, CORA, Frama-C, Arcanum, Astree, etc.) are descriptive, not executable.

**Strategy:** Team-driven

---

### Task 1: Perception Contract Specification Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/perception_contract_specification.md`

**Agent role:** senior-engineer

**Priority:** P1

**Step 1: Write the memo file with the following structure and content**

Create the file `/home/yotto/ad-adas-memo/.worktrees/memo/memo/perception_contract_specification.md` with 400-500 lines covering the following topics. Use the formatting conventions from `three_layer_formal_verification.md` (H1 title, H2 for major sections, H3 for subsections, tables for comparisons, inline math in code blocks, explicit assumption lists).

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - State the problem: Layer 1 safety proofs depend on assumptions about sensor/perception accuracy (e.g., `A_sensor`: sensor measurements within specified error bounds). These assumptions are currently ungrounded -- no formal specification or validation methodology exists to guarantee that perception meets them. Without perception contracts, the entire chain of guarantees collapses at the first link.
   - State the solution direction: formal assume-guarantee contracts for each sensor and perception component, with defined validation methodology and runtime degradation detection.
   - Connection to design doc Sections 5.1.1-5.1.6 and the contract framework in Section 2.

2. **Sensor Specification Contracts** (80-100 lines)
   - Define the contract formalism for sensors: `(A_sensor, G_sensor)` where assumptions cover operating conditions (illumination, temperature, contamination, EMI) and guarantees cover detection performance (recall, precision, latency, accuracy).
   - Provide concrete contract examples for:
     - Front camera (detection recall, position error, latency, false positive rate under stated illumination/contamination conditions)
     - Lidar (range accuracy, point density, angular resolution under stated weather conditions)
     - Radar (velocity accuracy, range accuracy under stated clutter conditions)
   - Discuss the structure: each guarantee is conditional on all assumptions being met. When an assumption is violated, the guarantee degrades according to a degradation model.
   - Reference: IEEE 2846 (Assumptions for Autonomous Driving), NHTSA AV framework.

3. **Validation Methodology for Contracts** (80-100 lines)
   - Statistical confidence intervals for performance claims. If we claim "detection recall >= 99.9%", what sample size is needed for a given confidence level? Derive using Clopper-Pearson exact binomial confidence intervals.
   - Provide the formula: for a claimed recall of `p` with confidence `1-alpha`, the required number of test cases `n` satisfies `P(X >= k | n, p_lower) >= 1-alpha` where `k` is the observed number of successes and `p_lower` is the lower confidence bound.
   - Concrete example: to validate recall >= 99.9% at 95% confidence with zero misses, need `n >= -ln(alpha) / (1 - p) = -ln(0.05) / 0.001 ~ 2996` positive examples.
   - Discuss the corner case problem: performance at the boundary of operating conditions (edge cases) vs. average performance. Need stratified testing across ODD parameter space.
   - Reference: ISO/PAS 21448 (SOTIF) Clause 10 (validation of absence of unreasonable risk), Koopman & Wagner "Challenges in Autonomous Vehicle Testing and Validation" (SAE, 2016).

4. **Degradation Models** (60-80 lines)
   - How sensor performance degrades as operating conditions approach contract boundary (e.g., rain intensity increases, illumination decreases).
   - Formalize as a function: `G_effective(condition) = G_nominal * degradation_factor(condition, boundary)`.
   - Discuss graceful vs. cliff-edge degradation. Sensors often exhibit cliff-edge degradation (works fine until it suddenly fails). Contracts must capture this.
   - Link to ODD boundary monitoring (design doc Section 4.4.4 and Task 3 of this plan): when degradation is detected, tighten safety margins or trigger degraded mode.
   - Reference: existing automotive sensor characterization studies, SAE J3016 ODD parameters.

5. **Probabilistic Contracts and Conformal Prediction** (60-80 lines)
   - Extend deterministic contracts to probabilistic ones: `P(guarantee | assumptions) >= p`.
   - Conformal prediction as a methodology for producing calibrated prediction intervals with guaranteed coverage. Explain the key property: `P(Y in C(X)) >= 1-alpha` holds for any model and any data distribution, requiring only exchangeability.
   - Discuss the challenge of sequential/temporal data in automotive perception (violation of exchangeability assumption). Reference adaptive conformal prediction (Gibbs & Candes, 2021).
   - How conformal prediction provides the probabilistic contracts needed by the quantitative proof composition framework (design doc Section 7.2 and Task 7 of this plan).

6. **Sensor Fusion Contract Composition** (40-60 lines)
   - How individual sensor contracts compose under fusion. If camera has position error <= 0.5m and lidar has position error <= 0.3m, what can be guaranteed about the fused estimate?
   - Discuss Bayesian fusion with verified interval bounds, Dempster-Shafer theory.
   - The key challenge: fusion can improve accuracy (when sensors agree) but can also introduce new failure modes (when sensors disagree due to different failure modes).
   - Redundancy analysis: prove system remains safe with any single sensor failed.

7. **Open Research Challenges** (30-40 lines)
   - No industry standard for perception contracts (IEEE 2846 is closest but insufficient).
   - Scalability of statistical validation across full ODD parameter space.
   - Temporal dependencies in sequential perception data.
   - Adversarial robustness as a perception contract property.
   - Connection to NN verification (Task 10 of this plan): property-specific NN verification can strengthen perception contracts beyond statistical evidence.

8. **Connection to Other Memos** (20-30 lines)
   - Link to `three_layer_formal_verification.md`: perception contracts ground the `A_sensor` assumptions of Layer 1 proofs.
   - Link to `deriving_anomarly_path_en.md`: perception degradation detection is analogous to anomaly path monitoring for hardware faults.
   - Link to this plan's other memos: ODD specification (Task 3), safety envelope (Task 2), quantitative composition (Task 7).

**Key references to research:**
- Benveniste et al., "Contracts for System Design" (2018)
- SPEEDS FP7 project on contract-based design
- Volk et al., conformal prediction surveys
- Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift" (2021)
- IEEE 2846 standard
- ISO/PAS 21448 (SOTIF)
- Koopman & Wagner, "Challenges in Autonomous Vehicle Testing and Validation" (2016)
- Czarnecki & Salay, "Towards a Framework to Manage Perceptual Uncertainty for Safe Automated Driving" (2018)

**Expected length:** 400-500 lines

**Step 2: Self-review the memo for completeness and consistency**

Verify: all 8 sections are present, formulas use code-block formatting, tables use markdown table syntax, references are specific (author/year/venue), connections to other memos are explicit, and the overall depth is comparable to `three_layer_formal_verification.md`.

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/perception_contract_specification.md
git commit -m "add perception contract specification research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 400 lines; contains formulas, concrete examples, tables | |
| All required sections present | 8 sections covering contracts, validation, degradation, conformal prediction, fusion, open challenges, connections | |
| References are specific | Each reference includes author, year, venue or standard number | |
| Connections to other memos are explicit | References to `three_layer_formal_verification.md`, `deriving_anomarly_path_en.md`, and at least 3 other planned memos | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 2: Safety Envelope Enforcement Pattern Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/safety_envelope_enforcement_pattern.md`

**Agent role:** senior-engineer

**Priority:** P1

**Step 1: Write the memo file with the following structure and content**

Create the file with 450-550 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - The safety envelope enforcement pattern is the core architectural mechanism that makes the "proved core + monitored envelope" approach work. It is the bridge between Zone 1 (proved) and Zone 2 (monitored).
   - Pattern: Zone 2 component produces candidate output -> Zone 1 safety filter checks against formal safety contract -> accept and forward, or reject and substitute verified fallback.
   - Connection to design doc Section 6.1, 6.2, and 4.4.

2. **Pattern Definition and Formal Properties** (80-100 lines)
   - Define the pattern formally using assume-guarantee contracts.
   - Six properties that must be formally proved about the safety filter (from design doc Section 6.1):
     - **Soundness**: If filter accepts, safety property holds. Verification method: part of Layer 1 proof (KeYmaera X / CBF).
     - **Completeness**: Filter checks ALL conditions required by safety proof. Verification method: traceability from Layer 1 proof assumptions to filter checks.
     - **Implementation correctness**: Filter code correctly implements check formula. Verification method: Layer 2 (Arcanum/Frama-C).
     - **Runtime error freedom**: Filter code has no undefined behavior. Verification method: Layer 3 (Astree/CBMC).
     - **Timeliness**: Filter + fallback execute within WCET budget. Verification method: WCET analysis.
     - **Availability**: Filter does not reject safe outputs excessively. Verification method: false rejection rate analysis.
   - Provide a table with verification method and evidence type for each property.
   - Discuss the distinction between safety (soundness) and performance (availability) requirements.

3. **ModelPlex: Automatic Monitor Derivation from Offline Proofs** (80-100 lines)
   - Full explanation of the ModelPlex pipeline (design doc Section 6.2):
     - Step 1: Offline safety proof using KeYmaera X (hybrid program + dL formula -> machine-checked proof)
     - Step 2: ModelPlex extracts monitor condition (arithmetic formula phi_monitor over observable state variables)
     - Step 3: Proved property: "If phi_monitor holds at each control step, offline safety proof applies to actual execution"
     - Step 4: Implement monitor condition as runtime check (simple arithmetic comparisons)
     - Step 5: Verify monitor implementation through Layers 2 and 3
   - Concrete example: derive monitor condition for adaptive cruise control. Show the dL formula, the hybrid program, and the extracted monitor condition as arithmetic inequalities.
   - Reference: Mitsch & Platzer, "ModelPlex: Verified Runtime Validation of Verified Cyber-Physical System Models" (FM 2014, extended version in FMSD 2016).

4. **Control Barrier Function (CBF) Based Enforcement** (60-80 lines)
   - CBFs as an alternative to ModelPlex for safety envelope enforcement.
   - Define CBF formally: scalar function B(x) where B(x) <= 0 is safe set, B(x) > 0 is unsafe.
   - Enforcement: at each control step, solve QP to find control input that satisfies `dB/dt <= -alpha * B(x)` while minimizing deviation from desired input.
   - Advantage: CBFs synthesize safe control directly (not just monitor), so the fallback IS the filter.
   - Disadvantage: CBF construction for complex systems is non-trivial; QP must be solved in real-time.
   - Reference: Ames et al., "Control Barrier Functions: Theory and Applications" (ECC 2019).

5. **RSS Enforcement Layer** (50-70 lines)
   - RSS as a specific instantiation of the safety envelope pattern.
   - RSS defines formal safety distances (longitudinal, lateral, intersection). Show the formulas from the design doc.
   - RSS enforcement: compute safety distances in real-time, override planner output if RSS violation imminent.
   - RSS advantage: derived from traffic regulations and physics, not from a specific algorithm model.
   - RSS limitation: conservative (may reject aggressive but safe maneuvers); assumes correct perception of other agents' velocities.
   - Reference: Shalev-Shwartz, Shammah, Shashua, "On a Formal Model of Safe and Scalable Self-driving Cars" (2017).

6. **Research Frontiers: Extensions Beyond Current ModelPlex** (60-80 lines)
   - **Multi-agent monitors**: Monitor conditions for scenarios with multiple interacting vehicles. Current ModelPlex handles single ego vehicle; multi-agent requires compositional reasoning.
   - **Perception-aware monitors**: Monitor conditions that account for perception uncertainty. Current ModelPlex assumes perfect state observation; need to incorporate observation error bounds.
   - **Temporal monitors**: Properties spanning multiple time steps (e.g., "vehicle has been decelerating for >= 2 seconds before entering intersection").
   - **Adaptive monitors**: Monitor conditions that adapt to current ODD parameters (e.g., tighter margins in rain).
   - For each extension, describe the technical challenge and current state of research.

7. **Implementation Considerations** (40-50 lines)
   - Monitor condition implementation: typically a conjunction of arithmetic inequalities, tens of lines of code, not thousands.
   - WCET budget: safety filter + fallback must complete within one control cycle (e.g., 10ms). This constrains monitor complexity.
   - Combining multiple safety criteria (RSS + CBF + road boundaries) without over-constraining the planner.
   - False rejection rate: a filter that rejects too aggressively makes the vehicle unusable (stops too often). Tuning between safety and availability.

8. **Connection to Other Memos** (20-30 lines)
   - Link to `three_layer_formal_verification.md`: the safety filter is verified through all three layers (Layer 1 for soundness, Layer 2 for implementation, Layer 3 for runtime error freedom).
   - Link to `deriving_anomarly_path_en.md`: the anomaly path architecture is a different kind of safety envelope (for hardware faults rather than algorithmic faults).
   - Link to other planned memos: perception contracts (Task 1), verified safety filter for planning (Task 4), graceful degradation (Task 5), ODD specification (Task 3).

**Key references to research:**
- Mitsch & Platzer, "ModelPlex: Verified Runtime Validation" (FMSD 2016)
- Ames et al., "Control Barrier Functions: Theory and Applications" (ECC 2019)
- Shalev-Shwartz et al., "On a Formal Model of Safe and Scalable Self-driving Cars" (2017)
- Platzer, "Logical Foundations of Cyber-Physical Systems" (Springer 2018)
- Gao, Kong, Clarke, "dReal: An SMT Solver for Nonlinear Theories of Reals" (CADE 2013)

**Expected length:** 450-550 lines

**Step 2: Self-review the memo for completeness and consistency**

Verify: all 8 sections present, concrete ACC example included, formulas formatted correctly, references specific, connections to other memos explicit.

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/safety_envelope_enforcement_pattern.md
git commit -m "add safety envelope enforcement pattern research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 450 lines; concrete ACC example with dL formula and monitor condition | |
| All required sections present | 8 sections covering pattern definition, ModelPlex, CBF, RSS, research frontiers, implementation, connections | |
| Six properties table present | Table with soundness, completeness, implementation correctness, runtime error freedom, timeliness, availability | |
| References are specific | Each reference includes author, year, venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 3: ODD Formal Specification Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/odd_formal_specification.md`

**Agent role:** senior-engineer

**Priority:** P1

**Step 1: Write the memo file with the following structure and content**

Create the file with 350-450 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - The ODD defines the boundary of the entire safety guarantee. Every Layer 1 proof has assumptions about operating conditions (speed, road curvature, weather, sensor availability). The ODD is the conjunction of all these assumptions.
   - Problem: ODD specifications are currently natural-language descriptions, not formal mathematical objects. This makes automated runtime checking, regulatory submission, and proof assumption traceability impossible.
   - Connection to design doc Section 5.3.2, 4.4.4, and Section 9 priority P1.

2. **Current State: ODD Specification Practice** (40-60 lines)
   - SAE J3016 defines ODD conceptually but provides no formal specification language.
   - BSI PAS 1883:2020 provides ODD taxonomy (static attributes, dynamic attributes, connectivity attributes) but no formalism.
   - ASAM OpenODD (in development): aims to provide a standardized, machine-readable ODD specification format. Describe current status and approach.
   - NHTSA Voluntary Safety Self-Assessment guidance: requires ODD description but no format.
   - Current practice in industry: natural language ODD descriptions in safety cases, manually checked.

3. **Formal ODD Parameter Language** (80-100 lines)
   - Propose a formal structure for ODD specification. Each ODD parameter is a typed variable with a domain:
     - Continuous parameters: speed in [0, 130] km/h, visibility >= 100m, friction >= 0.5
     - Discrete parameters: road_type in {highway, limited_access}, weather in {clear, light_rain, heavy_rain}
     - Boolean parameters: all_sensors_operational = true, v2x_available = false
   - The ODD is a conjunction: `ODD = P1 AND P2 AND ... AND Pn`
   - Show a concrete formal ODD specification (the example from design doc Section 5.3.2, expanded with more parameters and structure).
   - Discuss hierarchical ODD: base ODD for the system, extended ODD for specific features.
   - Discuss temporal ODD constraints: "visibility >= 100m for at least the past 10 seconds" (not just instantaneous).

4. **Mapping ODD Parameters to Layer 1 Proof Assumptions** (60-80 lines)
   - Each Layer 1 safety proof has explicit assumptions. These assumptions must map to ODD parameters.
   - Provide a traceability table showing the mapping. For example:
     - Layer 1 proof of lane keeping assumes road curvature <= 0.01 m^-1 -> ODD parameter: curvature <= 0.01 m^-1
     - Layer 1 proof of ACC assumes sensor error <= 0.1m -> ODD parameter: all primary sensors operational AND visibility >= 100m (which implies sensor accuracy within bounds)
   - Discuss the challenge: some Layer 1 assumptions map directly to measurable ODD parameters; others are indirect (e.g., "other vehicles behave rationally" is an assumption that cannot be directly measured).
   - The ODD must be conservative: if any Layer 1 assumption cannot be guaranteed by runtime monitoring, the ODD must exclude the corresponding conditions.

5. **Runtime ODD Boundary Monitoring** (60-80 lines)
   - Implementation of the ODD boundary monitor (design doc Section 4.4.4).
   - For each ODD parameter, define a runtime monitoring mechanism:
     - Weather conditions: rain sensor intensity, visibility estimation from camera
     - Road type: HD map lookup, lane marking detection
     - Sensor health: self-diagnostics, cross-sensor consistency
     - Localization quality: GNSS accuracy estimate, map-matching confidence
   - Define hysteresis and debouncing for ODD boundary transitions (avoid oscillation at boundaries).
   - When ODD boundary is approached: tighten safety margins. When ODD is exited: trigger graceful degradation (link to Task 5).
   - Verification of ODD monitor: the monitor itself is a Zone 1 component, verified through Layers 2 and 3.

6. **Regulatory Submission Considerations** (30-40 lines)
   - UN R157 (ALKS): requires formal ODD definition for type approval. Discuss the requirements.
   - How a formal ODD specification facilitates regulatory review: machine-readable, unambiguous, traceable to safety proofs.
   - Alignment with ISO 34503 (ODD taxonomy standard, in development).
   - The regulatory case: "the system is provably safe within this formally defined ODD, and the ODD boundary monitor provably detects when the system leaves the ODD."

7. **Open Research Challenges** (30-40 lines)
   - Machine-readable AND human-readable ODD specification language.
   - Dynamic ODD: ODD that can be expanded/contracted based on system confidence.
   - ODD validation: how to prove that the ODD boundary monitor covers all ODD parameters.
   - Multi-system ODD interaction: when multiple AD systems operate in overlapping ODDs.

8. **Connection to Other Memos** (20-30 lines)
   - Link to `three_layer_formal_verification.md`: ODD parameters ground Layer 1 proof assumptions.
   - Link to perception contracts (Task 1): sensor operating conditions are ODD parameters.
   - Link to graceful degradation (Task 5): ODD boundary violation triggers degradation.
   - Link to safety envelope (Task 2): ODD monitoring is a specific instance of the safety envelope pattern.

**Key references to research:**
- SAE J3016 (Taxonomy and Definitions for Terms Related to Driving Automation)
- BSI PAS 1883:2020 (ODD taxonomy)
- ASAM OpenODD specification (draft)
- UN R157 (Automated Lane Keeping Systems)
- ISO 34503 (ODD taxonomy, in development)
- Czarnecki, "Operational Design Domain for Automated Driving Systems" (Waterloo, 2018)
- Thorn et al., "A Framework for Automated Driving System Testable Cases and Scenarios" (NHTSA, 2018)

**Expected length:** 350-450 lines

**Step 2: Self-review the memo for completeness and consistency**

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/odd_formal_specification.md
git commit -m "add ODD formal specification research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 350 lines; concrete ODD specification example, traceability table | |
| All required sections present | 8 sections covering current state, formal language, proof mapping, runtime monitoring, regulatory, challenges, connections | |
| Concrete ODD example present | Machine-readable ODD specification with typed parameters | |
| References are specific | Standards cited with number/year, academic papers with author/year/venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 4: Verified Safety Filter for Motion Planning Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/verified_safety_filter_motion_planning.md`

**Agent role:** senior-engineer

**Priority:** P2

**Step 1: Write the memo file with the following structure and content**

Create the file with 400-500 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - Motion planners (trajectory optimization, sampling-based planning, RL-based planning) are too complex to verify directly. The safety filter pattern allows using unverified planners with formal safety guarantees.
   - Pattern: unverified planner produces candidate trajectory -> verified safety filter checks collision freedom and physical feasibility -> accept or substitute verified fallback.
   - Connection to design doc Section 5.2.2, 5.2.3, and Section 6.1.

2. **Control Barrier Function (CBF) Theory for Safety Filtering** (80-100 lines)
   - Full mathematical treatment of CBFs for safety filtering.
   - Define CBF: `B : R^n -> R` such that the zero-superlevel set `{x : B(x) >= 0}` is the safe set (note: convention varies; follow Ames et al. 2019 convention).
   - Safety condition: `dB/dt >= -alpha(B(x))` where alpha is a class-K function.
   - For control-affine systems `dx/dt = f(x) + g(x)u`, the safety condition becomes a linear constraint on u: `L_f B(x) + L_g B(x) * u >= -alpha(B(x))`.
   - QP-based safety filter: minimize `||u - u_ref||^2` subject to CBF constraint. This modifies the planned control input minimally to ensure safety.
   - Provide concrete example: CBF for collision avoidance with a leading vehicle. Define B(x) = d - d_safe(v) where d is distance and d_safe(v) is RSS safe distance at velocity v.
   - Reference: Ames et al., "Control Barrier Functions: Theory and Applications" (ECC 2019), Ames et al., "Control Barrier Function Based Quadratic Programs for Safety Critical Systems" (IEEE TAC 2017).

3. **Reachability-Based Trajectory Validation** (60-80 lines)
   - Alternative to CBF: compute reachable set of planned trajectory and verify it does not intersect obstacles.
   - Forward reachable set computation: given initial state and control sequence, compute all possible states accounting for tracking error and disturbances.
   - If reachable set does not intersect any obstacle's predicted reachable set (from Task 1 / design doc Section 5.3.1): trajectory is safe.
   - Comparison with CBF: reachability-based validation checks entire trajectory ahead of time (open-loop); CBF enforces safety at each control step (closed-loop).
   - Tools: CORA (zonotope-based), JuliaReach, CommonRoad safety verification.
   - Reference: Althoff, "An Introduction to CORA" (ARCH 2015), Althoff & Dolan, "Online Verification of Automated Road Vehicles using Reachability Analysis" (IEEE T-RO 2014).

4. **RSS as a Safety Filter** (40-60 lines)
   - RSS constraints as an alternative safety filter implementation.
   - Check planned trajectory at each time step against RSS safety distances for all detected traffic participants.
   - Advantage: no CBF construction needed; RSS distances are simple formulas.
   - Disadvantage: RSS is conservative; may reject trajectories that are safe but violate RSS distance formulas.
   - Combining RSS with CBF: use RSS for multi-agent safety distances, CBF for road boundary safety.

5. **Formal Verification of the Safety Filter Itself** (60-80 lines)
   - The safety filter must be verified through all three layers:
     - Layer 1: Prove that the CBF/RSS condition implies safety (KeYmaera X or barrier certificate proof).
     - Layer 2: Prove the QP solver implementation correctly solves the optimization problem (Arcanum/Frama-C contracts on the solver function).
     - Layer 3: Prove no runtime errors in the safety filter code (Astree/Polyspace).
   - Special challenge: QP solver verification. The solver must be simple enough to verify (e.g., explicit active-set solver for small QPs, not a general-purpose solver).
   - Reference: verification of optimization-based controllers, Roux et al., "Formal Proofs of Rounding Error Bounds" (JAR 2014).

6. **Liveness: The Filter Must Not Block Progress** (50-60 lines)
   - Safety alone is insufficient: the system must also make progress (liveness).
   - A safety filter that keeps the vehicle stopped forever is "safe" but useless.
   - Formal liveness property: "For any safe state, there exists a control input that satisfies the CBF constraint AND makes progress toward the goal."
   - This is much harder to prove than safety. Current CBF theory primarily addresses safety; liveness proofs are an active research area.
   - Discuss controlled invariance with progress: the safe set must be both forward invariant AND contain paths to the goal.
   - Reference: Mestres & Cortés, "Optimization-Based Safe Stabilizing Feedback with Guaranteed Region of Attraction" (CDC 2021).

7. **Real-Time Computational Challenges** (40-50 lines)
   - QP solve time must fit within control cycle (e.g., 10ms). For simple CBF constraints, the QP is small (a few variables, a few constraints) and solvable in microseconds.
   - For complex scenarios (multiple obstacles, multiple CBF constraints, road boundary constraints), the QP grows. Discuss scalability.
   - Reachability computation time: zonotope propagation for a fixed trajectory is typically O(milliseconds); full reachable set computation for nonlinear dynamics may be O(seconds).
   - Pre-computation strategies: pre-compute reachable sets offline for common scenarios; use online computation only for the residual.

8. **Open Research Challenges** (30-40 lines)
   - CBF construction for complex vehicle dynamics (beyond simple point-mass models).
   - Multi-objective safety filtering (combine collision avoidance, road boundary, comfort constraints).
   - Safety filter for discrete planning decisions (behavior planning, not just trajectory tracking).
   - Integration of learning-based planners with verified safety filters (safe RL).

9. **Connection to Other Memos** (20-30 lines)
   - Link to safety envelope (Task 2): the safety filter is a concrete instantiation of the safety envelope pattern.
   - Link to `three_layer_formal_verification.md`: the filter is verified through all three layers.
   - Link to graceful degradation (Task 5): when the safety filter rejects too many trajectories, escalate to degraded mode.

**Key references to research:**
- Ames et al., "Control Barrier Functions: Theory and Applications" (ECC 2019)
- Ames et al., "Control Barrier Function Based Quadratic Programs" (IEEE TAC 2017)
- Althoff, "An Introduction to CORA" (ARCH 2015)
- Shalev-Shwartz et al., "On a Formal Model of Safe and Scalable Self-driving Cars" (2017)
- Wabersich & Zeilinger, "A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems" (Automatica 2023)

**Expected length:** 400-500 lines

**Step 2: Self-review the memo for completeness and consistency**

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/verified_safety_filter_motion_planning.md
git commit -m "add verified safety filter for motion planning research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 400 lines; CBF math with Lie derivatives, concrete collision avoidance example | |
| All required sections present | 9 sections covering CBF theory, reachability, RSS, verification, liveness, real-time, challenges, connections | |
| CBF QP formulation present | Explicit optimization problem with objective and constraints | |
| References are specific | Each reference includes author, year, venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 5: Graceful Degradation and Fallback Hierarchy Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/graceful_degradation_fallback_hierarchy.md`

**Agent role:** senior-engineer

**Priority:** P2

**Step 1: Write the memo file with the following structure and content**

Create the file with 350-450 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - For Level 3+ autonomy, the system cannot simply shut down when a fault is detected -- it must degrade gracefully. The driver may not be able to take over immediately.
   - The design doc (Section 6.3) defines a four-level degradation hierarchy: Full Autonomy -> Degraded Autonomy -> Minimal Risk Condition (MRC) -> Emergency Stop.
   - Key challenge: each level must have its own safety proof, AND the transitions between levels must be safe.

2. **Four-Level Degradation Hierarchy** (80-100 lines)
   - Describe each level in detail:
     - **Level 0 (Full Autonomy)**: Zone 2 active (NN perception + complex planner). Zone 1 monitors GREEN. Safety argument: Zone 1 monitors + Zone 2 statistical evidence.
     - **Level 1 (Degraded Autonomy)**: Simplified perception, verified simple planner, reduced speed/ODD. Safety argument: full formal proof of simplified system.
     - **Level 2 (Minimal Risk Condition)**: Controlled stop in safe location. Pre-planned MRC trajectory verified offline. Safety argument: reachability analysis of MRC trajectory.
     - **Level 3 (Emergency Stop)**: Immediate controlled braking to standstill. Safety argument: braking dynamics reachability analysis.
   - For each level: what components are active, what safety proof applies, what ODD is assumed, what triggers entry, what triggers exit.
   - The key invariant: the system can ALWAYS reach at least Level 3 (emergency stop is always available).

3. **Per-Level Safety Proof Requirements** (60-80 lines)
   - Level 0: safety relies on Zone 1 monitors being sound and complete (proved) + Zone 2 meeting statistical contracts (validated). Mixed deterministic/probabilistic argument (link to Task 7).
   - Level 1: safety must be FULLY formally proved (all three layers). This is achievable because the simplified system is small enough for complete verification.
   - Level 2: MRC trajectory safety is proved offline using reachability analysis. Implementation of MRC trajectory tracking is verified through Layers 2 and 3.
   - Level 3: braking dynamics proved safe via reachability analysis. Braking controller implementation verified through Layers 2 and 3.
   - Discuss the tradeoff: higher levels provide more functionality but weaker safety guarantees; lower levels provide less functionality but stronger guarantees.

4. **Transition Logic Verification** (60-80 lines)
   - Formal specification of transitions between levels: triggers, guards, actions.
   - Model as a state machine with discrete states (levels) and transition conditions.
   - Verify using model checking (UPPAAL for timed automata, NuSMV for discrete state machines):
     - Safety: no transition leaves the system in an unprotected state.
     - Liveness: the system can always transition downward (toward safer modes) when needed.
     - Determinism: transition conditions are non-overlapping (no ambiguity about which level to enter).
   - Temporal properties:
     - "If a monitor triggers, the system reaches Level 1 or lower within T_transition milliseconds."
     - "Once in Level 2 (MRC), the vehicle reaches standstill within T_mrc seconds."
   - Show example LTL/CTL formulas for these properties.

5. **Dwell Time and Hysteresis Design** (40-60 lines)
   - Dwell time: minimum time in each level before allowing de-escalation (upward transition).
   - Purpose: prevent oscillation between levels when conditions are borderline.
   - Hysteresis: different thresholds for entering vs. leaving a degraded level. Enter Level 1 when confidence < 0.8; exit Level 1 only when confidence > 0.9 for at least T_dwell seconds.
   - Formal verification of hysteresis: prove that the hysteresis parameters prevent Zeno behavior (infinite switches in finite time).

6. **Concurrent Failure Analysis** (40-60 lines)
   - What happens when multiple monitors trigger simultaneously?
   - Worst case: perception fails AND hardware fault detected simultaneously.
   - The degradation logic must handle compound events: take the most conservative action (deepest degradation level).
   - Formal analysis: enumerate all possible monitor combinations and verify that for each combination, the degradation logic reaches a safe state.
   - Link to `deriving_anomarly_path_en.md`: dual-point fault analysis methodology applies here too.

7. **Open Research Challenges** (30-40 lines)
   - Formal verification of transition logic for continuous-time systems (not just discrete state machines).
   - Recovery: conditions under which the system can safely return to higher autonomy levels.
   - MRC planning: how to compute safe stop locations in real-time (depends on traffic, road geometry, available shoulder).
   - Fail-operational vs. fail-safe: design doc assumes fail-safe; Level 4/5 may require fail-operational degradation (backup control, not just shutdown).

8. **Connection to Other Memos** (20-30 lines)
   - Link to `deriving_anomarly_path_en.md`: fail-safe vs. fail-operational discussion, E-Gas monitoring architecture.
   - Link to safety envelope (Task 2): degradation is triggered by safety envelope violations.
   - Link to ODD specification (Task 3): ODD boundary violation is a degradation trigger.
   - Link to perception contracts (Task 1): perception degradation detection triggers Level 1.

**Key references to research:**
- ISO 26262 Part 3 (Concept phase, safe state definition)
- ISO/PAS 21448 (SOTIF, Clause 8: design verification and validation)
- SAE J3016 (Minimal Risk Condition definition)
- E-Gas monitoring concept (3-level monitoring)
- Maurer et al., "Autonomous Driving" (Springer, 2016) -- degradation concepts
- Alur & Dill, "A Theory of Timed Automata" (TCS 1994) -- for timed model checking
- UPPAAL documentation and case studies

**Expected length:** 350-450 lines

**Step 2: Self-review the memo for completeness and consistency**

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/graceful_degradation_fallback_hierarchy.md
git commit -m "add graceful degradation and fallback hierarchy research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 350 lines; state machine formalization, LTL/CTL formulas | |
| All required sections present | 8 sections covering hierarchy, per-level proofs, transitions, dwell time, concurrent failures, challenges, connections | |
| Four-level table present | Table with entry triggers, active components, safety proof method for each level | |
| References are specific | Standards and academic papers with author/year/venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 6: Runtime Monitor Extensions Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/runtime_monitor_extensions.md`

**Agent role:** senior-engineer

**Priority:** P2

**Step 1: Write the memo file with the following structure and content**

Create the file with 350-450 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - Current ModelPlex technology derives runtime monitors from KeYmaera X offline proofs. This is powerful but limited: it handles single ego vehicle with perfect state observation.
   - This memo explores four research extensions: multi-agent monitors, perception-aware monitors, temporal monitors, and adaptive monitors.
   - Connection to design doc Section 6.2 "Research needed" and Section 4.4.

2. **Background: ModelPlex Monitor Derivation** (40-60 lines)
   - Recap of the ModelPlex pipeline (brief, since the safety envelope memo covers it in detail -- reference Task 2).
   - Key limitation: the monitor condition phi_monitor is derived from a dL proof that models the ego vehicle as a hybrid system with perfectly observed state. This means:
     - Other vehicles are modeled as worst-case disturbances, not as agents with their own dynamics.
     - State variables (position, velocity, heading) are assumed to be exactly known.
     - The monitor checks conditions at a single time step; history is not considered.
     - The monitor condition is fixed; it does not adapt to changing conditions.

3. **Extension 1: Multi-Agent Monitors** (60-80 lines)
   - Problem: highway driving involves multiple interacting vehicles. The ego vehicle's safety depends not just on its own behavior but on the combined behavior of all agents.
   - Current approach: model other vehicles as worst-case bounded disturbances. This is sound but overly conservative.
   - Research direction: compositional reasoning. Model each vehicle as a separate hybrid system with its own safety envelope. The system-level monitor is the conjunction of pairwise monitors.
   - Challenge: the number of pairwise monitors grows quadratically with the number of agents. Need scalable composition.
   - Reference: Muller et al., "Compositional Verification of Cyber-Physical Systems" (Springer, 2021), Loos et al., "Adaptive Cruise Control: Hybrid, Distributed, and Now Formally Verified" (FM 2011).

4. **Extension 2: Perception-Aware Monitors** (60-80 lines)
   - Problem: ModelPlex assumes perfect state observation. In reality, state is estimated by the perception pipeline with bounded error.
   - Research direction: incorporate observation error bounds into the monitor condition. Instead of checking `B(x) >= 0`, check `B(x - e) >= 0` for all `||e|| <= e_max` where e_max is the perception error bound from the perception contract (Task 1).
   - This requires robust monitor conditions: `for all e in E: phi_monitor(x_observed - e)` holds.
   - For linear monitor conditions, this reduces to tightening thresholds by the perception error margin.
   - For nonlinear monitor conditions, this requires interval arithmetic or Taylor model evaluation.
   - Challenge: perception error bounds may be state-dependent (better at close range, worse at far range). The monitor must account for this.
   - Reference: Fan et al., "DryVR: Data-Driven Verification and Compositional Reasoning for Automotive Systems" (CAV 2017).

5. **Extension 3: Temporal Monitors** (50-70 lines)
   - Problem: some safety properties span multiple time steps. "The vehicle has been decelerating for at least 2 seconds before entering the intersection" is not a single-step check.
   - Current ModelPlex: checks a condition at each control step independently.
   - Research direction: extend to past-time temporal logic. Use Signal Temporal Logic (STL) monitors that evaluate formulas over a sliding window of past states.
   - STL monitoring is well-studied: tools like Breach, S-TaLiRo, rtamt provide efficient online STL monitors.
   - Challenge: integrating STL monitors with the dL proof framework. The offline proof must guarantee that the STL monitor condition is sufficient for safety.
   - Reference: Donze & Maler, "Robust Satisfaction of Temporal Logic over Real-Valued Signals" (FORMATS 2010), Deshmukh et al., "Robust Online Monitoring of Signal Temporal Logic" (FMSD 2017).

6. **Extension 4: Adaptive Monitors** (50-70 lines)
   - Problem: safety margins should adapt to current conditions. In rain, braking distance increases, so safe following distance should increase.
   - Current ModelPlex: monitor condition is fixed at design time.
   - Research direction: parameterized monitor conditions where parameters are ODD variables. The monitor condition becomes `phi_monitor(x, theta)` where theta includes weather, road conditions, etc.
   - The offline proof must show: for all theta in ODD, if `phi_monitor(x, theta)` holds at each step, the system is safe under conditions theta.
   - This connects to ODD formal specification (Task 3): ODD parameters become monitor parameters.
   - Challenge: the proof must cover the entire ODD parameter space, not just fixed parameter values.

7. **Perception Sanity Monitors** (40-50 lines)
   - Separate from model-based monitors (ModelPlex): simple runtime checks on perception output.
   - Object count bounds, velocity consistency, spatial consistency, temporal coherence, confidence calibration (from design doc Section 4.4.3).
   - These are simple enough to verify through Layers 2 and 3 without needing Layer 1 proofs.
   - They complement model-based monitors by catching perception failures that model-based monitors cannot detect (e.g., phantom objects).

8. **Open Research Challenges** (30-40 lines)
   - Scalability of multi-agent compositional verification.
   - Integration of STL monitoring with dL proof framework.
   - Adaptive monitors with formal guarantees across the full ODD parameter space.
   - Monitor synthesis: automated generation of perception sanity monitors from perception contracts.

9. **Connection to Other Memos** (20-30 lines)
   - Link to safety envelope (Task 2): monitors are the enforcement mechanism of the safety envelope.
   - Link to perception contracts (Task 1): perception-aware monitors depend on perception error bounds.
   - Link to ODD specification (Task 3): adaptive monitors are parameterized by ODD variables.
   - Link to `three_layer_formal_verification.md`: monitors are verified through all three layers.

**Key references to research:**
- Mitsch & Platzer, "ModelPlex" (FMSD 2016)
- Donze & Maler, "Robust Satisfaction of Temporal Logic" (FORMATS 2010)
- Deshmukh et al., "Robust Online Monitoring of Signal Temporal Logic" (FMSD 2017)
- Fan et al., "DryVR" (CAV 2017)
- Loos et al., "Adaptive Cruise Control: Hybrid, Distributed, and Now Formally Verified" (FM 2011)
- Heffernan et al., "An Analysis of the Effects of Resolution on an Automated Runtime Monitoring Approach for ISO 26262" (IET Software 2014)

**Expected length:** 350-450 lines

**Step 2: Self-review the memo for completeness and consistency**

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/runtime_monitor_extensions.md
git commit -m "add runtime monitor extensions research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 350 lines; formulas for robust monitoring, STL notation | |
| All required sections present | 9 sections covering four extensions, perception sanity monitors, challenges, connections | |
| Each extension has problem/direction/challenge structure | Clear problem statement, research direction, and technical challenge for each extension | |
| References are specific | Each reference includes author, year, venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 7: Quantitative Proof Composition Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/quantitative_proof_composition.md`

**Agent role:** senior-engineer

**Priority:** P2

**Step 1: Write the memo file with the following structure and content**

Create the file with 400-500 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - Zone 1 provides deterministic guarantees ("always safe IF assumptions hold"). Zone 2 provides probabilistic evidence ("perception meets contract with probability >= p"). The system-level safety argument must combine both.
   - Current safety standards (ISO 26262, ISO/PAS 21448) handle deterministic and probabilistic evidence separately. A unified quantitative framework is an open research problem.
   - Connection to design doc Section 7.1, 7.2, 7.3.

2. **Scenario Decomposition** (60-80 lines)
   - Decompose end-to-end failure into mutually exclusive scenarios (from design doc Section 7.2):
     - Scenario A: Zone 2 correct AND Zone 1 processes correctly -> Safe (deterministic)
     - Scenario B: Zone 2 incorrect BUT Zone 1 monitor detects it AND fallback safe -> Safe (deterministic, by monitor + fallback proofs)
     - Scenario C: Zone 2 incorrect AND Zone 1 monitor fails to detect -> UNSAFE
   - `P(unsafe) = P(Zone 2 incorrect) * P(monitor miss | Zone 2 incorrect)`
   - Discuss each term:
     - `P(Zone 2 incorrect)`: bounded by perception contract validation (statistical evidence, conformal prediction bounds from Task 1).
     - `P(monitor miss | Zone 2 incorrect)`: depends on monitor coverage. For model-based monitors (ModelPlex): if error violates dL model assumptions, monitor detects with probability 1. Only errors consistent with model but outside safety envelope are missed.
   - This decomposition makes the problem tractable: instead of proving the entire system, prove the deterministic parts formally and bound the probabilistic parts statistically.

3. **GSN Safety Case Structure** (60-80 lines)
   - Goal Structuring Notation (GSN) for organizing the safety argument.
   - Full GSN tree from design doc Section 7.1:
     - G1: Vehicle is safe within ODD -> decomposed into G2 through G8.
     - Show the full tree with evidence types for each goal.
   - Context nodes (C1: ODD, C2: HW fault rates, C3: platform correctness): these are the assumptions that, if violated, invalidate the safety case.
   - Discuss the distinction between:
     - Goals backed by formal proof (G2, G3, G4, G5, G6, G7): deterministic evidence.
     - Goals backed by statistical evidence (G8): probabilistic evidence.
   - Reference: Kelly, "The Goal Structuring Notation" (2004), GSN Community Standard V3.

4. **P(unsafe) Formulation and Computation** (80-100 lines)
   - Mathematical framework for computing P(unsafe).
   - For a single safety property (e.g., collision avoidance):
     ```
     P(collision) = P(perception error exceeds contract) * P(monitor miss | perception error)
                  + P(SW bug in Zone 1) * (proved to be 0 by formal verification)
                  + P(HW fault undetected) * P(safety impact | HW fault)
     ```
   - For multiple safety properties: treat each independently (conservative) or model correlations.
   - Discuss the challenge of bounding `P(monitor miss | perception error)`:
     - For errors that violate model assumptions (e.g., object teleportation): monitor detects with certainty.
     - For errors within model but outside safety bounds (e.g., object position off by exactly the right amount): monitor may miss. How to bound this?
     - Monitor coverage analysis: systematically enumerate perception failure modes and determine which are caught by the monitor.
   - Numerical example: show a concrete computation for a simplified ACC scenario.

5. **Gaps in ISO 26262 and ISO 21448** (40-60 lines)
   - ISO 26262 handles deterministic hardware failure rates (PMHF) and software verification. It does not address probabilistic performance of perception systems.
   - ISO 21448 (SOTIF) addresses perception inadequacies but provides no quantitative framework for combining deterministic and probabilistic evidence.
   - The gap: no standard provides a method for computing "P(unsafe) <= target" when some components have deterministic proofs and others have statistical bounds.
   - Potential approach: extend ISO 26262 PMHF framework to include software/perception failure rates. Define PMSF (Probabilistic Metric for Software Failures)?
   - Reference: ISO 26262:2018, ISO/PAS 21448:2022, UL 4600 (Standard for Safety for the Evaluation of Autonomous Products).

6. **ASIL Decomposition Across Zones** (40-60 lines)
   - From design doc Section 7.3:
     - Zone 2 (perception, planning): ASIL B for primary function.
     - Zone 1 monitors: ASIL D for safety mechanism.
     - Zone 1 core: ASIL D.
   - Independence requirements for ASIL decomposition: functional, data, execution independence between Zone 1 and Zone 2.
   - Discuss how independence is achieved in the architecture: separate cores/partitions, separate code, verified input integrity.
   - Reference: ISO 26262 Part 9 (ASIL-oriented and safety-oriented analyses), Annex D (ASIL decomposition).

7. **Open Research Challenges** (30-40 lines)
   - Unified quantitative safety framework combining formal proofs with statistical evidence.
   - How to quantify residual risk from unverified components (perception) in a way that regulators accept.
   - Compositional safety arguments: how to compose safety cases for interacting subsystems.
   - Machine-checkable safety cases: GSN that can be automatically verified for completeness and consistency.

8. **Connection to Other Memos** (20-30 lines)
   - Link to `three_layer_formal_verification.md`: the three layers provide the deterministic evidence for Zone 1 goals.
   - Link to `deriving_anomarly_path_en.md`: FMEDA-style analysis for HW fault contribution to P(unsafe).
   - Link to perception contracts (Task 1): P(Zone 2 incorrect) comes from perception contract validation.
   - Link to safety envelope (Task 2): monitor soundness is the deterministic component that bounds P(monitor miss).

**Key references to research:**
- Kelly, "The Goal Structuring Notation" (York, 2004)
- GSN Community Standard V3 (2021)
- ISO 26262:2018 Parts 5, 9
- ISO/PAS 21448:2022
- UL 4600 (Standard for Safety for the Evaluation of Autonomous Products)
- Bloomfield & Bishop, "Safety and Assurance Cases: Past, Present and Possible Future" (Safety Science, 2010)
- Koopman, "How Safe is Safe Enough?" (SAE, 2019)

**Expected length:** 400-500 lines

**Step 2: Self-review the memo for completeness and consistency**

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/quantitative_proof_composition.md
git commit -m "add quantitative proof composition research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 400 lines; P(unsafe) formula, numerical example, GSN tree | |
| All required sections present | 8 sections covering decomposition, GSN, P(unsafe), standards gaps, ASIL decomposition, challenges, connections | |
| Numerical example present | Concrete P(unsafe) computation for simplified scenario | |
| References are specific | Standards with numbers/years, papers with author/year/venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 8: Cybersecurity-Safety Integration Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/cybersecurity_safety_integration.md`

**Agent role:** senior-engineer

**Priority:** P3

**Step 1: Write the memo file with the following structure and content**

Create the file with 350-450 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - A cybersecurity breach can violate safety assumptions. Spoofed sensor data violates `A_sensor`; injected control commands violate `A_impl`; manipulated calibration parameters invalidate Layer 1 proofs.
   - Current practice treats cybersecurity (ISO 21434, UN R155/R156) and safety (ISO 26262) as separate concerns. This memo explores their integration within the proved architecture.
   - Connection to design doc Section 8.1.

2. **Threat Landscape for Safety-Critical AD Systems** (50-70 lines)
   - Map specific cybersecurity threats to safety contract violations (from design doc Section 8.1 table):
     - Sensor spoofing (GPS, radar, camera injection attacks) -> violates `A_sensor` (sensor data within error bounds)
     - Command injection via compromised ECU -> violates output range constraints
     - Calibration parameter manipulation via diagnostic port -> invalidates Layer 1 proof constants
     - Malicious software update -> violates software integrity assumption
   - For each threat: attack vector, required attacker capability (local/remote, physical access), and which safety contracts are broken.
   - Reference: Miller & Valasek, "Remote Exploitation of an Unaltered Passenger Vehicle" (2015), Upstream Automotive Cybersecurity Report (annual).

3. **ISO 21434 and UN R155/R156 Requirements** (40-60 lines)
   - ISO 21434 (Road vehicles -- Cybersecurity engineering): lifecycle cybersecurity management, TARA methodology, cybersecurity goals.
   - UN R155: Cyber Security Management System (CSMS) requirements for type approval.
   - UN R156: Software Update Management System (SUMS) requirements.
   - Discuss how these standards relate to but do not integrate with ISO 26262.

4. **TARA-to-Safety-Contract Methodology** (60-80 lines)
   - Propose a systematic mapping from TARA (Threat Analysis and Risk Assessment, ISO 21434) to safety contract violations.
   - For each threat identified in TARA:
     - Identify which safety contracts it can violate.
     - Determine the safety impact (which Layer 1 proofs are invalidated).
     - Define cybersecurity mitigation as a safety mechanism with its own contract.
   - Example: GPS spoofing -> violates localization contract -> invalidates Lane 1 proof of lane keeping -> Mitigation: cross-sensor consistency check (camera-based localization vs. GPS) -> Mitigation contract: "If GPS position deviates from camera-based position by more than X meters, flag localization as degraded."
   - The mitigation becomes a Zone 1 safety monitor (verified through Layers 2 and 3).

5. **Attack Detection as Safety Monitoring** (50-70 lines)
   - Key insight: attack detection and safety monitoring share the same architectural pattern.
   - Anomalous patterns in sensor data or system behavior can be detected by the same mechanisms that detect perception failures.
   - Cross-sensor consistency checks detect both sensor failures AND sensor spoofing.
   - Runtime parameter integrity checks (CRC) detect both hardware faults AND parameter manipulation.
   - This unification simplifies the architecture: a single monitoring framework handles both safety and security concerns.
   - Link to `deriving_anomarly_path_en.md`: the CRC/signature/encoding mechanisms designed for hardware fault detection also provide some cybersecurity protection.

6. **Formal Verification of Cryptographic Implementations** (40-60 lines)
   - Authenticated communication (AUTOSAR SecOC, TLS) protects against command injection and message spoofing.
   - Challenge: the cryptographic protocol implementation must itself be verified. A bug in the crypto implementation undermines the security claim.
   - State of the art: verified implementations of cryptographic primitives (HACL*, Vale, Jasmin).
   - Application to automotive: secure boot verification, firmware signature verification, inter-ECU authentication protocol verification.
   - This is a specialized application of Layer 2/3 verification to security-critical code.

7. **Open Research Challenges** (30-40 lines)
   - Systematic TARA-to-safety-contract mapping methodology.
   - Formal threat modeling integrated with contract-based design.
   - Quantifying cybersecurity risk in the P(unsafe) framework (Task 7): what is the probability of a successful attack?
   - Defense against adversarial ML attacks on perception NNs.
   - Supply chain security for software components (third-party libraries, open-source dependencies).

8. **Connection to Other Memos** (20-30 lines)
   - Link to `deriving_anomarly_path_en.md`: shared monitoring mechanisms for hardware faults and cyber attacks.
   - Link to perception contracts (Task 1): sensor spoofing violates perception contracts.
   - Link to safety envelope (Task 2): attack detection uses the safety envelope enforcement pattern.
   - Link to OTA re-verification (Task 9): secure update is a cybersecurity-safety integration point.

**Key references to research:**
- ISO 21434:2021 (Road vehicles -- Cybersecurity engineering)
- UN R155 (Cyber security) and R156 (Software update)
- Miller & Valasek, "Remote Exploitation of an Unaltered Passenger Vehicle" (2015)
- Protzenko et al., "Verified Low-Level Programming Embedded in F*" (ICFP 2017) -- HACL*
- SAE J3061 (Cybersecurity Guidebook for Cyber-Physical Vehicle Systems)
- Petit & Shladover, "Potential Cyberattacks on Automated Vehicles" (IEEE T-ITS 2015)

**Expected length:** 350-450 lines

**Step 2: Self-review the memo for completeness and consistency**

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/cybersecurity_safety_integration.md
git commit -m "add cybersecurity-safety integration research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 350 lines; threat-to-contract mapping table, TARA methodology example | |
| All required sections present | 8 sections covering threats, standards, TARA mapping, attack detection, crypto verification, challenges, connections | |
| Threat-to-contract table present | Table mapping specific threats to specific safety contract violations | |
| References are specific | Standards with numbers/years, papers with author/year/venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 9: OTA Re-Verification and Incremental Proof Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/ota_reverification_incremental_proof.md`

**Agent role:** senior-engineer

**Priority:** P4

**Step 1: Write the memo file with the following structure and content**

Create the file with 350-450 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - Over-the-air software updates change verified code. Which proofs must be re-done?
   - The modular re-verification principle: if the contract at a component boundary is unchanged, only the updated component needs re-verification.
   - This memo explores the practical challenges and open research problems in incremental re-verification.
   - Connection to design doc Section 8.2.

2. **Modular Re-Verification Principles** (60-80 lines)
   - Contract-based modularity: each component has a contract (requires/ensures). If the external contract is unchanged after a code modification, downstream components do not need re-verification.
   - Formal definition: component C1 depends on C2's contract (A2, G2). If C2 is updated to C2' with the same contract (A2, G2), the proof that C1 satisfies its own contract (A1, G1) remains valid.
   - The critical question: did any external contract change?
     - If NO: re-verify only the modified component (Layers 2+3).
     - If YES: re-verify modified component + all components whose assumptions depended on the changed guarantee. Layer 1 proofs may need update if algorithm behavior changed.
   - Provide a concrete example: updating a filter coefficient in a control algorithm. If the coefficient change does not affect the function's postcondition bounds, only the modified function needs re-verification. If it changes the output range, all callers must be re-checked.

3. **Contract Change Impact Analysis** (60-80 lines)
   - Given a code diff, determine which contracts changed and which proofs are affected.
   - Proposed approach:
     - Parse the diff to identify modified functions.
     - For each modified function, compare pre- and post-modification contracts.
     - If contract unchanged: mark only the function for re-verification.
     - If contract changed: trace all dependencies (which functions call this function or depend on its guarantees) and mark them for re-verification.
   - Dependency graph model: verification artifacts as a directed acyclic graph where nodes are function contracts and edges are dependencies.
   - Invalidation propagation: a contract change at node N invalidates all nodes reachable from N.
   - Tooling gap: no existing tool performs automated contract change impact analysis.

4. **Incremental Proof Strategies by Layer** (80-100 lines)
   - **Layer 3 (Astree/Polyspace)**: Abstract interpretation tools support incremental analysis. When a module changes, re-analyze only that module (if the module's interface is unchanged). Astree maintains analysis results for unchanged modules.
   - **Layer 2 (Frama-C/Arcanum)**: SMT solvers can cache proof obligations. Unchanged verification conditions reuse previous proof results. Frama-C WP supports this natively. Discuss cache invalidation: when does a changed function invalidate cached proofs of unchanged functions?
   - **Layer 1 (KeYmaera X)**: dL proofs are typically monolithic (a single proof for the entire hybrid system). Incremental proof techniques for dL are under-explored. Changing one part of the hybrid program (e.g., modifying the control law) invalidates the entire proof. Research needed: modular dL proofs where each component has its own sub-proof that can be re-verified independently.
   - For each layer, provide estimated re-verification time for typical changes (minutes for Layer 3, hours for Layer 2, days for Layer 1).

5. **Incremental SMT Caching** (40-60 lines)
   - SMT solvers (Z3, CVC5, Alt-Ergo) can cache satisfiability results.
   - When a function is modified, new verification conditions are generated. Many VCs are identical to previous versions (if the function change was localized).
   - SMT incremental mode: push/pop assertions, check satisfiability incrementally.
   - Practical challenge: VC generation may produce syntactically different but semantically equivalent formulas after minor code changes, defeating caching.
   - Reference: de Moura & Bjorner, "Z3: An Efficient SMT Solver" (TACAS 2008).

6. **Challenges in dL Proof Incrementality** (50-70 lines)
   - KeYmaera X proofs are based on sequent calculus with proof terms. Changing the hybrid program model requires re-proving from the changed point.
   - Current state: KeYmaera X does not support modular proofs. A change to the control law requires re-running the entire proof search.
   - Research direction: modular dL proofs. Decompose the hybrid system into components with contracts. Each component has its own sub-proof. When a component changes, only its sub-proof needs re-verification, provided its contract is unchanged.
   - This is analogous to modular software verification (Frama-C verifies each function independently given its contract).
   - Open problem: how to decompose a hybrid program into modular components with well-defined contracts in dL.
   - Reference: Muller et al., "A Component-Based Approach to Hybrid Systems Safety Verification" (IFM 2017).

7. **OTA Update Workflow Integration** (30-40 lines)
   - Practical workflow for OTA updates with re-verification:
     1. Developer modifies code.
     2. CI/CD pipeline runs contract change analysis.
     3. If no contract changes: run Layers 2+3 re-verification for modified components only.
     4. If contract changes: flag for manual review, run full re-verification of affected components.
     5. Generate re-verification report documenting what was re-verified and what was reused.
     6. Sign and deploy update.
   - Connection to cybersecurity (Task 8): OTA update authentication and integrity verification.

8. **Open Research Challenges** (30-40 lines)
   - Automated contract change impact analysis tooling.
   - Modular dL proofs for KeYmaera X.
   - Formal model of verification artifacts as a dependency graph with invalidation propagation.
   - Quantifying re-verification cost savings: how much faster is incremental vs. full re-verification?
   - Regulatory acceptance of incremental re-verification (can a regulator accept a partial re-verification report?).

9. **Connection to Other Memos** (20-30 lines)
   - Link to `three_layer_formal_verification.md`: each layer has different incremental verification capabilities.
   - Link to cybersecurity-safety integration (Task 8): secure OTA update process.
   - Link to safety envelope (Task 2): if a safety monitor is updated, its Layer 1 proof must be re-verified.

**Key references to research:**
- de Moura & Bjorner, "Z3: An Efficient SMT Solver" (TACAS 2008)
- Baudin et al., "ACSL: ANSI/ISO C Specification Language" (2022)
- Muller et al., "A Component-Based Approach to Hybrid Systems Safety Verification" (IFM 2017)
- Cuoq et al., "Frama-C: A Software Analysis Perspective" (SEFM 2012)
- UN R156 (Software Update Management System)

**Expected length:** 350-450 lines

**Step 2: Self-review the memo for completeness and consistency**

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/ota_reverification_incremental_proof.md
git commit -m "add OTA re-verification and incremental proof research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 350 lines; dependency graph model, per-layer incremental strategies | |
| All required sections present | 9 sections covering modularity, impact analysis, incremental strategies per layer, SMT caching, dL incrementality, workflow, challenges, connections | |
| Per-layer comparison table | Table comparing incremental verification capabilities of Layer 1/2/3 tools | |
| References are specific | Each reference includes author, year, venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

### Task 10: Neural Network Verification for Safety Properties Memo

**Files:**
- Create: `/home/yotto/ad-adas-memo/.worktrees/memo/memo/nn_verification_safety_properties.md`

**Agent role:** senior-engineer

**Priority:** P4

**Step 1: Write the memo file with the following structure and content**

Create the file with 400-500 lines covering the following topics.

**Section structure and required content:**

1. **Overview** (20-30 lines)
   - Neural networks are the least verifiable components in the AD stack. Global correctness proofs are infeasible. However, specific safety-relevant properties CAN be verified for bounded input regions.
   - This memo surveys the state of the art in NN verification, focusing on properties relevant to the proved architecture.
   - Connection to design doc Section 5.1.3 and the property-specific verification research direction.

2. **What is Verifiable Today** (60-80 lines)
   - **Local robustness**: For a given input x, prove that no perturbation within L_inf epsilon-ball changes the classification. Tools: Marabou, alpha-beta-CROWN, ERAN, VeriNet.
   - **Output bounds**: For inputs in a specified region, prove the output is within specified bounds. Useful for verifying NN output ranges match downstream contract assumptions.
   - **Monotonicity properties**: Prove certain input changes produce monotonic output changes (e.g., closer object -> larger bounding box).
   - For each property type: formal definition, verification approach, tool capabilities, and scalability limits.
   - Provide concrete examples from perception: "For any image where a pedestrian occupies at least 32x64 pixels and contrast ratio >= 2:1, the detection confidence is >= 0.9."

3. **What is NOT Verifiable Today** (30-40 lines)
   - Global correctness ("this NN always detects pedestrians").
   - Performance under arbitrary distribution shift.
   - Behavior on truly novel inputs.
   - Why these are not verifiable: the input space is too large, the property is too complex, or the notion of "correct" is undefined for novel inputs.

4. **Verification Tools Deep Dive** (80-100 lines)
   - **Marabou** (Hebrew University): SMT-based NN verification. Encodes NN as piecewise-linear constraints, solves using specialized SMT theory. Handles ReLU networks. Scalability: networks with hundreds to low thousands of neurons. Reference: Katz et al., "The Marabou Framework for Verification and Analysis of Deep Neural Networks" (CAV 2019).
   - **alpha-beta-CROWN** (UCLA/CMU): Branch-and-bound with linear relaxation. Winner of VNN-COMP (International Verification of Neural Networks Competition) in multiple years. Handles ReLU, sigmoid, and other activations. Currently the fastest general-purpose NN verifier. Reference: Wang et al., "Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints" (NeurIPS 2021).
   - **ERAN** (ETH Zurich): Abstract interpretation for neural networks. Uses DeepPoly and k-ReLU abstract domains. Reference: Singh et al., "An Abstract Domain for Certifying Neural Networks" (POPL 2019).
   - **VeriNet** (NTU Singapore): Symbolic bound propagation with LP relaxation.
   - For each tool: input format, supported architectures, verification guarantees, scalability benchmarks from VNN-COMP.

5. **Property-Specific Verification for Automotive Perception** (60-80 lines)
   - Research direction from design doc Section 5.1.3: rather than proving global NN correctness, prove only the specific properties that safety monitors need.
   - Example property: "If a pedestrian exists within 30m and meets minimum size criteria, the NN output bounding box overlaps the pedestrian's true position with IoU >= 0.5."
   - This is a much weaker (and more provable) property than "the NN correctly detects all pedestrians."
   - How to formulate automotive-specific verification properties:
     - Minimum detection distance for objects of minimum size.
     - Output range bounds for position estimates.
     - Monotonicity of detection confidence with object size/distance.
   - Connection to perception contracts (Task 1): these verified properties strengthen the perception contract beyond statistical evidence.

6. **Conformal Prediction as Complementary Approach** (40-60 lines)
   - When formal verification is infeasible (too large network, too complex property), conformal prediction provides distribution-free uncertainty quantification.
   - Key property: `P(Y in C(X)) >= 1-alpha` with no model or distribution assumptions.
   - Comparison with formal verification:
     - Formal verification: deterministic guarantee for bounded input region.
     - Conformal prediction: probabilistic guarantee for the data distribution.
   - They are complementary: use formal verification for critical safety properties on small networks; use conformal prediction for uncertainty quantification on large production networks.
   - Reference: Vovk et al., "Algorithmic Learning in a Random World" (2005), Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2023).

7. **Scalability Challenges and Open Problems** (40-50 lines)
   - Current NN verification tools handle networks with thousands to tens of thousands of neurons. Automotive perception networks have millions to billions of parameters.
   - Approaches to bridge the gap:
     - Verify only the safety-critical sub-network (e.g., the final classification layers).
     - Use abstraction: replace parts of the network with verified approximations.
     - Composition: verify smaller components and compose guarantees.
   - VNN-COMP benchmarks: discuss the gap between competition benchmarks and automotive-scale networks.
   - Training for verifiability: design NN architectures that are easier to verify (e.g., constrained architectures, Lipschitz-bounded networks).

8. **Open Research Challenges** (30-40 lines)
   - Scaling formal verification to automotive-size perception networks.
   - Verification of non-standard architectures (transformers, attention mechanisms).
   - Verification under distribution shift (training vs. deployment distribution mismatch).
   - Integration of NN verification results into the GSN safety case framework (Task 7).
   - Certified training: training procedures that guarantee the trained network satisfies specified properties.

9. **Connection to Other Memos** (20-30 lines)
   - Link to perception contracts (Task 1): NN verification can prove specific perception contract properties.
   - Link to safety envelope (Task 2): the safety filter provides safety even when NN verification is incomplete.
   - Link to quantitative proof composition (Task 7): NN verification results contribute to the P(unsafe) computation.
   - Link to `three_layer_formal_verification.md`: NN verification is a specialized form of Layer 1 reasoning applied to learned components.

**Key references to research:**
- Katz et al., "The Marabou Framework" (CAV 2019)
- Wang et al., "Beta-CROWN: Efficient Bound Propagation" (NeurIPS 2021)
- Singh et al., "An Abstract Domain for Certifying Neural Networks" (POPL 2019)
- Vovk et al., "Algorithmic Learning in a Random World" (2005)
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2023)
- VNN-COMP benchmark results (annual)
- Huang et al., "A Survey of Safety and Trustworthiness of Deep Neural Networks" (CSUR 2020)

**Expected length:** 400-500 lines

**Step 2: Self-review the memo for completeness and consistency**

**Step 3: Commit**

```bash
git add /home/yotto/ad-adas-memo/.worktrees/memo/memo/nn_verification_safety_properties.md
git commit -m "add neural network verification for safety properties research memo"
```

**Step 4: Review checkpoint**

**Review Checkpoint:**

| Check | Criteria | Pass/Fail |
|-------|----------|-----------|
| Depth matches existing memos | At least 400 lines; tool comparison table, concrete verification property examples | |
| All required sections present | 9 sections covering verifiable/not verifiable, tool deep dive, property-specific verification, conformal prediction, scalability, challenges, connections | |
| Tool comparison table | Table with tool name, approach, supported architectures, scalability limits | |
| References are specific | Each reference includes author, year, venue | |

Reviewer: `e2e-memo-reviewer-1` reviews diff for this task.
Action on CHANGES NEEDED: fix the issues, then re-review before starting the next task.

---

## Execution: Team-Driven

> **For Claude:** REQUIRED SUB-SKILL: Use [oneteam:skill] `team-management` skill to orchestrate
> execution starting from Phase 2 (Team Setup).

**Fragments:** 4

### Team Composition

| Name | Type | Scope |
|------|------|-------|
| e2e-memo-reviewer-1 | code-reviewer | All fragments |
| e2e-memo-senior-engineer-1 | senior-engineer | Fragment 1, Tasks 1, 2, 3 |
| e2e-memo-senior-engineer-2 | senior-engineer | Fragment 2, Tasks 4, 5, 6 |
| e2e-memo-senior-engineer-3 | senior-engineer | Fragment 3, Tasks 7, 8 |
| e2e-memo-senior-engineer-4 | senior-engineer | Fragment 4, Tasks 9, 10 |

Names use the `{group}-{role}-{N}` convention from the `team-management` skill.
These names are used as agent names when spawning and as `SendMessage` recipients.

**Reviewer count:** 1 for all fragments.
**Engineers:** 1 per fragment, all senior (every task classified as senior complexity).

### Fragment 1: P1 Foundation Memos
- **Tasks:** Task 1 (Perception Contract Specification), Task 2 (Safety Envelope Enforcement Pattern), Task 3 (ODD Formal Specification)
- **File scope:** `/home/yotto/ad-adas-memo/.worktrees/memo/memo/perception_contract_specification.md`, `/home/yotto/ad-adas-memo/.worktrees/memo/memo/safety_envelope_enforcement_pattern.md`, `/home/yotto/ad-adas-memo/.worktrees/memo/memo/odd_formal_specification.md`
- **Agent role:** senior-engineer
- **Inter-fragment dependencies:** none

#### Fragment 1: Post-Completion Review

| Stage | Reviewer | Criteria | Status |
|-------|----------|----------|--------|
| 1. Spec compliance | e2e-memo-reviewer-1 | All 3 memos present with required sections, depth >= 350 lines each, formulas and concrete examples included | |
| 2. Content quality | e2e-memo-reviewer-1 | References specific, cross-memo connections present, style matches existing memos, no factual errors in formal methods descriptions | |

Both stages must PASS before fragment is merge-ready.

### Fragment 2: P2 Expansion Memos
- **Tasks:** Task 4 (Verified Safety Filter for Motion Planning), Task 5 (Graceful Degradation and Fallback Hierarchy), Task 6 (Runtime Monitor Extensions)
- **File scope:** `/home/yotto/ad-adas-memo/.worktrees/memo/memo/verified_safety_filter_motion_planning.md`, `/home/yotto/ad-adas-memo/.worktrees/memo/memo/graceful_degradation_fallback_hierarchy.md`, `/home/yotto/ad-adas-memo/.worktrees/memo/memo/runtime_monitor_extensions.md`
- **Agent role:** senior-engineer
- **Inter-fragment dependencies:** none (all memos are standalone research documents that reference each other by name but do not depend on each other being written first)

#### Fragment 2: Post-Completion Review

| Stage | Reviewer | Criteria | Status |
|-------|----------|----------|--------|
| 1. Spec compliance | e2e-memo-reviewer-1 | All 3 memos present with required sections, depth >= 350 lines each, formulas and concrete examples included | |
| 2. Content quality | e2e-memo-reviewer-1 | References specific, cross-memo connections present, style matches existing memos, CBF math correct, state machine formalization present | |

Both stages must PASS before fragment is merge-ready.

### Fragment 3: P2-P3 Integration Memos
- **Tasks:** Task 7 (Quantitative Proof Composition), Task 8 (Cybersecurity-Safety Integration)
- **File scope:** `/home/yotto/ad-adas-memo/.worktrees/memo/memo/quantitative_proof_composition.md`, `/home/yotto/ad-adas-memo/.worktrees/memo/memo/cybersecurity_safety_integration.md`
- **Agent role:** senior-engineer
- **Inter-fragment dependencies:** none

#### Fragment 3: Post-Completion Review

| Stage | Reviewer | Criteria | Status |
|-------|----------|----------|--------|
| 1. Spec compliance | e2e-memo-reviewer-1 | Both memos present with required sections, depth >= 350 lines each, P(unsafe) formula present, threat-to-contract table present | |
| 2. Content quality | e2e-memo-reviewer-1 | References specific, cross-memo connections present, style matches existing memos, numerical examples present | |

Both stages must PASS before fragment is merge-ready.

### Fragment 4: P4 Lifecycle Memos
- **Tasks:** Task 9 (OTA Re-Verification and Incremental Proof), Task 10 (Neural Network Verification for Safety Properties)
- **File scope:** `/home/yotto/ad-adas-memo/.worktrees/memo/memo/ota_reverification_incremental_proof.md`, `/home/yotto/ad-adas-memo/.worktrees/memo/memo/nn_verification_safety_properties.md`
- **Agent role:** senior-engineer
- **Inter-fragment dependencies:** none

#### Fragment 4: Post-Completion Review

| Stage | Reviewer | Criteria | Status |
|-------|----------|----------|--------|
| 1. Spec compliance | e2e-memo-reviewer-1 | Both memos present with required sections, depth >= 350 lines each, tool comparison table present, per-layer incremental analysis present | |
| 2. Content quality | e2e-memo-reviewer-1 | References specific, cross-memo connections present, style matches existing memos, VNN-COMP benchmarks referenced | |

Both stages must PASS before fragment is merge-ready.

Fragment groupings are designed for parallel execution with worktree isolation. All fragments are fully independent (no memo depends on another being written first). The priority ordering (P1 -> P2 -> P3 -> P4) reflects research importance, not execution dependencies. All four fragments can execute in parallel.
