# Deriving ASIL-D Anomaly Handling from Normal-Path Specs at Design Phase (Research Notes)

## 1. Problem Statement

Running everything on an ASIL-D core (lockstep etc.) is overkill. Want to asymmetrically split the normal path (ASIL-B core) and anomaly path (ASIL-D core), then systematically derive the minimum anomaly-path processing from **SW architecture artifacts at design phase** (input definitions, output specs, check algorithms, state transition design).

### Basic Idea

- **Normal path (ASIL-B core)**: Runs the algorithm + spec checks on I/O (plausibility, range checks, etc.). Produces reliable output as long as no HW faults (assuming no bugs)
- **Anomaly path (ASIL-D core)**: Doesn't re-execute the normal path's computation. Specializes in verifying the normal path's check mechanisms haven't been disabled by HW faults + minimal independent output verification
- **Derivable at design phase**: Don't need to wait for normal path implementation. Can determine anomaly path from SW architecture design info alone

### ISO 26262 Alignment

ASIL decomposition D = B(D) + B(D), asymmetrically separating intended function (normal path) and safety mechanism (anomaly path). Within scope of ISO 26262 Part 6.

---

## 2. Random Hardware Fault Model

Target is not SW bugs but random HW faults from external physical causes.

### 2.1 SEU — Memory/Register Bit Flips

Cosmic rays (neutrons) or alpha particles flip charge in storage elements. Transient, recovers on rewrite.

- **Where**: SRAM, cache, CPU registers, flip-flops
- **Effect**: Data value change (single or multi-bit)
- **Rate**: Hundreds to thousands of FIT/Mbit for automotive SRAM (FIT = failures per 10⁹ hours)
- **Scaling trend**: Qcrit drops with process shrink, increasing SEU susceptibility

Three natural masking effects: logical (fault doesn't propagate to output), electrical (pulse attenuates), temporal (pulse doesn't coincide with latch timing). Only manifests as soft error when all three fail.

### 2.2 SET — Transient Pulses in Combinational Logic

Particle strikes create transient voltage pulses in combinational circuits (ALU, comparators, etc.). Equivalent to SEU when latched.

- **Where**: ALU, address decoder, bus logic
- **Effect**: Temporary computation error, temporary address corruption
- **Increases with process shrink and higher clock frequencies**

### 2.3 Stuck-at Faults

Physical degradation (electromigration, hot carrier degradation, NBTI, etc.) causes signal lines or registers to stick at 0 or 1. Permanent fault.

- **Any logic element or wire**
- **Increases over time, no natural recovery**

### 2.4 Bridge Faults

Shorts between adjacent signal lines. One signal gets pulled to the other's value. Risk increases as wire spacing shrinks.

### 2.5 EMI (Electromagnetic Interference)

Automotive-specific (ignition noise, power line noise, external RF, etc.). Corrupts memory and bus data. Similar to SEU bit flips but can affect multiple bits in bursts. EMI can dominate over cosmic rays in automotive environments.

### 2.6 Power/Clock Anomalies

Power fluctuation or clock disturbance causes setup/hold time violations → metastability and data corruption. Can become Common Cause Failure (CCF) in lockstep cores.

### 2.7 Scope for This Work

Primary targets:
- **SEU/SET**: Highest frequency, easiest to analyze at SW level
- **EMI-induced bit flips**: Can be modeled similarly to SEU

Out of scope (handled by separate safety mechanisms):
- Stuck-at/bridge faults → BIST, latent fault detection
- Power/clock anomalies → Dedicated HW monitors

---

## 3. Fault Patterns Undetectable by Normal-Path Spec Checks

### 3.1 Practically Not a Problem

- **Small errors within spec tolerance**: Spec tolerances are designed for sensor noise etc., so HW-fault-induced small changes within tolerance are within acceptable inaccuracy
- **Input changes to noise-tolerant algorithms**: Image processing etc. barely affected by 1-bit input change. Input CRC would cause excessive false positives, hurting Availability. Output-side checks are sufficient

### 3.2 Requires Anomaly-Path Handling

| Pattern | Description | Why Undetectable |
|---|---|---|
| Input data corruption | Input corrupted in memory before reaching normal path | Normal path operates "correctly" on corrupted input |
| Control flow corruption | PC or branch condition bit flip | Spec check only sees "the result that was executed" |
| Check mechanism failure | Threshold/constant corruption, result flag corruption, check skip | Can't detect its own failure (self-reference problem) |

The most essential role of the ASIL-D core: **Verifying that the normal path's verification mechanisms are functioning correctly**

---

## 4. Anomaly Path (ASIL-D Core) Design

### 4.1 Architecture

Based on the **3-level monitoring pattern** derived from E-Gas. E-Gas itself was created by German OEMs (Audi, BMW, VW, Daimler, Porsche, etc.) for engine control, but the 3-level structural principle is widely referenced for brakes, steering, etc. Applying its 1oo1D (1-out-of-1 with Diagnostics) concept to AD/ADAS SW safety monitoring.

**Output data goes directly from normal path to actuator. Does not pass through ASIL-D core.** ASIL-D core is not a gatekeeper — it monitors normal path health and shuts off via independent HW shutoff path on anomaly detection.

```
[ASIL-B Core (L1: Function + L2: Function Monitoring)]
1. Acquire inputs
2. Execute algorithm
3. Output directly to actuator
4. Output validity check (spec check, plausibility, etc.)
5. Send {check result, check constant CRC, signature} to ASIL-D core

[ASIL-D Core (Part of L2 + L3: Controller Monitoring)]
1. Verify check result is "normal"
2. Verify check constant CRC matches startup value (constant corruption detection)
3. Verify signature matches expected value (control flow / check execution proof)
4. Minimal independent output plausibility check (fallback)
5. On anomaly → shut off via independent shutoff path, transition to safe state
```

This design structurally eliminates the output data transfer bit-flip problem. Only monitoring info (check results, etc.) is sent to the ASIL-D core, and transfer protection is covered by Section 4.2 countermeasures (encoding, signatures, CRC).

### 4.1.1 Fail-Safe vs Fail-Operational

This work assumes **fail-safe** (anomaly detection → transition to safe state). Consistent with 3-level monitoring pattern.

In AD/ADAS the assumption changes depending on automation level:

- **Fail-safe is OK**: Driver can intervene (Level 0–2, Level 3 if transition time is available). Shutdown or degraded mode reaches safe state. → Scope of this work
- **Fail-operational needed**: Driver can't intervene immediately (Level 3+, especially Level 4/5). Sudden shutdown on highway is dangerous. Must continue Minimal Risk Maneuver (retreat driving) after anomaly

Even for fail-operational, this monitoring architecture (spec check structure → anomaly monitor derivation, CRC/signature/encoding detection, DC evaluation) can be reused as-is. Only the post-detection action changes from "shutdown" to "switch to backup control." Additional design considerations though:

- **FTTI may get tighter**: Fail-safe only needs time to shutdown, but fail-operational requires backup control switchover + stable output within FTTI. If detection latency requirements tighten, might need partial signature verification mid-frame instead of waiting until frame end
- **Securing inputs for backup control**: If normal path has faulted, ASIL-D core needs independent sensor access for backup control. Can't trust inputs via the normal path anymore
- **Fault-type-dependent switching strategy**: Fail-safe can just "stop" for any anomaly, but fail-operational may need different backup control depending on what broke. Whether this detection mechanism can also provide diagnostic info on "what faulted" becomes an additional requirement

None of these change the core monitoring architecture. Future work.

### 4.2 Countermeasures per Fault Pattern

**Input data corruption**: Strategy depends on input type.

- Scalar values (control parameters, etc.): CRC immediately after input acquisition → send to ASIL-D core. Re-verify after computation
- Bulk data (images, etc.): CRC is overkill. Handle via normal path's output-side checks, ASIL-D core only does output plausibility

**Control flow corruption**: Send state variables to ASIL-D core every frame, verify legitimacy against transition table. Use E2E protection for transfer.

**Check mechanism failure**:

- Check constant table: Compute CRC at startup, hold in ASIL-D core. Recompute every frame and verify match
- Check result flag: Encoding (normal=0xA5A5, abnormal=0x5A5A) for 1-bit flip resilience
- Check execution proof: Signature method (XOR/CRC fold-in of unique constants at each checkpoint) → send to ASIL-D core. Detects skips and reordering

### 4.3 Residual Risk

Two-tier fault detection:

- Normal path I/O faults → detected by normal path (ASIL-B core) spec checks
- Spec check mechanism faults → detected by anomaly path (ASIL-D core)

Independent execution environments, so self-reference problem is structurally resolved.

However, the processes that **generate** the info sent to ASIL-D core (CRC, signature, etc.) run on the ASIL-B core. If these generation processes are corrupted by HW faults, that's the residual risk.

#### Scenario 1: Register Corruption During CRC Computation

```
ASIL-B Core:
  1. Check constant MAX_SAFE_VALUE corrupts 100→50000 (Fault A)
  2. Compute CRC of corrupted constant
     → Register corrupts during CRC computation,
       accidentally producing "the original correct CRC value" (Fault B)
  3. Send to ASIL-D core: {check result=normal, CRC=correct value}

ASIL-D Core:
  CRC matches startup value → misjudges as no anomaly
```

Fault A corrupts threshold so an actually-abnormal output passes the check + Fault B makes CRC look normal → missed detection

#### Scenario 2: Signature Corruption After Check Skip

```
ASIL-B Core:
  1. Execute algorithm
  2. Output check skipped due to control flow corruption (Fault A)
  3. Signature is an invalid value missing checkpoint fold-ins
  4. This invalid signature corrupts in memory before transmission,
     accidentally matching expected value (Fault B)

ASIL-D Core:
  Signature matches → misjudges as no anomaly
```

#### Why This Isn't a Problem

Both require **multiple independent faults occurring simultaneously within FTTI**.

**Scenario 1**: Fault A (constant bit flip) and Fault B (register corruption during CRC computation) are independent events. Furthermore, Fault B isn't just "CRC computation breaks" — it must **accidentally produce the correct CRC for the corrupted constant**. For CRC-32, probability ≈ 2⁻³² (2.3×10⁻¹⁰). Joint probability is the product of individual FIT rates, well below ASIL-D target (10⁻⁸/h = 10 FIT).

**Scenario 2**: Fault A (check skip) guarantees the signature differs from expected. To escape detection, Fault B (memory bit flip) must accidentally make it match. For 32-bit signature, ≈ 2⁻³².

→ **Infinite recursive protection is unnecessary**. Two-tier normal/anomaly architecture is quantitatively sufficient under ISO 26262's dual-point fault framework. Concrete numbers computable from HW SER and CRC/signature bit width.

### 4.4 Design-Phase Derivation Flow

Can be derived from the following SW architecture artifacts without waiting for normal path implementation:

```
SW Architecture Design
  │
  ├─ Input Interface Definition
  │    → Type & nature (scalar/bulk data, noise tolerance)
  │    → Protection strategy (CRC / E2E / output-side check only / none)
  │
  ├─ Output Spec Definition
  │    → Value range, constraints, physical consistency conditions
  │    → ASIL-D core independent plausibility check conditions
  │
  ├─ Spec Check Algorithm Definition
  │    → Constants & thresholds used → constant CRC verification
  │    → Result output format → encoding scheme
  │    → Execution flow → signature design
  │
  └─ State Transition Design
       → Allowed transition table → ASIL-D core transition verification logic
```

Normal path algorithm implementation details not needed anywhere.

---

## 5. Required Research

### 5.1 Theory 1: Automatic Derivation Rules from Spec Check Structure to Anomaly Monitor Requirements

Analyze normal path spec check structure, identify undetectable patterns against HW faults, mechanically derive ASIL-D core monitoring items. Want to build transformation rules.

Systematic mapping:

| Normal Path Design Artifact | Fault Pattern | Required Anomaly Processing |
|---|---|---|
| Range check (constant threshold) | Threshold constant bit flip | Constant CRC verification |
| State-transition-dependent processing | State variable bit flip | Transition table verification |
| Sequential check execution | Control flow corruption (check skip) | Signature execution proof |
| Bool check result output | Result flag bit flip | Encoded result value verification |
| Scalar-input-based computation | Input data bit flip | Input CRC verification |
| Bulk data input (images, etc.) | Input data bit flip | Output-side plausibility (absorbed by normal path) |

### 5.2 Theory 2: Design-Time Analytical Evaluation of SW Spec Check DC

Method to conservatively estimate DC from spec check structure and fault model before implementation.

- N-bit output with range check (tolerance = R% of full range) → 1-bit-flip SDC detection rate ≥ (1 - R/100)
- M check constants with CRC-32 → constant corruption detection rate (1 - 2⁻³²) ≈ 1 - 2.3×10⁻¹⁰
- Signature method (K checkpoints, W-bit signature) skip detection rate

Formalize to connect with ISO 26262 SPFM/LFM metrics.

#### Availability Perspective: False Positive Rate Evaluation

This architecture ("normal path does spec check + anomaly path does meta-check") has more fault judgment factors than directly doing spec checks on the ASIL-D core. Multiple paths where meta-check false-positives cause spurious shutdowns despite no actual fault:

- **Bit flip during constant CRC computation**: Register corrupts during CRC of a healthy constant → produces incorrect CRC → misjudged as constant corruption
- **Bit flip during signature computation**: Signature fold-in corrupts despite correct check execution → invalid signature → misjudged as check skip
- **Bit flip during transfer**: Check result / CRC / signature corrupts in transit to ASIL-D core
- **Encoded result flag corruption**: 0xA5A5 (normal) corrupts to a value that's neither 0xA5A5 nor 0x5A5A (abnormal) → misjudged as abnormal

Each mechanism's false positive rate can be computed independently:

- Constant CRC: Number of registers involved in CRC computation × bit width × SER × computation time
- Signature: Number of registers in fold-in processing × bit width × SER × computation time (increases with more checkpoints)
- Transfer: Transmitted data bit count × memory/bus SER × residence time
- Encoded flag: Flag bit count × SER × residence time

Overall false positive rate = 1 - (1-p₁)(1-p₂)(1-p₃)(1-p₄). Since each pᵢ is extremely small (≤10⁻¹⁰), ≈ p₁ + p₂ + p₃ + p₄ as approximation. Multiply by check frequency (every frame, several ms period, etc.) to get spurious shutdown rate per unit time.

Normally each mechanism's individual false positive rate is very low, but high check frequency can make the cumulative rate non-negligible.

The design-time tradeoff to evaluate depends on architecture choice:

- **This architecture (normal path spec check + anomaly path meta-check)**: DC (detection rate) vs false positive rate tradeoff. More meta-check items → higher DC but higher false positive rate
- **Architecture with spec check directly on ASIL-D core**: False positive problem disappears, but DC vs execution speed tradeoff emerges. ASIL-D core often has less processing power than normal path, and adding check items may not fit within FTTI

Quantitative pre-evaluation of these tradeoffs is also needed to choose between architectures.

### 5.3 Theory 3: Integrated Methodology

Integrate Theories 1 and 2 into a framework for end-to-end derivation of anomaly path processing specs from SW architecture design.

- **V-model left side (design)**: Derive anomaly path from input definitions, output specs, check algorithms, state transitions. Design-time DC estimate and false positive rate evaluation
- **V-model right side (verification)**: After normal path implementation, validate DC estimates and false positive rates via formal verification (Arcanum, etc.) and fault injection

---

## 6. Prior Work

### 6.1 Runtime Monitor Synthesis from Safety Requirements

**Heffernan et al. (IET Software, 2014)**: ISO 26262 safety requirements → past-time LTL formulas → automatic FPGA monitor circuit synthesis. Authors themselves say "this concept has not yet been sufficiently explored in the research literature." → Monitor synthesis framework is useful reference, but doesn't consider HW transient fault models for monitor derivation

**ESC-QC (IEEE, 2011)**: Event sequence charts + quantitative constraints → automatic Simulink/Stateflow monitor synthesis. → Spec-to-monitor automatic conversion pipeline is useful. But not fault-model-driven

**Runtime Monitoring for AVs (MDPI Electronics, 2025)**: Controlled natural language → LTL formal spec → runtime monitor. → Staged derivation process (safety requirements → formal spec → monitor) is useful reference

**BTC EmbeddedSpecifier**: Industrial tool. Safety requirements → semi-formal/formal notation → model checking monitor synthesis. TÜV SÜD certified ASIL D. Primarily Simulink model verification. → Formal spec-based automatic verification is useful. HW-fault-aware anomaly derivation is out of scope

### 6.2 Safety Patterns

**Safety Pattern Synthesis (AutoFOCUS3)**: Automatic recommendation of safety architecture patterns against faults in system architecture. Logic programming backend. → Recommendation mechanism is useful, but no rules for "spec check structure → monitor requirements"

**USF (Universal Safety Format)**: Describes safety mechanisms in a language-independent transformation language. Improves pattern reusability. → Description format is useful reference

**AUTOSAR WdgM**: Models valid program sequences at design phase, runtime checkpoint-based monitoring. → Directly corresponds to signature method. But WdgM's scope excludes output plausibility and data integrity

**Safety Patterns (Bitsch 2001, Konrad et al.)**: Catalog of 86 patterns in 19 classes. → Pattern systematization approach is useful. No mapping to HW fault models

### 6.3 Design-Phase DC Evaluation

**FMEDA**: ISO 26262 Part 5 HW quantitative safety analysis. DC = High(99%)/Medium(90%)/Low(60%). SPFM, LFM, PMHF computation. → HW DC evaluation framework is a reference model, but not directly applicable to SW spec check DC evaluation. Insufficient reference material for DC estimation methods has also been reported

**Safety-Oriented HW Exploration (MDPI, 2022)**: Fault tree vulnerability analysis + HW architecture exploration → automatic FMEDA report generation. → Design-phase FMEDA automation approach is useful reference

**Cone of Influence (Mentor/Siemens)**: RTL-level structural analysis to quantify safety mechanism effectiveness → estimated DC. → HW RTL design-time DC evaluation. Direction: extend to SW spec level

**Safety Synthesis (Mentor/Siemens)**: Automatic insertion of safety mechanisms into design structure. Register-level + module-level. → Direction: apply HW design auto-insertion concept to SW design

### 6.4 Other

**AVF (Mukherjee et al., MICRO 2003)**: Quantifies per-HW-structure probability of bit flip causing output error. ACE bit concept. → Fundamental concept for fault impact quantification. But bottom-up (post-implementation) analysis. AVF-style analysis useful for V-model right-side verification

**PVF (Sridharan & Kaeli, ISCA 2010)**: Extends AVF to SW side. Microarchitecture-independent instruction-level vulnerability quantification. → Useful as V-model right-side verification method

**AUTOSAR E2E Protection Library**: End-to-End protection for communication data (CRC, Counter, DataID, etc.). → Directly applicable for I/O data transfer protection

---

## 7. Research Gap Summary

| Required Theory | Existing State of the Art | Gap |
|---|---|---|
| **Spec check structure → automatic anomaly monitor derivation** | Safety requirements → monitor synthesis (Heffernan, ESC-QC), safety pattern catalogs (Bitsch, USF), WdgM | No transformation rules that take "spec check structure" + "HW fault model" → identify undetectable patterns → automatically derive anomaly monitor requirements |
| **Design-time analytical DC evaluation of SW spec checks** | FMEDA (HW), Cone of Influence (RTL), DC estimation tables (ISO 26262-5 Annex D) | No method to analytically estimate DC of SW range checks / plausibility checks from fault model before implementation |
| **Integrated methodology** | Individual building blocks exist (E2E, WdgM, Safety Pattern, FMEDA) | No end-to-end framework for "SW architecture design → fault pattern analysis → anomaly processing spec → DC estimate" |

### Novelty

1. **Fault-model-driven anomaly design**: Existing work has humans decide what to monitor from safety analysis. Here, formally analyze spec check structure + HW fault model to identify undetectable patterns and automatically derive anomaly path
2. **Complete at design phase**: AVF/PVF are post-implementation bottom-up. Can determine anomaly path at SW architecture design phase without waiting for normal path implementation
3. **Theorizing SW spec check DC evaluation**: No theory extending FMEDA DC evaluation to SW spec checks. Analytical models like range check SDC detection rate are novel contributions
