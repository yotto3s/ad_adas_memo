# ISO 26262 Software Qualification: What "Certified" Actually Means

## 1. The Common Misconception

"We can't use that compiler because it's not certified." This is one of the most frequently heard statements in automotive software development. It is technically incorrect, but practically meaningful. Understanding why requires unpacking what ISO 26262 actually requires — and what it does not.

## 2. ISO 26262 Does Not Certify Software Products

ISO 26262 defines two distinct mechanisms:

- **Process certification**: A third-party assessment body (TÜV SÜD, TÜV NORD, TÜV Rheinland, etc.) audits an organization's development process and confirms it conforms to ISO 26262. This is what OEMs and suppliers obtain when they claim "ISO 26262 certified."

- **Tool qualification** (Part 8, Clause 11): The **user** of a software tool provides evidence that the tool is suitable for developing safety-related systems at a given ASIL level.

There is no mechanism in ISO 26262 to certify a software product itself — not compilers, not operating systems, not AUTOSAR stacks. The standard certifies **processes** and qualifies **tools**. A tool vendor can provide supporting evidence, but the qualification responsibility always lies with the user.

## 3. Tool Qualification Process

### 3.1 Classification

Tools are classified based on two dimensions:

| Dimension | Question |
|---|---|
| **Tool Impact (TI)** | Can the tool introduce or fail to detect errors in the safety-related software? |
| **Tool Error Detection (TD)** | How likely is it that such errors would be caught by other measures (reviews, tests, other tools)? |

These two dimensions map to a **Tool Confidence Level (TCL)**:

```
                    Tool Error Detection
                    High (TD1)    Low (TD2)
Tool Impact
  Low  (TI1)       TCL 1         TCL 1
  High (TI2)       TCL 2         TCL 3
```

- **TCL 1**: No additional qualification needed. The tool is either low-impact or its errors are reliably caught by other means.
- **TCL 2**: Qualification required. Moderate effort.
- **TCL 3**: Qualification required. Maximum effort.

A compiler has **high Tool Impact** (TI2): a compiler bug can silently generate incorrect machine code from correct source code. Whether it classifies as TCL 2 or TCL 3 depends on the project's ability to detect compiler-introduced errors through other means (e.g., integration testing, back-to-back testing against a reference model).

### 3.2 Qualification Methods

ISO 26262 Part 8 defines four qualification methods, in order of increasing rigor:

| Method | Description | Suitable For |
|---|---|---|
| **1a** | Increased confidence from use | TCL 2 at lower ASILs. Requires evidence of successful use in similar projects. |
| **1b** | Evaluation of the development process | TCL 2. Assess the tool vendor's development process against a recognized standard. |
| **1c** | Validation of the software tool | TCL 2 and TCL 3. Test the tool against a validation suite. |
| **1d** | Development in accordance with a safety standard | TCL 3 at ASIL D. The tool itself was developed following a safety standard (e.g., IEC 61508). |

For compilers at ASIL C/D, methods **1c** or **1d** are typically required.

## 4. What "Certified Compiler" Actually Means

### 4.1 Compilers With Vendor Qualification Support

When the industry says a compiler is "certified," they mean: **the vendor provides a Compiler Qualification Kit (CQK)**.

A CQK typically includes:

- A comprehensive test suite covering the C/C++ language features used in the project
- Pre-generated qualification evidence documents
- Traceability from test cases to language standard clauses
- Reports formatted for ISO 26262 Part 8 compliance

Vendors offering CQKs include Green Hills, Wind River, HighTec, IAR Systems, and others. The user runs the kit against their specific compiler version, target architecture, and optimization flags, reviews the results, and archives the evidence. This costs license fees but minimal engineering effort.

### 4.2 Compilers Without Vendor Qualification Support

When someone says a compiler is "not certified," they mean: **no vendor provides the qualification kit, so the user must build their own.**

GCC and Clang fall into this category. They can be used under ISO 26262 — the standard does not prohibit any specific tool. But the user must:

1. Define the scope (compiler version, target, flags, language subset used)
2. Build or procure a validation test suite covering that scope
3. Execute the test suite and document results
4. Produce the qualification report
5. Have the evidence reviewed (internally or by an assessor)

This is months of engineering work per compiler version. Every compiler update requires re-qualification. This is the real barrier — not a technical prohibition, but a cost/effort burden.

Third-party services (Solid Sands, Japan Novel Corporation, etc.) offer compiler validation suites and qualification support for GCC/Clang to reduce this burden.

### 4.3 CompCert: The Formally Verified Alternative

CompCert takes a fundamentally different approach. Instead of validating the compiler through testing (method 1c), CompCert provides a **mathematical proof** that the compiler preserves the semantics of the source program — the generated machine code behaves exactly as the C source specifies.

This proof is mechanically checked by the Coq proof assistant. It covers the entire compilation pipeline from C source to assembly, including all optimizations. A compiler bug that changes program behavior would be a contradiction of the proof — it cannot exist (within the scope covered by the proof).

From a qualification perspective, this is strictly stronger than any test-suite-based qualification. A test suite can only cover the cases it tests; a formal proof covers all cases. CompCert has been qualified for ISO 26262 and DO-178C (avionics) projects.

The limitation: CompCert supports fewer targets and optimization levels than GCC/Clang, and carries a commercial license cost.

## 5. Static Analysis Tools

Static analysis tools (Astrée, Polyspace, TrustInSoft Analyzer, etc.) also require tool qualification. Their classification depends on usage:

- **Used to verify absence of defects** (replacement for testing): High TI, requires TCL 2 or TCL 3 qualification.
- **Used as supplementary analysis** (in addition to testing): Lower TI, may classify as TCL 1.

Vendors provide **Qualification Support Kits (QSKs)** that automate much of the qualification process. For example, AbsInt provides a QSK for Astrée that generates ISO 26262 Part 8-compliant qualification reports. TrustInSoft Analyzer has been qualified as a TCL 3 tool by TÜV SÜD.

## 6. The Process vs Property Gap

The entire tool qualification framework illustrates the same gap identified in the e2e proven architecture introduction (Section 1.2):

- **Hardware**: Quantitative metrics (PMHF, FIT) define clear pass/fail criteria.
- **Software tools**: Process-based qualification (did you test the tool adequately?) rather than property-based proof (is the tool mathematically correct?).

CompCert is the notable exception — it provides a property-based argument (proven semantic preservation) rather than a process-based argument (test suite passed). The three-layer verification approach extends this principle to the application software itself: instead of arguing "we tested enough," argue "we proved correctness."

## 7. Practical Implications

| Situation | What to Do |
|---|---|
| Using a vendor-supported compiler (Green Hills, etc.) | Use the vendor's CQK. Budget license cost. |
| Using GCC/Clang | Budget engineering effort for in-house qualification or procure third-party validation suites. |
| Highest assurance (ASIL D, critical path) | Consider CompCert for the formally verified guarantee. |
| Static analysis tools | Use vendor QSKs. Classify usage correctly (replacement vs. supplementary) to determine TCL. |
| Any tool update | Re-qualification is required for the new version. Plan for this in the release cycle. |
