// RUN: not %arcanum --mode=verify %s 2>&1 | %FileCheck %s

// Test that raw pointers are rejected by the SubsetEnforcer.
// Raw pointers are not in the safe C++ subset.

int32_t deref(int32_t* p) {
    // CHECK: error: raw pointers are not allowed in the Arcanum safe subset
    return *p;
}
