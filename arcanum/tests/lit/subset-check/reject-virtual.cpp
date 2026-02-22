// RUN: not %arcanum --mode=verify %s 2>&1 | %FileCheck %s

// Test that virtual functions are rejected by the SubsetEnforcer.
// Virtual dispatch is not in the safe C++ subset.

class Base {
    virtual int foo() { return 0; }
    // CHECK: error: virtual functions are not allowed in the Arcanum safe subset
};
