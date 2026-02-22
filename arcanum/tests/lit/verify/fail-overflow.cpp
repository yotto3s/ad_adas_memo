// RUN: not %arcanum --mode=verify %s 2>&1 | %FileCheck %s

// Test that unbounded addition is flagged for potential overflow.
// Without preconditions bounding the inputs, the addition of two
// arbitrary int32_t values can overflow.

#include <cstdint>

//@ ensures: \result == a + b
int32_t unsafe_add(int32_t a, int32_t b) {
    return a + b;
}

// CHECK: [FAIL]
// CHECK: unsafe_add
// CHECK: overflow
