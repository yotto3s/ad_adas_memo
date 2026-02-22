// RUN: %arcanum --mode=verify %s 2>&1 | %FileCheck %s

// Test end-to-end verification of a simple addition function.
// All obligations should pass given the bounded inputs and output.

#include <cstdint>

//@ requires: a >= 0 && a <= 1000
//@ requires: b >= 0 && b <= 1000
//@ ensures: \result >= 0 && \result <= 2000
int32_t safe_add(int32_t a, int32_t b) {
    return a + b;
}

// CHECK: [PASS]
// CHECK: safe_add
// CHECK: obligations proven
