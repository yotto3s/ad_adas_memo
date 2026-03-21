// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}halve_to_zero{{.*}}obligations proven

#include <cstdint>

//@ requires: x > 0
//@ ensures: \result >= 0
int32_t halve_to_zero(int32_t x) {
    //@ loop_invariant: x >= 0
    //@ loop_variant: x
    //@ loop_assigns: x
    while (x > 0) {
        x = x / 2;
    }
    return x;
}
