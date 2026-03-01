// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}count_digits{{.*}}obligations proven

#include <cstdint>

//@ requires: x > 0 && x <= 1000
//@ ensures: \result >= 1
int32_t count_digits(int32_t x) {
    int32_t count = 0;
    //@ loop_invariant: count >= 0 && x >= 0
    //@ loop_variant: x
    //@ loop_assigns: x, count
    do {
        x = x / 10;
        count = count + 1;
    } while (x > 0);
    return count;
}
