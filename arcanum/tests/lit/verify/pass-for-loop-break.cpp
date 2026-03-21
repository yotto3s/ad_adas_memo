// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}find_first_even{{.*}}obligations proven

#include <cstdint>

//@ requires: n > 0 && n <= 100
//@ ensures: \result >= 0
int32_t find_first_even(int32_t n) {
    int32_t result = 0;
    //@ loop_invariant: i >= 0 && i <= n && result >= 0
    //@ loop_variant: n - i
    //@ loop_assigns: i, result
    for (int32_t i = 0; i < n; i = i + 1) {
        if (i % 2 == 0) {
            result = i;
            break;
        }
    }
    return result;
}
