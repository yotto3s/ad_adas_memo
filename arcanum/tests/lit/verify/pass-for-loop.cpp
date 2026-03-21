// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}sum_to_n{{.*}}obligations proven

#include <cstdint>

//@ requires: n >= 0 && n <= 1000
//@ ensures: \result >= 0
int32_t sum_to_n(int32_t n) {
    int32_t sum = 0;
    //@ loop_invariant: sum >= 0 && i >= 0 && i <= n
    //@ loop_variant: n - i
    //@ loop_assigns: i, sum
    for (int32_t i = 0; i < n; i = i + 1) {
        sum = sum + i;
    }
    return sum;
}
