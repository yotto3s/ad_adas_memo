// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}sum_odd{{.*}}obligations proven

#include <cstdint>

//@ requires: n > 0 && n <= 100
//@ ensures: \result >= 0
int32_t sum_odd(int32_t n) {
    int32_t sum = 0;
    int32_t i = 0;
    //@ loop_invariant: i >= 0 && i <= n && sum >= 0
    //@ loop_variant: n - i
    //@ loop_assigns: i, sum
    while (i < n) {
        i = i + 1;
        if (i % 2 == 0) {
            continue;
        }
        sum = sum + i;
    }
    return sum;
}
