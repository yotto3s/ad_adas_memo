// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}i64_add{{.*}}obligations proven

#include <cstdint>

//@ requires: a >= 0 && a <= 1000000000
//@ requires: b >= 0 && b <= 1000000000
//@ ensures: \result >= 0 && \result <= 2000000000
int64_t i64_add(int64_t a, int64_t b) { return a + b; }
