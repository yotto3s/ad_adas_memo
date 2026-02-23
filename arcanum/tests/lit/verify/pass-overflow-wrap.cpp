// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}wrap_add{{.*}}obligations proven

#include <cstdint>

//@ overflow: wrap
//@ requires: a >= 0 && a <= 127
//@ requires: b >= 0 && b <= 127
//@ ensures: \result >= 0
int8_t wrap_add(int8_t a, int8_t b) { return static_cast<int8_t>(a + b); }
