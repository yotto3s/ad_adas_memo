// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}i8_add{{.*}}obligations proven

#include <cstdint>

//@ requires: a >= -100 && a <= 100
//@ requires: b >= -100 && b <= 100
//@ ensures: \result >= -128 && \result <= 127
int8_t i8_add(int8_t a, int8_t b) { return static_cast<int8_t>(a + b); }
