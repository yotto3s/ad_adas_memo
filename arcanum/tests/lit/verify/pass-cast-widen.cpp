// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}widen_i8_to_i32{{.*}}obligations proven

#include <cstdint>

//@ requires: x >= -128 && x <= 127
//@ ensures: \result >= -128 && \result <= 127
int32_t widen_i8_to_i32(int8_t x) { return static_cast<int32_t>(x); }
