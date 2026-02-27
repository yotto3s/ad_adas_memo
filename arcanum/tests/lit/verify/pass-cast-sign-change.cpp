// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}sign_change{{.*}}obligations proven

#include <cstdint>

//@ requires: x >= 0 && x <= 100
//@ ensures: \result >= 0 && \result <= 100
uint32_t sign_change(int32_t x) { return static_cast<uint32_t>(x); }
