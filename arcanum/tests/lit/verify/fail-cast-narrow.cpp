// REQUIRES: why3
// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: [FAIL]{{.*}}obligations

#include <cstdint>

//@ requires: x >= 0
int8_t narrow_i32_to_i8(int32_t x) { return static_cast<int8_t>(x); }
