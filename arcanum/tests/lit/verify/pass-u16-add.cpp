// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}u16_add{{.*}}obligations proven

#include <cstdint>

//@ requires: a <= 30000
//@ requires: b <= 30000
//@ ensures: \result <= 65535
uint16_t u16_add(uint16_t a, uint16_t b) {
  return static_cast<uint16_t>(a + b);
}
