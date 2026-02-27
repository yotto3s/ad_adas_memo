// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}i8_add{{.*}}obligations proven

#include <cstdint>

// Preconditions use tighter bounds so the sum stays within i8 range
// [-128, 127].  Arcanum verifies at source-level types (i8), not
// C++ promoted types (int), so the add overflow check uses i8 bounds.
//@ requires: a >= -64 && a <= 63
//@ requires: b >= -64 && b <= 63
//@ ensures: \result >= -128 && \result <= 127
int8_t i8_add(int8_t a, int8_t b) { return static_cast<int8_t>(a + b); }
