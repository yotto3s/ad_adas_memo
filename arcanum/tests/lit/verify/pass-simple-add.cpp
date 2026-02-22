// REQUIRES: why3
// RUN: %arcanum --mode=verify %s | %FileCheck %s
// CHECK: [PASS]{{.*}}safe_add{{.*}}obligations proven

#include <cstdint>

//@ requires: a >= 0 && a <= 1000
//@ requires: b >= 0 && b <= 1000
//@ ensures: \result >= 0 && \result <= 2000
int32_t safe_add(int32_t a, int32_t b) {
    return a + b;
}
