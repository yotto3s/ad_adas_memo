// REQUIRES: why3
// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: [FAIL]{{.*}}obligations

#include <cstdint>

//@ ensures: \result >= 0
int32_t unchecked_add(int32_t a, int32_t b) {
    return a + b;
}
