// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error:{{.*}}matching types{{.*}}

#include <cstdint>

int32_t mixed_add(int32_t a, int8_t b) { return a + b; }
