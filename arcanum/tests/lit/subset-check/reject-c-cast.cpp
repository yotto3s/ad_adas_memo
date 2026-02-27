// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error:{{.*}}static_cast{{.*}}

#include <cstdint>

int8_t bad_cast(int32_t x) { return (int8_t)x; }
