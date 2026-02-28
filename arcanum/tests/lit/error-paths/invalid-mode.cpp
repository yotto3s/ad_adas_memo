// RUN: %not %arcanum --mode=invalid %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error: unsupported mode 'invalid'{{.*}}

#include <cstdint>
int32_t foo(int32_t a) { return a; }
