// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error{{.*}}

// This is intentionally invalid C++ to test parse error handling.
int32_t foo(int32_t a {
  return a; }
