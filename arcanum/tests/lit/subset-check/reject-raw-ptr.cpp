// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: error: raw pointer types are not allowed

void foo(int* p) {}
