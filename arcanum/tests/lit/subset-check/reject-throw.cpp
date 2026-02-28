// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error:{{.*}}throw{{.*}}

void foo() { throw 42; }
