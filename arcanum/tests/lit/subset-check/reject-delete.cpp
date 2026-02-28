// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error:{{.*}}delete{{.*}}

void foo() { int* p = new int(42); delete p; }
