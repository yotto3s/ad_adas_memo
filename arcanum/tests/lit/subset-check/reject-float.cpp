// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error:{{.*}}floating-point{{.*}}

double foo() { return 3.14; }
