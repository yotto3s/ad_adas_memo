// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error:{{.*}}try{{.*}}

void foo() {
  try { } catch (...) { }
}
