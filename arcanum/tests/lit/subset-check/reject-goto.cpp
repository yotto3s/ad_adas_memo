// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error:{{.*}}goto{{.*}}

void foo() {
label:
  goto label;
}
