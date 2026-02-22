// RUN: %not %arcanum --mode=verify %s 2>&1 | %FileCheck %s
// CHECK: {{.*}}error: virtual functions are not allowed

class Base {
public:
  virtual int foo() { return 0; }
};
