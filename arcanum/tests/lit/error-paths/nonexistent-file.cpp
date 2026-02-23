// RUN: %not %arcanum --mode=verify /tmp/nonexistent_file_arcanum_test.cpp 2>&1 | %FileCheck %s
// CHECK: {{.*}}error: file not found: /tmp/nonexistent_file_arcanum_test.cpp
