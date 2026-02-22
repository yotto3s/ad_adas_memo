#include "frontend/SubsetEnforcer.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>
#include <string>

namespace arcanum {
namespace {

/// Helper: parse source string into ASTContext and run enforceSubset.
SubsetResult checkSubset(const std::string& code) {
  // Use Clang tooling to parse the code
  std::unique_ptr<clang::ASTUnit> ast = clang::tooling::buildASTFromCode(
      code, "test.cpp", std::make_shared<clang::PCHContainerOperations>());
  EXPECT_NE(ast, nullptr);
  return enforceSubset(ast->getASTContext());
}

TEST(SubsetEnforcerTest, AcceptsInt32ArithmeticFunction) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t add(int32_t a, int32_t b) { return a + b; }
  )");
  EXPECT_TRUE(result.passed);
  EXPECT_TRUE(result.diagnostics.empty());
}

TEST(SubsetEnforcerTest, AcceptsBoolFunction) {
  auto result = checkSubset(R"(
    #include <cstdint>
    bool isPositive(int32_t x) { return x > 0; }
  )");
  EXPECT_TRUE(result.passed);
}

TEST(SubsetEnforcerTest, AcceptsIfElse) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t abs(int32_t x) {
      if (x < 0) {
        return -x;
      } else {
        return x;
      }
    }
  )");
  EXPECT_TRUE(result.passed);
}

TEST(SubsetEnforcerTest, RejectsVirtualFunction) {
  auto result = checkSubset(R"(
    class Base {
    public:
      virtual int foo() { return 0; }
    };
  )");
  EXPECT_FALSE(result.passed);
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_NE(result.diagnostics[0].find("virtual"), std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsRawPointer) {
  auto result = checkSubset(R"(
    void foo(int* p) {}
  )");
  EXPECT_FALSE(result.passed);
  ASSERT_FALSE(result.diagnostics.empty());
  EXPECT_NE(result.diagnostics[0].find("pointer"), std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsNewExpression) {
  auto result = checkSubset(R"(
    void foo() { int* p = new int(42); }
  )");
  EXPECT_FALSE(result.passed);
}

TEST(SubsetEnforcerTest, RejectsThrow) {
  auto result = checkSubset(R"(
    void foo() { throw 42; }
  )");
  EXPECT_FALSE(result.passed);
}

TEST(SubsetEnforcerTest, RejectsDoubleType) {
  auto result = checkSubset(R"(
    double foo() { return 3.14; }
  )");
  EXPECT_FALSE(result.passed);
}

TEST(SubsetEnforcerTest, AcceptsAllArithmeticOps) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t compute(int32_t a, int32_t b) {
      int32_t sum = a + b;
      int32_t diff = a - b;
      int32_t prod = a * b;
      int32_t quot = a / b;
      int32_t rem = a % b;
      return sum;
    }
  )");
  EXPECT_TRUE(result.passed);
}

TEST(SubsetEnforcerTest, AcceptsAllComparisonAndLogicalOps) {
  auto result = checkSubset(R"(
    #include <cstdint>
    bool check(int32_t a, int32_t b) {
      bool r1 = a < b;
      bool r2 = a <= b;
      bool r3 = a > b;
      bool r4 = a >= b;
      bool r5 = a == b;
      bool r6 = a != b;
      bool r7 = r1 && r2;
      bool r8 = r3 || r4;
      bool r9 = !r5;
      return r9;
    }
  )");
  EXPECT_TRUE(result.passed);
}

} // namespace
} // namespace arcanum
