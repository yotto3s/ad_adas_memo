/// Regression tests for SubsetEnforcer bugs found during Slice 2 review.
/// Originally these tests demonstrated the bugs; now they verify the fixes.

#include "arcanum/frontend/SubsetEnforcer.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>
#include <string>

namespace arcanum {
namespace {

/// Helper: parse source string into ASTContext and run enforceSubset.
SubsetResult checkSubsetBugRepro(const std::string& code) {
  std::unique_ptr<clang::ASTUnit> ast = clang::tooling::buildASTFromCode(
      code, "test.cpp", std::make_shared<clang::PCHContainerOperations>());
  EXPECT_NE(ast, nullptr);
  return enforceSubset(ast->getASTContext());
}

// [F5] Bug: VisitBinaryOperator false positive with integer literal operands
// and narrow integer types.
//
// When comparing int8_t with a literal 0, Clang's AST has an implicit
// promotion of int8_t to int.  IgnoreParenImpCasts() strips this promotion,
// exposing the original int8_t type (width 8) on the LHS, while the literal
// 0 retains type int (width 32).  The mixed-type check fires despite this
// being completely valid C++ code.
TEST(BugReproSubsetEnforcerTest, F5_Int8ComparedToLiteralZeroFalsePositive) {
  auto result = checkSubsetBugRepro(R"(
    #include <cstdint>
    bool is_positive(int8_t x) {
      return x > 0;
    }
  )");

  // This should PASS -- comparing int8_t to literal 0 is valid C++.
  // Bug F5: the mixed-type checker fires because IgnoreParenImpCasts
  // strips the integral promotion, exposing int8_t (width 8) vs int (width 32).
  EXPECT_TRUE(result.passed)
      << "int8_t compared to literal 0 should not trigger mixed-type error.\n"
      << "This is bug F5: IgnoreParenImpCasts causes false positive.\n"
      << "Diagnostics:\n";
  for (const auto& diag : result.diagnostics) {
    // Print diagnostics for debugging
    EXPECT_TRUE(diag.find("matching types") == std::string::npos)
        << "False positive diagnostic: " << diag;
  }
}

// [F5] Same bug with int16_t and arithmetic with literal
TEST(BugReproSubsetEnforcerTest, F5_Int16AddLiteralOneFalsePositive) {
  auto result = checkSubsetBugRepro(R"(
    #include <cstdint>
    int16_t increment(int16_t x) {
      return x + static_cast<int16_t>(1);
    }
  )");

  // With static_cast, both operands should be int16_t after
  // IgnoreParenImpCasts. This test serves as a control: it should pass.
  EXPECT_TRUE(result.passed)
      << "int16_t + static_cast<int16_t>(1) should pass.\n";
}

// [F5] The actual failing case: int16_t + 1 (without static_cast)
TEST(BugReproSubsetEnforcerTest,
     F5_Int16AddLiteralOneWithoutCastFalsePositive) {
  auto result = checkSubsetBugRepro(R"(
    #include <cstdint>
    int16_t increment(int16_t x) {
      return x + 1;
    }
  )");

  // This should PASS -- adding 1 (int literal) to int16_t is valid C++.
  // Bug F5: the mixed-type checker fires (int16_t width 16 vs int width 32).
  EXPECT_TRUE(result.passed)
      << "int16_t + literal 1 should not trigger mixed-type error.\n"
      << "This is bug F5: IgnoreParenImpCasts causes false positive.\n";
  for (const auto& diag : result.diagnostics) {
    EXPECT_TRUE(diag.find("matching types") == std::string::npos)
        << "False positive diagnostic: " << diag;
  }
}

// [F5] Same bug with uint8_t comparison
TEST(BugReproSubsetEnforcerTest, F5_Uint8ComparedToLiteralFalsePositive) {
  auto result = checkSubsetBugRepro(R"(
    #include <cstdint>
    bool is_big(uint8_t x) {
      return x > 100;
    }
  )");

  // This should PASS -- comparing uint8_t to literal 100 is valid C++.
  EXPECT_TRUE(result.passed)
      << "uint8_t compared to literal 100 should not trigger mixed-type "
         "error.\n"
      << "This is bug F5: IgnoreParenImpCasts causes false positive.\n";
}

} // namespace
} // namespace arcanum
