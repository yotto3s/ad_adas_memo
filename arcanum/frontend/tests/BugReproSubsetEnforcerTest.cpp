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
// and narrow integer types (B4 parameterized).

struct F5Param {
  const char* name;
  const char* code;
  bool checkDiagnostics;
};

class F5FalsePositiveTest : public ::testing::TestWithParam<F5Param> {};

TEST_P(F5FalsePositiveTest, NarrowTypeWithLiteralPasses) {
  auto [name, code, checkDiagnostics] = GetParam();
  auto result = checkSubsetBugRepro(code);

  EXPECT_TRUE(result.passed)
      << name << " should not trigger mixed-type error.\n"
      << "This is bug F5: IgnoreParenImpCasts causes false positive.\n";

  if (checkDiagnostics) {
    for (const auto& diag : result.diagnostics) {
      EXPECT_TRUE(diag.find("matching types") == std::string::npos)
          << "False positive diagnostic: " << diag;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(BugReproSubsetEnforcer, F5FalsePositiveTest,
                         ::testing::Values(F5Param{"Int8ComparedToLiteralZero",
                                                   R"(
    #include <cstdint>
    bool is_positive(int8_t x) {
      return x > 0;
    }
  )",
                                                   true},
                                           F5Param{"Int16AddLiteralOne",
                                                   R"(
    #include <cstdint>
    int16_t increment(int16_t x) {
      return x + static_cast<int16_t>(1);
    }
  )",
                                                   false},
                                           F5Param{
                                               "Int16AddLiteralOneWithoutCast",
                                               R"(
    #include <cstdint>
    int16_t increment(int16_t x) {
      return x + 1;
    }
  )",
                                               true},
                                           F5Param{"Uint8ComparedToLiteral",
                                                   R"(
    #include <cstdint>
    bool is_big(uint8_t x) {
      return x > 100;
    }
  )",
                                                   false}),
                         [](const ::testing::TestParamInfo<F5Param>& info) {
                           return info.param.name;
                         });

} // namespace
} // namespace arcanum
