/// Bug reproduction tests for ContractParser.cpp findings.
/// These tests demonstrate bugs found by the bug hunter agent.
/// Each test targets exactly one finding and should FAIL before the bug is fixed.

#include "arcanum/frontend/ContractParser.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

std::map<const clang::FunctionDecl*, ContractInfo>
parseFromSource(const std::string& code,
                std::unique_ptr<clang::ASTUnit>& astOut) {
  astOut = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  EXPECT_NE(astOut, nullptr);
  return parseContracts(astOut->getASTContext());
}

/// [F3] matchString("true") matches identifier prefixes without word
/// boundary check.
///
/// A parameter named "trueVal" starts with "true".  The parser's
/// matchString("true") greedily matches this prefix and returns
/// BoolLiteral(true), leaving "Val" unconsumed.  The remaining "Val"
/// causes the parse to fail or produce an incorrect AST.
///
/// Expected: "trueVal >= 0" should parse as (ParamRef("trueVal") >= 0).
/// Before fix: "trueVal" is parsed as BoolLiteral(true) + garbage "Val",
/// resulting in a null/incorrect parse and the contract being silently
/// dropped.
TEST(BugHunterContractParserTest, F3_TruePrefixMatchesIdentifier) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: trueVal >= 0
    int32_t foo(int32_t trueVal) { return trueVal; }
  )",
                                   ast);

  // The contract "trueVal >= 0" should be successfully parsed.
  ASSERT_EQ(contracts.size(), 1u)
      << "Contract was not parsed (likely dropped due to parse failure)";

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u)
      << "Precondition was not parsed correctly";

  // The parsed expression should be a comparison: trueVal >= 0
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Ge);

  // The LHS should be a ParamRef to "trueVal", NOT a BoolLiteral
  ASSERT_NE(expr->left, nullptr);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ParamRef)
      << "LHS was parsed as kind="
      << static_cast<int>(expr->left->kind)
      << " instead of ParamRef.  'true' prefix was likely matched as a "
         "boolean literal.";
  EXPECT_EQ(expr->left->paramName, "trueVal");
}

/// [F3] Same bug with "false" prefix.
TEST(BugHunterContractParserTest, F3_FalsePrefixMatchesIdentifier) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: falsehood == 0
    int32_t foo(int32_t falsehood) { return falsehood; }
  )",
                                   ast);

  ASSERT_EQ(contracts.size(), 1u)
      << "Contract was not parsed (likely dropped due to parse failure)";

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u)
      << "Precondition was not parsed correctly";

  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Eq);

  ASSERT_NE(expr->left, nullptr);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ParamRef)
      << "LHS was parsed as kind="
      << static_cast<int>(expr->left->kind)
      << " instead of ParamRef.  'false' prefix was likely matched as a "
         "boolean literal.";
  EXPECT_EQ(expr->left->paramName, "falsehood");
}

} // namespace
} // namespace arcanum
