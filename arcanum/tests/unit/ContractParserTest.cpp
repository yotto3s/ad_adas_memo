#include "frontend/ContractParser.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

std::map<const clang::FunctionDecl*, ContractInfo>
parseFromSource(const std::string& code,
                std::unique_ptr<clang::ASTUnit>& astOut) {
  astOut = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp",
      "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  EXPECT_NE(astOut, nullptr);
  return parseContracts(astOut->getASTContext());
}

TEST(ContractParserTest, ParsesSimpleRequires) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )", ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.preconditions.size(), 1u);
  EXPECT_EQ(it->second.postconditions.size(), 0u);
}

TEST(ContractParserTest, ParsesMultipleRequires) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    //@ requires: a <= 1000
    int32_t foo(int32_t a) { return a; }
  )", ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.preconditions.size(), 2u);
}

TEST(ContractParserTest, ParsesEnsuresWithResult) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t foo(int32_t a) { return a; }
  )", ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.postconditions.size(), 1u);
}

TEST(ContractParserTest, ParsesRequiresAndEnsures) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) { return a + b; }
  )", ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.preconditions.size(), 2u);
  EXPECT_EQ(it->second.postconditions.size(), 1u);
}

TEST(ContractParserTest, NoContractsReturnsEmptyMap) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    int32_t foo(int32_t a) { return a; }
  )", ast);

  EXPECT_TRUE(contracts.empty());
}

TEST(ContractParserTest, ParsesBinaryComparisonExpr) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )", ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Ge);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ParamRef);
  EXPECT_EQ(expr->left->paramName, "a");
  EXPECT_EQ(expr->right->kind, ContractExprKind::IntLiteral);
  EXPECT_EQ(expr->right->intValue, 0);
}

TEST(ContractParserTest, ParsesAndExpression) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    int32_t foo(int32_t a) { return a; }
  )", ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::And);
}

TEST(ContractParserTest, ParsesResultRef) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t foo(int32_t a) { return a; }
  )", ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.postconditions.size(), 1u);
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ResultRef);
}

} // namespace
} // namespace arcanum
