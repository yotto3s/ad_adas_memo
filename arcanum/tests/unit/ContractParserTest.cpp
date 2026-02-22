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

TEST(ContractParserTest, ParsesSimpleRequires) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

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
  )",
                                   ast);

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
  )",
                                   ast);

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
  )",
                                   ast);

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
  )",
                                   ast);

  EXPECT_TRUE(contracts.empty());
}

TEST(ContractParserTest, ParsesBinaryComparisonExpr) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

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
  )",
                                   ast);

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
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.postconditions.size(), 1u);
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ResultRef);
}

// --- TC-8: Comprehensive contract expression parser tests ---

TEST(ContractParserTest, ParsesAllComparisonOperators) {
  // Test <, <=, >, >=, ==, !=
  std::unique_ptr<clang::ASTUnit> ast;

  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a < 10
    //@ requires: a <= 20
    //@ requires: a > 0
    //@ requires: a >= 1
    //@ requires: a == 5
    //@ requires: a != 3
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 6u);
  EXPECT_EQ(it->second.preconditions[0]->binaryOp, BinaryOpKind::Lt);
  EXPECT_EQ(it->second.preconditions[1]->binaryOp, BinaryOpKind::Le);
  EXPECT_EQ(it->second.preconditions[2]->binaryOp, BinaryOpKind::Gt);
  EXPECT_EQ(it->second.preconditions[3]->binaryOp, BinaryOpKind::Ge);
  EXPECT_EQ(it->second.preconditions[4]->binaryOp, BinaryOpKind::Eq);
  EXPECT_EQ(it->second.preconditions[5]->binaryOp, BinaryOpKind::Ne);
}

TEST(ContractParserTest, ParsesArithmeticOperators) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result == a + b
    int32_t foo(int32_t a, int32_t b) { return a + b; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.postconditions.size(), 1u);
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Eq);
  // rhs should be a + b
  EXPECT_EQ(expr->right->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->right->binaryOp, BinaryOpKind::Add);
}

TEST(ContractParserTest, ParsesSubtraction) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result == a - b
    int32_t foo(int32_t a, int32_t b) { return a - b; }
  )",
                                   ast);

  auto it = contracts.begin();
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->right->binaryOp, BinaryOpKind::Sub);
}

TEST(ContractParserTest, ParsesMultiplication) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result == a * b
    int32_t foo(int32_t a, int32_t b) { return a * b; }
  )",
                                   ast);

  auto it = contracts.begin();
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->right->binaryOp, BinaryOpKind::Mul);
}

TEST(ContractParserTest, ParsesDivision) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result == a / b
    int32_t foo(int32_t a, int32_t b) { return a / b; }
  )",
                                   ast);

  auto it = contracts.begin();
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->right->binaryOp, BinaryOpKind::Div);
}

TEST(ContractParserTest, ParsesRemainder) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result == a % b
    int32_t foo(int32_t a, int32_t b) { return a % b; }
  )",
                                   ast);

  auto it = contracts.begin();
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->right->binaryOp, BinaryOpKind::Rem);
}

TEST(ContractParserTest, ParsesOrExpression) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a > 0 || b > 0
    int32_t foo(int32_t a, int32_t b) { return a + b; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  EXPECT_EQ(it->second.preconditions[0]->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(it->second.preconditions[0]->binaryOp, BinaryOpKind::Or);
}

TEST(ContractParserTest, ParsesNotUnaryOp) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: !false
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::UnaryOp);
  EXPECT_EQ(expr->unaryOp, UnaryOpKind::Not);
  EXPECT_EQ(expr->operand->kind, ContractExprKind::BoolLiteral);
  EXPECT_EQ(expr->operand->boolValue, false);
}

TEST(ContractParserTest, ParsesNegationUnaryOp) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result == -a
    int32_t foo(int32_t a) { return -a; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.postconditions.size(), 1u);
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->right->kind, ContractExprKind::UnaryOp);
  EXPECT_EQ(expr->right->unaryOp, UnaryOpKind::Neg);
}

TEST(ContractParserTest, ParsesBoolLiterals) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: true
    //@ ensures: false
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  EXPECT_EQ(it->second.preconditions[0]->kind, ContractExprKind::BoolLiteral);
  EXPECT_EQ(it->second.preconditions[0]->boolValue, true);

  ASSERT_EQ(it->second.postconditions.size(), 1u);
  EXPECT_EQ(it->second.postconditions[0]->kind, ContractExprKind::BoolLiteral);
  EXPECT_EQ(it->second.postconditions[0]->boolValue, false);
}

TEST(ContractParserTest, ParsesParenthesizedExpression) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: (a + b) >= 0
    int32_t foo(int32_t a, int32_t b) { return a + b; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Ge);
  // Left side should be (a + b)
  EXPECT_EQ(expr->left->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->left->binaryOp, BinaryOpKind::Add);
}

TEST(ContractParserTest, ParsesOperatorPrecedence) {
  // a + b * c should parse as a + (b * c) due to precedence
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ ensures: \result == a + b * c
    int32_t foo(int32_t a, int32_t b, int32_t c) { return a + b * c; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_EQ(it->second.postconditions.size(), 1u);
  auto& expr = it->second.postconditions[0];
  // == at top level
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Eq);
  // right side: a + (b * c)
  auto& addExpr = expr->right;
  EXPECT_EQ(addExpr->binaryOp, BinaryOpKind::Add);
  EXPECT_EQ(addExpr->left->kind, ContractExprKind::ParamRef);
  EXPECT_EQ(addExpr->left->paramName, "a");
  EXPECT_EQ(addExpr->right->binaryOp, BinaryOpKind::Mul);
}

TEST(ContractParserTest, ParsesIntegerLiteral) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 42
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  auto it = contracts.begin();
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->right->kind, ContractExprKind::IntLiteral);
  EXPECT_EQ(expr->right->intValue, 42);
}

// TC-9: Malformed expression tests
TEST(ContractParserTest, MalformedExpressionReturnsNull) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: @#$invalid
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  // Malformed expression should result in no preconditions being added
  // (because parse returns nullptr which is filtered by the `if (auto expr)`
  // check)
  auto it = contracts.begin();
  if (it != contracts.end()) {
    EXPECT_TRUE(it->second.preconditions.empty());
  }
}

TEST(ContractParserTest, EmptyExpressionReturnsNull) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires:
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  // Empty expression should not add any preconditions
  if (!contracts.empty()) {
    auto it = contracts.begin();
    EXPECT_TRUE(it->second.preconditions.empty());
  }
}

} // namespace
} // namespace arcanum
