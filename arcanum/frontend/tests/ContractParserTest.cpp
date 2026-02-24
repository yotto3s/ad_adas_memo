#include "TestHelpers.h"

#include "arcanum/frontend/ContractParser.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

using ::arcanum::testing::parseFromSource;

// ---------------------------------------------------------------------------
// Parameterized: Comparison operators
// ---------------------------------------------------------------------------

struct ComparisonOpParam {
  const char* name;
  const char* opStr;
  BinaryOpKind expected;
};

class ComparisonOpTest : public ::testing::TestWithParam<ComparisonOpParam> {};

struct ComparisonOpName {
  std::string
  operator()(const ::testing::TestParamInfo<ComparisonOpParam>& info) const {
    return info.param.name;
  }
};

TEST_P(ComparisonOpTest, ParsesComparisonOperator) {
  auto [name, opStr, expected] = GetParam();
  std::string code = std::string(R"(
    #include <cstdint>
    //@ requires: a )") +
                     opStr + R"( 10
    int32_t foo(int32_t a) { return a; }
  )";
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(code, ast);

  auto it = contracts.begin();
  ASSERT_NE(it, contracts.end());
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, expected);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ParamRef);
  EXPECT_EQ(expr->left->paramName, "a");
  EXPECT_EQ(expr->right->kind, ContractExprKind::IntLiteral);
  EXPECT_EQ(expr->right->intValue, 10);
}

INSTANTIATE_TEST_SUITE_P(
    ContractParser, ComparisonOpTest,
    ::testing::Values(ComparisonOpParam{"Lt", "<", BinaryOpKind::Lt},
                      ComparisonOpParam{"Le", "<=", BinaryOpKind::Le},
                      ComparisonOpParam{"Gt", ">", BinaryOpKind::Gt},
                      ComparisonOpParam{"Ge", ">=", BinaryOpKind::Ge},
                      ComparisonOpParam{"Eq", "==", BinaryOpKind::Eq},
                      ComparisonOpParam{"Ne", "!=", BinaryOpKind::Ne}),
    ComparisonOpName{});

// ---------------------------------------------------------------------------
// Parameterized: Arithmetic operators
// ---------------------------------------------------------------------------

struct ArithmeticOpParam {
  const char* name;
  const char* opStr;
  BinaryOpKind expected;
};

class ArithmeticOpTest : public ::testing::TestWithParam<ArithmeticOpParam> {};

struct ArithmeticOpName {
  std::string
  operator()(const ::testing::TestParamInfo<ArithmeticOpParam>& info) const {
    return info.param.name;
  }
};

TEST_P(ArithmeticOpTest, ParsesArithmeticOperator) {
  auto [name, opStr, expected] = GetParam();
  std::string code = std::string(R"(
    #include <cstdint>
    //@ ensures: \result == a )") +
                     opStr + R"( b
    int32_t foo(int32_t a, int32_t b) { return a + b; }
  )";
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(code, ast);

  auto it = contracts.begin();
  ASSERT_NE(it, contracts.end());
  ASSERT_EQ(it->second.postconditions.size(), 1u);
  auto& expr = it->second.postconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Eq);
  ASSERT_NE(expr->right, nullptr);
  EXPECT_EQ(expr->right->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->right->binaryOp, expected);
}

INSTANTIATE_TEST_SUITE_P(
    ContractParser, ArithmeticOpTest,
    ::testing::Values(ArithmeticOpParam{"Add", "+", BinaryOpKind::Add},
                      ArithmeticOpParam{"Sub", "-", BinaryOpKind::Sub},
                      ArithmeticOpParam{"Mul", "*", BinaryOpKind::Mul},
                      ArithmeticOpParam{"Div", "/", BinaryOpKind::Div},
                      ArithmeticOpParam{"Rem", "%", BinaryOpKind::Rem}),
    ArithmeticOpName{});

// ---------------------------------------------------------------------------
// Parameterized: Unary operators
// ---------------------------------------------------------------------------

struct UnaryOpParam {
  const char* name;
  const char* contractLine;
  bool isPostcondition;
  UnaryOpKind expected;
};

class UnaryOpTest : public ::testing::TestWithParam<UnaryOpParam> {};

struct UnaryOpName {
  std::string
  operator()(const ::testing::TestParamInfo<UnaryOpParam>& info) const {
    return info.param.name;
  }
};

TEST_P(UnaryOpTest, ParsesUnaryOperator) {
  auto [name, contractLine, isPostcondition, expected] = GetParam();
  std::string code = std::string(R"(
    #include <cstdint>
    //@ )") + contractLine +
                     R"(
    int32_t foo(int32_t a) { return -a; }
  )";
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(code, ast);

  auto it = contracts.begin();
  ASSERT_NE(it, contracts.end());

  if (isPostcondition) {
    ASSERT_EQ(it->second.postconditions.size(), 1u);
    auto& expr = it->second.postconditions[0];
    ASSERT_NE(expr->right, nullptr);
    EXPECT_EQ(expr->right->kind, ContractExprKind::UnaryOp);
    EXPECT_EQ(expr->right->unaryOp, expected);
  } else {
    ASSERT_EQ(it->second.preconditions.size(), 1u);
    auto& expr = it->second.preconditions[0];
    EXPECT_EQ(expr->kind, ContractExprKind::UnaryOp);
    EXPECT_EQ(expr->unaryOp, expected);
    EXPECT_EQ(expr->operand->kind, ContractExprKind::BoolLiteral);
    EXPECT_EQ(expr->operand->boolValue, false);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ContractParser, UnaryOpTest,
    ::testing::Values(
        UnaryOpParam{"Not", "requires: !false", false, UnaryOpKind::Not},
        UnaryOpParam{"Neg", "ensures: \\result == -a", true, UnaryOpKind::Neg}),
    UnaryOpName{});

// ---------------------------------------------------------------------------
// Parameterized: Bool-prefix identifiers (regression: true/false prefix)
// ---------------------------------------------------------------------------

struct BoolPrefixParam {
  const char* paramName;
  const char* contractExpr;
  BinaryOpKind expectedOp;
};

class BoolPrefixIdentifierTest
    : public ::testing::TestWithParam<BoolPrefixParam> {};

struct BoolPrefixName {
  std::string
  operator()(const ::testing::TestParamInfo<BoolPrefixParam>& info) const {
    return info.param.paramName;
  }
};

TEST_P(BoolPrefixIdentifierTest, PrefixParsesAsIdentifier) {
  auto [paramName, contractExpr, expectedOp] = GetParam();
  std::string code = std::string(R"(
    #include <cstdint>
    //@ requires: )") +
                     contractExpr + "\n    int32_t foo(int32_t " + paramName +
                     ") { return " + paramName + "; }\n";
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(code, ast);

  ASSERT_EQ(contracts.size(), 1u)
      << "Contract was not parsed (likely dropped due to parse failure)";

  auto it = contracts.begin();
  ASSERT_EQ(it->second.preconditions.size(), 1u)
      << "Precondition was not parsed correctly";

  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, expectedOp);

  ASSERT_NE(expr->left, nullptr);
  EXPECT_EQ(expr->left->kind, ContractExprKind::ParamRef)
      << "LHS was parsed as kind=" << static_cast<int>(expr->left->kind)
      << " instead of ParamRef. Bool prefix was likely matched as a "
         "boolean literal.";
  EXPECT_EQ(expr->left->paramName, paramName);
}

INSTANTIATE_TEST_SUITE_P(
    ContractParser, BoolPrefixIdentifierTest,
    ::testing::Values(
        BoolPrefixParam{"trueVal", "trueVal >= 0", BinaryOpKind::Ge},
        BoolPrefixParam{"falsehood", "falsehood == 0", BinaryOpKind::Eq}),
    BoolPrefixName{});

// ---------------------------------------------------------------------------
// Individual tests: unique scenarios that don't benefit from parameterization
// ---------------------------------------------------------------------------

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
  EXPECT_EQ(expr->left->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->left->binaryOp, BinaryOpKind::Add);
}

TEST(ContractParserTest, ParsesOperatorPrecedence) {
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
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Eq);
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

TEST(ContractParserTest, MalformedExpressionReturnsNull) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: @#$invalid
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

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

  if (!contracts.empty()) {
    auto it = contracts.begin();
    EXPECT_TRUE(it->second.preconditions.empty());
  }
}

TEST(ContractParserTest, ChainedComparisonParsesLeftToRight) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: 0 <= a <= 1000
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_NE(it, contracts.end());
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Le);
}

// ---------------------------------------------------------------------------
// Coverage gap G2: Negative integer literals in contracts
// ---------------------------------------------------------------------------

TEST(ContractParserTest, ParsesNegativeIntegerLiteral) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= -5
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  auto it = contracts.begin();
  ASSERT_NE(it, contracts.end());
  ASSERT_EQ(it->second.preconditions.size(), 1u);
  auto& expr = it->second.preconditions[0];
  EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
  EXPECT_EQ(expr->binaryOp, BinaryOpKind::Ge);
  // -5 parses as UnaryOp(Neg, IntLiteral(5))
  ASSERT_NE(expr->right, nullptr);
  EXPECT_EQ(expr->right->kind, ContractExprKind::UnaryOp);
  EXPECT_EQ(expr->right->unaryOp, UnaryOpKind::Neg);
  ASSERT_NE(expr->right->operand, nullptr);
  EXPECT_EQ(expr->right->operand->kind, ContractExprKind::IntLiteral);
  EXPECT_EQ(expr->right->operand->intValue, 5);
}

// ---------------------------------------------------------------------------
// Coverage gap G3: Multiple functions with contracts in same TU
// ---------------------------------------------------------------------------

TEST(ContractParserTest, ParsesMultipleFunctionsWithContracts) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }

    //@ ensures: \result >= 0
    int32_t bar(int32_t b) { return b; }
  )",
                                   ast);

  EXPECT_EQ(contracts.size(), 2u);

  bool foundRequires = false;
  bool foundEnsures = false;
  for (const auto& [decl, info] : contracts) {
    if (!info.preconditions.empty()) {
      foundRequires = true;
      EXPECT_EQ(info.preconditions.size(), 1u);
    }
    if (!info.postconditions.empty()) {
      foundEnsures = true;
      EXPECT_EQ(info.postconditions.size(), 1u);
    }
  }
  EXPECT_TRUE(foundRequires);
  EXPECT_TRUE(foundEnsures);
}

} // namespace
} // namespace arcanum
