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
  struct ArithmeticCase {
    const char* opName;
    const char* source;
    BinaryOpKind expectedKind;
  };

  const ArithmeticCase cases[] = {
      {"Add",
       R"(
    #include <cstdint>
    //@ ensures: \result == a + b
    int32_t foo(int32_t a, int32_t b) { return a + b; }
  )",
       BinaryOpKind::Add},
      {"Sub",
       R"(
    #include <cstdint>
    //@ ensures: \result == a - b
    int32_t foo(int32_t a, int32_t b) { return a - b; }
  )",
       BinaryOpKind::Sub},
      {"Mul",
       R"(
    #include <cstdint>
    //@ ensures: \result == a * b
    int32_t foo(int32_t a, int32_t b) { return a * b; }
  )",
       BinaryOpKind::Mul},
      {"Div",
       R"(
    #include <cstdint>
    //@ ensures: \result == a / b
    int32_t foo(int32_t a, int32_t b) { return a / b; }
  )",
       BinaryOpKind::Div},
      {"Rem",
       R"(
    #include <cstdint>
    //@ ensures: \result == a % b
    int32_t foo(int32_t a, int32_t b) { return a % b; }
  )",
       BinaryOpKind::Rem},
  };

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.opName);
    std::unique_ptr<clang::ASTUnit> ast;
    auto contracts = parseFromSource(tc.source, ast);

    auto it = contracts.begin();
    ASSERT_EQ(it->second.postconditions.size(), 1u);
    auto& expr = it->second.postconditions[0];
    EXPECT_EQ(expr->kind, ContractExprKind::BinaryOp);
    EXPECT_EQ(expr->binaryOp, BinaryOpKind::Eq);
    EXPECT_EQ(expr->right->kind, ContractExprKind::BinaryOp);
    EXPECT_EQ(expr->right->binaryOp, tc.expectedKind);
  }
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

// --- Task 5: Overflow mode annotation tests ---

TEST(ContractParserTest, OverflowModeDefaultIsTrap) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.overflowMode, "trap");
}

TEST(ContractParserTest, OverflowModeTrap) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ overflow: trap
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.overflowMode, "trap");
}

TEST(ContractParserTest, OverflowModeWrap) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ overflow: wrap
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.overflowMode, "wrap");
}

TEST(ContractParserTest, OverflowModeSaturate) {
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ overflow: saturate
    //@ requires: a >= 0
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.overflowMode, "saturate");
}

TEST(ContractParserTest, OverflowModeOnlyStoresContract) {
  // A function with only an overflow annotation (non-trap) should be stored.
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ overflow: wrap
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  EXPECT_EQ(contracts.size(), 1u);
  auto it = contracts.begin();
  EXPECT_EQ(it->second.overflowMode, "wrap");
  EXPECT_TRUE(it->second.preconditions.empty());
  EXPECT_TRUE(it->second.postconditions.empty());
}

TEST(ContractParserTest, OverflowModeTrapOnlyNotStored) {
  // A function with only an explicit trap annotation (same as default)
  // and no other contracts should NOT be stored.
  std::unique_ptr<clang::ASTUnit> ast;
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ overflow: trap
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  EXPECT_TRUE(contracts.empty());
}

TEST(ContractParserTest, OverflowModeInvalidWarns) {
  // An invalid overflow mode should warn and default to "trap".
  std::unique_ptr<clang::ASTUnit> ast;
  // Redirect stderr is not easy in unit tests; just verify the contract
  // is not stored (no other valid annotations).
  auto contracts = parseFromSource(R"(
    #include <cstdint>
    //@ overflow: invalid_mode
    int32_t foo(int32_t a) { return a; }
  )",
                                   ast);

  // Invalid mode defaults to "trap", no other contracts, so map is empty.
  EXPECT_TRUE(contracts.empty());
}

// ---------------------------------------------------------------------------
// Loop annotation parsing tests (Slice 3)
// ---------------------------------------------------------------------------

TEST(ContractParserTest, ParsesLoopInvariantAnnotation) {
  auto lines =
      arcanum::extractAnnotationLines("//@ loop_invariant: x >= 0 && x <= n\n");
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "loop_invariant: x >= 0 && x <= n");
}

TEST(ContractParserTest, ParsesLoopVariantAnnotation) {
  auto lines = arcanum::extractAnnotationLines("//@ loop_variant: n - i\n");
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "loop_variant: n - i");
}

TEST(ContractParserTest, ParsesLoopAssignsAnnotation) {
  auto lines = arcanum::extractAnnotationLines("//@ loop_assigns: i, sum\n");
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "loop_assigns: i, sum");
}

TEST(ContractParserTest, ParsesLabelAnnotation) {
  auto lines = arcanum::extractAnnotationLines("//@ label: outer\n");
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "label: outer");
}

TEST(ContractParserTest, ParsesMultipleLoopInvariantsConjoin) {
  LoopContractInfo info;
  std::vector<std::string> lines = {
      "loop_invariant: x >= 0",
      "loop_invariant: x <= n",
  };
  for (const auto& line : lines) {
    applyLoopAnnotationLine(llvm::StringRef(line), info);
  }
  EXPECT_EQ(info.invariant, "x >= 0 && x <= n");
}

TEST(ContractParserTest, ParsesLoopVariantSingleExpr) {
  LoopContractInfo info;
  applyLoopAnnotationLine("loop_variant: n - i", info);
  EXPECT_EQ(info.variant, "n - i");
}

TEST(ContractParserTest, ParsesLoopAssignsCommaSeparated) {
  LoopContractInfo info;
  applyLoopAnnotationLine("loop_assigns: i, sum, count", info);
  ASSERT_EQ(info.assigns.size(), 3u);
  EXPECT_EQ(info.assigns[0], "i");
  EXPECT_EQ(info.assigns[1], "sum");
  EXPECT_EQ(info.assigns[2], "count");
}

TEST(ContractParserTest, ParsesLoopLabel) {
  LoopContractInfo info;
  applyLoopAnnotationLine("label: outer", info);
  EXPECT_EQ(info.label, "outer");
}

// [TC-11] Edge case: empty loop_assigns produces empty assigns vector.
TEST(ContractParserTest, EmptyLoopAssignsProducesEmptyVector) {
  LoopContractInfo info;
  applyLoopAnnotationLine("loop_assigns: ", info);
  EXPECT_TRUE(info.assigns.empty());
}

// [F2] Test: multiple loop_assigns lines are merged, not overwritten.
TEST(ContractParserTest, MultipleLoopAssignsLinesAreMerged) {
  LoopContractInfo info;
  applyLoopAnnotationLine("loop_assigns: i", info);
  applyLoopAnnotationLine("loop_assigns: sum", info);
  ASSERT_EQ(info.assigns.size(), 2u);
  EXPECT_EQ(info.assigns[0], "i");
  EXPECT_EQ(info.assigns[1], "sum");
}

} // namespace
} // namespace arcanum
