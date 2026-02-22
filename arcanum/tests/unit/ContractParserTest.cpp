#include "frontend/ContractParser.h"

#include "clang/AST/ASTContext.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace arcanum {
namespace {

// ============================================================
// Expression Parser Tests (standalone, no Clang AST needed)
// ============================================================

TEST(ContractExprParserTest, ParsesIntegerLiteral) {
    auto result = parseContractExpr("42");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<IntLiteral>(result.expr->node));
    EXPECT_EQ(std::get<IntLiteral>(result.expr->node).value, 42);
}

TEST(ContractExprParserTest, ParsesNegativeNumberAsUnary) {
    auto result = parseContractExpr("-5");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<UnaryExpr>(result.expr->node));
    auto &unary = std::get<UnaryExpr>(result.expr->node);
    EXPECT_EQ(unary.op, UnaryOp::Neg);
    ASSERT_TRUE(std::holds_alternative<IntLiteral>(unary.operand->node));
    EXPECT_EQ(std::get<IntLiteral>(unary.operand->node).value, 5);
}

TEST(ContractExprParserTest, ParsesBoolLiteralTrue) {
    auto result = parseContractExpr("true");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BoolLiteral>(result.expr->node));
    EXPECT_TRUE(std::get<BoolLiteral>(result.expr->node).value);
}

TEST(ContractExprParserTest, ParsesBoolLiteralFalse) {
    auto result = parseContractExpr("false");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BoolLiteral>(result.expr->node));
    EXPECT_FALSE(std::get<BoolLiteral>(result.expr->node).value);
}

TEST(ContractExprParserTest, ParsesIdentifier) {
    auto result = parseContractExpr("foo");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<Identifier>(result.expr->node));
    EXPECT_EQ(std::get<Identifier>(result.expr->node).name, "foo");
}

TEST(ContractExprParserTest, ParsesResultRef) {
    auto result = parseContractExpr("\\result");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<ResultRef>(result.expr->node));
}

TEST(ContractExprParserTest, ParsesSimpleAddition) {
    auto result = parseContractExpr("a + b");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    auto &binary = std::get<BinaryExpr>(result.expr->node);
    EXPECT_EQ(binary.op, BinOp::Add);
}

TEST(ContractExprParserTest, ParsesComparisonChain) {
    // "a >= 0 && a <= 1000" should parse as (a >= 0) && (a <= 1000)
    auto result = parseContractExpr("a >= 0 && a <= 1000");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    auto &andExpr = std::get<BinaryExpr>(result.expr->node);
    EXPECT_EQ(andExpr.op, BinOp::And);

    // LHS: a >= 0
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(andExpr.lhs->node));
    auto &lhs = std::get<BinaryExpr>(andExpr.lhs->node);
    EXPECT_EQ(lhs.op, BinOp::Ge);

    // RHS: a <= 1000
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(andExpr.rhs->node));
    auto &rhs = std::get<BinaryExpr>(andExpr.rhs->node);
    EXPECT_EQ(rhs.op, BinOp::Le);
}

TEST(ContractExprParserTest, ParsesResultInEnsures) {
    // "\\result >= 0 && \\result <= 2000"
    auto result = parseContractExpr("\\result >= 0 && \\result <= 2000");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    auto &andExpr = std::get<BinaryExpr>(result.expr->node);
    EXPECT_EQ(andExpr.op, BinOp::And);

    // LHS: \result >= 0
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(andExpr.lhs->node));
    auto &lhsBin = std::get<BinaryExpr>(andExpr.lhs->node);
    ASSERT_TRUE(std::holds_alternative<ResultRef>(lhsBin.lhs->node));
}

TEST(ContractExprParserTest, ParsesArithmeticPrecedence) {
    // "a + b * c" should parse as a + (b * c)
    auto result = parseContractExpr("a + b * c");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    auto &addExpr = std::get<BinaryExpr>(result.expr->node);
    EXPECT_EQ(addExpr.op, BinOp::Add);

    // LHS is just "a"
    ASSERT_TRUE(std::holds_alternative<Identifier>(addExpr.lhs->node));

    // RHS is "b * c"
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(addExpr.rhs->node));
    EXPECT_EQ(std::get<BinaryExpr>(addExpr.rhs->node).op, BinOp::Mul);
}

TEST(ContractExprParserTest, ParsesParentheses) {
    // "(a + b) * c" should parse as (a + b) * c
    auto result = parseContractExpr("(a + b) * c");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    auto &mulExpr = std::get<BinaryExpr>(result.expr->node);
    EXPECT_EQ(mulExpr.op, BinOp::Mul);

    // LHS is "(a + b)"
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(mulExpr.lhs->node));
    EXPECT_EQ(std::get<BinaryExpr>(mulExpr.lhs->node).op, BinOp::Add);
}

TEST(ContractExprParserTest, ParsesLogicalNot) {
    auto result = parseContractExpr("!x");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<UnaryExpr>(result.expr->node));
    auto &unary = std::get<UnaryExpr>(result.expr->node);
    EXPECT_EQ(unary.op, UnaryOp::Not);
    ASSERT_TRUE(std::holds_alternative<Identifier>(unary.operand->node));
}

TEST(ContractExprParserTest, ParsesLogicalOr) {
    auto result = parseContractExpr("a || b");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    EXPECT_EQ(std::get<BinaryExpr>(result.expr->node).op, BinOp::Or);
}

TEST(ContractExprParserTest, ParsesEqualityAndInequality) {
    auto result = parseContractExpr("a == 0 || b != 1");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    auto &orExpr = std::get<BinaryExpr>(result.expr->node);
    EXPECT_EQ(orExpr.op, BinOp::Or);

    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(orExpr.lhs->node));
    EXPECT_EQ(std::get<BinaryExpr>(orExpr.lhs->node).op, BinOp::Eq);

    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(orExpr.rhs->node));
    EXPECT_EQ(std::get<BinaryExpr>(orExpr.rhs->node).op, BinOp::Ne);
}

TEST(ContractExprParserTest, ParsesModulo) {
    auto result = parseContractExpr("x % 2");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    EXPECT_EQ(std::get<BinaryExpr>(result.expr->node).op, BinOp::Mod);
}

TEST(ContractExprParserTest, ParsesDivision) {
    auto result = parseContractExpr("x / y");
    ASSERT_TRUE(result.success());
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    EXPECT_EQ(std::get<BinaryExpr>(result.expr->node).op, BinOp::Div);
}

TEST(ContractExprParserTest, ParsesComplexExpression) {
    // Full Slice 1 example: a >= 0 && a <= 1000 && b >= 0 && b <= 1000
    auto result = parseContractExpr(
        "a >= 0 && a <= 1000 && b >= 0 && b <= 1000");
    ASSERT_TRUE(result.success());
    // Should be a tree of && operations
    ASSERT_TRUE(std::holds_alternative<BinaryExpr>(result.expr->node));
    EXPECT_EQ(std::get<BinaryExpr>(result.expr->node).op, BinOp::And);
}

// ============================================================
// Error handling
// ============================================================

TEST(ContractExprParserTest, ReportsErrorOnEmptyInput) {
    auto result = parseContractExpr("");
    EXPECT_FALSE(result.success());
    EXPECT_GE(result.errors.size(), 1u);
}

TEST(ContractExprParserTest, ReportsErrorOnInvalidToken) {
    auto result = parseContractExpr("@");
    EXPECT_FALSE(result.success());
}

TEST(ContractExprParserTest, ReportsErrorOnUnmatchedParen) {
    auto result = parseContractExpr("(a + b");
    EXPECT_FALSE(result.success());
}

TEST(ContractExprParserTest, ReportsErrorOnTrailingTokens) {
    auto result = parseContractExpr("a b");
    EXPECT_FALSE(result.success());
}

// ============================================================
// contractExprToString tests
// ============================================================

TEST(ContractExprToStringTest, PrintsIntLiteral) {
    auto expr = makeIntLiteral(42);
    EXPECT_EQ(contractExprToString(*expr), "42");
}

TEST(ContractExprToStringTest, PrintsBoolLiteral) {
    EXPECT_EQ(contractExprToString(*makeBoolLiteral(true)), "true");
    EXPECT_EQ(contractExprToString(*makeBoolLiteral(false)), "false");
}

TEST(ContractExprToStringTest, PrintsIdentifier) {
    auto expr = makeIdentifier("x");
    EXPECT_EQ(contractExprToString(*expr), "x");
}

TEST(ContractExprToStringTest, PrintsResultRef) {
    auto expr = makeResultRef();
    EXPECT_EQ(contractExprToString(*expr), "\\result");
}

TEST(ContractExprToStringTest, PrintsBinaryExpr) {
    auto expr = makeBinaryExpr(BinOp::Add, makeIdentifier("a"),
                               makeIdentifier("b"));
    EXPECT_EQ(contractExprToString(*expr), "(a + b)");
}

TEST(ContractExprToStringTest, PrintsUnaryNot) {
    auto expr = makeUnaryExpr(UnaryOp::Not, makeIdentifier("x"));
    EXPECT_EQ(contractExprToString(*expr), "!x");
}

TEST(ContractExprToStringTest, PrintsNestedExpression) {
    // (a >= 0) && (a <= 1000)
    auto lhs = makeBinaryExpr(BinOp::Ge, makeIdentifier("a"),
                              makeIntLiteral(0));
    auto rhs = makeBinaryExpr(BinOp::Le, makeIdentifier("a"),
                              makeIntLiteral(1000));
    auto expr = makeBinaryExpr(BinOp::And, std::move(lhs), std::move(rhs));
    EXPECT_EQ(contractExprToString(*expr), "((a >= 0) && (a <= 1000))");
}

// ============================================================
// ContractParser Integration Tests (with Clang AST)
// ============================================================

/// Helper: parse C++ source with contract annotations and run ContractParser.
ContractParserResult parseContractsFromSource(const std::string &source) {
    auto ast = clang::tooling::buildASTFromCodeWithArgs(
        source, {"-std=c++20", "-fsyntax-only", "-fparse-all-comments"},
        "test_input.cpp");
    EXPECT_NE(ast, nullptr);
    if (!ast)
        return ContractParserResult{};

    ContractParser parser(ast->getASTContext());
    return parser.parse();
}

TEST(ContractParserTest, ParsesRequiresAnnotation) {
    auto result = parseContractsFromSource(R"cpp(
        #include <cstdint>
        //@ requires: a >= 0
        int32_t f(int32_t a) { return a; }
    )cpp");
    EXPECT_FALSE(result.hasErrors());
    ASSERT_EQ(result.contracts.size(), 1u);

    auto &info = result.contracts[0].second;
    ASSERT_EQ(info.preconditions.size(), 1u);
    EXPECT_EQ(info.postconditions.size(), 0u);
    EXPECT_EQ(info.preconditions[0].kind, ContractKind::Requires);
}

TEST(ContractParserTest, ParsesEnsuresAnnotation) {
    auto result = parseContractsFromSource(R"cpp(
        #include <cstdint>
        //@ ensures: \result >= 0
        int32_t f(int32_t a) { return a; }
    )cpp");
    EXPECT_FALSE(result.hasErrors());
    ASSERT_EQ(result.contracts.size(), 1u);

    auto &info = result.contracts[0].second;
    EXPECT_EQ(info.preconditions.size(), 0u);
    ASSERT_EQ(info.postconditions.size(), 1u);
    EXPECT_EQ(info.postconditions[0].kind, ContractKind::Ensures);
}

TEST(ContractParserTest, ParsesMultipleAnnotations) {
    auto result = parseContractsFromSource(R"cpp(
        #include <cstdint>
        //@ requires: a >= 0 && a <= 1000
        //@ requires: b >= 0 && b <= 1000
        //@ ensures: \result >= 0 && \result <= 2000
        int32_t safe_add(int32_t a, int32_t b) {
            return a + b;
        }
    )cpp");
    EXPECT_FALSE(result.hasErrors());
    ASSERT_EQ(result.contracts.size(), 1u);

    auto &info = result.contracts[0].second;
    EXPECT_EQ(info.preconditions.size(), 2u);
    EXPECT_EQ(info.postconditions.size(), 1u);
}

TEST(ContractParserTest, ParsesNoAnnotations) {
    auto result = parseContractsFromSource(R"cpp(
        int f(int a) { return a; }
    )cpp");
    EXPECT_FALSE(result.hasErrors());
    EXPECT_EQ(result.contracts.size(), 0u);
}

TEST(ContractParserTest, IgnoresNonContractComments) {
    auto result = parseContractsFromSource(R"cpp(
        // This is a regular comment
        int f(int a) { return a; }
    )cpp");
    EXPECT_FALSE(result.hasErrors());
    EXPECT_EQ(result.contracts.size(), 0u);
}

TEST(ContractParserTest, ParsesMultipleFunctions) {
    auto result = parseContractsFromSource(R"cpp(
        #include <cstdint>

        //@ requires: a >= 0
        int32_t abs_val(int32_t a) {
            if (a < 0) return -a;
            return a;
        }

        //@ ensures: \result >= 0
        int32_t identity(int32_t x) { return x; }
    )cpp");
    EXPECT_FALSE(result.hasErrors());
    EXPECT_EQ(result.contracts.size(), 2u);
}

} // namespace
} // namespace arcanum
