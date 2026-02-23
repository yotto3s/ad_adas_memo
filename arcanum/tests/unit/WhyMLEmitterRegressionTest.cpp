/// Regression tests for WhyMLEmitter.cpp bug fixes.

#include "arcanum/backend/WhyMLEmitter.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"
#include "arcanum/dialect/Lowering.h"
#include "arcanum/frontend/ContractParser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

/// [F2] Missing "use int.ComputerDivision" import for division/modulo.
///
/// The WhyML emitter only emits "use int.Int" but division (div) and
/// modulo (mod) require "use int.ComputerDivision" in Why3.  Without
/// this import, Why3 reports an unresolved symbol and fails.
///
/// This test verifies that when a function uses division, the emitted
/// WhyML text contains "use int.ComputerDivision" (or equivalent).
TEST(WhyMLEmitterRegressionTest, DivisionEmitsComputerDivisionImport) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t mydiv(int32_t a, int32_t b) { return a / b; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // The WhyML output must import int.ComputerDivision (or
  // int.EuclideanDivision) for the "div" function to be defined.
  // Before the fix, only "use int.Int" is imported.
  bool hasComputerDiv =
      result->whymlText.find("use int.ComputerDivision") != std::string::npos;
  bool hasEuclideanDiv =
      result->whymlText.find("use int.EuclideanDivision") != std::string::npos;
  EXPECT_TRUE(hasComputerDiv || hasEuclideanDiv)
      << "WhyML output is missing division import.  Generated text:\n"
      << result->whymlText;
}

/// [F2] Same bug for modulo: missing ComputerDivision import.
TEST(WhyMLEmitterRegressionTest, ModuloEmitsComputerDivisionImport) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t mymod(int32_t a, int32_t b) { return a % b; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  bool hasComputerDiv =
      result->whymlText.find("use int.ComputerDivision") != std::string::npos;
  bool hasEuclideanDiv =
      result->whymlText.find("use int.EuclideanDivision") != std::string::npos;
  EXPECT_TRUE(hasComputerDiv || hasEuclideanDiv)
      << "WhyML output is missing division import for mod.  Generated text:\n"
      << result->whymlText;
}

/// [F4] Division overflow (INT_MIN / -1) not checked in WhyML emission.
///
/// emitDivLikeOp only emits a divisor-not-zero assertion but no overflow
/// check.  INT32_MIN / -1 overflows in C (undefined behavior).  The
/// WhyML output should contain an overflow assertion for division results,
/// similar to what emitArithWithOverflowCheck does for add/sub/mul.
TEST(WhyMLEmitterRegressionTest, DivisionEmitsOverflowAssertion) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t mydiv(int32_t a, int32_t b) { return a / b; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // The WhyML output should have an overflow bounds assertion for division,
  // just like it does for addition/subtraction/multiplication.
  // Before the fix, only the divisor-not-zero assertion is present.
  bool hasDivNotZero = result->whymlText.find("<> 0") != std::string::npos;
  EXPECT_TRUE(hasDivNotZero) << "Divisor-not-zero assertion missing";

  // Check that overflow bounds are also present for division
  // (i.e., -2147483648 <= result <= 2147483647)
  bool hasOverflowLowerBound =
      result->whymlText.find("-2147483648") != std::string::npos;
  bool hasOverflowUpperBound =
      result->whymlText.find("2147483647") != std::string::npos;
  EXPECT_TRUE(hasOverflowLowerBound && hasOverflowUpperBound)
      << "Division result overflow assertion is missing.  INT_MIN / -1 would "
         "overflow but is not caught.  Generated text:\n"
      << result->whymlText;
}

/// [F5] if-without-else emits invalid WhyML.
///
/// When an arc.IfOp has no else region, the emitter outputs:
///   if cond then (body)
/// without an else clause.  WhyML is expression-based and requires
/// an else clause (or the then-branch must have type unit).  The
/// emitter produces neither.
///
/// We test this indirectly: a C++ function with if-without-else where the
/// if body does NOT contain a return (so SubsetEnforcer does not reject it)
/// but the body is non-trivial.  Since assignment lowering is unimplemented,
/// we use a variable declaration inside the if to produce a non-empty then
/// region.
TEST(WhyMLEmitterRegressionTest, IfWithoutElseEmitsElseClause) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t foo(int32_t a) {
      int32_t x = 0;
      if (a > 0) {
        int32_t y = a;
      }
      return x;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // The WhyML output must have an else clause for every if.
  // Look for "then" which only appears in if-then-else constructs
  auto thenPos = result->whymlText.find("then");
  if (thenPos != std::string::npos) {
    auto elsePos = result->whymlText.find("else", thenPos);
    EXPECT_NE(elsePos, std::string::npos)
        << "WhyML output has 'if ... then' without matching 'else', which is "
           "invalid WhyML syntax.  Generated text:\n"
        << result->whymlText;
  }
}

} // namespace
} // namespace arcanum
