#include "backend/WhyMLEmitter.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"
#include "dialect/Lowering.h"
#include "frontend/ContractParser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(WhyMLEmitterTest, EmitsSafeAddModule) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) {
      return a + b;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // Check WhyML text contains expected constructs
  EXPECT_NE(result->whymlText.find("module"), std::string::npos);
  EXPECT_NE(result->whymlText.find("use int.Int"), std::string::npos);
  EXPECT_NE(result->whymlText.find("safe_add"), std::string::npos);
  EXPECT_NE(result->whymlText.find("requires"), std::string::npos);
  EXPECT_NE(result->whymlText.find("ensures"), std::string::npos);
  EXPECT_NE(result->whymlText.find("end"), std::string::npos);
}

TEST(WhyMLEmitterTest, EmitsOverflowAssertions) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) {
      return a + b;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // Check overflow bounds are emitted as a conjunction (valid WhyML)
  EXPECT_NE(result->whymlText.find("-2147483648"), std::string::npos);
  EXPECT_NE(result->whymlText.find("2147483647"), std::string::npos);
  // Verify conjunction form, not chained comparison
  EXPECT_NE(result->whymlText.find("/\\"), std::string::npos);
}

TEST(WhyMLEmitterTest, LocationMapPopulated) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t foo(int32_t a) { return a; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->locationMap.empty());
}

// TC-16: Test contractToWhyML translation rules
TEST(WhyMLEmitterTest, ContractTranslationInWhyML) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ ensures: \result >= 0
    int32_t foo(int32_t a) { return a; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // && should become /\ in WhyML
  EXPECT_NE(result->whymlText.find("/\\"), std::string::npos);
  // \result should become result in WhyML
  EXPECT_NE(result->whymlText.find("result"), std::string::npos);
}

// TC-17: Test module name conversion
TEST(WhyMLEmitterTest, SnakeCaseToModuleName) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t my_func_name(int32_t a) { return a; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // my_func_name should become MyFuncName module
  EXPECT_NE(result->whymlText.find("module MyFuncName"), std::string::npos);
}

// TC-18: Test subtraction overflow assertion
TEST(WhyMLEmitterTest, EmitsSubtractionOverflowAssertion) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t sub(int32_t a, int32_t b) { return a - b; }
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

  // Should have overflow assertion for subtraction
  EXPECT_NE(result->whymlText.find("assert"), std::string::npos);
  EXPECT_NE(result->whymlText.find("-2147483648"), std::string::npos);
}

// TC-18: Test division-by-zero assertion
TEST(WhyMLEmitterTest, EmitsDivisionByZeroAssertion) {
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

  // Should have divisor-not-zero assertion
  EXPECT_NE(result->whymlText.find("<> 0"), std::string::npos);
  EXPECT_NE(result->whymlText.find("div"), std::string::npos);
}

// TC-18: Test VarOp emission
TEST(WhyMLEmitterTest, EmitsLetBinding) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t withVar(int32_t a) {
      int32_t x = a + 1;
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

  // Should emit a "let x = ..." binding
  EXPECT_NE(result->whymlText.find("let x ="), std::string::npos);
}

// TC-19: Test empty module -> nullopt
TEST(WhyMLEmitterTest, EmptyModuleReturnsNullopt) {
  mlir::MLIRContext mlirCtx;
  mlirCtx.getOrLoadDialect<arc::ArcDialect>();
  mlir::OpBuilder builder(&mlirCtx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto result = emitWhyML(module);
  EXPECT_FALSE(result.has_value());

  module->destroy();
}

// TC-18: Test comparison emission
TEST(WhyMLEmitterTest, EmitsComparisonExpression) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    bool isPositive(int32_t a) { return a > 0; }
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

  // Should contain the > comparison
  EXPECT_NE(result->whymlText.find(">"), std::string::npos);
}

// TC-18: Test original C++ parameter names in WhyML
TEST(WhyMLEmitterTest, UsesOriginalParameterNames) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: count >= 0
    int32_t foo(int32_t count) { return count; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // Should use "count" not "arg0"
  EXPECT_NE(result->whymlText.find("(count: int)"), std::string::npos);
  EXPECT_EQ(result->whymlText.find("arg0"), std::string::npos);
}

} // namespace
} // namespace arcanum
