#include "backend/WhyMLEmitter.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"
#include "dialect/Lowering.h"
#include "frontend/ContractParser.h"

#include "clang/Tooling/Tooling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(WhyMLEmitterTest, EmitsSafeAddModule) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) {
      return a + b;
    }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
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
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ requires: b >= 0 && b <= 1000
    //@ ensures: \result >= 0 && \result <= 2000
    int32_t safe_add(int32_t a, int32_t b) {
      return a + b;
    }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // Check overflow bounds are emitted
  EXPECT_NE(result->whymlText.find("-2147483648"), std::string::npos);
  EXPECT_NE(result->whymlText.find("2147483647"), std::string::npos);
}

TEST(WhyMLEmitterTest, LocationMapPopulated) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t foo(int32_t a) { return a; }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
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

} // namespace
} // namespace arcanum
