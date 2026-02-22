#include "dialect/Lowering.h"
#include "frontend/ContractParser.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"

#include "clang/Tooling/Tooling.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(LoweringTest, LowersSimpleAddFunction) {
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

  // Check that we have at least one arc.func operation
  bool foundFunc = false;
  module->walk([&](arc::FuncOp funcOp) {
    EXPECT_EQ(funcOp.getSymName(), "safe_add");
    foundFunc = true;
  });
  EXPECT_TRUE(foundFunc);
}

TEST(LoweringTest, LowersIfElseFunction) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(R"(
    #include <cstdint>
    int32_t myAbs(int32_t x) {
      if (x < 0) {
        return -x;
      } else {
        return x;
      }
    }
  )", {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundIf = false;
  module->walk([&](arc::IfOp ifOp) {
    foundIf = true;
  });
  EXPECT_TRUE(foundIf);
}

} // namespace
} // namespace arcanum
