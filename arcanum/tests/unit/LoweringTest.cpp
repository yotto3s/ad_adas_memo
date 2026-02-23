#include "arcanum/DiagnosticTracker.h"
#include "arcanum/dialect/Lowering.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/frontend/ContractParser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

/// Test fixture that resets DiagnosticTracker before each test to avoid
/// cross-test contamination from fallback counts (CR-10).
class LoweringTestFixture : public ::testing::Test {
protected:
  void SetUp() override { DiagnosticTracker::reset(); }
  void TearDown() override { DiagnosticTracker::reset(); }
};

TEST_F(LoweringTestFixture, LowersSimpleAddFunction) {
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

  // Check that we have at least one arc.func operation
  bool foundFunc = false;
  module->walk([&](arc::FuncOp funcOp) {
    EXPECT_EQ(funcOp.getSymName(), "safe_add");
    foundFunc = true;
  });
  EXPECT_TRUE(foundFunc);
}

TEST_F(LoweringTestFixture, LowersIfElseFunction) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t myAbs(int32_t x) {
      if (x < 0) {
        return -x;
      } else {
        return x;
      }
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundIf = false;
  module->walk([&](arc::IfOp ifOp) { foundIf = true; });
  EXPECT_TRUE(foundIf);
}

// TC-12: Verify contract attributes and body operations on lowered FuncOp
TEST_F(LoweringTestFixture, FuncOpHasContractAttributesAndBody) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: a >= 0 && a <= 1000
    //@ ensures: \result >= 0
    int32_t identity(int32_t a) {
      return a;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  module->walk([&](arc::FuncOp funcOp) {
    EXPECT_EQ(funcOp.getSymName(), "identity");
    // Check contract attributes are set
    EXPECT_TRUE(funcOp.getRequiresAttr().has_value());
    EXPECT_TRUE(funcOp.getEnsuresAttr().has_value());
    // Check the function has a body with at least a return
    EXPECT_FALSE(funcOp.getBody().empty());
    auto& block = funcOp.getBody().front();
    EXPECT_EQ(block.getNumArguments(), 1u); // one parameter
    // Should end with a return op
    bool hasReturn = false;
    for (auto& op : block.getOperations()) {
      if (llvm::isa<arc::ReturnOp>(&op)) {
        hasReturn = true;
      }
    }
    EXPECT_TRUE(hasReturn);
  });
}

// TC-13: Lowering various expression types
TEST_F(LoweringTestFixture, LowersSubtraction) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t sub(int32_t a, int32_t b) {
      return a - b;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundSub = false;
  module->walk([&](arc::SubOp) { foundSub = true; });
  EXPECT_TRUE(foundSub);
}

TEST_F(LoweringTestFixture, LowersComparison) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    bool isPositive(int32_t a) {
      return a > 0;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundCmp = false;
  module->walk([&](arc::CmpOp) { foundCmp = true; });
  EXPECT_TRUE(foundCmp);
}

TEST_F(LoweringTestFixture, LowersVariableDeclaration) {
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

  bool foundVar = false;
  module->walk([&](arc::VarOp varOp) {
    EXPECT_EQ(varOp.getName(), "x");
    foundVar = true;
  });
  EXPECT_TRUE(foundVar);
}

TEST_F(LoweringTestFixture, LowersFunctionWithoutContracts) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t noContract(int32_t a) {
      return a;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  module->walk([&](arc::FuncOp funcOp) {
    // No contract attributes should be set
    EXPECT_FALSE(funcOp.getRequiresAttr().has_value());
    EXPECT_FALSE(funcOp.getEnsuresAttr().has_value());
  });
}

} // namespace
} // namespace arcanum
