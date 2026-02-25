#include "arcanum/dialect/Lowering.h"
#include "arcanum/DiagnosticTracker.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/frontend/ContractParser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <tuple>

namespace arcanum {
namespace {

/// Test fixture that resets DiagnosticTracker before each test to avoid
/// cross-test contamination from fallback counts (CR-10).
class LoweringTestFixture : public ::testing::Test {
protected:
  void SetUp() override { DiagnosticTracker::reset(); }
  void TearDown() override { DiagnosticTracker::reset(); }
};

// ---------------------------------------------------------------------------
// Individual tests for unique lowering scenarios
// ---------------------------------------------------------------------------

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
    EXPECT_TRUE(funcOp.getRequiresAttr().has_value());
    EXPECT_TRUE(funcOp.getEnsuresAttr().has_value());
    EXPECT_FALSE(funcOp.getBody().empty());
    auto& block = funcOp.getBody().front();
    EXPECT_EQ(block.getNumArguments(), 1u);
    bool hasReturn = false;
    for (auto& op : block.getOperations()) {
      if (llvm::isa<arc::ReturnOp>(&op)) {
        hasReturn = true;
      }
    }
    EXPECT_TRUE(hasReturn);
  });
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
    EXPECT_FALSE(funcOp.getRequiresAttr().has_value());
    EXPECT_FALSE(funcOp.getEnsuresAttr().has_value());
  });
}

/// Merged from LoweringRegressionTest.cpp:
/// [F1] Null pointer dereference when lowering void function with explicit
/// return; statement.
TEST_F(LoweringTestFixture, VoidReturnDoesNotCrash) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    void doNothing() { return; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;

  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  EXPECT_TRUE(module);
}

// B6: Assignment lowering test
TEST_F(LoweringTestFixture, LowersAssignment) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t withAssign(int32_t a) {
      int32_t x = a;
      x = a + 1;
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

  bool foundAssign = false;
  module->walk([&](arc::AssignOp) { foundAssign = true; });
  EXPECT_TRUE(foundAssign);
}

// ---------------------------------------------------------------------------
// TEST_P: Arithmetic lowering (Sub, Mul, Div, Rem)
// ---------------------------------------------------------------------------

using ArithLoweringParam = std::tuple<std::string, std::string, std::string>;

class ArithmeticLoweringTest
    : public LoweringTestFixture,
      public ::testing::WithParamInterface<ArithLoweringParam> {};

TEST_P(ArithmeticLoweringTest, LowersArithmeticOp) {
  auto [name, cppOperator, expectedOpName] = GetParam();

  std::string code = R"(
    #include <cstdint>
    int32_t arith(int32_t a, int32_t b) {
      return a )" + cppOperator +
                     R"( b;
    }
  )";

  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundOp = false;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == expectedOpName) {
      foundOp = true;
    }
  });
  EXPECT_TRUE(foundOp) << "Expected to find " << expectedOpName;
}

INSTANTIATE_TEST_SUITE_P(
    ArithOps, ArithmeticLoweringTest,
    ::testing::Values(ArithLoweringParam{"Sub", "- ", "arc.sub"},
                      ArithLoweringParam{"Mul", "* ", "arc.mul"},
                      ArithLoweringParam{"Div", "/ ", "arc.div"},
                      ArithLoweringParam{"Rem", "% ", "arc.rem"}),
    [](const ::testing::TestParamInfo<ArithLoweringParam>& info) {
      return std::get<0>(info.param);
    });

// ---------------------------------------------------------------------------
// TEST_P: Comparison predicate lowering (gt, le, ge, eq, ne)
// ---------------------------------------------------------------------------

using CmpLoweringParam = std::tuple<std::string, std::string, std::string>;

class ComparisonLoweringTest
    : public LoweringTestFixture,
      public ::testing::WithParamInterface<CmpLoweringParam> {};

TEST_P(ComparisonLoweringTest, LowersComparisonPredicate) {
  auto [name, cppOperator, expectedPredicate] = GetParam();

  std::string code = R"(
    #include <cstdint>
    bool cmp(int32_t a) {
      return a )" + cppOperator +
                     R"( 0;
    }
  )";

  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundPredicate = false;
  module->walk([&](arc::CmpOp cmpOp) {
    if (cmpOp.getPredicate() == expectedPredicate) {
      foundPredicate = true;
    }
  });
  EXPECT_TRUE(foundPredicate)
      << "Expected CmpOp with predicate '" << expectedPredicate << "'";
}

INSTANTIATE_TEST_SUITE_P(
    CmpPredicates, ComparisonLoweringTest,
    ::testing::Values(CmpLoweringParam{"Gt", "> ", "gt"},
                      CmpLoweringParam{"Le", "<= ", "le"},
                      CmpLoweringParam{"Ge", ">= ", "ge"},
                      CmpLoweringParam{"Eq", "== ", "eq"},
                      CmpLoweringParam{"Ne", "!= ", "ne"}),
    [](const ::testing::TestParamInfo<CmpLoweringParam>& info) {
      return std::get<0>(info.param);
    });

// ---------------------------------------------------------------------------
// TEST_P: Logical operator lowering (And, Or, Not)
// ---------------------------------------------------------------------------

using LogicalLoweringParam = std::tuple<std::string, std::string, std::string>;

class LogicalLoweringTest
    : public LoweringTestFixture,
      public ::testing::WithParamInterface<LogicalLoweringParam> {};

TEST_P(LogicalLoweringTest, LowersLogicalOp) {
  auto [name, code, expectedOpName] = GetParam();

  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundOp = false;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == expectedOpName) {
      foundOp = true;
    }
  });
  EXPECT_TRUE(foundOp) << "Expected to find " << expectedOpName;
}

INSTANTIATE_TEST_SUITE_P(
    LogicalOps, LogicalLoweringTest,
    ::testing::Values(LogicalLoweringParam{"And",
                                           R"(
    #include <cstdint>
    bool logicalAnd(int32_t a, int32_t b) {
      return a > 0 && b > 0;
    }
  )",
                                           "arc.and"},
                      LogicalLoweringParam{"Or",
                                           R"(
    #include <cstdint>
    bool logicalOr(int32_t a, int32_t b) {
      return a > 0 || b > 0;
    }
  )",
                                           "arc.or"},
                      LogicalLoweringParam{"Not",
                                           R"(
    bool logicalNot(bool a) {
      return !a;
    }
  )",
                                           "arc.not"}),
    [](const ::testing::TestParamInfo<LogicalLoweringParam>& info) {
      return std::get<0>(info.param);
    });

} // namespace
} // namespace arcanum
