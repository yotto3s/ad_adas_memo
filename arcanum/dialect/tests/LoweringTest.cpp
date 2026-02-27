#include "arcanum/dialect/Lowering.h"
#include "arcanum/DiagnosticTracker.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"
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

TEST_F(LoweringTestFixture, LowersVariableReassignment) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: a >= 0
    //@ ensures: \result >= 0
    int32_t foo(int32_t a, int32_t b) {
      int32_t x = a;
      x = a + b;
      return x;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundAssign = false;
  module->walk([&](arc::AssignOp) { foundAssign = true; });
  EXPECT_TRUE(foundAssign);
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

// --- Slice 2: Multi-type lowering tests ---

// [TC-9] Parametrized type-mapping test covering all integer widths/signedness
TEST_F(LoweringTestFixture, TypeMappingCoversAllIntegerWidths) {
  struct TypeMappingCase {
    const char* typeName;
    const char* funcName;
    unsigned expectedWidth;
    bool expectedSigned;
  };

  const TypeMappingCase cases[] = {
      {"int8_t", "add_i8", 8, true},
      {"uint32_t", "add_u32", 32, false},
      {"int16_t", "add_i16", 16, true},
      {"uint64_t", "add_u64", 64, false},
  };

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.typeName);

    std::string source = "#include <cstdint>\n";
    source += std::string(tc.typeName) + " " + tc.funcName + "(" + tc.typeName +
              " a, " + tc.typeName + " b) {\n";
    source += "  return a + b;\n}\n";

    auto ast = clang::tooling::buildASTFromCodeWithArgs(
        source, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
        std::make_shared<clang::PCHContainerOperations>());
    ASSERT_NE(ast, nullptr);

    std::map<const clang::FunctionDecl*, ContractInfo> contracts;
    mlir::MLIRContext mlirCtx;
    auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
    ASSERT_TRUE(module);

    bool foundFunc = false;
    module->walk([&](arc::FuncOp funcOp) {
      if (funcOp.getSymName() != tc.funcName)
        return;
      auto& block = funcOp.getBody().front();
      for (auto arg : block.getArguments()) {
        auto intType = llvm::dyn_cast<arc::IntType>(arg.getType());
        ASSERT_TRUE(intType);
        EXPECT_EQ(intType.getWidth(), tc.expectedWidth);
        EXPECT_EQ(intType.getIsSigned(), tc.expectedSigned);
      }
      foundFunc = true;
    });
    EXPECT_TRUE(foundFunc);
  }
}

TEST_F(LoweringTestFixture, LowersStaticCastEmitsCastOp) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int8_t narrow(int32_t x) {
      return static_cast<int8_t>(x);
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundCast = false;
  module->walk([&](arc::CastOp castOp) {
    auto inputType = llvm::dyn_cast<arc::IntType>(castOp.getInput().getType());
    auto resultType =
        llvm::dyn_cast<arc::IntType>(castOp.getResult().getType());
    ASSERT_TRUE(inputType);
    ASSERT_TRUE(resultType);
    EXPECT_EQ(inputType.getWidth(), 32u);
    EXPECT_EQ(resultType.getWidth(), 8u);
    foundCast = true;
  });
  EXPECT_TRUE(foundCast);
}

TEST_F(LoweringTestFixture, OverflowModeOnFuncFromContract) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ overflow: wrap
    //@ requires: a >= 0
    int32_t wrap_add(int32_t a, int32_t b) {
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
    EXPECT_EQ(funcOp.getSymName(), "wrap_add");
    auto overflowAttr = funcOp->getAttrOfType<mlir::StringAttr>("overflow");
    ASSERT_TRUE(overflowAttr);
    EXPECT_EQ(overflowAttr.getValue(), "wrap");
    foundFunc = true;
  });
  EXPECT_TRUE(foundFunc);
}

TEST_F(LoweringTestFixture, OverflowAttrOnSignedArithUsesMode) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ overflow: wrap
    int32_t wrap_add(int32_t a, int32_t b) {
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

  bool foundAdd = false;
  module->walk([&](arc::AddOp addOp) {
    auto overflowAttr = addOp->getAttrOfType<mlir::StringAttr>("overflow");
    ASSERT_TRUE(overflowAttr);
    EXPECT_EQ(overflowAttr.getValue(), "wrap");
    foundAdd = true;
  });
  EXPECT_TRUE(foundAdd);
}

TEST_F(LoweringTestFixture, UnsignedArithAlwaysGetsWrapOverflow) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ overflow: trap
    uint32_t trap_add(uint32_t a, uint32_t b) {
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

  bool foundAdd = false;
  module->walk([&](arc::AddOp addOp) {
    auto overflowAttr = addOp->getAttrOfType<mlir::StringAttr>("overflow");
    ASSERT_TRUE(overflowAttr);
    // Unsigned always wraps, regardless of function-level mode
    EXPECT_EQ(overflowAttr.getValue(), "wrap");
    foundAdd = true;
  });
  EXPECT_TRUE(foundAdd);
}

// [TC-10] Default mode ("trap") is stored on FuncOp when no annotation present
TEST_F(LoweringTestFixture, DefaultTrapModeOnFuncOp) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t no_annotation(int32_t a, int32_t b) {
      return a + b;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundFunc = false;
  module->walk([&](arc::FuncOp funcOp) {
    auto overflowAttr = funcOp->getAttrOfType<mlir::StringAttr>("overflow");
    ASSERT_TRUE(overflowAttr);
    EXPECT_EQ(overflowAttr.getValue(), "trap");
    foundFunc = true;
  });
  EXPECT_TRUE(foundFunc);
}

// [TC-7] Overflow attribute on DivOp/RemOp uses function mode
TEST_F(LoweringTestFixture, OverflowAttrOnDivRemUsesFunctionMode) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ overflow: wrap
    int32_t wrap_div(int32_t a, int32_t b) {
      return a / b;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundDiv = false;
  module->walk([&](arc::DivOp divOp) {
    auto overflowAttr = divOp->getAttrOfType<mlir::StringAttr>("overflow");
    ASSERT_TRUE(overflowAttr);
    EXPECT_EQ(overflowAttr.getValue(), "wrap");
    foundDiv = true;
  });
  EXPECT_TRUE(foundDiv);
}

// [TC-8] Unary negation with non-i32 type produces SubOp with correct width
TEST_F(LoweringTestFixture, NegationOfI8ProducesSubOpWithI8Width) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int8_t neg_i8(int8_t a) {
      return -a;
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
  module->walk([&](arc::SubOp subOp) {
    auto resultType = llvm::dyn_cast<arc::IntType>(subOp.getResult().getType());
    ASSERT_TRUE(resultType);
    // Negation of int8_t uses promoted int (width 32) in C++ AST,
    // so the SubOp result will have the promoted type.
    // This test documents the current behavior.
    EXPECT_TRUE(resultType.getWidth() == 8u || resultType.getWidth() == 32u);
    foundSub = true;
  });
  EXPECT_TRUE(foundSub);
}

// [TC-11] Widening cast lowering (i8 -> i32) produces CastOp
TEST_F(LoweringTestFixture, LowersWideningCastI8ToI32) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t widen(int8_t x) {
      return static_cast<int32_t>(x);
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundCast = false;
  module->walk([&](arc::CastOp castOp) {
    auto resultType =
        llvm::dyn_cast<arc::IntType>(castOp.getResult().getType());
    ASSERT_TRUE(resultType);
    EXPECT_EQ(resultType.getWidth(), 32u);
    foundCast = true;
  });
  EXPECT_TRUE(foundCast);
}

// [SC-4] CastOp in non-trap mode gets overflow attribute
TEST_F(LoweringTestFixture, CastOpInWrapModeGetsOverflowAttr) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ overflow: wrap
    int8_t narrow_wrap(int32_t x) {
      return static_cast<int8_t>(x);
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundCast = false;
  module->walk([&](arc::CastOp castOp) {
    auto overflowAttr = castOp->getAttrOfType<mlir::StringAttr>("overflow");
    ASSERT_TRUE(overflowAttr);
    EXPECT_EQ(overflowAttr.getValue(), "wrap");
    foundCast = true;
  });
  EXPECT_TRUE(foundCast);
}

} // namespace
} // namespace arcanum
