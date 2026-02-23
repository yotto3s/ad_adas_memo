#include "arcanum/backend/WhyMLEmitter.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"
#include "arcanum/dialect/Lowering.h"
#include "arcanum/frontend/ContractParser.h"

#include "mlir/IR/Builders.h"
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

// [W18/TC-12] Test remainder (%) WhyML emission
TEST(WhyMLEmitterTest, EmitsRemainderWithModAndDivisorCheck) {
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

  // Should contain "mod" and divisor assertion
  EXPECT_NE(result->whymlText.find("mod"), std::string::npos)
      << "WhyML output missing 'mod'.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("<> 0"), std::string::npos)
      << "WhyML output missing divisor check.  Text:\n"
      << result->whymlText;
}

// --- Parameterized: Contract Translation Tests ---
// Tests that C++ operators in contracts are correctly translated to WhyML.

struct ContractTranslationParam {
  std::string name;
  std::string contractLine; // The //@ line
  std::string expectedStr;  // Substring expected in WhyML output
  std::string absentStr;    // Substring that must NOT appear (empty = skip)
};

class ContractTranslationTest
    : public ::testing::TestWithParam<ContractTranslationParam> {};

TEST_P(ContractTranslationTest, TranslatesOperator) {
  const auto& param = GetParam();

  std::string code = R"(
    #include <cstdint>
    //@ )" + param.contractLine +
                     R"(
    int32_t foo(int32_t a) { return a; }
  )";

  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  if (!param.expectedStr.empty()) {
    EXPECT_NE(result->whymlText.find(param.expectedStr), std::string::npos)
        << "WhyML output missing '" << param.expectedStr << "'.  Text:\n"
        << result->whymlText;
  }

  if (!param.absentStr.empty()) {
    // For == -> = translation, check after "ensures" to avoid false matches
    if (param.absentStr == "==") {
      auto ensPos = result->whymlText.find("ensures");
      ASSERT_NE(ensPos, std::string::npos);
      auto afterEns = result->whymlText.substr(ensPos);
      EXPECT_EQ(afterEns.find(param.absentStr), std::string::npos)
          << "WhyML output still contains '" << param.absentStr << "'.  Text:\n"
          << result->whymlText;
    } else {
      EXPECT_EQ(result->whymlText.find(param.absentStr), std::string::npos)
          << "WhyML output still contains '" << param.absentStr << "'.  Text:\n"
          << result->whymlText;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    WhyMLEmitter, ContractTranslationTest,
    ::testing::Values(
        ContractTranslationParam{"AndToConjunction",
                                 "requires: a >= 0 && a <= 1000", "/\\", ""},
        ContractTranslationParam{"OrToDisjunction",
                                 "requires: a >= 0 || a <= 10", "\\/", ""},
        ContractTranslationParam{"NotEqualToDiamond", "requires: a != 0", "<>",
                                 ""},
        ContractTranslationParam{"EqualToSingle", "ensures: \\result == 0", "",
                                 "=="},
        ContractTranslationParam{"NotToWord", "requires: !false", "not ", ""}),
    [](const ::testing::TestParamInfo<ContractTranslationParam>& info) {
      return info.param.name;
    });

// --- Parameterized: Module Name Conversion Tests ---

struct ModuleNameParam {
  std::string name;
  std::string funcName;
  std::string expectedModule;
};

class ModuleNameTest : public ::testing::TestWithParam<ModuleNameParam> {};

TEST_P(ModuleNameTest, ConvertsName) {
  const auto& param = GetParam();

  std::string code = R"(
    #include <cstdint>
    //@ ensures: \result >= 0
    int32_t )" + param.funcName +
                     R"((int32_t a) { return a; }
  )";

  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  auto contracts = parseContracts(ast->getASTContext());
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("module " + param.expectedModule),
            std::string::npos)
      << "Expected 'module " << param.expectedModule
      << "' in WhyML output.  Text:\n"
      << result->whymlText;
}

INSTANTIATE_TEST_SUITE_P(
    WhyMLEmitter, ModuleNameTest,
    ::testing::Values(ModuleNameParam{"SnakeCase", "my_func_name",
                                      "MyFuncName"},
                      ModuleNameParam{"CamelCase", "myFunc", "MyFunc"}),
    [](const ::testing::TestParamInfo<ModuleNameParam>& info) {
      return info.param.name;
    });

// --- Parameterized: ComputerDivision Import Tests ---
// Merged from WhyMLEmitterRegressionTest: [F2] division/modulo require
// "use int.ComputerDivision".

struct ComputerDivisionImportParam {
  std::string name;
  std::string code;
};

class ComputerDivisionImportTest
    : public ::testing::TestWithParam<ComputerDivisionImportParam> {};

TEST_P(ComputerDivisionImportTest, EmitsImport) {
  const auto& param = GetParam();

  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      param.code, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
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
      << "WhyML output missing division import.  Generated text:\n"
      << result->whymlText;
}

INSTANTIATE_TEST_SUITE_P(
    WhyMLEmitter, ComputerDivisionImportTest,
    ::testing::Values(ComputerDivisionImportParam{"Division",
                                                  R"(
    #include <cstdint>
    int32_t mydiv(int32_t a, int32_t b) { return a / b; }
  )"},
                      ComputerDivisionImportParam{"Modulo",
                                                  R"(
    #include <cstdint>
    int32_t mymod(int32_t a, int32_t b) { return a % b; }
  )"}),
    [](const ::testing::TestParamInfo<ComputerDivisionImportParam>& info) {
      return info.param.name;
    });

// --- Merged from regression: [F4] Division overflow ---
TEST(WhyMLEmitterTest, DivisionEmitsOverflowAssertion) {
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

  // Divisor-not-zero assertion
  bool hasDivNotZero = result->whymlText.find("<> 0") != std::string::npos;
  EXPECT_TRUE(hasDivNotZero) << "Divisor-not-zero assertion missing";

  // Overflow bounds for division (INT_MIN / -1 would overflow)
  bool hasOverflowLowerBound =
      result->whymlText.find("-2147483648") != std::string::npos;
  bool hasOverflowUpperBound =
      result->whymlText.find("2147483647") != std::string::npos;
  EXPECT_TRUE(hasOverflowLowerBound && hasOverflowUpperBound)
      << "Division result overflow assertion is missing.  INT_MIN / -1 would "
         "overflow but is not caught.  Generated text:\n"
      << result->whymlText;
}

// --- Merged from regression: [F5] if-without-else ---
TEST(WhyMLEmitterTest, IfWithoutElseEmitsElseClause) {
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

  // WhyML is expression-based; every if must have an else clause
  auto thenPos = result->whymlText.find("then");
  if (thenPos != std::string::npos) {
    auto elsePos = result->whymlText.find("else", thenPos);
    EXPECT_NE(elsePos, std::string::npos)
        << "WhyML output has 'if ... then' without matching 'else', which is "
           "invalid WhyML syntax.  Generated text:\n"
        << result->whymlText;
  }
}

// --- G2: Multiplication overflow assertions ---
TEST(WhyMLEmitterTest, EmitsMultiplicationOverflowAssertion) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t mul(int32_t a, int32_t b) { return a * b; }
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

  // Multiplication should have INT32 overflow bounds assertion
  EXPECT_NE(result->whymlText.find("assert"), std::string::npos)
      << "WhyML output missing overflow assertion for multiplication.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("-2147483648"), std::string::npos)
      << "WhyML output missing INT32_MIN bound.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("2147483647"), std::string::npos)
      << "WhyML output missing INT32_MAX bound.  Text:\n"
      << result->whymlText;
}

// --- G4: AssignOp emission (variable reassignment) ---
TEST(WhyMLEmitterTest, EmitsAssignOpAsLetRebinding) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t reassign(int32_t a) {
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

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  // Variable reassignment should produce a let re-binding in WhyML
  // There should be at least two "let x =" bindings
  auto firstLet = result->whymlText.find("let x =");
  ASSERT_NE(firstLet, std::string::npos)
      << "WhyML output missing first 'let x =' binding.  Text:\n"
      << result->whymlText;
  auto secondLet = result->whymlText.find("let x =", firstLet + 1);
  EXPECT_NE(secondLet, std::string::npos)
      << "WhyML output missing second 'let x =' re-binding for assignment.  "
         "Text:\n"
      << result->whymlText;
}

// ============================================================
// Slice 2: Type-aware bounds, overflow modes, cast emission
// ============================================================

/// Test fixture for building Arc IR directly via OpBuilder.
class WhyMLEmitterSlice2Test : public ::testing::Test {
protected:
  void SetUp() override {
    ctx_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&ctx_);
  }

  /// Build a module with a single function containing one arithmetic op
  /// on the given type with the given overflow mode.
  /// Returns the emitted WhyML text.
  std::optional<WhyMLResult> buildAndEmitArithFunc(
      arc::IntType type, const std::string& overflowMode,
      const std::string& funcName,
      std::function<arc::AddOp(mlir::OpBuilder&, mlir::Location, arc::IntType,
                               mlir::Value, mlir::Value)>
          createOp = nullptr) {
    auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
    builder_->setInsertionPointToEnd(module.getBody());

    auto funcType = builder_->getFunctionType({type, type}, {type});
    auto funcOp = builder_->create<arc::FuncOp>(
        builder_->getUnknownLoc(), funcName, funcType,
        /*requires_attr=*/mlir::StringAttr{},
        /*ensures_attr=*/mlir::StringAttr{});

    // Set param_names
    funcOp->setAttr("param_names",
                    builder_->getArrayAttr({builder_->getStringAttr("a"),
                                            builder_->getStringAttr("b")}));

    auto& entryBlock = funcOp.getBody().emplaceBlock();
    entryBlock.addArgument(type, builder_->getUnknownLoc());
    entryBlock.addArgument(type, builder_->getUnknownLoc());

    builder_->setInsertionPointToEnd(&entryBlock);
    auto addOp = builder_->create<arc::AddOp>(builder_->getUnknownLoc(), type,
                                              entryBlock.getArgument(0),
                                              entryBlock.getArgument(1));

    if (!overflowMode.empty()) {
      addOp->setAttr("overflow", builder_->getStringAttr(overflowMode));
    }

    builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                    addOp.getResult());

    auto result = emitWhyML(module);
    module->destroy();
    return result;
  }

  /// Build a module with a single function containing a CastOp.
  std::optional<WhyMLResult> buildAndEmitCastFunc(arc::IntType srcType,
                                                  arc::IntType dstType,
                                                  const std::string& funcName) {
    auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
    builder_->setInsertionPointToEnd(module.getBody());

    auto funcType = builder_->getFunctionType({srcType}, {dstType});
    auto funcOp = builder_->create<arc::FuncOp>(
        builder_->getUnknownLoc(), funcName, funcType,
        /*requires_attr=*/mlir::StringAttr{},
        /*ensures_attr=*/mlir::StringAttr{});

    funcOp->setAttr("param_names",
                    builder_->getArrayAttr({builder_->getStringAttr("x")}));

    auto& entryBlock = funcOp.getBody().emplaceBlock();
    entryBlock.addArgument(srcType, builder_->getUnknownLoc());

    builder_->setInsertionPointToEnd(&entryBlock);
    auto castOp = builder_->create<arc::CastOp>(
        builder_->getUnknownLoc(), dstType, entryBlock.getArgument(0));
    builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                    castOp.getResult());

    auto result = emitWhyML(module);
    module->destroy();
    return result;
  }

  mlir::MLIRContext ctx_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

// [S2] i8 bounds in overflow assertion output
TEST_F(WhyMLEmitterSlice2Test, EmitsI8BoundsInTrapMode) {
  auto i8Type = arc::IntType::get(&ctx_, 8, true);
  auto result = buildAndEmitArithFunc(i8Type, "", "add_i8");
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("-128"), std::string::npos)
      << "Missing i8 min bound.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("127"), std::string::npos)
      << "Missing i8 max bound.  Text:\n"
      << result->whymlText;
}

// [S2] u8 wrap mode uses mod 256
TEST_F(WhyMLEmitterSlice2Test, EmitsU8WrapWithMod256) {
  auto u8Type = arc::IntType::get(&ctx_, 8, false);
  auto result = buildAndEmitArithFunc(u8Type, "wrap", "add_u8");
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("mod"), std::string::npos)
      << "Missing mod for wrap mode.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("256"), std::string::npos)
      << "Missing 2^8=256 for u8 wrap.  Text:\n"
      << result->whymlText;
  // Should NOT have an assert (wrap mode doesn't trap)
  EXPECT_EQ(result->whymlText.find("assert"), std::string::npos)
      << "Wrap mode should not emit assert.  Text:\n"
      << result->whymlText;
}

// [S2] Signed wrap mode uses mod with offset
TEST_F(WhyMLEmitterSlice2Test, EmitsSignedWrapWithModAndOffset) {
  auto i8Type = arc::IntType::get(&ctx_, 8, true);
  auto result = buildAndEmitArithFunc(i8Type, "wrap", "add_i8_wrap");
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("mod"), std::string::npos)
      << "Missing mod for signed wrap.  Text:\n"
      << result->whymlText;
  // Should contain 128 (half power) and 256 (full power) for i8
  EXPECT_NE(result->whymlText.find("128"), std::string::npos)
      << "Missing 2^7=128 offset for signed i8 wrap.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("256"), std::string::npos)
      << "Missing 2^8=256 modulus for signed i8 wrap.  Text:\n"
      << result->whymlText;
}

// [S2] Saturate mode emits clamping expression
TEST_F(WhyMLEmitterSlice2Test, EmitsSaturateClamp) {
  auto i8Type = arc::IntType::get(&ctx_, 8, true);
  auto result = buildAndEmitArithFunc(i8Type, "saturate", "add_i8_sat");
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("-128"), std::string::npos)
      << "Missing min clamp bound.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("127"), std::string::npos)
      << "Missing max clamp bound.  Text:\n"
      << result->whymlText;
  // Saturate mode uses if-then-else, not assert
  EXPECT_EQ(result->whymlText.find("assert"), std::string::npos)
      << "Saturate mode should not emit assert for the add.  Text:\n"
      << result->whymlText;
}

// [S2] ComputerDivision import for wrap mode
TEST_F(WhyMLEmitterSlice2Test, WrapModeImportsComputerDivision) {
  auto u8Type = arc::IntType::get(&ctx_, 8, false);
  auto result = buildAndEmitArithFunc(u8Type, "wrap", "add_wrap");
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("use int.ComputerDivision"),
            std::string::npos)
      << "Wrap mode needs ComputerDivision import for mod.  Text:\n"
      << result->whymlText;
}

// [S2] Widening cast (i8 -> i32) produces no assert
TEST_F(WhyMLEmitterSlice2Test, WideningCastNoAssert) {
  auto i8Type = arc::IntType::get(&ctx_, 8, true);
  auto i32Type = arc::IntType::get(&ctx_, 32, true);
  auto result = buildAndEmitCastFunc(i8Type, i32Type, "widen_i8_to_i32");
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result->whymlText.find("assert"), std::string::npos)
      << "Widening cast should not emit assert.  Text:\n"
      << result->whymlText;
}

// [S2] Narrowing cast (i32 -> i8) in trap mode asserts target range
TEST_F(WhyMLEmitterSlice2Test, NarrowingCastAssertsRange) {
  auto i32Type = arc::IntType::get(&ctx_, 32, true);
  auto i8Type = arc::IntType::get(&ctx_, 8, true);
  auto result = buildAndEmitCastFunc(i32Type, i8Type, "narrow_i32_to_i8");
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("assert"), std::string::npos)
      << "Narrowing cast should emit assert.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("-128"), std::string::npos)
      << "Should assert i8 min bound.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("127"), std::string::npos)
      << "Should assert i8 max bound.  Text:\n"
      << result->whymlText;
}

// [S2] Sign-change cast (i32 -> u32) in trap mode asserts target range
TEST_F(WhyMLEmitterSlice2Test, SignChangeCastAssertsRange) {
  auto i32Type = arc::IntType::get(&ctx_, 32, true);
  auto u32Type = arc::IntType::get(&ctx_, 32, false);
  auto result = buildAndEmitCastFunc(i32Type, u32Type, "signchange_i32_to_u32");
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("assert"), std::string::npos)
      << "Sign-change cast should emit assert.  Text:\n"
      << result->whymlText;
  // u32 range: 0 to 4294967295
  EXPECT_NE(result->whymlText.find("0 <="), std::string::npos)
      << "Should assert u32 min bound (0).  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("4294967295"), std::string::npos)
      << "Should assert u32 max bound.  Text:\n"
      << result->whymlText;
}

} // namespace
} // namespace arcanum
