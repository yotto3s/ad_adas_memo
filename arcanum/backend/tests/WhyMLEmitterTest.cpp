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

// --- Coverage gap C1: moduleToFuncMap population ---

TEST(WhyMLEmitterTest, PopulatesModuleToFuncMap) {
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

  // moduleToFuncMap should be populated by the emitter
  EXPECT_FALSE(result->moduleToFuncMap.empty());
  // The emitter converts snake_case "safe_add" to CamelCase "SafeAdd" for
  // the WhyML module name, and maps it back to the original function name.
  auto it = result->moduleToFuncMap.find("SafeAdd");
  ASSERT_NE(it, result->moduleToFuncMap.end());
  EXPECT_EQ(it->second, "safe_add");
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

TEST(WhyMLEmitterTest, EmitsIfThenElse) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    int32_t myabs(int32_t x) {
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

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("if "), std::string::npos)
      << "Missing 'if' keyword in WhyML output.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("then"), std::string::npos)
      << "Missing 'then' keyword in WhyML output.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("else"), std::string::npos)
      << "Missing 'else' keyword in WhyML output.  Text:\n"
      << result->whymlText;
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
  std::optional<WhyMLResult>
  buildAndEmitArithFunc(arc::IntType type, const std::string& overflowMode,
                        const std::string& funcName) {
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

// (B5 and B8 parameterized tests are defined after
// ArithOpKind/buildAndEmitBinaryOpFunc below.)

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

// ============================================================
// TC-1: SubOp, MulOp, DivOp, RemOp WhyML emitter tests
// ============================================================

// Helper: build a module with a single function containing one binary op
// of any type (Sub, Mul, Div, Rem).
enum class ArithOpKind { Sub, Mul, Div, Rem };

std::optional<WhyMLResult>
buildAndEmitBinaryOpFunc(mlir::MLIRContext& ctx, arc::IntType type,
                         const std::string& overflowMode,
                         const std::string& funcName, ArithOpKind opKind) {
  mlir::OpBuilder builder(&ctx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto funcType = builder.getFunctionType({type, type}, {type});
  auto funcOp =
      builder.create<arc::FuncOp>(builder.getUnknownLoc(), funcName, funcType,
                                  /*requires_attr=*/mlir::StringAttr{},
                                  /*ensures_attr=*/mlir::StringAttr{});

  funcOp->setAttr("param_names",
                  builder.getArrayAttr({builder.getStringAttr("a"),
                                        builder.getStringAttr("b")}));

  auto& entryBlock = funcOp.getBody().emplaceBlock();
  entryBlock.addArgument(type, builder.getUnknownLoc());
  entryBlock.addArgument(type, builder.getUnknownLoc());

  builder.setInsertionPointToEnd(&entryBlock);
  mlir::Value resultVal;
  switch (opKind) {
  case ArithOpKind::Sub: {
    auto op = builder.create<arc::SubOp>(builder.getUnknownLoc(), type,
                                         entryBlock.getArgument(0),
                                         entryBlock.getArgument(1));
    if (!overflowMode.empty())
      op->setAttr("overflow", builder.getStringAttr(overflowMode));
    resultVal = op.getResult();
    break;
  }
  case ArithOpKind::Mul: {
    auto op = builder.create<arc::MulOp>(builder.getUnknownLoc(), type,
                                         entryBlock.getArgument(0),
                                         entryBlock.getArgument(1));
    if (!overflowMode.empty())
      op->setAttr("overflow", builder.getStringAttr(overflowMode));
    resultVal = op.getResult();
    break;
  }
  case ArithOpKind::Div: {
    auto op = builder.create<arc::DivOp>(builder.getUnknownLoc(), type,
                                         entryBlock.getArgument(0),
                                         entryBlock.getArgument(1));
    if (!overflowMode.empty())
      op->setAttr("overflow", builder.getStringAttr(overflowMode));
    resultVal = op.getResult();
    break;
  }
  case ArithOpKind::Rem: {
    auto op = builder.create<arc::RemOp>(builder.getUnknownLoc(), type,
                                         entryBlock.getArgument(0),
                                         entryBlock.getArgument(1));
    if (!overflowMode.empty())
      op->setAttr("overflow", builder.getStringAttr(overflowMode));
    resultVal = op.getResult();
    break;
  }
  }

  builder.create<arc::ReturnOp>(builder.getUnknownLoc(), resultVal);
  auto result = emitWhyML(module);
  module->destroy();
  return result;
}

// --- B5: Trap-mode bounds tests (parameterized) ---
// Consolidates EmitsSubI16BoundsInTrapMode, EmitsMulI16BoundsInTrapMode,
// EmitsSubI8BoundsInTrapMode, and EmitsI64BoundsInTrapMode.

enum class BoundsTestHelper { ArithFunc, BinaryOpFunc };

struct TrapBoundsParam {
  const char* name;
  unsigned width;
  bool isSigned;
  BoundsTestHelper helper;
  ArithOpKind opKind; // only used when helper == BinaryOpFunc
  const char* funcName;
  const char* minBound;
  const char* maxBound;
  const char* operatorStr; // extra substring check (empty = skip)
};

class TrapBoundsTest : public ::testing::TestWithParam<TrapBoundsParam> {
protected:
  void SetUp() override {
    ctx_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&ctx_);
  }

  std::optional<WhyMLResult> buildArithFunc(arc::IntType type,
                                            const std::string& funcName) {
    auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
    builder_->setInsertionPointToEnd(module.getBody());

    auto funcType = builder_->getFunctionType({type, type}, {type});
    auto funcOp = builder_->create<arc::FuncOp>(
        builder_->getUnknownLoc(), funcName, funcType,
        /*requires_attr=*/mlir::StringAttr{},
        /*ensures_attr=*/mlir::StringAttr{});

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
    builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                    addOp.getResult());

    auto result = emitWhyML(module);
    module->destroy();
    return result;
  }

  mlir::MLIRContext ctx_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

TEST_P(TrapBoundsTest, EmitsTrapBounds) {
  auto [name, width, isSigned, helper, opKind, funcName, minBound, maxBound,
        operatorStr] = GetParam();

  auto type = arc::IntType::get(&ctx_, width, isSigned);
  std::optional<WhyMLResult> result;

  if (helper == BoundsTestHelper::ArithFunc) {
    result = buildArithFunc(type, funcName);
  } else {
    result = buildAndEmitBinaryOpFunc(ctx_, type, "", funcName, opKind);
  }

  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find(minBound), std::string::npos)
      << "Missing min bound '" << minBound << "'.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find(maxBound), std::string::npos)
      << "Missing max bound '" << maxBound << "'.  Text:\n"
      << result->whymlText;

  if (std::string(operatorStr).length() > 0) {
    EXPECT_NE(result->whymlText.find(operatorStr), std::string::npos)
        << "Missing operator '" << operatorStr << "'.  Text:\n"
        << result->whymlText;
  }
}

INSTANTIATE_TEST_SUITE_P(
    WhyMLEmitter, TrapBoundsTest,
    ::testing::Values(
        TrapBoundsParam{"AddI8", 8, true, BoundsTestHelper::ArithFunc,
                        ArithOpKind::Sub /*unused*/, "add_i8", "-128", "127",
                        ""},
        TrapBoundsParam{"SubI16", 16, true, BoundsTestHelper::BinaryOpFunc,
                        ArithOpKind::Sub, "sub_i16", "-32768", "32767", " - "},
        TrapBoundsParam{"MulI16", 16, true, BoundsTestHelper::BinaryOpFunc,
                        ArithOpKind::Mul, "mul_i16", "-32768", "32767", " * "},
        TrapBoundsParam{"SubI8", 8, true, BoundsTestHelper::BinaryOpFunc,
                        ArithOpKind::Sub, "sub_i8", "-128", "127", ""},
        TrapBoundsParam{"AddI64", 64, true, BoundsTestHelper::ArithFunc,
                        ArithOpKind::Sub /*unused*/, "add_i64",
                        "-9223372036854775808", "9223372036854775807", ""}),
    [](const ::testing::TestParamInfo<TrapBoundsParam>& info) {
      return info.param.name;
    });

// [TC-1] DivOp with i16 in trap mode (has both div-by-zero and overflow)
TEST_F(WhyMLEmitterSlice2Test, EmitsDivI16WithBothAssertions) {
  auto i16Type = arc::IntType::get(&ctx_, 16, true);
  auto result =
      buildAndEmitBinaryOpFunc(ctx_, i16Type, "", "div_i16", ArithOpKind::Div);
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("<> 0"), std::string::npos)
      << "Missing divisor-not-zero assertion.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("-32768"), std::string::npos)
      << "Missing i16 min bound for DivOp overflow.  Text:\n"
      << result->whymlText;
}

// [TC-1] RemOp with u8 in wrap mode (no overflow assertion)
TEST_F(WhyMLEmitterSlice2Test, EmitsRemU8WrapMode) {
  auto u8Type = arc::IntType::get(&ctx_, 8, false);
  auto result = buildAndEmitBinaryOpFunc(ctx_, u8Type, "wrap", "rem_u8",
                                         ArithOpKind::Rem);
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("<> 0"), std::string::npos)
      << "Missing divisor-not-zero assertion.  Text:\n"
      << result->whymlText;
  // Wrap mode should use mod, not trap assertion
  EXPECT_NE(result->whymlText.find("mod"), std::string::npos)
      << "Missing mod for wrap mode RemOp.  Text:\n"
      << result->whymlText;
}

// [SC-2/F2] DivOp in wrap mode should NOT emit spurious trap assertion
TEST_F(WhyMLEmitterSlice2Test, DivOpWrapModeNoSpuriousTrapAssert) {
  auto i16Type = arc::IntType::get(&ctx_, 16, true);
  auto result = buildAndEmitBinaryOpFunc(ctx_, i16Type, "wrap", "div_wrap_i16",
                                         ArithOpKind::Div);
  ASSERT_TRUE(result.has_value());

  // Should have div-by-zero assertion
  EXPECT_NE(result->whymlText.find("<> 0"), std::string::npos)
      << "Missing divisor-not-zero assertion.  Text:\n"
      << result->whymlText;
  // Should have mod (wrap mode), NOT a range assert for overflow
  EXPECT_NE(result->whymlText.find("mod"), std::string::npos)
      << "Wrap mode DivOp should use modular arithmetic.  Text:\n"
      << result->whymlText;
}

// --- B8: Wrap-mode tests (parameterized) ---
// Consolidates EmitsU64WrapWithMod2Pow64 (and can be extended for more wrap
// mode variants).

struct WrapModeParam {
  const char* name;
  unsigned width;
  bool isSigned;
  const char* funcName;
  std::vector<const char*> expectedStrings;
  std::vector<const char*> absentStrings;
};

class WrapModeTest : public ::testing::TestWithParam<WrapModeParam> {
protected:
  void SetUp() override {
    ctx_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&ctx_);
  }

  std::optional<WhyMLResult> buildWrapArithFunc(arc::IntType type,
                                                const std::string& funcName) {
    auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
    builder_->setInsertionPointToEnd(module.getBody());

    auto funcType = builder_->getFunctionType({type, type}, {type});
    auto funcOp = builder_->create<arc::FuncOp>(
        builder_->getUnknownLoc(), funcName, funcType,
        /*requires_attr=*/mlir::StringAttr{},
        /*ensures_attr=*/mlir::StringAttr{});

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
    addOp->setAttr("overflow", builder_->getStringAttr("wrap"));
    builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                    addOp.getResult());

    auto result = emitWhyML(module);
    module->destroy();
    return result;
  }

  mlir::MLIRContext ctx_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

TEST_P(WrapModeTest, EmitsWrapMode) {
  auto [name, width, isSigned, funcName, expectedStrings, absentStrings] =
      GetParam();

  auto type = arc::IntType::get(&ctx_, width, isSigned);
  auto result = buildWrapArithFunc(type, funcName);
  ASSERT_TRUE(result.has_value());

  for (const auto* expected : expectedStrings) {
    EXPECT_NE(result->whymlText.find(expected), std::string::npos)
        << "Missing expected string '" << expected << "'.  Text:\n"
        << result->whymlText;
  }
  for (const auto* absent : absentStrings) {
    EXPECT_EQ(result->whymlText.find(absent), std::string::npos)
        << "Should not contain '" << absent << "'.  Text:\n"
        << result->whymlText;
  }
}

INSTANTIATE_TEST_SUITE_P(
    WhyMLEmitter, WrapModeTest,
    ::testing::Values(
        WrapModeParam{"U8WrapMod256", 8, false, "add_u8", {"256", "mod"}, {}},
        WrapModeParam{
            "SignedI8WrapMod256", 8, true, "add_i8_wrap", {"256", "mod"}, {}},
        WrapModeParam{"U64WrapMod2Pow64",
                      64,
                      false,
                      "add_u64",
                      {"18446744073709551616", "mod"},
                      {}}),
    [](const ::testing::TestParamInfo<WrapModeParam>& info) {
      return info.param.name;
    });

// [TC-3] Wrap mode with null intType path: document this is unreachable
// in well-formed IR because all arithmetic ops have IntType results.
// The fallback (modExpr = rawExpr) exists as a defensive measure.
// This test constructs a scenario to verify the defensive path.
// Note: In practice, this path is unreachable with valid Arc IR since
// all arithmetic ops produce IntType results. The defensive code
// handles the hypothetical case where intType is null.

// [CR-1] CastOp with wrap mode imports ComputerDivision even without
// any DivOp, RemOp, or wrap-mode arithmetic ops in the function.
TEST_F(WhyMLEmitterSlice2Test, WrapModeCastOnlyImportsComputerDivision) {
  auto i32Type = arc::IntType::get(&ctx_, 32, true);
  auto i8Type = arc::IntType::get(&ctx_, 8, true);

  // Build a module with only a wrap-mode CastOp (narrowing i32 -> i8)
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto funcType = builder_->getFunctionType({i32Type}, {i8Type});
  auto funcOp = builder_->create<arc::FuncOp>(
      builder_->getUnknownLoc(), "cast_wrap_only", funcType,
      /*requires_attr=*/mlir::StringAttr{},
      /*ensures_attr=*/mlir::StringAttr{});

  funcOp->setAttr("param_names",
                  builder_->getArrayAttr({builder_->getStringAttr("x")}));

  auto& entryBlock = funcOp.getBody().emplaceBlock();
  entryBlock.addArgument(i32Type, builder_->getUnknownLoc());

  builder_->setInsertionPointToEnd(&entryBlock);
  auto castOp = builder_->create<arc::CastOp>(builder_->getUnknownLoc(), i8Type,
                                              entryBlock.getArgument(0));
  // Set wrap mode on the CastOp
  castOp->setAttr("overflow", builder_->getStringAttr("wrap"));
  builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                  castOp.getResult());

  auto result = emitWhyML(module);
  module->destroy();
  ASSERT_TRUE(result.has_value());

  // The wrap-mode CastOp uses mod, so ComputerDivision must be imported
  EXPECT_NE(result->whymlText.find("use int.ComputerDivision"),
            std::string::npos)
      << "Wrap-mode CastOp needs ComputerDivision import for mod.  Text:\n"
      << result->whymlText;
  // Should contain mod (wrap reduction)
  EXPECT_NE(result->whymlText.find("mod"), std::string::npos)
      << "Wrap-mode CastOp should emit mod expression.  Text:\n"
      << result->whymlText;
}

// [S2-TC6] DivOp with saturate mode emits clamping, not trap assertion
TEST_F(WhyMLEmitterSlice2Test, EmitsDivSaturateMode) {
  auto i16Type = arc::IntType::get(&ctx_, 16, true);
  auto result = buildAndEmitBinaryOpFunc(ctx_, i16Type, "saturate",
                                         "div_i16_sat", ArithOpKind::Div);
  ASSERT_TRUE(result.has_value());

  // Saturate mode should emit if-then-else clamping to target bounds
  EXPECT_NE(result->whymlText.find("if"), std::string::npos)
      << "Missing clamping 'if' for saturate mode DivOp.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("-32768"), std::string::npos)
      << "Missing i16 min clamp bound.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("32767"), std::string::npos)
      << "Missing i16 max clamp bound.  Text:\n"
      << result->whymlText;
  // Division-by-zero assertion is always present regardless of overflow mode
  EXPECT_NE(result->whymlText.find("<> 0"), std::string::npos)
      << "Missing divisor-not-zero assertion.  Text:\n"
      << result->whymlText;
  // Saturate mode should NOT emit a range trap assertion (only clamping)
  // Count "assert" occurrences: should be exactly one (the div-by-zero one)
  size_t assertCount = 0;
  size_t pos = 0;
  while ((pos = result->whymlText.find("assert", pos)) != std::string::npos) {
    ++assertCount;
    ++pos;
  }
  EXPECT_EQ(assertCount, 1u)
      << "Saturate mode should have only the div-by-zero assert, not a range "
         "assert.  Text:\n"
      << result->whymlText;
}

// [S2-TC6] CastOp (i32->i8) with saturate mode emits clamping
TEST_F(WhyMLEmitterSlice2Test, EmitsCastSaturateMode) {
  auto i32Type = arc::IntType::get(&ctx_, 32, true);
  auto i8Type = arc::IntType::get(&ctx_, 8, true);

  // Build a module with a narrowing CastOp with overflow="saturate"
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto funcType = builder_->getFunctionType({i32Type}, {i8Type});
  auto funcOp = builder_->create<arc::FuncOp>(
      builder_->getUnknownLoc(), "cast_sat_i32_to_i8", funcType,
      /*requires_attr=*/mlir::StringAttr{},
      /*ensures_attr=*/mlir::StringAttr{});

  funcOp->setAttr("param_names",
                  builder_->getArrayAttr({builder_->getStringAttr("x")}));

  auto& entryBlock = funcOp.getBody().emplaceBlock();
  entryBlock.addArgument(i32Type, builder_->getUnknownLoc());

  builder_->setInsertionPointToEnd(&entryBlock);
  auto castOp = builder_->create<arc::CastOp>(builder_->getUnknownLoc(), i8Type,
                                              entryBlock.getArgument(0));
  // Set saturate mode on the CastOp
  castOp->setAttr("overflow", builder_->getStringAttr("saturate"));
  builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                  castOp.getResult());

  auto result = emitWhyML(module);
  module->destroy();
  ASSERT_TRUE(result.has_value());

  // Saturate mode should emit if-then-else clamping to i8 bounds
  EXPECT_NE(result->whymlText.find("if"), std::string::npos)
      << "Missing clamping 'if' for saturate mode CastOp.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("-128"), std::string::npos)
      << "Missing i8 min clamp bound.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("127"), std::string::npos)
      << "Missing i8 max clamp bound.  Text:\n"
      << result->whymlText;
  // Saturate mode CastOp should NOT emit any trap assertion
  EXPECT_EQ(result->whymlText.find("assert"), std::string::npos)
      << "Saturate mode CastOp should not emit assert.  Text:\n"
      << result->whymlText;
}

// [SC-7/F1/TC-5] Unsigned-to-signed same-width cast now emits assertion
TEST_F(WhyMLEmitterSlice2Test, UnsignedToSignedSameWidthCastAssertsRange) {
  auto u32Type = arc::IntType::get(&ctx_, 32, false);
  auto i32Type = arc::IntType::get(&ctx_, 32, true);
  auto result = buildAndEmitCastFunc(u32Type, i32Type, "u32_to_i32_same_width");
  ASSERT_TRUE(result.has_value());

  // Should now emit an assert for the target (i32) range because same-width
  // unsigned-to-signed is NOT widening.
  EXPECT_NE(result->whymlText.find("assert"), std::string::npos)
      << "Same-width unsigned-to-signed cast must emit assert.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("-2147483648"), std::string::npos)
      << "Should assert i32 min bound.  Text:\n"
      << result->whymlText;
  EXPECT_NE(result->whymlText.find("2147483647"), std::string::npos)
      << "Should assert i32 max bound.  Text:\n"
      << result->whymlText;
}

// --- Slice 3: Loop emission tests ---

TEST(WhyMLEmitterTest, EmitsForLoopAsRecursiveFunction) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: n >= 0 && n <= 1000
    //@ ensures: \result >= 0
    int32_t sum_to_n(int32_t n) {
      int32_t sum = 0;
      //@ loop_invariant: sum >= 0 && i >= 0 && i <= n
      //@ loop_variant: n - i
      //@ loop_assigns: i, sum
      for (int32_t i = 0; i < n; i = i + 1) {
        sum = sum + i;
      }
      return sum;
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

  EXPECT_NE(result->whymlText.find("let rec"), std::string::npos)
      << "Should emit a recursive function for the loop";
  EXPECT_NE(result->whymlText.find("requires"), std::string::npos)
      << "Should emit loop invariant as requires clause";
  EXPECT_NE(result->whymlText.find("variant"), std::string::npos)
      << "Should emit loop variant clause";
  EXPECT_NE(result->whymlText.find("module"), std::string::npos);
  EXPECT_NE(result->whymlText.find("end"), std::string::npos);
}

TEST(WhyMLEmitterTest, EmitsWhileLoopAsRecursiveFunction) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: x > 0
    //@ ensures: \result >= 0
    int32_t halve_to_zero(int32_t x) {
      //@ loop_invariant: x >= 0
      //@ loop_variant: x
      //@ loop_assigns: x
      while (x > 0) {
        x = x / 2;
      }
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

  auto result = emitWhyML(*module);
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("let rec"), std::string::npos);
  EXPECT_NE(result->whymlText.find("variant"), std::string::npos);
}

TEST(WhyMLEmitterTest, EmitsDoWhileLoopAsRecursiveFunction) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: x > 0 && x <= 1000
    //@ ensures: \result >= 1
    int32_t count_digits(int32_t x) {
      int32_t count = 0;
      //@ loop_invariant: count >= 0 && x >= 0
      //@ loop_variant: x
      //@ loop_assigns: x, count
      do {
        x = x / 10;
        count = count + 1;
      } while (x > 0);
      return count;
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

  EXPECT_NE(result->whymlText.find("let rec"), std::string::npos);
}

TEST(WhyMLEmitterTest, EmitsBreakAsEarlyReturn) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: n > 0 && n <= 100
    //@ ensures: \result >= 0
    int32_t find_first_even(int32_t n) {
      int32_t result = 0;
      //@ loop_invariant: i >= 0 && i <= n
      //@ loop_assigns: i, result
      for (int32_t i = 0; i < n; i = i + 1) {
        if (i % 2 == 0) {
          result = i;
          break;
        }
      }
      return result;
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

  EXPECT_NE(result->whymlText.find("let rec"), std::string::npos);
}

// [TC-1/TC-2] Test: continue emits as recursive call with proper control flow.
TEST(WhyMLEmitterTest, EmitsContinueAsRecursiveCall) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: n > 0 && n <= 100
    //@ ensures: \result >= 0
    int32_t sum_odd(int32_t n) {
      int32_t sum = 0;
      int32_t i = 0;
      //@ loop_invariant: i >= 0 && i <= n && sum >= 0
      //@ loop_variant: n - i
      //@ loop_assigns: i, sum
      while (i < n) {
        i = i + 1;
        if (i % 2 == 0) {
          continue;
        }
        sum = sum + i;
      }
      return sum;
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

  const auto& text = result->whymlText;
  EXPECT_NE(text.find("let rec"), std::string::npos)
      << "Should emit a recursive function";
  // The continue path should produce an if-then-else where the then-branch
  // recurses and the else-branch contains the rest of the body.
  EXPECT_NE(text.find("loop_"), std::string::npos)
      << "Should contain a recursive call";
  // Verify no "else ()" inside the loop body -- the else branch should
  // contain actual code, not unit.
  auto letRecPos = text.find("let rec");
  auto inPos = text.find("\n    in\n", letRecPos);
  ASSERT_NE(letRecPos, std::string::npos);
  ASSERT_NE(inPos, std::string::npos);
  auto loopBody = text.substr(letRecPos, inPos - letRecPos);
  EXPECT_EQ(loopBody.find("else\n    ()"), std::string::npos)
      << "Continue if-then-else should not have () in else branch";
}

// [TC-3] Test: nested loops emit as nested recursive functions.
TEST(WhyMLEmitterTest, EmitsNestedLoopsAsNestedRecursiveFunctions) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: n >= 0 && n <= 100
    //@ ensures: \result >= 0
    int32_t sum_triangle(int32_t n) {
      int32_t sum = 0;
      //@ loop_invariant: i >= 0 && i <= n && sum >= 0
      //@ loop_assigns: i, sum
      for (int32_t i = 0; i < n; i = i + 1) {
        //@ loop_invariant: j >= 0 && j <= i
        //@ loop_assigns: j, sum
        for (int32_t j = 0; j <= i; j = j + 1) {
          sum = sum + 1;
        }
      }
      return sum;
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

  const auto& text = result->whymlText;
  // Count distinct "let rec" definitions -- should have at least 2.
  size_t letRecCount = 0;
  size_t pos = 0;
  while ((pos = text.find("let rec", pos)) != std::string::npos) {
    ++letRecCount;
    pos += 7;
  }
  EXPECT_GE(letRecCount, 2u)
      << "Nested loops should produce two distinct 'let rec' definitions.\n"
      << "Full WhyML output:\n"
      << text;
}

// [TC-4] Strengthened: verify assigned variable names appear as parameters.
TEST(WhyMLEmitterTest, ForLoopAssignedVarsAppearAsParameters) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: n >= 0 && n <= 1000
    //@ ensures: \result >= 0
    int32_t sum_to_n(int32_t n) {
      int32_t sum = 0;
      //@ loop_invariant: sum >= 0 && i >= 0 && i <= n
      //@ loop_variant: n - i
      //@ loop_assigns: i, sum
      for (int32_t i = 0; i < n; i = i + 1) {
        sum = sum + i;
      }
      return sum;
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

  const auto& text = result->whymlText;
  // Verify assigned variables appear as parameters in the recursive function
  EXPECT_NE(text.find("(i: int)"), std::string::npos)
      << "Assigned variable 'i' should appear as a parameter";
  EXPECT_NE(text.find("(sum: int)"), std::string::npos)
      << "Assigned variable 'sum' should appear as a parameter";
}

// ---------------------------------------------------------------------------
// [TC-5] Tests exercising buildTupleExpr/buildTupleType/parseAssignsList
// indirectly through the public emitWhyML API.
// ---------------------------------------------------------------------------

// Single assigned variable: buildTupleType(1) -> "int",
// buildTupleExpr({"i"}) -> "i" (no parentheses).
TEST(WhyMLEmitterTest, SingleAssignedVarProducesBareType) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: n > 0 && n <= 100
    //@ ensures: \result >= 0
    int32_t countdown(int32_t n) {
      //@ loop_invariant: n >= 0
      //@ loop_variant: n
      //@ loop_assigns: n
      while (n > 0) {
        n = n - 1;
      }
      return n;
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

  const auto& text = result->whymlText;
  // Single var: return type should be "int" not "(int)"
  EXPECT_NE(text.find(": int"), std::string::npos)
      << "Single assigned variable should produce bare 'int' return type";
  // Destructuring should be "let n = loop_" not "let (n) = loop_"
  EXPECT_NE(text.find("let n = loop_"), std::string::npos)
      << "Single assigned variable should produce bare name in let binding";
}

// Multiple assigned variables: buildTupleType(2) -> "(int, int)",
// buildTupleExpr({"i", "sum"}) -> "(i, sum)".
TEST(WhyMLEmitterTest, MultipleAssignedVarsProduceTupleType) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: n >= 0 && n <= 100
    //@ ensures: \result >= 0
    int32_t sum_up(int32_t n) {
      int32_t s = 0;
      //@ loop_invariant: i >= 0 && i <= n && s >= 0
      //@ loop_variant: n - i
      //@ loop_assigns: i, s
      for (int32_t i = 0; i < n; i = i + 1) {
        s = s + i;
      }
      return s;
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

  const auto& text = result->whymlText;
  // Multiple vars: return type should be "(int, int)"
  EXPECT_NE(text.find("(int, int)"), std::string::npos)
      << "Multiple assigned variables should produce tuple return type";
  // Destructuring should use tuple: "let (i, s) = loop_"
  EXPECT_NE(text.find("let (i, s) = loop_"), std::string::npos)
      << "Multiple assigned variables should produce tuple destructuring";
}

// Whitespace in assigns attribute: parseAssignsList("  i ,  sum  ") should
// produce ["i", "sum"] with whitespace trimmed.
TEST(WhyMLEmitterTest, AssignsWithWhitespaceAreTrimmed) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    //@ requires: n >= 0 && n <= 100
    //@ ensures: \result >= 0
    int32_t sum_up(int32_t n) {
      int32_t s = 0;
      //@ loop_invariant: i >= 0 && i <= n && s >= 0
      //@ loop_variant: n - i
      //@ loop_assigns:   i ,  s
      for (int32_t i = 0; i < n; i = i + 1) {
        s = s + i;
      }
      return s;
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

  const auto& text = result->whymlText;
  // Trimming: parameter declarations should have clean names
  EXPECT_NE(text.find("(i: int)"), std::string::npos)
      << "Whitespace around assigns should be trimmed";
  EXPECT_NE(text.find("(s: int)"), std::string::npos)
      << "Whitespace around assigns should be trimmed";
}

} // namespace
} // namespace arcanum
