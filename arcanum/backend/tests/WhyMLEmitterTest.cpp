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
  std::string contractLine;  // The //@ line
  std::string expectedStr;   // Substring expected in WhyML output
  std::string absentStr;     // Substring that must NOT appear (empty = skip)
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
          << "WhyML output still contains '" << param.absentStr
          << "'.  Text:\n"
          << result->whymlText;
    } else {
      EXPECT_EQ(result->whymlText.find(param.absentStr), std::string::npos)
          << "WhyML output still contains '" << param.absentStr
          << "'.  Text:\n"
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
    ::testing::Values(
        ModuleNameParam{"SnakeCase", "my_func_name", "MyFuncName"},
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
    ::testing::Values(
        ComputerDivisionImportParam{
            "Division",
            R"(
    #include <cstdint>
    int32_t mydiv(int32_t a, int32_t b) { return a / b; }
  )"},
        ComputerDivisionImportParam{
            "Modulo",
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

} // namespace
} // namespace arcanum
