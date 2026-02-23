/// Bug reproduction tests for Lowering.cpp findings.
/// These tests demonstrate bugs found by the bug hunter agent.
/// Each test targets exactly one finding and should FAIL before the bug is fixed.

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
class BugHunterLoweringTestFixture : public ::testing::Test {
protected:
  void SetUp() override { DiagnosticTracker::reset(); }
  void TearDown() override { DiagnosticTracker::reset(); }
};

/// [F1] Null pointer dereference when lowering void function with explicit
/// return; statement.
///
/// SubsetEnforcer accepts void functions.  When a void function has an
/// explicit "return;" statement, ReturnStmt::getRetValue() returns nullptr.
/// lowerExpr() is called with this nullptr and immediately dereferences it
/// via expr->IgnoreParenImpCasts(), causing a crash.
///
/// This test should crash (SEGFAULT / abort) without the fix.  With the
/// fix, it should produce a valid module (no crash).
TEST_F(BugHunterLoweringTestFixture, F1_VoidReturnCausesNullDeref) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    void doNothing() { return; }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;

  // This call should NOT crash.  Before the fix, it dereferences a null
  // pointer inside lowerExpr() when processing the void "return;" statement.
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);

  // If we reach here, the crash did not occur.
  EXPECT_TRUE(module);
}

} // namespace
} // namespace arcanum
