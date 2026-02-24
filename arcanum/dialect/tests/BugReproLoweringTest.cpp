/// Regression tests for Lowering bugs found during Slice 2 review.
/// Originally these tests demonstrated the bugs; now they verify the fixes.

#include "arcanum/DiagnosticTracker.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"
#include "arcanum/dialect/Lowering.h"
#include "arcanum/frontend/ContractParser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>
#include <map>

namespace arcanum {
namespace {

class BugReproLoweringTest : public ::testing::Test {
protected:
  void SetUp() override { DiagnosticTracker::reset(); }
};

// [F4] Bug: getSExtValue() on unsigned 64-bit literals > INT64_MAX
// produces incorrect negative values.
//
// When lowering a uint64_t literal like 18446744073709551615ULL (UINT64_MAX),
// intLit->getValue().getSExtValue() returns -1 (sign-extends the bit pattern).
// This -1 is then stored in the IntegerAttr, producing an incorrect constant.
// The WhyML emitter then calls getInt() which also returns -1, and
// std::to_string(-1) produces "-1" instead of "18446744073709551615".
TEST_F(BugReproLoweringTest, F4_U64MaxLiteralLoweredCorrectly) {
  // Use the literal 18446744073709551615ULL directly.
  // Clang parses this as a uint64_t integer literal with value UINT64_MAX.
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    uint64_t get_max() {
      return 18446744073709551615ULL;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundConst = false;
  module->walk([&](arc::ConstantOp constOp) {
    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
    if (intAttr) {
      // The APInt stores all-ones (correct bit pattern).
      llvm::APInt val = intAttr.getValue();
      EXPECT_TRUE(val.isAllOnes()) << "Expected UINT64_MAX (all 64 bits set)";

      // F4 fix: The bit pattern is correctly stored as all-ones (UINT64_MAX).
      // Note: getInt() still returns -1 (sign-extended int64_t) because the
      // MLIR IntegerAttr API always sign-extends.  This is expected behavior.
      // The fix is in the WhyML emitter (F3) which now uses APInt formatting
      // via toStringUnsigned() for unsigned types instead of getInt().
      EXPECT_EQ(val.getZExtValue(), UINT64_MAX)
          << "APInt should store UINT64_MAX correctly via zero-extension";
      foundConst = true;
    }
  });
  EXPECT_TRUE(foundConst) << "Should have found a ConstantOp for UINT64_MAX";
}

// [F4] Simpler test: large u64 literal directly
TEST_F(BugReproLoweringTest, F4_U64LiteralValuePreserved) {
  // Use a simpler large literal that can be represented
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      R"(
    #include <cstdint>
    uint64_t get_big() {
      return 10000000000ULL;
    }
  )",
      {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  ASSERT_NE(ast, nullptr);

  std::map<const clang::FunctionDecl*, ContractInfo> contracts;
  mlir::MLIRContext mlirCtx;
  auto module = lowerToArc(mlirCtx, ast->getASTContext(), contracts);
  ASSERT_TRUE(module);

  bool foundConst = false;
  module->walk([&](arc::ConstantOp constOp) {
    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
    if (intAttr) {
      // 10000000000 fits in int64_t, so getSExtValue is fine here.
      // This test should PASS -- it's a control for comparison.
      EXPECT_EQ(intAttr.getInt(), 10000000000LL)
          << "Expected 10000000000 but got: " << intAttr.getInt();
      foundConst = true;
    }
  });
  EXPECT_TRUE(foundConst)
      << "Should have found a ConstantOp for the literal 10000000000ULL";
}

} // namespace
} // namespace arcanum
