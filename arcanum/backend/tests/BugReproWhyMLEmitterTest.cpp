/// Regression tests for WhyMLEmitter bugs found during Slice 2 review.
/// Originally these tests demonstrated the bugs; now they verify the fixes.

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

/// Test fixture for building Arc IR directly via OpBuilder.
class BugReproWhyMLEmitterTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&ctx_);
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

  /// Build a module with a division op carrying a specific overflow mode.
  std::optional<WhyMLResult>
  buildAndEmitDivFunc(arc::IntType type, const std::string& overflowMode,
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
    auto divOp = builder_->create<arc::DivOp>(builder_->getUnknownLoc(), type,
                                              entryBlock.getArgument(0),
                                              entryBlock.getArgument(1));

    if (!overflowMode.empty()) {
      divOp->setAttr("overflow", builder_->getStringAttr(overflowMode));
    }

    builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                    divOp.getResult());

    auto result = emitWhyML(module);
    module->destroy();
    return result;
  }

  /// Build a module with a constant op holding a u64 value.
  std::optional<WhyMLResult>
  buildAndEmitConstFunc(arc::IntType type, int64_t rawValue,
                        const std::string& funcName) {
    auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
    builder_->setInsertionPointToEnd(module.getBody());

    auto funcType = builder_->getFunctionType({}, {type});
    auto funcOp = builder_->create<arc::FuncOp>(
        builder_->getUnknownLoc(), funcName, funcType,
        /*requires_attr=*/mlir::StringAttr{},
        /*ensures_attr=*/mlir::StringAttr{});

    funcOp->setAttr("param_names", builder_->getArrayAttr({}));

    auto& entryBlock = funcOp.getBody().emplaceBlock();

    builder_->setInsertionPointToEnd(&entryBlock);
    auto constOp = builder_->create<arc::ConstantOp>(
        builder_->getUnknownLoc(), type,
        builder_->getIntegerAttr(builder_->getIntegerType(type.getWidth()),
                                 rawValue));
    builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                    constOp.getResult());

    auto result = emitWhyML(module);
    module->destroy();
    return result;
  }

  mlir::MLIRContext ctx_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

// [F1] Bug: u32 -> i32 cast is misclassified as widening (no range assertion).
//
// The isWidening check in emitCastOp treats unsigned-to-signed same-width
// casts as widening.  But u32 values > INT32_MAX (e.g., 3000000000) do not
// fit in i32.  The emitter should emit an assertion: 0 <= x /\ x <= 2147483647
// but does not, because isWidening is true.
TEST_F(BugReproWhyMLEmitterTest,
       F1_UnsignedToSignedSameWidthCastMissingAssert) {
  auto u32Type = arc::IntType::get(&ctx_, 32, false);
  auto i32Type = arc::IntType::get(&ctx_, 32, true);
  auto result = buildAndEmitCastFunc(u32Type, i32Type, "u32_to_i32");
  ASSERT_TRUE(result.has_value());

  // The cast u32 -> i32 should emit a range assertion because u32 values
  // above 2147483647 do not fit in i32.  This assertion is expected to FAIL
  // because the code incorrectly classifies this cast as "widening".
  EXPECT_NE(result->whymlText.find("assert"), std::string::npos)
      << "u32->i32 cast should emit a range assertion but does not.\n"
      << "This is bug F1: unsigned-to-signed same-width cast treated as "
         "widening.\n"
      << "WhyML output:\n"
      << result->whymlText;
}

// [F1] Also check u16 -> i16 (same bug pattern, different width)
TEST_F(BugReproWhyMLEmitterTest, F1_U16ToI16CastMissingAssert) {
  auto u16Type = arc::IntType::get(&ctx_, 16, false);
  auto i16Type = arc::IntType::get(&ctx_, 16, true);
  auto result = buildAndEmitCastFunc(u16Type, i16Type, "u16_to_i16");
  ASSERT_TRUE(result.has_value());

  EXPECT_NE(result->whymlText.find("assert"), std::string::npos)
      << "u16->i16 cast should emit a range assertion but does not.\n"
      << "This is bug F1: unsigned-to-signed same-width cast treated as "
         "widening.\n"
      << "WhyML output:\n"
      << result->whymlText;
}

// [F2] Bug: emitDivLikeOp ignores overflow mode, always emits trap assertion.
//
// When a DivOp has overflow="wrap", the emitter should NOT emit a range
// assertion (wrap mode means the user opted out of trapping on overflow).
// But emitDivLikeOp unconditionally calls emitTrapAssertion.
TEST_F(BugReproWhyMLEmitterTest, F2_DivOpInWrapModeEmitsSpuriousTrapAssert) {
  auto i8Type = arc::IntType::get(&ctx_, 8, true);
  auto result = buildAndEmitDivFunc(i8Type, "wrap", "div_i8_wrap");
  ASSERT_TRUE(result.has_value());

  // Count the number of "assert" occurrences.  Division should have
  // exactly ONE assert for division-by-zero.  In wrap mode, there should be
  // NO overflow assertion.
  std::string text = result->whymlText;
  size_t assertCount = 0;
  size_t pos = 0;
  while ((pos = text.find("assert", pos)) != std::string::npos) {
    ++assertCount;
    pos += 6;
  }

  // Expected: 1 assert (division-by-zero only).
  // Actual (bug): 2 asserts (division-by-zero + overflow range check).
  EXPECT_EQ(assertCount, 1u)
      << "Wrap-mode division should have only 1 assert (div-by-zero), "
         "not an overflow assertion.\n"
      << "This is bug F2: emitDivLikeOp ignores overflow mode.\n"
      << "WhyML output:\n"
      << result->whymlText;
}

// [F3] Bug: ConstantOp emission of unsigned 64-bit values > INT64_MAX produces
// negative string.
//
// IntegerAttr::getInt() returns int64_t, which sign-extends the APInt.
// For u64 UINT64_MAX (0xFFFFFFFFFFFFFFFF), getInt() returns -1.
// std::to_string(-1) produces "-1" instead of "18446744073709551615".
TEST_F(BugReproWhyMLEmitterTest, F3_U64MaxConstantEmittedAsNegative) {
  auto u64Type = arc::IntType::get(&ctx_, 64, false);
  // -1 in int64_t corresponds to UINT64_MAX in unsigned 64-bit representation
  auto result = buildAndEmitConstFunc(u64Type, -1, "u64_max");
  ASSERT_TRUE(result.has_value());

  // The WhyML output should contain the unsigned decimal representation
  // "18446744073709551615", not "-1".
  EXPECT_EQ(result->whymlText.find("-1"), std::string::npos)
      << "u64 UINT64_MAX should NOT be emitted as -1.\n"
      << "This is bug F3: getInt() sign-extends unsigned 64-bit values.\n"
      << "WhyML output:\n"
      << result->whymlText;

  EXPECT_NE(result->whymlText.find("18446744073709551615"), std::string::npos)
      << "u64 UINT64_MAX should be emitted as 18446744073709551615.\n"
      << "WhyML output:\n"
      << result->whymlText;
}

} // namespace
} // namespace arcanum
