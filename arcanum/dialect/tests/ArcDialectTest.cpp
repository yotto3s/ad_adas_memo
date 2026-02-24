#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <tuple>

namespace arcanum {
namespace {

class ArcDialectTest : public ::testing::Test {
protected:
  void SetUp() override {
    context_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&context_);
  }

  mlir::MLIRContext context_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

// --- IntType creation tests for all 8 variants (parametrized) ---

TEST_F(ArcDialectTest, IntTypeCreationAllVariants) {
  struct TypeSpec {
    unsigned width;
    bool isSigned;
  };
  constexpr TypeSpec specs[] = {
      {8, true},  {16, true},  {32, true},  {64, true},
      {8, false}, {16, false}, {32, false}, {64, false},
  };

  for (const auto& spec : specs) {
    SCOPED_TRACE(std::string(spec.isSigned ? "i" : "u") +
                 std::to_string(spec.width));
    auto type = arc::IntType::get(&context_, spec.width, spec.isSigned);
    EXPECT_TRUE(type);
    EXPECT_EQ(type.getWidth(), spec.width);
    EXPECT_EQ(type.getIsSigned(), spec.isSigned);
  }
}

// --- Min/Max value bounds tests (parametrized) ---

TEST_F(ArcDialectTest, IntTypeBoundsAllVariants) {
  struct TypeSpec {
    unsigned width;
    bool isSigned;
  };
  constexpr TypeSpec specs[] = {
      {8, true},  {16, true},  {32, true},  {64, true},
      {8, false}, {16, false}, {32, false}, {64, false},
  };

  for (const auto& spec : specs) {
    SCOPED_TRACE(std::string(spec.isSigned ? "i" : "u") +
                 std::to_string(spec.width));
    auto type = arc::IntType::get(&context_, spec.width, spec.isSigned);

    llvm::APInt expectedMin = spec.isSigned
                                  ? llvm::APInt::getSignedMinValue(spec.width)
                                  : llvm::APInt(spec.width, 0);
    llvm::APInt expectedMax = spec.isSigned
                                  ? llvm::APInt::getSignedMaxValue(spec.width)
                                  : llvm::APInt::getMaxValue(spec.width);

    EXPECT_EQ(type.getMinValue(), expectedMin);
    EXPECT_EQ(type.getMaxValue(), expectedMax);
  }
}

// --- Backward compatibility: IntType(32, true) works as old I32Type ---

TEST_F(ArcDialectTest, IntTypeI32BackwardCompatible) {
  auto type = arc::IntType::get(&context_, 32, true);
  // Same type instance should be returned (uniqued by MLIR context)
  auto type2 = arc::IntType::get(&context_, 32, true);
  EXPECT_EQ(type, type2);
}

// --- BoolType unchanged ---

TEST_F(ArcDialectTest, BoolTypeCreation) {
  auto type = arc::BoolType::get(&context_);
  EXPECT_TRUE(type);
}

// --- Op creation tests (updated to use IntType) ---

TEST_F(ArcDialectTest, ConstantOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto constOp = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(42));

  EXPECT_TRUE(constOp);
  module->destroy();
}

// TC-11: Binary op creation tests (parametrized over all 5 arithmetic ops)
TEST_F(ArcDialectTest, BinaryArithOpCreationAllTypes) {
  auto testBinaryOp = [&](auto createFn, const char* name) {
    SCOPED_TRACE(name);
    auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
    builder_->setInsertionPointToEnd(module.getBody());

    auto i32Type = arc::IntType::get(&context_, 32, true);
    auto lhs = builder_->create<arc::ConstantOp>(
        builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(10));
    auto rhs = builder_->create<arc::ConstantOp>(
        builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(3));
    auto op = createFn(i32Type, lhs, rhs);

    EXPECT_TRUE(op);
    module->destroy();
  };

  testBinaryOp(
      [&](mlir::Type type, mlir::Value lhs, mlir::Value rhs) {
        return builder_->create<arc::AddOp>(builder_->getUnknownLoc(), type,
                                            lhs, rhs);
      },
      "AddOp");
  testBinaryOp(
      [&](mlir::Type type, mlir::Value lhs, mlir::Value rhs) {
        return builder_->create<arc::SubOp>(builder_->getUnknownLoc(), type,
                                            lhs, rhs);
      },
      "SubOp");
  testBinaryOp(
      [&](mlir::Type type, mlir::Value lhs, mlir::Value rhs) {
        return builder_->create<arc::MulOp>(builder_->getUnknownLoc(), type,
                                            lhs, rhs);
      },
      "MulOp");
  testBinaryOp(
      [&](mlir::Type type, mlir::Value lhs, mlir::Value rhs) {
        return builder_->create<arc::DivOp>(builder_->getUnknownLoc(), type,
                                            lhs, rhs);
      },
      "DivOp");
  testBinaryOp(
      [&](mlir::Type type, mlir::Value lhs, mlir::Value rhs) {
        return builder_->create<arc::RemOp>(builder_->getUnknownLoc(), type,
                                            lhs, rhs);
      },
      "RemOp");
}

TEST_F(ArcDialectTest, CmpOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto boolType = arc::BoolType::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto cmpOp =
      builder_->create<arc::CmpOp>(builder_->getUnknownLoc(), boolType,
                                   builder_->getStringAttr("lt"), lhs, rhs);

  EXPECT_TRUE(cmpOp);
  module->destroy();
}

TEST_F(ArcDialectTest, AndOrNotOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto boolType = arc::BoolType::get(&context_);
  auto t = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), boolType, builder_->getBoolAttr(true));
  auto f = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), boolType, builder_->getBoolAttr(false));

  auto andOp =
      builder_->create<arc::AndOp>(builder_->getUnknownLoc(), boolType, t, f);
  EXPECT_TRUE(andOp);

  auto orOp =
      builder_->create<arc::OrOp>(builder_->getUnknownLoc(), boolType, t, f);
  EXPECT_TRUE(orOp);

  auto notOp =
      builder_->create<arc::NotOp>(builder_->getUnknownLoc(), boolType, t);
  EXPECT_TRUE(notOp);

  module->destroy();
}

TEST_F(ArcDialectTest, ReturnOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto val = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(42));
  auto retOp = builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                               val.getResult());

  EXPECT_TRUE(retOp);
  module->destroy();
}

TEST_F(ArcDialectTest, VarOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto init = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(0));
  auto varOp = builder_->create<arc::VarOp>(builder_->getUnknownLoc(), i32Type,
                                            "x", init.getResult());

  EXPECT_TRUE(varOp);
  EXPECT_EQ(varOp.getName(), "x");
  module->destroy();
}

// --- Coverage gap tests for AssignOp, IfOp, FuncOp ---

TEST_F(ArcDialectTest, AssignOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto target = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(0));
  auto value = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(42));
  auto assignOp =
      builder_->create<arc::AssignOp>(builder_->getUnknownLoc(), target, value);

  EXPECT_TRUE(assignOp);
  module->destroy();
}

TEST_F(ArcDialectTest, IfOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto boolType = arc::BoolType::get(&context_);
  auto cond = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), boolType, builder_->getBoolAttr(true));

  auto ifOp = builder_->create<arc::IfOp>(builder_->getUnknownLoc(),
                                          mlir::TypeRange{}, cond);

  EXPECT_TRUE(ifOp);
  EXPECT_TRUE(ifOp.getThenRegion().empty());
  EXPECT_TRUE(ifOp.getElseRegion().empty());

  module->destroy();
}

TEST_F(ArcDialectTest, FuncOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto funcType = builder_->getFunctionType({i32Type}, {i32Type});

  auto funcOp = builder_->create<arc::FuncOp>(
      builder_->getUnknownLoc(), builder_->getStringAttr("test_func"),
      mlir::TypeAttr::get(funcType), mlir::StringAttr(), mlir::StringAttr());

  EXPECT_TRUE(funcOp);
  EXPECT_EQ(funcOp.getSymName(), "test_func");

  auto& block = funcOp.getBody().emplaceBlock();
  block.addArgument(i32Type, builder_->getUnknownLoc());

  auto savedIp = builder_->saveInsertionPoint();
  builder_->setInsertionPointToEnd(&block);
  builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                  block.getArgument(0));
  builder_->restoreInsertionPoint(savedIp);

  EXPECT_FALSE(funcOp.getBody().empty());
  EXPECT_EQ(funcOp.getBody().front().getNumArguments(), 1u);

  module->destroy();
}

// --- Test ops with different IntType widths ---

TEST_F(ArcDialectTest, AddOpWithU16Type) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto u16Type = arc::IntType::get(&context_, 16, false);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), u16Type, builder_->getI32IntegerAttr(100));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), u16Type, builder_->getI32IntegerAttr(200));
  auto addOp = builder_->create<arc::AddOp>(builder_->getUnknownLoc(), u16Type,
                                            lhs, rhs);

  EXPECT_TRUE(addOp);
  EXPECT_EQ(addOp.getResult().getType(), u16Type);
  module->destroy();
}

// --- Task 2: CastOp tests ---

TEST_F(ArcDialectTest, CastOpWideningI8ToI32) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i8Type = arc::IntType::get(&context_, 8, true);
  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto src = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i8Type, builder_->getI8IntegerAttr(42));
  auto castOp = builder_->create<arc::CastOp>(builder_->getUnknownLoc(),
                                              i32Type, src.getResult());

  EXPECT_TRUE(castOp);
  EXPECT_EQ(castOp.getInput().getType(), i8Type);
  EXPECT_EQ(castOp.getResult().getType(), i32Type);
  module->destroy();
}

TEST_F(ArcDialectTest, CastOpNarrowingI32ToI8) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto i8Type = arc::IntType::get(&context_, 8, true);
  auto src = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(100));
  auto castOp = builder_->create<arc::CastOp>(builder_->getUnknownLoc(), i8Type,
                                              src.getResult());

  EXPECT_TRUE(castOp);
  EXPECT_EQ(castOp.getInput().getType(), i32Type);
  EXPECT_EQ(castOp.getResult().getType(), i8Type);
  module->destroy();
}

TEST_F(ArcDialectTest, CastOpSignChangeI32ToU32) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto u32Type = arc::IntType::get(&context_, 32, false);
  auto src = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(5));
  auto castOp = builder_->create<arc::CastOp>(builder_->getUnknownLoc(),
                                              u32Type, src.getResult());

  EXPECT_TRUE(castOp);
  EXPECT_EQ(castOp.getInput().getType(), i32Type);
  EXPECT_EQ(castOp.getResult().getType(), u32Type);
  module->destroy();
}

// --- Task 3: Overflow mode string attribute tests ---

TEST_F(ArcDialectTest, AddOpWithOverflowWrapAttribute) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto addOp = builder_->create<arc::AddOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  // MLIR supports arbitrary attributes natively; attach overflow mode.
  addOp->setAttr("overflow", builder_->getStringAttr("wrap"));
  auto attr = addOp->getAttrOfType<mlir::StringAttr>("overflow");
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), "wrap");

  module->destroy();
}

TEST_F(ArcDialectTest, AddOpWithOverflowSaturateAttribute) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto addOp = builder_->create<arc::AddOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  addOp->setAttr("overflow", builder_->getStringAttr("saturate"));
  auto attr = addOp->getAttrOfType<mlir::StringAttr>("overflow");
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), "saturate");

  module->destroy();
}

TEST_F(ArcDialectTest, FuncOpWithOverflowTrapAttribute) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto funcType = mlir::FunctionType::get(&context_, {i32Type}, {i32Type});
  auto funcOp = builder_->create<arc::FuncOp>(
      builder_->getUnknownLoc(), "my_func", funcType,
      /*requires_attr=*/mlir::StringAttr{},
      /*ensures_attr=*/mlir::StringAttr{});

  funcOp->setAttr("overflow", builder_->getStringAttr("trap"));
  auto attr = funcOp->getAttrOfType<mlir::StringAttr>("overflow");
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), "trap");

  module->destroy();
}

TEST_F(ArcDialectTest, MulOpAbsentOverflowAttributeIsNull) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(3));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(4));
  auto mulOp = builder_->create<arc::MulOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  // No overflow attribute set; lookup should return null.
  auto attr = mulOp->getAttrOfType<mlir::StringAttr>("overflow");
  EXPECT_FALSE(attr);

  module->destroy();
}

// ============================================================
// TC-12: Type print tests — verify dialect printer output
// ============================================================

// Helper to print an MLIR type to string via the dialect printer.
static std::string typeToString(mlir::Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  type.print(os);
  return str;
}

// Verify that all 8 integer type mnemonics print correctly through
// ArcDialect::printType (invoked by type.print()).
TEST_F(ArcDialectTest, IntTypePrintMnemonics) {
  EXPECT_EQ(typeToString(arc::IntType::get(&context_, 8, true)), "!arc.i8");
  EXPECT_EQ(typeToString(arc::IntType::get(&context_, 16, true)), "!arc.i16");
  EXPECT_EQ(typeToString(arc::IntType::get(&context_, 32, true)), "!arc.i32");
  EXPECT_EQ(typeToString(arc::IntType::get(&context_, 64, true)), "!arc.i64");
  EXPECT_EQ(typeToString(arc::IntType::get(&context_, 8, false)), "!arc.u8");
  EXPECT_EQ(typeToString(arc::IntType::get(&context_, 16, false)), "!arc.u16");
  EXPECT_EQ(typeToString(arc::IntType::get(&context_, 32, false)), "!arc.u32");
  EXPECT_EQ(typeToString(arc::IntType::get(&context_, 64, false)), "!arc.u64");
}

// Verify BoolType prints correctly.
TEST_F(ArcDialectTest, BoolTypePrintMnemonic) {
  EXPECT_EQ(typeToString(arc::BoolType::get(&context_)), "!arc.bool");
}

// ============================================================
// TC-13: IntType width validation via verify()
// ============================================================

// Verify that IntType::verify() rejects unsupported widths and accepts valid
// ones. This tests the same validation logic used by IntType::parse() and
// IntType::getChecked().
TEST_F(ArcDialectTest, IntTypeVerifyRejectsInvalidWidth) {
  // Suppress diagnostics emitted by verify
  auto diagHandler = context_.getDiagEngine().registerHandler(
      [](mlir::Diagnostic&) { return mlir::success(); });

  auto emitError = [&]() {
    return mlir::emitError(mlir::UnknownLoc::get(&context_));
  };

  // Invalid widths should fail
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 0, true)));
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 1, true)));
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 12, true)));
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 24, false)));
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 128, true)));

  // Valid widths should succeed (both signed and unsigned)
  EXPECT_TRUE(mlir::succeeded(arc::IntType::verify(emitError, 8, true)));
  EXPECT_TRUE(mlir::succeeded(arc::IntType::verify(emitError, 16, false)));
  EXPECT_TRUE(mlir::succeeded(arc::IntType::verify(emitError, 32, true)));
  EXPECT_TRUE(mlir::succeeded(arc::IntType::verify(emitError, 64, false)));

  context_.getDiagEngine().eraseHandler(diagHandler);
}

} // namespace
} // namespace arcanum
