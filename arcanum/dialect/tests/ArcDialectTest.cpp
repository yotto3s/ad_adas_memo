#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

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

// --- IntType creation tests for all 8 variants ---

TEST_F(ArcDialectTest, IntTypeI8Creation) {
  auto type = arc::IntType::get(&context_, 8, true);
  EXPECT_TRUE(type);
  EXPECT_EQ(type.getWidth(), 8u);
  EXPECT_TRUE(type.getIsSigned());
}

TEST_F(ArcDialectTest, IntTypeI16Creation) {
  auto type = arc::IntType::get(&context_, 16, true);
  EXPECT_TRUE(type);
  EXPECT_EQ(type.getWidth(), 16u);
  EXPECT_TRUE(type.getIsSigned());
}

TEST_F(ArcDialectTest, IntTypeI32Creation) {
  auto type = arc::IntType::get(&context_, 32, true);
  EXPECT_TRUE(type);
  EXPECT_EQ(type.getWidth(), 32u);
  EXPECT_TRUE(type.getIsSigned());
}

TEST_F(ArcDialectTest, IntTypeI64Creation) {
  auto type = arc::IntType::get(&context_, 64, true);
  EXPECT_TRUE(type);
  EXPECT_EQ(type.getWidth(), 64u);
  EXPECT_TRUE(type.getIsSigned());
}

TEST_F(ArcDialectTest, IntTypeU8Creation) {
  auto type = arc::IntType::get(&context_, 8, false);
  EXPECT_TRUE(type);
  EXPECT_EQ(type.getWidth(), 8u);
  EXPECT_FALSE(type.getIsSigned());
}

TEST_F(ArcDialectTest, IntTypeU16Creation) {
  auto type = arc::IntType::get(&context_, 16, false);
  EXPECT_TRUE(type);
  EXPECT_EQ(type.getWidth(), 16u);
  EXPECT_FALSE(type.getIsSigned());
}

TEST_F(ArcDialectTest, IntTypeU32Creation) {
  auto type = arc::IntType::get(&context_, 32, false);
  EXPECT_TRUE(type);
  EXPECT_EQ(type.getWidth(), 32u);
  EXPECT_FALSE(type.getIsSigned());
}

TEST_F(ArcDialectTest, IntTypeU64Creation) {
  auto type = arc::IntType::get(&context_, 64, false);
  EXPECT_TRUE(type);
  EXPECT_EQ(type.getWidth(), 64u);
  EXPECT_FALSE(type.getIsSigned());
}

// --- Min/Max value bounds tests ---

TEST_F(ArcDialectTest, IntTypeI8Bounds) {
  auto type = arc::IntType::get(&context_, 8, true);
  EXPECT_EQ(type.getMinValue(), llvm::APInt(8, -128, true));
  EXPECT_EQ(type.getMaxValue(), llvm::APInt(8, 127, true));
}

TEST_F(ArcDialectTest, IntTypeI16Bounds) {
  auto type = arc::IntType::get(&context_, 16, true);
  EXPECT_EQ(type.getMinValue(), llvm::APInt(16, -32768, true));
  EXPECT_EQ(type.getMaxValue(), llvm::APInt(16, 32767, true));
}

TEST_F(ArcDialectTest, IntTypeI32Bounds) {
  auto type = arc::IntType::get(&context_, 32, true);
  EXPECT_EQ(type.getMinValue(), llvm::APInt(32, -2147483648LL, true));
  EXPECT_EQ(type.getMaxValue(), llvm::APInt(32, 2147483647, true));
}

TEST_F(ArcDialectTest, IntTypeI64Bounds) {
  auto type = arc::IntType::get(&context_, 64, true);
  EXPECT_EQ(type.getMinValue(), llvm::APInt::getSignedMinValue(64));
  EXPECT_EQ(type.getMaxValue(), llvm::APInt::getSignedMaxValue(64));
}

TEST_F(ArcDialectTest, IntTypeU8Bounds) {
  auto type = arc::IntType::get(&context_, 8, false);
  EXPECT_EQ(type.getMinValue(), llvm::APInt(8, 0));
  EXPECT_EQ(type.getMaxValue(), llvm::APInt(8, 255));
}

TEST_F(ArcDialectTest, IntTypeU16Bounds) {
  auto type = arc::IntType::get(&context_, 16, false);
  EXPECT_EQ(type.getMinValue(), llvm::APInt(16, 0));
  EXPECT_EQ(type.getMaxValue(), llvm::APInt(16, 65535));
}

TEST_F(ArcDialectTest, IntTypeU32Bounds) {
  auto type = arc::IntType::get(&context_, 32, false);
  EXPECT_EQ(type.getMinValue(), llvm::APInt(32, 0));
  EXPECT_EQ(type.getMaxValue(), llvm::APInt::getMaxValue(32));
}

TEST_F(ArcDialectTest, IntTypeU64Bounds) {
  auto type = arc::IntType::get(&context_, 64, false);
  EXPECT_EQ(type.getMinValue(), llvm::APInt(64, 0));
  EXPECT_EQ(type.getMaxValue(), llvm::APInt::getMaxValue(64));
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

TEST_F(ArcDialectTest, AddOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto addOp = builder_->create<arc::AddOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  EXPECT_TRUE(addOp);
  module->destroy();
}

// TC-11: Tests for additional ops
TEST_F(ArcDialectTest, SubOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(5));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(3));
  auto subOp = builder_->create<arc::SubOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  EXPECT_TRUE(subOp);
  module->destroy();
}

TEST_F(ArcDialectTest, MulOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(4));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(7));
  auto mulOp = builder_->create<arc::MulOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  EXPECT_TRUE(mulOp);
  module->destroy();
}

TEST_F(ArcDialectTest, DivOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(10));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto divOp = builder_->create<arc::DivOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  EXPECT_TRUE(divOp);
  module->destroy();
}

TEST_F(ArcDialectTest, RemOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(10));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(3));
  auto remOp = builder_->create<arc::RemOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  EXPECT_TRUE(remOp);
  module->destroy();
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

} // namespace
} // namespace arcanum
