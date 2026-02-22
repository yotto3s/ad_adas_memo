#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <gtest/gtest.h>

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

TEST_F(ArcDialectTest, I32TypeCreation) {
  auto type = arc::I32Type::get(&context_);
  EXPECT_TRUE(type);
}

TEST_F(ArcDialectTest, BoolTypeCreation) {
  auto type = arc::BoolType::get(&context_);
  EXPECT_TRUE(type);
}

TEST_F(ArcDialectTest, ConstantOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto constOp = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type,
      builder_->getI32IntegerAttr(42));

  EXPECT_TRUE(constOp);
  module->destroy();
}

TEST_F(ArcDialectTest, AddOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto addOp = builder_->create<arc::AddOp>(
      builder_->getUnknownLoc(), i32Type, lhs, rhs);

  EXPECT_TRUE(addOp);
  module->destroy();
}

// TC-11: Tests for additional ops
TEST_F(ArcDialectTest, SubOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(5));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(3));
  auto subOp = builder_->create<arc::SubOp>(
      builder_->getUnknownLoc(), i32Type, lhs, rhs);

  EXPECT_TRUE(subOp);
  module->destroy();
}

TEST_F(ArcDialectTest, MulOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(4));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(7));
  auto mulOp = builder_->create<arc::MulOp>(
      builder_->getUnknownLoc(), i32Type, lhs, rhs);

  EXPECT_TRUE(mulOp);
  module->destroy();
}

TEST_F(ArcDialectTest, DivOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(10));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto divOp = builder_->create<arc::DivOp>(
      builder_->getUnknownLoc(), i32Type, lhs, rhs);

  EXPECT_TRUE(divOp);
  module->destroy();
}

TEST_F(ArcDialectTest, RemOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(10));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(3));
  auto remOp = builder_->create<arc::RemOp>(
      builder_->getUnknownLoc(), i32Type, lhs, rhs);

  EXPECT_TRUE(remOp);
  module->destroy();
}

TEST_F(ArcDialectTest, CmpOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto boolType = arc::BoolType::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto cmpOp = builder_->create<arc::CmpOp>(
      builder_->getUnknownLoc(), boolType,
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

  auto andOp = builder_->create<arc::AndOp>(
      builder_->getUnknownLoc(), boolType, t, f);
  EXPECT_TRUE(andOp);

  auto orOp = builder_->create<arc::OrOp>(
      builder_->getUnknownLoc(), boolType, t, f);
  EXPECT_TRUE(orOp);

  auto notOp = builder_->create<arc::NotOp>(
      builder_->getUnknownLoc(), boolType, t);
  EXPECT_TRUE(notOp);

  module->destroy();
}

TEST_F(ArcDialectTest, ReturnOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto val = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(42));
  auto retOp = builder_->create<arc::ReturnOp>(
      builder_->getUnknownLoc(), val.getResult());

  EXPECT_TRUE(retOp);
  module->destroy();
}

TEST_F(ArcDialectTest, VarOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto init = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(0));
  auto varOp = builder_->create<arc::VarOp>(
      builder_->getUnknownLoc(), i32Type, "x", init.getResult());

  EXPECT_TRUE(varOp);
  EXPECT_EQ(varOp.getName(), "x");
  module->destroy();
}

} // namespace
} // namespace arcanum
