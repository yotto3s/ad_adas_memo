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

} // namespace
} // namespace arcanum
