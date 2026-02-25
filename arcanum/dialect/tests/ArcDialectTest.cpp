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

// ---------------------------------------------------------------------------
// TEST_P: Type creation (I32, Bool)
// ---------------------------------------------------------------------------

using TypeFactory = std::function<mlir::Type(mlir::MLIRContext*)>;
using TypeCreationParam = std::tuple<std::string, TypeFactory>;

class TypeCreationTest
    : public ArcDialectTest,
      public ::testing::WithParamInterface<TypeCreationParam> {};

TEST_P(TypeCreationTest, CreatesType) {
  auto [name, factory] = GetParam();
  auto type = factory(&context_);
  EXPECT_TRUE(type);
}

INSTANTIATE_TEST_SUITE_P(
    ArcTypes, TypeCreationTest,
    ::testing::Values(
        TypeCreationParam{"I32",
                          [](mlir::MLIRContext* ctx) -> mlir::Type {
                            return arc::I32Type::get(ctx);
                          }},
        TypeCreationParam{"Bool",
                          [](mlir::MLIRContext* ctx) -> mlir::Type {
                            return arc::BoolType::get(ctx);
                          }}),
    [](const ::testing::TestParamInfo<TypeCreationParam>& info) {
      return std::get<0>(info.param);
    });

// ---------------------------------------------------------------------------
// TEST_P: Binary op creation (Add, Sub, Mul, Div, Rem)
// ---------------------------------------------------------------------------

using BinaryOpCreator = std::function<mlir::Operation*(
    mlir::OpBuilder&, mlir::Location, mlir::Type, mlir::Value, mlir::Value)>;
using BinaryOpParam = std::tuple<std::string, BinaryOpCreator>;

class BinaryOpCreationTest
    : public ArcDialectTest,
      public ::testing::WithParamInterface<BinaryOpParam> {};

TEST_P(BinaryOpCreationTest, CreatesOp) {
  auto [name, creator] = GetParam();
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));

  auto* op = creator(*builder_, builder_->getUnknownLoc(), i32Type, lhs, rhs);
  EXPECT_NE(op, nullptr);

  module->destroy();
}

INSTANTIATE_TEST_SUITE_P(
    ArcBinaryOps, BinaryOpCreationTest,
    ::testing::Values(
        BinaryOpParam{"Add",
                      [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
                         mlir::Value lhs, mlir::Value rhs) -> mlir::Operation* {
                        return b.create<arc::AddOp>(loc, ty, lhs, rhs);
                      }},
        BinaryOpParam{"Sub",
                      [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
                         mlir::Value lhs, mlir::Value rhs) -> mlir::Operation* {
                        return b.create<arc::SubOp>(loc, ty, lhs, rhs);
                      }},
        BinaryOpParam{"Mul",
                      [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
                         mlir::Value lhs, mlir::Value rhs) -> mlir::Operation* {
                        return b.create<arc::MulOp>(loc, ty, lhs, rhs);
                      }},
        BinaryOpParam{"Div",
                      [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
                         mlir::Value lhs, mlir::Value rhs) -> mlir::Operation* {
                        return b.create<arc::DivOp>(loc, ty, lhs, rhs);
                      }},
        BinaryOpParam{"Rem",
                      [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
                         mlir::Value lhs, mlir::Value rhs) -> mlir::Operation* {
                        return b.create<arc::RemOp>(loc, ty, lhs, rhs);
                      }}),
    [](const ::testing::TestParamInfo<BinaryOpParam>& info) {
      return std::get<0>(info.param);
    });

// ---------------------------------------------------------------------------
// Individual tests for ops with unique structure
// ---------------------------------------------------------------------------

TEST_F(ArcDialectTest, ConstantOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
  auto constOp = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(42));

  EXPECT_TRUE(constOp);
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

  auto i32Type = arc::I32Type::get(&context_);
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

  auto i32Type = arc::I32Type::get(&context_);
  auto init = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(0));
  auto varOp = builder_->create<arc::VarOp>(builder_->getUnknownLoc(), i32Type,
                                            "x", init.getResult());

  EXPECT_TRUE(varOp);
  EXPECT_EQ(varOp.getName(), "x");
  module->destroy();
}

// ---------------------------------------------------------------------------
// B1: Coverage gap tests for AssignOp, IfOp, FuncOp
// ---------------------------------------------------------------------------

TEST_F(ArcDialectTest, AssignOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::I32Type::get(&context_);
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

  auto i32Type = arc::I32Type::get(&context_);
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

} // namespace
} // namespace arcanum
