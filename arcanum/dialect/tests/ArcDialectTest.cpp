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
#include <vector>

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

// --- B2: CastOp type-pair tests (parameterized) ---

struct CastOpParam {
  const char* name;
  unsigned srcW;
  bool srcS;
  unsigned dstW;
  bool dstS;
};

class CastOpParamTest : public ::testing::TestWithParam<CastOpParam> {
protected:
  void SetUp() override {
    context_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&context_);
  }

  mlir::MLIRContext context_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

TEST_P(CastOpParamTest, CastOpTypePair) {
  auto [name, srcW, srcS, dstW, dstS] = GetParam();

  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto srcType = arc::IntType::get(&context_, srcW, srcS);
  auto dstType = arc::IntType::get(&context_, dstW, dstS);
  auto src = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), srcType, builder_->getI32IntegerAttr(5));
  auto castOp = builder_->create<arc::CastOp>(builder_->getUnknownLoc(),
                                              dstType, src.getResult());

  EXPECT_TRUE(castOp);
  EXPECT_EQ(castOp.getInput().getType(), srcType);
  EXPECT_EQ(castOp.getResult().getType(), dstType);
  module->destroy();
}

INSTANTIATE_TEST_SUITE_P(
    ArcDialect, CastOpParamTest,
    ::testing::Values(CastOpParam{"WideningI8ToI32", 8, true, 32, true},
                      CastOpParam{"NarrowingI32ToI8", 32, true, 8, true},
                      CastOpParam{"SignChangeI32ToU32", 32, true, 32, false}),
    [](const ::testing::TestParamInfo<CastOpParam>& info) {
      return info.param.name;
    });

// --- B1: AddOp overflow attribute tests (parameterized) ---

struct AddOpOverflowParam {
  const char* name;
  const char* overflowValue;
};

class AddOpOverflowTest : public ::testing::TestWithParam<AddOpOverflowParam> {
protected:
  void SetUp() override {
    context_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&context_);
  }

  mlir::MLIRContext context_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

TEST_P(AddOpOverflowTest, SetsAndRetrievesOverflowAttribute) {
  auto [name, overflowValue] = GetParam();

  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Type = arc::IntType::get(&context_, 32, true);
  auto lhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(1));
  auto rhs = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Type, builder_->getI32IntegerAttr(2));
  auto addOp = builder_->create<arc::AddOp>(builder_->getUnknownLoc(), i32Type,
                                            lhs, rhs);

  addOp->setAttr("overflow", builder_->getStringAttr(overflowValue));
  auto attr = addOp->getAttrOfType<mlir::StringAttr>("overflow");
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), overflowValue);

  module->destroy();
}

INSTANTIATE_TEST_SUITE_P(
    ArcDialect, AddOpOverflowTest,
    ::testing::Values(AddOpOverflowParam{"Wrap", "wrap"},
                      AddOpOverflowParam{"Saturate", "saturate"}),
    [](const ::testing::TestParamInfo<AddOpOverflowParam>& info) {
      return info.param.name;
    });

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

  auto attr = mulOp->getAttrOfType<mlir::StringAttr>("overflow");
  EXPECT_FALSE(attr);

  module->destroy();
}

// ============================================================
// TC-12: Type print tests -- verify dialect printer output
// ============================================================

static std::string typeToString(mlir::Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  type.print(os);
  return str;
}

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

TEST_F(ArcDialectTest, BoolTypePrintMnemonic) {
  EXPECT_EQ(typeToString(arc::BoolType::get(&context_)), "!arc.bool");
}

// ============================================================
// TC-13: IntType width validation via verify()
// ============================================================

TEST_F(ArcDialectTest, IntTypeVerifyRejectsInvalidWidth) {
  auto diagHandler = context_.getDiagEngine().registerHandler(
      [](mlir::Diagnostic&) { return mlir::success(); });

  auto emitError = [&]() {
    return mlir::emitError(mlir::UnknownLoc::get(&context_));
  };

  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 0, true)));
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 1, true)));
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 12, true)));
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 24, false)));
  EXPECT_TRUE(mlir::failed(arc::IntType::verify(emitError, 128, true)));

  EXPECT_TRUE(mlir::succeeded(arc::IntType::verify(emitError, 8, true)));
  EXPECT_TRUE(mlir::succeeded(arc::IntType::verify(emitError, 16, false)));
  EXPECT_TRUE(mlir::succeeded(arc::IntType::verify(emitError, 32, true)));
  EXPECT_TRUE(mlir::succeeded(arc::IntType::verify(emitError, 64, false)));

  context_.getDiagEngine().eraseHandler(diagHandler);
}

// ============================================================
// Loop operations (Slice 3)
// ============================================================

// --- B6: LoopOp condition_first tests (parameterized) ---

struct LoopConditionFirstParam {
  const char* name;
  bool conditionFirst;
};

class LoopConditionFirstTest
    : public ::testing::TestWithParam<LoopConditionFirstParam> {
protected:
  void SetUp() override {
    context_.getOrLoadDialect<arc::ArcDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(&context_);
  }

  mlir::MLIRContext context_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

TEST_P(LoopConditionFirstTest, SetsConditionFirstAttribute) {
  auto [name, conditionFirst] = GetParam();

  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto loopOp = builder_->create<arc::LoopOp>(builder_->getUnknownLoc());
  loopOp->setAttr("condition_first", builder_->getBoolAttr(conditionFirst));

  if (conditionFirst) {
    loopOp->setAttr("invariant", builder_->getStringAttr("i >= 0 && i <= n"));
    loopOp->setAttr("variant", builder_->getStringAttr("n - i"));
    loopOp->setAttr("assigns", builder_->getStringAttr("i, sum"));
  }

  EXPECT_TRUE(loopOp);
  auto condFirst = loopOp->getAttrOfType<mlir::BoolAttr>("condition_first");
  ASSERT_TRUE(condFirst);
  EXPECT_EQ(condFirst.getValue(), conditionFirst);

  if (conditionFirst) {
    auto inv = loopOp->getAttrOfType<mlir::StringAttr>("invariant");
    ASSERT_TRUE(inv);
    EXPECT_EQ(inv.getValue(), "i >= 0 && i <= n");
  }

  module->destroy();
}

INSTANTIATE_TEST_SUITE_P(
    ArcDialect, LoopConditionFirstTest,
    ::testing::Values(LoopConditionFirstParam{"ForLoop", true},
                      LoopConditionFirstParam{"DoWhile", false}),
    [](const ::testing::TestParamInfo<LoopConditionFirstParam>& info) {
      return info.param.name;
    });

// --- B3: Simple zero-operand op creation tests (parameterized) ---

TEST_F(ArcDialectTest, SimpleOpCreation) {
  struct SimpleOpCase {
    const char* name;
    std::function<bool(mlir::OpBuilder&, mlir::Location)> create;
  };

  std::vector<SimpleOpCase> cases = {
      {"BreakOp",
       [](mlir::OpBuilder& b, mlir::Location loc) {
         return static_cast<bool>(b.create<arc::BreakOp>(loc));
       }},
      {"ContinueOp",
       [](mlir::OpBuilder& b, mlir::Location loc) {
         return static_cast<bool>(b.create<arc::ContinueOp>(loc));
       }},
      {"YieldOp",
       [](mlir::OpBuilder& b, mlir::Location loc) {
         return static_cast<bool>(b.create<arc::YieldOp>(loc));
       }},
  };

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.name);
    auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
    builder_->setInsertionPointToEnd(module.getBody());

    EXPECT_TRUE(tc.create(*builder_, builder_->getUnknownLoc()));

    module->destroy();
  }
}

TEST_F(ArcDialectTest, ConditionOpCreation) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto boolType = arc::BoolType::get(&context_);
  auto cond = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), boolType, builder_->getBoolAttr(true));
  auto condOp = builder_->create<arc::ConditionOp>(builder_->getUnknownLoc(),
                                                   cond.getResult());
  EXPECT_TRUE(condOp);

  module->destroy();
}

TEST_F(ArcDialectTest, LoopOpHasFourRegions) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto loopOp = builder_->create<arc::LoopOp>(builder_->getUnknownLoc());
  EXPECT_EQ(loopOp->getNumRegions(), 4u);

  module->destroy();
}

// ---------------------------------------------------------------------------
// [TC-10] LoopOp roundtrip test
// ---------------------------------------------------------------------------

TEST_F(ArcDialectTest, LoopOpPrintRoundtrip) {
  auto module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  builder_->setInsertionPointToEnd(module.getBody());

  auto i32Ty = arc::IntType::get(&context_, 32, true);
  auto boolTy = arc::BoolType::get(&context_);
  auto funcType = builder_->getFunctionType({i32Ty}, {i32Ty});

  auto funcOp = builder_->create<arc::FuncOp>(
      builder_->getUnknownLoc(), builder_->getStringAttr("loop_test"),
      mlir::TypeAttr::get(funcType), mlir::StringAttr(), mlir::StringAttr());
  auto& entryBlock = funcOp.getBody().emplaceBlock();
  entryBlock.addArgument(i32Ty, builder_->getUnknownLoc());

  builder_->setInsertionPointToEnd(&entryBlock);

  auto initConst = builder_->create<arc::ConstantOp>(
      builder_->getUnknownLoc(), i32Ty, builder_->getI32IntegerAttr(0));
  auto varI = builder_->create<arc::VarOp>(builder_->getUnknownLoc(), i32Ty,
                                           "i", initConst);

  auto loopOp = builder_->create<arc::LoopOp>(builder_->getUnknownLoc());
  loopOp->setAttr("condition_first", builder_->getBoolAttr(true));
  loopOp->setAttr("invariant", builder_->getStringAttr("i >= 0"));
  loopOp->setAttr("variant", builder_->getStringAttr("n - i"));
  loopOp->setAttr("assigns", builder_->getStringAttr("i"));

  {
    auto& block = loopOp.getInitRegion().emplaceBlock();
    builder_->setInsertionPointToEnd(&block);
    builder_->create<arc::YieldOp>(builder_->getUnknownLoc());
  }

  {
    auto& block = loopOp.getCondRegion().emplaceBlock();
    builder_->setInsertionPointToEnd(&block);
    auto trueCond = builder_->create<arc::ConstantOp>(
        builder_->getUnknownLoc(), boolTy, builder_->getBoolAttr(true));
    builder_->create<arc::ConditionOp>(builder_->getUnknownLoc(), trueCond);
  }

  {
    auto& block = loopOp.getBodyRegion().emplaceBlock();
    builder_->setInsertionPointToEnd(&block);
    auto one = builder_->create<arc::ConstantOp>(
        builder_->getUnknownLoc(), i32Ty, builder_->getI32IntegerAttr(1));
    builder_->create<arc::AssignOp>(builder_->getUnknownLoc(), varI.getResult(),
                                    one);
    builder_->create<arc::YieldOp>(builder_->getUnknownLoc());
  }

  {
    auto& block = loopOp.getUpdateRegion().emplaceBlock();
    builder_->setInsertionPointToEnd(&block);
    builder_->create<arc::YieldOp>(builder_->getUnknownLoc());
  }

  builder_->setInsertionPointAfter(loopOp);
  builder_->create<arc::ReturnOp>(builder_->getUnknownLoc(),
                                  entryBlock.getArgument(0));

  std::string printed;
  llvm::raw_string_ostream os(printed);
  module->print(os);

  EXPECT_NE(printed.find("arc.loop"), std::string::npos)
      << "Printed output should contain 'arc.loop'";
  EXPECT_NE(printed.find("condition_first"), std::string::npos)
      << "Printed output should contain 'condition_first'";
  EXPECT_NE(printed.find("invariant"), std::string::npos)
      << "Printed output should contain 'invariant'";
  EXPECT_NE(printed.find("variant"), std::string::npos)
      << "Printed output should contain 'variant'";
  EXPECT_NE(printed.find("assigns"), std::string::npos)
      << "Printed output should contain 'assigns'";
  EXPECT_NE(printed.find("arc.condition"), std::string::npos)
      << "Printed output should contain 'arc.condition'";
  EXPECT_NE(printed.find("arc.yield"), std::string::npos)
      << "Printed output should contain 'arc.yield'";
  EXPECT_NE(printed.find("arc.assign"), std::string::npos)
      << "Printed output should contain 'arc.assign'";

  module->destroy();
}

} // namespace
} // namespace arcanum
