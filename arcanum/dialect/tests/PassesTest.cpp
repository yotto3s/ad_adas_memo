#include "arcanum/passes/Passes.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace arcanum {
namespace {

// ---------------------------------------------------------------------------
// Test helper: builds a module with a function containing a loop.
// The loop has a body region with arc.assign ops targeting arc.var ops,
// a cond region with arc.condition, and optionally an init region.
// ---------------------------------------------------------------------------

struct LoopTestModule {
  mlir::MLIRContext ctx;
  mlir::ModuleOp module;

  LoopTestModule() { ctx.getOrLoadDialect<arc::ArcDialect>(); }
  ~LoopTestModule() { module->destroy(); }
};

mlir::Block& createFuncWithBlock(mlir::OpBuilder& builder,
                                 LoopTestModule& env) {
  auto i32Ty = arc::IntType::get(&env.ctx, 32, true);
  builder.setInsertionPointToEnd(env.module.getBody());
  auto funcType = builder.getFunctionType({i32Ty}, {i32Ty});
  auto funcOp = builder.create<arc::FuncOp>(
      builder.getUnknownLoc(), builder.getStringAttr("test_func"),
      mlir::TypeAttr::get(funcType), mlir::StringAttr(), mlir::StringAttr());
  auto& block = funcOp.getBody().emplaceBlock();
  block.addArgument(i32Ty, builder.getUnknownLoc());
  return block;
}

void populateCondRegion(mlir::OpBuilder& builder, arc::LoopOp loopOp,
                        mlir::MLIRContext& ctx) {
  auto loc = builder.getUnknownLoc();
  auto boolTy = arc::BoolType::get(&ctx);
  auto& condBlock = loopOp.getCondRegion().emplaceBlock();
  builder.setInsertionPointToEnd(&condBlock);
  auto trueCond =
      builder.create<arc::ConstantOp>(loc, boolTy, builder.getBoolAttr(true));
  builder.create<arc::ConditionOp>(loc, trueCond);
}

void populateBodyWithAssigns(mlir::OpBuilder& builder, arc::LoopOp loopOp,
                             arc::VarOp varX, arc::VarOp varY,
                             mlir::Value newVal) {
  auto loc = builder.getUnknownLoc();
  auto& bodyBlock = loopOp.getBodyRegion().emplaceBlock();
  builder.setInsertionPointToEnd(&bodyBlock);
  builder.create<arc::AssignOp>(loc, varX.getResult(), newVal);
  builder.create<arc::AssignOp>(loc, varY.getResult(), newVal);
  builder.create<arc::YieldOp>(loc);
}

void populateInitRegion(mlir::OpBuilder& builder, arc::LoopOp loopOp) {
  auto& initBlock = loopOp.getInitRegion().emplaceBlock();
  builder.setInsertionPointToEnd(&initBlock);
  builder.create<arc::YieldOp>(builder.getUnknownLoc());
}

arc::LoopOp createLoopWithAssignsInBody(mlir::OpBuilder& builder,
                                        LoopTestModule& env,
                                        mlir::Block& funcBlock,
                                        bool withInitRegion) {
  auto loc = builder.getUnknownLoc();
  auto i32Ty = arc::IntType::get(&env.ctx, 32, true);

  builder.setInsertionPointToEnd(&funcBlock);

  auto initConst =
      builder.create<arc::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(0));
  auto varX = builder.create<arc::VarOp>(loc, i32Ty, "x", initConst);
  auto varY = builder.create<arc::VarOp>(loc, i32Ty, "y", initConst);
  auto newVal =
      builder.create<arc::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(1));

  auto loopOp = builder.create<arc::LoopOp>(loc);

  populateCondRegion(builder, loopOp, env.ctx);
  populateBodyWithAssigns(builder, loopOp, varX, varY, newVal);

  if (withInitRegion) {
    populateInitRegion(builder, loopOp);
  }

  builder.setInsertionPointAfter(loopOp);
  builder.create<arc::ReturnOp>(loc, funcBlock.getArgument(0));

  return loopOp;
}

// ---------------------------------------------------------------------------
// Diagnostic capture helper
// ---------------------------------------------------------------------------

struct DiagnosticCapture {
  mlir::MLIRContext& ctx;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  mlir::ScopedDiagnosticHandler handler;

  explicit DiagnosticCapture(mlir::MLIRContext& context)
      : ctx(context), handler(&context, [this](mlir::Diagnostic& diag) {
          if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
            errors.push_back(diag.str());
          } else if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning) {
            warnings.push_back(diag.str());
          }
          return mlir::success();
        }) {}
};

// ---------------------------------------------------------------------------
// Slice 1 tests
// ---------------------------------------------------------------------------

TEST(PassesTest, SucceedsOnValidModule) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<arc::ArcDialect>();
  mlir::OpBuilder builder(&ctx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create a valid arc.func
  builder.setInsertionPointToEnd(module.getBody());
  auto funcType = builder.getFunctionType({arc::IntType::get(&ctx, 32, true)},
                                          {arc::IntType::get(&ctx, 32, true)});
  auto funcOp = builder.create<arc::FuncOp>(
      builder.getUnknownLoc(), builder.getStringAttr("test_func"),
      mlir::TypeAttr::get(funcType), mlir::StringAttr(), mlir::StringAttr());
  auto& block = funcOp.getBody().emplaceBlock();
  block.addArgument(arc::IntType::get(&ctx, 32, true), builder.getUnknownLoc());

  auto savedIp = builder.saveInsertionPoint();
  builder.setInsertionPointToEnd(&block);
  builder.create<arc::ReturnOp>(builder.getUnknownLoc(), block.getArgument(0));
  builder.restoreInsertionPoint(savedIp);

  auto result = runPasses(module);
  EXPECT_TRUE(result.succeeded());

  module->destroy();
}

TEST(PassesTest, SucceedsOnEmptyModule) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<arc::ArcDialect>();
  mlir::OpBuilder builder(&ctx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto result = runPasses(module);
  EXPECT_TRUE(result.succeeded());

  module->destroy();
}

// B9: Invalid module should cause MLIR verification to detect the error.
// runPasses() uses a PassManager which only runs the verifier after each pass;
// with zero passes added, the verifier does not run. We verify the intent by
// calling mlir::verify() directly on the invalid module.
TEST(PassesTest, VerifierRejectsInvalidModule) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<arc::ArcDialect>();
  mlir::OpBuilder builder(&ctx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create a FuncOp with a non-empty body but no terminator.
  builder.setInsertionPointToEnd(module.getBody());
  auto funcType = builder.getFunctionType({arc::IntType::get(&ctx, 32, true)},
                                          {arc::IntType::get(&ctx, 32, true)});
  auto funcOp = builder.create<arc::FuncOp>(
      builder.getUnknownLoc(), builder.getStringAttr("bad_func"),
      mlir::TypeAttr::get(funcType), mlir::StringAttr(), mlir::StringAttr());
  auto& block = funcOp.getBody().emplaceBlock();
  block.addArgument(arc::IntType::get(&ctx, 32, true), builder.getUnknownLoc());
  // Intentionally leave the block without a terminator

  EXPECT_TRUE(mlir::failed(mlir::verify(module)));

  module->destroy();
}

// ---------------------------------------------------------------------------
// LoopContractPass tests
// ---------------------------------------------------------------------------

TEST(PassesTest, LoopContractPassAutoComputesAssigns) {
  LoopTestModule env;
  mlir::OpBuilder builder(&env.ctx);
  env.module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto& funcBlock = createFuncWithBlock(builder, env);

  // Create a for-loop (with init region) with assigns in body but no
  // "assigns" attribute. Also add invariant+variant so validation passes.
  auto loopOp = createLoopWithAssignsInBody(builder, env, funcBlock,
                                            /*withInitRegion=*/true);
  loopOp->setAttr("invariant", builder.getStringAttr("true"));
  loopOp->setAttr("variant", builder.getStringAttr("n - i"));

  DiagnosticCapture diag(env.ctx);
  auto result = runPasses(env.module);
  EXPECT_TRUE(result.succeeded());

  // The pass should auto-compute assigns from the two arc.assign targets.
  auto assignsAttr = loopOp->getAttrOfType<mlir::StringAttr>("assigns");
  ASSERT_TRUE(assignsAttr != nullptr);

  auto assignsStr = assignsAttr.getValue().str();
  EXPECT_NE(assignsStr.find("x"), std::string::npos);
  EXPECT_NE(assignsStr.find("y"), std::string::npos);
}

TEST(PassesTest, LoopContractPassPreservesUserAssigns) {
  LoopTestModule env;
  mlir::OpBuilder builder(&env.ctx);
  env.module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto& funcBlock = createFuncWithBlock(builder, env);

  auto loopOp = createLoopWithAssignsInBody(builder, env, funcBlock,
                                            /*withInitRegion=*/true);
  loopOp->setAttr("assigns", builder.getStringAttr("user_provided"));
  loopOp->setAttr("invariant", builder.getStringAttr("true"));
  loopOp->setAttr("variant", builder.getStringAttr("n - i"));

  DiagnosticCapture diag(env.ctx);
  auto result = runPasses(env.module);
  EXPECT_TRUE(result.succeeded());

  // User-provided assigns should not be overwritten.
  auto assignsAttr = loopOp->getAttrOfType<mlir::StringAttr>("assigns");
  ASSERT_TRUE(assignsAttr != nullptr);
  EXPECT_EQ(assignsAttr.getValue().str(), "user_provided");
}

TEST(PassesTest, LoopContractPassWarnsOnMissingInvariant) {
  LoopTestModule env;
  mlir::OpBuilder builder(&env.ctx);
  env.module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto& funcBlock = createFuncWithBlock(builder, env);

  // For-loop (with init) without invariant. Variant not required for for-loops.
  createLoopWithAssignsInBody(builder, env, funcBlock, /*withInitRegion=*/true);

  DiagnosticCapture diag(env.ctx);
  auto result = runPasses(env.module);
  EXPECT_TRUE(result.succeeded());

  // Should emit a warning about missing invariant (may also emit variant
  // auto-inference warning).
  ASSERT_FALSE(diag.warnings.empty());
  bool foundInvariantWarning = false;
  for (const auto& w : diag.warnings) {
    if (w.find("loop_invariant") != std::string::npos) {
      foundInvariantWarning = true;
    }
  }
  EXPECT_TRUE(foundInvariantWarning);
}

TEST(PassesTest, LoopContractPassErrorsOnWhileWithoutVariant) {
  LoopTestModule env;
  mlir::OpBuilder builder(&env.ctx);
  env.module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto& funcBlock = createFuncWithBlock(builder, env);

  // While-loop (no init region) without variant attribute.
  auto loopOp = createLoopWithAssignsInBody(builder, env, funcBlock,
                                            /*withInitRegion=*/false);
  loopOp->setAttr("invariant", builder.getStringAttr("true"));
  // No variant attribute set.

  DiagnosticCapture diag(env.ctx);
  auto result = runPasses(env.module);
  EXPECT_TRUE(result.failed());

  // Should emit an error about missing variant.
  ASSERT_FALSE(diag.errors.empty());
  EXPECT_NE(diag.errors[0].find("loop_variant"), std::string::npos);
}

TEST(PassesTest, LoopContractPassAllowsForLoopWithoutVariant) {
  LoopTestModule env;
  mlir::OpBuilder builder(&env.ctx);
  env.module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto& funcBlock = createFuncWithBlock(builder, env);

  // For-loop (with init region) without variant -- should succeed.
  auto loopOp = createLoopWithAssignsInBody(builder, env, funcBlock,
                                            /*withInitRegion=*/true);
  loopOp->setAttr("invariant", builder.getStringAttr("true"));
  // No variant attribute, but for-loop so this is OK.

  DiagnosticCapture diag(env.ctx);
  auto result = runPasses(env.module);
  EXPECT_TRUE(result.succeeded());
  EXPECT_TRUE(diag.errors.empty());
}

// [TC-8] Test: auto-compute assigns finds variables in both body and update.
TEST(PassesTest, LoopContractPassAutoComputesAssignsFromUpdateRegion) {
  LoopTestModule env;
  mlir::OpBuilder builder(&env.ctx);
  env.module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto i32Ty = arc::IntType::get(&env.ctx, 32, true);
  auto& funcBlock = createFuncWithBlock(builder, env);

  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(&funcBlock);

  auto initConst =
      builder.create<arc::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(0));
  auto varX = builder.create<arc::VarOp>(loc, i32Ty, "x", initConst);
  auto varZ = builder.create<arc::VarOp>(loc, i32Ty, "z", initConst);
  auto newVal =
      builder.create<arc::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(1));

  auto loopOp = builder.create<arc::LoopOp>(loc);
  loopOp->setAttr("invariant", builder.getStringAttr("true"));
  loopOp->setAttr("variant", builder.getStringAttr("n - i"));

  populateCondRegion(builder, loopOp, env.ctx);

  // Body region: assign to x
  {
    auto& bodyBlock = loopOp.getBodyRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&bodyBlock);
    builder.create<arc::AssignOp>(loc, varX.getResult(), newVal);
    builder.create<arc::YieldOp>(loc);
  }

  // Update region: assign to z
  {
    auto& updateBlock = loopOp.getUpdateRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&updateBlock);
    builder.create<arc::AssignOp>(loc, varZ.getResult(), newVal);
    builder.create<arc::YieldOp>(loc);
  }

  // Init region (makes it a for-loop)
  populateInitRegion(builder, loopOp);

  builder.setInsertionPointAfter(loopOp);
  builder.create<arc::ReturnOp>(loc, funcBlock.getArgument(0));

  DiagnosticCapture diag(env.ctx);
  auto result = runPasses(env.module);
  EXPECT_TRUE(result.succeeded());

  auto assignsAttr = loopOp->getAttrOfType<mlir::StringAttr>("assigns");
  ASSERT_TRUE(assignsAttr != nullptr);

  auto assignsStr = assignsAttr.getValue().str();
  EXPECT_NE(assignsStr.find("x"), std::string::npos)
      << "Should find 'x' from body region";
  EXPECT_NE(assignsStr.find("z"), std::string::npos)
      << "Should find 'z' from update region";
}

} // namespace
} // namespace arcanum
