#include "arcanum/passes/Passes.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(PassesTest, SucceedsOnValidModule) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<arc::ArcDialect>();
  mlir::OpBuilder builder(&ctx);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create a valid arc.func
  builder.setInsertionPointToEnd(module.getBody());
  auto funcType = builder.getFunctionType({arc::I32Type::get(&ctx)},
                                          {arc::I32Type::get(&ctx)});
  auto funcOp = builder.create<arc::FuncOp>(
      builder.getUnknownLoc(), builder.getStringAttr("test_func"),
      mlir::TypeAttr::get(funcType), mlir::StringAttr(), mlir::StringAttr());
  auto& block = funcOp.getBody().emplaceBlock();
  block.addArgument(arc::I32Type::get(&ctx), builder.getUnknownLoc());

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
  auto funcType = builder.getFunctionType({arc::I32Type::get(&ctx)},
                                          {arc::I32Type::get(&ctx)});
  auto funcOp = builder.create<arc::FuncOp>(
      builder.getUnknownLoc(), builder.getStringAttr("bad_func"),
      mlir::TypeAttr::get(funcType), mlir::StringAttr(), mlir::StringAttr());
  auto& block = funcOp.getBody().emplaceBlock();
  block.addArgument(arc::I32Type::get(&ctx), builder.getUnknownLoc());
  // Intentionally leave the block without a terminator

  EXPECT_TRUE(mlir::failed(mlir::verify(module)));

  module->destroy();
}

} // namespace
} // namespace arcanum
