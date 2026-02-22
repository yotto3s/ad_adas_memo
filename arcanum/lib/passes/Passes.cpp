#include "passes/Passes.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

namespace arcanum {

mlir::LogicalResult runPasses(mlir::ModuleOp module) {
  // Set up PassManager infrastructure for future slices.
  // Slice 1: identity pass-through with built-in verifier.
  mlir::PassManager pm(module->getContext());

  // The PassManager's built-in verifier runs automatically after each pass.
  // No optimization passes are added for Slice 1.
  if (pm.run(module).failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

} // namespace arcanum
