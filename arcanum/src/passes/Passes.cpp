#include "passes/Passes.h"

#include "mlir/IR/Verifier.h"

namespace arcanum {

mlir::LogicalResult runPasses(mlir::ModuleOp module) {
  // Slice 1: identity pass-through.
  // Just run the MLIR verifier to catch malformed IR.
  // The PassManager's built-in verifier runs automatically after each pass.
  // Since we have no passes in Slice 1, just verify the module manually.
  if (mlir::verify(module).failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

} // namespace arcanum
