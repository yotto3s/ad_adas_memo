#ifndef ARCANUM_PASSES_PASSES_H
#define ARCANUM_PASSES_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace arcanum {

/// Run MLIR passes on the Arc module.
/// Slice 1: identity pass-through + MLIR verifier only.
mlir::LogicalResult runPasses(mlir::ModuleOp module);

} // namespace arcanum

#endif // ARCANUM_PASSES_PASSES_H
