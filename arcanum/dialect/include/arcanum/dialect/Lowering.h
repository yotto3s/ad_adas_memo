#ifndef ARCANUM_DIALECT_LOWERING_H
#define ARCANUM_DIALECT_LOWERING_H

#include "arcanum/frontend/ContractParser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/AST/ASTContext.h"

#include <map>
#include <memory>

namespace arcanum {

/// Lower a Clang ASTContext + parsed contracts into an Arc MLIR ModuleOp.
///
/// Error contract: Returns a valid ModuleOp.  When an expression cannot
/// be lowered, lowerExpr returns std::nullopt and the containing
/// statement is skipped (no zero-constant fallback is emitted).  The
/// caller must check DiagnosticTracker::getFallbackCount() after this
/// call to detect incomplete lowering.  Returns nullptr only on fatal
/// infrastructure errors.
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
lowerToArc(mlir::MLIRContext& context, clang::ASTContext& astContext,
           const std::map<const clang::FunctionDecl*, ContractInfo>& contracts);

} // namespace arcanum

#endif // ARCANUM_DIALECT_LOWERING_H
