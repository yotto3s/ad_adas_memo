#ifndef ARCANUM_DIALECT_LOWERING_H
#define ARCANUM_DIALECT_LOWERING_H

#include "frontend/ContractParser.h"
#include "clang/AST/ASTContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <map>
#include <memory>

namespace arcanum {

/// Lower a Clang ASTContext + parsed contracts into an Arc MLIR ModuleOp.
mlir::OwningOpRef<mlir::ModuleOp> lowerToArc(
    mlir::MLIRContext& context,
    clang::ASTContext& astContext,
    const std::map<const clang::FunctionDecl*, ContractInfo>& contracts);

} // namespace arcanum

#endif // ARCANUM_DIALECT_LOWERING_H
