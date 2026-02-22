#ifndef ARCANUM_DIALECT_ARCOPS_H
#define ARCANUM_DIALECT_ARCOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "dialect/ArcDialect.h"
#include "dialect/ArcTypes.h"

#define GET_OP_CLASSES
#include "dialect/ArcOps.h.inc"

#endif // ARCANUM_DIALECT_ARCOPS_H
