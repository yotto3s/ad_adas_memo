#ifndef ARCANUM_DIALECT_ARCOPS_H
#define ARCANUM_DIALECT_ARCOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcTypes.h"

#define GET_OP_CLASSES
#include "arcanum/dialect/ArcOps.h.inc"

#endif // ARCANUM_DIALECT_ARCOPS_H
