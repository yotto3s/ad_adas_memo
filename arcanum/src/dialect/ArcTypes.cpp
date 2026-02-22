#include "dialect/ArcTypes.h"
#include "dialect/ArcDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace arcanum::arc;

#define GET_TYPEDEF_CLASSES
#include "dialect/ArcTypes.cpp.inc"

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "dialect/ArcTypes.h.inc"
      >();
}
