#include "arcanum/dialect/ArcTypes.h"
#include "arcanum/dialect/ArcDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace arcanum::arc;

#define GET_TYPEDEF_CLASSES
#include "arcanum/dialect/ArcTypes.cpp.inc"

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "arcanum/dialect/ArcTypes.cpp.inc"
      >();
}
