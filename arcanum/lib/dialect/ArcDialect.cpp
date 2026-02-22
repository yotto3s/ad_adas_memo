#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

using namespace arcanum::arc;

#include "dialect/ArcDialect.cpp.inc"

void ArcDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "dialect/ArcOps.cpp.inc"
      >();
}
