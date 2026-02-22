#include "dialect/ArcOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace arcanum::arc;

#define GET_OP_CLASSES
#include "dialect/ArcOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
  // Minimal custom parsing for Slice 1 — will be refined
  // For now, delegate to simple attribute-based format
  return mlir::failure(); // TODO: implement full parser
}

void FuncOp::print(mlir::OpAsmPrinter& printer) {
  printer << " @" << getSymName();
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"sym_name", "function_type"});
  printer << " ";
  printer.printRegion(getBody());
}

//===----------------------------------------------------------------------===//
// ConstantOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser& parser,
                                    mlir::OperationState& result) {
  return mlir::failure(); // TODO: implement full parser
}

void ConstantOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getValue();
  printer << " : " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// IfOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult IfOp::parse(mlir::OpAsmParser& parser,
                              mlir::OperationState& result) {
  return mlir::failure(); // TODO: implement full parser
}

void IfOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getCondition();
  printer << " ";
  printer.printRegion(getThenRegion());
  if (!getElseRegion().empty()) {
    printer << " else ";
    printer.printRegion(getElseRegion());
  }
}
