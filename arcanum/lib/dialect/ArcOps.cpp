#include "arcanum/dialect/ArcOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace arcanum::arc;

#define GET_OP_CLASSES
#include "arcanum/dialect/ArcOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Helper for binary op custom assembly (print/parse)
//===----------------------------------------------------------------------===//

namespace {

void printBinaryOp(mlir::OpAsmPrinter& printer, mlir::Operation* op) {
  printer << " " << op->getOperand(0) << ", " << op->getOperand(1);
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op->getResult(0).getType();
}

mlir::ParseResult parseBinaryOp(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
  mlir::OpAsmParser::UnresolvedOperand lhs;
  mlir::OpAsmParser::UnresolvedOperand rhs;
  mlir::Type type;
  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(lhs, type, result.operands) ||
      parser.resolveOperand(rhs, type, result.operands)) {
    return mlir::failure();
  }
  result.addTypes(type);
  return mlir::success();
}

} // namespace

//===----------------------------------------------------------------------===//
// FuncOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
  // Parse: arc.func @name { <optional attrs> } <region>
  mlir::StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, "sym_name", result.attributes)) {
    return mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }
  auto* body = result.addRegion();
  if (parser.parseRegion(*body)) {
    return mlir::failure();
  }
  // function_type is required; if absent from the parsed attrs, this will
  // fail MLIR verification, which is acceptable for round-trip testing.
  return mlir::success();
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
  // Parse: arc.constant <value> : <type>
  mlir::Attribute valueAttr;
  mlir::Type type;
  if (parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.parseColonType(type)) {
    return mlir::failure();
  }
  result.addTypes(type);
  return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getValue();
  printer << " : " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// Arithmetic ops custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult AddOp::parse(mlir::OpAsmParser& p, mlir::OperationState& r) {
  return parseBinaryOp(p, r);
}
void AddOp::print(mlir::OpAsmPrinter& p) { printBinaryOp(p, *this); }

mlir::ParseResult SubOp::parse(mlir::OpAsmParser& p, mlir::OperationState& r) {
  return parseBinaryOp(p, r);
}
void SubOp::print(mlir::OpAsmPrinter& p) { printBinaryOp(p, *this); }

mlir::ParseResult MulOp::parse(mlir::OpAsmParser& p, mlir::OperationState& r) {
  return parseBinaryOp(p, r);
}
void MulOp::print(mlir::OpAsmPrinter& p) { printBinaryOp(p, *this); }

mlir::ParseResult DivOp::parse(mlir::OpAsmParser& p, mlir::OperationState& r) {
  return parseBinaryOp(p, r);
}
void DivOp::print(mlir::OpAsmPrinter& p) { printBinaryOp(p, *this); }

mlir::ParseResult RemOp::parse(mlir::OpAsmParser& p, mlir::OperationState& r) {
  return parseBinaryOp(p, r);
}
void RemOp::print(mlir::OpAsmPrinter& p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// CmpOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult CmpOp::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
  // TODO: implement full parser.  Round-trip testing (print -> parse -> print)
  // is deferred to a future slice.  Currently only print() is exercised.
  return mlir::failure();
}

void CmpOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getPredicate() << ", " << getLhs() << ", " << getRhs();
  printer.printOptionalAttrDict((*this)->getAttrs(), {"predicate"});
  printer << " : " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// Logical ops custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult AndOp::parse(mlir::OpAsmParser& p, mlir::OperationState& r) {
  return parseBinaryOp(p, r);
}
void AndOp::print(mlir::OpAsmPrinter& p) { printBinaryOp(p, *this); }

mlir::ParseResult OrOp::parse(mlir::OpAsmParser& p, mlir::OperationState& r) {
  return parseBinaryOp(p, r);
}
void OrOp::print(mlir::OpAsmPrinter& p) { printBinaryOp(p, *this); }

mlir::ParseResult NotOp::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
  mlir::OpAsmParser::UnresolvedOperand operand;
  mlir::Type type;
  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(operand, type, result.operands)) {
    return mlir::failure();
  }
  result.addTypes(type);
  return mlir::success();
}

void NotOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getOperand();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// VarOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult VarOp::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
  // TODO: implement full parser.  Round-trip testing is deferred to a
  // future slice.  Currently only print() is exercised.
  return mlir::failure();
}

void VarOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getName() << " = " << getInit();
  printer.printOptionalAttrDict((*this)->getAttrs(), {"name"});
  printer << " : " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// AssignOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult AssignOp::parse(mlir::OpAsmParser& parser,
                                  mlir::OperationState& result) {
  // TODO: implement full parser.  Round-trip testing is deferred to a
  // future slice.  Currently only print() is exercised.
  return mlir::failure();
}

void AssignOp::print(mlir::OpAsmPrinter& printer) {
  printer << " " << getTarget() << " = " << getValue();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getValue().getType();
}

//===----------------------------------------------------------------------===//
// ReturnOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult ReturnOp::parse(mlir::OpAsmParser& parser,
                                  mlir::OperationState& result) {
  mlir::OpAsmParser::UnresolvedOperand operand;
  mlir::Type type;
  if (parser.parseOptionalOperand(operand).has_value()) {
    if (parser.parseColonType(type) ||
        parser.resolveOperand(operand, type, result.operands)) {
      return mlir::failure();
    }
  }
  return parser.parseOptionalAttrDict(result.attributes);
}

void ReturnOp::print(mlir::OpAsmPrinter& printer) {
  if (getValue()) {
    printer << " " << getValue() << " : " << getValue().getType();
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// IfOp custom assembly format
//===----------------------------------------------------------------------===//

mlir::ParseResult IfOp::parse(mlir::OpAsmParser& parser,
                              mlir::OperationState& result) {
  // TODO: implement full parser.  Round-trip testing is deferred to a
  // future slice.  Currently only print() is exercised.
  return mlir::failure();
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
