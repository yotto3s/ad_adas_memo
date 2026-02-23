//===- ArcOps.td - Arc operation definitions ---------------*- tablegen -*-===//
//
// Defines the operations for the Arc dialect (Slice 1).
//
//===----------------------------------------------------------------------===//

#ifndef ARC_OPS_TD
#define ARC_OPS_TD

include "arcanum/dialect/ArcDialect.td"
include "arcanum/dialect/ArcTypes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Function operation
//===----------------------------------------------------------------------===//

def Arc_FuncOp : Arc_Op<"func", [
    Symbol,
    IsolatedFromAbove
]> {
  let summary = "Function with verification contracts";
  let description = [{
    Represents a verified function with optional requires/ensures attributes.
    The body is a single region.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$requires_attr,
    OptionalAttr<StrAttr>:$ensures_attr
  );
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Constant operations
//===----------------------------------------------------------------------===//

def Arc_ConstantOp : Arc_Op<"constant", [Pure]> {
  let summary = "Integer or boolean constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Arithmetic operations (overflow-checked in trap mode)
//===----------------------------------------------------------------------===//

def Arc_AddOp : Arc_Op<"add", [Pure]> {
  let summary = "Checked integer addition";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Arc_SubOp : Arc_Op<"sub", [Pure]> {
  let summary = "Checked integer subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Arc_MulOp : Arc_Op<"mul", [Pure]> {
  let summary = "Checked integer multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Arc_DivOp : Arc_Op<"div", [Pure]> {
  let summary = "Checked integer division";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Arc_RemOp : Arc_Op<"rem", [Pure]> {
  let summary = "Checked integer remainder";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Comparison operation
//===----------------------------------------------------------------------===//

def Arc_CmpOp : Arc_Op<"cmp", [Pure]> {
  let summary = "Integer comparison";
  let arguments = (ins StrAttr:$predicate, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Logical operations
//===----------------------------------------------------------------------===//

def Arc_AndOp : Arc_Op<"and", [Pure]> {
  let summary = "Logical AND";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Arc_OrOp : Arc_Op<"or", [Pure]> {
  let summary = "Logical OR";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Arc_NotOp : Arc_Op<"not", [Pure]> {
  let summary = "Logical NOT";
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Variable operations
//===----------------------------------------------------------------------===//

def Arc_VarOp : Arc_Op<"var"> {
  let summary = "Local variable declaration";
  let arguments = (ins StrAttr:$name, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Arc_AssignOp : Arc_Op<"assign"> {
  let summary = "Variable assignment";
  let arguments = (ins AnyType:$target, AnyType:$value);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

def Arc_ReturnOp : Arc_Op<"return", [Terminator]> {
  let summary = "Function return";
  let arguments = (ins Optional<AnyType>:$value);
  let hasCustomAssemblyFormat = 1;
}

def Arc_IfOp : Arc_Op<"if"> {
  let summary = "Conditional branch";
  let arguments = (ins AnyType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
  let results = (outs Optional<AnyType>:$result);
  let hasCustomAssemblyFormat = 1;
}

#endif // ARC_OPS_TD
