//===- ArcTypes.td - Arc type definitions -----------------*- tablegen -*-===//
//
// Defines the types for the Arc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_TYPES_TD
#define ARC_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "dialect/ArcDialect.td"

//===----------------------------------------------------------------------===//
// Arc type definitions
//===----------------------------------------------------------------------===//

class Arc_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Arc_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Arc_I32Type : Arc_Type<"I32", "i32"> {
  let summary = "32-bit signed integer type";
  let description = [{
    Represents int32_t in the Arc dialect. Arithmetic on this type
    generates overflow proof obligations in trap mode.
  }];
}

def Arc_BoolType : Arc_Type<"Bool", "bool"> {
  let summary = "Boolean type";
  let description = [{
    Represents bool in the Arc dialect.
  }];
}

#endif // ARC_TYPES_TD
