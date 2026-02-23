//===- ArcTypes.td - Arc type definitions -----------------*- tablegen -*-===//
//
// Defines the types for the Arc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_TYPES_TD
#define ARC_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "arcanum/dialect/ArcDialect.td"

//===----------------------------------------------------------------------===//
// Arc type definitions
//===----------------------------------------------------------------------===//

class Arc_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Arc_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Arc_IntType : Arc_Type<"Int", "int"> {
  let summary = "Fixed-width integer type";
  let description = [{
    Parameterized integer type supporting i8/i16/i32/i64 (signed)
    and u8/u16/u32/u64 (unsigned). Arithmetic on signed types
    generates overflow proof obligations in trap mode.
  }];
  let parameters = (ins "unsigned":$width, "bool":$isSigned);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    llvm::APInt getMinValue() const;
    llvm::APInt getMaxValue() const;
  }];
}

def Arc_BoolType : Arc_Type<"Bool", "bool"> {
  let summary = "Boolean type";
  let description = [{
    Represents bool in the Arc dialect.
  }];
}

#endif // ARC_TYPES_TD
