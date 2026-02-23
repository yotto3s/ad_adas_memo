//===- ArcDialect.td - Arc dialect definition ----------*- tablegen -*-===//
//
// Defines the Arc dialect for Arcanum formal verification.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_DIALECT_TD
#define ARC_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Arc dialect definition
//===----------------------------------------------------------------------===//

def Arc_Dialect : Dialect {
  let name = "arc";
  let summary = "Arc dialect for formal verification of C++ programs";
  let description = [{
    The Arc dialect provides operations and types for representing
    verified C++ programs. It captures verification-relevant semantics
    (contracts, overflow checking, etc.) in a small set of MLIR operations
    that translate nearly 1:1 to WhyML.
  }];
  let cppNamespace = "::arcanum::arc";

  let useDefaultTypePrinterParser = 0;
  let extraClassDeclaration = [{
    void registerTypes();
    /// Custom type parsing to support short mnemonics (!arc.i32, !arc.u8, etc.)
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    /// Custom type printing to emit short mnemonics
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

//===----------------------------------------------------------------------===//
// Base Arc operation definition
//===----------------------------------------------------------------------===//

class Arc_Op<string mnemonic, list<Trait> traits = []> :
    Op<Arc_Dialect, mnemonic, traits>;

#endif // ARC_DIALECT_TD
