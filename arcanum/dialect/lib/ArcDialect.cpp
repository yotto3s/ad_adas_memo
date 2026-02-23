#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSwitch.h"

using namespace arcanum::arc;

#include "arcanum/dialect/ArcDialect.cpp.inc"

void ArcDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "arcanum/dialect/ArcOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Custom type parsing: supports short mnemonics like i32, u8, etc.
// Also supports the verbose form int<32, true>.
//===----------------------------------------------------------------------===//

mlir::Type ArcDialect::parseType(mlir::DialectAsmParser& parser) const {
  llvm::StringRef keyword;
  mlir::SMLoc keywordLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword)) {
    return {};
  }

  // Handle short integer mnemonics: i8, i16, i32, i64, u8, u16, u32, u64
  struct IntTypeInfo {
    unsigned width;
    bool isSigned;
  };
  auto intInfo = llvm::StringSwitch<std::optional<IntTypeInfo>>(keyword)
                     .Case("i8", IntTypeInfo{8, true})
                     .Case("i16", IntTypeInfo{16, true})
                     .Case("i32", IntTypeInfo{32, true})
                     .Case("i64", IntTypeInfo{64, true})
                     .Case("u8", IntTypeInfo{8, false})
                     .Case("u16", IntTypeInfo{16, false})
                     .Case("u32", IntTypeInfo{32, false})
                     .Case("u64", IntTypeInfo{64, false})
                     .Default(std::nullopt);

  if (intInfo) {
    return IntType::get(getContext(), intInfo->width, intInfo->isSigned);
  }

  // Handle verbose int<width, signed> form
  if (keyword == "int") {
    return IntType::parse(parser);
  }

  // Handle bool
  if (keyword == "bool") {
    return BoolType::get(getContext());
  }

  parser.emitError(keywordLoc, "unknown Arc type: ") << keyword;
  return {};
}

//===----------------------------------------------------------------------===//
// Custom type printing: emits short mnemonics
//===----------------------------------------------------------------------===//

void ArcDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter& printer) const {
  if (auto intType = llvm::dyn_cast<IntType>(type)) {
    if (intType.getIsSigned()) {
      printer << "i" << intType.getWidth();
    } else {
      printer << "u" << intType.getWidth();
    }
    return;
  }
  if (llvm::isa<BoolType>(type)) {
    printer << "bool";
    return;
  }
  llvm_unreachable("unknown Arc type in printType");
}
