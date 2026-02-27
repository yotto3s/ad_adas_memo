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

namespace {
struct IntTypeInfo {
  unsigned width;
  bool isSigned;
};
} // namespace

static std::optional<IntTypeInfo> resolveIntMnemonic(llvm::StringRef keyword) {
  return llvm::StringSwitch<std::optional<IntTypeInfo>>(keyword)
      .Case("i8", IntTypeInfo{.width = 8, .isSigned = true})
      .Case("i16", IntTypeInfo{.width = 16, .isSigned = true})
      .Case("i32", IntTypeInfo{.width = 32, .isSigned = true})
      .Case("i64", IntTypeInfo{.width = 64, .isSigned = true})
      .Case("u8", IntTypeInfo{.width = 8, .isSigned = false})
      .Case("u16", IntTypeInfo{.width = 16, .isSigned = false})
      .Case("u32", IntTypeInfo{.width = 32, .isSigned = false})
      .Case("u64", IntTypeInfo{.width = 64, .isSigned = false})
      .Default(std::nullopt);
}

mlir::Type ArcDialect::parseType(mlir::DialectAsmParser& parser) const {
  llvm::StringRef keyword;
  mlir::SMLoc keywordLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword)) {
    return {};
  }

  // Handle short integer mnemonics: i8, i16, i32, i64, u8, u16, u32, u64
  if (auto info = resolveIntMnemonic(keyword)) {
    return IntType::get(getContext(), info->width, info->isSigned);
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

static void printIntTypeMnemonic(mlir::DialectAsmPrinter& p, IntType t) {
  p << (t.getIsSigned() ? "i" : "u") << t.getWidth();
}

static void printBoolTypeKeyword(mlir::DialectAsmPrinter& p) { p << "bool"; }

void ArcDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter& printer) const {
  if (auto intType = llvm::dyn_cast<IntType>(type)) {
    printIntTypeMnemonic(printer, intType);
    return;
  }
  if (llvm::isa<BoolType>(type)) {
    printBoolTypeKeyword(printer);
    return;
  }
  // CQ-10: llvm_unreachable is acceptable here because the Arc dialect has a
  // closed type system (IntType, BoolType) and this code path is unreachable
  // for well-formed IR.  MLIR dialects conventionally use llvm_unreachable
  // for exhaustive type dispatch.
  llvm_unreachable("unknown Arc type in printType");
}
