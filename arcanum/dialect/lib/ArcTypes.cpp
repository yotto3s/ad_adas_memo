#include "arcanum/dialect/ArcTypes.h"
#include "arcanum/dialect/ArcDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace arcanum::arc;

#define GET_TYPEDEF_CLASSES
#include "arcanum/dialect/ArcTypes.cpp.inc"

// Suppress unused-function warnings for TableGen-generated helpers that are
// only used when useDefaultTypePrinterParser = 1.  We use custom dialect-level
// parseType/printType instead.
[[maybe_unused]] static auto* unusedParser_ = &generatedTypeParser;
[[maybe_unused]] static auto* unusedPrinter_ = &generatedTypePrinter;

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "arcanum/dialect/ArcTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// IntType custom assembly format
//===----------------------------------------------------------------------===//

void IntType::print(mlir::AsmPrinter& printer) const {
  // Print as <width, isSigned> for the verbose/generic form.
  // The dialect-level printType handles the short mnemonic form.
  printer << "<" << getWidth() << ", " << (getIsSigned() ? "true" : "false")
          << ">";
}

mlir::Type IntType::parse(mlir::AsmParser& parser) {
  unsigned width;
  bool isSigned;
  if (parser.parseLess() || parser.parseInteger(width) || parser.parseComma()) {
    return {};
  }
  // Parse "true" or "false" for isSigned
  llvm::StringRef signedStr;
  if (parser.parseKeyword(&signedStr)) {
    return {};
  }
  if (signedStr == "true") {
    isSigned = true;
  } else if (signedStr == "false") {
    isSigned = false;
  } else {
    parser.emitError(parser.getCurrentLocation(),
                     "expected 'true' or 'false' for isSigned");
    return {};
  }
  if (parser.parseGreater()) {
    return {};
  }
  // Validate width
  if (width != 8 && width != 16 && width != 32 && width != 64) {
    parser.emitError(parser.getCurrentLocation(),
                     "unsupported integer width; expected 8, 16, 32, or 64");
    return {};
  }
  return IntType::get(parser.getContext(), width, isSigned);
}

//===----------------------------------------------------------------------===//
// IntType min/max value helpers
//===----------------------------------------------------------------------===//

llvm::APInt IntType::getMinValue() const {
  if (getIsSigned()) {
    return llvm::APInt::getSignedMinValue(getWidth());
  }
  return llvm::APInt(getWidth(), 0);
}

llvm::APInt IntType::getMaxValue() const {
  if (getIsSigned()) {
    return llvm::APInt::getSignedMaxValue(getWidth());
  }
  return llvm::APInt::getMaxValue(getWidth());
}
