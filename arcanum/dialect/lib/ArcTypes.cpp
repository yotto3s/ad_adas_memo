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
// IntType construction validation (defense-in-depth for CR-3)
//===----------------------------------------------------------------------===//

static bool isValidIntWidth(unsigned width) {
  return width == 8 || width == 16 || width == 32 || width == 64;
}

mlir::LogicalResult
IntType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                unsigned width, bool /*isSigned*/) {
  if (!isValidIntWidth(width)) {
    return emitError() << "unsupported IntType width " << width
                       << "; expected 8, 16, 32, or 64";
  }
  return mlir::success();
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

static std::optional<bool> parseSignednessKeyword(mlir::AsmParser& parser) {
  llvm::StringRef signedStr;
  if (parser.parseKeyword(&signedStr)) {
    return std::nullopt;
  }
  if (signedStr == "true") {
    return true;
  }
  if (signedStr == "false") {
    return false;
  }
  parser.emitError(parser.getCurrentLocation(),
                   "expected 'true' or 'false' for isSigned");
  return std::nullopt;
}

mlir::Type IntType::parse(mlir::AsmParser& parser) {
  unsigned width;
  if (parser.parseLess() || parser.parseInteger(width) || parser.parseComma()) {
    return {};
  }
  auto isSigned = parseSignednessKeyword(parser);
  if (!isSigned) {
    return {};
  }
  if (parser.parseGreater()) {
    return {};
  }
  if (!isValidIntWidth(width)) {
    parser.emitError(parser.getCurrentLocation(),
                     "unsupported integer width; expected 8, 16, 32, or 64");
    return {};
  }
  return IntType::get(parser.getContext(), width, *isSigned);
}

//===----------------------------------------------------------------------===//
// IntType min/max value helpers
//===----------------------------------------------------------------------===//

llvm::APInt IntType::getMinValue() const {
  if (getIsSigned()) {
    return llvm::APInt::getSignedMinValue(getWidth());
  }
  return {getWidth(), 0};
}

llvm::APInt IntType::getMaxValue() const {
  if (getIsSigned()) {
    return llvm::APInt::getSignedMaxValue(getWidth());
  }
  return llvm::APInt::getMaxValue(getWidth());
}
