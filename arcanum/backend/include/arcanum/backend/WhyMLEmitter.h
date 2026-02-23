#ifndef ARCANUM_BACKEND_WHYMLEMITTER_H
#define ARCANUM_BACKEND_WHYMLEMITTER_H

#include "mlir/IR/BuiltinOps.h"

#include <map>
#include <optional>
#include <string>

namespace arcanum {

/// Source location mapping: maps WhyML construct identifiers back to
/// original C++ source locations.
struct LocationEntry {
  std::string functionName;
  std::string fileName;
  unsigned line = 0;
};

struct WhyMLResult {
  std::string whymlText; // The generated WhyML source
  std::string filePath;  // Path to temporary .mlw file
  std::map<std::string, LocationEntry> locationMap;
  /// Maps WhyML module names (CamelCase) back to original C++ function names.
  /// Used by Why3Runner to attribute proof obligations to source functions.
  std::map<std::string, std::string> moduleToFuncMap;
};

/// Emit WhyML from an Arc MLIR module. Writes the .mlw file to a temp path.
///
/// Error contract: Returns std::nullopt on failure (empty module, I/O
/// error).  Unmapped MLIR values emit "?unknown?" with a warning,
/// producing invalid WhyML that Why3 will reject at parse time.
[[nodiscard]] std::optional<WhyMLResult> emitWhyML(mlir::ModuleOp module);

} // namespace arcanum

#endif // ARCANUM_BACKEND_WHYMLEMITTER_H
