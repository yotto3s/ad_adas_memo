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
};

/// Emit WhyML from an Arc MLIR module. Writes the .mlw file to a temp path.
std::optional<WhyMLResult> emitWhyML(mlir::ModuleOp module);

} // namespace arcanum

#endif // ARCANUM_BACKEND_WHYMLEMITTER_H
