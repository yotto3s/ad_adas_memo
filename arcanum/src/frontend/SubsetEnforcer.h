#ifndef ARCANUM_FRONTEND_SUBSETENFORCER_H
#define ARCANUM_FRONTEND_SUBSETENFORCER_H

#include "clang/AST/ASTContext.h"
#include <string>
#include <vector>

namespace arcanum {

struct SubsetResult {
  bool passed = true;
  std::vector<std::string> diagnostics;
};

/// Walk the Clang AST and reject any constructs outside the Slice 1 subset.
/// Allowed: int32_t, bool, non-template non-recursive functions with single
/// return, variable declarations, assignments, if/else, return,
/// +, -, *, /, %, comparisons, &&, ||, !.
SubsetResult enforceSubset(clang::ASTContext& context);

} // namespace arcanum

#endif // ARCANUM_FRONTEND_SUBSETENFORCER_H
