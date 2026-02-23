#ifndef ARCANUM_EXITCODES_H
#define ARCANUM_EXITCODES_H

namespace arcanum {

/// Centralized exit code definitions for the arcanum CLI.
/// Each code represents a distinct failure category.
/// Exit code 2 is intentionally skipped: it is conventionally reserved
/// for usage/command-line errors by many Unix tools (e.g., grep, diff).
enum class ExitCode : int {
  Success = 0,            // All verification obligations proven
  VerificationFailed = 1, // One or more obligations not proven
  SubsetViolation = 3,    // Input violates the allowed C++ subset
  ParseError = 4,         // Clang failed to parse the input file
  InternalError = 5, // Internal pipeline error (lowering, emission, etc.)
};

} // namespace arcanum

#endif // ARCANUM_EXITCODES_H
