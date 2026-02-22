#ifndef ARCANUM_EXITCODES_H
#define ARCANUM_EXITCODES_H

namespace arcanum {

/// Centralized exit code definitions for the arcanum CLI.
/// Each code represents a distinct failure category.
enum ExitCode {
  ExitSuccess = 0,            // All verification obligations proven
  ExitVerificationFailed = 1, // One or more obligations not proven
  ExitSubsetViolation = 3,    // Input violates the allowed C++ subset
  ExitParseError = 4,         // Clang failed to parse the input file
  ExitInternalError = 5, // Internal pipeline error (lowering, emission, etc.)
};

} // namespace arcanum

#endif // ARCANUM_EXITCODES_H
