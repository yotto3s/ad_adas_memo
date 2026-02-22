#ifndef ARCANUM_DIAGNOSTICTRACKER_H
#define ARCANUM_DIAGNOSTICTRACKER_H

namespace arcanum {

/// Simple counter tracking how many fallback substitutions occurred during
/// lowering.  When non-zero, the report generator includes a warning in
/// stdout so that users piping only stdout are aware the results may be
/// based on fabricated values.
struct DiagnosticTracker {
  static inline int fallbackCount = 0;

  static void reset() { fallbackCount = 0; }
  static void recordFallback() { ++fallbackCount; }
  static int getFallbackCount() { return fallbackCount; }
};

} // namespace arcanum

#endif // ARCANUM_DIAGNOSTICTRACKER_H
