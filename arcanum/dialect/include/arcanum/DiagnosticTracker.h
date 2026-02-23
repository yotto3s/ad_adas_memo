#ifndef ARCANUM_DIAGNOSTICTRACKER_H
#define ARCANUM_DIAGNOSTICTRACKER_H

#include <atomic>

namespace arcanum {

/// Simple counter tracking how many fallback substitutions occurred during
/// lowering.  When non-zero, the pipeline aborts with a non-zero exit code
/// so that users are aware the results may be based on fabricated values.
/// Uses std::atomic for thread safety.
struct DiagnosticTracker {
  static inline std::atomic<int> fallbackCount{0};

  static void reset() noexcept {
    fallbackCount.store(0, std::memory_order_relaxed);
  }
  static void recordFallback() noexcept {
    fallbackCount.fetch_add(1, std::memory_order_relaxed);
  }
  [[nodiscard]] static int getFallbackCount() noexcept {
    return fallbackCount.load(std::memory_order_relaxed);
  }
};

} // namespace arcanum

#endif // ARCANUM_DIAGNOSTICTRACKER_H
