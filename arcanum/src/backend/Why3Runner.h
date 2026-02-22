#ifndef ARCANUM_BACKEND_WHY3RUNNER_H
#define ARCANUM_BACKEND_WHY3RUNNER_H

#include <chrono>
#include <string>
#include <vector>

namespace arcanum {

enum class ObligationStatus {
  Valid,
  Unknown,
  Timeout,
  Failure,
};

struct ObligationResult {
  std::string name;
  ObligationStatus status = ObligationStatus::Unknown;
  std::chrono::milliseconds duration{0};
  /// Name of the function this obligation belongs to.
  /// Populated when multi-function reporting is implemented; empty for now.
  /// This field is a data model preparation for grouping obligations by
  /// function in future slices (see CR-8).
  std::string functionName;
};

/// Run Why3 on a .mlw file with the given solver and timeout.
/// Returns per-obligation results parsed from Why3 stdout.
std::vector<ObligationResult> runWhy3(const std::string& mlwPath,
                                      const std::string& why3Binary = "why3",
                                      int timeoutSeconds = 30);

/// Parse Why3 stdout output into obligation results (exposed for testing).
std::vector<ObligationResult> parseWhy3Output(const std::string& output);

} // namespace arcanum

#endif // ARCANUM_BACKEND_WHY3RUNNER_H
