#ifndef ARCANUM_BACKEND_WHY3RUNNER_H
#define ARCANUM_BACKEND_WHY3RUNNER_H

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace arcanum {

/// Default per-obligation timeout in seconds for Why3 solver invocations.
constexpr int DEFAULT_TIMEOUT_SECONDS = 30;

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
  /// Populated by parseWhy3Output() from Why3's Theory context lines.
  /// The module name (CamelCase) is converted back to the original
  /// snake_case function name for matching against the location map.
  std::string functionName;
};

/// Run Why3 on a .mlw file with the given solver and timeout.
/// Returns per-obligation results parsed from Why3 stdout.
/// The moduleToFuncMap maps WhyML module names to original C++ function names,
/// used to attribute proof obligations to source functions.
[[nodiscard]] std::vector<ObligationResult>
runWhy3(const std::string& mlwPath,
        const std::map<std::string, std::string>& moduleToFuncMap = {},
        const std::string& why3Binary = "why3",
        int timeoutSeconds = DEFAULT_TIMEOUT_SECONDS);

/// Parse Why3 stdout output into obligation results (exposed for testing).
/// The moduleToFuncMap maps WhyML module names to original C++ function names.
[[nodiscard]] std::vector<ObligationResult>
parseWhy3Output(const std::string& output,
                const std::map<std::string, std::string>& moduleToFuncMap = {});

} // namespace arcanum

#endif // ARCANUM_BACKEND_WHY3RUNNER_H
