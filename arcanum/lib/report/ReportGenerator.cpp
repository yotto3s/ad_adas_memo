#include "arcanum/report/ReportGenerator.h"

#include <chrono>
#include <sstream>

namespace arcanum {

Report generateReport(const std::vector<ObligationResult>& obligations,
                      const std::map<std::string, LocationEntry>& locationMap) {
  Report report;
  std::ostringstream out;
  constexpr double MS_PER_SECOND = 1000.0;

  // Group obligations by function name using the locationMap keys.
  // If obligations have functionName set, group by that. Otherwise,
  // fall back to assigning all obligations to the first function.
  struct FuncStats {
    int validCount = 0;
    int totalCount = 0;
    bool hasUnknown = false;
    bool hasTimeout = false;
    std::chrono::milliseconds totalDuration{0};
  };

  // Determine if obligations carry function name info
  bool hasPerObFuncNames = false;
  for (const auto& ob : obligations) {
    if (!ob.functionName.empty()) {
      hasPerObFuncNames = true;
      break;
    }
  }

  std::map<std::string, FuncStats> funcStatsMap;

  if (hasPerObFuncNames && locationMap.size() > 1) {
    // Group by per-obligation function name
    for (const auto& ob : obligations) {
      std::string funcName =
          ob.functionName.empty() ? "unknown" : ob.functionName;
      auto& stats = funcStatsMap[funcName];
      stats.totalCount++;
      stats.totalDuration += ob.duration;
      switch (ob.status) {
      case ObligationStatus::Valid:
        stats.validCount++;
        break;
      case ObligationStatus::Unknown:
      case ObligationStatus::Failure:
        stats.hasUnknown = true;
        break;
      case ObligationStatus::Timeout:
        stats.hasTimeout = true;
        break;
      }
    }
  } else {
    // Single function or no per-obligation function info: aggregate all
    std::string funcName = "unknown";
    if (!locationMap.empty()) {
      funcName = locationMap.begin()->first;
    }
    auto& stats = funcStatsMap[funcName];
    for (const auto& ob : obligations) {
      stats.totalCount++;
      stats.totalDuration += ob.duration;
      switch (ob.status) {
      case ObligationStatus::Valid:
        stats.validCount++;
        break;
      case ObligationStatus::Unknown:
      case ObligationStatus::Failure:
        stats.hasUnknown = true;
        break;
      case ObligationStatus::Timeout:
        stats.hasTimeout = true;
        break;
      }
    }
  }

  // Emit one report line per function
  for (const auto& [funcName, stats] : funcStatsMap) {
    if (stats.totalCount == 0) {
      continue;
    }
    double totalSeconds =
        static_cast<double>(stats.totalDuration.count()) / MS_PER_SECOND;

    // Build the display line from the location map
    std::string funcLine;
    auto locIt = locationMap.find(funcName);
    if (locIt != locationMap.end()) {
      funcLine = locIt->second.fileName + ":" + locIt->second.functionName;
    } else {
      funcLine = funcName;
    }

    if (stats.validCount == stats.totalCount) {
      out << "[PASS]  " << funcLine << "    " << stats.validCount << "/"
          << stats.totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.passCount++;
    } else if (stats.hasTimeout && !stats.hasUnknown) {
      out << "[TIMEOUT]  " << funcLine << "    " << stats.validCount << "/"
          << stats.totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.timeoutCount++;
      report.allPassed = false;
    } else {
      out << "[FAIL]  " << funcLine << "    " << stats.validCount << "/"
          << stats.totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.failCount++;
      report.allPassed = false;
    }
  }

  // Summary line
  out << "\nSummary: " << report.passCount << " passed, " << report.failCount
      << " failed, " << report.timeoutCount << " timeout\n";

  report.text = out.str();
  return report;
}

} // namespace arcanum
