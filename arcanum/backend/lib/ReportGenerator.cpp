#include "arcanum/report/ReportGenerator.h"

#include <chrono>
#include <sstream>

namespace arcanum {
namespace {

struct FuncStats {
  int validCount = 0;
  int totalCount = 0;
  bool hasUnknown = false;
  bool hasTimeout = false;
  std::chrono::milliseconds totalDuration{0};
};

void accumulateObligationStats(FuncStats& stats, const ObligationResult& ob) {
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

std::string resolveObligationFuncName(const ObligationResult& ob,
                                      const std::string& fallback) {
  return ob.functionName.empty() ? fallback : ob.functionName;
}

bool obligationsHavePerFunctionNames(
    const std::vector<ObligationResult>& obligations) {
  for (const auto& ob : obligations) {
    if (!ob.functionName.empty()) {
      return true;
    }
  }
  return false;
}

std::string
resolveDisplayName(const std::string& funcName,
                   const std::map<std::string, LocationEntry>& locationMap) {
  auto locIt = locationMap.find(funcName);
  if (locIt != locationMap.end()) {
    return locIt->second.fileName + ":" + locIt->second.functionName;
  }
  return funcName;
}

std::string obligationStatusLabel(const FuncStats& stats) {
  if (stats.validCount == stats.totalCount) {
    return "[PASS]";
  }
  if (stats.hasTimeout && !stats.hasUnknown) {
    return "[TIMEOUT]";
  }
  return "[FAIL]";
}

void emitFunctionReportLine(std::ostream& out, const std::string& label,
                            const std::string& funcLine, const FuncStats& stats,
                            double totalSeconds) {
  out << label << "  " << funcLine << "    " << stats.validCount << "/"
      << stats.totalCount << " obligations proven (" << std::fixed;
  out.precision(1);
  out << totalSeconds << "s)\n";
}

} // namespace

Report generateReport(const std::vector<ObligationResult>& obligations,
                      const std::map<std::string, LocationEntry>& locationMap) {
  Report report;
  std::ostringstream out;
  constexpr double MS_PER_SECOND = 1000.0;

  // Group obligations by function name using the locationMap keys.
  // If obligations have functionName set, group by that. Otherwise,
  // fall back to assigning all obligations to the first function.

  bool hasPerObFuncNames = obligationsHavePerFunctionNames(obligations);

  std::map<std::string, FuncStats> funcStatsMap;

  if (hasPerObFuncNames && locationMap.size() > 1) {
    // Group by per-obligation function name
    for (const auto& ob : obligations) {
      std::string funcName = resolveObligationFuncName(ob, "unknown");
      accumulateObligationStats(funcStatsMap[funcName], ob);
    }
  } else {
    // Single function or no per-obligation function info: aggregate all
    std::string funcName =
        locationMap.empty() ? "unknown" : locationMap.begin()->first;
    auto& stats = funcStatsMap[funcName];
    for (const auto& ob : obligations) {
      accumulateObligationStats(stats, ob);
    }
  }

  // Emit one report line per function
  for (const auto& [funcName, stats] : funcStatsMap) {
    if (stats.totalCount == 0) {
      continue;
    }
    double totalSeconds =
        static_cast<double>(stats.totalDuration.count()) / MS_PER_SECOND;

    std::string funcLine = resolveDisplayName(funcName, locationMap);
    std::string label = obligationStatusLabel(stats);
    emitFunctionReportLine(out, label, funcLine, stats, totalSeconds);

    if (stats.validCount == stats.totalCount) {
      report.passCount++;
    } else if (stats.hasTimeout && !stats.hasUnknown) {
      report.timeoutCount++;
      report.allPassed = false;
    } else {
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
