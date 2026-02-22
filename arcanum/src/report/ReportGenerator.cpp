#include "report/ReportGenerator.h"

#include <chrono>
#include <sstream>

namespace arcanum {

Report generateReport(
    const std::vector<ObligationResult>& obligations,
    const std::map<std::string, LocationEntry>& locationMap) {
  Report report;

  int validCount = 0;
  int totalCount = static_cast<int>(obligations.size());
  bool hasUnknown = false;
  bool hasTimeout = false;
  auto totalDuration = std::chrono::milliseconds(0);

  for (const auto& ob : obligations) {
    totalDuration += ob.duration;
    switch (ob.status) {
    case ObligationStatus::Valid:
      ++validCount;
      break;
    case ObligationStatus::Unknown:
    case ObligationStatus::Failure:
      hasUnknown = true;
      break;
    case ObligationStatus::Timeout:
      hasTimeout = true;
      break;
    }
  }

  std::ostringstream out;

  double totalSeconds =
      static_cast<double>(totalDuration.count()) / 1000.0;

  // Per-function report line
  if (totalCount > 0) {
    // Try to get the first function from the location map
    std::string funcLine;
    if (!locationMap.empty()) {
      auto& entry = locationMap.begin()->second;
      funcLine = entry.fileName + ":" + entry.functionName;
    } else {
      funcLine = "unknown";
    }

    if (validCount == totalCount) {
      out << "[PASS]  " << funcLine << "    " << validCount << "/"
          << totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.passCount = 1;
    } else if (hasTimeout && !hasUnknown) {
      out << "[TIMEOUT]  " << funcLine << "    " << validCount << "/"
          << totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.timeoutCount = 1;
      report.allPassed = false;
    } else {
      out << "[FAIL]  " << funcLine << "    " << validCount << "/"
          << totalCount << " obligations proven (" << std::fixed;
      out.precision(1);
      out << totalSeconds << "s)\n";
      report.failCount = 1;
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
