#ifndef ARCANUM_REPORT_REPORTGENERATOR_H
#define ARCANUM_REPORT_REPORTGENERATOR_H

#include "backend/Why3Runner.h"
#include "backend/WhyMLEmitter.h"

#include <string>
#include <vector>
#include <map>

namespace arcanum {

struct Report {
  std::string text;
  bool allPassed = true;
  int passCount = 0;
  int failCount = 0;
  int timeoutCount = 0;
};

/// Generate a human-readable verification report.
Report generateReport(
    const std::vector<ObligationResult>& obligations,
    const std::map<std::string, LocationEntry>& locationMap);

} // namespace arcanum

#endif // ARCANUM_REPORT_REPORTGENERATOR_H
