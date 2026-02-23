/// Regression tests for ReportGenerator.cpp bug fixes.

#include "arcanum/report/ReportGenerator.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

/// [F6] Report only shows the first function when multiple functions are
/// present.
///
/// generateReport uses locationMap.begin()->second to get the function
/// name for the report line.  When multiple functions exist, all
/// obligations are combined into one line attributed to the
/// alphabetically first function name, producing misleading output.
///
/// Expected: The report should show separate results for each function.
/// Before fix: Only "alpha_func" appears; "beta_func" obligations are
/// counted but not attributed.
TEST(ReportGeneratorRegressionTest, MultipleFunctionsReportedSeparately) {
  std::vector<ObligationResult> obligations;
  // Two obligations from alpha_func
  obligations.push_back({"alpha_overflow'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(100), "alpha_func"});
  obligations.push_back({"alpha_postcondition'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(100), "alpha_func"});
  // Two obligations from beta_func -- one fails
  obligations.push_back({"beta_overflow'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(100), "beta_func"});
  obligations.push_back({"beta_postcondition'vc", ObligationStatus::Unknown,
                         std::chrono::milliseconds(100), "beta_func"});

  std::map<std::string, LocationEntry> locMap;
  locMap["alpha_func"] = {"alpha_func", "input.cpp", 5};
  locMap["beta_func"] = {"beta_func", "input.cpp", 15};

  auto report = generateReport(obligations, locMap);

  // The report should mention beta_func somewhere, not just alpha_func.
  // Before the fix, only alpha_func appears in the output.
  EXPECT_NE(report.text.find("beta_func"), std::string::npos)
      << "Report does not mention beta_func.  All obligations are attributed "
         "to the first function only.  Report text:\n"
      << report.text;
}

} // namespace
} // namespace arcanum
