#include "arcanum/report/ReportGenerator.h"

#include <gtest/gtest.h>

#include <initializer_list>

namespace arcanum {
namespace {

struct ReportCase {
  const char* label;
  std::vector<ObligationResult> obligations;
  std::map<std::string, LocationEntry> locMap;
  bool expectedAllPassed;
  int expectedPassCount;
  int expectedFailCount;
  int expectedTimeoutCount;
  std::vector<std::string> expectedPatterns;
};

// Consolidated test for AllPassed, Failed, Timeout, and SingleFailure reports.
TEST(ReportGeneratorTest, SingleOutcomeReports) {
  std::vector<ReportCase> cases;

  // AllPassedReport
  {
    ReportCase tc;
    tc.label = "AllPassed";
    tc.obligations = {
        {"overflow_check'vc", ObligationStatus::Valid,
         std::chrono::milliseconds(100), ""},
        {"postcondition'vc", ObligationStatus::Valid,
         std::chrono::milliseconds(200), ""},
    };
    tc.locMap["safe_add"] = {"safe_add", "input.cpp", 6};
    tc.expectedAllPassed = true;
    tc.expectedPassCount = 1;
    tc.expectedFailCount = 0;
    tc.expectedTimeoutCount = 0;
    tc.expectedPatterns = {"[PASS]", "2/2", "Summary"};
    cases.push_back(std::move(tc));
  }

  // FailedReport
  {
    ReportCase tc;
    tc.label = "Failed";
    tc.obligations = {
        {"overflow_check'vc", ObligationStatus::Valid,
         std::chrono::milliseconds(100), ""},
        {"postcondition'vc", ObligationStatus::Unknown,
         std::chrono::milliseconds(200), ""},
    };
    tc.locMap["bad_func"] = {"bad_func", "input.cpp", 10};
    tc.expectedAllPassed = false;
    tc.expectedPassCount = -1;    // not checked
    tc.expectedFailCount = -1;    // not checked
    tc.expectedTimeoutCount = -1; // not checked
    tc.expectedPatterns = {"[FAIL]"};
    cases.push_back(std::move(tc));
  }

  // TimeoutReport
  {
    ReportCase tc;
    tc.label = "Timeout";
    tc.obligations = {
        {"invariant'vc", ObligationStatus::Timeout,
         std::chrono::milliseconds(30000), ""},
    };
    tc.expectedAllPassed = false;
    tc.expectedPassCount = -1; // not checked
    tc.expectedFailCount = -1; // not checked
    tc.expectedTimeoutCount = 1;
    tc.expectedPatterns = {};
    cases.push_back(std::move(tc));
  }

  // [W19/TC-9] SingleFailureReport
  {
    ReportCase tc;
    tc.label = "SingleFailure";
    tc.obligations = {
        {"postcondition'vc", ObligationStatus::Failure,
         std::chrono::milliseconds(50), ""},
    };
    tc.locMap["fail_func"] = {"fail_func", "input.cpp", 10};
    tc.expectedAllPassed = false;
    tc.expectedPassCount = 0;
    tc.expectedFailCount = 1;
    tc.expectedTimeoutCount = -1; // not checked
    tc.expectedPatterns = {"[FAIL]", "0/1", "fail_func"};
    cases.push_back(std::move(tc));
  }

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.label);
    auto report = generateReport(tc.obligations, tc.locMap);

    EXPECT_EQ(report.allPassed, tc.expectedAllPassed);
    if (tc.expectedPassCount >= 0) {
      EXPECT_EQ(report.passCount, tc.expectedPassCount);
    }
    if (tc.expectedFailCount >= 0) {
      EXPECT_EQ(report.failCount, tc.expectedFailCount);
    }
    if (tc.expectedTimeoutCount >= 0) {
      EXPECT_EQ(report.timeoutCount, tc.expectedTimeoutCount);
    }
    for (const auto& pattern : tc.expectedPatterns) {
      EXPECT_NE(report.text.find(pattern), std::string::npos)
          << "Expected pattern not found: " << pattern;
    }
  }
}

TEST(ReportGeneratorTest, EmptyObligations) {
  std::vector<ObligationResult> obligations;
  std::map<std::string, LocationEntry> locMap;

  auto report = generateReport(obligations, locMap);

  EXPECT_TRUE(report.allPassed);
  EXPECT_NE(report.text.find("Summary"), std::string::npos);
}

// TC-23: Mixed obligation statuses
TEST(ReportGeneratorTest, MixedObligationStatuses) {
  std::vector<ObligationResult> obligations;
  obligations.push_back({"overflow'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(50), ""});
  obligations.push_back({"postcondition'vc", ObligationStatus::Timeout,
                         std::chrono::milliseconds(30000), ""});
  obligations.push_back({"precondition'vc", ObligationStatus::Unknown,
                         std::chrono::milliseconds(0), ""});
  obligations.push_back({"assertion'vc", ObligationStatus::Failure,
                         std::chrono::milliseconds(0), ""});

  std::map<std::string, LocationEntry> locMap;
  locMap["mixed_func"] = {"mixed_func", "input.cpp", 5};

  auto report = generateReport(obligations, locMap);

  EXPECT_FALSE(report.allPassed);
  // With mixed unknown+timeout, FAIL takes priority over TIMEOUT
  EXPECT_EQ(report.failCount, 1);
  EXPECT_NE(report.text.find("[FAIL]"), std::string::npos);
  EXPECT_NE(report.text.find("Summary"), std::string::npos);
}

// [W13] Multi-function report test
TEST(ReportGeneratorTest, MultiFunctionReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back({"alpha_overflow'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(100), "alpha_func"});
  obligations.push_back({"alpha_postcondition'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(100), "alpha_func"});
  obligations.push_back({"beta_overflow'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(100), "beta_func"});
  obligations.push_back({"beta_postcondition'vc", ObligationStatus::Unknown,
                         std::chrono::milliseconds(100), "beta_func"});

  std::map<std::string, LocationEntry> locMap;
  locMap["alpha_func"] = {"alpha_func", "input.cpp", 5};
  locMap["beta_func"] = {"beta_func", "input.cpp", 15};

  auto report = generateReport(obligations, locMap);

  EXPECT_FALSE(report.allPassed);
  // Should mention both functions
  EXPECT_NE(report.text.find("alpha_func"), std::string::npos);
  EXPECT_NE(report.text.find("beta_func"), std::string::npos);
  // alpha_func should PASS, beta_func should FAIL
  EXPECT_NE(report.text.find("[PASS]"), std::string::npos);
  EXPECT_NE(report.text.find("[FAIL]"), std::string::npos);
  EXPECT_EQ(report.passCount, 1);
  EXPECT_EQ(report.failCount, 1);
}

} // namespace
} // namespace arcanum
