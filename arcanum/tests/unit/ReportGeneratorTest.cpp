#include "report/ReportGenerator.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(ReportGeneratorTest, AllPassedReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back({"overflow_check'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(100), ""});
  obligations.push_back({"postcondition'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(200), ""});

  std::map<std::string, LocationEntry> locMap;
  locMap["safe_add"] = {"safe_add", "input.cpp", 6};

  auto report = generateReport(obligations, locMap);

  EXPECT_TRUE(report.allPassed);
  EXPECT_EQ(report.passCount, 1);
  EXPECT_EQ(report.failCount, 0);
  EXPECT_EQ(report.timeoutCount, 0);
  EXPECT_NE(report.text.find("[PASS]"), std::string::npos);
  EXPECT_NE(report.text.find("2/2"), std::string::npos);
  EXPECT_NE(report.text.find("Summary"), std::string::npos);
}

TEST(ReportGeneratorTest, FailedReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back({"overflow_check'vc", ObligationStatus::Valid,
                         std::chrono::milliseconds(100), ""});
  obligations.push_back({"postcondition'vc", ObligationStatus::Unknown,
                         std::chrono::milliseconds(200), ""});

  std::map<std::string, LocationEntry> locMap;
  locMap["bad_func"] = {"bad_func", "input.cpp", 10};

  auto report = generateReport(obligations, locMap);

  EXPECT_FALSE(report.allPassed);
  EXPECT_NE(report.text.find("[FAIL]"), std::string::npos);
}

TEST(ReportGeneratorTest, TimeoutReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back({"invariant'vc", ObligationStatus::Timeout,
                         std::chrono::milliseconds(30000), ""});

  std::map<std::string, LocationEntry> locMap;

  auto report = generateReport(obligations, locMap);

  EXPECT_FALSE(report.allPassed);
  EXPECT_EQ(report.timeoutCount, 1);
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
  obligations.push_back(
      {"overflow'vc", ObligationStatus::Valid, std::chrono::milliseconds(50), ""});
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

} // namespace
} // namespace arcanum
