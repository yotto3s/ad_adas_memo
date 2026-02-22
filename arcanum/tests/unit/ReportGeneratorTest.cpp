#include "report/ReportGenerator.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(ReportGeneratorTest, AllPassedReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back(
      {"overflow_check'vc", ObligationStatus::Valid,
       std::chrono::milliseconds(100)});
  obligations.push_back(
      {"postcondition'vc", ObligationStatus::Valid,
       std::chrono::milliseconds(200)});

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
  obligations.push_back(
      {"overflow_check'vc", ObligationStatus::Valid,
       std::chrono::milliseconds(100)});
  obligations.push_back(
      {"postcondition'vc", ObligationStatus::Unknown,
       std::chrono::milliseconds(200)});

  std::map<std::string, LocationEntry> locMap;
  locMap["bad_func"] = {"bad_func", "input.cpp", 10};

  auto report = generateReport(obligations, locMap);

  EXPECT_FALSE(report.allPassed);
  EXPECT_NE(report.text.find("[FAIL]"), std::string::npos);
}

TEST(ReportGeneratorTest, TimeoutReport) {
  std::vector<ObligationResult> obligations;
  obligations.push_back(
      {"invariant'vc", ObligationStatus::Timeout,
       std::chrono::milliseconds(30000)});

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

} // namespace
} // namespace arcanum
