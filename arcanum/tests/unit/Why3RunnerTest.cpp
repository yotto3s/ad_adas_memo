#include "arcanum/backend/Why3Runner.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

// TC-20/TC-21: ParsesValidObligation with name and duration assertions
TEST(Why3RunnerTest, ParsesValidObligation) {
  std::string output = R"(
File "test.mlw", line 5, characters 10-30:
    Goal safe_add'vc. Valid (0.01s, 0 steps).
)";
  auto results = parseWhy3Output(output);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Valid);
  // TC-21: Assert parsed obligation name
  EXPECT_EQ(results[0].name, "safe_add'vc");
  // TC-20: Assert parsed duration value (0.01s = 10ms)
  EXPECT_EQ(results[0].duration.count(), 10);
}

TEST(Why3RunnerTest, ParsesTimeoutObligation) {
  std::string output = R"(
File "test.mlw", line 5, characters 10-30:
    Goal safe_add'vc. Timeout.
)";
  auto results = parseWhy3Output(output);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Timeout);
}

TEST(Why3RunnerTest, ParsesUnknownObligation) {
  std::string output = R"(
File "test.mlw", line 5, characters 10-30:
    Goal safe_add'vc. Unknown ("unknown").
)";
  auto results = parseWhy3Output(output);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Unknown);
}

TEST(Why3RunnerTest, ParsesMultipleObligations) {
  std::string output = R"(
File "test.mlw", line 3, characters 10-30:
    Goal overflow_check'vc. Valid (0.01s, 0 steps).
File "test.mlw", line 5, characters 10-30:
    Goal postcondition'vc. Valid (0.02s, 0 steps).
)";
  auto results = parseWhy3Output(output);
  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].status, ObligationStatus::Valid);
  EXPECT_EQ(results[1].status, ObligationStatus::Valid);
}

TEST(Why3RunnerTest, ParsesEmptyOutput) {
  auto results = parseWhy3Output("");
  EXPECT_TRUE(results.empty());
}

// TC-14: Valid goal with no duration info -- duration should default to 0.
TEST(Why3RunnerTest, ValidGoalNoDurationDefaultsToZero) {
  std::string output = R"(
Theory SafeAdd
    Goal postcondition'vc. Valid.
)";
  auto results = parseWhy3Output(output);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Valid);
  EXPECT_EQ(results[0].duration.count(), 0);
}

} // namespace
} // namespace arcanum
