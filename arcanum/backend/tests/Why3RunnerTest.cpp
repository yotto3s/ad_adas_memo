#include "arcanum/backend/Why3Runner.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

struct SingleGoalCase {
  const char* label;
  const char* goalLine;
  const char* expectedName;
  ObligationStatus expectedStatus;
  int64_t expectedDurationMs;
};

// TC-20/TC-21: Consolidated single-goal parsing tests covering
// Valid (with name + duration), Timeout, and Unknown statuses.
TEST(Why3RunnerTest, ParsesSingleGoalStatuses) {
  const SingleGoalCase cases[] = {
      {"Valid", "Goal safe_add'vc. Valid (0.01s, 0 steps).", "safe_add'vc",
       ObligationStatus::Valid, 10},
      {"Timeout", "Goal safe_add'vc. Timeout.", "safe_add'vc",
       ObligationStatus::Timeout, 0},
      {"Unknown", R"(Goal safe_add'vc. Unknown ("unknown").)", "safe_add'vc",
       ObligationStatus::Unknown, 0},
  };
  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.label);
    std::string output = std::string("File \"test.mlw\", line 5, "
                                     "characters 10-30:\n    ") +
                         tc.goalLine + "\n";
    auto results = parseWhy3Output(output);
    ASSERT_GE(results.size(), 1u);
    EXPECT_EQ(results[0].name, tc.expectedName);
    EXPECT_EQ(results[0].status, tc.expectedStatus);
    EXPECT_EQ(results[0].duration.count(), tc.expectedDurationMs);
  }
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

// Merged from Why3RunnerRunTest.cpp:
// [W16/TC-3] Test runWhy3() with a non-existent binary name.
TEST(Why3RunnerTest, NonExistentBinaryReturnsFailure) {
  auto results =
      runWhy3("/tmp/nonexistent.mlw", {}, "nonexistent_why3_binary_xyz");
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Failure);
  EXPECT_EQ(results[0].name, "why3_not_found");
}

// --- G7: parseWhy3Output() with moduleToFuncMap ---
// Tests that Theory context lines are used to attribute obligations to
// original C++ function names via the moduleToFuncMap.
TEST(Why3RunnerTest, ParsesTheoryContextWithModuleToFuncMap) {
  std::string output = R"(
Theory SafeAdd
    Goal overflow_check'vc. Valid (0.01s, 0 steps).
    Goal postcondition'vc. Valid (0.02s, 0 steps).
)";
  std::map<std::string, std::string> moduleToFuncMap;
  moduleToFuncMap["SafeAdd"] = "safe_add";

  auto results = parseWhy3Output(output, moduleToFuncMap);
  ASSERT_EQ(results.size(), 2u);
  // Both obligations should be attributed to safe_add via the map
  EXPECT_EQ(results[0].functionName, "safe_add");
  EXPECT_EQ(results[1].functionName, "safe_add");
}

} // namespace
} // namespace arcanum
