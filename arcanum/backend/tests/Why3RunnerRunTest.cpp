#include "arcanum/backend/Why3Runner.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

/// [W16/TC-3] Test runWhy3() with a non-existent binary name.
/// Should return a Failure result with "why3_not_found" name.
TEST(Why3RunnerRunTest, NonExistentBinaryReturnsFailure) {
  auto results =
      runWhy3("/tmp/nonexistent.mlw", {}, "nonexistent_why3_binary_xyz");
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].status, ObligationStatus::Failure);
  EXPECT_EQ(results[0].name, "why3_not_found");
}

} // namespace
} // namespace arcanum
