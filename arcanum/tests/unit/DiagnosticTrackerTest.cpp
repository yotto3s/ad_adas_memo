#include "arcanum/DiagnosticTracker.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace {

TEST(DiagnosticTrackerTest, InitialCountIsZero) {
  DiagnosticTracker::reset();
  EXPECT_EQ(DiagnosticTracker::getFallbackCount(), 0);
}

TEST(DiagnosticTrackerTest, RecordFallbackIncrements) {
  DiagnosticTracker::reset();
  DiagnosticTracker::recordFallback();
  EXPECT_EQ(DiagnosticTracker::getFallbackCount(), 1);
  DiagnosticTracker::recordFallback();
  EXPECT_EQ(DiagnosticTracker::getFallbackCount(), 2);
}

TEST(DiagnosticTrackerTest, ResetClearsCount) {
  DiagnosticTracker::reset();
  DiagnosticTracker::recordFallback();
  DiagnosticTracker::recordFallback();
  DiagnosticTracker::recordFallback();
  EXPECT_EQ(DiagnosticTracker::getFallbackCount(), 3);
  DiagnosticTracker::reset();
  EXPECT_EQ(DiagnosticTracker::getFallbackCount(), 0);
}

} // namespace
} // namespace arcanum
