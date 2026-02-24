# --- ArcDialectTest ---
add_executable(ArcDialectTest ArcDialectTest.cpp)
target_link_libraries(ArcDialectTest PRIVATE
  ArcDialect
  gtest_main
  MLIRIR
)
gtest_discover_tests(ArcDialectTest)

# --- LoweringTest ---
add_executable(LoweringTest LoweringTest.cpp)
target_link_libraries(LoweringTest PRIVATE
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
)
gtest_discover_tests(LoweringTest)

# --- BugReproLoweringTest ---
add_executable(BugReproLoweringTest BugReproLoweringTest.cpp)
target_link_libraries(BugReproLoweringTest PRIVATE
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
)
gtest_discover_tests(BugReproLoweringTest)

# --- PassesTest ---
add_executable(PassesTest PassesTest.cpp)
target_link_libraries(PassesTest PRIVATE
  ArcanumPasses
  ArcDialect
  gtest_main
  MLIRIR
  MLIRPass
)
gtest_discover_tests(PassesTest)

# --- DiagnosticTrackerTest ---
add_executable(DiagnosticTrackerTest DiagnosticTrackerTest.cpp)
target_include_directories(DiagnosticTrackerTest PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(DiagnosticTrackerTest PRIVATE
  gtest_main
)
gtest_discover_tests(DiagnosticTrackerTest)
