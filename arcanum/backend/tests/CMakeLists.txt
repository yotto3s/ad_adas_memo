# --- WhyMLEmitterTest ---
add_executable(WhyMLEmitterTest WhyMLEmitterTest.cpp)
target_link_libraries(WhyMLEmitterTest PRIVATE
  ArcanumBackend
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
  LLVMSupport
)
gtest_discover_tests(WhyMLEmitterTest)

# --- WhyMLEmitterRegressionTest ---
add_executable(WhyMLEmitterRegressionTest WhyMLEmitterRegressionTest.cpp)
target_link_libraries(WhyMLEmitterRegressionTest PRIVATE
  ArcanumBackend
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
  LLVMSupport
)
gtest_discover_tests(WhyMLEmitterRegressionTest)

# --- Why3RunnerTest ---
add_executable(Why3RunnerTest Why3RunnerTest.cpp)
target_link_libraries(Why3RunnerTest PRIVATE
  ArcanumBackend
  gtest_main
  LLVMSupport
)
gtest_discover_tests(Why3RunnerTest)

# --- Why3RunnerRunTest ---
add_executable(Why3RunnerRunTest Why3RunnerRunTest.cpp)
target_link_libraries(Why3RunnerRunTest PRIVATE
  ArcanumBackend
  gtest_main
  LLVMSupport
)
gtest_discover_tests(Why3RunnerRunTest)

# --- ReportGeneratorTest ---
add_executable(ReportGeneratorTest ReportGeneratorTest.cpp)
target_link_libraries(ReportGeneratorTest PRIVATE
  ArcanumReport
  ArcanumBackend
  ArcDialect
  gtest_main
  MLIRIR
  LLVMSupport
)
gtest_discover_tests(ReportGeneratorTest)

# --- ReportGeneratorRegressionTest ---
add_executable(ReportGeneratorRegressionTest ReportGeneratorRegressionTest.cpp)
target_link_libraries(ReportGeneratorRegressionTest PRIVATE
  ArcanumReport
  ArcanumBackend
  ArcDialect
  gtest_main
  MLIRIR
  LLVMSupport
)
gtest_discover_tests(ReportGeneratorRegressionTest)
