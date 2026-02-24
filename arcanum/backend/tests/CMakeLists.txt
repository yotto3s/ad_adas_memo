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

# --- Why3RunnerTest ---
add_executable(Why3RunnerTest Why3RunnerTest.cpp)
target_link_libraries(Why3RunnerTest PRIVATE
  ArcanumBackend
  gtest_main
  LLVMSupport
)
gtest_discover_tests(Why3RunnerTest)

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
