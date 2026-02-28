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

# --- BugReproWhyMLEmitterTest ---
add_executable(BugReproWhyMLEmitterTest BugReproWhyMLEmitterTest.cpp)
target_link_libraries(BugReproWhyMLEmitterTest PRIVATE
  ArcanumBackend
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
  LLVMSupport
)
gtest_discover_tests(BugReproWhyMLEmitterTest)

# --- BugReproSlice3WhyMLEmitterTest ---
add_executable(BugReproSlice3WhyMLEmitterTest BugReproSlice3WhyMLEmitterTest.cpp)
target_link_libraries(BugReproSlice3WhyMLEmitterTest PRIVATE
  ArcanumBackend
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
  LLVMSupport
)
gtest_discover_tests(BugReproSlice3WhyMLEmitterTest)

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
