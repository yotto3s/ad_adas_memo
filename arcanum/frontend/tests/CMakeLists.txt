# --- SubsetEnforcerTest ---
add_executable(SubsetEnforcerTest SubsetEnforcerTest.cpp)
target_link_libraries(SubsetEnforcerTest PRIVATE
  ArcanumFrontend
  gtest_main
  clangTooling
)
gtest_discover_tests(SubsetEnforcerTest)

# --- ContractParserTest ---
add_executable(ContractParserTest ContractParserTest.cpp)
target_link_libraries(ContractParserTest PRIVATE
  ArcanumFrontend
  gtest_main
  clangTooling
)
gtest_discover_tests(ContractParserTest)
