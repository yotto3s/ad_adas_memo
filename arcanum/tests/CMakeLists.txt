# Test targets are added by the test infrastructure task (Task 10).
# This file is a placeholder to satisfy add_subdirectory(tests).
# Arcanum test infrastructure
# GoogleTest via FetchContent + lit integration tests

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
# Prevent overriding parent project compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

# SubsetEnforcer unit test
add_executable(SubsetEnforcerTest
  unit/SubsetEnforcerTest.cpp
)
target_link_libraries(SubsetEnforcerTest
  PRIVATE
    GTest::gtest_main
    ArcanumFrontend
)
target_include_directories(SubsetEnforcerTest PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME SubsetEnforcerTest COMMAND SubsetEnforcerTest)

# ContractParser unit test
add_executable(ContractParserTest
  unit/ContractParserTest.cpp
)
target_link_libraries(ContractParserTest
  PRIVATE
    GTest::gtest_main
    ArcanumFrontend
)
target_include_directories(ContractParserTest PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME ContractParserTest COMMAND ContractParserTest)

# WhyMLEmitter unit test
add_executable(WhyMLEmitterTest
  unit/WhyMLEmitterTest.cpp
)
target_link_libraries(WhyMLEmitterTest
  PRIVATE
    GTest::gtest_main
    ArcanumBackend
)
target_include_directories(WhyMLEmitterTest PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME WhyMLEmitterTest COMMAND WhyMLEmitterTest)

# ---------------------------------------------------------------------------
# Lit integration tests
# ---------------------------------------------------------------------------

# Find lit (usually installed via pip: pip install lit)
find_program(LIT_COMMAND lit)
find_program(FILECHECK_COMMAND FileCheck)

if(LIT_COMMAND AND FILECHECK_COMMAND)
  add_custom_target(check-arcanum
    COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/lit
      --path $<TARGET_FILE_DIR:arcanum>
      -v
    DEPENDS arcanum
    COMMENT "Running Arcanum lit tests"
  )
else()
  message(STATUS "lit or FileCheck not found; lit tests (check-arcanum) disabled")
endif()
>>>>>>> 72786f1 (Add test infrastructure, lit integration tests, and CI workflow)
