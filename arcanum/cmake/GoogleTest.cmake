# Shared GoogleTest infrastructure for all component test directories.
# Include this once from the root CMakeLists.txt before any add_subdirectory().

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Suppress Clang warnings in GoogleTest code
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(gtest PRIVATE -Wno-covered-switch-default)
  target_compile_options(gmock PRIVATE -Wno-covered-switch-default)
endif()

include(GoogleTest)
