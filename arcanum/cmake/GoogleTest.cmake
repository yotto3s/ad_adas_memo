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

# Suppress warnings-as-errors for third-party GoogleTest code.
# C++23 triggers new warnings (e.g., Clang -Wcharacter-conversion for char8_t)
# that we cannot fix upstream.
foreach(_gtest_target gtest gmock gtest_main gmock_main)
  if(TARGET ${_gtest_target})
    target_compile_options(${_gtest_target} PRIVATE -Wno-error)
  endif()
endforeach()

include(GoogleTest)
