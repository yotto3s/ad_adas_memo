# Arcanum test infrastructure
# GoogleTest via FetchContent + lit integration tests

# Fetch Google Test
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

# --- SubsetEnforcerTest ---
add_executable(SubsetEnforcerTest unit/SubsetEnforcerTest.cpp)
target_include_directories(SubsetEnforcerTest SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(SubsetEnforcerTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(SubsetEnforcerTest PRIVATE
  ArcanumFrontend
  gtest_main
  clangTooling
)
gtest_discover_tests(SubsetEnforcerTest)

# --- ContractParserTest ---
add_executable(ContractParserTest unit/ContractParserTest.cpp)
target_include_directories(ContractParserTest SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(ContractParserTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(ContractParserTest PRIVATE
  ArcanumFrontend
  gtest_main
  clangTooling
)
gtest_discover_tests(ContractParserTest)

# --- ArcDialectTest ---
add_executable(ArcDialectTest unit/ArcDialectTest.cpp)
target_include_directories(ArcDialectTest SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_include_directories(ArcDialectTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(ArcDialectTest PRIVATE
  ArcDialect
  gtest_main
  MLIRIR
)
gtest_discover_tests(ArcDialectTest)

# --- LoweringTest ---
add_executable(LoweringTest unit/LoweringTest.cpp)
target_include_directories(LoweringTest SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(LoweringTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(LoweringTest PRIVATE
  ArcanumLowering
  ArcanumFrontend
  ArcDialect
  gtest_main
  clangTooling
  MLIRIR
)
gtest_discover_tests(LoweringTest)

# --- WhyMLEmitterTest ---
add_executable(WhyMLEmitterTest unit/WhyMLEmitterTest.cpp)
target_include_directories(WhyMLEmitterTest SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(WhyMLEmitterTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
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
add_executable(Why3RunnerTest unit/Why3RunnerTest.cpp)
target_include_directories(Why3RunnerTest SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
)
target_include_directories(Why3RunnerTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(Why3RunnerTest PRIVATE
  ArcanumBackend
  gtest_main
  LLVMSupport
)
gtest_discover_tests(Why3RunnerTest)

# --- ReportGeneratorTest ---
add_executable(ReportGeneratorTest unit/ReportGeneratorTest.cpp)
target_include_directories(ReportGeneratorTest SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_include_directories(ReportGeneratorTest PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(ReportGeneratorTest PRIVATE
  ArcanumReport
  ArcanumBackend
  ArcDialect
  gtest_main
  MLIRIR
  LLVMSupport
)
gtest_discover_tests(ReportGeneratorTest)

# --- Lit tests ---
add_subdirectory(lit)
