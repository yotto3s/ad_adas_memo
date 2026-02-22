cmake_minimum_required(VERSION 3.20)
project(Arcanum LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Common warning flags
add_compile_options(-Wall -Wextra -Werror)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Find MLIR (which includes LLVM) ---
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Store LLVM/MLIR include directories for use by targets that need them.
# We DON'T add these globally to avoid LLVM's cxxabi.h conflicting with
# system headers when building GoogleTest.
set(ARCANUM_LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
set(ARCANUM_MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS})

# Project include directories (for TableGen outputs and dialect headers)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# --- Find Clang for LibTooling ---
find_package(Clang REQUIRED CONFIG)
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
set(ARCANUM_CLANG_INCLUDE_DIRS ${CLANG_INCLUDE_DIRS})

# --- TableGen for Arc dialect ---
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/dialect)

# Generate dialect declarations/definitions from ArcOps.td
set(LLVM_TARGET_DEFINITIONS src/dialect/ArcOps.td)
mlir_tablegen(src/dialect/ArcOps.h.inc -gen-op-decls)
mlir_tablegen(src/dialect/ArcOps.cpp.inc -gen-op-defs)
mlir_tablegen(src/dialect/ArcDialect.h.inc -gen-dialect-decls)
mlir_tablegen(src/dialect/ArcDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ArcOpsIncGen)

# Generate type declarations/definitions from ArcTypes.td
set(LLVM_TARGET_DEFINITIONS src/dialect/ArcTypes.td)
mlir_tablegen(src/dialect/ArcTypes.h.inc -gen-typedef-decls)
mlir_tablegen(src/dialect/ArcTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(ArcTypesIncGen)

# --- Arc dialect library ---
add_mlir_dialect_library(ArcDialect
  src/dialect/ArcDialect.cpp
  src/dialect/ArcOps.cpp
  src/dialect/ArcTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/src/dialect

  DEPENDS
  ArcOpsIncGen
  ArcTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
)
target_include_directories(ArcDialect PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)

# --- Frontend library (SubsetEnforcer + ContractParser) ---
add_library(ArcanumFrontend
  src/frontend/SubsetEnforcer.cpp
  src/frontend/ContractParser.cpp
)
target_include_directories(ArcanumFrontend PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(ArcanumFrontend PUBLIC
  clangAST
  clangBasic
  clangFrontend
  clangTooling
  clangASTMatchers
)

# --- Lowering library ---
add_library(ArcanumLowering
  src/dialect/Lowering.cpp
)
target_include_directories(ArcanumLowering PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(ArcanumLowering PUBLIC
  ArcDialect
  ArcanumFrontend
  MLIRIR
)

# --- Passes library ---
add_library(ArcanumPasses
  src/passes/Passes.cpp
)
target_include_directories(ArcanumPasses PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(ArcanumPasses PUBLIC
  ArcDialect
  MLIRPass
  MLIRIR
)

# --- Backend library (WhyMLEmitter + Why3Runner) ---
add_library(ArcanumBackend
  src/backend/WhyMLEmitter.cpp
  src/backend/Why3Runner.cpp
)
target_include_directories(ArcanumBackend PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(ArcanumBackend PUBLIC
  ArcDialect
  MLIRIR
  LLVMSupport
)

# --- Report library ---
add_library(ArcanumReport
  src/report/ReportGenerator.cpp
)
target_include_directories(ArcanumReport PUBLIC
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(ArcanumReport PUBLIC
  LLVMSupport
)

# --- CLI executable ---
add_executable(arcanum src/main.cpp)
target_include_directories(arcanum PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)
target_link_libraries(arcanum PRIVATE
  ArcanumFrontend
  ArcanumLowering
  ArcanumPasses
  ArcanumBackend
  ArcanumReport
  ArcDialect
  clangTooling
  clangFrontend
  clangSerialization
  clangDriver
  clangParse
  clangSema
  clangAnalysis
  clangEdit
  clangAST
  clangLex
  clangBasic
  LLVMSupport
  MLIRIR
  MLIRPass
)

# --- Testing ---
enable_testing()
add_subdirectory(tests)
