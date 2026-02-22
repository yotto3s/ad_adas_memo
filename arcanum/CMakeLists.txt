cmake_minimum_required(VERSION 3.20)
project(Arcanum LANGUAGES C CXX)

# C++17 is used instead of C++23 (spec target) because GCC 13 on Ubuntu 24.04
# does not fully support C++23.  No C++23 features are required by Slice 1.
# Re-evaluate when upgrading to GCC 14+ or using Clang exclusively.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Common warning flags
# -Wno-unused-parameter is needed because MLIR headers have unused parameters
# -Wno-covered-switch-default suppresses Clang warnings in MLIR-generated code
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wno-covered-switch-default)
endif()

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
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# --- Find Clang for LibTooling ---
find_package(Clang REQUIRED CONFIG)
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
set(ARCANUM_CLANG_INCLUDE_DIRS ${CLANG_INCLUDE_DIRS})

# --- TableGen for Arc dialect ---
# TableGen needs LLVM/MLIR include dirs in the INCLUDE_DIRECTORIES
# directory property to resolve .td includes. We add them here temporarily
# for TableGen, then remove them to avoid polluting GoogleTest with LLVM
# headers (which cause cxxabi.h conflicts with GCC).
set_property(DIRECTORY APPEND PROPERTY INCLUDE_DIRECTORIES
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/arcanum/dialect)

# Generate dialect declarations/definitions from ArcOps.td
set(LLVM_TARGET_DEFINITIONS include/arcanum/dialect/ArcOps.td)
mlir_tablegen(include/arcanum/dialect/ArcOps.h.inc -gen-op-decls)
mlir_tablegen(include/arcanum/dialect/ArcOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/arcanum/dialect/ArcDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/arcanum/dialect/ArcDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ArcOpsIncGen)

# Generate type declarations/definitions from ArcTypes.td
set(LLVM_TARGET_DEFINITIONS include/arcanum/dialect/ArcTypes.td)
mlir_tablegen(include/arcanum/dialect/ArcTypes.h.inc -gen-typedef-decls)
mlir_tablegen(include/arcanum/dialect/ArcTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(ArcTypesIncGen)

# Remove LLVM/MLIR includes from directory property after TableGen is done.
# This prevents them from polluting GoogleTest builds.
get_property(_current_includes DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
list(REMOVE_ITEM _current_includes ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES ${_current_includes})

# --- Arc dialect library ---
add_mlir_dialect_library(ArcDialect
  lib/dialect/ArcDialect.cpp
  lib/dialect/ArcOps.cpp
  lib/dialect/ArcTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/arcanum/dialect

  DEPENDS
  ArcOpsIncGen
  ArcTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
)
target_include_directories(ArcDialect SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)

# --- Frontend library (SubsetEnforcer + ContractParser) ---
add_library(ArcanumFrontend
  lib/frontend/SubsetEnforcer.cpp
  lib/frontend/ContractParser.cpp
)
target_include_directories(ArcanumFrontend SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(ArcanumFrontend PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
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
  lib/dialect/Lowering.cpp
)
target_include_directories(ArcanumLowering SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(ArcanumLowering PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumLowering PUBLIC
  ArcDialect
  ArcanumFrontend
  MLIRIR
)

# --- Passes library ---
add_library(ArcanumPasses
  lib/passes/Passes.cpp
)
target_include_directories(ArcanumPasses SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_include_directories(ArcanumPasses PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumPasses PUBLIC
  ArcDialect
  MLIRPass
  MLIRIR
)

# --- Backend library (WhyMLEmitter + Why3Runner) ---
add_library(ArcanumBackend
  lib/backend/WhyMLEmitter.cpp
  lib/backend/Why3Runner.cpp
)
target_include_directories(ArcanumBackend SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_include_directories(ArcanumBackend PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumBackend PUBLIC
  ArcDialect
  MLIRIR
  LLVMSupport
)

# --- Report library ---
add_library(ArcanumReport
  lib/report/ReportGenerator.cpp
)
target_include_directories(ArcanumReport SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_include_directories(ArcanumReport PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumReport PUBLIC
  LLVMSupport
)

# --- CLI executable ---
add_executable(arcanum tools/arcanum/main.cpp)
target_include_directories(arcanum SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(arcanum PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
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

# --- Install & Export ---
# Public headers
install(DIRECTORY include/arcanum
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.td"
)

# Generated .inc files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/arcanum/dialect
  DESTINATION include/arcanum
  FILES_MATCHING PATTERN "*.inc"
)

# Library targets
install(TARGETS ArcDialect ArcanumPasses ArcanumLowering
        ArcanumFrontend ArcanumBackend ArcanumReport
  EXPORT ArcanumTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# CMake package export
install(EXPORT ArcanumTargets
  NAMESPACE Arcanum::
  DESTINATION lib/cmake/Arcanum
)

# Package config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/ArcanumConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ArcanumConfig.cmake
  INSTALL_DESTINATION lib/cmake/Arcanum
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ArcanumConfig.cmake
  DESTINATION lib/cmake/Arcanum
)

# --- Testing ---
enable_testing()
add_subdirectory(tests)
