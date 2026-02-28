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

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# --- Find Clang for LibTooling ---
find_package(Clang REQUIRED CONFIG)
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
set(ARCANUM_CLANG_INCLUDE_DIRS ${CLANG_INCLUDE_DIRS})

# --- Testing ---
enable_testing()
include(cmake/GoogleTest.cmake)

# --- Component subdirectories (order matters: dependency chain) ---
add_subdirectory(frontend)
add_subdirectory(dialect)
add_subdirectory(backend)
add_subdirectory(tools/arcanum)
add_subdirectory(tests/lit)

# --- Install & Export ---
# Public headers (from each component)
install(DIRECTORY
  frontend/include/arcanum
  dialect/include/arcanum
  backend/include/arcanum
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.td"
)

# Generated .inc files
install(DIRECTORY ${CMAKE_BINARY_DIR}/dialect/include/arcanum/dialect
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
