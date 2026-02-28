# --- TableGen for Arc dialect ---
# TableGen needs LLVM/MLIR include dirs in the INCLUDE_DIRECTORIES
# directory property to resolve .td includes. We add them here (scoped to this
# directory) so they don't pollute GoogleTest with LLVM headers.
set_property(DIRECTORY APPEND PROPERTY INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

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
list(REMOVE_ITEM _current_includes
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES ${_current_includes})

# --- Arc dialect library ---
add_mlir_dialect_library(ArcDialect
  lib/ArcDialect.cpp
  lib/ArcOps.cpp
  lib/ArcTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/arcanum/dialect

  DEPENDS
  ArcOpsIncGen
  ArcTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
)
# add_mlir_dialect_library creates an obj.ArcDialect OBJECT library that
# compiles the sources.  SYSTEM includes must also be set on it so that
# MLIR/LLVM header warnings are suppressed (C++20 reversed-comparison
# ambiguity in GCC).
target_include_directories(ArcDialect SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
if(TARGET obj.ArcDialect)
  target_include_directories(obj.ArcDialect SYSTEM PRIVATE
    ${ARCANUM_LLVM_INCLUDE_DIRS}
    ${ARCANUM_MLIR_INCLUDE_DIRS}
  )
endif()
target_include_directories(ArcDialect PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# --- Lowering library ---
add_library(ArcanumLowering
  lib/Lowering.cpp
)
target_include_directories(ArcanumLowering SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(ArcanumLowering PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumLowering PUBLIC
  ArcDialect
  ArcanumFrontend
  MLIRIR
)

# --- Passes library ---
add_library(ArcanumPasses
  lib/Passes.cpp
)
target_include_directories(ArcanumPasses SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_include_directories(ArcanumPasses PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumPasses PUBLIC
  ArcDialect
  MLIRPass
  MLIRIR
)

add_subdirectory(tests)
