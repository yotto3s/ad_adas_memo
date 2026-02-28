# --- Backend library (WhyMLEmitter + Why3Runner) ---
add_library(ArcanumBackend
  lib/WhyMLEmitter.cpp
  lib/Why3Runner.cpp
)
target_include_directories(ArcanumBackend SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_include_directories(ArcanumBackend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumBackend PUBLIC
  ArcDialect
  MLIRIR
  LLVMSupport
)

# --- Report library ---
add_library(ArcanumReport
  lib/ReportGenerator.cpp
)
target_include_directories(ArcanumReport SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
)
target_include_directories(ArcanumReport PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumReport PUBLIC
  LLVMSupport
)

add_subdirectory(tests)
