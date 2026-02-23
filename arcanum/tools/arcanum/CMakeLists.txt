# --- CLI executable ---
add_executable(arcanum main.cpp)
target_include_directories(arcanum SYSTEM PRIVATE
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_MLIR_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
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
