# --- Frontend library (SubsetEnforcer + ContractParser) ---
add_library(ArcanumFrontend
  lib/SubsetEnforcer.cpp
  lib/ContractParser.cpp
)
target_include_directories(ArcanumFrontend SYSTEM PUBLIC
  ${ARCANUM_LLVM_INCLUDE_DIRS}
  ${ARCANUM_CLANG_INCLUDE_DIRS}
)
target_include_directories(ArcanumFrontend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ArcanumFrontend PUBLIC
  clangAST
  clangBasic
  clangFrontend
  clangTooling
  clangASTMatchers
)

add_subdirectory(tests)
