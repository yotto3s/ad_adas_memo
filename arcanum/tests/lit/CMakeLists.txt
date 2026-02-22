find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Locate lit: prefer LLVM_EXTERNAL_LIT, then search common paths, then fallback.
# Always recompute (FORCE) so a stale cached fallback doesn't persist after the
# environment is fixed.  Users should set LLVM_EXTERNAL_LIT to override.
if(LLVM_EXTERNAL_LIT)
  set(_LIT_DEFAULT "${LLVM_EXTERNAL_LIT}")
elseif(LLVM_DIR)
  # LLVM_DIR layout varies:
  #   Debian/Ubuntu: <root>/cmake          (1 level up to root)
  #   Standard:      <root>/lib/cmake/llvm  (3 levels up to root)
  # Try Debian layout first (1 level), then standard (3 levels).
  get_filename_component(_LLVM_ROOT_1 "${LLVM_DIR}" DIRECTORY)
  get_filename_component(_LLVM_ROOT_3 "${_LLVM_ROOT_1}" DIRECTORY)
  get_filename_component(_LLVM_ROOT_3 "${_LLVM_ROOT_3}" DIRECTORY)
  find_program(_LIT_VIA_LLVM NAMES lit llvm-lit lit.py
      HINTS "${_LLVM_ROOT_1}/bin" "${_LLVM_ROOT_1}/build/utils/lit"
            "${_LLVM_ROOT_3}/bin" "${_LLVM_ROOT_3}/build/utils/lit")
  if(_LIT_VIA_LLVM)
    set(_LIT_DEFAULT "${_LIT_VIA_LLVM}")
  endif()
endif()
if(NOT _LIT_DEFAULT)
  find_program(_LIT_VIA_SYSTEM NAMES lit llvm-lit lit.py
      HINTS "/usr/lib/llvm-21/build/utils/lit" "/usr/lib/llvm-21/bin")
  if(_LIT_VIA_SYSTEM)
    set(_LIT_DEFAULT "${_LIT_VIA_SYSTEM}")
  else()
    set(_LIT_DEFAULT "${Python3_EXECUTABLE}" "-m" "lit")
  endif()
endif()
set(LIT_COMMAND ${_LIT_DEFAULT} CACHE STRING "Command to run lit" FORCE)

# Locate LLVM tools directory (handles both Debian and standard layouts)
if(LLVM_TOOLS_BINARY_DIR)
  set(_TOOLS_DEFAULT "${LLVM_TOOLS_BINARY_DIR}")
elseif(LLVM_DIR)
  # Try Debian layout first (1 level up), then standard (3 levels up)
  get_filename_component(_TOOLS_ROOT_1 "${LLVM_DIR}/.." ABSOLUTE)
  get_filename_component(_TOOLS_ROOT_3 "${LLVM_DIR}/../../.." ABSOLUTE)
  if(EXISTS "${_TOOLS_ROOT_1}/bin/FileCheck")
    set(_TOOLS_DEFAULT "${_TOOLS_ROOT_1}/bin")
  elseif(EXISTS "${_TOOLS_ROOT_3}/bin/FileCheck")
    set(_TOOLS_DEFAULT "${_TOOLS_ROOT_3}/bin")
  else()
    set(_TOOLS_DEFAULT "/usr/lib/llvm-21/bin")
  endif()
else()
  set(_TOOLS_DEFAULT "/usr/lib/llvm-21/bin")
endif()
set(LLVM_TOOLS_DIR "${_TOOLS_DEFAULT}" CACHE PATH "Path to LLVM tools" FORCE)
set(FILECHECK_PATH "${LLVM_TOOLS_DIR}/FileCheck" CACHE FILEPATH "Path to FileCheck")
set(NOT_PATH "${LLVM_TOOLS_DIR}/not" CACHE FILEPATH "Path to not tool")

set(ARCANUM_OBJ_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(ARCANUM_SRC_ROOT "${CMAKE_SOURCE_DIR}")
set(ARCANUM_PATH "${CMAKE_BINARY_DIR}/bin/arcanum")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
    "${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py"
    COPYONLY
)

# Copy test files to build directory
set(TEST_SUBDIRS subset-check verify)
foreach(SUBDIR ${TEST_SUBDIRS})
    file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/*.cpp")
    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME)
        configure_file(
            "${TEST_FILE}"
            "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${TEST_NAME}"
            COPYONLY
        )
    endforeach()
endforeach()

add_custom_target(check-arcanum-lit
    COMMAND ${LIT_COMMAND} -v "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS arcanum
    COMMENT "Running Arcanum lit tests"
    USES_TERMINAL
)

add_test(
    NAME arcanum-lit-tests
    COMMAND ${LIT_COMMAND} -v "${CMAKE_CURRENT_BINARY_DIR}"
)
set_tests_properties(arcanum-lit-tests PROPERTIES
    DEPENDS "arcanum"
)
