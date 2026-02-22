find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Locate lit: prefer LLVM_EXTERNAL_LIT, then search common paths, then fallback
if(LLVM_EXTERNAL_LIT)
  set(_LIT_DEFAULT "${LLVM_EXTERNAL_LIT}")
elseif(LLVM_DIR)
  get_filename_component(_LLVM_ROOT "${LLVM_DIR}" DIRECTORY)
  get_filename_component(_LLVM_ROOT "${_LLVM_ROOT}" DIRECTORY)
  get_filename_component(_LLVM_ROOT "${_LLVM_ROOT}" DIRECTORY)
  find_program(_LIT_FOUND NAMES lit llvm-lit lit.py
      HINTS "${_LLVM_ROOT}/bin" "${_LLVM_ROOT}/build/utils/lit")
  if(_LIT_FOUND)
    set(_LIT_DEFAULT "${_LIT_FOUND}")
  endif()
endif()
if(NOT _LIT_DEFAULT)
  find_program(_LIT_FOUND NAMES lit llvm-lit lit.py
      HINTS "/usr/lib/llvm-21/build/utils/lit" "/usr/lib/llvm-21/bin")
  if(_LIT_FOUND)
    set(_LIT_DEFAULT "${_LIT_FOUND}")
  else()
    set(_LIT_DEFAULT "${Python3_EXECUTABLE}" "-m" "lit")
  endif()
endif()
set(LIT_COMMAND ${_LIT_DEFAULT} CACHE STRING "Command to run lit")

# Locate LLVM tools directory
if(LLVM_TOOLS_BINARY_DIR)
  set(_TOOLS_DEFAULT "${LLVM_TOOLS_BINARY_DIR}")
elseif(LLVM_DIR)
  get_filename_component(_TOOLS_DEFAULT "${LLVM_DIR}/../../../bin" ABSOLUTE)
else()
  set(_TOOLS_DEFAULT "/usr/lib/llvm-21/bin")
endif()
set(LLVM_TOOLS_DIR "${_TOOLS_DEFAULT}" CACHE PATH "Path to LLVM tools")
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
