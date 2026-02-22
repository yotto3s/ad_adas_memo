find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LIT_COMMAND "${Python3_EXECUTABLE}" "/usr/lib/llvm-21/build/utils/lit/lit.py"
    CACHE STRING "Command to run lit")

set(LLVM_TOOLS_DIR "/usr/lib/llvm-21/bin" CACHE PATH "Path to LLVM tools")
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
