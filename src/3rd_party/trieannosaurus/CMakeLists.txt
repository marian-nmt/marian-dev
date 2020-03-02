cmake_minimum_required(VERSION 3.5)
project(trieannosaurus)
string(ASCII 27 Esc)
set(Orange "${Esc}[33m")
set(ColourReset "${Esc}[m")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(example
  example.cpp
)

#TESTS
configure_file(${PROJECT_SOURCE_DIR}/Test/test_sents ${PROJECT_BINARY_DIR}/Test/test_sents COPYONLY)
add_executable(tests
  Test/allTests.cpp
)
set_target_properties(tests
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Test"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Test"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Test"
)

#CTest integration with Catch2
include(CMake/Catch.cmake)
include(CTest)
catch_discover_tests(tests)
