cmake_minimum_required(VERSION 3.27)

project(LinearAlgebra)

# Specify the C++ standard
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_STANDARD_REQUIRED True)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find GoogleTest
set(CMAKE_PREFIX_PATH "/opt/homebrew")
find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})

# Add sources for tests
set(TEST_SOURCES 
    ${CMAKE_SOURCE_DIR}/src/Matrix.tpp
    ${CMAKE_SOURCE_DIR}/src/Vector.tpp
    ${CMAKE_SOURCE_DIR}/tests/MatrixTests/test_matrix_basics.cpp
    ${CMAKE_SOURCE_DIR}/tests/MatrixTests/test_matrix_properties.cpp
    ${CMAKE_SOURCE_DIR}/tests/MatrixTests/test_matrix_operations.cpp
    ${CMAKE_SOURCE_DIR}/tests/MatrixTests/test_system_operations.cpp
    ${CMAKE_SOURCE_DIR}/tests/MatrixTests/test_system_performance.cpp
    ${CMAKE_SOURCE_DIR}/tests/VectorTests/test_vector_basics.cpp
    ${CMAKE_SOURCE_DIR}/tests/VectorTests/test_vector_operations.cpp
)

# Add sources for examples
set(EXAMPLE_SOURCES
    ${CMAKE_SOURCE_DIR}/examples/simple_examples.cpp
)

add_executable(runTests ${TEST_SOURCES}) # add executable for testing
target_link_libraries(runTests GTest::GTest GTest::Main pthread) # link with gtest libraries
target_include_directories(runTests PRIVATE ${GTEST_INCLUDE_DIRS}) # specify include directory

add_executable(runExamples ${EXAMPLE_SOURCES}) # add executable for examples
target_include_directories(runTests PRIVATE ${CMAKE_SOURCE_DIR}/include)