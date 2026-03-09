## LinearAlgebra: Intro

I wrote this linear algebra library to get familiar with C++ and touch up on some computational linear algebra. It accounts for many basic properties of vectors and matrices, can perform basic operations between them, and can solve systems of linear equations using back- and forward-substitution, QR-decomposition, LU-decomposition, and Cholesky decomposition. The library does not currently support eigenvalue/eigenvector operations or error estimation; both of these are priorities if I continue with development.

## Organization

There are two main template classes, Matrix and Vector. Basic member functions include getData, setData (these need new names), transpose, etc. System-solving functions are members of the Matrix class, and are called on a Matrix given a vector input and a tolerance, the level of user-tolerated error. Tolerance specifications are currently rudimentary.

To see some basic use cases in action, check out the examples folder. To print the results, build the project and run the examples (see Building and Running).

This project uses gtest as its testing framework. Test folders correspond to class, and each contains tests for basic member functions and operations. Additional Matrix testing files test the member functions that check Matrix properties and solve systems of equations. With the exception of a few Vector constructor tests that I still need to update, all tests use test fixtures for consistency's sake. Base fixtures are located outside of files in the class testing folders.

## Priorities

After reorganization, adding eigenvector/value functionality, robust error estimation, and tolerance calculation given types and floating point operations are natural next steps for development.

## Building
This project uses CMake as its build system. To build using CMake, cd into the project and run:
- mkdir build
- cd build
- cmake ..
- make

## Running
To run tests:
- ./runTests

To run examples:
- ./runExamples