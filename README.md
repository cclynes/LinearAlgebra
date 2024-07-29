## LinearAlgebra: Intro

I wrote this linear algebra program to get familiar with C++ and touch up on some computational linear algebra. It accounts for some basic properties of vectors and matrices, can perform basic operations between them, and can solve systems of linear equations using back- and forward-substitution, QR-decomposition, LU-decomposition, and Cholesky decomposition. The library does not currently support eigenvalue/eigenvector operations or error estimation; both of these are features I want to add, but only after taking stock of the program's structure and reorganizing to make better use of the object-orientedness of C++.

## Organization

There are two main template classes, Matrix and Vector. Member functions for each are declared in header files and defined in tpp files which are included in the headers to adhere to best practices for template classes (or at least, my understanding of best practices). I probably need more classes for specific kinds of vectors and matrices and to manage some operations between them, but I don't have much formal object-oriented programming experience, so I want to learn more before setting out to reorganize without a clear direction.

Basic member functions like getData, setData, transpose(), etc. are fairly self-explanatory from their definitions. System-solving functions are members of the Matrix class, and are called on a Matrix given a vector input and a tolerance, the level of user-tolerated error. (Tolerance specifications are currently rudimentary; I'd like to flesh this out with error estimation.)

To see some basic use cases in action, check out the examples folder. To print the results, build the project and run the examples (see Building and Running).

This project uses gtest as its testing framework. Test folders correspond to class, and each contains tests for basic member functions and operations. Additional Matrix testing files test the member functions that check Matrix properties and solve systems of equations. With the exception of a few Vector constructor tests that I still need to update, all tests use test fixtures for consistency's sake. Base fixtures are located outside of files in the class testing folders.

## Priorities

After reorganization, adding robust error estimation and tolerance calculation given types and floating point operations is the priority. Exploiting some of C++'s systems-level features, optimizing FLOPS, and making the library fast is the other major priority.

## Building
This project uses CMake as its build system. To build using CMake, navigate to the project directory and run the following terminal commands:
- mkdir build
- cd build
- cmake ..
- make

## Running
To run tests:
- ./runTests

To run examples:
- ./runExamples

## To do

The following is a running list of some things I need to take care of, in no particular order:

- Separate out the functions for returning a decomposition (e.g. LU, QR) and the functions for solving systems using that decomposition
    - Create structures or classes for decompositions to allow for easy access and abstract system-solving functionality.
- Add eigenvector/eigenvalue functionality
- Test rank calculation
- Add an inverse function, probably using QR
- Test with valgrind
- Make a simple data accessor method that returns a vector/matrix and allows pythonic, list-like selection of data (i.e. add subscript operators that support either arithmetic types or pairs of arithmetic types)
- Look into conforming to BLAS and LAPACK
- Test typecasting
- Check getData methods, esp. the overloaded ones: these might (?) be malfunctioning when matrices are empty, because they go off m_cols or m_rows or something, but then end up returning an empty vector {} when they should be returning {{}}
- Incorporate type checks into tests
- Review function return types and try to make type assumptions/typecasting easier
- Look into maybe updating a lot of methods to static methods?
- Hunt for edge cases, incl. with constructors (e.g. empty args, etc.)
- Abstract size/dimension-compatibility checks to helper functions (maybe)
- Does it actually make sense to allow empty vectors/matrices? Are there practical/theoretical arguments for allowing them?
- To test: what happens when you call Matrix(0, 1)? Do we get some sort of error? Guessing not
- Implement warnings for unstable, “almost singular” matrices
- Look into memory allocation - maybe arena allocation/memory arenas for objects
- Add support for multithreading
- Look into preallocated blocks for vectors too, esp. variable-size vectors, to avoid reallocating all the time
- Implement Givens rotation for QR-decomposition
- Add functions to compute tolerance/allowable error
- Write a function to generate random invertible nxn matrices
- Add hessenberg and permutated triangular solvers
- Reorganize a bit - you should maybe make the system solving functions members of classes inheriting from a centralized template class specific to system solving
- Add function to generate random SPD matrix
- Build functionality to measure time-performance