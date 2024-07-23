## LinearAlgebra: Intro

I wrote this linear algebra program to get familiar with C++ and touch up on some computational linear algebra. It accounts for some basic properties of vectors and matrices, can perform basic operations between those objects, and can solve systems of linear equations using back- and forward-substitution, QR-decomposition, LU-decomposition, and Cholesky decomposition. The library does not currently support eigenvalue/eigenvector operations or error estimation; both of these are features I want to add, but only after taking stock of the program's structure and reorganizing to make better use of the object-orientedness of C++. There’s a lot to be improved.

## Organization

There are two main template classes, Matrix and Vector. Member functions for each are declared in header files and defined in tpp files which are included in the headers to adhere to best practices for template classes (or at least, my understanding of best practices). I probably need more classes for specific kinds of vectors and matrices and to manage some operations between them, but with no object-oriented programming experience, I want to learn more before setting out to reorganize with no clear direction.

Basic member functions like getData, setData, transpose(), etc. are fairly self-explanatory from their definitions. System-solving functions are members of the Matrix class, and are called on a Matrix given a vector input and a tolerance, the level of user-tolerated error. (Tolerance specifications are currently rudimentary; I'd like to flesh this out with error estimation.)

This project uses gtest as its testing framework. Test folders correspond to class, and each contains tests for basic member functions and operations. Additional Matrix testing files test the member functions that check Matrix properties and solve systems of equations. With the exception of a few Vector constructor tests that I still need to update, all tests use test fixtures for consistency's sake. Base fixtures are located outside of files in the class testing folders.

## Priorities

Since this is intended as a library, I'd like to wrap it in a namespace; I just can't get this to work without defining friend functions directly in the header files. After reorganization, adding robust error estimation and tolerance calculation given types and floating point operations is a major priority. Exploiting C++'s systems-level features to optimize FLOPS and make the library fast is the other major priority.

## Building
This project uses CMake as its build system. To build using CMake, navigate to the project directory and run the following terminal commands:
mkdir build
cd build
cmake ..
make

## Running
There's not much to run right now except tests. To run tests:
./runTests

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
- Check getData methods, esp. The overloaded ones: these might (?) be malfunctioning when matrices are empty, because they go off m_cols or m_rows or something, but then end up returning an empty vector {} when they should be returning {{}}
- Incorporate type checks into tests
- Go through and check types - especially make sure iterator variables have type size_t
- Also check method return types and try to make type assumptions/typecasting easier
- Check into maybe updating a lot of methods to static methods?
- Add Vector constructor that just lets you input 1-d list and gives you a column vector
- Hunt for edge cases, esp. with constructors (e.g. empty args, etc.)
- Abstract size/dimension-compatibility checks to helper functions (maybe)
- Another question to ask: does it actually make sense to allow empty vectors/matrices? Are there practical/theoretical arguments for allowing them?
- To test: what happens when you call Matrix(0, 1)? Do we get some sort of error? Guessing not
- Allow for variable type passing (if possible) to accommodate situations where, for instance, you might want to getData either as a 2d vector or as a float
- Implement warnings for unstable, “almost singular” matrices
- Look into memory allocation - maybe arena allocation/memory arenas for objects? Add pointers?
- Multithreading?
- Look into preallocated blocks for vectors too, esp. Variable-size vectors, to avoid reallocating all the time
- Implement Givens rotation for QR-decomposition
- Add methods to compute tolerance/allowable error
- Write an algorithm to generate random invertible nxn matrices
- Go through constructors and make sure all of them define every attribute (this caused problems in LU testing due to issues with MathVector(const Matrix& mat) constructor that were not detected before)
- Add hessenberg and permutated triangular solvers
- Clean up fixtures and reorganize a bit - you should maybe make the system solving functions members of classes inheriting from a centralized template class specific to system solving
- Add function to generate random SPD matrix
- Build functionality to track floating point error and calculate tolerances
- Build functionality to measure time-performance