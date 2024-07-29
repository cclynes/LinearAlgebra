#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Vector.hpp"
#include "Utility.hpp"
#include <vector>
#include <functional>

namespace LinearAlgebra {

template<typename T>
class Vector;

template<typename T>
class Matrix {
public:

    // default matrix constructor
    Matrix();

    // constructor for a rows-by-cols matrix
    Matrix(size_t rows, size_t cols);

    // constructor from std::vector
    Matrix(const std::vector<std::vector<T>>& data);

    // constructor from initializer list
    Matrix(std::initializer_list<std::initializer_list<T>> data);

    // constructor that takes a vector argument
    Matrix(const Vector<T>& vec);

    // identity constructor
    static Matrix<T> identity(size_t dim);

    // destructor
    ~Matrix();

    // get dimensions
    std::pair<size_t, size_t> getDims() const;

    // methods to retrieve matrix data as a 2D vector
    std::vector<std::vector<T>> getData(
        const std::pair<size_t, size_t>& rowRange, 
        const std::pair<size_t, size_t>& colRange) const;

    std::vector<std::vector<T>> getData() const;

    T getData(
        size_t row,
        size_t col) const;

    std::vector<std::vector<T>> getData(
        size_t row, 
        const std::pair<size_t, size_t>& colRange) const;
    
    std::vector<std::vector<T>> getData(
        const std::pair<size_t, size_t>& rowRange, 
        size_t col) const;

    void print() const;
    void print(size_t precision) const;

    // methods to set matrix data
    void setData(
        const std::vector<std::vector<T>>& dataToSet, 
        const std::pair<size_t, size_t>& rowRange, 
        const std::pair<size_t, size_t>& colRange);

    void setData(
        const std::vector<std::vector<T>>& toSet);

    void setData(
        const std::vector<T>& toSet, 
        size_t row, 
        const std::pair<size_t, size_t>& colRange);

    void setData(
        const std::vector<T>& toSet, 
        const std::pair<size_t, size_t>& rowRange, 
        size_t col);

        void setData(
        T toSet,
        size_t row,
        size_t col);

    template<typename U>
    bool operator==(const Matrix<U>& toCompare);

    template<typename U, typename V>
    bool isNear(const Matrix<U>& toCompare, V tolerance);

    // method to add two matrices
    template<typename U>
    auto operator+(const Matrix<U>& toAdd) const -> Matrix<decltype(T{} + U{})>;

    // method to subtract two matrices
    template<typename U>
    auto operator-(const Matrix<U>& toSubtract) const -> Matrix<decltype(T{} - U{})>;

    // method to multiply two matrices
    template<typename U>
    auto operator*(const Matrix<U>& rightMultiple) const -> Matrix<decltype(T{} * U{})>;

    // method to multiply a matrix by a scalar
    template<typename U, typename = std::enable_if_t<std::is_arithmetic<U>::value>>
    auto operator*(U scalar) const -> Matrix<decltype(T{} * U{})>;

    template<typename U>
    friend auto operator*(U scalar, const Matrix<T>& mat) -> typename std::enable_if<std::is_arithmetic<U>::value, Matrix<decltype(U{} * T{})>>::type {
        assertTypesAreArithmetic<U>();
        return mat * scalar;
    }

    template<typename U>
    auto operator*(const Vector<U>& vec) const -> Matrix<decltype(T{} * U{})>;

    // method to compute matrix transpose
    Matrix<T> transpose() const;

    // method to swap a portion of a row or column
    void interchange(const std::pair<size_t, size_t>& toSwap, const std::pair<size_t, size_t>& range, size_t index);

    void interchange(const std::pair<size_t, size_t>& toSwap, size_t index);

    // method to determine whether matrix is symmetric
    template<typename U>
    bool isSymmetric(U tol) const;

    // method to throw exception if matrix and vector cannot form a sytem of equations
    template<typename U>
    void assertCanFormSystemWith(const Vector<U>& vecB) const;

    // methods to solve a linear system
    template<typename U, typename V>
    auto solveSystem(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveQR(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveLU(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveBackSub(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveForwardSub(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveCholesky(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;

    // helper functions to determine matrix properties
    bool isSquare() const;
    template<typename U>
    bool isUpperTriangular(U tol) const;
    template<typename U>
    bool isLowerTriangular(U tol) const;
    template<typename U>
    bool isUpperHessenberg(U tol) const;
    template<typename U>
    bool isLowerHessenberg(U tol) const;
    template<typename U>
    bool isHermitian(U tol) const;
    bool diagHasSameSign() const;
    template<typename U>
    bool diagIsNonZero(U tol) const;

    // function to sort the rows/columns of a matrix by some rule that returns an arithmetic type
    // returns a vector vec of indices representing the new row/column locations, where vec[i] = j means i should go to j
    template<typename U>
    std::vector<size_t> sortWithScrambler(std::function<U(const std::vector<T>&)> functional, const size_t index);

    // function to scramble the rows/columns of a matrix according to the given index vector
    void scramble(const std::vector<size_t>& indices, size_t index);

    // method to compute rank
    size_t rank() const;

private:
    // variables to hold matrix dimensions and data
    size_t m_rows;
    size_t m_cols;
    std::vector<std::vector<T>> m_data;

    // helper functions to solve a system using a specified algorithm
    template<typename U, typename V>
    auto solveQRUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveLUUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveBackSubUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveForwardSubUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;
    template<typename U, typename V>
    auto solveCholeskyUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})>;

    // helper function to return R during QR factorization using Householder vectors
    template<typename V>
    auto getQRDecomp(V tol) const 
        -> std::tuple<std::vector<Vector<decltype(T{} * 1.0)>>, Matrix<decltype(T{} * 1.0)>, std::vector<size_t>, size_t>;
};

} // namespace LinearAlgebra

#include "../src/Matrix.tpp"

#endif // MATRIX_CPP