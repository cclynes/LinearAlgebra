#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "Matrix.hpp"
#include "Utility.hpp"
#include <vector>

namespace LinearAlgebra {

template<typename T>
class Matrix;

template<typename T>
class Vector {
public:

    // default constructor
    Vector();

    // constructor for a Vector
    Vector(const std::vector<T>& vec, bool isRow=false);

    // constructor for 2d argument
    Vector(const std::vector<std::vector<T>>& vec);

    // constructor for dimensional input
    Vector(size_t dim, bool isRow);

    // constructor that takes a matrix argument
    Vector(const Matrix<T>& matrix);

    // destructor
    ~Vector();

    // methods to set and get data from specified ranges
    std::vector<T> getData(const std::pair<size_t, size_t>& range) const;
    T getData(size_t index) const;
    std::vector<T> getData() const;
    void print() const;
    void print(size_t precision) const;

    void setData(const std::vector<T>& toSet, const std::pair<size_t, size_t>& range);
    void setData(T toSet, size_t index);
    void setData(const std::vector<T>& toSet);


    // method to check if Vector is row or column vector
    bool isRow() const;

    // method to return dimension of vector
    size_t dim() const;

    // method to take transpose (i.e. transform a row vector to a column vector or vice versa)
    Vector<T> transpose() const;

    // method to reverse a Vector
    Vector<T> reverse() const;

    // method to scramble entries of a Vector based on given index vector
    void scramble(std::vector<size_t>& indices);

    // method to check for equality
    template<typename U>
    bool operator==(const Vector<U>& toCompare) const;

    template<typename U, typename V>
    bool isNear(const Vector<U>& toCompare, V tolerance);

    // method to add two Vectors
    template<typename U>
    auto operator+(const Vector<U>& toAdd) const -> Vector<decltype(T{} + U{})>;

    // method to subtract two Vectors
    template<typename U>
    auto operator-(const Vector<U>& toSubtract) const -> Vector<decltype(T{} - U{})>;

    // method to compute dot product of two Vectors
    template<typename U>
    decltype(T{} * U{}) dot(const Vector<U>& toMultiply) const;

    // method to compute matrix product of two Vectors
    template<typename U>
    auto operator*(const Vector<U>& toMultiply) const -> Matrix<decltype(T{} * U{})>;

    // method to left-multipy a Vector with a Matrix
    template<typename U>
    auto operator*(const Matrix<U>& mat) const -> Matrix<decltype(T{} * U{})>;

    // method to scalar multiply a Vector
    template<typename U, typename = std::enable_if_t<std::is_arithmetic<U>::value>>
    auto operator*(U scalar) const -> Vector<decltype(T{} * U{})>;

    template<typename U>
    friend auto operator*(U scalar, const Vector<T>& vec) -> typename std::enable_if<std::is_arithmetic<U>::value, Vector<decltype(U{} * T{})>>::type {
        assertTypesAreArithmetic<U>();
        return vec * scalar; // use previously defined operator
    }

    // method to compute vector 2-norm
    T norm() const;

private:

    bool m_is_row;
    size_t m_dim;
    std::vector<T> m_data;

};

} // namespace LinearAlgebra

#include "../src/Vector.tpp"

#endif // VECTOR_CPP