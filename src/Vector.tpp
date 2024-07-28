#include "../include/Vector.hpp"
#include "../include/Matrix.hpp"
#include "../include/Utility.hpp"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <iostream>
#include <vector>
#include <iomanip>

namespace LinearAlgebra {

// default constructor
template<typename T>
Vector<T>::Vector()
    : m_data({}), m_is_row(false), m_dim(0) {
        assertTypesAreArithmetic<T>();
    }

// 1D constructor
template<typename T>
Vector<T>::Vector(const std::vector<T>& vec, bool isRow) 
    : m_data(vec), m_is_row(isRow), m_dim(vec.size()) {
        assertTypesAreArithmetic<T>();
    }

// 2D constructor - must take a 1-by-n or n-by-1 vector
template<typename T>
Vector<T>::Vector(const std::vector<std::vector<T>>& vec) {
    assertTypesAreArithmetic<T>();

    if ((vec.size() == 0) || (vec[0].size() == 0)) {
        m_data = {};
        m_dim = 0;
    }

    else if (vec.size() == 1) {
        m_data = vec[0];
        m_is_row = true;
        m_dim = vec[0].size();
    }

    else {
        m_data = std::vector<T>(vec.size(), 0.0f);
        for (size_t i = 0; i < vec.size(); i++) {
            
            // check that each entry holds exactly one element
            if (vec[i].size() != 1) {
                throw std::invalid_argument("Vector can only hold one row or one column of data.");
            }

            m_data[i] = vec[i][0];
        }

        m_is_row = false;
        m_dim = vec.size();
    }
}

// dimensional constructor
template<typename T>
Vector<T>::Vector(size_t dim, bool isRow)
    : Vector(vector<T>(dim, 0.0), isRow) {
        assertTypesAreArithmetic<T>();
    }

// constructor from Matrix
template<typename T>
Vector<T>::Vector(const Matrix<T>& matrix) {
    assertTypesAreArithmetic<T>();

    size_t numRows = matrix.getDims().first;
    size_t numCols = matrix.getDims().second;
    if (!((numRows == 1) || (numCols == 1))) {
        throw std::invalid_argument("Row or column dimension must be 1 to typecast a matrix to a Vector.");
    }
    if (numCols == 1) {
        m_dim = numRows;
        m_data = std::vector<T>(numRows);
        m_is_row = false;
        for (size_t i = 0; i < numRows; i++) {
            m_data[i] = matrix.getData(i, 0);
        }
    }
    else if (numRows == 1) {
        m_dim = numCols;
        m_data = matrix.getData()[0];
    }
}

// Vector destructor
template<typename T>
Vector<T>::~Vector() {}

template<typename T>
std::vector<T> Vector<T>::getData(const std::pair<size_t, size_t>& range) const {
    size_t start = range.first;
    size_t end = range.second;

    // check that range indices are within bounds
    if ((end > m_data.size()) || (start > m_data.size())) {
        throw std::invalid_argument("Range must be within bounds of data.");
    }

    // return the data

    size_t toGetLength = (end >= start) ? (end - start) : 0;
    vector<T> toGet(toGetLength);

    for (size_t i = start; i < end; i++) {
        toGet[i - start] = m_data[i];
    }

    return toGet;
}

template<typename T>
T Vector<T>::getData(size_t index) const {
    return getData({index, index + 1})[0];
}

template<typename T>
std::vector<T> Vector<T>::getData() const {
    return getData({0, m_data.size()});
}

template<typename T>
void Vector<T>::print() const {
    print(5);
}

template<typename T>
void Vector<T>::print(size_t precision) const {
    for (size_t i=0; i < m_dim; i++) {
        std::cout << std::setprecision(precision) << m_data[i] << " ";
        if (!m_is_row) {std::cout << endl;}
    }
}

template<typename T>
void Vector<T>::setData(const vector<T>& toSet, const std::pair<size_t, size_t>& range) {
    assertTypesAreArithmetic<T>();

    size_t start = range.first;
    size_t end = range.second;

    // check that range indices are within bounds
    if (end > m_data.size()) {
        throw std::invalid_argument("Range must be within bounds of data.");
    }

    if ((end - start) != toSet.size()) {
        throw std::invalid_argument("Range must match size of input data.");
    }

    // set the data
    for (size_t i = start; i < end; i++) {
        m_data[i] = toSet[i - start];
    }
}

template<typename T>
void Vector<T>::setData(T toSet, size_t index) {
    assertTypesAreArithmetic<T>();
    setData({toSet}, {index, index + 1});
}

template<typename T>
void Vector<T>::setData(const vector<T>& toSet) {
    assertTypesAreArithmetic<T>();
    setData({toSet}, {0, m_dim});
}

template<typename T>
bool Vector<T>::isRow() const {
    return m_is_row;
}

template<typename T>
size_t Vector<T>::dim() const {
    return m_dim;
}

template<typename T>
Vector<T> Vector<T>::transpose() const {
    return Vector(m_data, !m_is_row);
}

template<typename T>
Vector<T> Vector<T>::reverse() const {
    std::vector<T> new_m_data = m_data;
    std::reverse(new_m_data.begin(), new_m_data.end());
    return Vector(new_m_data, m_is_row);
}

template<typename T>
void Vector<T>::scramble(std::vector<size_t>& indices) {
    if (indices.size() != dim()) {
        throw std::invalid_argument("Index vector must have same length as Vector.");
    }
    
    std::vector<bool> visitedIndices(dim(), false);
    std::vector<T> scrambledData(dim());

    for (size_t i=0; i < dim(); i++) {
        size_t index = indices[i];
        if (index >= dim()) {
            throw std::invalid_argument("All indices must be within range to scramble.");
        }
        if (visitedIndices[index] == true) {
            throw std::invalid_argument("List of indices must be unique to scramble.");
        }
        scrambledData[index] = m_data[i];
        visitedIndices[index] = true;
    }

    m_data = scrambledData;
}

template<typename T>
template<typename U>
bool Vector<T>::operator==(const Vector<U>& toCompare) const {
    // check that dimensions are equal
    if (m_dim != toCompare.getData().size()) {
        return false;
    }
    // check that orientation is equal
    if (m_is_row != toCompare.isRow()) {
        return false;
    }
    // check that data is equal
    for (size_t i = 0; i < m_dim; i++) {
        if (m_data[i] != toCompare.getData(i)) {
            return false;
        }
    }
    return true;
}

template<typename T>
template<typename U, typename V>
bool Vector<T>::isNear(const Vector<U>& toCompare, V tolerance) {
    assertTypesAreArithmetic<V>();
    // check that dimensions are equal
    if (m_dim != toCompare.getData().size()) {
        return false;
    }
    // check that orientation is equal
    if (m_is_row != toCompare.isRow()) {
        return false;
    }
    // check that data is equal within given tolerance
    for (size_t i = 0; i < m_dim; i++) {
        if (abs(m_data[i] - toCompare.getData(i)) > tolerance) {
            return false;
        }
    }
    return true;
}

template<typename T>
template<typename U>
auto Vector<T>::operator+(const Vector<U>& toAdd) const -> Vector<decltype(T{} + U{})> {
    // check that dimensions are equal
    if ((m_data.size() != toAdd.m_data.size()) || m_is_row != toAdd.m_is_row) {
        throw std::invalid_argument("Vector addition requires that dimensions are equal.");
    }

    // add the two vectors
    vector<T> sum(m_data.size());

    for (int i=0; i<m_data.size(); i++) {
        sum[i] = m_data[i] + toAdd.m_data[i];
    }

    return Vector(sum, m_is_row);
}

template<typename T>
template<typename U>
auto Vector<T>::operator-(const Vector<U>& toSubtract) const -> Vector<decltype(T{} - U{})> {
    // check that dimensions are equal
    if ((m_data.size() != toSubtract.m_data.size()) || (m_is_row != toSubtract.m_is_row)) {
        throw std::invalid_argument("Vector subtraction requires that dimensions are equal.");
    }

    // add the two vectors
    vector<T> diff(m_data.size());

    for (int i=0; i<m_data.size(); i++) {
        diff[i] = m_data[i] - toSubtract.m_data[i];
    }

    return Vector(diff, m_is_row);
}

// dot product
template<typename T>
template<typename U>
decltype(T{} * U{}) Vector<T>::dot(const Vector<U>& toMultiply) const {
    
    // check that vector dimensions are equal
    if (m_dim != toMultiply.m_dim) {
        throw std::invalid_argument("Dot product requires that vector dimensions are equal.");
    }

    // throw error if vectors are empty
    if ((m_dim == 0) && (toMultiply.m_dim == 0)) {
        throw std::invalid_argument("Dot product requires that vectors are non-empty.");
    }

    // compute dot product
    T dotProduct = 0;

    for (size_t i = 0; i < m_dim; i++) {
        dotProduct += m_data[i]*toMultiply.m_data[i];
    }

    return dotProduct;
}

// method to matrix-multiply two vectors; this always returns a matrix
template<typename T>
template<typename U>
auto Vector<T>::operator*(const Vector<U>& toMultiply) const -> Matrix<decltype(T{} * U{})> {

    if ((m_dim == 0) && (toMultiply.m_dim == 0)) {
        std::vector<std::vector<T>> emptyVec = {{}};
        return Matrix(emptyVec);
    }

    if ((m_dim == 1) && (toMultiply.m_dim == 1)) {
        std::vector<std::vector<T>> matData(1, std::vector<T>(1, m_data[0] * toMultiply.m_data[0]));
        return Matrix(matData);
    }
    
    if (m_is_row) {
        if (toMultiply.m_is_row) { // if trying to multiply a row vector with a row vector
            throw std::invalid_argument("Cannot multiply two row Vectors.");
        }
        if (m_dim == toMultiply.m_dim) {
            T dot = 0;
            for (size_t i = 0; i < m_dim; i++) {
                dot += m_data[i] * toMultiply.m_data[i];
            }
            std::vector<std::vector<T>> matData(1, std::vector<T>(1, dot));
            return Matrix(matData);
        }
        else {
            throw std::invalid_argument("Cannot multiply two Vectors of different dimensions.");
        }
    }

    else { // if left vector is a column vector
        if (toMultiply.m_is_row) {
            std::vector<std::vector<T>> matData(m_dim, vector<T>(toMultiply.m_dim, 0.0f));
            for (size_t i = 0; i < m_dim; i++) {
                for (size_t j = 0; j < toMultiply.m_dim; j++) {
                    matData[i][j] = m_data[i]*toMultiply.m_data[j];
                }
            }
            return Matrix(matData);
        }
        else { // if toMultiply is a column vector
            if (toMultiply.m_dim == 1) {
                std::vector<std::vector<T>> matData(m_dim, vector<T>(1, 0.0f));
                for (size_t i = 0; i < m_dim; i++) {
                    matData[i][0] = m_data[i] * toMultiply.m_data[0];
                }
                return Matrix(matData);
            }
            else {
                throw std::invalid_argument("Cannot multiply two Vectors without corresponding dimensions.");
            }
        }
    }
}

// method to left-multiply a matrix by a row Vector; returns a matrix
template<typename T>
template<typename U>
auto Vector<T>::operator*(const Matrix<U>& mat) const -> Matrix<decltype(T{} * U{})> {
    // check that dimensions match
    if (!(m_is_row) && (m_dim != 1) && (m_dim != 0)) {
        throw std::invalid_argument("Vector must be a row vector to left-multiply a Matrix.");
    }

    if (m_dim != mat.getDims().first) {
        throw std::invalid_argument("Dimension of Vector must equal row-dimension of Matrix for left multiplication to proceed.");
    }
    using common_type = decltype(T{} * U{});
    if (m_dim == 0) {
        Matrix<common_type> product(0, 0);
        return product;
    }

    // compute product
    size_t numCols = mat.getDims().second;
    Matrix<common_type> product(1, numCols);
    for (size_t j = 0; j < numCols; j++) {

        // compute dot product of vector with each column
        T dotProduct = 0;
        for (size_t i = 0; i < m_dim; i++) {
            dotProduct += m_data[i] * mat.getData(i,j);

        }
        product.setData(dotProduct, 0, j);
    }
    return product;
}

// method to scalar multiply a vector
template<typename T>
template<typename U, typename>
auto Vector<T>::operator*(U scalar) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<U>();
    Vector product(m_data,m_is_row);

    for (size_t i = 0; i < m_dim; i++) {
        product.m_data[i] *= scalar;
    }

    return product;
}
/*
template<typename T, typename U>
auto operator*(U scalar, const Vector<T>& vec) -> typename std::enable_if<std::is_arithmetic<U>::value, Vector<decltype(U{} * T{})>>::type {
    assertTypesAreArithmetic<U>();
    return vec * scalar; // use previously defined operator
}
*/
// returns the 2norm of a vector
template<typename T>
T Vector<T>::norm() const {
    
    return sqrt(
        std::accumulate(m_data.begin(), m_data.end(), static_cast<long double>(0), [](long double acc, T val) {
            return acc + static_cast<long double>(val) * static_cast<long double>(val);
    }));
}

} // namespace LinearAlgebra