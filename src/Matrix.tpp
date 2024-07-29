#include "../include/Matrix.hpp"
#include "../include/Vector.hpp"
#include "../include/Utility.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <memory>
#include <iomanip>

namespace LinearAlgebra {

// helper function for 2d-std::vectors to determine if std::vectors are "empty"
template<typename T>
bool isEmpty(const std::vector<std::vector<T>>& vec) {
    assertTypesAreArithmetic<T>();
    return vec.empty() || std::all_of(
        vec.begin(), vec.end(), [](const std::vector<T>& elt) {
            return elt.empty();
        });
}

// default constructor for Matrix
template<typename T>
Matrix<T>::Matrix()
    : m_rows(0), m_cols(0), m_data({{}}) {
        assertTypesAreArithmetic<T>();
    }

// dimensional constructor for Matrix
template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : m_rows(rows), m_cols(cols), m_data(rows, std::vector<T>(cols, 0.0)) {
        assertTypesAreArithmetic<T>();
    }

// content constructor for Matrix
template<typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& data) {
    assertTypesAreArithmetic<T>();
    if (data.size() == 0) { // vector is empty
        std::vector<std::vector<T>> data2D = {{}};
        m_data = data2D;
        m_rows = 0;
        m_cols = 0;
    }

    else {
        // throw error if rows are of differing lengths
        size_t firstRowLen = data[0].size();
        for (size_t i = 0; i < data.size(); i++) {
            if (data[i].size() != firstRowLen) {
                throw std::invalid_argument("Rows must be of equal dimension.");
            }
        }
        
        if ((data.size() == 1) && (data[0].size() == 0)) { // vector is 2d-empty
            std::vector<std::vector<T>> data2D = {{}};
            m_data = data2D;
            m_rows = 0;
            m_cols = 0;
        }
        else {
            m_rows = data.size();
            m_cols = firstRowLen;
            m_data = data;
        }
    }
}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> data) {
    assertTypesAreArithmetic<T>();

    m_rows = data.size();
    m_cols = (m_rows > 0) ? data.begin()->size() : 0;

    m_data.reserve(m_rows);

    for (const auto& row : data) {
        if (row.size() != m_cols) {
            throw std::invalid_argument("Rows must be of equal dimension.");
        }
        m_data.push_back(std::vector<T>(row));
    }
}

template<typename T>
Matrix<T>::Matrix(const Vector<T>& vec) {
    assertTypesAreArithmetic<T>();
    size_t dim = vec.dim();
    if (dim == 0) {
        m_rows = 0;
        m_cols = 0;
        std::vector<std::vector<T>> zeroVec = {{}};
        m_data = zeroVec;
    }
    else if (vec.isRow()) {
        m_rows = 1;
        m_cols = dim;
        m_data = std::vector<std::vector<T>>(1);
        m_data[0] = vec.getData();
    }
    else if (!(vec.isRow())) {
        m_cols = 1;
        m_rows = dim;
        m_data = std::vector<std::vector<T>>(dim, std::vector<T>(1));
        for (size_t i = 0; i < dim; i++) {
            m_data[i][0] = vec.getData(i);
        }
    }
}

// identity "constructor"
template<typename T>
Matrix<T> Matrix<T>::identity(size_t dim) {
    Matrix<T> id(dim, dim);

    for (size_t i=0; i<dim; i++) {
        id.setData(1.0, i, i);
    }

    return id;
}

// Matrix destructor
template<typename T>
Matrix<T>::~Matrix() {}

// method to retrieve matrix dimensions
template<typename T>
std::pair<size_t, size_t> Matrix<T>::getDims() const {
    assertTypesAreArithmetic();
    std::pair<size_t, size_t> dims = {m_rows, m_cols};
    return dims;
}

template<typename T>
size_t Matrix<T>::rank() const {
    auto decompTuple = getQRDecomp(1e-6);
    return std::get<3>(decompTuple);
}

// method to retrieve matrix data
template<typename T>
std::vector<std::vector<T>> Matrix<T>::getData(
    const std::pair<size_t, size_t>& rowRange, 
    const std::pair<size_t, size_t>& colRange) const {

    size_t rowStart = rowRange.first;
    size_t rowEnd = rowRange.second;
    size_t colStart = colRange.first;
    size_t colEnd = colRange.second;

    // check that bounds are valid
    if ((rowEnd > m_rows) || (colEnd > m_cols) || (rowEnd < rowStart) || (colEnd < colStart)) {
        throw std::invalid_argument("Data range must be contained within matrix.");
    }

    // return empty 2D vector if no data is requested
    if ((rowEnd == rowStart) || (colEnd == colStart)) {
        std::vector<std::vector<T>> empty2DVec = {{}};
        return empty2DVec;
    }

    // otherwise, return data in specified range as a 2d vector
    std::vector<std::vector<T>> dataToGet(rowEnd - rowStart, std::vector<T>(colEnd - colStart));

    for (size_t i=rowStart; i<rowEnd; i++) {
        for (size_t j=colStart; j<colEnd; j++) {
            dataToGet[i - rowStart][j - colStart] = m_data[i][j];
        }
    }
    return dataToGet;
}

// overloaded methods to set default args for getData
template<typename T>
std::vector<std::vector<T>> Matrix<T>::getData() const {
    // note that data vector must contain at least one elt, the empty vector
    if (m_rows == 0) {
        return std::vector<std::vector<T>>(1); // return 2d empty vector
    }
    else {
        return getData({0, m_rows},{0, m_cols});
    }
}

template<typename T>
T Matrix<T>::getData(size_t row, size_t col) const {
    return getData({row,row+1}, {col,col+1})[0][0];
}

template<typename T>
std::vector<std::vector<T>> Matrix<T>::getData(size_t row, const std::pair<size_t, size_t>& colRange) const {
    return getData({row,row+1}, colRange);
}

template<typename T>
std::vector<std::vector<T>> Matrix<T>::getData(const std::pair<size_t, size_t>& rowRange, size_t col) const {
    // get unflattened data using range-based getData() method
    return getData(rowRange, {col,col+1});
}

template<typename T>
void Matrix<T>::print() const {
    print(5);
}
template<typename T>
void Matrix<T>::print(size_t precision) const {
    for (size_t i = 0; i < m_rows; i++) {
        for (size_t j = 0; j < m_cols; j++) {
            std::cout << std::setprecision(15) << m_data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// method to set matrix data
template<typename T>
void Matrix<T>::setData(
    const std::vector<std::vector<T>>& dataToSet, 
    const std::pair<size_t, size_t>& rowRange, 
    const std::pair<size_t, size_t>& colRange) {
    
    assertTypesAreArithmetic<T>();

    size_t rowStart = rowRange.first;
    size_t rowEnd = rowRange.second;
    size_t colStart = colRange.first;
    size_t colEnd = colRange.second;

    size_t rowInsertDiff = rowEnd - rowStart;
    size_t rowInsertLength = (rowInsertDiff) > 0 ? rowInsertDiff : 0;
    size_t colInsertDiff = colEnd - colStart;
    size_t colInsertLength = (colInsertDiff) > 0 ? colInsertDiff : 0;

    // check that matrix dimensions are zero if empty vector is passed
    if ((dataToSet.empty() || dataToSet[0].empty()) && ((rowInsertLength != 0) || (colInsertLength != 0))) {
        throw std::invalid_argument("Dimensions of input vector must match those of matrix.");
    }

    // check that vector length matches number of rows
    if (!(isEmpty(dataToSet)) && (dataToSet.size() != rowInsertLength)) {
        throw std::invalid_argument("Dimensions of input vector must match those of matrix.");
    }

    // check that vector element length matches number of columns
    if (!(dataToSet.empty())) {
        if (dataToSet[0].size() != colInsertLength) {
            throw std::invalid_argument("Dimensions of input vector must match those of matrix.");
        }
    }

    // check that given index bounds are within range
    if ((rowEnd > m_rows) || (colEnd > m_cols)) {
        throw std::invalid_argument("Data range must be contained within matrix.");
    }

    // set matrix data
    for (size_t i=rowStart; i<rowEnd; i++) {
        for (size_t j=colStart; j<colEnd; j++) {
            m_data[i][j] = dataToSet[i - rowStart][j - colStart];
        }
    }
}

// overloaded methods to set default data
template<typename T>
void Matrix<T>::setData(
    const std::vector<std::vector<T>>& dataToSet) {
        assertTypesAreArithmetic<T>();
        setData(dataToSet, {0,m_rows}, {0,m_cols});
}

template<typename T>
void Matrix<T>::setData(
    const std::vector<T>& dataToSet, 
    size_t row, 
    const std::pair<size_t, size_t>& colRange) {
        assertTypesAreArithmetic<T>();
        std::vector<std::vector<T>> toSetExpanded = {dataToSet};
        setData(toSetExpanded, {row,row+1}, colRange);
}

template<typename T>
void Matrix<T>::setData(
    const std::vector<T>& dataToSet, 
    const std::pair<size_t, size_t>& rowRange, 
    size_t col) {
        assertTypesAreArithmetic<T>();
        size_t rowDiff = (rowRange.second - rowRange.first > 0) ? rowRange.second - rowRange.first : 0;
        std::vector<std::vector<T>> toSetExpanded(
            rowDiff, std::vector<T>(1));
        for (size_t i = 0; i < rowDiff; i++) {
            toSetExpanded[i][0] = dataToSet[i];
        }
        setData(toSetExpanded, rowRange, {col,col+1});
}

template<typename T>
void Matrix<T>::setData(
    T dataToSet, 
    size_t row, 
    size_t col) {
        assertTypesAreArithmetic<T>();
        std::vector<std::vector<T>> dataAsVec = {{dataToSet}};
        setData(dataAsVec, {row,row+1}, {col,col+1});
}

template<typename T>
template<typename U>
bool Matrix<T>::operator==(const Matrix<U>& toCompare) {
    // check that dimensions are equal
    if ((m_rows != toCompare.m_rows) || (m_cols != toCompare.m_cols)) {
        return false;
    }
    // check that data is equal
    for (size_t i = 0; i < m_rows; i++) {
        for (size_t j = 0; j < m_cols; j++) {
            if (static_cast<decltype(T{}+U{})>(m_data[i][j]) != static_cast<decltype(T{}+U{})>(toCompare.m_data[i][j])) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
template<typename U, typename V>
bool Matrix<T>::isNear(const Matrix<U>& toCompare, V tolerance) {
    assertTypesAreArithmetic<V>();

    // check that dimensions are equal
    if ((m_rows != toCompare.m_rows) || (m_cols != toCompare.m_cols)) {
        return false;
    }

    // check that data is equal
    for (size_t i = 0; i < m_rows; i++) {
        for (size_t j = 0; j < m_cols; j++) {
            if (abs(m_data[i][j] - toCompare.m_data[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// method to add two matrices
template<typename T>
template<typename U>
auto Matrix<T>::operator+(const Matrix<U>& toAdd) const -> Matrix<decltype(T{} + U{})> {
    using common_type = decltype(T{} + U{});

    // check that summand dimensions match
    if ((m_rows != toAdd.m_rows) || (m_cols != toAdd.m_cols)) {
        throw std::invalid_argument("Matrix addition requires that summand dimensions match.");
    }

    // return the sum of each matrix
    Matrix<common_type> result(m_rows, m_cols);

    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=0; j<m_cols; j++) {
            result.m_data[i][j] = m_data[i][j] + toAdd.m_data[i][j];
        }
    }

    return result;
}

// method to subtract two matrices
template<typename T>
template<typename U>
auto Matrix<T>::operator-(const Matrix<U>& toSubtract) const -> Matrix<decltype(T{} - U{})> {
    using common_type = decltype(T{} - U{});

    // check that summand dimensions match
    if ((m_rows != toSubtract.m_rows) || (m_cols != toSubtract.m_cols)) {
        throw std::invalid_argument("Matrix subtraction requires that subtrahend and minuend dimensions match.");
    }

    // return the sum of each matrix
    Matrix<common_type> result(m_rows, m_cols);

    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=0; j<m_cols; j++) {
            result.m_data[i][j] = m_data[i][j] - toSubtract.m_data[i][j];
        }
    }

    return result;
}

// method to multiply two matrices
template<typename T>
template<typename U>
auto Matrix<T>::operator*(const Matrix<U>& rightMultiplier) const -> Matrix<decltype(T{} * U{})> {
    using common_type = decltype(T{} * U{});

    // check that dimensions match
    if (m_cols != rightMultiplier.m_rows) {
        throw std::invalid_argument("Matrix multiplication requires that columns of left multiplier match rows of right multiplier.");
    }

    if ((m_rows == 0) || (rightMultiplier.m_cols == 0)) {
        throw std::invalid_argument("Matrix multiples must be nonempty.");
    }

    // multiply matrices element by element
    Matrix<common_type> result(m_rows, rightMultiplier.m_cols);

    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=0; j<rightMultiplier.m_cols; j++) {

            // compute dot product of current column-row pair
            T dotProduct = 0;

            for (size_t index=0; index<m_cols; index++) {
                dotProduct += m_data[i][index]*rightMultiplier.m_data[index][j];
            }

            result.m_data[i][j] = dotProduct;
        }
    }

    return result;
}

// method to left-multiply a Vector by a matrix
template<typename T>
template<typename U>
auto Matrix<T>::operator*(const Vector<U>& vec) const -> Matrix<decltype(T{} * U{})> {
    using common_type = decltype(T{} * U{});

    if ((vec.dim() == 0) && (m_cols == 0)) {
        return Matrix<common_type>(0,0);
    }
    
    if (vec.dim() == 1) {
        return Matrix<common_type>::operator*(vec.getData(0));
    }

    // check that dimensions match
    if (vec.isRow()) {
        if (m_cols == 1) {
            return m_data[0][0] * vec;
        }
        else {
            throw std::invalid_argument("Vector must be a column vector to right-multiply a Matrix.");
        }
    }

    if (vec.dim() != m_cols) {
        throw std::invalid_argument("Dimension of Vector must equal column-dimension of Matrix for left multiplication to proceed.");
    }

    // compute product
    Matrix<common_type> product(m_rows, 1);
    for (size_t i = 0; i < m_rows; i++) {

        // compute dot product of row with vector
        T dotProduct = 0;
        for (size_t j = 0; j < m_cols; j++) {
            dotProduct += m_data[i][j]*vec.getData(j);
        }
        product.m_data[i][0] = dotProduct;
    }

    return product;
}

template<typename T>
template<typename U, typename>
auto Matrix<T>::operator*(U scalar) const -> Matrix<decltype(T{} * U{})> {
    assertTypesAreArithmetic<U>();

    using common_type = decltype(T{} * U{});
    Matrix<common_type> result(m_rows, m_cols);

    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=0; j<m_cols; j++) {
            result.m_data[i][j] = scalar * m_data[i][j];
        }
    }

    return result;
}
/*
template<typename T, typename U>
auto operator*(U scalar, const Matrix<T>& mat) -> typename std::enable_if<std::is_arithmetic<U>::value, Matrix<decltype(U{} * T{})>>::type {
    return mat * scalar;
}
*/
// method to compute the transpose of a matrix
template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    std::vector<std::vector<T>> transpose_data(m_cols, std::vector<T>(m_rows));

    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=0; j<m_cols; j++) {
            transpose_data[j][i] = m_data[i][j];
        }
    }
    return Matrix(transpose_data);
}

// method to swap a portion of a row or column with another
template<typename T>
void Matrix<T>::interchange(const std::pair<size_t, size_t>& toSwap, const std::pair<size_t, size_t>& range, size_t index) {
    // check index
    size_t maxBound;
    size_t maxBoundForDimToSwap;
    if (index == 0) {
        maxBound = m_cols;
        maxBoundForDimToSwap = m_rows;
    }
    else if (index == 1) {
        maxBound = m_rows;
        maxBoundForDimToSwap = m_cols;
    }
    else {
        throw std::invalid_argument("Index must be 0 or 1 to interchange rows or columns.");
    }
    
    // check that bounds are valid
    if ((toSwap.first >= maxBoundForDimToSwap) || (toSwap.second >= maxBoundForDimToSwap)) {
        throw std::invalid_argument("Row or column indices must be less than total row or column dimensionality.");
    }

    if ((range.first >= maxBound) || (range.second > maxBound)) {
        throw std::invalid_argument("Range of elements to interchange must be within range of matrix.");
    }

    size_t swapIndexA = toSwap.first;
    size_t swapIndexB = toSwap.second;
    size_t rangeStart = range.first;
    size_t rangeEnd = range.second;
    // swap elements according to indices and range
    if (index == 0) {
        for (size_t i = rangeStart; i < rangeEnd; i++) {
            T tempData = m_data[swapIndexA][i];
            m_data[swapIndexA][i] = m_data[swapIndexB][i];
            m_data[swapIndexB][i] = tempData;
        }
    }
    else {
        for (size_t i = rangeStart; i < rangeEnd; i++) {
            T tempData = m_data[i][swapIndexA];
            m_data[i][swapIndexA] = m_data[i][swapIndexB];
            m_data[i][swapIndexB] = tempData;
        }
    }
}

template<typename T>
void Matrix<T>::interchange(const std::pair<size_t, size_t>& toSwap, size_t index) {
    size_t maxBound;
    // perform index check. You should abstract this out to optimize, since it gets called in the other interchange, too.
    if (index == 0) {
        maxBound = m_cols;
    }
    else if (index == 1) {
        maxBound = m_rows;
    }
    else {
        throw std::invalid_argument("Index must be 0 or 1 to interchange rows or columns.");

    }
    interchange(toSwap, {0, maxBound}, index);
}

// method to determine whether matrix is square
template<typename T>
bool Matrix<T>::isSquare() const {
    return (m_rows == m_cols);
}

// method to determine whether matrix is symmetric
template<typename T>
template<typename U>
bool Matrix<T>::isSymmetric(U tol) const {
    assertTypesAreArithmetic<U>();
    
    if (!(isSquare())) {return false;}

    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=0; j<i; j++) {
            if (abs(m_data[i][j] - m_data[j][i]) > tol) {
                return false;
            }
        }
    }

    return true;
}

// method to solve a system of linear equations
template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveSystem(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();
    assertCanFormSystemWith(vecB);

    // call helper functions depending on nature of system

    // use QR decomposition if matrix is not square
    if (m_rows != m_cols) {
        return solveQRUnsafe(vecB, tol);
    }

    // use forward- or back-substition if matrix is triangular
    if (isUpperTriangular(tol)) {
        return solveBackSubUnsafe(vecB, tol);
    }

    if (isLowerTriangular(tol)) {
        return solveForwardSubUnsafe(vecB, tol);
    }

    // use LU decomposition if matrix is 16-by-16 or smaller
    if ((m_rows <= 16) && (m_cols <= 16)) {
        return solveLUUnsafe(vecB, tol);
    }

    // if matrix could be symmetric positive-definite, attempt Cholesky decomposition
    if (isSymmetric(tol) && (m_data[0][0] > 0) && diagHasSameSign()) {
        try {
            Vector choleskySol = solveCholeskyUnsafe(vecB, tol);
            return choleskySol;
        }
        catch (const std::runtime_error& error) {}
    }

    // default to LU decomposition
    return solveLUUnsafe(vecB, tol); 
}

// determines if a square matrix is upper-triangular
template<typename T>
template<typename U>
bool Matrix<T>::isUpperTriangular(U tol) const {
    assertTypesAreArithmetic<U>();

    if (!(isSquare())) {        return false;}

    // check that each element below the diagonal is 0
    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=0; j<i; j++) {
            if (abs(m_data[i][j]) > tol) {
                return false;
            }
        }
    }

    return true;
}

// determines if a square matrix is lower-triangular
template<typename T>
template<typename U>
bool Matrix<T>::isLowerTriangular(U tol) const {
    assertTypesAreArithmetic<U>();

    if (!(isSquare())) {return false;}

    // check that each element above the diagonal is 0
    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=i+1; j<m_rows; j++) {
            if (abs(m_data[i][j]) > tol) {
                return false;
            }
        }
    }

    return true;
}

// determines if a square matrix is upper Hessenberg
template<typename T>
template<typename U>
bool Matrix<T>::isUpperHessenberg(U tol) const {
    assertTypesAreArithmetic<U>();

    if (!(isSquare())) {return false;}

    // check that each element below the first subdiagonal is 0
    for (size_t j = 0; j < m_cols; j++) {
        for (size_t i = j+2; i < m_rows; i++) {
            if (abs(m_data[i][j]) > tol) {
                return false;
            }
        }
    }

    return true;
}

// determines if a square matrix is lower Hessenberg
template<typename T>
template<typename U>
bool Matrix<T>::isLowerHessenberg(U tol) const {
    assertTypesAreArithmetic<U>();

    if (!(isSquare())) {return false;}

    // check that each element above the first superdiagonal is 0
    for (size_t i = 0; i < m_rows; i++) {
        for (size_t j = i+2; j < m_cols; j++) {
            if (abs(m_data[i][j]) > tol) {
                return false;
            }
        }
    }

    return true;
}

// determines if a square matrix is Hermitian 
// (to be expanded once complex no. functionality is added)
template<typename T>
template<typename U>
bool Matrix<T>::isHermitian(U tol) const {
    assertTypesAreArithmetic<U>();

    if (!(isSquare())) {return false;}

    for (size_t i=0; i<m_rows; i++) {
        for (size_t j=0; j<i; j++) {
            if (abs(m_data[i][j] - m_data[j][i]) > tol) {
                return false;
            }
        }
    }

    return true;
}

// determines if all elements on the diagonal of a square matrix have the same sign
template<typename T>
bool Matrix<T>::diagHasSameSign() const {

    if (!(isSquare())) {return false;}
    if ((m_rows == 0) || (m_cols == 0)) {return true;}
    if (m_data[0][0] == 0.0) {return false;}

    bool firstEltIsPositive = (m_data[0][0] > 0.0) ? true : false;
    
    for (size_t i=0; i<m_rows; i++) {
        bool eltIsPositive = (m_data[i][i] > 0.0) ? true : false;
        if ((firstEltIsPositive != eltIsPositive) || (m_data[i][i] == 0.0)) {
            return false;
        }
    }

    return true;
}

// NEED TO TEST THIS FUNCTION
template<typename T>
template<typename U>
bool Matrix<T>::diagIsNonZero(U tol) const {
    
    assertTypesAreArithmetic<U>();
    if (!(isSquare())) {return false;}
    if ((m_rows == 0) || (m_cols == 0)) {return true;}

    for (size_t i=0; i<m_rows; i++) {
        if (abs(U(m_data[i][i])) < tol) {
            return false;
        }
    }

    return true;
}

// determines if a matrix can form a system of equations with the given vector
template<typename T>
template<typename U>
void Matrix<T>::assertCanFormSystemWith(const Vector<U>& vecB) const {
    if (m_rows != vecB.dim()) {
        throw std::invalid_argument("Matrix and vector dimensions must agree to form a system.");
    }

    if (vecB.isRow() && (vecB.dim() > 1)) {
        throw std::invalid_argument("Cannot form a system with a row vector.");
    }
}

template<typename T>
template<typename U>
std::vector<size_t> Matrix<T>::sortWithScrambler(std::function<U(const std::vector<T>&)> functional, const size_t index) {
    // make sure the function to sort by returns a type with defined ordering operations
    assertTypesAreArithmetic<U>();

    // ensure index is 0 or 1
    if (!(index == 0) && !(index == 1)) {
        throw std::invalid_argument("Index must be 0 or 1 to sort rows or columns.");
    }

    // make vector of indices
    size_t maxBound = (index == 0) ? m_rows : m_cols;
    std::vector<size_t> indices(maxBound);
    std::iota(indices.begin(), indices.end(), 0);

    // define lambda that sorts indices based on the corresponding values of the given vector
    auto sortByVal = [](std::vector<size_t>& indices, const std::vector<U>& vec) {
        std::sort(
            indices.begin(), indices.end(), [&vec](const size_t indexOne, const size_t indexTwo) {
                return vec[indexOne] > vec[indexTwo];
        });
    };

    // apply desired function to each row (if index is 0) or column (if index is 1), then apply sorting function
    if (index == 0) {
        std::vector<U> rowVals(m_rows);

        for (size_t i = 0; i < m_rows; i++) {
            rowVals[i] = functional(m_data[i]);
        }

        sortByVal(indices, rowVals);

        // convert to scrambler format (i.e., if the vector represents a mapping from index to index, construct its inverse)
        std::vector<size_t> rowScrambler(m_rows);
        for (size_t i=0; i < m_rows; i++) {
            rowScrambler[indices[i]] = i;
        }
        return rowScrambler;
    }

    else { // index == 1
        std::vector<U> colVals(m_cols);

        for (size_t j = 0; j < m_cols; j++) {
            std::vector<T> colVec(m_rows);
            for (size_t i = 0; i < m_rows; i++) {
                colVec[i] = m_data[i][j];
            }
            colVals[j] = functional(colVec);
        }

        sortByVal(indices, colVals);

        // convert to scrambler format (i.e., if the vector represents a mapping from index to index, construct its inverse)
        std::vector<size_t> colScrambler(m_cols);
        for (size_t i=0; i < m_cols; i++) {
            colScrambler[indices[i]] = i;
        }

        return colScrambler;
    }
}

template<typename T>
void Matrix<T>::scramble(const std::vector<size_t>& indices, size_t index) {
    if (!(index == 0) && !(index == 1)) {
        throw std::invalid_argument("Index must be 0 or 1 to sort rows or columns.");
    }

    size_t maxDim = (index == 0) ? m_rows : m_cols;
    if (indices.size() != maxDim) {
        throw std::invalid_argument("Index vector must have same length as Vector.");
    }
    
    std::vector<bool> visitedIndices(maxDim, false);
    std::vector<std::vector<T>> scrambledData(m_rows, std::vector<T>(m_cols));

    for (size_t i=0; i < maxDim; i++) {
        size_t indexFromVec = indices[i];
        if (indexFromVec >= maxDim) {
            throw std::invalid_argument("All indices must be within range to scramble.");
        }
        if (visitedIndices[indexFromVec] == true) {
            throw std::invalid_argument("List of indices must be unique to scramble.");
        }
        if (index == 0) {
            scrambledData[indexFromVec] = m_data[i];
        }
        if (index == 1) {
            for (size_t j=0; j < m_rows; j++) {
                scrambledData[j][indexFromVec] = m_data[j][i];
            }
        }
        visitedIndices[indexFromVec] = true;
    }

    m_data = scrambledData;
}

template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveQR(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertCanFormSystemWith(vecB);
    return solveQRUnsafe(vecB, tol);
}

// solves an overdetermined, full-column-rank linear system of equations using QR-decomposition with the Householder method
template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveQRUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    if ((m_rows == 0) || (m_cols == 0)) {
        return Vector<T>(0, false);
    }

    using common_type = decltype(T{} * 1.0);
    
    // get QR decomposition
    auto decompTuple = getQRDecomp(tol);
    std::vector<Vector<common_type>> householderVecs = std::get<0>(decompTuple);
    Matrix<common_type> R = std::get<1>(decompTuple);
    std::vector<size_t> pivotTracker = std::get<2>(decompTuple);
    size_t rank = std::get<3>(decompTuple);

    // compute cHat, which will take the value Q.transpose() * vecB, where Q is an orthonormal matrix and the product
    // is computed implicitly using the Householder Vectors
    Vector<common_type> cHat(vecB.dim(), false);

    size_t maxDim = (m_rows > m_cols) ? m_rows : m_cols;
    size_t redundantDims = maxDim - rank;
    Matrix<common_type> RExtracted = Matrix(R.getData({0, rank}, {0, rank}));
    cHat = vecB;
    for (size_t k = 0; k < rank; k++) {
        // update cHat
        Vector<common_type> cHatDataToGet = Vector(cHat.getData({k, m_rows}));
        Vector<common_type> cHatDataToSet = cHatDataToGet - Vector(
            2 * householderVecs[k] * householderVecs[k].transpose() * cHatDataToGet);
        cHat.setData(cHatDataToSet.getData(), {k, m_rows});
    }
    // extract the first m_cols rows of R and cHat, and solve the system using back-substitution
    Vector<common_type> c = Vector<common_type>(cHat.getData({0, rank}));

    // solve or approximate the solution
    Vector<decltype(U{} * T{})> fullRankSol = RExtracted.solveBackSub(c, tol); // you should actually calculate different tolerances depending on the floating-point error of the accumulated calculations
    Vector<decltype(U{} * T{})> sol(m_cols, false);
    sol.setData(fullRankSol.getData(), {0, rank});
    sol.scramble(pivotTracker); // since pivoting produced a column permutation, permute the solution before returning
    return sol;
}

template<typename T>
template<typename V>
// when called on a non-empty, overdetermined Matrix, returns relevant components of QR decomposition
auto Matrix<T>::getQRDecomp(V tol) const 
    -> std::tuple<std::vector<Vector<decltype(T{} * 1.0)>>, Matrix<decltype(T{} * 1.0)>, std::vector<size_t>, size_t> {

    size_t rank = 0;

    using common_type = decltype(T{} * 1.0);
    // initialize copy of given matrix in order to modify data
    Matrix<common_type> R(m_data);
    // initialize std::vector of Householder Vectors
    std::vector<Vector<common_type>> householderVecs(m_cols);
    
    // initialize pivot tracker
    std::vector<size_t> pivotTracker(m_cols);
    std::iota(pivotTracker.begin(), pivotTracker.end(), 0);

    // reduce R to upper-triangular form
    for (size_t k = 0; k < m_cols; k++) {

        // compute greatest norm in the sub-matrix being considered
        T maxNorm = 0.0;
        size_t maxNormCol = 0;
        for (size_t i = k; i < m_cols; i++) {
            Vector<T> colAsVector(R.getData({k, m_rows}, i));
            T colNorm = colAsVector.norm();
            
            if (colNorm > maxNorm) {
                maxNorm = colNorm;
                maxNormCol = i;
            }
        }

        // break if all remaining norms are 0 (within the tolerance)
        if (abs(maxNorm) < tol) {
            break;
        }

        rank += 1; // increment rank since there are more linearly independent columns

        // swap the current column with the column with the greatest norm 
        R.interchange({k, maxNormCol}, 1);
        size_t colIndex = pivotTracker[k];
        pivotTracker[k] = pivotTracker[maxNormCol];
        pivotTracker[maxNormCol] = colIndex;

        // compute vectors
        Vector<common_type> yVec = Vector<common_type>(R.getData({k, m_rows}, k));
        if (yVec.isRow()) {yVec = yVec.transpose();}; // transpose if is row vec, which can happen with 1x1 data
        int sign_y = (yVec.getData(0) > 0) ? 1 : -1; // note that yVec[0] will always be accessible since we assume m_rows > m_cols

        // create first basis vector with the same dimension as yVec
        Vector<common_type> e1 = Vector(std::vector<common_type>(m_rows - k), false);
        e1.setData(common_type(1.0), 0);

        // calculate and normalize k-th Householder Vector
        Vector<common_type> vk = yVec + (sign_y * yVec.norm() * e1); // calculate
        common_type vkNorm = vk.norm();
        if (vkNorm > tol) { // you seriously need to stop using a constant tolerance for all your operations. this should depend directly on accumulated error
            vk = vk * (1 / vkNorm); // normalize - YOU NEED TO WRITE A SCALAR DIVISION OPERATOR FOR VECTORS AND MATRICES
        }

        householderVecs[k] = vk; // save
        // update R - this update involves a ton of flip-flop typecasting and needs to be streamlined, probably with the help of an additional data accessor member function
        Matrix dataToGet(R.getData({k, m_rows}, {k, m_cols}));
        Matrix dataToSet = dataToGet - (2 * vk * vk.transpose() * Matrix(dataToGet));
        R.setData(dataToSet.getData(), {k, m_rows}, {k, m_cols});
    }

    return std::tuple<std::vector<Vector<common_type>>, Matrix<common_type>, std::vector<size_t>, size_t>{householderVecs, R, pivotTracker, rank};
}

template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveLU(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();
    if (m_rows != m_cols) {
        throw std::invalid_argument("Only square matrices can be decomposed into LU matrices.");
    }
    assertCanFormSystemWith(vecB);
    return solveLUUnsafe(vecB, tol);
}

template<typename T>
template<typename U, typename V>
// solves a square linear system of equations using LU factorization, assuming matrix is square and vector has correct dimensionality
auto Matrix<T>::solveLUUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();

    using common_type = decltype(T{} * U{});
    // cover empty case
    if (m_rows == 0) {
        return Vector<common_type>(0, false);
    }

    // instantiate L, U, and pivot-tracking vector
    Matrix<common_type> LMat = identity(m_rows);
    Matrix<common_type> UMat = Matrix(m_data);
    std::vector<size_t> pivotTracker(m_rows);
    std::iota(pivotTracker.begin(), pivotTracker.end(), 0);

    // perform LU factorization
    for (size_t k = 0; k < m_rows-1; k++) {
        
        // find row with the greatest pivot
        T maxPivot = abs(m_data[k][k]);
        size_t maxPivotRow = k;
        for (size_t i = k+1; i < m_rows; i++) {
            if (abs(m_data[i][k]) > maxPivot) {
                maxPivot = m_data[i][k];
                maxPivotRow = i;
            }
        }

        if (maxPivot < tol) {
            throw std::runtime_error("No pivot meets the tolerance threshold for stable factorization.");
        }

        UMat.interchange({k, maxPivotRow}, {k, m_rows}, 0);
        LMat.interchange({k, maxPivotRow}, {0, k}, 0);

        // update pivot tracker, swapping indices in the corresponding rows
        size_t rowIndex = pivotTracker[k];
        pivotTracker[k] = pivotTracker[maxPivotRow];
        pivotTracker[maxPivotRow] = rowIndex;

        for (size_t i=k+1; i<m_rows; i++) {
            
            // compute multipliers
            LMat.m_data[i][k] = UMat.m_data[i][k]/UMat.m_data[k][k];

            // perform row operations
            Matrix subMatOfU(UMat.getData(i, {k, m_rows}));
            Matrix product(UMat.getData(k, {k, m_rows}));
            Matrix subtrahend = LMat.m_data[i][k] * product;
            
            subMatOfU = subMatOfU - subtrahend;

            // update data in U to match
            UMat.setData(subMatOfU.getData(), {i,i+1}, {k,m_rows});
        }
    }

    // solve the systems of equations
    Vector<common_type> vecBCopy = vecB;
    vecBCopy.scramble(pivotTracker);
    Vector<common_type> vecY = LMat.solveForwardSubUnsafe(vecBCopy, tol);
    Vector<common_type> solution = UMat.solveBackSubUnsafe(vecY, tol);

    return solution;
}

template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveBackSub(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();
    assertCanFormSystemWith(vecB);
    
    if (!isUpperTriangular(tol)) {
        throw std::invalid_argument("Matrix must be upper-triangular to perform back substitution.");
    }
    return solveBackSubUnsafe(vecB, tol);   
}

// solves a square linear system of equations using back substitution;
// assumes that matrix is upper-triangular
template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveBackSubUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();

    using common_type = decltype(T{} * U{});

    // instantiate solution vector
    Vector<common_type> solution(m_rows, false);
    // iteratively compute values of solution vector, beginning with the last element
    for (size_t i=0; i < m_rows; i++) {
        size_t currRow = m_rows - 1 - i;
        // throw an error if matrix is singular
        if (abs(m_data[currRow][currRow]) < tol) {
            throw std::invalid_argument("Matrix is not non-singular enough to perform back substitution.");
        }
        T varSolution = vecB.getData(currRow);
        
        for (size_t currCol = currRow+1; currCol < m_rows; currCol++) {
            varSolution -= m_data[currRow][currCol]*solution.getData(currCol);
        }
        varSolution /= m_data[currRow][currRow];
        solution.setData(varSolution, currRow);
    }

    return solution;
}

template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveForwardSub(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();
    assertCanFormSystemWith(vecB);

    if (!isLowerTriangular(tol)) {
        throw std::invalid_argument("Matrix must be lower-triangular to perform forward substitution.");
    }
    return solveForwardSubUnsafe(vecB, tol);
}

// solves a square linear system of equations using forward substitution;
// assumes matrix is lower-triangular
template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveForwardSubUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();
    using common_type = decltype(T{} * U{});

    // instantiate solution vector
    Vector<common_type> solution(m_rows, false);

    // iteratively compute values of solution vector, beginning with the first element
    for (size_t currRow=0; currRow<m_rows; currRow++) {

        // throw an error if matrix is singular
        if (abs(m_data[currRow][currRow]) < tol) {
            throw std::invalid_argument("Matrix is not non-singular enough to perform forward substitution.");
        }

        T varSolution = vecB.getData(currRow);

        for (size_t currCol = 0; currCol < currRow; currCol++) {
            varSolution -= m_data[currRow][currCol]*solution.getData(currCol);
        }
        varSolution /= m_data[currRow][currRow];
        solution.setData(varSolution, currRow);
    }

    return solution;
}

template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveCholesky(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();
    assertCanFormSystemWith(vecB);
    return solveCholeskyUnsafe(vecB, tol);
}

// solves a square linear system of equations using Cholesky decomposition,
// or throws an error if Cholesky fails (i.e. matrix is not symmetric positive-definite)
template<typename T>
template<typename U, typename V>
auto Matrix<T>::solveCholeskyUnsafe(const Vector<U>& vecB, V tol) const -> Vector<decltype(T{} * U{})> {
    assertTypesAreArithmetic<V>();

    // check that matrix is symmetric
    if (!isSymmetric(tol)) {
        throw std::invalid_argument("Cholesky decomposition requires a symmetric matrix.");
    }

    using common_type = decltype(T{} * U{});
    // initialize matrix R as the given matrix
    Matrix<common_type> L(m_data);

    for (size_t i = 0; i < m_rows; i++) {
        for (size_t j = 0; j <= i; j++) {
            T sum = 0;
            for (size_t k = 0; k < j; k++)
                sum += L.m_data[i][k] * L.m_data[j][k];

            if (i == j) {
                if (m_data[i][i] - sum < 0) {
                    throw std::invalid_argument("Cholesky failed due to non-positive element in the diagonal of L.");
                }
                L.m_data[i][j] = sqrt(m_data[i][i] - sum);
            }
            else {
                if (abs(L.m_data[i][i]) < tol) {
                    throw std::invalid_argument("Cholesky failed due to division by a number intolerably close to zero.");
                }
                L.m_data[i][j] = (1.0 / L.m_data[j][j]) * (m_data[i][j] - sum);
            }
        }
    }
    Vector<common_type> vecY = L.solveForwardSubUnsafe(vecB, tol);
    Vector<common_type> solution = L.transpose().solveBackSubUnsafe(vecY, tol);
    return solution;
}
} // namespace LinearAlgebra