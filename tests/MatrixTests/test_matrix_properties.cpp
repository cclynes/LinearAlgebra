#include "../../include/Matrix.hpp"
#include "../../include/Vector.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "BaseMatrixFixture.hpp"

using ::testing::ContainerEq;
using namespace LinearAlgebra;

class MatrixPropsFixture : public ::BaseMatrixFixture {
protected:
    std::vector<std::vector<double>> threeDimSymmetricData;
    std::vector<std::vector<double>> threeDimAsymmetricData;
    std::vector<std::vector<double>> almostSymmetricExtraColData;
    std::vector<std::vector<double>> almostSymmetricExtraRowData;

    Matrix<double> zerosMat;
    Matrix<double> zerosMatExtraRow;
    Matrix<double> zerosMatExtraCol;
    Matrix<double> threeDimSymmetric;
    Matrix<double> threeDimAsymmetric;
    Matrix<double> almostSymmetricExtraCol;
    Matrix<double> almostSymmetricExtraRow;

    virtual void SetUp() override {
        BaseMatrixFixture::SetUp();

        zerosMat = Matrix<double>(5,5);
        zerosMatExtraRow = Matrix<double>(6,5);
        zerosMatExtraCol = Matrix<double>(5,6);

        threeDimSymmetricData = {
            {1.0, -2.0, 15.5},
            {-2.0, 3.3, -12.0},
            {15.5, -12.0, -100.0}};
        threeDimSymmetric = Matrix<double>(threeDimSymmetricData);

        threeDimAsymmetricData = {
            {1.0, -2.0, 15.5},
            {-2.0, 3.3, 12.0},
            {15.5, -12.0, -100.0}};
        threeDimAsymmetric = Matrix<double>(threeDimAsymmetricData);

        almostSymmetricExtraColData = {
            {1.0, 2.0, 0.0},
            {2.0, 1.0, 0.0}};
        almostSymmetricExtraCol = Matrix<double>(almostSymmetricExtraColData);

        almostSymmetricExtraRowData = {
            {1.0, 2.0},
            {2.0, 1.0},
            {0.0, 0.0}};
        almostSymmetricExtraRow = Matrix<double>(almostSymmetricExtraRowData);
    }
};

TEST_F(MatrixPropsFixture, MatrixIsSymmetricWorks) {
    EXPECT_TRUE(defaultMat.isSymmetric(0));
    EXPECT_TRUE(singleEltMat.isSymmetric(0));
    EXPECT_TRUE(Matrix<double>::identity(5).isSymmetric(0));
    EXPECT_TRUE(threeDimSymmetric.isSymmetric(0));
    EXPECT_FALSE(threeDimAsymmetric.isSymmetric(0));
    EXPECT_FALSE(almostSymmetricExtraCol.isSymmetric(0));
    EXPECT_FALSE(almostSymmetricExtraRow.isSymmetric(0));
}

TEST_F(MatrixPropsFixture, MatrixIsUpperTriangularWorks) {
    EXPECT_TRUE(defaultMat.isUpperTriangular(0));
    EXPECT_TRUE(singleEltMat.isUpperTriangular(0));
    EXPECT_TRUE(zerosMat.isUpperTriangular(0));

    std::vector<std::vector<double>> threeDimUTData = {
        {1.0, -7.0, 2.5},
        {0.0, 0.5, -12.0},
        {0.0, 0.0, -1.0}};
    Matrix<double> threeDimUT(threeDimUTData);
    EXPECT_TRUE(threeDimUT.isUpperTriangular(0));

    std::vector<std::vector<double>> threeDimAlmostUTData = threeDimUTData;
    threeDimAlmostUTData[2][1] = 0.1;
    Matrix<double> threeDimAlmostUT(threeDimAlmostUTData);
    EXPECT_FALSE(threeDimAlmostUT.isUpperTriangular(0));

    std::vector<std::vector<double>> threeDimAlmostUTExtraRowData = {
        {1.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}};
    Matrix<double> threeDimAlmostUTExtraRow(threeDimAlmostUTExtraRowData);
    EXPECT_FALSE(threeDimAlmostUTExtraRow.isUpperTriangular(0));

    std::vector<std::vector<double>> threeDimAlmostUTExtraColData = {
        {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}};
    Matrix<double> threeDimAlmostUTExtraCol(threeDimAlmostUTExtraColData);
    EXPECT_FALSE(threeDimAlmostUTExtraCol.isUpperTriangular(0));
}

TEST_F(MatrixPropsFixture, MatrixIsLowerTriangularWorks) {
    EXPECT_TRUE(defaultMat.isLowerTriangular(0));
    EXPECT_TRUE(singleEltMat.isLowerTriangular(0));
    EXPECT_TRUE(zerosMat.isLowerTriangular(0));

    std::vector<std::vector<double>> threeDimLTData = {
        {1.0, 0.0, 0.0},
        {-7.0, 0.0, 0.0},
        {2.0, -12.0, -1.0}};
    Matrix<double> threeDimLT(threeDimLTData);
    EXPECT_TRUE(threeDimLT.isLowerTriangular(0));

    std::vector<std::vector<double>> threeDimAlmostLTData = threeDimLTData;
    threeDimAlmostLTData[1][2] = 0.1;
    Matrix<double> threeDimAlmostLT(threeDimAlmostLTData);
    EXPECT_FALSE(threeDimAlmostLT.isLowerTriangular(0));

    std::vector<std::vector<double>> threeDimAlmostLTExtraRowData = {
        {1.0, 0.0}, {1.0, 1.0}, {1.0, 1.0}};
    Matrix<double> threeDimAlmostLTExtraRow(threeDimAlmostLTExtraRowData);
    EXPECT_FALSE(threeDimAlmostLTExtraRow.isUpperTriangular(0));

    std::vector<std::vector<double>> threeDimAlmostLTExtraColData = {
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    Matrix<double> threeDimAlmostLTExtraCol(threeDimAlmostLTExtraColData);
    EXPECT_FALSE(threeDimAlmostLTExtraCol.isUpperTriangular(0));
}

TEST_F(MatrixPropsFixture, MatrixIsUpperHessenbergWorks) {
    EXPECT_TRUE(defaultMat.isUpperHessenberg(0));
    EXPECT_TRUE(singleEltMat.isUpperHessenberg(0));
    EXPECT_TRUE(zerosMat.isUpperHessenberg(0));
    EXPECT_TRUE(twoByTwo.isUpperHessenberg(0));
    EXPECT_FALSE(zerosMatExtraRow.isUpperHessenberg(0));
    EXPECT_FALSE(zerosMatExtraCol.isUpperHessenberg(0));

    std::vector<std::vector<double>> upperHessData = {
        {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}};
    Matrix<double> upperHess(upperHessData);
    EXPECT_TRUE(upperHess.isUpperHessenberg(0));

    upperHessData[2][0] = 1.0f;
    Matrix<double> notUpperHess(upperHessData);
    EXPECT_FALSE(notUpperHess.isUpperHessenberg(0));
}

TEST_F(MatrixPropsFixture, MatrixIsLowerHessenbergWorks) {
    EXPECT_TRUE(defaultMat.isLowerHessenberg(0));
    EXPECT_TRUE(singleEltMat.isLowerHessenberg(0));
    EXPECT_TRUE(zerosMat.isLowerHessenberg(0));
    EXPECT_TRUE(twoByTwo.isLowerHessenberg(0));
    EXPECT_FALSE(zerosMatExtraRow.isLowerHessenberg(0));
    EXPECT_FALSE(zerosMatExtraCol.isLowerHessenberg(0));

    std::vector<std::vector<double>> lowerHessData = {
        {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}};
    Matrix lowerHess(lowerHessData);
    EXPECT_TRUE(lowerHess.isLowerHessenberg(0));

    lowerHessData[0][2] = 1.0f;
    Matrix notLowerHess(lowerHessData);
    EXPECT_FALSE(notLowerHess.isLowerHessenberg(0));
}

TEST_F(MatrixPropsFixture, MatrixIsHermitianWorks) {
    EXPECT_TRUE(defaultMat.isHermitian(0));
    EXPECT_TRUE(singleEltMat.isHermitian(0));
    EXPECT_TRUE(threeDimSymmetric.isHermitian(0));
    EXPECT_TRUE(Matrix<double>::identity(5).isHermitian(0));
    EXPECT_FALSE(threeDimAsymmetric.isHermitian(0));
    EXPECT_FALSE(almostSymmetricExtraCol.isHermitian(0));
    EXPECT_FALSE(almostSymmetricExtraRow.isHermitian(0));
}

TEST_F(MatrixPropsFixture, MatrixDiagHasSameSignWorks) {
    EXPECT_TRUE(defaultMat.diagHasSameSign());
    EXPECT_TRUE(singleEltMat.diagHasSameSign());
    EXPECT_TRUE(Matrix<double>::identity(5).diagHasSameSign());
    EXPECT_FALSE(threeDimSymmetric.diagHasSameSign());
    EXPECT_FALSE(almostSymmetricExtraCol.diagHasSameSign());
    EXPECT_FALSE(almostSymmetricExtraRow.diagHasSameSign());

    // note that a matrix with any zeros in the diag will return false
    Matrix<double> zerosMat(5,5);
    EXPECT_FALSE(zerosMat.diagHasSameSign());
    Matrix<double> almostId = Matrix<double>::identity(5);
    almostId.setData(0.0f, 2, 2);
    EXPECT_FALSE(almostId.diagHasSameSign());
}