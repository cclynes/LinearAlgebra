#include "../../include/Matrix.hpp"
#include "../../include/Vector.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "BaseMatrixFixture.hpp"

using ::testing::ContainerEq;

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
    EXPECT_TRUE(defaultMat.isSymmetric());
    EXPECT_TRUE(singleEltMat.isSymmetric());
    EXPECT_TRUE(Matrix<double>::identity(5).isSymmetric());
    EXPECT_TRUE(threeDimSymmetric.isSymmetric());
    EXPECT_FALSE(threeDimAsymmetric.isSymmetric());
    EXPECT_FALSE(almostSymmetricExtraCol.isSymmetric());
    EXPECT_FALSE(almostSymmetricExtraRow.isSymmetric());
}

TEST_F(MatrixPropsFixture, MatrixIsUpperTriangularWorks) {
    EXPECT_TRUE(defaultMat.isUpperTriangular());
    EXPECT_TRUE(singleEltMat.isUpperTriangular());
    EXPECT_TRUE(zerosMat.isUpperTriangular());

    std::vector<std::vector<double>> threeDimUTData = {
        {1.0, -7.0, 2.5},
        {0.0, 0.5, -12.0},
        {0.0, 0.0, -1.0}};
    Matrix<double> threeDimUT(threeDimUTData);
    EXPECT_TRUE(threeDimUT.isUpperTriangular());

    std::vector<std::vector<double>> threeDimAlmostUTData = threeDimUTData;
    threeDimAlmostUTData[2][1] = 0.1;
    Matrix<double> threeDimAlmostUT(threeDimAlmostUTData);
    EXPECT_FALSE(threeDimAlmostUT.isUpperTriangular());

    std::vector<std::vector<double>> threeDimAlmostUTExtraRowData = {
        {1.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}};
    Matrix<double> threeDimAlmostUTExtraRow(threeDimAlmostUTExtraRowData);
    EXPECT_FALSE(threeDimAlmostUTExtraRow.isUpperTriangular());

    std::vector<std::vector<double>> threeDimAlmostUTExtraColData = {
        {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}};
    Matrix<double> threeDimAlmostUTExtraCol(threeDimAlmostUTExtraColData);
    EXPECT_FALSE(threeDimAlmostUTExtraCol.isUpperTriangular());
}

TEST_F(MatrixPropsFixture, MatrixIsLowerTriangularWorks) {
    EXPECT_TRUE(defaultMat.isLowerTriangular());
    EXPECT_TRUE(singleEltMat.isLowerTriangular());
    EXPECT_TRUE(zerosMat.isLowerTriangular());

    std::vector<std::vector<double>> threeDimLTData = {
        {1.0, 0.0, 0.0},
        {-7.0, 0.0, 0.0},
        {2.0, -12.0, -1.0}};
    Matrix<double> threeDimLT(threeDimLTData);
    EXPECT_TRUE(threeDimLT.isLowerTriangular());

    std::vector<std::vector<double>> threeDimAlmostLTData = threeDimLTData;
    threeDimAlmostLTData[1][2] = 0.1;
    Matrix<double> threeDimAlmostLT(threeDimAlmostLTData);
    EXPECT_FALSE(threeDimAlmostLT.isLowerTriangular());

    std::vector<std::vector<double>> threeDimAlmostLTExtraRowData = {
        {1.0, 0.0}, {1.0, 1.0}, {1.0, 1.0}};
    Matrix<double> threeDimAlmostLTExtraRow(threeDimAlmostLTExtraRowData);
    EXPECT_FALSE(threeDimAlmostLTExtraRow.isUpperTriangular());

    std::vector<std::vector<double>> threeDimAlmostLTExtraColData = {
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    Matrix<double> threeDimAlmostLTExtraCol(threeDimAlmostLTExtraColData);
    EXPECT_FALSE(threeDimAlmostLTExtraCol.isUpperTriangular());
}

TEST_F(MatrixPropsFixture, MatrixIsUpperHessenbergWorks) {
    EXPECT_TRUE(defaultMat.isUpperHessenberg());
    EXPECT_TRUE(singleEltMat.isUpperHessenberg());
    EXPECT_TRUE(zerosMat.isUpperHessenberg());
    EXPECT_TRUE(twoByTwo.isUpperHessenberg());
    EXPECT_FALSE(zerosMatExtraRow.isUpperHessenberg());
    EXPECT_FALSE(zerosMatExtraCol.isUpperHessenberg());

    std::vector<std::vector<double>> upperHessData = {
        {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}};
    Matrix<double> upperHess(upperHessData);
    EXPECT_TRUE(upperHess.isUpperHessenberg());

    upperHessData[2][0] = 1.0f;
    Matrix<double> notUpperHess(upperHessData);
    EXPECT_FALSE(notUpperHess.isUpperHessenberg());
}

TEST_F(MatrixPropsFixture, MatrixIsLowerHessenbergWorks) {
    EXPECT_TRUE(defaultMat.isLowerHessenberg());
    EXPECT_TRUE(singleEltMat.isLowerHessenberg());
    EXPECT_TRUE(zerosMat.isLowerHessenberg());
    EXPECT_TRUE(twoByTwo.isLowerHessenberg());
    EXPECT_FALSE(zerosMatExtraRow.isLowerHessenberg());
    EXPECT_FALSE(zerosMatExtraCol.isLowerHessenberg());

    std::vector<std::vector<double>> lowerHessData = {
        {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}};
    Matrix lowerHess(lowerHessData);
    EXPECT_TRUE(lowerHess.isLowerHessenberg());

    lowerHessData[0][2] = 1.0f;
    Matrix notLowerHess(lowerHessData);
    EXPECT_FALSE(notLowerHess.isLowerHessenberg());
}

TEST_F(MatrixPropsFixture, MatrixIsHermitianWorks) {
    EXPECT_TRUE(defaultMat.isHermitian());
    EXPECT_TRUE(singleEltMat.isHermitian());
    EXPECT_TRUE(threeDimSymmetric.isHermitian());
    EXPECT_TRUE(Matrix<double>::identity(5).isHermitian());
    EXPECT_FALSE(threeDimAsymmetric.isHermitian());
    EXPECT_FALSE(almostSymmetricExtraCol.isHermitian());
    EXPECT_FALSE(almostSymmetricExtraRow.isHermitian());
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