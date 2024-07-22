#include "../../include/Matrix.hpp"
#include "../../include/Vector.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "BaseMatrixFixture.hpp"

using ::testing::ContainerEq;

class MatrixSubstitutionFixture : public ::BaseMatrixFixture {
protected:
    Vector<double> defaultVec; // will not be redeclared
    Vector<double> singleEltVec;
    Vector<double> twoEltNullVec;
    Vector<double> threeEltNullVec;
    Vector<double> threeEltVec;
    Vector<double> fiveEltVec;


    std::vector<std::vector<double>> threeDimUTData;
    std::vector<std::vector<double>> threeDimUTSingularData;
    std::vector<std::vector<double>> fiveDimUTData;

    Matrix<double> twoByTwoFull;
    Matrix<double> threeDimUT;
    Matrix<double> threeDimUTSingular;
    Matrix<double> fiveDimUT;

    Matrix<double> threeDimLT;
    Matrix<double> threeDimLTSingular;
    Matrix<double> fiveDimLT;

    Vector<double> singleEltSol;
    Vector<double> threeEltUTSol;
    Vector<double> fiveEltUTSol;
    Vector<double> threeEltLTSol;
    Vector<double> fiveEltLTSol;

    virtual void SetUp() override {
        BaseMatrixFixture::SetUp();

        singleEltVec = Vector<double>(1, false);
        singleEltVec.setData({2.0});
        twoEltNullVec = Vector<double>(2, false);
        threeEltNullVec = Vector<double>(3, false);
        threeEltVec = Vector<double>({2.0, -1.5, 10.0}, false);
        fiveEltVec = Vector<double>({-7.5, 0.0, 2.5, 4.5, 5.0}, false);

        Matrix<double> twoByTwoFull({{1.0, 2.0}, {-2.0, 1.0}});
        threeDimUTData = {
            {2.0, 4.5, -1.0},
            {0.0, 6.0, 3.0},
            {0.0, 0.0, -7.0}};
        threeDimUTSingularData = threeDimUTData;
        threeDimUTSingularData[1][1] = 0.0f;
        fiveDimUTData = {
            {-15.0, 12.5, 3.2, -18.0, -9.0},
            {0.0, 6.5, -12.5, -8.0, -4.0},
            {0.0, 0.0, 3.5, 20.0, 10.5},
            {0.0, 0.0, 0.0, 14.0, 14.5},
            {0.0, 0.0, 0.0, 0.0, 2.0}};

        threeDimUT = Matrix<double>(threeDimUTData);
        threeDimUTSingular = Matrix<double>(threeDimUTSingularData);
        fiveDimUT = Matrix<double>(fiveDimUTData);

        threeDimLT = threeDimUT.transpose();
        threeDimLTSingular = threeDimUTSingular.transpose();
        fiveDimLT = fiveDimUT.transpose();

        singleEltSol = Vector<double>(1, false);
        singleEltSol.setData({-0.8264462});
        threeEltUTSol = Vector<double>({-0.758929, 0.464286, -1.428571}, false);
        fiveEltUTSol = Vector<double>({11.88790f, 10.61931f, 6.173469f, -2.267857f, 2.5f}, false);
        threeEltLTSol = Vector<double>({1.0f, -1.0f, -2.0f}, false);
        fiveEltLTSol = Vector<double>({0.5f, -0.961538f, -3.17692f, 4.95330f, -16.4056f});
    }
};

TEST_F(MatrixSubstitutionFixture, SolveBackSubWorks) {
    double tol = 1e-6;
    EXPECT_TRUE(defaultMat.solveBackSub(defaultVec, tol) == defaultVec);
    EXPECT_TRUE(singleEltMat.solveBackSub(singleEltVec, tol).isNear(singleEltSol, tol));
    EXPECT_TRUE(singleEltMat.solveBackSub(singleEltVec.transpose(), tol).isNear(singleEltSol, tol));
    EXPECT_TRUE(threeDimUT.solveBackSub(threeEltVec, tol).isNear(threeEltUTSol, tol));
    EXPECT_TRUE(threeDimUT.solveBackSub(threeEltNullVec, tol) == threeEltNullVec);

    // tolerance must increase with size - suggests need for performance optimization
    double highTol = 1e-4;
    EXPECT_TRUE(fiveDimUT.solveBackSub(fiveEltVec, tol).isNear(fiveEltUTSol, highTol));
}

TEST_F(MatrixSubstitutionFixture, SolveBackSubBreaksAsIntended) {
    double tol = 1e-6;
    // cannot solve systems with improper dimensions
    EXPECT_THROW(defaultMat.solveBackSub(singleEltVec, tol), std::invalid_argument);
    EXPECT_THROW(singleEltMat.solveBackSub(defaultVec, tol), std::invalid_argument);
    EXPECT_THROW(fiveDimUT.solveBackSub(threeEltVec, tol), std::invalid_argument);
    EXPECT_THROW(threeDimUT.solveBackSub(fiveEltVec, tol), std::invalid_argument);

    // cannot solve systems with row vectors (unless they're 1x1)
    EXPECT_THROW(threeDimUT.solveBackSub(threeEltVec.transpose(), tol), std::invalid_argument);

    // cannot back-substitute in systems with non-upper-triangular matrices
    EXPECT_THROW(twoByTwoFull.solveBackSub(twoEltNullVec, tol), std::invalid_argument);

    // cannot back-substitute in singular systems
    EXPECT_THROW(threeDimUTSingular.solveBackSub(threeEltVec, tol), std::invalid_argument);
}

TEST_F(MatrixSubstitutionFixture, SolveForwardSubWorks) {
    double tol = 1e-6;
    EXPECT_TRUE(defaultMat.solveForwardSub(defaultVec, tol) == defaultVec);
    EXPECT_TRUE(singleEltMat.solveForwardSub(singleEltVec, tol).isNear(singleEltSol, tol));
    EXPECT_TRUE(singleEltMat.solveForwardSub(singleEltVec.transpose(), tol).isNear(singleEltSol, tol));
    EXPECT_TRUE(threeDimLT.solveForwardSub(threeEltVec, tol).isNear(threeEltLTSol, tol));
    EXPECT_TRUE(threeDimLT.solveForwardSub(threeEltNullVec, tol) == threeEltNullVec);
    // tolerance must increase with size - suggests need for performance optimization
    double highTol = 1e-4;
    EXPECT_TRUE(fiveDimLT.solveForwardSub(fiveEltVec, tol).isNear(fiveEltLTSol, highTol));
}

TEST_F(MatrixSubstitutionFixture, SolveForwardSubBreaksAsIntended) {
    double tol = 1e-6;
    // cannot solve systems with improper dimensions
    EXPECT_THROW(defaultMat.solveForwardSub(singleEltVec, tol), std::invalid_argument);
    EXPECT_THROW(singleEltMat.solveForwardSub(defaultVec, tol), std::invalid_argument);
    EXPECT_THROW(fiveDimLT.solveForwardSub(threeEltVec, tol), std::invalid_argument);
    EXPECT_THROW(threeDimLT.solveForwardSub(fiveEltVec, tol), std::invalid_argument);

    // cannot solve systems with row vectors (unless they're 1x1)
    EXPECT_THROW(threeDimLT.solveForwardSub(threeEltVec.transpose(), tol), std::invalid_argument);

    // cannot back-substitute in systems with non-upper-triangular matrices
    EXPECT_THROW(twoByTwoFull.solveForwardSub(twoEltNullVec, tol), std::invalid_argument);

    // cannot back-substitute in singular systems
    EXPECT_THROW(threeDimLTSingular.solveForwardSub(threeEltVec, tol), std::invalid_argument);
}

class MatrixQRFixture : public ::BaseMatrixFixture {
protected:
    Vector<double> defaultVec;

    Matrix<double> threeByTwoMat;
    Matrix<double> fourByThreeColRankDeficientMat;

    Vector<double> twoDimVecOfOnes;
    Vector<double> twoDimMultipleForProjectedSol;
    Vector<double> threeDimVecOfOnes;

    Vector<double> threeDimVecSol;
    Vector<double> threeDimVecUnreachable;
    Vector<double> fourDimRankDeficientVecSol;

    virtual void SetUp() override {
        BaseMatrixFixture::SetUp();

        threeByTwoMat = Matrix<double>({
            {1.0, 2.0},
            {-1.0, 2.0},
            {3.0, 6.0}});

        fourByThreeColRankDeficientMat = Matrix<double>({
            {1.0, 1.0, 2.0},
            {2.0, -2.0, 4.0},
            {3.0, 3.0, 6.0},
            {4.0, 4.0, 8.0}});

        twoDimVecOfOnes = Vector<double>({1.0, 1.0}, false);
        twoDimMultipleForProjectedSol = Vector<double>({0.6, -0.2}, false);
        threeDimVecOfOnes = Vector<double>({1.0, 1.0, 1.0}, false);

        threeDimVecSol = Vector<double>(threeByTwoMat * twoDimVecOfOnes);
        threeDimVecUnreachable = Vector<double>({2.0, -1.0, 0.0}, false);
        fourDimRankDeficientVecSol = Vector<double>(fourByThreeColRankDeficientMat * threeDimVecOfOnes);
    }
};

TEST_F(MatrixQRFixture, QRWorksWithSquareMatrices) {
    double tol = 1e-14;
    EXPECT_TRUE(defaultMat.solveQR(defaultVec, 0.0) == defaultVec);
    EXPECT_TRUE(threeByTwoMat.solveQR(threeDimVecSol, tol).isNear(twoDimVecOfOnes, tol));
    EXPECT_TRUE(threeByTwoMat.solveQR(threeDimVecUnreachable, tol).isNear(twoDimMultipleForProjectedSol, tol));
    // EXPECT_TRUE(fourByThreeColRankDeficientMat.solveQR(fourDimRankDeficientVecSol, tol).isNear(threeDimVecOfOnes, tol));
}

class MatrixLUFixture : public ::BaseMatrixFixture {
protected:
    Matrix<double> invertibleTwoByTwo;
    Matrix<double> invertibleThreeByThree;
    Matrix<double> invertibleFourByFour;
    Matrix<double> singularFourByFour;

    Vector<double> defaultVec; // will not be further defined
    Vector<double> oneDimVecMultiple;
    Vector<double> twoDimVecMultiple;
    Vector<double> threeDimVecMultiple;
    Vector<double> fourDimVecMultiple;

    Vector<double> oneDimVecSol;
    Vector<double> twoDimVecSol;
    Vector<double> threeDimVecSol;
    Vector<double> fourDimVecSol;
    Vector<double> fourDimVecSingularSol;
    Vector<double> fourDimUnreachableVec;

    virtual void SetUp() override {
        BaseMatrixFixture::SetUp();

        invertibleTwoByTwo = Matrix<double>({
            {2.0f, 1.0f}, 
            {5.0f, 3.0f}});
        invertibleThreeByThree = Matrix<double>({
            {0.0f, 0.125f, 0.25f}, 
            {0.5f, -0.125f, -1.0f}, 
            {0.25f, -0.125f, -0.5f}});
        invertibleFourByFour = Matrix<double>({
            {1.0f, 1.0f, 1.0f, 0.0f},
            {0.0f, 3.0f, 1.0f, 2.0f},
            {1.0f, 0.0f, 2.0f, 1.0f},
            {2.0f, 3.0f, 1.0f, 0.0f}});
        singularFourByFour = Matrix<double>({
            {1.0f, 2.0f, 3.0f, -1.0f},
            {-2.0f, -2.0f, -4.0f, 0.0f},
            {5.5f, -1.0f, 4.5f, 1.0f},
            {-7.0f, 2.0f, -5.0f, 3.0f}}); // note col_2 is in span(col_0, col_1)
        
        std::vector<double> oneDimMultipleData = {2.0f};
        oneDimVecMultiple = Vector<double>(oneDimMultipleData, false);
        twoDimVecMultiple = Vector<double>({4.5f, -1.0f}, false);
        threeDimVecMultiple = Vector<double>({2.0f, -10.0f, 14.0f}, false);
        fourDimVecMultiple = Vector<double>({2.5f, -5.0f, 5.0f, 1.0f}, false);

        oneDimVecSol = Vector<double>(singleEltMat * oneDimVecMultiple);
        twoDimVecSol = Vector<double>(invertibleTwoByTwo * twoDimVecMultiple);
        threeDimVecSol = Vector<double>(invertibleThreeByThree * threeDimVecMultiple);
        fourDimVecSol = Vector<double>(invertibleFourByFour * fourDimVecMultiple);
        fourDimVecSingularSol = Vector<double>(singularFourByFour * fourDimVecMultiple);
        fourDimUnreachableVec = Vector<double>({57.0f, 56.0f, 24.0f, 11.0f}, false); // in null(singularFourByFour.transpose()), i.e. not in singularFourByFour's column space
    }
};

TEST_F(MatrixLUFixture, LUWorks) {
    double tol = 1e-5;

    EXPECT_TRUE(defaultMat.solveLU(defaultVec, 0.0f) == defaultVec);
    EXPECT_TRUE(singleEltMat.solveLU(oneDimVecSol, 0.0f) == oneDimVecMultiple);
    EXPECT_TRUE(invertibleTwoByTwo.solveLU(twoDimVecSol, tol).isNear(twoDimVecMultiple, tol));
    EXPECT_TRUE(invertibleThreeByThree.solveLU(threeDimVecSol, tol).isNear(threeDimVecMultiple, tol));
    EXPECT_TRUE(invertibleFourByFour.solveLU(fourDimVecSol, tol).isNear(fourDimVecMultiple, tol));
}

TEST_F(MatrixLUFixture, LUBreaksAsIntended) {
    double tol = 1e-5;
    // LU solver checks for dimensionality problems
    EXPECT_THROW(singleEltMat.solveLU(twoDimVecSol, tol), std::invalid_argument);
    EXPECT_THROW(oneByTwo.solveLU(twoDimVecSol, tol), std::invalid_argument);

    // LU throws an error when system is inconsistent
    EXPECT_THROW(singularFourByFour.solveLU(fourDimUnreachableVec, tol), std::runtime_error);

    // but it also throws an error when system is over-determined (fix this once system-solving is abstracted from LU-decomposition)
    EXPECT_THROW(singularFourByFour.solveLU(fourDimVecSingularSol, tol), std::runtime_error);
}

class MatrixCholeskyFixture : public ::BaseMatrixFixture { // this class replicates some attributes from the LU fixture - you should restructure classes so you only declare once
protected:
    Matrix<double> singleEltCholeskyMat;
    Matrix<double> spdTwoByTwo;
    Matrix<double> nonspdTwoByTwo;
    Matrix<double> spdThreeByThree;
    Matrix<double> nonspdFourByFour;
    Matrix<double> spdFourByFour;

    Vector<double> defaultVec; // will not be further defined
    Vector<double> oneDimVecMultiple;
    Vector<double> twoDimVecMultiple;
    Vector<double> threeDimVecMultiple;
    Vector<double> fourDimVecMultiple;

    Vector<double> oneDimVecSol;
    Vector<double> twoDimVecSol;
    Vector<double> twoDimNonSpdVecSol;
    Vector<double> threeDimVecSol;
    Vector<double> fourDimVecSol;
    Vector<double> fourDimNonCholeskyVecSol;

    virtual void SetUp() override {
        BaseMatrixFixture::SetUp();

        singleEltCholeskyMat = -1 * singleEltMat;
        spdTwoByTwo = Matrix<double>({
            {4.0f, 3.0f},
            {3.0f, 5.0f}});
        nonspdTwoByTwo = Matrix<double>({
            {1.0f, 2.0f},
            {2.0f, 1.0f}});
        spdThreeByThree = Matrix<double>({
            {2.0f, -1.0f, 0.0f},
            {-1.0f, 2.0f, -1.0f},
            {0.0f, -1.0f, 2.0f}});
        nonspdFourByFour = Matrix<double>({
            {2.0f, 4.0f, 3.5f, 2.0f},
            {-2.0f, 7.0f, -15.0f, -13.0f},
            {0.0f, -11.0f, 12.0f, 4.0f},
            {7.5f, 10.0f, -9.0f, 9.0f}});
        spdFourByFour = nonspdFourByFour * nonspdFourByFour.transpose();

        std::vector<double> oneDimMultipleData = {2.0f};
        oneDimVecMultiple = Vector<double>(oneDimMultipleData, false);
        twoDimVecMultiple = Vector<double>({4.5f, -1.0f}, false);
        threeDimVecMultiple = Vector<double>({2.0f, -10.0f, 14.0f}, false);
        fourDimVecMultiple = Vector<double>({2.5f, -5.0f, 5.0f, 1.0f}, false);

        oneDimVecSol = Vector<double>(singleEltCholeskyMat * oneDimVecMultiple);
        twoDimVecSol = Vector<double>(spdTwoByTwo * twoDimVecMultiple);
        twoDimNonSpdVecSol = Vector<double>(nonspdTwoByTwo * twoDimVecMultiple);
        threeDimVecSol = Vector<double>(spdThreeByThree * threeDimVecMultiple);
        fourDimVecSol = Vector<double>(spdFourByFour * fourDimVecMultiple);
        fourDimNonCholeskyVecSol = Vector<double>(nonspdFourByFour * fourDimVecMultiple);
    }
};

TEST_F(MatrixCholeskyFixture, CholeskyWorks) {
    double tol = 1e-5;
    
    EXPECT_TRUE(defaultMat.solveCholesky(defaultVec, 0.0f) == defaultVec);
    EXPECT_TRUE(singleEltCholeskyMat.solveCholesky(oneDimVecSol, 0.0f).isNear(oneDimVecMultiple, tol));
    EXPECT_TRUE(spdTwoByTwo.solveCholesky(twoDimVecSol, tol).isNear(twoDimVecMultiple, tol));
    EXPECT_TRUE(spdThreeByThree.solveCholesky(threeDimVecSol, tol).isNear(threeDimVecMultiple, tol));
    EXPECT_TRUE(spdFourByFour.solveCholesky(fourDimVecSol, tol).isNear(fourDimVecMultiple, tol));
}

TEST_F(MatrixCholeskyFixture, CholeskyBreaksAsIntended) {
    double tol = 1e-5;
    // Cholesky solver checks for dimensionality problems
    EXPECT_THROW(singleEltMat.solveCholesky(twoDimVecSol, tol), std::invalid_argument);
    EXPECT_THROW(oneByTwo.solveCholesky(twoDimVecSol, tol), std::invalid_argument);

    // Cholesky throws an error when matrix is not symmetric positive-definite
    EXPECT_THROW(nonspdFourByFour.solveCholesky(fourDimVecSol, tol), std::invalid_argument);
    EXPECT_THROW(nonspdTwoByTwo.solveCholesky(twoDimNonSpdVecSol, tol), std::invalid_argument);
}
