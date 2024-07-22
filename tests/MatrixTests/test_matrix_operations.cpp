#include "../../include/Matrix.hpp"
#include "../../include/Vector.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "BaseMatrixFixture.hpp"

using ::testing::ContainerEq;

// doesn't inherit; redundant declarations are made for readability's sake
class MatrixOpsFixture : public ::testing::Test {
protected:
    std::vector<std::vector<double>> singleEltMatData;
    std::vector<std::vector<double>> singleEltMatDataTwo;
    std::vector<std::vector<double>> oneByTwoData;
    std::vector<std::vector<double>> twoByOneData;
    std::vector<std::vector<double>> twoByOneTwoData;
    std::vector<std::vector<double>> twoByTwoData;
    std::vector<std::vector<double>> threeByTwoData;
    std::vector<std::vector<double>> twoByThreeData;

    Matrix<double> defaultMat; // will not be further defined in this class
    Matrix<double> emptyMat;
    Matrix<double> singleEltMat;
    Matrix<double> singleEltMatTwo;
    Matrix<double> oneByTwo;
    Matrix<double> twoByOne;
    Matrix<double> twoByOneTwo;
    Matrix<double> twoByTwo;
    Matrix<double> threeByTwo;
    Matrix<double> twoByThree;
    Matrix<double> threeDimZeros;
    Matrix<double> fourByThreeZeros;
    Matrix<double> threeByFourZeros;

    virtual void SetUp() override {
        emptyMat = Matrix<double>(0,0);

        singleEltMatData = {{-2.32f}};
        singleEltMat = Matrix<double>(singleEltMatData);

        singleEltMatDataTwo = {{-1.0f}};
        singleEltMatTwo = Matrix<double>(singleEltMatDataTwo);

        oneByTwoData = {{1.0, -2.0}};
        oneByTwo = Matrix<double>(oneByTwoData);

        twoByOneData = {{-2.5}, {1.5}};
        twoByOne = Matrix<double>(twoByOneData);

        twoByOneTwoData = {{2.5}, {1.0}};
        twoByOneTwo = Matrix<double>(twoByOneTwoData);

        twoByTwoData = {{-2.5, 0.0}, {0.0, 2.0}};
        twoByTwo = Matrix<double>(twoByTwoData);

        threeByTwoData = {{5.5, 7.0}, {2.5, 5.5}, {-3.0, -4.0}};
        threeByTwo = Matrix<double>(threeByTwoData);

        twoByThreeData = {{-2.0, 10.0, 11.0}, {-5.0, 4.5, 2.0}};
        twoByThree = Matrix<double>(twoByThreeData);
        
        threeDimZeros = Matrix<double>(3,3);
        fourByThreeZeros = Matrix<double>(4,3);
        threeByFourZeros = Matrix<double>(3,4);
    }
};

TEST_F(MatrixOpsFixture, MatrixEqualsWorks) {
    EXPECT_TRUE(emptyMat == defaultMat);
    Matrix<double> twoByThreeCopy = Matrix<double>(twoByThreeData);
    EXPECT_TRUE(twoByThree == twoByThreeCopy);

    EXPECT_FALSE(singleEltMat == singleEltMatTwo);
    Matrix<double> threeDimZeros(3,3);
    EXPECT_FALSE(threeDimZeros == fourByThreeZeros);
    EXPECT_FALSE(threeDimZeros == threeByFourZeros);
    }

TEST_F(MatrixOpsFixture, MatrixIsNearWorks) {
    EXPECT_TRUE(emptyMat.isNear(defaultMat, 0.0));
    Matrix<double> twoByTwoCopy(twoByTwoData);
    EXPECT_TRUE(twoByTwo.isNear(twoByTwoCopy, 0.0));
    threeByTwoData[2][0] = -3.1;
    Matrix<double> threeByTwoAlmost(threeByTwoData);
    EXPECT_TRUE(threeByTwo.isNear(threeByTwoAlmost, 0.2));
    EXPECT_TRUE(threeByTwo.isNear(threeByTwoAlmost, 0.10000001));
    EXPECT_FALSE(threeByTwo.isNear(threeByTwoAlmost, 0.05));
    EXPECT_FALSE(threeDimZeros.isNear(threeByFourZeros, 100.0));
    EXPECT_FALSE(threeDimZeros.isNear(fourByThreeZeros, 100.0));
}

TEST_F(MatrixOpsFixture, MatrixAdditionWorks) {
    double tol = 1e-7;
    EXPECT_TRUE(emptyMat + defaultMat == emptyMat);

    Matrix<double> summedSingleElts(1,1);
    summedSingleElts.setData({{-3.32}});
    EXPECT_TRUE((singleEltMat + singleEltMatTwo).isNear(summedSingleElts, tol));

    std::vector<std::vector<double>> summedTwoByOneData = {{0.0f}, {2.5f}};
    Matrix<double> summedTwoByOne(summedTwoByOneData);
    EXPECT_TRUE(twoByOne + twoByOneTwo == summedTwoByOne);

    std::vector<std::vector<double>> summedTwoByTwoData = {{-5.0f, 0.0f}, {0.0f, 4.0f}};
    Matrix<double> summedTwoByTwo(summedTwoByTwoData);
    EXPECT_TRUE(twoByTwo + twoByTwo == summedTwoByTwo);
}

TEST_F(MatrixOpsFixture, MatrixAdditionBreaksAsIntended) {
    EXPECT_THROW(emptyMat + singleEltMat, std::invalid_argument);
    EXPECT_THROW(twoByOne + oneByTwo, std::invalid_argument);
    EXPECT_THROW(threeDimZeros + threeByFourZeros, std::invalid_argument);
    EXPECT_THROW(threeDimZeros + fourByThreeZeros, std::invalid_argument);
}

TEST_F(MatrixOpsFixture, MatrixSubtractionWorks) {
    EXPECT_TRUE(emptyMat - defaultMat == emptyMat);

    Matrix<double> subtractedSingleElts(1,1);
    subtractedSingleElts.setData({{-1.32f}});
    EXPECT_TRUE((singleEltMat - singleEltMatTwo).isNear(subtractedSingleElts, 1e-6f));

    std::vector<std::vector<double>> subtractedTwoByOneData = {{-5.0f}, {0.5f}};
    Matrix<double> subtractedTwoByOne(subtractedTwoByOneData);
    EXPECT_TRUE(twoByOne - twoByOneTwo == subtractedTwoByOne);

    Matrix<double> subtractedTwoByTwo(2,2);
    EXPECT_TRUE(twoByTwo - twoByTwo == subtractedTwoByTwo);
}

TEST_F(MatrixOpsFixture, MatrixSubtractionBreaksAsIntended) {
    EXPECT_THROW(emptyMat - singleEltMat, std::invalid_argument);
    EXPECT_THROW(twoByOne - oneByTwo, std::invalid_argument);
    EXPECT_THROW(threeDimZeros - threeByFourZeros, std::invalid_argument);
    EXPECT_THROW(threeDimZeros - fourByThreeZeros, std::invalid_argument);
}

TEST_F(MatrixOpsFixture, MatrixTransposeWorks) {
    EXPECT_TRUE(emptyMat == emptyMat.transpose());
    EXPECT_TRUE(threeDimZeros == threeDimZeros.transpose());
    EXPECT_TRUE(twoByTwo == twoByTwo.transpose());

    std::vector<std::vector<double>> threeByTwoTransData = {
        {5.5, 2.5, -3.0}, {7.0, 5.5, -4.0}};
    Matrix<double> threeByTwoTrans(threeByTwoTransData);
    EXPECT_TRUE(threeByTwoTrans == threeByTwo.transpose());

    EXPECT_FALSE(threeDimZeros == threeByFourZeros.transpose());

    std::vector<std::vector<double>> threeDimData = {
        {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}};
    Matrix<double> threeDim(threeDimData);
    EXPECT_FALSE(threeDim == threeDim.transpose());
}

class MatrixInterchangeFixture : public ::BaseMatrixFixture {
protected:
    Matrix<double> threeByFour;
    Matrix<double> threeByFourLastTwoRowsInterchanged;
    Matrix<double> threeByFourMiddleTwoColumnsInterchanged;

    virtual void SetUp() override {
        threeByFour = Matrix<double>({
            {3.0, -5.0, 4.5, 2.0},
            {2.3, -10.0, -20.0, 0.0},
            {3.0, 1.5, -3.0, -3.5}});
        threeByFourLastTwoRowsInterchanged = Matrix<double>({
            {3.0, -5.0, 4.5, 2.0},
            {3.0, 1.5, -3.0, -3.5},
            {2.3, -10.0, -20.0, 0.0}
        });
        threeByFourMiddleTwoColumnsInterchanged = Matrix<double>({
            {3.0, 4.5, -5.0, 2.0},
            {2.3, -20.0, -10.0, 0.0},
            {3.0, -3.0, 1.5, -3.5}
        });
    }
};

TEST_F(MatrixInterchangeFixture, MatrixInterchangeWorks) {
    threeByFour.interchange({1,2}, 0);
    EXPECT_TRUE(threeByFour == threeByFourLastTwoRowsInterchanged);

    Matrix<double> subColInterchange({
        {3.0, -5.0, 4.5, 2.0},
        {3.0, -3.0, 1.5, -3.5},
        {2.3, -20.0, -10.0, 0.0}
    });
    threeByFour.interchange({1,2}, {1,3}, 1);
    EXPECT_TRUE(threeByFour == subColInterchange);

    Matrix<double> subRowInterchange({        
        {3.0, -20.0, -10.0, 2.0},
        {3.0, -3.0, 1.5, -3.5},
        {2.3, -5.0, 4.5, 0.0}});
    threeByFour.interchange({0,2}, {1,3}, 0);
    EXPECT_TRUE(threeByFour == subRowInterchange);
}

TEST_F(MatrixInterchangeFixture, MatrixInterchangeBreaksAsIntended) {
    EXPECT_THROW(defaultMat.interchange({0,0}, 0), std::invalid_argument);
    EXPECT_THROW(threeByFour.interchange({1,3}, 0), std::invalid_argument);
    EXPECT_THROW(threeByFour.interchange({1,4}, 1), std::invalid_argument);
    EXPECT_THROW(threeByFour.interchange({1,2}, {0,5}, 0), std::invalid_argument);
    EXPECT_THROW(threeByFour.interchange({1,2}, {0,4}, 1), std::invalid_argument);    
}

TEST_F(MatrixInterchangeFixture, MatrixScrambleWorks) {
    Matrix<double> defaultMatCopy = defaultMat;
    std::vector<size_t> emptyVec(0);

    // scramble works with empty Matrix
    defaultMat.scramble(emptyVec, 0);
    EXPECT_TRUE(defaultMat == defaultMatCopy);
    defaultMat.scramble(emptyVec, 1);
    EXPECT_TRUE(defaultMat == defaultMatCopy);
    
    // scramble works with single interchange
    std::vector<size_t> threeByFourRowScrambler = {0, 2, 1};
    threeByFour.scramble(threeByFourRowScrambler, 0);
    EXPECT_TRUE(threeByFour == threeByFourLastTwoRowsInterchanged);

    // scramble works with multiple interchanges
    std::vector<size_t> threeByFourColScrambler = {1, 3, 2, 0};
    Matrix<double> threeByFourColsScrambled({
        {2.0, 3.0, 4.5, -5.0},
        {-3.5, 3.0, -3.0, 1.5},
        {0.0, 2.3, -20.0, -10.0}
    });
    threeByFour.scramble(threeByFourColScrambler, 1);
    EXPECT_TRUE(threeByFour == threeByFourColsScrambled);
}

TEST_F(MatrixInterchangeFixture, MatrixScrambleBreaksAsIntended) {
    std::vector<size_t> threeByFourRowScrambler = {0, 2, 1};
    std::vector<size_t> threeByFourColScrambler = {1, 3, 2, 0};
    EXPECT_THROW(threeByFour.scramble(threeByFourRowScrambler, 2), std::invalid_argument);
    EXPECT_THROW(threeByFour.scramble(threeByFourRowScrambler, 1), std::invalid_argument);
    EXPECT_THROW(threeByFour.scramble(threeByFourColScrambler, 0), std::invalid_argument);

    std::vector<size_t> outOfRangeScrambler = {0, 1, 3};
    std::vector<size_t> nonUniqueScrambler = {2, 0, 2, 1};
    EXPECT_THROW(threeByFour.scramble(outOfRangeScrambler, 0), std::invalid_argument);
    EXPECT_THROW(threeByFour.scramble(nonUniqueScrambler, 1), std::invalid_argument);
}

class MatrixSortWithScramblerFixture : public :: BaseMatrixFixture {
protected:
    Matrix<double> threeByFour;
    Matrix<double> identicalNorms;

    virtual void SetUp() override {
        BaseMatrixFixture::SetUp();

        threeByFour = Matrix<double>({
            {3.0, -5.0, 4.5, 2.0},
            {2.3, -10.0, -20.0, 0.0},
            {3.0, 1.5, -3.0, -3.5}});

        identicalNorms = Matrix<double>({
            {1.0, 2.0, 3.0},
            {2.0, 3.0, 1.0},
            {3.0, 1.0, 2.0}});
    }
};

TEST_F(MatrixSortWithScramblerFixture, MatrixSortWithScramblerWorks) {
    std::vector<size_t> defaultVec;
    auto normFunc = [](const std::vector<double>& toCompare) {return Vector<double>(toCompare, false).norm();};


    // sorting works with empty Matrix
    EXPECT_EQ(defaultMat.sortWithScrambler<double>(normFunc, 0), defaultVec);
    EXPECT_EQ(defaultMat.sortWithScrambler<double>(normFunc, 1), defaultVec);

    // sorting works with normal matrices
    std::vector<size_t> expectedScramblerForOrderedColNorms = {2, 1, 0, 3};
    EXPECT_EQ(threeByFour.sortWithScrambler<double>(normFunc, 1), expectedScramblerForOrderedColNorms);

    std::vector<size_t> expectedScramblerForOrderedRowNorms = {1, 0, 2};
    EXPECT_EQ(threeByFour.sortWithScrambler<double>(normFunc, 0), expectedScramblerForOrderedRowNorms);

    // order is preserved by sorter in case of equality
    std::vector<size_t> expectedScramblerForIdenticalNorms = {0, 1, 2};
    EXPECT_EQ(identicalNorms.sortWithScrambler<double>(normFunc, 0), expectedScramblerForIdenticalNorms);
    EXPECT_EQ(identicalNorms.sortWithScrambler<double>(normFunc, 1), expectedScramblerForIdenticalNorms);
}

TEST_F(MatrixSortWithScramblerFixture, MatrixSortWithScramblerBreaksAsIntended) {
    auto normFunc = [](const std::vector<double>& toCompare) {return Vector<double>(toCompare, false).norm();};

    EXPECT_THROW(threeByFour.sortWithScrambler<double>(normFunc, 2), std::invalid_argument);
}

TEST_F(MatrixOpsFixture, MatrixMultiplicationWorks) {
    double tol = 1e-7;
    Matrix<double> singleEltMatProd(1,1);
    singleEltMatProd.setData(2.32, 0, 0);
    EXPECT_TRUE((singleEltMat * singleEltMatTwo).isNear(singleEltMatProd, tol));

    Matrix<double> oneByTwoTwoByOne(1,1);
    oneByTwoTwoByOne.setData(-5.5, 0, 0);
    EXPECT_TRUE(oneByTwo * twoByOne == oneByTwoTwoByOne);

    Matrix<double> twoByTwoTwoByThree({
        {5.0, -25.0, -27.5},
        {-10.0, 9.0, 4.0}});
    EXPECT_TRUE(twoByTwo * twoByThree == twoByTwoTwoByThree);
}

TEST_F(MatrixOpsFixture, MatrixMultiplicationBreaksAsIntended) {
    EXPECT_THROW(emptyMat * defaultMat, std::invalid_argument);
    EXPECT_THROW(emptyMat * singleEltMat, std::invalid_argument);
    EXPECT_THROW(singleEltMat * emptyMat, std::invalid_argument);

    EXPECT_THROW(oneByTwo * oneByTwo, std::invalid_argument);
    EXPECT_THROW(threeDimZeros * fourByThreeZeros, std::invalid_argument);
}

TEST_F(MatrixOpsFixture, MultiplyMatAndVecWorks) {
    Vector<double> defaultVec;
    Vector<double> singleEltColVec(1, false);
    singleEltColVec.setData(2.0f, 0);

    EXPECT_TRUE(defaultMat * defaultVec == defaultMat);
    EXPECT_TRUE(defaultMat * singleEltColVec == defaultMat);

    // row vecs can be multiplied when dimensions allow
    Vector<double> singleEltRowVec(1, true);
    singleEltRowVec.setData(2.0f, 0);
    Matrix<double> twoByOneProd(2,1);
    twoByOneProd.setData({{-5.0f}, {3.0f}});
    EXPECT_TRUE((twoByOne * singleEltRowVec).isNear(twoByOneProd, 1e-6f));

    Vector<double> twoEltRowVec({2.0f, -3.0f}, true);
    Matrix<double> oneByTwoProd(1,2);
    oneByTwoProd.setData({{-2.0f, 3.0f}});
    EXPECT_TRUE(singleEltMatTwo * twoEltRowVec == oneByTwoProd);

    // non—edge cases are as expected
    Vector<double> twoEltColVec({2.0f, -3.0f}, false);
    Matrix<double> threeByOneProd(3,1);
    threeByOneProd.setData({{-10.0f}, {-11.5f}, {6.0f}});
    EXPECT_TRUE(threeByTwo * twoEltColVec == threeByOneProd);
    

    Vector<double> threeEltColVec({-1.0f, 2.0f, 1.0f}, false);
    twoByOneProd.setData({{33.0f}, {16.0f}});
    EXPECT_TRUE(twoByThree * threeEltColVec == twoByOneProd);
}

TEST_F(MatrixOpsFixture, MultiplyMatAndVecBreaksAsIntended) {
    Vector<double> defaultVec;
    EXPECT_THROW(singleEltMat * defaultVec, std::invalid_argument);

    Vector<double> twoEltRowVec({1.0f, 1.0f}, true);
    EXPECT_THROW(twoByTwo * twoEltRowVec, std::invalid_argument);

    Vector<double> twoEltColVec({1.0f, 1.0f}, false);
    EXPECT_THROW(twoByThree * twoEltColVec, std::invalid_argument);
}

TEST_F(MatrixOpsFixture, ScalarMatrixMultiplicationWorks) {
    EXPECT_TRUE(-5.0f * defaultMat == defaultMat);
    EXPECT_TRUE(defaultMat * -5.0f == defaultMat);
    EXPECT_TRUE(3.0f * threeDimZeros == threeDimZeros);
    EXPECT_TRUE(threeDimZeros * 3.0f == threeDimZeros);
    Matrix<double> twoByThreeTimesTwo({{-4.0f, 20.0f, 22.0f}, {-10.0f, 9.0f, 4.0f}});
    EXPECT_TRUE(2.0f * twoByThree == twoByThreeTimesTwo);
    EXPECT_TRUE(twoByThree * 2.0f == twoByThreeTimesTwo);
}
