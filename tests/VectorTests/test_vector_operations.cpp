#include "../../include/Vector.hpp"
#include "../../include/Matrix.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <vector>
#include "BaseVectorFixture.hpp"
#include <iostream>

using ::testing::ContainerEq;
using namespace LinearAlgebra;

TEST_F(BaseVectorFixture, IsRowWorks) {
    EXPECT_FALSE(defaultVec.isRow());
    EXPECT_TRUE(singleEltRowVec.isRow());
    EXPECT_FALSE(fourEltColVec.isRow());
}

TEST_F(BaseVectorFixture, DimWorks) {
    EXPECT_EQ(defaultVec.dim(), 0);
    EXPECT_EQ(singleEltRowVec.dim(), 1);
    EXPECT_EQ(fourEltColVec.dim(), 4);
}

TEST_F(BaseVectorFixture, TransposeWorks) {
    EXPECT_TRUE(defaultVec.transpose().isRow());
    EXPECT_TRUE(fourEltColVec.transpose().isRow());

    fourEltRowVec.transpose();
    EXPECT_TRUE(fourEltRowVec.isRow()); // transpose is non-mutating
}

TEST_F(BaseVectorFixture, ReverseWorks) {
    EXPECT_TRUE(defaultVec == defaultVec.reverse());
    EXPECT_TRUE(singleEltRowVec == singleEltRowVec.reverse());
    EXPECT_TRUE(fourEltRowVec == fourEltReverseRowVec.reverse());
}

TEST_F(BaseVectorFixture, ScrambleWorks) {
    Vector<double> oldDefault = defaultVec;
    std::vector<size_t> emptyScrambler(0);
    defaultVec.scramble(emptyScrambler);
    EXPECT_TRUE(oldDefault == defaultVec);

    // {2.0f, -7.32f, 1000.5554f, -42.0f}
    Vector<double> oldThreeEltColVec = threeEltColVec;
    std::vector<size_t> constantScrambler = {0, 1, 2};
    threeEltColVec.scramble(constantScrambler);
    EXPECT_TRUE(oldThreeEltColVec == threeEltColVec);

    Vector<double> fourEltsShiftedLeft({-7.32f, 1000.5554f, -42.0f, 2.0f}, false);
    std::vector<size_t> shiftLeftScrambler = {3, 0, 1, 2};
    fourEltColVec.scramble(shiftLeftScrambler);
    EXPECT_TRUE(fourEltColVec == fourEltsShiftedLeft);

    Vector<double> fourEltsScrambled({-7.32f, -42.0f, 2.0f, 1000.5554f}, false);
    std::vector<size_t> ranScrambler = {0, 3, 1, 2};
    fourEltColVec.scramble(ranScrambler);
    EXPECT_TRUE(fourEltColVec == fourEltsScrambled);
}

TEST_F(BaseVectorFixture, ScrambleBreaksAsIntended) {
    std::vector<size_t> scrambler2D = {1, 2};
    EXPECT_THROW(threeEltColVec.scramble(scrambler2D), std::invalid_argument);
    std::vector<size_t> scramblerNonUnique = {2, 1, 2};
    EXPECT_THROW(threeEltRowVec.scramble(scramblerNonUnique), std::invalid_argument);
    std::vector<size_t> scramblerOutOfRange = {2, 3, 1};
    EXPECT_THROW(threeEltColVec.scramble(scramblerOutOfRange), std::invalid_argument);
}

class VectorOpsFixture : public ::BaseVectorFixture {
protected:

    Vector<double> newDefaultVec;
    Vector<double> twoEltRowVec1;
    Vector<double> twoEltRowVec2;
    Vector<double> twoEltColVec1;
    Vector<double> twoEltColVec2;
    Vector<double> fourEltRowVec1;
    Vector<double> fourEltRowVec2;
    Vector<double> fourEltColVec1;
    Vector<double> fourEltColVec2;

    virtual void SetUp() override {
        BaseVectorFixture::SetUp();
        newDefaultVec = defaultVec;
        twoEltRowVec1 = Vector<double>({1.2, -2.0}, true);
        twoEltRowVec2 = Vector<double>({3.0, 4.0}, true);
        twoEltColVec1 = twoEltRowVec1.transpose();
        twoEltColVec2 = twoEltRowVec2.transpose();
        fourEltRowVec1 = Vector<double>({1.0, 2.0, 3.0, 4.0}, true);
        fourEltRowVec2 = Vector<double>({5.0, 6.0, 7.0, 8.0}, true);
        fourEltColVec1 = fourEltRowVec1.transpose();
        fourEltColVec2 = fourEltRowVec2.transpose();
    }
};

TEST_F(VectorOpsFixture, VectorEqualsWorks) {
    EXPECT_TRUE(newDefaultVec == defaultVec);
    EXPECT_FALSE(fourEltRowVec == fourEltColVec);
    EXPECT_FALSE(defaultVec == singleEltColVec);
}

TEST_F(VectorOpsFixture, VectorIsNearWorks) {
    EXPECT_TRUE(newDefaultVec.isNear(defaultVec, 0.0f));
    EXPECT_TRUE(fourEltRowVec1.isNear(fourEltRowVec2, 4.0f));
    EXPECT_FALSE(fourEltColVec1.isNear(fourEltColVec2, 3.9999f));
    EXPECT_FALSE(fourEltColVec1.isNear(fourEltRowVec1, 0.0f));
}

TEST_F(VectorOpsFixture, VectorAdditionWorks) {
    Vector<double> summedDefaults = defaultVec + newDefaultVec;
    EXPECT_TRUE(summedDefaults == defaultVec);

    Vector<double> summedRowVecs = twoEltRowVec1 + twoEltRowVec2;
    Vector<double> expectedSumRow = Vector<double>({4.2, 2.0}, true);
    EXPECT_TRUE(expectedSumRow == summedRowVecs);

    Vector<double> summedColVecs = fourEltColVec1 + fourEltColVec2;
    Vector<double> expectedSumCol = Vector<double>({6.0, 8.0, 10.0, 12.0}, false);
    EXPECT_TRUE(expectedSumCol == summedColVecs);
}

TEST_F(VectorOpsFixture, VectorAdditionBreaksAsIntended) {
    EXPECT_THROW(twoEltRowVec1 + twoEltColVec1, std::invalid_argument);
    EXPECT_THROW(twoEltRowVec1 + fourEltRowVec1, std::invalid_argument);
}

TEST_F(VectorOpsFixture, VectorSubtractionWorks) {
    Vector<double> subtractedDefaults = defaultVec + newDefaultVec;
    EXPECT_TRUE(subtractedDefaults == defaultVec);

    Vector<double> subtractedRowVecs = twoEltRowVec1 - twoEltRowVec2;
    Vector<double> expectedDiffRow = Vector<double>({-1.8, -6.0}, true);
    EXPECT_TRUE(expectedDiffRow == subtractedRowVecs);

    Vector<double> subtractedColVecs = fourEltColVec2 - fourEltColVec1;
    Vector<double> expectedDiffCol = Vector<double>({4.0, 4.0, 4.0, 4.0}, false);
    EXPECT_TRUE(expectedDiffCol == subtractedColVecs);
}

TEST_F(VectorOpsFixture, VectorSubtractionBreaksAsIntended) {
    EXPECT_THROW(twoEltRowVec1 - twoEltColVec1, std::invalid_argument);
    EXPECT_THROW(twoEltRowVec1 - fourEltRowVec1, std::invalid_argument);
}

class VectorMultFixture : public ::VectorOpsFixture {}; // for readability

TEST_F(VectorMultFixture, DotProductWorks) {
    double tolerance = 1e-6;
    double dotRow = twoEltRowVec1.dot(twoEltRowVec2);

    double expectedDotRow = -4.4;
    EXPECT_NEAR(dotRow, expectedDotRow, tolerance);

    double dotRowAndCol = twoEltRowVec1.dot(twoEltColVec2);
    EXPECT_NEAR(dotRowAndCol, expectedDotRow, tolerance);
}

TEST_F(VectorMultFixture, DotProductBreaksAsIntended) {
    // may need additional stress testing to gauge error at high dimensions
    EXPECT_THROW(defaultVec.dot(newDefaultVec), std::invalid_argument);
    EXPECT_THROW(twoEltRowVec1.dot(fourEltRowVec1), std::invalid_argument);
}

TEST_F(VectorMultFixture, VectorMultiplicationWorks) {
    double tolerance = 1e-6;
    // zero-by-zero works
    EXPECT_TRUE((defaultVec * newDefaultVec) == Matrix<double>(defaultVec));
    // one-by-one works
    std::vector<std::vector<double>> expectedOneElt(1, std::vector<double>(1, 100.0f));
    EXPECT_THAT((singleEltRowVec * singleEltColVec).getData(), ContainerEq(expectedOneElt));
    EXPECT_THAT((singleEltRowVec * singleEltRowVec).getData(), ContainerEq(expectedOneElt));
    EXPECT_THAT((singleEltColVec * singleEltColVec).getData(), ContainerEq(expectedOneElt));

    // 1xn with nx1 works
    std::vector<std::vector<double>> expectedDotMatData(1, std::vector<double>(1, -4.4f));
    Matrix<double> expectedDotMat(expectedDotMatData);
    EXPECT_TRUE((twoEltRowVec1 * twoEltColVec2).isNear(expectedDotMat, tolerance));

    // column with row works
    Matrix<double> productMat = Matrix<double>({
        {6.0, 7.2, 8.4, 9.6}, 
        {-10.0, -12.0, -14.0, -16.0}});
    EXPECT_TRUE((twoEltColVec1 * fourEltRowVec2).isNear(productMat, tolerance));

    // column with 1-elt column works
    Matrix<double> colProduct = Matrix<double>({{-10.0}, {-20.0}, {-30.0}, {-40.0}});
    EXPECT_TRUE((fourEltColVec1 * singleEltColVec).isNear(colProduct, tolerance));
}

TEST_F(VectorMultFixture, VectorMultiplicationBreaksAsIntended) {
    EXPECT_THROW(twoEltColVec1 * twoEltColVec2, std::invalid_argument);
    EXPECT_THROW(twoEltRowVec1 * twoEltRowVec2, std::invalid_argument);
    EXPECT_THROW(twoEltRowVec2 * fourEltColVec1, std::invalid_argument);
}

TEST_F(VectorMultFixture, ScalarVectorMultiplicationWorks) {
    // may need additional stress testing to gauge error at high dimensions
    EXPECT_EQ(defaultVec.getData(), (0.0 * defaultVec).getData());
    EXPECT_EQ(defaultVec.getData(), (defaultVec * -325.1).getData());

    Vector<double> newFourEltRowVec = fourEltRowVec1 * 2.5;
    vector<double> expectedData = {2.5, 5.0, 7.5, 10.0};
    EXPECT_EQ(expectedData, newFourEltRowVec.getData());
}

class VectorAndMatFixture : public ::BaseVectorFixture {
protected:
    Matrix<double> emptyMat;

    std::vector<std::vector<double>> twoByThreeData;
    Matrix<double> twoByThree;

    std::vector<std::vector<double>> oneByTwoData;
    Matrix<double> oneByTwo;

    std::vector<std::vector<double>> threeByThreeData;
    Matrix<double> threeByThree;

    virtual void SetUp() override {
        BaseVectorFixture::SetUp();

        twoByThreeData = {{3.0f, 2.5f, -2.0f}, {-1.0f, 5.0f, 2.0f}};
        twoByThree = Matrix<double>(twoByThreeData);

        oneByTwoData = {{1.5f, 2.0f}};
        oneByTwo = Matrix<double>(oneByTwoData);

        threeByThreeData = {{5.5, 7.0, 2.0}, {2.5, 5.5, -10.0}, {-3.0, -4.0, 0.4}};
        threeByThree = Matrix<double>(threeByThreeData);
    }
};

TEST_F(VectorAndMatFixture, LeftMultiplyVecAndMatWorks) {
    // 0-dim multiples work
    Matrix<double> emptyMat = Matrix<double>(defaultVec);
    EXPECT_TRUE((defaultVec * emptyMat) == emptyMat);

    std::vector<std::vector<double>> oneTwoProdData = {{-15.0, -20.0}};
    Matrix oneTwoProd(oneTwoProdData);
    EXPECT_TRUE(singleEltRowVec * oneByTwo == oneTwoProd);
    EXPECT_TRUE(singleEltColVec * oneByTwo == oneTwoProd);

    std::vector<std::vector<double>> threesProdData = {{17.75, 26.25, -11.4}};
    Matrix threesProd(threesProdData);
    EXPECT_TRUE(threeEltRowVec * threeByThree == threesProd);
}

TEST_F(VectorAndMatFixture, LeftMultiplyVecAndMatBreaksAsIntended) {
    EXPECT_THROW(threeEltColVec * threeByThree, std::invalid_argument);
    EXPECT_THROW(singleEltRowVec * emptyMat, std::invalid_argument);
    EXPECT_THROW(defaultVec * oneByTwo, std::invalid_argument);
    EXPECT_THROW(threeEltRowVec * oneByTwo, std::invalid_argument);
}

TEST_F(VectorOpsFixture, VectorNormWorks) {
    double tol = 1e-6;
    Vector threeFourFive = Vector<double>({3.0, 4.0}, false);
    EXPECT_EQ(threeFourFive.norm(), 5.0);

    Vector manyDimVec = Vector<double>({3.5, -7.2, -25.0, -1.111, -0.22}, true);
    double expectedNorm = 26.2749447; // this is hardcoded - maybe go back and test more rigorously
    EXPECT_TRUE(abs(manyDimVec.norm() - expectedNorm) < tol);
}
