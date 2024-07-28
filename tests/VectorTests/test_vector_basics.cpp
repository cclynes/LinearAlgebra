#include "../../include/Vector.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <vector>
#include "BaseVectorFixture.hpp"

using ::testing::ContainerEq;
using namespace LinearAlgebra;

// Test Vector constructors
// These tests won't pass if getData, dim, or isRow are broken.
// They also don't make use of BaseVectorFixture; changing this
// would slightly enhance efficiency and consistency at a slight cost to readability.

TEST(VectorConstructorTests, DefaultConstructorWorks) {
    std::vector<double> expectedData = {};
    size_t dim = 0;
    Vector<double> defaultVec;


    EXPECT_EQ(defaultVec.getData(), expectedData);
    EXPECT_FALSE(defaultVec.isRow());
    EXPECT_EQ(defaultVec.dim(), dim);
}

TEST(VectorConstructorTests, 1DConstructorWorks) {
    // Create data for 1D vector
    std::vector<double> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    // Construct row vector
    Vector rowVec(vec, true);
    EXPECT_EQ(rowVec.getData({0, 4}), vec);
    EXPECT_TRUE(rowVec.isRow());
    EXPECT_EQ(rowVec.dim(), vec.size());

    // Construct column vector
    Vector colVec(vec, false);
    EXPECT_EQ(colVec.getData({0, 4}), vec);
    EXPECT_FALSE(colVec.isRow());
    EXPECT_EQ(colVec.dim(), vec.size());
}

TEST(VectorConstructorTests, 2DConstructorWorksForColVecs) {
    // Create data for 2D vectors
    std::vector<std::vector<double>> colVecData = {{1.0f}, {3.0f}, {5.0f}};

    // Construct vector using 2D data
    Vector<double> colVec(colVecData);

    // Flatten vector
    std::vector<double> expectedData(colVecData.size());
    for (size_t i = 0; i < colVecData.size(); i++) {
        expectedData[i] = colVecData[i][0];
    }

    EXPECT_EQ(colVec.getData(), expectedData);
    EXPECT_FALSE(colVec.isRow());
    EXPECT_EQ(colVec.dim(), expectedData.size());
}

TEST(VectorConstructorTests, 2DConstructorWorksForRowVecs) {
    // Create data for 2D vector
    std::vector<std::vector<double>> rowVecData = {{-4.2f, 1.0f, 4.0f}};

    // Construct vector using 2D data
    Vector rowVec(rowVecData);

    // Flatten vector
    std::vector<double> expectedData = rowVecData[0];

    EXPECT_EQ(rowVec.getData(), expectedData);
    EXPECT_TRUE(rowVec.isRow());
    EXPECT_EQ(rowVec.dim(), expectedData.size());
}

TEST(VectorConstructorTests, DimensionalConstructorWorks) {
    size_t dim = 5;

    // Construct row vector with given dimension
    Vector<double> rowVec(dim, true);
    EXPECT_EQ(rowVec.dim(), dim);
    EXPECT_TRUE(rowVec.isRow());

    // Construct column vector with given dimension
    Vector<double> colVec(dim, false);
    EXPECT_EQ(colVec.dim(), dim);
    EXPECT_FALSE(colVec.isRow());

    // Check if vectors are initialized with zeros
    std::vector<double> expectedData(dim, 0.0);
    EXPECT_EQ(rowVec.getData({0, dim}), expectedData);
    EXPECT_EQ(colVec.getData({0, dim}), expectedData);
}

// inheritance is for readability's sake
class GetDataVectorFixture : public BaseVectorFixture {};

TEST_F(GetDataVectorFixture, GetDataWorksWithNoArgs) {

    std::vector<double> emptyData = {};

    EXPECT_EQ(defaultVec.getData(), emptyData);
    EXPECT_EQ(singleEltRowVec.getData(), singleEltVecData);
    EXPECT_EQ(singleEltColVec.getData(), singleEltVecData);
    EXPECT_EQ(fourEltRowVec.getData(), fourEltVecData);
    EXPECT_EQ(fourEltColVec.getData(), fourEltVecData);
}

TEST_F(GetDataVectorFixture, GetDataWorksWithIndex) {

    EXPECT_EQ(singleEltRowVec.getData(0), singleEltVecData[0]);
    EXPECT_EQ(fourEltRowVec.getData(2), fourEltVecData[2]);
    EXPECT_EQ(fourEltRowVec.getData(3), fourEltVecData[3]);
    EXPECT_EQ(fourEltColVec.getData(3), fourEltVecData[3]);
}

TEST_F(GetDataVectorFixture, GetDataWorksWithRange) {

    std::vector<double> emptyData = {};

    // create vector of data to be extracted using getData
    std::vector<double> fourEltVecMiddleData(2);
    for (size_t i = 0; i < 2; i++) {
        fourEltVecMiddleData[i] = fourEltVecData[i+1];
    }

    EXPECT_EQ(fourEltRowVec.getData({0,0}), emptyData);
    EXPECT_EQ(singleEltColVec.getData({0,1}), singleEltVecData);
    EXPECT_EQ(fourEltRowVec.getData({0,4}), fourEltVecData);
    EXPECT_EQ(fourEltRowVec.getData({1,3}), fourEltVecMiddleData);
    EXPECT_EQ(fourEltColVec.getData({1,3}), fourEltVecMiddleData);
    EXPECT_EQ(fourEltRowVec.getData({2,1}), emptyData);
}

TEST_F(GetDataVectorFixture, GetDataBreaksOutOfRange) {
    EXPECT_THROW(defaultVec.getData(0), std::invalid_argument);
    EXPECT_THROW(singleEltColVec.getData({0,2}), std::invalid_argument);
    EXPECT_THROW(fourEltColVec.getData({5,2}), std::invalid_argument);
}

class SetDataVectorFixture : public BaseVectorFixture { // doesn't inherit from BaseVectorFixture; redeclarations improve readability
protected:

    std::vector<double> singleElt;
    std::vector<double> twoElts;
    std::vector<double> fourElts;
    Vector<double> singleEltVec2;
    Vector<double> twoEltRowVec2;
    Vector<double> twoEltColVec2;
    Vector<double> fourEltRowVec2;
    Vector<double> fourEltColVec2;

    void SetUp() override {

        BaseVectorFixture::SetUp();

        singleElt = {79.0f};
        twoElts = {5.23f, -4.4f};
        fourElts = {94.0f, -21.4f, 7.2f, 902.3f};

        // prepare data for comparison
        singleEltVec2 = Vector<double>(singleElt, false); // for consistency in tests
        twoEltRowVec2 = Vector<double>(twoElts, true);
        twoEltColVec2 = Vector<double>(twoElts, false);
        fourEltRowVec2 = Vector<double>(fourElts, true);
        fourEltColVec2 = Vector<double>(fourElts, false);
    }
    
};

TEST_F(SetDataVectorFixture, SetDataWorksWithIndex) { // there might be some redundant tests here
    // alter data by index
    singleEltVec2.setData(17.0f, 0);
    // change corresponding vector for check
    singleElt[0] = 17.0f;
    // check
    EXPECT_EQ(singleEltVec2.getData(), singleElt);

    // etc.
    twoEltRowVec2.setData(-9.0f, 0);
    twoElts[0] = -9.0f;
    EXPECT_EQ(twoEltRowVec2.getData(), twoElts);
    twoElts[0] = 5.23f;

    twoEltColVec2.setData(-9.0f, 1);
    twoElts[1] = -9.0f;
    EXPECT_EQ(twoEltColVec2.getData(), twoElts);
    twoElts[1] = -4.4f;

    fourEltRowVec2.setData(1000.0f, 2);
    fourElts[2] = 1000.0f;
    EXPECT_EQ(fourEltRowVec2.getData(), fourElts);
    fourElts[2] = 7.2f;

    fourEltColVec2.setData(1000.0f, 3);
    fourElts[3] = 1000.0f;
    EXPECT_EQ(fourEltColVec2.getData(), fourElts);
    fourElts[3] = 902.3f;
}

TEST_F(SetDataVectorFixture, SetDataWorksWithRange) {
    singleEltVec2.setData({17.0f},{0,1});
    singleElt[0] = 17.0f;
    EXPECT_EQ(singleEltVec2.getData(), singleElt);

    fourEltRowVec2.setData({-9.0f, 1000.0f}, {1,3});
    fourElts[1] = -9.0f;
    fourElts[2] = 1000.0f;
    EXPECT_EQ(fourEltRowVec2.getData(), fourElts);
    fourElts[1] = -21.4f; 
    fourElts[2] = 7.2f;

    fourEltColVec2.setData({-55.2f, 21.21f, 3.14f}, {0,3});
    fourElts[0] = -55.2f;
    fourElts[1] = 21.21f;
    fourElts[2] = 3.14f;
    EXPECT_EQ(fourEltColVec2.getData(), fourElts);
    fourElts[0] = 94.0f;
    fourElts[1] = -21.4f;
    fourElts[2] = 7.2f;
}

TEST_F(SetDataVectorFixture, SetDataWorksWithOneArg) {
    std::vector<double> newOneElt = {9.9f};
    singleEltVec2.setData(newOneElt);
    EXPECT_EQ(singleEltVec2.getData(), newOneElt);

    std::vector<double> newFourElts = {33.0f, 22.2f, 10.0f, -9.9f};
    fourEltRowVec2.setData(newFourElts);
    fourEltColVec2.setData(newFourElts);
    EXPECT_EQ(fourEltRowVec2.getData(), newFourElts);
    EXPECT_EQ(fourEltColVec2.getData(), newFourElts);
}

TEST_F(SetDataVectorFixture, SetDataBreaksOutOfRange) {
    EXPECT_THROW(defaultVec.setData(0.0f, 0), std::invalid_argument);
    EXPECT_THROW(fourEltColVec.setData(2.0f, -2), std::invalid_argument);
    EXPECT_THROW(fourEltRowVec.setData(3.5f, 6), std::invalid_argument);
    EXPECT_THROW(fourEltRowVec.setData(fourElts, {1, 5}), std::invalid_argument);
    EXPECT_THROW(fourEltColVec.setData({0.0f, 0.0f, 0.0f, 0.0f, 0.0f}), std::invalid_argument);
    EXPECT_THROW(fourEltRowVec.setData({-1.0f},{1,0}), std::invalid_argument);
}
