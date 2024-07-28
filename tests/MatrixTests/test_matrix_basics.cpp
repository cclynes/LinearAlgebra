#include "../../include/Matrix.hpp"
#include "../../include/Vector.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <random>
#include "BaseMatrixFixture.hpp"

using ::testing::ContainerEq;
using namespace LinearAlgebra;

class MatrixConstructorFixture : public :: BaseMatrixFixture {
protected:
    std::vector<std::vector<double>> emptyVecData;
    std::vector<std::vector<double>> empty2DVecData;
    std::vector<std::vector<double>> singleEltVecData;
    std::vector<std::vector<double>> squareVecData;
    std::vector<std::vector<double>> largeVecData;

    Vector<double> emptyVec;
    Vector<double> singleEltVec;
    Vector<double> threeEltRowVec;

    virtual void SetUp() override {
        BaseMatrixFixture::SetUp();

        emptyVecData = {};
        empty2DVecData = {{}};
        singleEltVecData = {{3.71}};
        squareVecData = {{4.0, 2.5, 9401.42}, {525.0, -34.9, 0.0}, {0.2, -90.9, 42.4}};
        largeVecData = std::vector<std::vector<double>>(85, vector<double>(31, 2.7f));

        emptyVec = Vector<double>(emptyVecData);
        singleEltVec = Vector<double>(singleEltVec);
        threeEltRowVec = Vector<double>({1.0, 2.0, 3.0}, true);
    }
};

// this test is premised on getData and vector constructors working properly
TEST_F(MatrixConstructorFixture, DefaultConstructorWorks) {
    // call constructors with matrix dimensions passed as arguments
    Matrix<double> emptyMat(0,0);
    Matrix<double> singleEltMat(1,1);
    Matrix<double> squareMat(4,4);
    Matrix<double> largeMat(27,44);

    // create zero vectors with corresponding dimensions
    std::vector<std::vector<double>> singleZeroVec(1, std::vector<double>(1.0 ,0.0));
    std::vector<std::vector<double>> squareZeroVec(4, std::vector<double>(4.0, 0.0));
    std::vector<std::vector<double>> largeZeroVec(27, std::vector<double>(44.0, 0.0));

    // check for equality
    EXPECT_EQ(emptyMat.getData(), empty2DVecData);
    EXPECT_EQ(singleEltMat.getData(), singleZeroVec);
    EXPECT_EQ(squareMat.getData(), squareZeroVec);
    EXPECT_EQ(largeMat.getData(), largeZeroVec);
}

// this test is premised on getData and vector constructors working properly
TEST_F(MatrixConstructorFixture, DataConstructorWorks) {

    // call constructors with data passed as arguments
    Matrix<double> emptyMat(emptyVecData);
    Matrix<double> singleEltMat(singleEltVecData);
    Matrix<double> squareMat(squareVecData);
    Matrix<double> largeMat(largeVecData);

    // check for equality
    EXPECT_EQ(emptyMat.getData(), empty2DVecData);
    EXPECT_EQ(emptyMat.getDims().first, emptyVecData.size());
    EXPECT_EQ(emptyMat.getDims().second, emptyVecData.size());

    EXPECT_EQ(singleEltMat.getData(), singleEltVecData);
    EXPECT_EQ(singleEltMat.getDims().first, singleEltVecData.size());
    EXPECT_EQ(singleEltMat.getDims().second, singleEltVecData.size());

    EXPECT_EQ(squareMat.getData(), squareVecData);
    EXPECT_EQ(squareMat.getDims().first, squareVecData.size());
    EXPECT_EQ(squareMat.getDims().second, squareVecData[0].size());

    EXPECT_EQ(largeMat.getData(), largeVecData);
    EXPECT_EQ(largeMat.getDims().first, largeVecData.size());
    EXPECT_EQ(largeMat.getDims().second, largeVecData[0].size());
}

TEST_F(MatrixConstructorFixture, DataConstructorBreaksAsIntended) {
    std::vector<std::vector<double>> jaggedVec = {{1.0f}, {1.2f}, {2.2f, 1.0f}, {1.0f}};
    EXPECT_THROW(Matrix<double> brokenMat(jaggedVec), std::invalid_argument);
}

TEST_F(MatrixConstructorFixture, VectorConstructorWorks) {
    Matrix<double> emptyMat(emptyVec);
    Matrix<double> singleEltMat(singleEltVecData);
    Matrix<double> threeEltRowMat(threeEltRowVec);

// check for equality 

    EXPECT_EQ(emptyMat.getData(), empty2DVecData);
    EXPECT_EQ(emptyMat.getDims().first, emptyVecData.size());
    EXPECT_EQ(emptyMat.getDims().second, emptyVecData.size());

    EXPECT_EQ(singleEltMat.getData(), singleEltVecData);
    EXPECT_EQ(singleEltMat.getDims().first, singleEltVecData.size());
    EXPECT_EQ(singleEltMat.getDims().second, singleEltVecData.size()); 
}

TEST_F(MatrixConstructorFixture, MatrixIdentityConstructorWorks) {
    EXPECT_TRUE(Matrix<double>::identity(0) == defaultMat);

    singleEltMat.setData({{1.0f}});
    EXPECT_TRUE(Matrix<double>::identity(1) == singleEltMat);

    std::vector<std::vector<double>> threeByThreeIdData = {
        {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}
    };
    Matrix<double> threeByThreeId(threeByThreeIdData);
    EXPECT_TRUE(Matrix<double>::identity(3) == threeByThreeId);
}

// test getData methods - all tests premised on matrix and vector constructors working properly

class GetMatDataFixture : public BaseMatrixFixture {
public:
    std::vector<std::vector<double>> make2D(const std::vector<double>& vec) {
        return {vec};
    }
};

TEST_F(GetMatDataFixture, getDataWithTwoRangesWorks) {
    EXPECT_THAT(emptyMat.getData({0,0},{0,0}), ContainerEq(make2D({})));
    EXPECT_EQ(singleEltMat.getData({0,1},{0,1}), singleEltMatData);

    std::vector<std::vector<double>> threeByTwoSeg = {{7.0f},{5.5f}};
    EXPECT_EQ(threeByTwo.getData({0,2},{1,2}), threeByTwoSeg);
    EXPECT_EQ(threeByTwo.getData({0,3},{0,2}), threeByTwoData);
}

TEST_F(GetMatDataFixture, getDataWithTwoRangesBreaksAsIntendeded) {
    // entering indices in descending order throws exception
    EXPECT_THROW(threeByTwo.getData({2,1},{1,2}), std::invalid_argument);
    EXPECT_THROW(threeByTwo.getData({1,2},{2,1}), std::invalid_argument);
    
    // entering indices exceeding dimension throws exception
    EXPECT_THROW(threeByTwo.getData({2,4},{1,2}), std::invalid_argument);
    EXPECT_THROW(threeByTwo.getData({1,2},{1,3}), std::invalid_argument);
}

TEST_F(GetMatDataFixture, getDataDefaultWorks) {
    EXPECT_EQ(emptyMat.getData(), make2D({}));
    EXPECT_EQ(singleEltMat.getData(), singleEltMatData);
    EXPECT_EQ(oneByTwo.getData(), oneByTwoData);
    EXPECT_EQ(threeByTwo.getData(), threeByTwoData);
}

// I don't test error handling for the col- and row-range getData overloads
// because they execute a simple call to the double-ranged getData

TEST_F(GetMatDataFixture, getDataWithColRangeWorks) {
    EXPECT_EQ(singleEltMat.getData(0, {0,1}), singleEltMatData);

    std::vector<std::vector<double>> threeByTwoSeg = {{5.5f}}; 
    EXPECT_EQ(threeByTwo.getData(1, {1,2}), threeByTwoSeg);
}

TEST_F(GetMatDataFixture, getDataWithRowRangeWorks) {
    EXPECT_EQ(singleEltMat.getData({0,1}, 0), singleEltMatData);

    std::vector<std::vector<double>> threeByTwoSeg = {{5.5f}};
    EXPECT_EQ(threeByTwo.getData({1,2}, 1), threeByTwoSeg);
}

TEST_F(GetMatDataFixture, getDataWithTwoIndicesWorks) {
    EXPECT_EQ(singleEltMat.getData(0,0), singleEltMatData[0][0]);
    EXPECT_EQ(threeByTwo.getData(2, 1), -4.0f);
    EXPECT_EQ(twoByTwo.getData(1, 0), 0.0f);
}

TEST_F(GetMatDataFixture, getDataWithTwoIndicesBreaksAsIntendeded) {
    EXPECT_THROW(emptyMat.getData(0,0), std::invalid_argument);
    EXPECT_THROW(threeByTwo.getData(3,1), std::invalid_argument);
    EXPECT_THROW(threeByTwo.getData(2,2), std::invalid_argument);
}

class SetMatDataFixture : public ::BaseMatrixFixture {};

// test setData methods
TEST_F(SetMatDataFixture, setDataWithTwoRangesWorks) {
    
    // setData works with empty matrices
    emptyMat.setData({{}}, {0,0}, {0,0});
    EXPECT_EQ(defaultMat.getData(), emptyMat.getData());
    oneByTwo.setData({{}}, {0,0}, {1,1});
    EXPECT_EQ(oneByTwo.getData(), oneByTwoData);

    // note that an "empty" vector of any size can be passed without error,
    // as long as index bounds correspond to emptiness
    oneByTwo.setData({{}, {}}, {0,0}, {1,1});
    EXPECT_EQ(oneByTwo.getData(), oneByTwoData);
    oneByTwo.setData({}, {0,0}, {1,1});
    EXPECT_EQ(oneByTwo.getData(), oneByTwoData);
    
    // full range works
    Matrix<double> threeByTwoNew(3,2);
    threeByTwoNew.setData(threeByTwoData, {0,3}, {0,2});
    EXPECT_EQ(threeByTwoNew.getData(), threeByTwoData);

    // partial range works
    std::vector<std::vector<double>> dataToAdd = {{1.2, 1.3}, {1.4, 1.5}};
    std::pair<size_t, size_t> rowRange = {1,3};
    std::pair<size_t, size_t> colRange = {0,2};
    threeByTwo.setData(dataToAdd, rowRange, colRange);    
    threeByTwoData[1][0] = 1.2;
    threeByTwoData[1][1] = 1.3;
    threeByTwoData[2][0] = 1.4;
    threeByTwoData[2][1] = 1.5;
    EXPECT_EQ(threeByTwo.getData(), threeByTwoData);
}

TEST_F(SetMatDataFixture, setDataWithTwoRangesBreaksAsIntended) {
    // index bounds must correspond to dimensions of data to set
    EXPECT_THROW(threeByTwo.setData({{1.0}}, {0,2}, {0,1}), std::invalid_argument);
    EXPECT_THROW(threeByTwo.setData({{1.0}}, {0,1}, {0,2}), std::invalid_argument);

    // index bounds cannot exceed those of matrix
    EXPECT_THROW(threeByTwo.setData({{1.0}}, {3,4}, {0,1}), std::invalid_argument);
    EXPECT_THROW(threeByTwo.setData({{1.0}}, {0,1}, {2,3}), std::invalid_argument);

    // "empty" vectors must be accompanied by corresponding indices,
    // even if indices technically match the size of the given std::vector<std::vector<double>>
    EXPECT_THROW(threeByTwo.setData({{}, {}, {}}, {0,3}, {0,1}), std::invalid_argument);
}

TEST_F(SetMatDataFixture, setDataDefaultWorks) { 
    Matrix<double> singleEltNew(1,1);
    singleEltNew.setData(singleEltMatData);
    EXPECT_EQ(singleEltNew.getData(), singleEltMatData);

    std::vector<std::vector<double>> oneByTwoNewData = {{5.0f, 5.0f}};
    Matrix<double> oneByTwoNew(oneByTwoNewData);
    oneByTwoNew.setData(oneByTwoData);
    EXPECT_EQ(oneByTwoNew.getData(), oneByTwoData);

    std::vector<std::vector<double>> threeByTwoNewData = {{7.1f, 3.4f}, {-1.3f, -7.2f}, {3.1f, 3.1f}};
    Matrix<double> threeByTwoNew(threeByTwoNewData);
    threeByTwoNew.setData(threeByTwoData);
    EXPECT_EQ(threeByTwoNew.getData(), threeByTwoData);
}

TEST_F(SetMatDataFixture, setDataDefaultBreaksAsIntended) {
    EXPECT_THROW(singleEltMat.setData(threeByTwoData), std::invalid_argument);
}

// I don't test error handling for the following setData overloads
// because they execute a simple call to the double-ranged setData
TEST_F(SetMatDataFixture, setDataWithColRangeWorks) {
    singleEltMat.setData({-6.0}, 0, {0,1});

    threeByTwo.setData({18.0, 12.0}, 1, {0,2});
    threeByTwoData[1][0] = 18.0;
    threeByTwoData[1][1] = 12.0;
    EXPECT_EQ(threeByTwo.getData(), threeByTwoData);
}

TEST_F(SetMatDataFixture, setDataWithRowRangeWorks) {
    singleEltMat.setData({-6.0}, {0,1}, 0);

    threeByTwo.setData({18.0, 12.0}, {1,3}, 1);
    threeByTwoData[1][1] = 18.0;
    threeByTwoData[2][1] = 12.0;
    EXPECT_EQ(threeByTwo.getData(), threeByTwoData);
}

TEST_F(SetMatDataFixture, setDataWithRowRangeBreaksAsIntended) {
    EXPECT_THROW(threeByTwo.setData({18.0f, 12.0f, 6.0f}, {1,4}, 1), std::invalid_argument);
}

TEST_F(SetMatDataFixture, setDataWithTwoIndicesWorks) {
    singleEltMat.setData(-6.0f, 0, 0);

    threeByTwo.setData(12.0f, 2, 1);
    threeByTwoData[2][1] = 12.0f;
    EXPECT_EQ(threeByTwo.getData(), threeByTwoData);
}
