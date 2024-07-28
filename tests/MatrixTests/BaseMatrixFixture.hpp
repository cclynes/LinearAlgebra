#ifndef BASE_MATRIX_FIXTURE_HPP    
#define BASE_MATRIX_FIXTURE_HPP

#include "../../include/Matrix.hpp"
#include "../../include/Vector.hpp"
#include "gtest/gtest.h"
#include <random>

using namespace LinearAlgebra;

// base fixture to define matrices that are common among many tests
class BaseMatrixFixture : public ::testing::Test {
protected:
    std::vector<std::vector<double>> singleEltMatData;
    std::vector<std::vector<double>> oneByTwoData;
    std::vector<std::vector<double>> twoByTwoData;
    std::vector<std::vector<double>> threeByTwoData;

    Matrix<double> defaultMat; // will not be further defined in this class
    Matrix<double> emptyMat;
    Matrix<double> singleEltMat;
    Matrix<double> oneByTwo;
    Matrix<double> twoByTwo;
    Matrix<double> threeByTwo;

    virtual void SetUp() override {
        emptyMat = Matrix<double>(0,0);

        singleEltMatData = {{-2.42f}};
        singleEltMat = Matrix<double>(singleEltMatData);

        oneByTwoData = {{1.0f, -2.0f}};
        oneByTwo = Matrix<double>(oneByTwoData);

        twoByTwoData = {{-2.5f, 0.0f}, {0.0f, 2.0f}};
        twoByTwo = Matrix<double>(twoByTwoData);

        threeByTwoData = {{5.5f, 7.0f}, {2.5f, 5.5f}, {-3.0f, -4.0f}};
        threeByTwo = Matrix<double>(threeByTwoData);
    }
};

#endif