#ifndef BASE_VECTOR_FIXTURE_HPP
#define BASE_VECTOR_FIXTURE_HPP

#include <gtest/gtest.h>
#include "../../include/Vector.hpp"


class BaseVectorFixture : public ::testing::Test {
protected:

    Vector<double> defaultVec; // will not be further defined
    Vector<double> singleEltRowVec;
    Vector<double> singleEltColVec;
    Vector<double> threeEltRowVec;
    Vector<double> threeEltColVec;
    Vector<double> fourEltRowVec;
    Vector<double> fourEltColVec;
    Vector<double> fourEltReverseRowVec;

    std::vector<double> singleEltVecData;
    std::vector<double> threeEltVecData;
    std::vector<double> fourEltVecData;
    std::vector<double> fourEltReverseVecData;

    // base vectors will go here

    virtual void SetUp() override {

        singleEltVecData = {-10.0f};
        threeEltVecData = {2.0f, 1.5f, -1.0f};
        fourEltVecData = {2.0f, -7.32f, 1000.5554f, -42.0f};
        fourEltReverseVecData = {-42.0f, 1000.5554f, -7.32f, 2.0f};

        singleEltRowVec = Vector<double>(singleEltVecData, true);
        singleEltColVec = Vector<double>(singleEltVecData, false);
        threeEltRowVec = Vector<double>(threeEltVecData, true);
        threeEltColVec = Vector<double>(threeEltVecData, false);
        fourEltRowVec = Vector<double>(fourEltVecData, true);
        fourEltColVec = Vector<double>(fourEltVecData, false);
        fourEltReverseRowVec = Vector<double>(fourEltReverseVecData, true);
    }
};

#endif