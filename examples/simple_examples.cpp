#include "../include/LinearAlgebra.hpp"
#include <iostream>

using namespace LinearAlgebra;
namespace la = LinearAlgebra;

// create some Vectors and Matrices

la::Vector<double> rowVec = Vector<double>({1.0, 2.0, 3.0}, true); // the "true" tells us this should be a row vector
la::Vector<double> colVec = {-3, -4, -5, 6}; // column vectors are simpler to define

la::Matrix<double> threeByFour = {
    {2, -3, 4, -1},
    {1, -1, 1, -1},
    {3, 4.5, 5, -1}};

la::Matrix<double> threeByThree = {
    {2, 1, 0},
    {-1, -1, -1},
    {1, 0, -2}};

// multiply the row Vector, three-by-four Matrix, and column Vector
// this will give us a 1x1 Matrix
la::Matrix<double> multipleAsMat = rowVec * threeByFour * colVec;

// if we want a scalar instead, we can call Vector's dot method:
la::Vector<double> rightMultiple = threeByFour * colVec;
double multiple = rowVec.dot(rightMultiple);

// multiply the square Matrix by the scalar:
la::Matrix<double> newThreeByThree = multiple * threeByThree;

// get some information about our objects
double colVecNorm = colVec.norm();
double threeByFourRank = threeByFour.rank();

// solve some systems of equations
la::Vector<double> threeEltColVec = {1.0, 2.0, -1.0};
double tol = 1e-8; // define the error tolerance

la::Vector<double> sol = threeByThree.solveSystem(threeEltColVec, tol); // where threeByThree * sol = threeEltColVec
// if we wanted, we could specify a solution method, e.g. using solveQR, solveLU, etc.

// we can also solve under- and over-determined systems
// systems with no solution will yield the least-squares solution
// systems with many solutions will yield the basic solution

la::Matrix<double> fourByThree = threeByFour.transpose();

// any non-square system will be solved by QR-decomposition using Householder vectors and pivoting
la::Vector<double> overDeterminedSol = fourByThree.solveSystem(colVec, tol);
la::Vector<double> underDeterminedSol = threeByFour.solveSystem(rowVec.transpose(), tol);

void printExamples() {
    std::cout << "rowVec: " << std::endl;
    rowVec.print();
    std::cout << std::endl;

    std::cout << "colVec: " << std::endl;
    colVec.print();
    std::cout << std::endl;

    std::cout << "threeByFour: " << std::endl;
    threeByFour.print();
    std::cout << std::endl;

    std::cout << "multiple: " << std::endl << multiple << std::endl << std::endl;

    std::cout << "newThreeByThree: " << std::endl;
    newThreeByThree.print();
    std::cout << std::endl;

    std::cout << "colVec norm: " << colVecNorm << std::endl;
    std::cout << "threeByFour rank: " << threeByFourRank << std::endl << std::endl;

    std::cout << "sol: " << std::endl;
    sol.print();
    std::cout << std::endl;

    std::cout << "overDeterminedSol: " << std::endl;
    overDeterminedSol.print();
    std::cout << std::endl;

    std::cout << "underDeterminedSol: " << std::endl;
    underDeterminedSol.print();
    std::cout << std::endl;
}

int main() {
    printExamples();
    return 0;
}