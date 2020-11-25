#pragma once

//#include <cstdlib>
//#include <cmath>
//#include <iostream>
//#include <iomanip>
//#include <vector>
//#include <cmath>

//#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <Eigen/LU>
//#include <Eigen/CXX11/Tensor>

using namespace Eigen;

// The order that the above two header files are loaded seems to affect the result slightly.

double grid_interpolate_at_one_point(const VectorXd p,
                                     const double xmin, const double xmax,
                                     const double ymin, const double ymax,
                                     const MatrixXd & grid_values);


VectorXd grid_interpolate(const MatrixXd & eval_coords,
                          double xmin, double xmax, double ymin, double ymax,
                          const MatrixXd & grid_values);


VectorXd grid_interpolate_vectorized(const MatrixXd & eval_coords,
                                     double xmin, double xmax, double ymin, double ymax,
                                     const MatrixXd & grid_values);

