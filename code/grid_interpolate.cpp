#include "grid_interpolate.h"

#include <cstdlib>
//#include <cmath>
#include <iostream>
//#include <iomanip>
//#include <vector>
//#include <cmath>

//#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <Eigen/LU>
//#include <Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;

// The order that the above two header files are loaded seems to affect the result slightly.

double grid_interpolate_at_one_point(const VectorXd p,
                                     const double xmin, const double xmax,
                                     const double ymin, const double ymax,
                                     const MatrixXd & grid_values)
{
    const int nx = grid_values.rows();
    const int ny = grid_values.cols();

    double x_width = (xmax - xmin);
    double y_width = (ymax - ymin);
    double num_cells_x = nx-1;
    double num_cells_y = ny-1;
    double hx = x_width / num_cells_x;
    double hy = y_width / num_cells_y;

    double value_at_p;
    if( (p(0) < xmin) || (p(0) > xmax) || (p(1) < ymin) || (p(1) > ymax))
        value_at_p = 0.0;
    else
    {
        double quotx = (p(0) - xmin) / hx;
        int i = (int)quotx;
        double s = quotx - ((double)i);

        double quoty = (p(1) - ymin) / hy;
        int j = (int)quoty;
        double t = quoty - ((double)j);

        double v00 = grid_values(i,   j);
        double v01 = grid_values(i,   j+1);
        double v10 = grid_values(i+1, j);
        double v11 = grid_values(i+1, j+1);

        value_at_p = (1.0-s)*(1.0-t)*v00 + (1.0-s)*t*v01 + s*(1.0-t)*v10 + s*t*v11;
    }
    return value_at_p;
}

VectorXd grid_interpolate(const MatrixXd & eval_coords,
                          double xmin, double xmax, double ymin, double ymax,
                          const MatrixXd & grid_values)
{
    const int N = eval_coords.rows();
    VectorXd eval_values(N);
    eval_values.setZero();
    for ( int  k = 0; k < N; ++k )
    {
        VectorXd pk = eval_coords.row(k);
        eval_values(k) = grid_interpolate_at_one_point(pk, xmin, xmax, ymin, ymax, grid_values);
    }
    return eval_values;
}

VectorXd grid_interpolate_vectorized(const MatrixXd & eval_coords,
                                     double xmin, double xmax, double ymin, double ymax,
                                     const MatrixXd & grid_values)
{
//    int d = min_point.size()
    int d = 2;
    const int N = eval_coords.rows();
    const int nx = grid_values.rows();
    const int ny = grid_values.cols();
//    VectorXd widths = max_point - min_point
    double x_width = (xmax - xmin);
    double y_width = (ymax - ymin);
    double num_cells_x = nx-1;
    double num_cells_y = ny-1;
    double hx = x_width / num_cells_x;
    double hy = y_width / num_cells_y;

//    if(eval_coords.cols() != d)
//        throw runtime_error(std::string('points of different dimension than grid'));

    VectorXd eval_values(N);
    eval_values.setZero();
//    eval_values.resize(N);
    for ( int  k = 0; k < N; ++k )
    {
        double px = eval_coords(k,0);
        double py = eval_coords(k,1);

        if( (px < xmin) || (px >= xmax) || (py < ymin) || (py >= ymax))
            eval_values(k) = 0.0;
        else
        {
            double quotx = (px - xmin) / hx;
            int i = (int)quotx;
            double s = quotx - ((double)i);

            double quoty = (py - ymin) / hy;
            int j = (int)quoty;
            double t = quoty - ((double)j);

            double v00 = grid_values(i,   j);
            double v01 = grid_values(i,   j+1);
            double v10 = grid_values(i+1, j);
            double v11 = grid_values(i+1, j+1);

            eval_values(k) = (1.0-s)*(1.0-t)*v00 + (1.0-s)*t*v01 + s*(1.0-t)*v10 + s*t*v11;
        }
    }
    return eval_values;
}