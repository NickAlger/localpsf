//
// Project     : HLib
// File        : bem1d.cc
// Description : example for dense H-matrix using a 1d integral equation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2014. All Rights Reserved.
//

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

#include <pybind11/pybind11.h>
#include <hlib.hh>

#include <Eigen/Dense>
//#include <Eigen/CXX11/Tensor>
#include <pybind11/eigen.h>

using namespace Eigen;


// The order that the above two header files are loaded seems to affect the result slightly.

namespace py = pybind11;

using namespace std;
using namespace HLIB;

#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif
//
// coefficient function for log|x-y| in [0,1]
//
class TLogCoeffFn : public TCoeffFn< real_t >
{
private:
    // stepwidth
    const double  _h;

public:
    // constructor
    TLogCoeffFn ( const double  h )
            : _h(h)
    {}

    //
    // coefficient evaluation
    //
    virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                         const std::vector< idx_t > &  colidxs,
                         real_t *                      matrix ) const
    {
        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();

        for ( size_t  j = 0; j < m; ++j )
        {
            const idx_t  idx1 = colidxs[ j ];
            
            for ( size_t  i = 0; i < n; ++i )
            {
                const idx_t  idx0 = rowidxs[ i ];
                double       value;

                if ( idx0 == idx1 ) 
                    value = -1.5*_h*_h + _h*_h*std::log(_h);
                else
                {
                    const double dist = _h * ( std::abs( double( idx0 - idx1 ) ) - 1.0 );
                    const double t1   = dist+1.0*_h;
                    const double t2   = dist+2.0*_h;
            
                    value = ( - 1.5*_h*_h + 0.5*t2*t2*std::log(t2) - t1*t1*std::log(t1) );
            
                    if ( std::abs(dist) > 1e-8 )
                        value += 0.5*dist*dist*std::log(dist);
                }
        
                matrix[ j*n + i ] = real_t(-value);
            }// for
        }// for
    }
    using TCoeffFn< real_t >::eval;

    //
    // return format of matrix, e.g. symmetric or hermitian
    //
    virtual matform_t  matrix_format  () const { return MATFORM_SYM; }
    
};

//
// function for evaluating rhs (for solution u = -1)
//
real_t
rhs ( const idx_t  i,
      const idx_t  n )
{
    const real_t  a     = real_t(i) / real_t(n);
    const real_t  b     = (real_t(i)+real_t(1)) / real_t(n);
    real_t        value = real_t(-1.5) * (b - a);
    
    if ( std::abs( b )       > real_t(1e-8) ) value += real_t(0.5)*b*b*std::log(b);
    if ( std::abs( a )       > real_t(1e-8) ) value -= real_t(0.5)*a*a*std::log(a);
    if ( std::abs( 1.0 - b ) > real_t(1e-8) ) value -= real_t(0.5)*(real_t(1)-b)*(real_t(1)-b)*std::log(real_t(1)-b);
    if ( std::abs( 1.0 - a ) > real_t(1e-8) ) value += real_t(0.5)*(real_t(1)-a)*(real_t(1)-a)*std::log(real_t(1)-a); 
    
    return value;
}


VectorXd grid_interpolate(MatrixXd eval_coords,
//                          VectorXd min_point, VectorXd max_pt,
                          double xmin, double xmax, double ymin, double ymax,
                          MatrixXd grid_values)
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

    VectorXd eval_values;
    eval_values.resize(N);
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


class CustomTLogCoeffFn : public TCoeffFn< real_t >
{
private:
    MatrixXd dof_coords;
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    MatrixXd grid_values;

public:
    // constructor
    CustomTLogCoeffFn (MatrixXd dof_coords,
                 double xmin, double xmax, double ymin, double ymax,
                 MatrixXd grid_values)
            : dof_coords(dof_coords), xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), grid_values(grid_values)
    {}

    //
    // coefficient evaluation
    //
    virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                         const std::vector< idx_t > &  colidxs,
                         real_t *                      matrix ) const
    {
        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();

        MatrixXd eval_coords;
        eval_coords.resize(n*m, 2);

        for ( size_t  j = 0; j < m; ++j )
        {
            const idx_t  idxj = colidxs[ j ];
            for ( size_t  i = 0; i < n; ++i )
            {
                const idx_t  idxi = rowidxs[ i ];
                eval_coords(j*n + i, 0) = dof_coords(idxj,0) - dof_coords(idxi,0);
                eval_coords(j*n + i, 1) = dof_coords(idxj,1) - dof_coords(idxi,1);
            }// for
        }// for

        VectorXd eval_values = grid_interpolate(eval_coords, xmin, xmax, ymin, ymax, grid_values);

        for ( size_t  j = 0; j < m; ++j )
            {
                for ( size_t  i = 0; i < n; ++i )
                {
                    matrix[ j*n + i ] = eval_values(j*n + i);
                }// for
            }// for
    }
    using TCoeffFn< real_t >::eval;

    //
    // return format of matrix, e.g. symmetric or hermitian
    //
    virtual matform_t  matrix_format  () const { return MATFORM_SYM; }

};

int bem1d ( int user_input )
{
    real_t        eps  = real_t(1e-4);
    size_t        n    = 512;
    const size_t  nmin = 60;

    printf("The argument passed in: %d\n", user_input);

    double h = 1.0 / double(n);
    
    printf("This is user_bem1d\n\n");

    try
    {
        //
        // init HLIBpro
        //
        
        INIT();

        CFG::set_verbosity( 3 );

        //
        // build coordinates
        //

        vector< double * >  vertices( n, NULL );
        vector< double * >  bbmin( n, NULL );
        vector< double * >  bbmax( n, NULL );

        for ( size_t i = 0; i < n; i++ )
        {
            vertices[i]    = new double;
            vertices[i][0] = h * double(i) + ( h / 2.0 ); // center of [i/h,(i+1)/h]

            // set bounding box (support) to [i/h,(i+1)/h]
            bbmin[i]       = new double;
            bbmin[i][0]    = h * double(i);
            bbmax[i]       = new double;
            bbmax[i][0]    = h * double(i+1);
        }// for

        unique_ptr< TCoordinate >  coord( new TCoordinate( vertices, 1, bbmin, bbmax ) );

        //
        // build cluster tree and block cluster tree
        //

        TAutoBSPPartStrat  part_strat;
        TBSPCTBuilder      ct_builder( & part_strat, nmin );
        auto               ct = ct_builder.build( coord.get() );
        TStdGeomAdmCond    adm_cond( 2.0 );
        TBCBuilder         bct_builder;
        auto               bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );

        if( verbose( 2 ) )
        {
            TPSClusterVis        c_vis;
            TPSBlockClusterVis   bc_vis;
            
            c_vis.print( ct->root(), "bem1d_ct" );
            bc_vis.print( bct->root(), "bem1d_bct" );
        }// if
                
        //
        // build matrix
        //
        
        std::cout << "━━ building H-matrix ( eps = " << eps << " )" << std::endl;

        TTimer                    timer( WALL_TIME );
        TConsoleProgressBar       progress;
        TTruncAcc                 acc( eps, 0.0 );
        TLogCoeffFn               log_coefffn( h );
        TPermCoeffFn< real_t >    coefffn( & log_coefffn, ct->perm_i2e(), ct->perm_i2e() );
        TACAPlus< real_t >        aca( & coefffn );
        TDenseMBuilder< real_t >  h_builder( & coefffn, & aca );
        TPSMatrixVis              mvis;
        
        // enable coarsening during construction
        h_builder.set_coarsening( false );

        timer.start();

        auto  A = h_builder.build( bct.get(), acc, & progress );
    
        timer.pause();
        std::cout << "    done in " << timer << std::endl;
        std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
        
        if( verbose( 2 ) )
        {
            mvis.svd( true );
            mvis.print( A.get(), "bem1d_A" );
        }// if

        {
            std::cout << std::endl << "━━ solving system" << std::endl;

            TMINRES      solver( 100 );
            TSolverInfo  solve_info( false, verbose( 2 ) );
            auto         b = A->row_vector();
            auto         x = A->col_vector();

            for ( size_t  i = 0; i < n; i++ )
                b->set_entry( i, rhs( i, n ) );

            // bring into H-ordering
            ct->perm_e2i()->permute( b.get() );
            
            timer.start();
    
            solver.solve( A.get(), x.get(), b.get(), NULL, & solve_info );
    
            if ( solve_info.has_converged() )
                std::cout << "  converged in " << timer << " and "
                          << solve_info.n_iter() << " steps with rate " << solve_info.conv_rate()
                          << ", |r| = " << solve_info.res_norm() << std::endl;
            else
                std::cout << "  not converged in " << timer << " and "
                          << solve_info.n_iter() << " steps " << std::endl;

            {
                auto  sol = b->copy();

                sol->fill( 1.0 );

                // bring into external ordering to compare with exact solution
                ct->perm_i2e()->permute( x.get() );
                
                x->axpy( 1.0, sol.get() );
                std::cout << "  |x-x~| = " << x->norm2() << std::endl;
            }
        }
        
        //
        // LU decomposition
        //

        auto  B = A->copy();
        
        std::cout << std::endl << "━━ LU factorisation ( eps = " << eps << " )" << std::endl;

        timer.start();

        auto  A_inv = factorise_inv( B.get(), acc, & progress );
    
        timer.pause();
        std::cout << "    done in " << timer << std::endl;

        std::cout << "    size of LU factor = " << Mem::to_string( B->byte_size() ) << std::endl;
        std::cout << "    inversion error   = " << std::scientific << std::setprecision( 4 )
                  << inv_approx_2( A.get(), A_inv.get() ) << std::endl;

        if( verbose( 2 ) )
            mvis.print( B.get(), "bem1d_LU" );

        //
        // solve with LU decomposition
        //

        auto  b = A->row_vector();

        for ( size_t  i = 0; i < n; i++ )
            b->set_entry( i, rhs( i, n ) );

        std::cout << std::endl << "━━ solving system" << std::endl;

        TAutoSolver  solver( 1000 );
        TSolverInfo  solve_info( false, verbose( 2 ) );
        auto         x = A->col_vector();

        timer.start();
    
        solver.solve( A.get(), x.get(), b.get(), A_inv.get(), & solve_info );
    
        if ( solve_info.has_converged() )
            std::cout << "  converged in " << timer << " and "
                      << solve_info.n_iter() << " steps with rate " << solve_info.conv_rate()
                      << ", |r| = " << solve_info.res_norm() << std::endl;
        else
            std::cout << "  not converged in " << timer << " and "
                      << solve_info.n_iter() << " steps " << std::endl;

        {
            auto  sol = b->copy();

            sol->fill( 1.0 );
            x->axpy( 1.0, sol.get() );
            
            std::cout << "  |x-x~| = " << x->norm2() << std::endl;
        }
    
        DONE();
    }// try
    catch ( Error & e )
    {
        std::cout << e.to_string() << std::endl;
    }// catch
    
    return 0;
}


int Custom_bem1d (MatrixXd dof_coords, double xmin, double xmax, double ymin, double ymax, MatrixXd grid_values)
{
    real_t        eps  = real_t(1e-4);
    size_t        n    = 512;
    const size_t  nmin = 60;


    double h = 1.0 / double(n);

    printf("This is user_bem1d\n\n");

    try
    {
        //
        // init HLIBpro
        //

        INIT();

        CFG::set_verbosity( 3 );

        //
        // build coordinates
        //

        vector< double * >  vertices( n, NULL );
        vector< double * >  bbmin( n, NULL );
        vector< double * >  bbmax( n, NULL );

        for ( size_t i = 0; i < n; i++ )
        {
            vertices[i]    = new double;
            vertices[i][0] = h * double(i) + ( h / 2.0 ); // center of [i/h,(i+1)/h]

            // set bounding box (support) to [i/h,(i+1)/h]
            bbmin[i]       = new double;
            bbmin[i][0]    = h * double(i);
            bbmax[i]       = new double;
            bbmax[i][0]    = h * double(i+1);
        }// for

        unique_ptr< TCoordinate >  coord( new TCoordinate( vertices, 1, bbmin, bbmax ) );

        //
        // build cluster tree and block cluster tree
        //

        TAutoBSPPartStrat  part_strat;
        TBSPCTBuilder      ct_builder( & part_strat, nmin );
        auto               ct = ct_builder.build( coord.get() );
        TStdGeomAdmCond    adm_cond( 2.0 );
        TBCBuilder         bct_builder;
        auto               bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );

        if( verbose( 2 ) )
        {
            TPSClusterVis        c_vis;
            TPSBlockClusterVis   bc_vis;

            c_vis.print( ct->root(), "bem1d_ct" );
            bc_vis.print( bct->root(), "bem1d_bct" );
        }// if

        //
        // build matrix
        //

        std::cout << "━━ building H-matrix ( eps = " << eps << " )" << std::endl;

        TTimer                    timer( WALL_TIME );
        TConsoleProgressBar       progress;
        TTruncAcc                 acc( eps, 0.0 );
        TLogCoeffFn               log_coefffn( h );
        TPermCoeffFn< real_t >    coefffn( & log_coefffn, ct->perm_i2e(), ct->perm_i2e() );
        TACAPlus< real_t >        aca( & coefffn );
        TDenseMBuilder< real_t >  h_builder( & coefffn, & aca );
        TPSMatrixVis              mvis;

        // enable coarsening during construction
        h_builder.set_coarsening( false );

        timer.start();

        auto  A = h_builder.build( bct.get(), acc, & progress );

        timer.pause();
        std::cout << "    done in " << timer << std::endl;
        std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;

        if( verbose( 2 ) )
        {
            mvis.svd( true );
            mvis.print( A.get(), "bem1d_A" );
        }// if

        {
            std::cout << std::endl << "━━ solving system" << std::endl;

            TMINRES      solver( 100 );
            TSolverInfo  solve_info( false, verbose( 2 ) );
            auto         b = A->row_vector();
            auto         x = A->col_vector();

            for ( size_t  i = 0; i < n; i++ )
                b->set_entry( i, rhs( i, n ) );

            // bring into H-ordering
            ct->perm_e2i()->permute( b.get() );

            timer.start();

            solver.solve( A.get(), x.get(), b.get(), NULL, & solve_info );

            if ( solve_info.has_converged() )
                std::cout << "  converged in " << timer << " and "
                          << solve_info.n_iter() << " steps with rate " << solve_info.conv_rate()
                          << ", |r| = " << solve_info.res_norm() << std::endl;
            else
                std::cout << "  not converged in " << timer << " and "
                          << solve_info.n_iter() << " steps " << std::endl;

            {
                auto  sol = b->copy();

                sol->fill( 1.0 );

                // bring into external ordering to compare with exact solution
                ct->perm_i2e()->permute( x.get() );

                x->axpy( 1.0, sol.get() );
                std::cout << "  |x-x~| = " << x->norm2() << std::endl;
            }
        }

        //
        // LU decomposition
        //

        auto  B = A->copy();

        std::cout << std::endl << "━━ LU factorisation ( eps = " << eps << " )" << std::endl;

        timer.start();

        auto  A_inv = factorise_inv( B.get(), acc, & progress );

        timer.pause();
        std::cout << "    done in " << timer << std::endl;

        std::cout << "    size of LU factor = " << Mem::to_string( B->byte_size() ) << std::endl;
        std::cout << "    inversion error   = " << std::scientific << std::setprecision( 4 )
                  << inv_approx_2( A.get(), A_inv.get() ) << std::endl;

        if( verbose( 2 ) )
            mvis.print( B.get(), "bem1d_LU" );

        //
        // solve with LU decomposition
        //

        auto  b = A->row_vector();

        for ( size_t  i = 0; i < n; i++ )
            b->set_entry( i, rhs( i, n ) );

        std::cout << std::endl << "━━ solving system" << std::endl;

        TAutoSolver  solver( 1000 );
        TSolverInfo  solve_info( false, verbose( 2 ) );
        auto         x = A->col_vector();

        timer.start();

        solver.solve( A.get(), x.get(), b.get(), A_inv.get(), & solve_info );

        if ( solve_info.has_converged() )
            std::cout << "  converged in " << timer << " and "
                      << solve_info.n_iter() << " steps with rate " << solve_info.conv_rate()
                      << ", |r| = " << solve_info.res_norm() << std::endl;
        else
            std::cout << "  not converged in " << timer << " and "
                      << solve_info.n_iter() << " steps " << std::endl;

        {
            auto  sol = b->copy();

            sol->fill( 1.0 );
            x->axpy( 1.0, sol.get() );

            std::cout << "  |x-x~| = " << x->norm2() << std::endl;
        }

        DONE();
    }// try
    catch ( Error & e )
    {
        std::cout << e.to_string() << std::endl;
    }// catch

    return 0;
}


PYBIND11_MODULE(user, m) {
    m.doc() = "pybind11 bem1d plugin"; // optional module docstring

    m.def("bem1d", &bem1d, "bem1d from hlibpro");
    m.def("grid_interpolate", &grid_interpolate, "grid_interpolate from cpp");
}
