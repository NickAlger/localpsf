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
#include <cmath>

#include <pybind11/pybind11.h>
#include <hlib.hh>

#include <Eigen/Dense>
#include <Eigen/LU>
//#include <Eigen/CXX11/Tensor>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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
    if( (p(0) < xmin) || (p(0) >= xmax) || (p(1) < ymin) || (p(1) >= ymax))
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

bool point_is_in_ellipsoid(VectorXd z, VectorXd mu, MatrixXd Sigma, double tau)
{
    VectorXd p = z - mu;
    return ( p.dot(Sigma.lu().solve(p)) < pow(tau, 2) );
}

class ProductConvolutionOneBatch
{
private:
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    MatrixXd eta_array;
    std::vector<MatrixXd> ww_arrays;
    MatrixXd pp;
    MatrixXd mus;
    std::vector<MatrixXd> Sigmas;
    double tau;

public:
    ProductConvolutionOneBatch(): xmin(), xmax(), ymin(), ymax(),
                                  eta_array(), ww_arrays(), pp(),
                                  mus(), Sigmas(), tau()
                                  {}

    ProductConvolutionOneBatch(MatrixXd eta_array,
                               std::vector<MatrixXd> ww_arrays,
                               MatrixXd pp,
                               MatrixXd mus,
                               std::vector<MatrixXd> Sigmas,
                               double tau, double xmin, double xmax, double ymin, double ymax)
                                : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax),
                                  eta_array(eta_array), ww_arrays(ww_arrays), pp(pp),
                                  mus(mus), Sigmas(Sigmas), tau(tau)
                                  {}

    VectorXd compute_entries(const MatrixXd & yy, const MatrixXd & xx) const
    {
        int num_batch_points = ww_arrays.size();
        int num_eval_points = xx.rows();

        VectorXd pc_entries(num_eval_points);
        pc_entries.setZero();
        for ( int  i = 0; i < num_batch_points; ++i )
        {
            for ( int k = 0; k < num_eval_points; ++k )
            {
                Vector2d z = pp.row(i) + yy.row(k) - xx.row(k);
                if (point_is_in_ellipsoid(z, mus.row(i), Sigmas[i], tau))
                {
                    double w_ik = grid_interpolate_at_one_point(xx.row(k), xmin, xmax, ymin, ymax, ww_arrays[i]);
                    double phi_ik = grid_interpolate_at_one_point(z, xmin, xmax, ymin, ymax, eta_array);
                    pc_entries(k) += w_ik * phi_ik;
                }
            }
        }
        return pc_entries;
    }
};


class ProductConvolutionMultipleBatches
{
private:
    std::vector<ProductConvolutionOneBatch> pc_batches;

public:
    ProductConvolutionMultipleBatches(std::vector<MatrixXd> eta_array_batches,
                                      std::vector<std::vector<MatrixXd>> ww_array_batches,
                                      std::vector<MatrixXd> pp_batches,
                                      std::vector<MatrixXd> mus_batches,
                                      std::vector<std::vector<MatrixXd>> Sigmas_batches,
                                      double tau, double xmin, double xmax, double ymin, double ymax) : pc_batches()
                                      {
                                        int num_batches = eta_array_batches.size();
                                        pc_batches.resize(num_batches);
                                        for (int i = 0; i < num_batches; ++i)
                                        {
                                            pc_batches[i] = ProductConvolutionOneBatch(eta_array_batches[i],
                                                                                       ww_array_batches[i],
                                                                                       pp_batches[i],
                                                                                       mus_batches[i],
                                                                                       Sigmas_batches[i],
                                                                                       tau, xmin, xmax, ymin, ymax);
                                        }
                                      }

    VectorXd compute_entries(const MatrixXd & yy, const MatrixXd & xx) const
    {
        int num_batches = pc_batches.size();
        int num_eval_points = xx.rows();

        VectorXd pc_entries(num_eval_points);
        pc_entries.setZero();
        for (int i = 0; i < num_batches; ++i)
        {
            pc_entries += pc_batches[i].compute_entries(yy, xx);
        }
        return pc_entries;
    }
};


class ProductConvolutionCoeffFn : public TCoeffFn< real_t >
{
private:
    ProductConvolutionMultipleBatches pcb;
    MatrixXd dof_coords;

public:
    ProductConvolutionCoeffFn (const ProductConvolutionMultipleBatches & pcb, MatrixXd dof_coords)
            : pcb(pcb), dof_coords(dof_coords)
    {}

    virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                         const std::vector< idx_t > &  colidxs,
                         real_t *                      matrix ) const
    {
        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();

        MatrixXd xx(n,2);
        for (int i = 0; i < n; ++i)
        {
            const idx_t  idxi = rowidxs[ i ];
            xx.row(i) = dof_coords.row(idxi);
        }

        MatrixXd yy(m,2);
        for (int j = 0; j < n; ++j)
        {
            const idx_t  idxj = colidxs[ j ];
            yy.row(j) = dof_coords.row(idxj);
        }

        VectorXd eval_values = pcb.compute_entries(yy, xx);

        for ( size_t  j = 0; j < m; ++j )
            {
                for ( size_t  i = 0; i < n; ++i )
                {
                    matrix[ j*n + i ] = eval_values(j*n + i);
                }// for
            }// for
    }
    using TCoeffFn< real_t >::eval;

    virtual matform_t  matrix_format  () const { return MATFORM_SYM; }

};


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

        VectorXd eval_values = grid_interpolate_vectorized(eval_coords, xmin, xmax, ymin, ymax, grid_values);

        for ( size_t  j = 0; j < m; ++j )
            {
                const idx_t  idxj = colidxs[ j ];
                for ( size_t  i = 0; i < n; ++i )
                {
                    const idx_t  idxi = rowidxs[ i ];
                    matrix[ j*n + i ] = eval_values(j*n + i);
//                    if (idxj == idxi)
//                    {
////                        matrix[ j*n + i ] = 1.0;
//                        matrix[ j*n + i ] = matrix[ j*n + i ] + 20.0;
//                    }
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


std::unique_ptr<HLIB::TClusterTree> build_cluster_tree_from_dof_coords(const MatrixXd & dof_coords, const double nmin)
{
    size_t N = dof_coords.rows();
    size_t d = dof_coords.cols();

    vector< double * >  vertices( N );

    for ( size_t i = 0; i < N; i++ )
    {
        double * v    = new double[d];
        for (size_t j=0; j < d; ++j)
            v[j] = dof_coords(i,j);

        vertices[i] = v;
    }// for

    auto coord = make_unique< TCoordinate >( vertices, d );

//    TAutoBSPPartStrat  part_strat;
    TCardBSPPartStrat  part_strat;
    TBSPCTBuilder      ct_builder( & part_strat, nmin );
    std::unique_ptr<HLIB::TClusterTree>  ct = ct_builder.build( coord.get() );
    return ct;
}

std::unique_ptr<HLIB::TBlockClusterTree> build_block_cluster_tree(HLIB::TClusterTree *  row_ct_ptr,
                                                                  HLIB::TClusterTree * col_ct_ptr,
                                                                  double admissibility_eta)
{
        TStdGeomAdmCond    adm_cond( admissibility_eta );
        TBCBuilder         bct_builder;
        std::unique_ptr<HLIB::TBlockClusterTree>  bct = bct_builder.build( row_ct_ptr, col_ct_ptr, & adm_cond );
        return bct;
}

void initialize_hlibpro()
{
    int verbosity_level = 3;
    INIT();
    CFG::set_verbosity( verbosity_level );
}

void visualize_cluster_tree(HLIB::TClusterTree * ct_ptr, string title)
{
    TPSClusterVis        c_vis;
    c_vis.print( ct_ptr->root(), title );
}

void visualize_block_cluster_tree(HLIB::TBlockClusterTree * bct_ptr, string title)
{
    TPSBlockClusterVis   bc_vis;
    bc_vis.print( bct_ptr->root(), title );
}

std::unique_ptr<HLIB::TMatrix> build_hmatrix(TCoeffFn<real_t> & coefffn,
                                             HLIB::TClusterTree * row_ct_ptr,
                                             HLIB::TClusterTree * col_ct_ptr,
                                             HLIB::TBlockClusterTree * bct_ptr,
                                             double tol)
{
    std::cout << "━━ building H-matrix ( tol = " << tol << " )" << std::endl;
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;
    TTruncAcc                 acc( tol, 0.0 );
    TPermCoeffFn< real_t >    permuted_coefffn( & coefffn, row_ct_ptr->perm_i2e(), col_ct_ptr->perm_i2e() );
    TACAPlus< real_t >        aca( & permuted_coefffn );
    TDenseMBuilder< real_t >  h_builder( & permuted_coefffn, & aca );
    h_builder.set_coarsening( false );

    timer.start();

    std::unique_ptr<HLIB::TMatrix>  A = h_builder.build( bct_ptr, acc, & progress );

    timer.pause();
    std::cout << "    done in " << timer << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    return A;
}

void add_identity_to_hmatrix(HLIB::TMatrix * A_ptr, double s)
{
    add_identity(A_ptr, s);
}

void visualize_hmatrix(HLIB::TMatrix * A_ptr, string title)
{
    TPSMatrixVis              mvis;
    mvis.svd( true );
    mvis.print( A_ptr, title );
}

VectorXd hmatrix_matvec(HLIB::TMatrix * A_ptr,
                        HLIB::TClusterTree * row_ct_ptr,
                        HLIB::TClusterTree * col_ct_ptr,
                        VectorXd x)
{
    // y = A * x
    std::unique_ptr<HLIB::TVector> y_hlib = A_ptr->row_vector();
    std::unique_ptr<HLIB::TVector> x_hlib = A_ptr->col_vector();

    int n = y_hlib->size();
    int m = x_hlib->size();

    for ( size_t  i = 0; i < m; i++ )
        x_hlib->set_entry( i, x(i) );

    col_ct_ptr->perm_e2i()->permute( x_hlib.get() );

    A_ptr->mul_vec(1.0, x_hlib.get(), 0.0, y_hlib.get());

    row_ct_ptr->perm_i2e()->permute( y_hlib.get() );

    VectorXd y(n);
    for ( size_t  i = 0; i < n; i++ )
        y(i) = x_hlib->entry( i );

    return y;
}

std::unique_ptr< HLIB::TFacInvMatrix > hmatrix_factorized_inverse(HLIB::TMatrix * A_ptr, double tol)
{
    TTruncAcc                 acc( tol, 0.0 );
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;

    auto  B = A_ptr->copy(); // B gets deleted when this function ends ACK.

    std::cout << std::endl << "━━ LU factorisation ( tol = " << tol << " )" << std::endl;

    timer.start();

    std::unique_ptr<HLIB::TFacInvMatrix> A_inv = factorise_inv( B.get(), acc, & progress );

    timer.pause();
    std::cout << "    done in " << timer << std::endl;

    std::cout << "    size of LU factor = " << Mem::to_string( B->byte_size() ) << std::endl;
    std::cout << "    inversion error   = " << std::scientific << std::setprecision( 4 )
              << inv_approx_2( A_ptr, A_inv.get() ) << std::endl;

    return A_inv;
}

int Custom_bem1d (MatrixXd dof_coords, double xmin, double xmax, double ymin, double ymax,
                  MatrixXd grid_values, VectorXd rhs_b)
{
    real_t        eps  = real_t(1e-4);
    size_t        n    = dof_coords.rows();
    const size_t  nmin = 60;
    double admissibility_eta = 2.0;

    printf("This is user.Custom_bem1d\n\n");

    try
    {
        initialize_hlibpro();
        std::unique_ptr<HLIB::TClusterTree>  ct = build_cluster_tree_from_dof_coords(dof_coords, nmin);
        std::unique_ptr<HLIB::TBlockClusterTree>  bct = build_block_cluster_tree(ct.get(), ct.get(), admissibility_eta);

        if( verbose( 2 ) )
        {
            visualize_cluster_tree(ct.get(), "custom_bem2d_ct");
            visualize_block_cluster_tree(bct.get(), "custom_bem2d_bct");
        }// if

        CustomTLogCoeffFn log_coefffn( dof_coords, xmin, xmax, ymin, ymax, grid_values );
        std::unique_ptr<HLIB::TMatrix> A = build_hmatrix(log_coefffn, ct.get(), ct.get(), bct.get(), eps);

        add_identity_to_hmatrix(A.get(), 20.0);

        if( verbose( 2 ) )
        {
            visualize_hmatrix(A.get(), "custom_bem2d_A");
        }// if

        TPSMatrixVis              mvis;
        TTimer                    timer( WALL_TIME );
        TConsoleProgressBar       progress;
        TTruncAcc                 acc( eps, 0.0 );

        {
            std::cout << std::endl << "━━ solving system" << std::endl;

            TMINRES      solver( 100 );
            TSolverInfo  solve_info( false, verbose( 2 ) );
            std::unique_ptr<HLIB::TVector>         b = A->row_vector();
            std::unique_ptr<HLIB::TVector>         x = A->col_vector();

            for ( size_t  i = 0; i < n; i++ )
                b->set_entry( i, rhs_b( i) );

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

        std::unique_ptr<HLIB::TFacInvMatrix> A_inv = hmatrix_factorized_inverse(A.get(), eps);

//        auto  B = A->copy();
//
//        std::cout << std::endl << "━━ LU factorisation ( eps = " << eps << " )" << std::endl;
//
//        timer.start();
//
//        auto  A_inv = factorise_inv( B.get(), acc, & progress );
//
//        timer.pause();
//        std::cout << "    done in " << timer << std::endl;
//
//        std::cout << "    size of LU factor = " << Mem::to_string( B->byte_size() ) << std::endl;
//        std::cout << "    inversion error   = " << std::scientific << std::setprecision( 4 )
//                  << inv_approx_2( A.get(), A_inv.get() ) << std::endl;
//


//        char asdf = A_inv->matrix();
        if( verbose( 2 ) )
            mvis.print( A_inv->matrix(), "custom_bem2d_LU" );
//            mvis.print( B.get(), "custom_bem2d_LU" );

        //
        // solve with LU decomposition
        //

        auto  b = A->row_vector();

        for ( size_t  i = 0; i < n; i++ )
            b->set_entry( i, rhs_b( i ) );

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



PYBIND11_MODULE(hlibpro_experiments1, m) {
    py::class_<HLIB::TFacInvMatrix>(m, "HLIB::TFacInvMatrix");

    py::class_<HLIB::TMatrix>(m, "HLIB::TMatrix");

    py::class_<TCoeffFn<real_t>>(m, "TCoeffFn<real_t>");

    py::class_<CustomTLogCoeffFn, TCoeffFn<real_t>>(m, "CustomTLogCoeffFn")
        .def(py::init<MatrixXd, // dof_coords
             double, // xmin
             double, // xmax
             double, // ymin
             double,  // ymax
             MatrixXd // grid_values
             >());

    py::class_<ProductConvolutionOneBatch>(m, "ProductConvolutionOneBatch")
        .def(py::init<MatrixXd, // eta
             std::vector<MatrixXd>, // ww
             MatrixXd, // pp
             MatrixXd, // mus
             std::vector<MatrixXd>, // Sigmas
             double, // tau
             double, // xmin
             double, // xmax
             double, // ymin
             double  // ymax
             >())
        .def("compute_entries", &ProductConvolutionOneBatch::compute_entries);

    py::class_<ProductConvolutionMultipleBatches>(m, "ProductConvolutionMultipleBatches")
        .def(py::init<std::vector<MatrixXd>, // eta_array_batches
                      std::vector<std::vector<MatrixXd>>, // ww_array_batches
                      std::vector<MatrixXd>, // pp_batches
                      std::vector<MatrixXd>, // mus_batches
                      std::vector<std::vector<MatrixXd>>, // Sigmas_batches
                      double, // tau
                      double, // xmin
                      double, // xmax
                      double, // ymin
                      double // ymax
                      >())
        .def("compute_entries", &ProductConvolutionMultipleBatches::compute_entries);

    py::class_<HLIB::TClusterTree>(m, "HLIB::TClusterTree");
    py::class_<HLIB::TBlockClusterTree>(m, "HLIB::TBlockClusterTree");

    m.doc() = "pybind11 bem1d plugin"; // optional module docstring

    m.def("bem1d", &bem1d, "bem1d from hlibpro");
    m.def("Custom_bem1d", &Custom_bem1d, "Custom_bem1d from hlibpro");
    m.def("grid_interpolate", &grid_interpolate, "grid_interpolate from cpp");
    m.def("grid_interpolate_vectorized", &grid_interpolate_vectorized, "grid_interpolate_vectorized from cpp");
//    m.def("compute_product_convolution_entries", &compute_product_convolution_entries, "compute_product_convolution_entries from cpp");
    m.def("build_cluster_tree_from_dof_coords", &build_cluster_tree_from_dof_coords, "build_cluster_tree_from_dof_coords from hlibpro");
    m.def("build_block_cluster_tree", &build_block_cluster_tree, "build_block_cluster_tree");
    m.def("initialize_hlibpro", &initialize_hlibpro, "initialize_hlibpro");
    m.def("visualize_cluster_tree", &visualize_cluster_tree, "visualize_cluster_tree");
    m.def("visualize_block_cluster_tree", &visualize_block_cluster_tree, "visualize_block_cluster_tree");
    m.def("build_hmatrix", &build_hmatrix, "build_hmatrix from hlibpro");
    m.def("add_identity_to_hmatrix", &add_identity_to_hmatrix, "add_identity from hlibpro");
    m.def("visualize_hmatrix", &visualize_hmatrix, "visualize_hmatrix from hlibpro");
    m.def("hmatrix_matvec", &hmatrix_matvec, "hmatrix_matvec from hlibpro");
    m.def("hmatrix_factorized_inverse", &hmatrix_factorized_inverse, "hmatrix_factorized_inverse from hlibpro");
}

