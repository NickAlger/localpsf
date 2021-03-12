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

#include "grid_interpolate.h"
#include "product_convolution_hmatrix.h"

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


std::shared_ptr<HLIB::TClusterTree> build_cluster_tree_from_dof_coords(const MatrixXd & dof_coords, const double nmin)
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
    std::shared_ptr<HLIB::TClusterTree>  ct = ct_builder.build( coord.get() );
    return ct;
}

std::shared_ptr<HLIB::TBlockClusterTree> build_block_cluster_tree(HLIB::TClusterTree *  row_ct_ptr,
                                                                  HLIB::TClusterTree * col_ct_ptr,
                                                                  double admissibility_eta)
{
        TStdGeomAdmCond    adm_cond( admissibility_eta );
        TBCBuilder         bct_builder;
        std::shared_ptr<HLIB::TBlockClusterTree>  bct = bct_builder.build( row_ct_ptr, col_ct_ptr, & adm_cond );
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


std::shared_ptr<HLIB::TMatrix> build_hmatrix_from_sparse_matfile (string mat_file,
                                                                  HLIB::TBlockClusterTree * bct_ptr)
{
    auto row_ct_ptr = bct_ptr->row_ct();
    auto col_ct_ptr = bct_ptr->col_ct();

    auto               M = read_matrix( mat_file );

    if ( ! IS_TYPE( M, TSparseMatrix ) )
    {
        cout << "given matrix is not sparse (" << M->typestr() << ")" << endl;
        exit( 1 );
    }

    auto               S = ptrcast( M.get(), TSparseMatrix );

    cout << "  matrix has dimension " << S->rows() << " x " << S->cols() << endl
         << "    no of non-zeroes    = " << S->n_non_zero() << endl
         << "    matrix is             " << ( S->is_complex() ? "complex" : "real" )
         << " valued" << endl
         << "    format              = ";
    if      ( S->is_nonsym()    ) cout << "non symmetric" << endl;
    else if ( S->is_symmetric() ) cout << "symmetric" << endl;
    else if ( S->is_hermitian() ) cout << "hermitian" << endl;
    cout << "  size of sparse matrix = " << Mem::to_string( S->byte_size() ) << endl;
    cout << "  |S|_F                 = " << norm_F( S ) << endl;

    cout << "    sparsity constant = " << bct_ptr->compute_c_sp() << endl;

    TSparseMBuilder    h_builder( S, row_ct_ptr->perm_i2e(), col_ct_ptr->perm_e2i() );

    TTruncAcc                 acc(0.0, 0.0);
//    auto               A = h_builder.build( bct_ptr, acc );
    std::shared_ptr<HLIB::TMatrix> A = h_builder.build( bct_ptr, acc );

    cout << "    size of H-matrix  = " << Mem::to_string( A->byte_size() ) << endl;
    cout << "    |A|_F             = " << norm_F( A.get() ) << endl;

    {
        auto  PA = make_unique< TPermMatrix >( row_ct_ptr->perm_i2e(), A.get(), col_ct_ptr->perm_e2i() );

        cout << " |S-A|_2 = " << diff_norm_2( S, PA.get() ) << endl;
    }

    return A;
}


std::shared_ptr<HLIB::TMatrix> build_hmatrix_from_coefffn(TCoeffFn<real_t> & coefffn,
//                                                          HLIB::TClusterTree * row_ct_ptr,
//                                                          HLIB::TClusterTree * col_ct_ptr,
                                                          HLIB::TBlockClusterTree * bct_ptr,
                                                          double tol)
{
    const HLIB::TClusterTree * row_ct_ptr = bct_ptr->row_ct();
    const HLIB::TClusterTree * col_ct_ptr = bct_ptr->col_ct();
    std::cout << "━━ building H-matrix ( tol = " << tol << " )" << std::endl;
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;
    TTruncAcc                 acc( tol, 0.0 );
    TPermCoeffFn< real_t >    permuted_coefffn( & coefffn, row_ct_ptr->perm_i2e(), col_ct_ptr->perm_i2e() );
    TACAPlus< real_t >        aca( & permuted_coefffn );
    TDenseMBuilder< real_t >  h_builder( & permuted_coefffn, & aca );
    h_builder.set_coarsening( false );

    timer.start();

//    std::unique_ptr<HLIB::TMatrix>  A = h_builder.build( bct_ptr, acc, & progress );
    std::shared_ptr<HLIB::TMatrix>  A = h_builder.build( bct_ptr, acc, & progress );

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

VectorXd h_matvec(HLIB::TMatrix * A_ptr,
                  HLIB::TClusterTree * row_ct_ptr,
                  HLIB::TClusterTree * col_ct_ptr,
                  VectorXd x)
{
    // y = A * x
    std::unique_ptr<HLIB::TVector> y_hlib = A_ptr->row_vector();
    std::unique_ptr<HLIB::TVector> x_hlib = A_ptr->col_vector();

//    int n = y_hlib->size();
//    int m = x_hlib->size();
    int n = x.size();
    int m = x.size();

    for ( size_t  i = 0; i < m; i++ )
        x_hlib->set_entry( i, x(i) );

    col_ct_ptr->perm_e2i()->permute( x_hlib.get() );

    A_ptr->apply(x_hlib.get(), y_hlib.get());

    row_ct_ptr->perm_i2e()->permute( y_hlib.get() );

    VectorXd y(n);
    for ( size_t  i = 0; i < n; i++ )
        y(i) = y_hlib->entry( i );

    return y;
}

VectorXd h_factorized_inverse_matvec(HLIB::TFacInvMatrix * inv_A_ptr,
                                           HLIB::TClusterTree * row_ct_ptr,
                                           HLIB::TClusterTree * col_ct_ptr,
                                           VectorXd x)
{
    // y = inv_A * x
    std::unique_ptr<HLIB::TVector> y_hlib = inv_A_ptr->matrix()->row_vector();
    std::unique_ptr<HLIB::TVector> x_hlib = inv_A_ptr->matrix()->col_vector();

//    int n = y_hlib->size();
//    int m = x_hlib->size();
    int n = x.size();
    int m = x.size();

    for ( size_t  i = 0; i < m; i++ )
        x_hlib->set_entry( i, x(i) );

    col_ct_ptr->perm_e2i()->permute( x_hlib.get() );

    inv_A_ptr->apply(x_hlib.get(), y_hlib.get());

    row_ct_ptr->perm_i2e()->permute( y_hlib.get() );

    VectorXd y(n);
    for ( size_t  i = 0; i < n; i++ )
        y(i) = y_hlib->entry( i );

    return y;
}

std::shared_ptr<HLIB::TFacInvMatrix> factorize_inv_with_progress_bar(HLIB::TMatrix * A_ptr, TTruncAcc acc)
{
    double rtol = acc.rel_eps();
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;

    std::cout << std::endl << "━━ LU factorisation ( rtol = " << rtol << " )" << std::endl;

    timer.start();

    std::unique_ptr<HLIB::TFacInvMatrix> A_inv = factorise_inv( A_ptr, acc, & progress );

    timer.pause();
    std::cout << "    done in " << timer << std::endl;

    std::cout << "    size of LU factor = " << Mem::to_string( A_ptr->byte_size() ) << std::endl;

    return A_inv;
}

std::unique_ptr< HLIB::TFacInvMatrix > hmatrix_factorized_inverse_destructive(HLIB::TMatrix * A_ptr, double tol)
{
    TTruncAcc                 acc( tol, 0.0 );
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;

    std::cout << std::endl << "━━ LU factorisation ( tol = " << tol << " )" << std::endl;

    timer.start();

    std::unique_ptr<HLIB::TFacInvMatrix> A_inv = factorise_inv( A_ptr, acc, & progress );

    timer.pause();
    std::cout << "    done in " << timer << std::endl;

    std::cout << "    size of LU factor = " << Mem::to_string( A_ptr->byte_size() ) << std::endl;

    return A_inv;
}


void hmatrix_add_overwrites_second (const HLIB::TMatrix * A, HLIB::TMatrix* B, double tol)
{
    TTruncAcc                 acc( tol, 0.0 );
    add(1.0, A, 1.0, B, acc);
}

template < typename T_value >
void multiply_without_progress_bar(const T_value  	    alpha,
		                           const matop_t  	    op_A,
                                   const TMatrix *      A,
                                   const matop_t  	    op_B,
                                   const TMatrix *      B,
                                   const T_value  	    beta,
                                   TMatrix *  	        C,
                                   const TTruncAcc &  	acc)
{
    multiply(alpha, op_A, A, op_B, B, beta, C, acc);
}

template < typename T_value >
void multiply_with_progress_bar(const T_value  	    alpha,
		                        const matop_t  	    op_A,
                                const TMatrix *      A,
                                const matop_t  	    op_B,
                                const TMatrix *      B,
                                const T_value  	    beta,
                                TMatrix *  	        C,
                                const TTruncAcc &  	acc)
{
//    TTruncAcc              acc2( 1e-6 );
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;

    std::cout << std::endl << "━━ H-matrix multiplication C=A*B " << std::endl;

    timer.start();

    multiply(alpha, op_A, A, op_B, B, beta, C, acc, & progress);

    timer.pause();
    std::cout << "    done in " << timer << std::endl;

    std::cout << "    size of C = " << Mem::to_string( C->byte_size() ) << std::endl;
}





PYBIND11_MODULE(hlibpro_bindings, m) {
    m.doc() = "hlibpro wrapper plus product convolution hmatrix stuff";

    // -----------------------------------------------
    // --------      H-matrix Bindings        --------
    // -----------------------------------------------

    py::class_<HLIB::TProgressBar>(m, "TProgressBar");
    py::class_<HLIB::TFacInvMatrix>(m, "TFacInvMatrix");
    py::class_<HLIB::TBlockCluster>(m, "TBlockCluster");

    py::class_<HLIB::TBlockMatrix>(m, "TBlockMatrix")
        .def(py::init<const TBlockCluster *>(), py::arg("bct")=nullptr);

    py::enum_<HLIB::matop_t>(m, "matop_t")
        .value("MATOP_NORM", HLIB::matop_t::MATOP_NORM)
        .value("apply_normal", HLIB::matop_t::apply_normal)
        .value("MATOP_TRANS", HLIB::matop_t::MATOP_TRANS)
        .value("apply_trans", HLIB::matop_t::apply_trans)
        .value("apply_transposed", HLIB::matop_t::apply_transposed)
        .value("MATOP_ADJ", HLIB::matop_t::MATOP_ADJ)
        .value("MATOP_CONJTRANS", HLIB::matop_t::MATOP_CONJTRANS)
        .value("apply_adj", HLIB::matop_t::apply_adj)
        .value("apply_adjoint", HLIB::matop_t::apply_adjoint)
        .value("apply_conjtrans", HLIB::matop_t::apply_conjtrans)
        .export_values();

    py::class_<HLIB::TTruncAcc>(m, "TTruncAcc")
        .def("max_rank", &HLIB::TTruncAcc::max_rank)
        .def("has_max_rank", &HLIB::TTruncAcc::has_max_rank)
        .def("rel_eps", &HLIB::TTruncAcc::rel_eps)
        .def("abs_eps", &HLIB::TTruncAcc::abs_eps)
        .def(py::init<>())
//        .def(py::init<const int, double>(), py::arg("k"), py::arg("absolute_eps")=CFG::Arith::abs_eps)
        .def(py::init<const double, double>(), py::arg("relative_eps"), py::arg("absolute_eps")=CFG::Arith::abs_eps);

    py::class_<HLIB::TVector>(m, "TVector");

    py::class_<HLIB::TMatrix, std::shared_ptr<HLIB::TMatrix>>(m, "TMatrix")
//    py::class_<HLIB::TMatrix, std::unique_ptr<HLIB::TMatrix>>(m, "TMatrix")
//    py::class_<HLIB::TMatrix>(m, "TMatrix")
        .def("id", &HLIB::TMatrix::id)
        .def("rows", &HLIB::TMatrix::rows)
        .def("cols", &HLIB::TMatrix::cols)
        .def("is_nonsym", &HLIB::TMatrix::is_nonsym)
        .def("is_symmetric", &HLIB::TMatrix::is_symmetric)
        .def("is_hermitian", &HLIB::TMatrix::is_hermitian)
        .def("set_nonsym", &HLIB::TMatrix::set_nonsym)
        .def("is_real", &HLIB::TMatrix::is_real)
        .def("is_complex", &HLIB::TMatrix::is_complex)
        .def("to_real", &HLIB::TMatrix::to_real)
        .def("to_complex", &HLIB::TMatrix::to_complex)
        .def("add_update", &HLIB::TMatrix::add_update)
        .def("entry", &HLIB::TMatrix::entry)
        .def("apply", &HLIB::TMatrix::apply, py::arg("x"), py::arg("y"), py::arg("op")=apply_normal)
//        .def("apply_add", static_cast<void
//                                      (HLIB::TMatrix::*)(const real_t,
//                                                         const HLIB::TVector *,
//                                                         HLIB::TVector *,
//                                                         const matop_t) const>(&HLIB::TMatrix::apply_add),
//                                      py::arg("alpha"), py::arg("x"), py::arg("y"), py::arg("op")=apply_normal)
        .def("set_symmetric", &HLIB::TMatrix::set_symmetric)
        .def("set_hermitian", &HLIB::TMatrix::set_hermitian)
        .def("domain_dim", &HLIB::TMatrix::domain_dim)
        .def("range_dim", &HLIB::TMatrix::range_dim)
        .def("domain_vector", &HLIB::TMatrix::domain_vector)
        .def("range_vector", &HLIB::TMatrix::range_vector)
        .def("transpose", &HLIB::TMatrix::transpose)
        .def("conjugate", &HLIB::TMatrix::conjugate)
        .def("add", &HLIB::TMatrix::add)
        .def("scale", &HLIB::TMatrix::scale)
        .def("truncate", &HLIB::TMatrix::truncate)
        .def("mul_vec", &HLIB::TMatrix::mul_vec, py::arg("alpha"), py::arg("x"),
                                                 py::arg("beta"), py::arg("y"), py::arg("op")=MATOP_NORM)
//        .def("mul_right", &HLIB::TMatrix::mul_right) // not implemented apparently
//        .def("mul_left", &HLIB::TMatrix::mul_left) // not implemented apparently
        .def("check_data", &HLIB::TMatrix::check_data)
        .def("byte_size", &HLIB::TMatrix::byte_size)
        .def("print", &HLIB::TMatrix::print)
//        .def("copy", &HLIB::TMatrix::copy)
        .def("copy", static_cast<std::unique_ptr<TMatrix> (HLIB::TMatrix::*)() const>(&HLIB::TMatrix::copy)) //Won't compile without this static_cast... Why??
//        .def("copy", static_cast<std::shared_ptr<TMatrix> (HLIB::TMatrix::*)() const>(&HLIB::TMatrix::copy))
        .def("copy_struct", &HLIB::TMatrix::copy_struct)
        .def("create", &HLIB::TMatrix::create)
        .def("cluster", &HLIB::TMatrix::cluster)
        .def("row_vector", &HLIB::TMatrix::row_vector)
        .def("col_vector", &HLIB::TMatrix::col_vector);


    m.def("add", &add<real_t>);
    m.def("multiply_without_progress_bar", &multiply_without_progress_bar<real_t>);
    m.def("multiply_with_progress_bar", &multiply_with_progress_bar<real_t>);
    m.def("factorize_inv_with_progress_bar", &factorize_inv_with_progress_bar);

    py::class_<HLIB::TCoeffFn<real_t>>(m, "TCoeffFn<real_t>");

    py::class_<HLIB::TClusterTree, std::shared_ptr<HLIB::TClusterTree>>(m, "HLIB::TClusterTree")
        .def("perm_i2e", &HLIB::TClusterTree::perm_i2e)
        .def("perm_e2i", &HLIB::TClusterTree::perm_e2i)
        .def("nnodes", &HLIB::TClusterTree::nnodes)
        .def("depth", &HLIB::TClusterTree::depth)
        .def("byte_size", &HLIB::TClusterTree::byte_size);

    py::class_<HLIB::TBlockClusterTree, std::shared_ptr<HLIB::TBlockClusterTree>>(m, "HLIB::TBlockClusterTree")
        .def("row_ct", &HLIB::TBlockClusterTree::row_ct)
//        .def("row_ct", static_cast<std::unique_ptr<TClusterTree> (HLIB::TClusterTree::*)() const>(&HLIB::TBlockClusterTree::row_ct))
        .def("col_ct", &HLIB::TBlockClusterTree::col_ct)
        .def("nnodes", &HLIB::TBlockClusterTree::nnodes)
        .def("depth", &HLIB::TBlockClusterTree::depth)
        .def("nnodes", &HLIB::TBlockClusterTree::nnodes)
        .def("depth", &HLIB::TBlockClusterTree::depth)
        .def("compute_c_sp", &HLIB::TBlockClusterTree::compute_c_sp)
        .def("byte_size", &HLIB::TBlockClusterTree::byte_size);

    m.def("build_cluster_tree_from_dof_coords", &build_cluster_tree_from_dof_coords);
    m.def("build_block_cluster_tree", &build_block_cluster_tree);
    m.def("initialize_hlibpro", &initialize_hlibpro);
    m.def("visualize_cluster_tree", &visualize_cluster_tree);
    m.def("visualize_block_cluster_tree", &visualize_block_cluster_tree);
    m.def("build_hmatrix_from_coefffn", &build_hmatrix_from_coefffn);
    m.def("add_identity_to_hmatrix", &add_identity_to_hmatrix);
    m.def("visualize_hmatrix", &visualize_hmatrix);
    m.def("h_matvec", &h_matvec);
    m.def("hmatrix_factorized_inverse_destructive", &hmatrix_factorized_inverse_destructive);
    m.def("h_factorized_inverse_matvec", &h_factorized_inverse_matvec);
    m.def("build_hmatrix_from_sparse_matfile", &build_hmatrix_from_sparse_matfile);
    m.def("hmatrix_add_overwrites_second", &hmatrix_add_overwrites_second);

    // ----------------------------------------------------------
    // --------      Product Convolution Bindings        --------
    // ----------------------------------------------------------

    py::class_<ProductConvolutionCoeffFn, HLIB::TCoeffFn<real_t>>(m, "ProductConvolutionCoeffFn")
        .def(py::init<const ProductConvolutionMultipleBatches &, MatrixXd>());

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

    m.def("grid_interpolate", &grid_interpolate);
    m.def("grid_interpolate_vectorized", &grid_interpolate_vectorized);
    m.def("point_is_in_ellipsoid", &point_is_in_ellipsoid);
}

