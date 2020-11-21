#include <cstdlib>
#include <iostream>

#include "hlib.hh"

using namespace std;
using namespace HLIB;

void print_sparse_matrix_info(HLIB::TSparseMatrix * S, string matrix_name)
{
    cout << "matrix " << matrix_name << " has dimension " << S->rows() << " x " << S->cols() << endl
         << "no of non-zeroes = " << S->n_non_zero() << endl
         << "matrix is " << ( S->is_complex() ? "complex" : "real" )
         << " valued" << endl
         << "format = ";
    if      ( S->is_nonsym()    ) cout << "non symmetric" << endl;
    else if ( S->is_symmetric() ) cout << "symmetric" << endl;
    else if ( S->is_hermitian() ) cout << "hermitian" << endl;
    cout << "size of sparse matrix = " << Mem::to_string( S->byte_size() ) << endl;
    cout << "|" << matrix_name << "|_F = " << norm_F( S ) << endl << endl;
}

void print_hmatrix_info(HLIB::TMatrix * A, HLIB::TClusterTree * ct, HLIB::TSparseMatrix * S, string matrix_name)
{
    cout << "size of H-matrix " << matrix_name << " = " << Mem::to_string( A->byte_size() ) << endl;
    cout << "|" << matrix_name << "|_F = " << norm_F( A ) << endl;
    {
        auto  PA = make_unique< TPermMatrix >( ct->perm_i2e(), A, ct->perm_e2i() );
        cout << matrix_name << " H-matrix error = " << diff_norm_2( S, PA.get() ) << endl << endl;
    }
}

std::unique_ptr<HLIB::TClusterTree> build_cluster_tree_from_dof_coords(HLIB::TVector * x_coords,
                                                                       HLIB::TVector * y_coords,
                                                                       const double nmin)
{
    size_t N = x_coords->size();
    size_t d = 2;

    vector< double * >  vertices( N );

    for ( size_t i = 0; i < N; i++ )
    {
        double * v    = new double[d];
        v[0] = x_coords->entry(i);
        v[1] = y_coords->entry(i);
        vertices[i] = v;
    }// for

    auto coord = make_unique< TCoordinate >( vertices, d );

//    TAutoBSPPartStrat  part_strat;
    TCardBSPPartStrat  part_strat;
    TBSPCTBuilder      ct_builder( & part_strat, nmin );
    std::unique_ptr<HLIB::TClusterTree>  ct = ct_builder.build( coord.get() );
    return ct;
}

int
main ( int argc, char ** argv )
{
    cout << (bool)argv[1] << endl;
    bool use_perturbed_matrix = (bool)argv[1];
    try
    {
        cout << "use_perturbed_matrix = " << use_perturbed_matrix << endl << endl;
        INIT();

        string M1_fname;
        string M2_fname;
        string M12_fname;

//        M1_fname = "test_matrix_1_perturbed.mat";
//        M2_fname = "test_matrix_2_perturbed.mat";
//        M12_fname = "test_matrix_12_perturbed.mat";
        if(use_perturbed_matrix)
        {
            M1_fname = "test_matrix_1_perturbed.mat";
            M2_fname = "test_matrix_2_perturbed.mat";
            M12_fname = "test_matrix_12_perturbed.mat";
        }
        else
        {
            M1_fname = "test_matrix_1.mat";
            M2_fname = "test_matrix_2.mat";
            M12_fname = "test_matrix_12.mat";
        }
        auto               M1 = read_matrix( M1_fname );
        auto               M2 = read_matrix( M2_fname );
        auto               M12 = read_matrix( M12_fname );

        auto               S1 = ptrcast( M1.get(), TSparseMatrix );
        auto               S2 = ptrcast( M2.get(), TSparseMatrix );
        auto               S12 = ptrcast( M12.get(), TSparseMatrix );

        print_sparse_matrix_info(S1, "S1");
        print_sparse_matrix_info(S2, "S2");

        auto x_coords = read_vector( "test_x_coords.mat" );
        auto y_coords = read_vector( "test_y_coords.mat" );

        // Geometric clustering
        auto ct = build_cluster_tree_from_dof_coords(x_coords.get(), y_coords.get(), 60);
        TStdGeomAdmCond    adm_cond( 2.0 );

//        // Algebraic clustering
//        TBFSAlgPartStrat   part_strat;
//        TAlgCTBuilder      ct_builder( & part_strat );
//        TAlgNDCTBuilder    nd_ct_builder( & ct_builder );
//        auto               ct = nd_ct_builder.build( S1 );
//        TWeakAlgAdmCond    adm_cond( S1, ct->perm_i2e() );

        TBCBuilder         bct_builder;
        auto               bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );

        cout << "sparsity constant = " << bct->compute_c_sp() << endl << endl;

        TSparseMBuilder    h_builder1( S1, ct->perm_i2e(), ct->perm_e2i() );
        TSparseMBuilder    h_builder2( S2, ct->perm_i2e(), ct->perm_e2i() );
//        TTruncAcc          acc( real(0.0) );
        TTruncAcc          acc( 1e-12, 1e-16 );
        auto               A1 = h_builder1.build( bct.get(), acc );
        auto               A2 = h_builder2.build( bct.get(), acc );

        print_hmatrix_info(A1.get(), ct.get(), S1, "A1");
        print_hmatrix_info(A2.get(), ct.get(), S2, "A2");

        auto               x = read_vector( "test_vector.mat" );
        auto               y = S1->col_vector();
        auto               z = S1->col_vector();
        auto               z2 = S1->col_vector();

        // y = A2*x, z = A1*y
        A2->apply(x.get(), y.get());
        A1->apply(y.get(), z.get());

        auto A1_A2 = A1->copy();
        multiply(1.0, apply_normal, A1.get(), apply_normal, A2.get(), 0.0, A1_A2.get(), acc); // Error happens here
//        multiply(1.0, A1.get(), A2.get(), 0.0, A1_A2.get(), acc); // Error happens here

        print_hmatrix_info(A1_A2.get(), ct.get(), S12, "A1_A2");

        A1_A2->apply(x.get(), z2.get());

        auto z_err = z->copy();
        z_err->axpy(-1.0, z2.get());

        cout << "A1*A2*z error = " << z_err->norm2() << endl;

        DONE();

    }
    catch ( Error & e )
    {
        cout << e.to_string() << endl;
    }

    return 0;
}
