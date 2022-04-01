import numpy as np
import dolfin as dl

from .product_convolution_kernel import ProductConvolutionKernel

from nalger_helper_functions import *
import hlibpro_python_wrapper as hpro


def make_hmatrix_from_kernel( Phi_pc : ProductConvolutionKernel,
                              hmatrix_tol=1e-5,
                              bct_admissibility_eta=2.0,
                              cluster_size_cutoff=50,
                              make_positive_definite=False,
                              use_lumped_mass_matrix=True ):
    print('Making row and column cluster trees')
    dof_coords_in = Phi_pc.V_in.tabulate_dof_coordinates()
    dof_coords_out = Phi_pc.V_out.tabulate_dof_coordinates()
    ct_in = hpro.build_cluster_tree_from_pointcloud(dof_coords_in, cluster_size_cutoff=cluster_size_cutoff)
    ct_out = hpro.build_cluster_tree_from_pointcloud(dof_coords_out, cluster_size_cutoff=cluster_size_cutoff)

    print('Making block cluster trees')
    bct_in = hpro.build_block_cluster_tree(ct_in, ct_in, admissibility_eta=bct_admissibility_eta)
    bct_out = hpro.build_block_cluster_tree(ct_out, ct_out, admissibility_eta=bct_admissibility_eta)
    bct_kernel = hpro.build_block_cluster_tree(ct_out, ct_in, admissibility_eta=bct_admissibility_eta)

    print('Building A kernel hmatrix')
    A_kernel_hmatrix = Phi_pc.build_hmatrix(bct_kernel,tol=hmatrix_tol)

    extras = {'A_kernel_hmatrix': A_kernel_hmatrix}

    print('Making input and output mass matrix hmatrices')
    if use_lumped_mass_matrix:
        mass_lumps_in_fenics = dl.Vector()
        Phi_pc.col_batches.ML_in.init_vector(mass_lumps_in_fenics,1)
        Phi_pc.col_batches.ML_in.get_diagonal(mass_lumps_in_fenics)
        mass_lumps_in = mass_lumps_in_fenics[:]

        mass_lumps_out_fenics = dl.Vector()
        Phi_pc.col_batches.ML_out.init_vector(mass_lumps_out_fenics,1)
        Phi_pc.col_batches.ML_out.get_diagonal(mass_lumps_out_fenics)
        mass_lumps_out = mass_lumps_out_fenics[:]

        print('Computing A_hmatrix = M_out_hmatrix * A_kernel_hmatrix * M_in_hmatrix')
        A_hmatrix = A_kernel_hmatrix.copy()
        A_hmatrix.mul_diag_left(mass_lumps_out)
        A_hmatrix.mul_diag_right(mass_lumps_in)

        extras['mass_lumps_in'] = mass_lumps_in
        extras['mass_lumps_out'] = mass_lumps_out
    else:
        M_in_fenics = Phi_pc.col_batches.M_in
        M_out_fenics = Phi_pc.col_batches.M_out
        M_in_scipy = csr_fenics2scipy(M_in_fenics)
        M_in_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_in_scipy, bct_in)
        M_out_scipy = csr_fenics2scipy(M_out_fenics)
        M_out_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_out_scipy, bct_out)

        print('Computing A_hmatrix = M_out_hmatrix * A_kernel_hmatrix * M_in_hmatrix')
        A_hmatrix = M_out_hmatrix * (A_kernel_hmatrix * M_in_hmatrix)

        extras['M_in_hmatrix'] = M_in_hmatrix
        extras['M_out_hmatrix'] = M_out_hmatrix

    if make_positive_definite:
        A_hmatrix_nonsym = A_hmatrix
        A_hmatrix = A_hmatrix.spd(overwrite=False, rtol_inv=hmatrix_tol)
        extras['A_hmatrix_nonsym'] = A_hmatrix_nonsym

    return A_hmatrix, extras







def product_convolution_hmatrix(V_in, V_out,
                                apply_A, apply_At,
                                num_batches,
                                tau=2.5,
                                max_candidate_points=None,
                                hmatrix_tol=1e-4,
                                bct_admissibility_eta=2.0,
                                cluster_size_cutoff=50,
                                make_positive_definite=False,
                                return_extras=False,
                                use_lumped_mass_matrix_for_impulse_response_moments=True):
    '''Builds hierarchical matrix representation of product convolution approximation of operator
        A: V_in -> V_out
    using only matrix-vector products with A and A^T.

    Parameters
    ----------
    V_in : fenics FunctionSpace
    V_out : fenics FunctionSpace
    apply_A : callable. Takes fenics Vector as input and returns fenics Vector as output
        apply_A(v) computes v -> A*v.
    apply_At : callable. Takes fenics Vector as input and returns fenics Vector as output
        apply_At(v) computes v -> A^T*v.
    num_batches : nonnegative int. Number of batches used in product convolution approximation
    tau : nonnegative float
        impulse response ellipsoid size parameter
        k'th ellipsoid = {x: (x-mu[k,:])^T Sigma[k,:,:]^{-1} (x-mu[k,:]) <= tau}
        support of k'th impulse response should be contained inside k'th ellipsoid
    max_candidate_points : nonnegative int. Maximum number of candidate points to consider when choosing sample points
    hmatrix_tol : nonnegative float. Accuracy tolerance for H-matrix
    bct_admissibility_eta : nonnegative float. Admissibility tolerance for block cluster tree.
        Only used of block_cluster_tree is not supplied
        A block of the matrix is admissible (low rank) if:
            distance(A, B) <= eta * min(diameter(A), diameter(B))
        where A is the cluster of points associated with the rows of the block, and
              B is the cluster of points associated with the columns of the block.
    cluster_size_cutoff : positive int. number of points below which clusters are not subdivided.
    use_boundary_extension : bool.
        if True (default), fill in convolution kernel missing values using neighboring convolution kernels
    make_positive_definite : bool. Default=False
        if True, modify hmatrix via rational approximation to make it symmetric positive definite
    return_extras : bool.
        If False (default), only return hmatrix. Otherwise, return other intermediate objects.

    Returns
    -------
    A_hmatrix : HMatrix. hierarchical matrix representation of product-convolution operator
    vol : fenics Function. Spatially varying impulse response volume
    mu : Vector-valued fenics Function. Spatially varying impulse response mean
    Sigma : Tensor-valued fenics Function. Spatially varying impulse response covariance
    point_batches : list of numpy arrays. point_batches[b].shape=(num_points_in_batch_b, spatial_dimension)
        Batches of sample points
    mu_batches : list of numpy arrays. mu_batches[b].shape=(num_points_in_batch_b, spatial_dimension)
        Impulse response means evaluated at the sample points
    Sigma_batches : list of numpy arrays. mu_batches[b].shape=(num_points_in_batch_b, spatial_dimension, spatial_dimension)
        Impulse response covariances evaluated at the sample points
    ww : list of fenics Functions, len(ww)=num_pts
        weighting functions
    ff_batches : list of fenics Functions, len(ff)=num_batches
        impulse response function batches
    WW : list of BoxFunctions. Weighting functions for each patch
    FF : list of BoxFunctions. Convolution kernels for each patch (with extension if done)
    initial_FF : list of BoxFunctions. Convolution kernels for each patch without extension

    '''
    if use_boundary_extension:
        kernel_fill_value=np.nan
    else:
        kernel_fill_value = 0.0

    print('Making mass matrices and solvers')
    M_in, solve_M_in = make_mass_matrix(V_in, make_solver=True)
    ML_in, solve_ML_in = make_mass_matrix(V_in, lumping='simple', make_solver=True)

    M_out, solve_M_out = make_mass_matrix(V_out, make_solver=True)
    ML_out, solve_ML_out = make_mass_matrix(V_out, lumping='simple', make_solver=True)

    if use_lumped_mass_matrix_for_impulse_response_moments:
        vol, mu, Sigma = impulse_response_moments(V_in, V_out, apply_At, solve_ML_in)
    else:
        vol, mu, Sigma = impulse_response_moments(V_in, V_out, apply_At, solve_M_in)

    point_batches, mu_batches, Sigma_batches = choose_sample_point_batches(num_batches, V_in, mu, Sigma, tau,
                                                                           max_candidate_points=max_candidate_points)

    pp = np.vstack(point_batches)
    all_mu = np.vstack(mu_batches)
    all_Sigma = np.vstack(Sigma_batches)
    batch_lengths = [pp_batch.shape[0] for pp_batch in point_batches]

    ff_batches = compute_impulse_response_batches(point_batches, V_in, V_out, apply_A, solve_M_in, solve_M_out)

    # ww = make_poisson_weighting_functions(V_in, pp)
    ww = make_rbf_weighting_functions(V_in, pp)

    WW, initial_FF = \
        build_product_convolution_patches_from_fenics_functions(ww, ff_batches, pp,
                                                                all_mu, all_Sigma, tau,
                                                                batch_lengths,
                                                                grid_density_multiplier=grid_density_multiplier,
                                                                w_support_rtol=w_support_rtol,
                                                                fill_value=kernel_fill_value)

    if use_boundary_extension:
        FF = compute_convolution_kernel_boundary_extensions(initial_FF, pp,
                                                            num_extension_kernels=num_extension_kernels)
    else:
        FF = initial_FF

    print('Making row and column cluster trees')
    dof_coords_in = V_in.tabulate_dof_coordinates()
    dof_coords_out = V_out.tabulate_dof_coordinates()
    ct_in = hpro.build_cluster_tree_from_pointcloud(dof_coords_in, cluster_size_cutoff=cluster_size_cutoff)
    ct_out = hpro.build_cluster_tree_from_pointcloud(dof_coords_out, cluster_size_cutoff=cluster_size_cutoff)

    print('Making block cluster trees')
    bct_in = hpro.build_block_cluster_tree(ct_in, ct_in, admissibility_eta=bct_admissibility_eta)
    bct_out = hpro.build_block_cluster_tree(ct_out, ct_out, admissibility_eta=bct_admissibility_eta)
    bct_kernel = hpro.build_block_cluster_tree(ct_out, ct_in, admissibility_eta=bct_admissibility_eta)

    print('Building A kernel hmatrix')
    A_kernel_hmatrix = build_product_convolution_hmatrix_from_patches(WW, FF, bct_kernel,
                                                                      dof_coords_out, dof_coords_in,
                                                                      tol=hmatrix_tol)

    print('Making input and output mass matrix hmatrices')
    M_in_scipy = csr_fenics2scipy(M_in)
    M_in_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_in_scipy, bct_in)
    M_out_scipy = csr_fenics2scipy(M_in)
    M_out_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_out_scipy, bct_out)

    print('Computing A_hmatrix = M_out_hmatrix * A_kernel_hmatrix * M_in_hmatrix')
    A_hmatrix = M_out_hmatrix * (A_kernel_hmatrix * M_in_hmatrix)
    # hpro.h_mul(A_kernel_hmatrix, M_out_hmatrix, alpha_A_B_hmatrix=A_kernel_hmatrix, rtol=hmatrix_tol)
    # hpro.h_mul(M_in_hmatrix, A_kernel_hmatrix, alpha_A_B_hmatrix=A_kernel_hmatrix, rtol=hmatrix_tol)
    # A_hmatrix = A_kernel_hmatrix

    extras = {'vol' : vol,
              'mu' : mu,
              'Sigma' : Sigma,
              'point_batches' : point_batches,
              'mu_batches' : mu_batches,
              'Sigma_batches' : Sigma_batches,
              'tau' : tau,
              'ww' : ww,
              'ff_batches' : ff_batches,
              'WW' : WW,
              'FF' : FF,
              'initial_FF' : initial_FF,
              'A_kernel_hmatrix' : A_kernel_hmatrix}

    if make_positive_definite:
        A_hmatrix_nonsym = A_hmatrix
        # A_hmatrix = hpro.rational_positive_definite_approximation_method1(A_hmatrix_nonsym, overwrite=False, rtol_inv=hmatrix_tol)
        A_hmatrix = A_hmatrix.spd(overwrite=False, rtol_inv=hmatrix_tol)
        extras['A_hmatrix_nonsym'] = A_hmatrix_nonsym

    if return_extras:
        return A_hmatrix, extras
    else:
        return A_hmatrix

