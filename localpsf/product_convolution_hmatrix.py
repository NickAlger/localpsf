import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from nalger_helper_functions import *
import hlibpro_python_wrapper as hpro

from .ellipsoid import points_which_are_not_in_ellipsoid_numba
from .impulse_response_moments import impulse_response_moments
from .sample_point_batches import choose_sample_point_batches
from .impulse_response_batches import compute_impulse_response_batches
from .poisson_weighting_functions import make_poisson_weighting_functions
from .rbf_weighting_functions import make_rbf_weighting_functions
from .visualization import visualize_impulse_response_batch, visualize_weighting_function


def polyharmonic_spline(rr, k):
    ff = np.zeros(rr.shape)
    if np.mod(k,2) == 0:
        ff[rr > 0] = np.power(rr[rr>0], k) * np.log(rr[rr>0])
    else:
        ff = np.power(rr, k)
    return ff


def eval_polyharmonic_spline_kernels_at_points(xx, pp, k):
    # xx: evaluation points. shape=(M,d)
    # pp: kernel centers. shape=(N,d)
    # k: polyharmonic spline parameter. int
    # ff: rbfs evaluated at points. shape=(N,M)
    rr = np.linalg.norm(xx[None,:,:] - pp[:,None,:], axis=2)
    return polyharmonic_spline(rr, k)


def eval_fenics_function_at_points(f, pp):
    num_pts = pp.shape[0]
    ff = np.zeros(num_pts)
    for ii in range(num_pts):
        ff[ii] = f(pp[ii, :])
    return ff


class ProductConvolutionKernelRBF:
    def __init__(me, impulse_response_batches, sample_points_batches, mu_batches, Sigma_batches, tau,
                 vol, mu, Sigma, V_in, V_out, rbf_kernel_parameter=2):
        me.impulse_response_batches = impulse_response_batches
        me.sample_points_batches = sample_points_batches
        me.mu_batches = mu_batches
        me.Sigma_batches = Sigma_batches
        me.tau = tau
        me.vol = vol
        me.mu = mu
        me.Sigma = Sigma
        me.V_in = V_in
        me.V_out = V_out
        me.rbf_kernel_parameter = rbf_kernel_parameter

        me.sample_points = np.concatenate(sample_points_batches, axis=0)

        me.mesh_out = me.V_out.mesh()

        me.dof_coords_in = me.V_in.tabulate_dof_coordinates()
        me.dof_coords_out = me.V_out.tabulate_dof_coordinates()

        me.num_batches = len(me.impulse_response_batches)
        me.num_sample_points = me.sample_points.shape[0]

        for f in me.impulse_response_batches:
            f.set_allow_extrapolation(True)

        all_points = np.vstack(sample_points_batches)
        all_mu = np.vstack(mu_batches)
        all_Sigma = np.vstack(Sigma_batches)

        me.vertex2dof_out = dl.vertex_to_dof_map(V_out)

        me.all_points_list = [all_points[ii,:].copy() for ii in range(me.num_sample_points)]
        me.all_mu_list = [all_mu[ii, :].copy() for ii in range(me.num_sample_points)]
        me.all_Sigma_list = [all_Sigma[ii, :, :].copy() for ii in range(me.num_sample_points)]
        me.impulse_response_batches_vectors = [IB.vector()[me.vertex2dof_out].copy() for IB in impulse_response_batches]
        me.batch_lengths = [point_batch.shape[0] for point_batch in sample_points_batches]

        me.mesh_vertices = np.array(me.mesh_out.coordinates().T, order='F')
        me.mesh_cells = np.array(me.mesh_out.cells().T, order='F')

        me.integral_kernel = hpro.hpro_cpp.ProductConvolutionKernelRBF(me.all_points_list,
                                                                       me.all_mu_list,
                                                                       me.all_Sigma_list,
                                                                       me.tau,
                                                                       me.impulse_response_batches_vectors,
                                                                       me.batch_lengths,
                                                                       me.mesh_vertices,
                                                                       me.mesh_cells)

    def __call__(me, y, x):
        return me.integral_kernel.eval_integral_kernel(y, x)


class ProductConvolutionKernel:
    def __init__(me, impulse_response_batches, sample_points_batches, mu_batches, Sigma_batches, tau,
                 vol, mu, Sigma, V_in, V_out, rbf_kernel_parameter=2):
        me.impulse_response_batches = impulse_response_batches
        me.sample_points_batches = sample_points_batches
        me.mu_batches = mu_batches
        me.Sigma_batches = Sigma_batches
        me.tau = tau
        me.vol = vol
        me.mu = mu
        me.Sigma = Sigma
        me.V_in = V_in
        me.V_out = V_out
        me.rbf_kernel_parameter = rbf_kernel_parameter

        me.sample_points = np.concatenate(sample_points_batches, axis=0)

        me.mesh_out = me.V_out.mesh()

        me.dof_coords_in = me.V_in.tabulate_dof_coordinates()
        me.dof_coords_out = me.V_out.tabulate_dof_coordinates()

        me.interpolation_matrix = eval_polyharmonic_spline_kernels_at_points(me.sample_points,
                                                                             me.sample_points,
                                                                             me.rbf_kernel_parameter)

        me.solve_interpolation_matrix = factorized(me.interpolation_matrix)

        me.num_batches = len(me.impulse_response_batches)
        me.num_sample_points = me.sample_points.shape[0]

        for f in me.impulse_response_batches:
            f.set_allow_extrapolation(True)

    def eval_integral_kernel_at_points(me, yy, xx, display_progress=False):
        # yy.shape = (N, d)
        # xx.shape = (M, d)
        # QQ[ii, jj] = Phi(yy[ii,:], xx[jj,:]), qq.shape = (N,M)
        N, d = yy.shape
        M, d = xx.shape
        W = me.eval_all_weighting_functions_at_points(xx) # shape=(num_sample_points, M)
        QQ = np.zeros((N, M))
        for jj in tqdm(range(M), disable=(not display_progress)):
            w = W[:,jj].reshape(-1) # shape=(num_sample_points,)
            zz = yy - xx[jj,:]
            cc = me.eval_all_convolution_kernels_at_points(zz) # shape=(num_sample_points, N)
            QQ[:,jj] = np.dot(w, cc)
        return QQ

    def build_dense_integral_kernel(me, V_in=None, V_out=None):
        if V_in is None:
            V_in = me.V_in
        if V_out is None:
            V_out = me.V_out

        xx = V_in.tabulate_dof_coordinates()
        yy = V_out.tabulate_dof_coordinates()

        print('Building dense integral kernel column-by-column')
        Phi = me.eval_integral_kernel_at_points(yy, xx, display_progress=True)
        print('done.')

        return Phi

    def eval_all_weighting_functions_at_points(me, xx):
        # xx = points to evaluate weighting functions wk at. shape=(M,d)
        # W[i,j] = w_i(x_j). W.shape = (num_weighting_functions, M)
        if len(xx.shape) == 1:
            xx = xx.reshape((1,-1))
        M, d = xx.shape

        B = eval_polyharmonic_spline_kernels_at_points(xx, me.sample_points, me.rbf_kernel_parameter)
        W = me.solve_interpolation_matrix(B)

        if len(xx.shape) == 1:
            W = W.reshape(-1)
        return W

    def eval_one_batch_of_convolution_kernels_at_points(me, zz, batch_ind, reflect=True):
        # zz = points to evaluate convolution kernels at. shape = (num_z, d)
        num_z, d = zz.shape
        # if reflect:
        #     zz = reflect_exterior_points_across_boundary(zz, me.mesh_out)

        pp = me.sample_points_batches[batch_ind]
        f = me.impulse_response_batches[batch_ind]
        mu_batch = me.mu_batches[batch_ind]
        Sigma_batch = me.Sigma_batches[batch_ind]

        batch_size = pp.shape[0]
        cc = np.zeros((batch_size, num_z))
        for kk in range(batch_size):
            mu = mu_batch[kk, :]
            Sigma = Sigma_batch[kk, :, :]
            shifted_zz = zz + pp[kk,:]
            if reflect:
                shifted_zz = reflect_exterior_points_across_boundary(shifted_zz, me.mesh_out).copy()

            not_in_ellipsoid = points_which_are_not_in_ellipsoid_numba(Sigma, mu, shifted_zz, me.tau)
            in_ellipsoid = np.logical_not(not_in_ellipsoid)
            if np.any(in_ellipsoid):
                cc[kk, in_ellipsoid] = \
                    eval_fenics_function_at_points(f, shifted_zz[in_ellipsoid, :]).reshape(-1)

        return cc

    def eval_all_convolution_kernels_at_points(me, zz):
        # zz = points to evaluate convolution kernels at. shape = (num_z, d)
        # cc.shape = (num_sample_points, num_z)
        num_eval_points, d = zz.shape
        # zzR = reflect_exterior_points_across_boundary(zz, me.mesh_out).copy()

        cc_list = list()
        for b in range(me.num_batches):
            cc_b = me.eval_one_batch_of_convolution_kernels_at_points(zz, b)
            cc_list.append(cc_b)

        cc = np.concatenate(cc_list, axis=0)
        return cc

    def visualize_impulse_response_batch(me, k):
        f = me.impulse_response_batches[k]
        pp = me.sample_points_batches[k]
        mu_batch = me.mu_batches[k]
        Sigma_batch = me.Sigma_batches[k]

        plt.figure()
        cm = dl.plot(f)
        plt.colorbar(cm)

        plt.scatter(pp[:, 0], pp[:, 1], c='k', s=2)

        for k in range(mu_batch.shape[0]):
            plot_ellipse(mu_batch[k, :], Sigma_batch[k, :, :],
                         n_std_tau=me.tau, facecolor='none', edgecolor='k',
                         linewidth=1)

    def visualize_weighting_function(me, k, V=None): # not efficient
        if V is None:
            V = me.V_in
        dof_coords = V.tabulate_dof_coordinates()
        WW = me.eval_all_weighting_functions_at_points(dof_coords)

        w = dl.Function(V)
        w.vector()[:] = WW[k,:].copy()

        plt.figure()
        cm = dl.plot(w)
        plt.colorbar(cm)
        plt.title('Weighting function ' + str(k))
        plt.scatter(me.sample_points[:, 0], me.sample_points[:, 1], c='k', s=2)
        plt.scatter(me.sample_points[k, 0], me.sample_points[k, 1], c='r', s=2)


def build_product_convolution_kernel(V_in, V_out,
                                     apply_A, apply_At,
                                     num_batches,
                                     tau=2.5,
                                     max_candidate_points=None,
                                     use_lumped_mass_matrix_for_impulse_response_moments=True,
                                     rbf_kernel_parameter=2):
    print('Making mass matrices and solvers')
    M_in, solve_M_in = make_mass_matrix(V_in, make_solver=True)
    ML_in, solve_ML_in = make_mass_matrix(V_in, lumping='simple', make_solver=True)

    M_out, solve_M_out = make_mass_matrix(V_out, make_solver=True)
    ML_out, solve_ML_out = make_mass_matrix(V_out, lumping='simple', make_solver=True)

    print('Computing impulse response moments')
    if use_lumped_mass_matrix_for_impulse_response_moments:
        vol, mu, Sigma = impulse_response_moments(V_in, V_out, apply_At, solve_ML_in)
    else:
        vol, mu, Sigma = impulse_response_moments(V_in, V_out, apply_At, solve_M_in)

    print('Choosing sample point batches')
    point_batches, mu_batches, Sigma_batches = choose_sample_point_batches(num_batches, V_in, mu, Sigma, tau,
                                                                           max_candidate_points=max_candidate_points)

    print('Computing impulse response batches')
    ff_batches = compute_impulse_response_batches(point_batches, V_in, V_out, apply_A, solve_M_in, solve_M_out)

    print('Forming ProductConvolutionKernel')
    # PCK = ProductConvolutionKernel(ff_batches, point_batches, mu_batches, Sigma_batches, tau,
    #                                vol, mu, Sigma, V_in, V_out,
    #                                rbf_kernel_parameter=rbf_kernel_parameter)

    PCK = ProductConvolutionKernelRBF(ff_batches, point_batches, mu_batches, Sigma_batches, tau,
                                      vol, mu, Sigma, V_in, V_out,
                                      rbf_kernel_parameter=rbf_kernel_parameter)

    return PCK


# class ProductConvolutionHmatrix:
#     def __init__(me, V_in, V_out,
#                  apply_A, apply_At,
#                  num_batches,
#                  tau=2.5,
#                  max_candidate_points=None,
#                  hmatrix_tol=1e-4,
#                  bct_admissibility_eta=2.0,
#                  cluster_size_cutoff=50,
#                  make_positive_definite=False,
#                  use_lumped_mass_matrix_for_impulse_response_moments=True):
#         me.V_in = V_in
#         me.V_out = V_out
#         me.apply_A = apply_A
#         me.apply_At = apply_At
#         me.num_batches = num_batches
#         me.tau = tau
#         me.max_candidate_points = max_candidate_points
#         me.hmatrix_tol = hmatrix_tol
#         me.bct_admissibility_eta = bct_admissibility_eta
#         me.cluster_size_cutoff = cluster_size_cutoff
#
#
#         if use_boundary_extension:
#             kernel_fill_value = np.nan
#         else:
#             kernel_fill_value = 0.0
#
#         print('Making mass matrices and solvers')
#         M_in, solve_M_in = make_mass_matrix(V_in, make_solver=True)
#         ML_in, solve_ML_in = make_mass_matrix(V_in, lumped=True, make_solver=True)
#
#         M_out, solve_M_out = make_mass_matrix(V_out, make_solver=True)
#         ML_out, solve_ML_out = make_mass_matrix(V_out, lumped=True, make_solver=True)
#
#         if use_lumped_mass_matrix_for_impulse_response_moments:
#             vol, mu, Sigma = impulse_response_moments(V_in, V_out, apply_At, solve_ML_in)
#         else:
#             vol, mu, Sigma = impulse_response_moments(V_in, V_out, apply_At, solve_M_in)
#
#         point_batches, mu_batches, Sigma_batches = choose_sample_point_batches(num_batches, V_in, mu, Sigma, tau,
#                                                                                max_candidate_points=max_candidate_points)
#
#         pp = np.vstack(point_batches)
#         all_mu = np.vstack(mu_batches)
#         all_Sigma = np.vstack(Sigma_batches)
#         batch_lengths = [pp_batch.shape[0] for pp_batch in point_batches]
#
#         ff_batches = compute_impulse_response_batches(point_batches, V_in, V_out, apply_A, solve_M_in, solve_M_out)
#
#         # ww = make_poisson_weighting_functions(V_in, pp)
#         ww = make_rbf_weighting_functions(V_in, pp)
#
#         WW, initial_FF = \
#             build_product_convolution_patches_from_fenics_functions(ww, ff_batches, pp,
#                                                                     all_mu, all_Sigma, tau,
#                                                                     batch_lengths,
#                                                                     grid_density_multiplier=grid_density_multiplier,
#                                                                     w_support_rtol=w_support_rtol,
#                                                                     fill_value=kernel_fill_value)
#
#         if use_boundary_extension:
#             FF = compute_convolution_kernel_boundary_extensions(initial_FF, pp,
#                                                                 num_extension_kernels=num_extension_kernels)
#         else:
#             FF = initial_FF
#
#         print('Making row and column cluster trees')
#         dof_coords_in = V_in.tabulate_dof_coordinates()
#         dof_coords_out = V_out.tabulate_dof_coordinates()
#         ct_in = hpro.build_cluster_tree_from_pointcloud(dof_coords_in, cluster_size_cutoff=cluster_size_cutoff)
#         ct_out = hpro.build_cluster_tree_from_pointcloud(dof_coords_out, cluster_size_cutoff=cluster_size_cutoff)
#
#         print('Making block cluster trees')
#         bct_in = hpro.build_block_cluster_tree(ct_in, ct_in, admissibility_eta=bct_admissibility_eta)
#         bct_out = hpro.build_block_cluster_tree(ct_out, ct_out, admissibility_eta=bct_admissibility_eta)
#         bct_kernel = hpro.build_block_cluster_tree(ct_out, ct_in, admissibility_eta=bct_admissibility_eta)
#
#         print('Building A kernel hmatrix')
#         A_kernel_hmatrix = build_product_convolution_hmatrix_from_patches(WW, FF, bct_kernel,
#                                                                           dof_coords_out, dof_coords_in,
#                                                                           tol=hmatrix_tol)
#
#         print('Making input and output mass matrix hmatrices')
#         M_in_scipy = csr_fenics2scipy(M_in)
#         M_in_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_in_scipy, bct_in)
#         M_out_scipy = csr_fenics2scipy(M_in)
#         M_out_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_out_scipy, bct_out)
#
#         print('Computing A_hmatrix = M_out_hmatrix * A_kernel_hmatrix * M_in_hmatrix')
#         A_hmatrix = M_out_hmatrix * (A_kernel_hmatrix * M_in_hmatrix)
#         # hpro.h_mul(A_kernel_hmatrix, M_out_hmatrix, alpha_A_B_hmatrix=A_kernel_hmatrix, rtol=hmatrix_tol)
#         # hpro.h_mul(M_in_hmatrix, A_kernel_hmatrix, alpha_A_B_hmatrix=A_kernel_hmatrix, rtol=hmatrix_tol)
#         # A_hmatrix = A_kernel_hmatrix
#
#         extras = {'vol': vol,
#                   'mu': mu,
#                   'Sigma': Sigma,
#                   'point_batches': point_batches,
#                   'mu_batches': mu_batches,
#                   'Sigma_batches': Sigma_batches,
#                   'tau': tau,
#                   'ww': ww,
#                   'ff_batches': ff_batches,
#                   'WW': WW,
#                   'FF': FF,
#                   'initial_FF': initial_FF,
#                   'A_kernel_hmatrix': A_kernel_hmatrix}
#
#         if make_positive_definite:
#             A_hmatrix_nonsym = A_hmatrix
#             # A_hmatrix = hpro.rational_positive_definite_approximation_method1(A_hmatrix_nonsym, overwrite=False, rtol_inv=hmatrix_tol)
#             A_hmatrix = A_hmatrix.spd(overwrite=False, rtol_inv=hmatrix_tol)
#             extras['A_hmatrix_nonsym'] = A_hmatrix_nonsym
#
#         if return_extras:
#             return A_hmatrix, extras
#         else:
#             return A_hmatrix



def product_convolution_hmatrix(V_in, V_out,
                                apply_A, apply_At,
                                num_batches,
                                tau=2.5,
                                max_candidate_points=None,
                                grid_density_multiplier=0.25,
                                w_support_rtol=2e-2,
                                num_extension_kernels=8,
                                hmatrix_tol=1e-4,
                                bct_admissibility_eta=2.0,
                                cluster_size_cutoff=50,
                                use_boundary_extension=False,
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
    grid_density_multiplier : nonnegative float
        grid density scaling factor for converting functions on irregular grids to regular grid patches
        If grid_density_multiplier=1.0, then the spacing between gridpoints
        in a given BoxFunction equals the minimum distance between mesh nodes in the box.
        If grid_density_multiplier=0.5, then the spacing between gridpoints
        in a given BoxFunction equals one half the minimum distance between mesh nodes in the box.
    w_support_rtol : nonnegative float
        truncation tolerance determining how big the support boxes are for the weighting functions.
        If w_support_rtol=2e-2, then values of w outside the box will be no more than 2% of the maximum
        value of w.
    num_extension_kernels : positive int
        number of initial impulse responses used to fill in the missing information for each impulse response
        E.g., if num_extension_kernels=8, then a given filled in impulse response will contain information
        from itself, and the 7 other nearest impulse responses.
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


def build_product_convolution_hmatrix_from_patches(WW, FF, block_cluster_tree, row_dof_coords, col_dof_coords, tol=1e-6):
    '''Build H-Matrix for product-convolution operator based on rectangular patches

    Parameters
    ----------
    WW : list of BoxFunctions. len=num_patches. W0eighting functions
    FF : list of BoxFunctions. len=num_patches. Convolution kernels
    row_dof_coords : numpy array. shape=(num_points, spatial_dimension). spatial locations of row degrees of freedom
    col_dof_coords : numpy array. shape=(num_points, spatial_dimension). spatial locations of column degrees of freedom
    block_cluster_tree : BlockClusterTree. Block cluster tree for H-matrix. None (Default): build block cluster tree from scratch
    tol : nonnegative float. Accuracy tolerance for H-matrix
    admissibility_eta : nonnegative float. Admissibility tolerance for block cluster tree.
        Only used of block_cluster_tree is not supplied
        A block of the matrix is admissible (low rank) if:
            distance(A, B) <= eta * min(diameter(A), diameter(B))
        where A is the cluster of points associated with the rows of the block, and
              B is the cluster of points associated with the columns of the block.
    cluster_size_cutoff : positive int. number of points below which clusters are not subdivided.

    Returns
    -------
    A_hmatrix : HMatrix

    '''
    WW_mins = [W.min for W in WW]
    WW_maxes = [W.max for W in WW]
    WW_arrays = [W.array for W in WW]

    FF_mins = [F.min for F in FF]
    FF_maxes = [F.max for F in FF]
    FF_arrays = [F.array for F in FF]

    print('Building product convolution hmatrix from patches')
    A_hmatrix = hpro.build_product_convolution_hmatrix_2d(WW_mins, WW_maxes, WW_arrays,
                                                          FF_mins, FF_maxes, FF_arrays,
                                                          row_dof_coords, col_dof_coords,
                                                          block_cluster_tree, tol=tol)

    return A_hmatrix


def build_product_convolution_patches_from_fenics_functions(ww, ff_batches, pp,
                                                            all_mu, all_Sigma, tau,
                                                            batch_lengths,
                                                            grid_density_multiplier=1.0,
                                                            w_support_rtol=2e-2,
                                                            fill_value=np.nan):
    '''Constructs weighting function and convolution kernel BoxFunctions
    by sampling fenics weighting functions and impulse response batches on regular grids

    Parameters
    ----------
    ww : list of fenics Functions, len(ww)=num_pts
        weighting functions
    ff_batches : list of fenics Functions, len(ff)=num_batches
        impulse response function batches
    pp : numpy array, pp.shape=(num_pts,d)
        sample points
        pp[k,:] is the k'th sample point
    all_mu : numpy array, all_mu.shape=(num_pts,d)
        impulse response means
        all_mu[k,:] is the mean of the k'th impulse response
    all_Sigma : numpy array, all_Sigma.shape=(num_pts,d,d)
        impulse response covariance matrices
        all_Sigma[k,:,:] is the covariance matrix for the k'th impulse response
    tau : nonnegative float
        impulse response ellipsoid size parameter
        k'th ellipsoid = {x: (x-mu[k,:])^T Sigma[k,:,:]^{-1} (x-mu[k,:]) <= tau}
        support of k'th impulse response should be contained inside k'th ellipsoid
    batch_lengths : list of ints, len(batch_lengths)=num_batches
        number of impulse responses in each batch.
        E.g., if batch_lengths=[3,5,4], then
        ff_batches[0] contains the first 3 impulse responses,
        ff_batches[1] contains the next 5 impulse responses,
        ff_batches[2] contains the last 4 impulse responses,
    grid_density_multiplier : nonnegative float
        grid density scaling factor
        If grid_density_multiplier=1.0, then the spacing between gridpoints
        in a given BoxFunction equals the minimum distance between mesh nodes in the box.
        If grid_density_multiplier=0.5, then the spacing between gridpoints
        in a given BoxFunction equals one half the minimum distance between mesh nodes in the box.
    w_support_rtol : nonnegative float
        truncation tolerance determining how big the support boxes are for the weighting functions.
        If w_support_rtol=2e-2, then values of w outside the box will be no more than 2% of the maximum
        value of w.
    fill_value : float or numpy float. Default: np.nan
        Value to fill in missing convolution kernel information due to domain truncation.

    Returns
    -------
    WW : list of BoxFunctions. Weighting functions for each patch
    FF : list of BoxFunctions. Convolution kernels for each patch
    '''
    num_pts, d = pp.shape

    print('Forming weighting function patches and (un-extended) convolution kernel patches')
    WW = list()
    FF = list()
    for ii in tqdm(range(num_pts)):
        b, k = ind2sub_batches(ii, batch_lengths)
        W, F = get_W_and_initial_F(ww[ii], ff_batches[b], pp[ii,:], all_mu[ii,:], all_Sigma[ii,:,:], tau,
                                   grid_density_multiplier=grid_density_multiplier, w_support_rtol=w_support_rtol)
        WW.append(W)

        F.array[np.isnan(F.array)] = fill_value
        FF.append(F)

    return WW, FF


def compute_convolution_kernel_boundary_extensions(initial_FF, pp, num_extension_kernels):
    '''Fill in missing values in convolution kernels using neighboring kernels

    Parameters
    ----------
    initial_FF : list of BoxFunctions. len=num_kernels. Convolution kernels
    pp : numpy array. shape=(num_kernels, spatial_dimension). Sample points
    num_extension_kernels : number of neighboring kernels used to fill in missing values

    Returns
    -------
    FF : list of BoxFunctions. len=num_kernels. Convolution kernels with missing values filled in

    '''
    num_pts, d = pp.shape
    num_extension_kernels = np.min([num_extension_kernels, num_pts])

    print('Filling in missing kernel entries using neighboring kernels')
    pp_cKDTree = cKDTree(pp)
    FF = list()
    for ii in tqdm(range(num_pts)):
        _, neighbor_inds = pp_cKDTree.query(pp[ii, :], num_extension_kernels)
        neighbor_inds = list(neighbor_inds.reshape(-1))
        pp_nbrs = pp[neighbor_inds, :]
        FF_nbrs = [initial_FF[jj] for jj in neighbor_inds]

        if np.any(np.isnan(initial_FF[ii].array)):
            filled_in_F = fill_in_missing_kernel_values_using_other_kernels(FF_nbrs, pp_nbrs)
        else:
            filled_in_F = initial_FF[ii]

        FF.append(filled_in_F)

    return FF


def fill_in_missing_kernel_values_using_other_kernels(FF, pp):
    num_kernels, d = pp.shape
    new_F_min0 = np.min([F.min for F in FF], axis=0)
    new_F_max0 = np.max([F.max for F in FF], axis=0)
    new_F_min, new_F_max, new_F_shape = conforming_box(new_F_min0, new_F_max0, np.zeros(d), FF[0].h)

    new_FF = list()
    for k in range(num_kernels):
        Fk = BoxFunction(new_F_min, new_F_max, np.zeros(new_F_shape))
        Fk.array = FF[k](Fk.gridpoints, fill_value=np.nan).reshape(new_F_shape)
        new_FF.append(Fk)

        # new_Fk_array = eval_fenics_function_on_regular_grid(ff[k],
        #                                                     new_F_min + pp[k,:],
        #                                                     new_F_max + pp[k,:],
        #                                                     new_F_shape, outside_mesh_fill_value=np.nan)
        # Fk = BoxFunction(new_F_min, new_F_max, new_Fk_array)
        # Ek = ellipsoid_characteristic_function(new_F_min, new_F_max, new_F_shape, mus[k,:]-pp[k,:], Sigmas[k,:,:], tau)
        # new_FF.append(Ek * Fk)

    valid_mask_0 = np.logical_not(np.isnan(new_FF[0].array))
    cc = np.zeros(num_kernels)
    JJ = np.zeros(num_kernels)
    for k in range(num_kernels):
        valid_mask_k = np.logical_not(np.isnan(new_FF[k].array))
        valid_mask_joint = np.logical_and(valid_mask_0, valid_mask_k)
        f0 = new_FF[0].array[valid_mask_joint]
        fk = new_FF[k].array[valid_mask_joint]
        cc[k] = np.sum(f0 * fk) / np.sum(fk * fk) # what if there is no overlap?
        JJ[k] = np.linalg.norm(f0 - cc[k] * fk)

    best_match_sort_inds = np.argsort(JJ).reshape(-1)
    filled_in_F_array = np.nan * np.ones(new_F_shape)
    for k in list(best_match_sort_inds):
        nan_mask = np.isnan(filled_in_F_array)
        filled_in_F_array[nan_mask] = cc[k] * new_FF[k].array[nan_mask]

    nan_mask = np.isnan(filled_in_F_array)
    filled_in_F_array[nan_mask] = 0.0
    filled_in_F = BoxFunction(new_F_min, new_F_max, filled_in_F_array)
    return filled_in_F


def get_W_and_initial_F(w, f, p, mu, Sigma, tau,
                        grid_density_multiplier=0.5, w_support_rtol=2e-2,
                        boundary_extension_method='reflect'):
    '''Constructs initial BoxFunctions for one weighting function and associated convolution kernel

    Parameters
    ----------
    w : fenics Function
        weighting function
    f : fenics Function
        impulse response batch
    p : numpy array, p.shape=(d,)
        sample point
    mu : numpy array, mu.shape=(d,)
        impulse response mean
    Sigma : numpy array, Sigma.shape=(d,d)
        impulse response covariance matrix
    tau : nonnegative float
        scalar ellipsoid size for impulse response
        impulse response support should be contained in ellipsoid E, given by
        E={x: (x-mu)^T Sigma^{-1} (x-mu) <= tau}
    grid_density_multiplier : nonnegative float
        grid density scaling factor
        If grid_density_multiplier=1.0, then the spacing between gridpoints
        in a given BoxFunction equals the minimum distance between mesh nodes in the box.
        If grid_density_multiplier=0.5, then the spacing between gridpoints
        in a given BoxFunction equals one half the minimum distance between mesh nodes in the box.
    w_support_rtol : nonnegative float
        truncation tolerance determining how big the support boxes are for the weighting functions.
        If w_support_rtol=2e-2, then values of w outside the box will be no more than 2% of the maximum
        value of w.

    Returns
    -------
    W : BoxFunction
        weighting function on regular grid in a box
    F : BoxFunction
        convolution kernel on regular grid in a box
    '''
    V = w.function_space()
    dof_coords = V.tabulate_dof_coordinates()

    W_min0, W_max0 = function_support_box(w.vector()[:], dof_coords, support_rtol=w_support_rtol)

    F_min_plus_p0, F_max_plus_p0 = ellipsoid_bounding_box(mu, Sigma, tau)
    F_min_plus_p0 = F_min_plus_p0 + 2. * (p - mu) * ((p - mu) < 0) # Make the box bigger
    F_max_plus_p0 = F_max_plus_p0 + 2. * (p - mu) * ((p - mu) > 0)

    h_w = shortest_distance_between_points_in_box(W_min0, W_max0, dof_coords)
    h_f = shortest_distance_between_points_in_box(F_min_plus_p0, F_max_plus_p0, dof_coords)
    h = np.min([h_w, h_f]) * grid_density_multiplier

    W_min, W_max, W_shape = conforming_box(W_min0, W_max0, p, h)
    F_min_plus_p, F_max_plus_p, F_shape = conforming_box(F_min_plus_p0, F_max_plus_p0, p, h)

    W_array = eval_fenics_function_on_regular_grid(w, W_min, W_max, W_shape,
                                                   outside_mesh_fill_value=None)
    # W_array = eval_fenics_function_on_regular_grid(w, W_min, W_max, W_shape, outside_mesh_fill_value=0.0)
    W = BoxFunction(W_min, W_max, W_array)

    if boundary_extension_method == 'reflect':
        ellipsoid_mask_function = lambda pp: point_is_in_ellipsoid(pp, mu, Sigma, tau)
        F_array = eval_fenics_function_on_regular_grid(f, F_min_plus_p, F_max_plus_p, F_shape,
                                                       boundary_reflection=True,
                                                       mask_function = ellipsoid_mask_function,
                                                       outside_mask_fill_value=np.nan)
        F = BoxFunction(F_min_plus_p, F_max_plus_p, F_array).translate(-p)
    else:
        F_array = eval_fenics_function_on_regular_grid(f, F_min_plus_p, F_max_plus_p, F_shape,
                                                       outside_mesh_fill_value=np.nan)
        E = ellipsoid_characteristic_function(F_min_plus_p, F_max_plus_p, F_shape, mu, Sigma, tau)
        F = (E * BoxFunction(F_min_plus_p, F_max_plus_p, F_array)).translate(-p)

    return W, F
