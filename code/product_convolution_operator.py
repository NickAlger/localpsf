import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interpn
from tqdm.auto import tqdm

from nalger_helper_functions import *
import hlibpro_python_wrapper as hpro


def build_product_convolution_hmatrix_from_fenics_functions(ww, ff_batches, batch_lengths, pp,
                                                            all_mu, all_Sigma, tau,
                                                            V_out=None,
                                                            grid_density_multiplier=1.0,
                                                            w_support_rtol=2e-2,
                                                            num_extension_kernels=8,
                                                            block_cluster_tree=None,
                                                            hmatrix_tol=1e-6,
                                                            bct_admissibility_eta=2.0,
                                                            cluster_size_cutoff=50,
                                                            use_boundary_extension=True,
                                                            return_extras=False):
    '''Builds hierarchical matrix representation of product convolution operator
    with given weighting functions and impulse response batches

    Parameters
    ----------
    ww : list of fenics Functions, len(ww)=num_pts
        weighting functions
    ff_batches : list of fenics Functions, len(ff)=num_batches
        impulse response function batches
    batch_lengths : list of ints, len(batch_lengths)=num_batches
        number of impulse responses in each batch.
        E.g., if batch_lengths=[3,5,4], then
        ff_batches[0] contains the first 3 impulse responses,
        ff_batches[1] contains the next 5 impulse responses,
        ff_batches[2] contains the last 4 impulse responses
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
    V_out : fenics FunctionSpace
        function space for output of product-convolution operator.
        If V_out is None (default), then the input FunctionSpace (inferred from weighting functions)
        is also used for the output
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
    num_extension_kernels : positive int
        number of initial impulse responses used to fill in the missing information for each impulse response
        E.g., if num_extension_kernels=8, then a given filled in impulse response will contain information
        from itself, and the 7 other nearest impulse responses.
    block_cluster_tree : BlockClusterTree.
        Block cluster tree for H-matrix.
        None (Default): build block cluster tree from scratch
    hmatrix_tol : nonnegative float. Accuracy tolerance for H-matrix
    bct_admissibility_eta : nonnegative float. Admissibility tolerance for block cluster tree.
        Only used of block_cluster_tree is not supplied
        A block of the matrix is admissible (low rank) if:
            distance(A, B) <= eta * min(diameter(A), diameter(B))
        where A is the cluster of points associated with the rows of the block, and
              B is the cluster of points associated with the columns of the block.
    cluster_size_cutoff : positive int. number of points below which clusters are not subdivided.
    use_boundary_extension : bool. True:
        fill in convolution kernel missing values using neighboring convolution kernels
    return_extras

    Returns
    -------
    A_hmatrix : HMatrix. hierarchical matrix representation of product-convolution operator
    WW : list of BoxFunctions. Weighting functions for each patch
    FF : list of BoxFunctions. Convolution kernels for each patch (with extension if done)
    initial_FF : list of BoxFunctions. Convolution kernels for each patch without extension

    '''
    if use_boundary_extension:
        kernel_fill_value=np.nan
    else:
        kernel_fill_value = 0.0

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

    V_in = ww[0].function_space()
    row_dof_coords = V_in.tabulate_dof_coordinates()
    if V_out is not None:
        col_dof_coords = V_out.tabulate_dof_coordinates()
    else:
        col_dof_coords = row_dof_coords

    A_hmatrix = build_product_convolution_hmatrix_from_patches(WW, FF, row_dof_coords, col_dof_coords,
                                                               block_cluster_tree=block_cluster_tree,
                                                               tol=hmatrix_tol,
                                                               admissibility_eta=bct_admissibility_eta,
                                                               cluster_size_cutoff=cluster_size_cutoff)

    if return_extras:
        return A_hmatrix, WW, FF, initial_FF
    else:
        return A_hmatrix



def build_product_convolution_hmatrix_from_patches(WW, FF, row_dof_coords, col_dof_coords,
                                                   block_cluster_tree=None, tol=1e-6,
                                                   admissibility_eta=2.0, cluster_size_cutoff=50):
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

    if block_cluster_tree is None:
        print('Building cluster trees and block cluster tree')
        row_ct = hpro.build_cluster_tree_from_pointcloud(row_dof_coords, cluster_size_cutoff=cluster_size_cutoff)
        col_ct = hpro.build_cluster_tree_from_pointcloud(col_dof_coords, cluster_size_cutoff=cluster_size_cutoff)
        block_cluster_tree = hpro.build_block_cluster_tree(row_ct, col_ct, admissibility_eta=admissibility_eta)

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


def build_product_convolution_operator_from_fenics_functions(ww, ff_batches, pp, all_mu, all_Sigma, tau, batch_lengths,
                                                             grid_density_multiplier=1.0, w_support_rtol=2e-2,
                                                             num_extension_kernels=8, V_out=None):
    '''Constructs a ProductConvolutionOperator from fenics weighting functions and impulse response batches

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
    V_out : fenics FunctionSpace
        function space for output of product-convolution operator.
        If V_out is None, then the input FunctionSpace is also used for the output
    num_extension_kernels : positive int
        number of initial impulse responses used to fill in the missing information for each impulse response
        E.g., if num_extension_kernels=8, then a given filled in impulse response will contain information
        from itself, and the 7 other nearest impulse responses.

    Returns
    -------
    PC : ProductConvolutionOperator
    '''
    num_pts, d = pp.shape
    num_extension_kernels = np.min([num_extension_kernels, num_pts])

    WW, initial_FF = build_product_convolution_patches_from_fenics_functions(ww, ff_batches, pp, all_mu, all_Sigma, tau, batch_lengths,
                                                                             grid_density_multiplier=grid_density_multiplier,
                                                                             w_support_rtol=w_support_rtol, fill_value=np.nan)

    FF = compute_convolution_kernel_boundary_extensions(initial_FF, pp, num_extension_kernels)

    V_in = ww[0].function_space()
    if V_out is None:
        V_out = V_in
    PC = form_product_convolution_operator_from_patches(WW, FF, V_in, V_out)
    return PC


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


def get_W_and_initial_F(w, f, p, mu, Sigma, tau, grid_density_multiplier=1.0, w_support_rtol=2e-2):
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

    h_w = shortest_distance_between_points_in_box(W_min0, W_max0, dof_coords)
    h_f = shortest_distance_between_points_in_box(F_min_plus_p0, F_max_plus_p0, dof_coords)
    h = np.min([h_w, h_f]) * grid_density_multiplier

    W_min, W_max, W_shape = conforming_box(W_min0, W_max0, p, h)
    F_min_plus_p, F_max_plus_p, F_shape = conforming_box(F_min_plus_p0, F_max_plus_p0, p, h)

    W_array = eval_fenics_function_on_regular_grid(w, W_min, W_max, W_shape, outside_mesh_fill_value=None)
    # W_array = eval_fenics_function_on_regular_grid(w, W_min, W_max, W_shape, outside_mesh_fill_value=0.0)
    F_array = eval_fenics_function_on_regular_grid(f, F_min_plus_p, F_max_plus_p, F_shape, outside_mesh_fill_value=np.nan)
    E = ellipsoid_characteristic_function(F_min_plus_p, F_max_plus_p, F_shape, mu, Sigma, tau)

    W = BoxFunction(W_min, W_max, W_array)
    F = (E * BoxFunction(F_min_plus_p, F_max_plus_p, F_array)).translate(-p)
    return W, F


def form_product_convolution_operator_from_patches(WW, FF, V_in, V_out):
    shape = (V_out.dim(), V_in.dim())

    print('constructing dofs2patch transfer matrices')
    patch_cols = list()
    TT_dof2patch = list()
    for k in tqdm(range(len(WW))):
        T_dof2patch, col_dof_inds = make_dofs2patch_transfer_matrix(WW[k], V_in)
        patch_cols.append(col_dof_inds)
        TT_dof2patch.append(T_dof2patch)

    print('constructing patch2dofs transfer matrices')
    patch_rows = list()
    TT_patch2dof = list()
    for k in tqdm(range(len(WW))):
        T_patch2dof, row_dof_inds = make_patch2dofs_transfer_matrix(WW[k], FF[k], V_out)
        patch_rows.append(row_dof_inds)
        TT_patch2dof.append(T_patch2dof)

    row_coords = V_out.tabulate_dof_coordinates()
    col_coords = V_in.tabulate_dof_coordinates()
    return ProductConvolutionOperator(WW, FF, shape, patch_cols, patch_rows,
                                      TT_dof2patch, TT_patch2dof, row_coords, col_coords)


def convolution_support(F, G):
    C_min = F.min + G.min
    C_max = F.max + G.max
    C_shape = tuple(np.array(F.shape) + np.array(G.shape) - 1)
    return C_min, C_max, C_shape


def make_patch2dofs_transfer_matrix(Wk, Fk, V_out):
    if np.linalg.norm(Wk.h - Fk.h) > 1e-10:
        raise RuntimeError('BoxFunctions have different spacings h')

    pp = V_out.tabulate_dof_coordinates()

    C_min, C_max, C_shape = convolution_support(Wk, Fk)
    dof_inds = np.argwhere(point_is_in_box(pp, C_min, C_max)).reshape(-1)
    T_patch2dof = multilinear_interpolation_matrix(pp[dof_inds, :], C_min, C_max, C_shape)
    return T_patch2dof, dof_inds


def make_dofs2patch_transfer_matrix(Wk, V_in):
    T_dof2patch, dof_inds = pointwise_observation_matrix(Wk.gridpoints, V_in, nonzero_columns_only=True)
    return T_dof2patch, dof_inds


class ProductConvolutionOperator:
    def __init__(me, WW, FF, shape, patch_cols, patch_rows, TT_dof2patch, TT_patch2dof, row_coords, col_coords):
        '''Product-convolution operator with weighting functions
        and convolution kernels defined on regular grid patches

        Parameters
        ----------
        WW : list of BoxFunctions, len(WW)=num_patches
            weighting functions for each patch
        FF : list of BoxFunctions, len(FF)=num_patches
            convolution kernels for each patch (zero-centered)
        shape : tuple
            shape of the operator. shape=(num_rows, num_cols)
        patch_cols : list of numpy arrays, len(patch_cols)=num_patches, patch_cols[k].dtype=int
            list of arrays of potentially nonzero columns associated with each patch
        patch_rows : list of numpy arrays, len(patch_rows)=num_patches, patch_rows[k].dtype=int
            list of arrays of potentially nonzero rows associated with each patch
        TT_dof2patch : list of scipy.sparse.csr_matrix, len(TT_dof2patch)=num_patches
            list of dof_vector-to-patch transfer matrices
            U = (TT_dof2patch[k] * u[patch_rows[k]]).reshape(WW[k].shape)
            u is vector of function values at dof locations, u.shape=(num_rows,)
            U is array of function values on weighting function patch grid, U.shape=WW[k].shape
        TT_patch2dof : list of scipy.sparse.csr_matrix, len(TT_patch2dof)=num_patches
            list of patch-to-dof_vector transfer matrices
            let q be a vector of function values at dof locations, q.shape=(num_cols,)
            q[output_ind_groups[k]] = TT_dof2grid[k] * Q.reshape(-1)
            Q is array of function values on convolution patch grid, Q.shape=boxconv(FF[k], WW[k]).shape
        row_coords : numpy array, row_coords.shape=(num_rows,d), d=spatial dimension
            coordinates for row dof locations
            row_coords[k,:] is the point in R^d associated with the k'th row of the operator
        col_coords : numpy array, row_coords.shape=(num_rows,d), d=spatial dimension
            coordinates for column dof locations
            col_coords[k,:] is the point in R^d associated with the k'th column of the operator
        '''
        me.WW = WW
        me.FF = FF
        me.patch_cols = patch_cols
        me.patch_rows = patch_rows
        me.TT_dof2grid = TT_dof2patch
        me.TT_grid2dof = TT_patch2dof
        me.row_coords = row_coords
        me.col_coords = col_coords

        me.num_patches = len(me.WW)
        me.d = WW[0].ndim
        me.shape = shape

        me.FF_star = [F.flip().conj() for F in me.FF]

        me.dtype = dtype_max([W.dtype for W in me.WW] + [F.dtype for F in me.FF])

        print('precomputing which patches are relevant for each row')
        me.row_patches = [set() for _ in range(me.shape[0])]
        for p in tqdm(range(len(me.patch_rows))):
            for row in me.patch_rows[p]:
                me.row_patches[row].add(p)

        print('precomputing which patches are relevant for each column')
        me.col_patches = [set() for _ in range(me.shape[1])]
        for p in tqdm(range(len(me.patch_cols))):
            for col in me.patch_cols[p]:
                me.col_patches[col].add(p)

    def get_scattered_entries_slow(me, rows, cols):
        num_entries = len(rows)
        print('computing ' + str(num_entries) + ' entries')
        entries = np.zeros(num_entries, dtype=me.dtype)
        for k in tqdm(range(num_entries)):
            row = rows[k]
            col = cols[k]
            patches = me.col_patches[col].intersection(me.row_patches[row])
            if patches:
                x = me.col_coords[col, :]
                y = me.row_coords[row, :]
                for p in patches:
                    entries[k] += me.WW[p](x) * me.FF[p](y - x)
        return entries

    def get_scattered_entries(me, rows, cols, entries_per_batch=int(1e6)):
        num_entries = len(rows)
        batch_size = np.min([entries_per_batch, num_entries])
        print('computing ' + str(num_entries) + ' matrix entries in batches of ' + str(batch_size))
        entries = list()
        for start in tqdm(range(0, num_entries, batch_size)):
            stop = np.min([start + batch_size, num_entries])
            entries.append(me._get_scattered_entries(rows[start:stop], cols[start:stop]))
        entries = np.concatenate(entries)
        return entries

    def _get_scattered_entries(me, rows, cols):
        num_entries = len(rows)

        kk_per_patch = [list() for _ in range(me.num_patches)] # index into entries
        for k in range(num_entries):
            row = rows[k]
            col = cols[k]
            patches = me.col_patches[col].intersection(me.row_patches[row])
            if patches:
                for p in patches:
                    kk_per_patch[p].append(k)

        entries = np.zeros(num_entries, dtype=me.dtype)
        for p in range(me.num_patches):
            kk = kk_per_patch[p]
            if kk:
                xx = me.col_coords[cols[kk], :]
                yy = me.row_coords[rows[kk], :]
                entries[kk] += me.WW[p](xx) * me.FF[p](yy - xx)

        return entries

    def get_block(me, block_rows, block_cols, entries_per_batch=int(1e6)):
        block_shape = (len(block_rows), len(block_cols))
        col_batch_size = int(np.ceil(entries_per_batch / block_shape[1]))
        print('computing ' + str(block_shape[0]) + ' x ' + str(block_shape[1]) +
              ' matrix block in batches of '+ str(col_batch_size) + ' cols')
        entries = list()
        starts = range(0,block_shape[1], col_batch_size)
        for start in tqdm(starts):
            stop = np.min([start + col_batch_size, block_shape[1]])
            cols_batch = block_cols[start:stop]
            entries.append(me._get_block(block_rows, cols_batch))
        entries = np.hstack(entries)
        return entries

    def _get_block(me, block_rows, block_cols):
        block_shape = (len(block_rows), len(block_cols))

        ii_per_patch = [list() for _ in range(me.num_patches)]
        jj_per_patch = [list() for _ in range(me.num_patches)]

        for jj in range(block_shape[1]):
            col = block_cols[jj]
            for ii in range(block_shape[0]):
                row = block_rows[ii]
                patches = me.col_patches[col].intersection(me.row_patches[row])
                if patches:
                    for p in patches:
                        ii_per_patch[p].append(ii)
                        jj_per_patch[p].append(jj)

        unique_jj_per_patch = [list() for _ in range(me.num_patches)]
        unique_jj_inverse_per_patch = [list() for _ in range(me.num_patches)]
        for p in range(me.num_patches):
            unique_jj_per_patch[p], unique_jj_inverse_per_patch[p] = \
                np.unique(jj_per_patch[p], return_inverse=True)

        entries = np.zeros(block_shape, dtype=me.dtype)
        for p in range(me.num_patches):
            ii = ii_per_patch[p]
            if ii:
                jj = jj_per_patch[p]
                unique_jj = unique_jj_per_patch[p]
                unique_jj_inverse = unique_jj_inverse_per_patch[p]

                Wp = me.WW[p](me.col_coords[block_cols[unique_jj],:])[unique_jj_inverse]
                Fp = me.FF[p](me.row_coords[block_rows[ii],:] - me.col_coords[block_cols[jj],:])

                entries[ii,jj] += Wp * Fp

        return entries


    def matvec(me, u):
        v = np.zeros(me.shape[0], dtype=dtype_max([me.dtype, u.dtype]))
        for p in range(me.num_patches):
            ii = me.patch_cols[p]
            oo = me.patch_rows[p]
            Ti = me.TT_dof2grid[p]
            To = me.TT_grid2dof[p]
            Wk = me.WW[p]
            Fk = me.FF[p]

            Uk_array = (Ti * u[ii]).reshape(Wk.shape)
            WUk = BoxFunction(Wk.min, Wk.max, Wk.array * Uk_array)
            Vk = boxconv(Fk, WUk)
            v[oo] += To * Vk.array.reshape(-1)
        return v

    def rmatvec(me, v):
        u = np.zeros(me.shape[1], dtype=dtype_max([me.dtype, v.dtype]))
        for p in range(me.num_patches):
            ii = me.patch_cols[p]
            oo = me.patch_rows[p]
            Ti = me.TT_dof2grid[p]
            To = me.TT_grid2dof[p]
            Wk = me.WW[p]
            Fk = me.FF[p]
            Fk_star = me.FF_star[p]

            Vk_min, Vk_max, Vk_shape = convolution_support(Fk, Wk)

            Vk_array = (To.T * v[oo]).reshape(Vk_shape)
            Vk = BoxFunction(Vk_min, Vk_max, Vk_array)
            Sk = boxconv(Fk_star, Vk)
            Uk_big = Sk * Wk
            Uk = Uk_big.restrict_to_new_box(Wk.min, Wk.max)
            u[ii] += Ti.T * Uk.array.reshape(-1)
        return u

    def astype(me, dtype):
        WW_new = [W.astype(dtype) for W in me.WW]
        FF_new = [F.astype(dtype) for F in me.FF]
        TT_dof2grid_new = [T.astype(dtype) for T in me.TT_dof2grid]
        TT_grid2dof_new = [T.astype(dtype) for T in me.TT_grid2dof]
        return ProductConvolutionOperator(WW_new, FF_new,
                                          me.patch_cols, me.patch_rows,
                                          TT_dof2grid_new, TT_grid2dof_new)

    @property
    def real(me):
        WW_new = [W.real for W in me.WW]
        FF_new = [F.real for F in me.FF]
        TT_dof2grid_new = [T.real for T in me.TT_dof2grid]
        TT_grid2dof_new = [T.real for T in me.TT_grid2dof]
        return ProductConvolutionOperator(WW_new, FF_new,
                                          me.patch_cols, me.patch_rows,
                                          TT_dof2grid_new, TT_grid2dof_new)

    @property
    def imag(me):
        FF_new = [F.imag for F in me.FF]
        return ProductConvolutionOperator(me.WW, FF_new,
                                          me.patch_cols, me.patch_rows,
                                          me.TT_dof2grid, me.TT_grid2dof)


def square_root_of_product_convolution_operator(PC):
    sqrt_FF = [convolution_square_root(F, positive_real_branch_cut=np.pi,
                                       negative_real_branch_cut=np.pi).real for F in PC.FF]
    sqrt_PC = ProductConvolutionOperator(PC.WW, sqrt_FF, PC.shape,
                                         PC.patch_cols, PC.patch_rows,
                                         PC.TT_dof2grid, PC.TT_grid2dof)
    return sqrt_PC

