import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from nalger_helper_functions import *


def build_product_convolution_operator_from_fenics_functions(ww, ff_batches, pp, all_mu, all_Sigma, tau, batch_lengths,
                                                             grid_density_multiplier=1.0, w_support_rtol=2e-2,
                                                             num_f_extension_kernels=8, V_out=None):
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
    num_f_extension_kernels : positive int
        number of initial impulse responses used to fill in the missing information for each impulse response
        E.g., if num_f_extension_kernels=8, then a given filled in impulse response will contain information
        from itself, and the 7 other nearest impulse responses.

    Returns
    -------
    PC : ProductConvolutionOperator
    '''
    num_pts, d = pp.shape
    num_f_extension_kernels = np.min([num_f_extension_kernels,  num_pts])

    print('Forming weighting function patches and initial kernel patches')
    WW = list()
    initial_FF = list()
    for ii in tqdm(range(num_pts)):
        b, k = ind2sub_batches(ii, batch_lengths)
        W, F = get_W_and_initial_F(ww[ii], ff_batches[b], pp[ii,:], all_mu[ii,:], all_Sigma[ii,:,:], tau,
                                   grid_density_multiplier=grid_density_multiplier, w_support_rtol=w_support_rtol)
        WW.append(W)
        initial_FF.append(F)

    print('Filling in missing kernel entries using neighboring kernels')
    pp_cKDTree = cKDTree(pp)
    FF = list()
    for ii in tqdm(range(num_pts)):
        _, neighbor_inds = pp_cKDTree.query(pp[ii, :], num_f_extension_kernels)
        neighbor_inds = list(neighbor_inds.reshape(-1))
        pp_nbrs = pp[neighbor_inds, :]
        mu_nbrs = all_mu[neighbor_inds, :]
        Sigma_nbrs = all_Sigma[neighbor_inds, :, :]
        FF_nbrs = [initial_FF[jj] for jj in neighbor_inds]
        ff_nbrs = list()
        for jj in neighbor_inds:
            b, k = ind2sub_batches(jj, batch_lengths)
            ff_nbrs.append(ff_batches[b])

        filled_in_F = fill_in_missing_kernel_values_using_other_kernels(FF_nbrs, ff_nbrs, pp_nbrs,
                                                                        mu_nbrs, Sigma_nbrs, tau)
        FF.append(filled_in_F)

    V_in = ww[0].function_space()
    if V_out is None:
        V_out = V_in
    PC = form_product_convolution_operator_from_patches(WW, FF, V_in, V_out)
    return PC


def fill_in_missing_kernel_values_using_other_kernels(FF, ff, pp, mus, Sigmas, tau):
    num_kernels, d = pp.shape
    new_F_min0 = np.min([F.min for F in FF], axis=0)
    new_F_max0 = np.max([F.max for F in FF], axis=0)
    new_F_min, new_F_max, new_F_shape = conforming_box(new_F_min0, new_F_max0, np.zeros(d), FF[0].h)

    new_FF = list()
    for k in range(num_kernels):
        new_Fk_array = eval_fenics_function_on_regular_grid(ff[k],
                                                            new_F_min + pp[k,:],
                                                            new_F_max + pp[k,:],
                                                            new_F_shape, outside_mesh_fill_value=np.nan)
        Fk = BoxFunction(new_F_min, new_F_max, new_Fk_array)
        Ek = ellipsoid_characteristic_function(new_F_min, new_F_max, new_F_shape, mus[k,:]-pp[k,:], Sigmas[k,:,:], tau)
        new_FF.append(Ek * Fk)

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

    W_array = eval_fenics_function_on_regular_grid(w, W_min, W_max, W_shape, outside_mesh_fill_value=0.0)
    F_array = eval_fenics_function_on_regular_grid(f, F_min_plus_p, F_max_plus_p, F_shape, outside_mesh_fill_value=np.nan)
    E = ellipsoid_characteristic_function(F_min_plus_p, F_max_plus_p, F_shape, mu, Sigma, tau)

    W = BoxFunction(W_min, W_max, W_array)
    F = (E * BoxFunction(F_min_plus_p, F_max_plus_p, F_array)).translate(-p)
    return W, F


def form_product_convolution_operator_from_patches(WW, FF, V_in, V_out):
    shape = (V_out.dim(), V_in.dim())

    print('constructing patch transfer matrices')
    input_ind_groups = list()
    TT_dof2grid = list()
    output_ind_groups = list()
    TT_grid2dof = list()
    for k in tqdm(range(len(WW))):
        T_dof2grid, input_dof_inds = make_dofs2grid_transfer_matrix(WW[k], V_in)
        input_ind_groups.append(input_dof_inds)
        TT_dof2grid.append(T_dof2grid)

        T_grid2dof, output_dof_inds = make_grid2dofs_transfer_matrix(WW[k], FF[k], V_out)
        output_ind_groups.append(output_dof_inds)
        TT_grid2dof.append(T_grid2dof)

    return ProductConvolutionOperator(WW, FF, shape, input_ind_groups, output_ind_groups, TT_dof2grid, TT_grid2dof)


def convolution_support(F, G):
    C_min = F.min + G.min
    C_max = F.max + G.max
    C_shape = tuple(np.array(F.shape) + np.array(G.shape) - 1)
    return C_min, C_max, C_shape


def make_grid2dofs_transfer_matrix(Wk, Fk, V_out):
    if np.linalg.norm(Wk.h - Fk.h) > 1e-10:
        raise RuntimeError('BoxFunctions have different spacings h')

    pp = V_out.tabulate_dof_coordinates()

    C_min, C_max, C_shape = convolution_support(Wk, Fk)
    # C_min = Wk.min + Fk.min
    # C_max = Wk.max + Fk.max
    # C_shape = tuple(np.array(Wk.shape) + np.array(Fk.shape) - 1)
    output_dof_inds = np.argwhere(point_is_in_box(pp, C_min, C_max)).reshape(-1)
    T_grid2dof = multilinear_interpolation_matrix(pp[output_dof_inds, :], C_min, C_max, C_shape)
    return T_grid2dof, output_dof_inds


def make_dofs2grid_transfer_matrix(Wk, V_in):
    T_dof2grid, input_dof_inds = pointwise_observation_matrix(Wk.gridpoints, V_in, nonzero_columns_only=True)
    return T_dof2grid, input_dof_inds


class ProductConvolutionOperator:
    def __init__(me, WW, FF, shape, input_ind_groups, output_ind_groups, TT_dof2grid, TT_grid2dof):
        # WW: weighting functions (a length-k list of BoxFunctions)
        # FF: convolution kernels (a length-k list of BoxFunctions)
        # shape: shape of the operator. tuple. shape=(num_rows, num_cols)
        #
        # input_ind_groups: length-k list of arrays of indices for dofs that contribute to each convolution input
        #     u[input_ind_groups[k]] = dof values from u that are relevant to WW[k]
        #
        # output_ind_groups: length-k list of arrays of indices for dofs that are affected by each convolution output
        #     v[input_ind_groups[k]] = dof values from v that are affected to boxconv(FF[k], WW[k])
        #
        # TT_dof2grid: list of sparse dof-to-grid transfer matrices.
        #     U = (TT_dof2grid[k] * u[input_ind_groups[k]]).reshape(WW[k].shape)
        #     u is vector of function values at dof locations, u.shape=(num_pts,)
        #     U is array of function values on weighting function grid, U.shape=WW[k].shape
        #
        # TT_grid2dof: list of sparse grid-to-dof transfer matrices.
        #     q[output_ind_groups[k]] = TT_dof2grid[k] * Q.reshape(-1)
        #     q is vector of function values at dof locations, q.shape=(num_pts,)
        #     Q is array of function values on convolution grid, Q.shape=boxconv(FF[k], WW[k]).shape
        me.WW = WW
        me.FF = FF
        me.input_ind_groups = input_ind_groups
        me.output_ind_groups = output_ind_groups
        me.TT_dof2grid = TT_dof2grid
        me.TT_grid2dof = TT_grid2dof

        me.num_patches = len(me.WW)
        me.d = WW[0].ndim
        me.shape = shape

        me.FF_star = [F.flip().conj() for F in me.FF]

        me.dtype = dtype_max([W.dtype for W in me.WW] + [F.dtype for F in me.FF])

    def matvec(me, u):
        v = np.zeros(me.shape[0], dtype=dtype_max([me.dtype, u.dtype]))
        for k in range(me.num_patches):
            ii = me.input_ind_groups[k]
            oo = me.output_ind_groups[k]
            Ti = me.TT_dof2grid[k]
            To = me.TT_grid2dof[k]
            Wk = me.WW[k]
            Fk = me.FF[k]

            Uk_array = (Ti * u[ii]).reshape(Wk.shape)
            WUk = BoxFunction(Wk.min, Wk.max, Wk.array * Uk_array)
            Vk = boxconv(Fk, WUk)
            v[oo] += To * Vk.array.reshape(-1)
        return v

    def rmatvec(me, v):
        u = np.zeros(me.shape[1], dtype=dtype_max([me.dtype, v.dtype]))
        for k in range(me.num_patches):
            ii = me.input_ind_groups[k]
            oo = me.output_ind_groups[k]
            Ti = me.TT_dof2grid[k]
            To = me.TT_grid2dof[k]
            Wk = me.WW[k]
            Fk = me.FF[k]
            Fk_star = me.FF_star[k]

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
                                          me.input_ind_groups, me.output_ind_groups,
                                          TT_dof2grid_new, TT_grid2dof_new)

    @property
    def real(me):
        WW_new = [W.real for W in me.WW]
        FF_new = [F.real for F in me.FF]
        TT_dof2grid_new = [T.real for T in me.TT_dof2grid]
        TT_grid2dof_new = [T.real for T in me.TT_grid2dof]
        return ProductConvolutionOperator(WW_new, FF_new,
                                          me.input_ind_groups, me.output_ind_groups,
                                          TT_dof2grid_new, TT_grid2dof_new)

    @property
    def imag(me):
        FF_new = [F.imag for F in me.FF]
        return ProductConvolutionOperator(me.WW, FF_new,
                                          me.input_ind_groups, me.output_ind_groups,
                                          me.TT_dof2grid, me.TT_grid2dof)


def square_root_of_product_convolution_operator(PC):
    sqrt_FF = [convolution_square_root(F, positive_real_branch_cut=np.pi,
                                       negative_real_branch_cut=np.pi).real for F in PC.FF]
    sqrt_PC = ProductConvolutionOperator(PC.WW, sqrt_FF, PC.shape,
                                         PC.input_ind_groups, PC.output_ind_groups,
                                         PC.TT_dof2grid, PC.TT_grid2dof)
    return sqrt_PC

