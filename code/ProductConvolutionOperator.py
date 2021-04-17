import numpy as np

from nalger_helper_functions import *


def build_product_convolution_operator_from_fenics_functions(ww, ff_batches, pp, all_mu, all_Sigma, tau, batch_lengths,
                                                             grid_density_multiplier=1.0, w_support_rtol=2e-2,
                                                             max_neighbors_for_f_extension=10):
    # Input:
    #   ww: weighting functions (list of fenics Functions, len(ww) = num_pts)
    #   ff_batches: impulse response batches (list of fenic Functions, len(ff) = num_batches)
    #   pp: sample points (np.array, pp.shape = (num_pts, d))
    #   all_mu: impulse response means (np.array, all_mu.shape = (num_pts, d))
    #   all_Sigma: impulse response covariance matrices (np.array, all_Sigma.shape = (num_pts, d, d))
    #   tau: scalar ellipsoid size. Ellipsoid={x: (x-mu)^T Sigma^{-1} (x-mu) <= tau}
    #   grid_density_multiplier: scalar determining density of grid for BoxFunctions. 1.0 => grid h = min mesh h in box
    #   w_support_rtol: scalar determining support box for weighting function
    #   max_neighbors_for_f_extension: number of neighboring impulse response neighbors used to fill in missing information
    num_pts, d = pp.shape

    WW = list()
    initial_FF = list()
    for ii in range(num_pts):
        b, k = ind2sub_batches(ii, batch_lengths)
        W, F = get_W_and_initial_F(ww[ii], ff_batches[b], pp[ii,:], all_mu[ii,:], all_Sigma[ii,:,:], tau,
                                   grid_density_multiplier=grid_density_multiplier, w_support_rtol=w_support_rtol)
        WW.append(W)
        initial_FF.append(F)


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
        Ek = ellipsoid_characteristic_function(new_F_min, new_F_max, new_F_shape, mus[k,:], Sigmas[k,:,:], tau)
        new_FF.append(Ek * Fk)

    valid_mask_0 = np.logical_not(np.isnan(new_FF[0].array))
    cc = np.zeros(num_kernels)
    JJ = np.zeros(num_kernels)
    for k in range(num_kernels):
        valid_mask_k = np.logical_not(np.isnan(new_FF[k].array))
        valid_mask_joint = np.logical_and(valid_mask_0, valid_mask_k)
        f0 = FF[0].array[valid_mask_joint]
        fk = FF[k].array[valid_mask_joint]
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
    # Input:
    #   w: weighting function (fenics Function)
    #   f: impulse response batch function (fenics Function)
    #   p: sample point coordinates (np.array, p.shape=(d,)
    #   mu: impulse response mean (np.array, mu.shape=(d,))
    #   Sigma: impulse response covariance matrix (np.array, Sigma.shape=(d,d))
    #   tau: scalar ellipsoid size. Ellipsoid={x: (x-mu)^T Sigma^{-1} (x-mu) <= tau}
    #   grid_density_multiplier: scalar determining density of grid for BoxFunctions. 1.0 => grid h = min mesh h in box
    #   w_support_rtol: scalar determining support box for weighting function
    # Output:
    #   W: weighting BoxFunction
    #   F: initial convolution kernel BoxFunction
    V = w[0].function_space()
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

    input_ind_groups = list()
    TT_dof2grid = list()
    output_ind_groups = list()
    TT_grid2dof = list()
    for k in range(len(WW)):
        T_dof2grid, input_dof_inds = make_dofs2grid_transfer_matrix(WW[k], V_in)
        input_ind_groups.append(input_dof_inds)
        TT_dof2grid.append(T_dof2grid)

        T_grid2dof, output_dof_inds = make_grid2dofs_transfer_matrix(WW[k], FF[k], V_out)
        output_ind_groups.append(output_dof_inds)
        TT_grid2dof.append(T_grid2dof)

    return ProductConvolutionOperator(WW, FF, shape, input_ind_groups, output_ind_groups, TT_dof2grid, TT_grid2dof)


def make_grid2dofs_transfer_matrix(Wk, Fk, V_out):
    if np.linalg.norm(Wk.h - Fk.h) > 1e-10:
        raise RuntimeError('BoxFunctions have different spacings h')

    pp = V_out.tabulate_dof_coordinates()

    C_min = Wk.min + Fk.min
    C_max = Wk.max + Fk.max
    C_shape = tuple(np.array(Wk.shape) + np.array(Fk.shape) - 1)
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
            To = me.TT_grid2dof
            Wk = me.WW[k]
            Fk = me.FF[k]

            Uk_array = (Ti * u[ii]).reshape(Wk.shape)
            WUk = BoxFunction(Wk.min, Wk.max, Wk.array * Uk_array)
            Vk = boxconv(Fk, WUk)
            v[oo] += To * Vk.reshape(-1)
        return v

    def rmatvec(me, v):
        u = np.zeros(me.shape[1], dtype=dtype_max([me.dtype, v.dtype]))
        for k in range(me.num_patches):
            ii = me.input_ind_groups[k]
            oo = me.output_ind_groups[k]
            Ti = me.TT_dof2grid[k]
            To = me.TT_grid2dof
            Wk = me.WW[k]
            Fk_star = me.FF_star[k]

            Vk = To.T * v[oo]
            Sk = boxconv(Fk_star, Vk)
            Uk_big = Sk * Wk
            Uk = Uk_big.restrict_to_new_box(Wk.min, Wk.max)
            u[ii] += Ti.T * Uk.reshape(-1)
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


