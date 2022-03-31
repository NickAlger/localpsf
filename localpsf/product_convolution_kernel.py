import numpy as np

from .impulse_response_batches import ImpulseResponseBatches

import hlibpro_python_wrapper as hpro


class ProductConvolutionKernel:
    def __init__(me, V_in, V_out, apply_A, apply_At, num_row_batches, num_col_batches,
                 tau_rows=3.0, tau_cols=3.0,
                 num_neighbors_rows=10, num_neighbors_cols=10,
                 symmetric=False, gamma=1e-8, sigma_min=1e-6,
                 max_scale_discrepancy=1e2):
        me.V_in = V_in
        me.V_out = V_out
        me.apply_A = apply_A
        me.apply_At = apply_At
        me.max_scale_discrepancy = max_scale_discrepancy


        me.col_batches = ImpulseResponseBatches(V_in, V_out, apply_A, apply_At,
                                                num_initial_batches=num_col_batches,
                                                tau=tau_cols,
                                                num_neighbors=num_neighbors_cols,
                                                sigma_min=sigma_min,
                                                max_scale_discrepancy=max_scale_discrepancy)

        if symmetric:
            me.row_batches = me.col_batches
        else:
            me.row_batches = ImpulseResponseBatches(V_out, V_in, apply_At, apply_A,
                                                    num_initial_batches=num_row_batches,
                                                    tau=tau_rows,
                                                    num_neighbors=num_neighbors_rows,
                                                    sigma_min=sigma_min,
                                                    max_scale_discrepancy=max_scale_discrepancy)

        me.col_coords = me.col_batches.dof_coords_in
        me.row_coords = me.col_batches.dof_coords_out
        me.shape = (me.V_out.dim(), me.V_in.dim())

        me.cpp_object = hpro.hpro_cpp.ProductConvolutionKernelRBF(me.col_batches.cpp_object,
                                                                  me.row_batches.cpp_object,
                                                                  me.col_coords,
                                                                  me.row_coords,
                                                                  gamma)

    def __call__(me, yy, xx):
        if len(xx.shape) == 1 and len(yy.shape) == 1:
            return me.cpp_object.eval_integral_kernel(yy, xx)
        else:
            return me.cpp_object.eval_integral_kernel_block(yy, xx)

    def __getitem__(me, ii_jj):
        ii, jj = ii_jj
        yy = np.array(me.row_coords[ii,:].T, order='F')
        xx = np.array(me.col_coords[jj, :].T, order='F')
        return me.__call__(yy, xx)

    def build_hmatrix(me, bct, tol=1e-5):
        hmatrix_cpp_object = hpro.hpro_cpp.build_hmatrix_from_coefffn( me.cpp_object, bct.cpp_object, tol )
        return hpro.HMatrix(hmatrix_cpp_object, bct)

    @property
    def mean_shift(me):
        return me.cpp_object.mean_shift

    @mean_shift.setter
    def mean_shift(me, new_mean_shift_bool : bool):
        me.cpp_object.mean_shift = new_mean_shift_bool

    @property
    def vol_preconditioning(me):
        return me.cpp_object.vol_preconditioning

    @vol_preconditioning.setter
    def vol_preconditioning(me, new_vol_preconditioning_bool : bool):
        me.cpp_object.vol_preconditioning = new_vol_preconditioning_bool

    @property
    def gamma(me):
        return me.cpp_object.gamma

    @gamma.setter
    def gamma(me, new_gamma):
        me.cpp_object.gamma = new_gamma

    @property
    def tau_rows(me):
        return me.row_batches.tau

    @tau_rows.setter
    def tau_rows(me, new_tau):
        me.row_batches.tau = new_tau

    @property
    def tau_cols(me):
        return me.col_batches.tau

    @tau_cols.setter
    def tau_cols(me, new_tau):
        me.col_batches.tau = new_tau

    @property
    def num_neighbors_rows(me):
        return me.row_batches.num_neighbors

    @num_neighbors_rows.setter
    def num_neighbors_rows(me, new_num_neighbors):
        me.row_batches.num_neighbors = new_num_neighbors

    @property
    def num_neighbors_cols(me):
        return me.col_batches.num_neighbors

    @num_neighbors_cols.setter
    def num_neighbors_cols(me, new_num_neighbors):
        me.col_batches.num_neighbors = new_num_neighbors