import numpy as np
import typing as typ
from dataclasses import dataclass
from functools import cached_property

from .assertion_helpers import *
# from .impulse_response_moments import impulse_response_moments_simplified
from .impulse_response_batches import ImpulseResponseBatches
from .mass_matrix import MassMatrixHelper

import hlibpro_python_wrapper as hpro


class ProductConvolutionKernel:
    def __init__(me, V_in, V_out, apply_A, apply_At, num_row_batches, num_col_batches,
                 tau_rows=3.0, tau_cols=3.0,
                 num_neighbors_rows=10, num_neighbors_cols=10,
                 symmetric=False, gamma=1e-4,
                 max_scale_discrepancy=1e5,
                 cols_only=True,
                 use_lumped_mass_moments=True,
                 use_lumped_mass_impulses=True):
        me.V_in = V_in
        me.V_out = V_out
        me.apply_A = apply_A
        me.apply_At = apply_At
        me.max_scale_discrepancy = max_scale_discrepancy
        me.cols_only = cols_only
        me.use_lumped_mass_moments = use_lumped_mass_moments
        me.use_lumped_mass_impulses = use_lumped_mass_impulses


        ####    SET UP MASS MATRIX STUFF    ####

        me.MMH_in = MassMatrixHelper(me.V_in)
        if symmetric:
            me.MMH_out = me.MMH_in
        else:
            me.MMH_out = MassMatrixHelper(me.V_out)

        if use_lumped_mass_moments:
            me.apply_M_in_moments = me.MMH_in.apply_ML_fenics
            me.solve_M_in_moments = me.MMH_in.solve_ML_fenics

            me.apply_M_out_moments = me.MMH_out.apply_ML_fenics
            me.solve_M_out_moments = me.MMH_out.solve_ML_fenics
        else:
            me.apply_M_in_moments = me.MMH_in.apply_M_fenics
            me.solve_M_in_moments = me.MMH_in.solve_M_fenics

            me.apply_M_out_moments = me.MMH_out.apply_M_fenics
            me.solve_M_out_moments = me.MMH_out.solve_M_fenics

        if use_lumped_mass_impulses:
            me.apply_M_in_impulses = me.MMH_in.apply_ML_fenics
            me.solve_M_in_impulses = me.MMH_in.solve_ML_fenics

            me.apply_M_out_impulses = me.MMH_out.apply_ML_fenics
            me.solve_M_out_impulses = me.MMH_out.solve_ML_fenics
        else:
            me.apply_M_in_impulses = me.MMH_in.apply_M_fenics
            me.solve_M_in_impulses = me.MMH_in.solve_M_fenics

            me.apply_M_out_impulses = me.MMH_out.apply_M_fenics
            me.solve_M_out_impulses = me.MMH_out.solve_M_fenics


        ####    COMPUTE IMPULSE RESPONSE BATCHES    ####

        me.col_batches = ImpulseResponseBatches(V_in, V_out,
                                                apply_A, apply_At,
                                                me.solve_M_in_moments,
                                                me.solve_M_in_impulses,
                                                me.solve_M_out_impulses,
                                                num_initial_batches=num_col_batches,
                                                tau=tau_cols,
                                                num_neighbors=num_neighbors_cols,
                                                max_scale_discrepancy=max_scale_discrepancy)

        me.col_coords = me.col_batches.dof_coords_in
        me.row_coords = me.col_batches.dof_coords_out
        me.shape = (me.V_out.dim(), me.V_in.dim())

        if me.cols_only:
            me.cpp_object = hpro.hpro_cpp.ProductConvolutionKernelRBFColsOnly(me.col_batches.cpp_object,
                                                                              me.col_coords,
                                                                              me.row_coords)
        else:
            if symmetric:
                me.row_batches = me.col_batches
            else:
                me.row_batches = ImpulseResponseBatches(V_out, V_in, apply_At, apply_A,
                                                        me.solve_M_out_moments,
                                                        me.solve_M_out_impulses,
                                                        me.solve_M_in_impulses,
                                                        num_initial_batches=num_row_batches,
                                                        tau=tau_rows,
                                                        num_neighbors=num_neighbors_rows,
                                                        max_scale_discrepancy=max_scale_discrepancy)

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
        if not me.cols_only:
            return me.cpp_object.gamma

    @gamma.setter
    def gamma(me, new_gamma):
        if not me.cols_only:
            me.cpp_object.gamma = new_gamma

    @property
    def tau_rows(me):
        if not me.cols_only:
            return me.row_batches.tau

    @tau_rows.setter
    def tau_rows(me, new_tau):
        if not me.cols_only:
            me.row_batches.tau = new_tau

    @property
    def tau_cols(me):
        return me.col_batches.tau

    @tau_cols.setter
    def tau_cols(me, new_tau):
        me.col_batches.tau = new_tau

    @property
    def num_neighbors_rows(me):
        if not me.cols_only:
            return me.row_batches.num_neighbors

    @num_neighbors_rows.setter
    def num_neighbors_rows(me, new_num_neighbors):
        if not me.cols_only:
            me.row_batches.num_neighbors = new_num_neighbors

    @property
    def num_neighbors_cols(me):
        return me.col_batches.num_neighbors

    @num_neighbors_cols.setter
    def num_neighbors_cols(me, new_num_neighbors):
        me.col_batches.num_neighbors = new_num_neighbors


