import numpy as np
import typing as typ
import scipy.sparse as sps
from dataclasses import dataclass
from functools import cached_property
from .product_convolution_kernel import ProductConvolutionKernel
import hlibpro_python_wrapper as hpro


@dataclass
class HessianPreconditioner:
    HR_lumped_scipy: sps.csr_matrix  # Regularization Hessian, using lumped mass, not including regularization parameter
    apply_misfit_gauss_newton_hessian: typ.Callable[[np.ndarray], np.ndarray] # Hgn(m)p = apply_misfit_gauss_newton_hessian(p)
    bct: hpro.BlockClusterTree
    mass_lumps: np.ndarray

    invHR_fac: hpro.FactorizedInverseHMatrix = None
    PCK: ProductConvolutionKernel = None
    Hd_hmatrix: hpro.HMatrix = None # data misfit Hessian
    shifted_inverse_interpolator: hpro.HMatrixShiftedInverseInterpolator = None

    a_reg_min: float = 1.0
    a_reg_max: float = 1.0
    use_regularization_preconditioning_initially = True
    hmatrix_rtol=1e-8

    def __post_init__(me):
        assert(me.a_reg_min <= me.a_reg_max)
        assert(me.HR_hmatrix.shape == (me.N, me.N))

        if me.invHR_fac is None:
            me.invHR_fac = me.HR_hmatrix.factorized_inverse(rtol=me.hmatrix_rtol)

    @cached_property
    def N(me):
        return len(me.HR_hmatrix.shape[0])



    def build_hessian_preconditioner(me):
        print('building PCH preconditioner')
        me.PCK = ProductConvolutionKernel(Vh2, Vh2,
                                          apply_misfit_gauss_newton_hessian_petsc,
                                          apply_misfit_gauss_newton_hessian_petsc,
                                          num_batches, num_batches,
                                          tau_rows=tau, tau_cols=tau,
                                          num_neighbors_rows=num_neighbors,
                                          num_neighbors_cols=num_neighbors)
        Hd_pch_nonsym, extras = make_hmatrix_from_kernel(me.PCK, hmatrix_tol=hmatrix_tol)
        me.Hd_pch = Hd_pch_nonsym.sym()

    def solve_hessian_preconditioner(me, b: np.ndarray, a_reg: float, display=False) -> np.ndarray:
        assert(a_reg > 0.0)
        assert(b.shape == (me.N,))
        if me.shifted_inverse_interpolator is not None:
            return me.shifted_inverse_interpolator.solve_shifted_deflated_preconditioner(b, a_reg, display=display)
        elif me.use_regularization_preconditioning_initially:
            return me.invHR_fac.matvec(b) / a_reg
        else:
            return b


