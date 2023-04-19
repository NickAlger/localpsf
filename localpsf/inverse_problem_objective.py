import numpy as np
import typing as typ
import scipy.sparse as sps
from dataclasses import dataclass
from functools import cached_property

from .assertion_helpers import *
import hlibpro_python_wrapper as hpro

@dataclass
class InverseProblemObjective:
    misfit: typ.Any
    regularization: typ.Any
    regularization_parameter: float

    def __post_init__(me):
        f1: typ.Callable[[np.ndarray, float], float]                    = me.regularization.cost # Jr(m) = cost(m, a_reg)
        f2: typ.Callable[[np.ndarray, float], np.ndarray]               = me.regularization.gradient # gr(m) = gradient(m, a_reg)
        f3: typ.Callable[[np.ndarray, np.ndarray, float], np.ndarray]   = me.regularization.apply_hessian # Hr(m)p = apply_hessian(p, m, a_reg)
        f4: typ.Callable[[np.ndarray, np.ndarray, float], np.ndarray]   = me.regularization.solve_hessian  # invHr(m)p = solve_hessian(p, m, a_reg)
        f5: typ.Callable[[hpro.BlockClusterTree, float], hpro.HMatrix]  = me.regularization.make_hess_hmatrix

        f6: typ.Callable[[], np.ndarray]            = me.misfit.get_parameter # m = get_parameter()
        f7: typ.Callable[[np.ndarray], None]        = me.misfit.update_parameter # update_parameter(new_m)
        f8: typ.Callable[[], float]                 = me.misfit.misfit # Jd(m) = misfit()
        f9: typ.Callable[[], np.ndarray]            = me.misfit.gradient # gd(m) = gradient()
        f10: typ.Callable[[np.ndarray], np.ndarray]  = me.misfit.apply_hessian # Hd(m)p = apply_hessian(p)
        f11: typ.Callable[[np.ndarray], np.ndarray] = me.misfit.apply_gauss_newton_hessian # Hdgn(m)p = apply_gauss_newton_hessian(p)

    @cached_property
    def N(me):
        return len(me.misfit.get_parameter())

    def get_optimization_variable(me) -> np.ndarray:
        m: np.ndarray = me.misfit.get_parameter()
        assert_equal(m.shape, (me.N,))
        return m

    def set_optimization_variable(me, new_m: np.ndarray) -> None:
        assert_equal(new_m.shape, (me.N,))
        me.misfit.update_parameter(new_m)

    def update_regularization_parameter(me, new_a_reg) -> None:
        assert_gt(new_a_reg, 0.0)
        me.regularization_parameter = new_a_reg

    def misfit_cost(me) -> float:
        Jd: float = me.misfit.misfit()
        return Jd

    def regularization_cost(me) -> float:
        Jr: float = me.regularization.cost(me.get_optimization_variable(), me.regularization_parameter)
        return Jr

    def cost_triple(me) -> typ.Tuple[float, float, float]:
        Jd: float = me.misfit_cost()
        Jr: float = me.regularization_cost()
        J: float = Jd + Jr
        return J, Jd, Jr

    def misfit_gradient(me) -> np.ndarray:
        gd = me.misfit.gradient()
        assert_equal(gd.shape, (me.N,))
        return gd

    def regularization_gradient(me) -> np.ndarray:
        gr = me.regularization.gradient(me.get_optimization_variable(), me.regularization_parameter)
        assert_equal(gr.shape, (me.N,))
        return gr

    def gradient_triple(me) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gd = me.misfit_gradient()
        gr = me.regularization_gradient()
        g = gd + gr
        return g, gd, gr

    def apply_hessian_triple(me, p: np.ndarray) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Hd_p = me.apply_misfit_hessian(p)
        Hr_p = me.apply_regularization_hessian(p)
        H_p = Hd_p + Hr_p
        return H_p, Hd_p, Hr_p

    def apply_gauss_newton_hessian_triple(me, p: np.ndarray) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Hdgn_p = me.apply_misfit_gauss_newton_hessian(p)
        Hr_p = me.apply_regularization_hessian(p)
        Hgn_p = Hdgn_p + Hr_p
        return Hgn_p, Hdgn_p, Hr_p

    def cost(me) -> float:
        return me.cost_triple()[0]

    def gradient(me) -> np.ndarray:
        return me.gradient_triple()[0]

    def apply_misfit_hessian(me, p: np.ndarray) -> np.ndarray:
        assert_equal(p.shape, (me.N,))
        Hd_p = me.misfit.apply_hessian(p)
        assert_equal(Hd_p.shape, (me.N,))
        return Hd_p

    def apply_misfit_gauss_newton_hessian(me, p: np.ndarray) -> np.ndarray:
        assert_equal(p.shape, (me.N,))
        Hdgn_p = me.misfit.apply_gauss_newton_hessian(p)
        assert_equal(Hdgn_p.shape, (me.N,))
        return Hdgn_p

    def apply_regularization_hessian(me, p: np.ndarray) -> np.ndarray:
        assert_equal(p.shape, (me.N,))
        Hr_p = me.regularization.apply_hessian(p, me.get_optimization_variable(), me.regularization_parameter)
        assert_equal(Hr_p.shape, (me.N,))
        return Hr_p

    def solve_regularization_hessian(me, p: np.ndarray) -> np.ndarray:
        assert_equal(p.shape, (me.N,))
        invHr_p = me.regularization.solve_hessian(p, me.get_optimization_variable(), me.regularization_parameter)
        assert_equal(invHr_p.shape, (me.N,))
        return invHr_p

    def apply_hessian(me, p: np.ndarray) -> np.ndarray:
        return me.apply_hessian_triple(p)[0]

    def apply_gauss_newton_hessian(me, p: np.ndarray) -> np.ndarray:
        return me.apply_gauss_newton_hessian_triple(p)[0]

    def make_regularization_hessian_hmatrix(me, bct: hpro.BlockClusterTree):
        return me.regularization.make_hess_hmatrix(bct, me.regularization_parameter)


@dataclass
class InverseProblemPSFHessianPreconditioner:
    IP: InverseProblemObjective
    bct: hpro.BlockClusterTree
    areg_min: float # minimum range limit for regularization parameter
    areg_max: float # maximum range limit for regularization parameter

    Hd_hmatrix_options: typ.Dict[str, typ.Any]
    shifted_inverse_interpolator_options: typ.Dict[str, typ.Any]

    HR_hmatrix: hpro.HMatrix=None
    Hd_hmatrix: hpro.HMatrix=None
    shifted_inverse_interpolator: hpro.HMatrixShiftedInverseInterpolator=None

    def __post_init__(me):
        assert_gt(me.areg_min, 0.0)
        assert_gt(me.areg_max, me.areg_min)

    @cached_property
    def N(me) -> int:
        return me.IP.N

    def solve_hessian_preconditioner(me, b: np.ndarray, display=False) -> np.ndarray:
        if me.shifted_inverse_interpolator is None:
            return me.IP.regularization.solve_regularization_hessian(b)
        else:
            return me.shifted_inverse_interpolator.solve_shifted_deflated_preconditioner(
                b, me.IP.regularization_parameter, display=display)

    def build_HR_hmatrix(me) -> hpro.HMatrix:
        hpro.build_hmatrix_from_scipy_sparse_matrix(me.IP.regularization, me.bct)

    def build_hessian_preconditioner(me):
        if me.HR_hmatrix is None:
            me.HR_hmatrix = me.build_HR_hmatrix()

        me.Hd_hmatrix = me.build_Hd_hmatrix()

        me.shifted_inverse_interpolator = hpro.deflate_negative_eigs_then_make_shifted_hmatrix_inverse_interpolator(
            me.Hd_hmatrix, me.HR_hmatrix, me.areg_min, me.areg_max, **me.shifted_inverse_interpolator_options)


