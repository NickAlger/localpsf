import numpy as np
import dolfin as dl
import typing as typ
import scipy.sparse as sps
from dataclasses import dataclass
from functools import cached_property
from collections import namedtuple
import matplotlib.pyplot as plt

from .assertion_helpers import *
from .bilaplacian_regularization_lumped import BilaplacianRegularization
from .localpsf_cg1_lumped import PSFObjectFenicsWrapper, make_psf_fenics
from .newtoncg import newtoncg_ls
import hlibpro_python_wrapper as hpro


@dataclass(frozen=True)
class InverseProblemObjective:
    misfit: typ.Any
    regularization: BilaplacianRegularization

    def __post_init__(me):
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

    def misfit_cost(me) -> float:
        Jd: float = me.misfit.misfit()
        return Jd

    def regularization_cost(me, areg: float) -> float:
        Jr: float = me.regularization.cost(me.get_optimization_variable(), areg)
        return Jr

    def cost_triple(me, areg: float) -> typ.Tuple[float, float, float]:
        Jd: float = me.misfit_cost()
        Jr: float = me.regularization_cost(areg)
        J: float = Jd + Jr
        return J, Jd, Jr

    def misfit_gradient(me) -> np.ndarray:
        gd = me.misfit.gradient()
        assert_equal(gd.shape, (me.N,))
        return gd

    def regularization_gradient(me, areg: float) -> np.ndarray:
        gr = me.regularization.gradient(me.get_optimization_variable(), areg)
        assert_equal(gr.shape, (me.N,))
        return gr

    def gradient_triple(me, areg: float) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gd = me.misfit_gradient()
        gr = me.regularization_gradient(areg)
        g = gd + gr
        return g, gd, gr

    def apply_hessian_triple(me, p: np.ndarray, areg: float) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Hd_p = me.apply_misfit_hessian(p)
        Hr_p = me.apply_regularization_hessian(p, areg)
        H_p = Hd_p + Hr_p
        return H_p, Hd_p, Hr_p

    def apply_gauss_newton_hessian_triple(me, p: np.ndarray, areg: float) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Hdgn_p = me.apply_misfit_gauss_newton_hessian(p)
        Hr_p = me.apply_regularization_hessian(p, areg)
        Hgn_p = Hdgn_p + Hr_p
        return Hgn_p, Hdgn_p, Hr_p

    def cost(me, areg: float) -> float:
        return me.cost_triple(areg)[0]

    def gradient(me, areg: float) -> np.ndarray:
        return me.gradient_triple(areg)[0]

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

    def apply_regularization_hessian(me, p: np.ndarray, areg: float) -> np.ndarray:
        assert_equal(p.shape, (me.N,))
        Hr_p = me.regularization.apply_hessian(p, me.get_optimization_variable(), areg)
        assert_equal(Hr_p.shape, (me.N,))
        return Hr_p

    def solve_regularization_hessian(me, p: np.ndarray, areg: float) -> np.ndarray:
        assert_equal(p.shape, (me.N,))
        invHr_p = me.regularization.solve_hessian(p, me.get_optimization_variable(), areg)
        assert_equal(invHr_p.shape, (me.N,))
        return invHr_p

    def apply_hessian(me, p: np.ndarray, areg: float) -> np.ndarray:
        return me.apply_hessian_triple(p, areg)[0]

    def apply_gauss_newton_hessian(me, p: np.ndarray, areg: float) -> np.ndarray:
        return me.apply_gauss_newton_hessian_triple(p, areg)[0]


def finite_difference_check(
        IP: InverseProblemObjective,
        m0: np.ndarray,
        dm: np.ndarray,
        areg: float=1.0,
        make_plots: bool=True
) -> typ.Tuple[np.ndarray,  # ss
               np.ndarray,  # errs_grad_d
               np.ndarray,  # errs_hess_d
               np.ndarray,  # errs_grad_r
               np.ndarray,  # errs_hess_r
               np.ndarray,  # errs_grad
               np.ndarray]: # errs_hess
    assert_equal(m0.shape, (IP.N,))
    assert_equal(dm.shape, (IP.N,))

    old_m = IP.get_optimization_variable().copy()

    IP.set_optimization_variable(m0)
    J0, J0_d, J0_r = IP.cost_triple(areg)
    g0, g0_d, g0_r = IP.gradient_triple(areg)
    H0dm, H0dm_d, H0dm_r = IP.apply_hessian_triple(dm, areg)

    dJ_d = np.dot(g0_d, dm)
    dJ_r = np.dot(g0_r, dm)
    dJ = np.dot(g0, dm)

    # ss = [1e-6]
    ss = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    errs_grad_d = []
    errs_hess_d = []
    errs_grad_r = []
    errs_hess_r = []
    errs_grad = []
    errs_hess = []
    for s in ss:
        m1 = m0 + s * dm

        IP.set_optimization_variable(m1)
        J1, J1_d, J1_r = IP.cost_triple(areg)
        g1, g1_d, g1_r = IP.gradient_triple(areg)

        dJ_diff_d = (J1_d - J0_d) / s
        err_grad_d = np.abs(dJ_diff_d - dJ_d) / np.abs(dJ_diff_d)
        errs_grad_d.append(err_grad_d)

        dJ_diff_r = (J1_r - J0_r) / s
        err_grad_r = np.abs(dJ_diff_r - dJ_r) / np.abs(dJ_diff_r)
        errs_grad_r.append(err_grad_r)

        dJ_diff = (J1 - J0) / s
        err_grad = np.abs(dJ_diff - dJ) / np.abs(dJ_diff)
        errs_grad.append(err_grad)

        # Hessian

        dg_diff_d = (g1_d - g0_d) / s
        err_hess_d = np.linalg.norm(dg_diff_d - H0dm_d) / np.linalg.norm(dg_diff_d)
        errs_hess_d.append(err_hess_d)

        dg_diff_r = (g1_r - g0_r) / s
        err_hess_r = np.linalg.norm(dg_diff_r - H0dm_r) / np.linalg.norm(dg_diff_r)
        errs_hess_r.append(err_hess_r)

        dg_diff = (g1 - g0) / s
        err_hess = np.linalg.norm(dg_diff - H0dm) / np.linalg.norm(dg_diff)
        errs_hess.append(err_hess)

        print('s=', s)
        print('err_grad_d=', err_grad_d, ', err_grad_r=', err_grad_r, ', err_grad=', err_grad)
        print('err_hess_d=', err_hess_d, ', err_hess_r=', err_hess_r, ', err_hess=', err_hess)

    ss = np.array(ss)
    errs_grad_d = np.array(errs_grad_d)
    errs_hess_d = np.array(errs_hess_d)
    errs_grad_r = np.array(errs_grad_r)
    errs_hess_r = np.array(errs_hess_r)
    errs_grad = np.array(errs_grad)
    errs_hess = np.array(errs_hess)

    if make_plots:
        plt.figure()
        plt.loglog(ss, errs_grad_d)
        plt.loglog(ss, errs_hess_d)
        plt.title('finite difference check for misfit')
        plt.xlabel('step size s')
        plt.ylabel('error')
        plt.legend(['misfit gradient', 'misfit hessian'])

        plt.figure()
        plt.loglog(ss, errs_grad_r)
        plt.loglog(ss, errs_hess_r)
        plt.title('finite difference check for regularization')
        plt.xlabel('step size s')
        plt.ylabel('error')
        plt.legend(['regularization gradient', 'regularization hessian'])

        plt.figure()
        plt.loglog(ss, errs_grad)
        plt.loglog(ss, errs_hess)
        plt.title('finite difference check for overall cost')
        plt.xlabel('step size s')
        plt.ylabel('error')
        plt.legend(['gradient', 'hessian'])

    IP.set_optimization_variable(old_m)

    return ss, errs_grad_d, errs_hess_d, errs_grad_r, errs_hess_r, errs_grad, errs_hess


@dataclass
class PSFHessianPreconditioner:
    apply_misfit_gauss_newton_hessian: typ.Callable[[np.ndarray], np.ndarray]
    Vh: dl.FunctionSpace
    mass_lumps: np.ndarray
    HR_hmatrix: hpro.HMatrix

    bct: hpro.BlockClusterTree = None
    HR_hmatrix_factorized_inverse: hpro.FactorizedInverseHMatrix = None

    current_preconditioning_type: str='none' # current_preconditioning_type in {'none', 'reg', 'psf'}
    psf_options: typ.Dict[str, typ.Any] = None
    Hd_hmatrix_options: typ.Dict[str, typ.Any] = None
    shifted_inverse_options: typ.Dict[str, typ.Any] = None
    deflation_options: typ.Dict[str, typ.Any] = None
    display: bool = False
    refactor_distance: float = 5.0
    deflation_factor = 0.1 # Deflate until negative eigenvalues of regularization preconditioned Hessian are less than this in magnitude

    psf_object: PSFObjectFenicsWrapper = None
    Hd_kernel_hmatrix: hpro.HMatrix = None
    Hd_hmatrix_nonsym: hpro.HMatrix = None
    Hd_hmatrix: hpro.HMatrix=None
    shifted_inverse_interpolator: hpro.HMatrixShiftedInverseInterpolator=None

    def __post_init__(me):
        assert_equal(me.mass_lumps.shape, (me.Vh.dim(),))
        assert(np.all(me.mass_lumps > 0.0))

        me.psf_options = dict() if me.psf_options is None else me.psf_options
        me.Hd_hmatrix_options = dict() if me.Hd_hmatrix_options is None else me.Hd_hmatrix_options
        me.shifted_inverse_options = dict() if me.shifted_inverse_options is None else me.shifted_inverse_options
        me.deflation_options = dict() if me.deflation_options is None else me.deflation_options

        if me.bct is None:
            me.bct = me.HR_hmatrix.bct

        if me.HR_hmatrix_factorized_inverse is None:
            me.HR_hmatrix_factorized_inverse = me.HR_hmatrix.factorized_inverse()

    def solve_hessian_preconditioner(me, b: np.ndarray, areg: float) -> np.ndarray:
        if me.current_preconditioning_type.lower() == 'reg':
            return me.HR_hmatrix_factorized_inverse.matvec(b) / areg
        elif me.current_preconditioning_type.lower() == 'psf':
            me.update_deflation(areg)
            me.update_factorizations(areg)
            return me.shifted_inverse_interpolator.solve_shifted_deflated_preconditioner(
                b, areg, display=me.display)
        else:
            return b

    def update_deflation(me, areg: float) -> None:
        deflation_threshold = -areg * me.deflation_factor
        if deflation_threshold > me.shifted_inverse_interpolator.spectrum_lower_bound:
            me.shifted_inverse_interpolator.deflate_more(deflation_threshold, **me.deflation_options)

    def update_factorizations(me, areg: float):
        nearest_mu_factor = me.shifted_inverse_interpolator.nearest_mu_factor(areg)
        if nearest_mu_factor > me.refactor_distance:
            me.shifted_inverse_interpolator.insert_new_mu(areg)

    def build_hessian_preconditioner(me, use_psf_now: bool=True) -> None:
        if me.display:
            print('building psf object')
        me.psf_object = make_psf_fenics(
            me.apply_misfit_gauss_newton_hessian,
            me.apply_misfit_gauss_newton_hessian,
            me.Vh, me.Vh, me.mass_lumps, me.mass_lumps, **me.psf_options)

        if me.display:
            print('Building hmatrix')
        me.Hd_hmatrix_nonsym, me.Hdkernel_hmatrix = me.psf_object.construct_hmatrices(
            me.bct, **me.Hd_hmatrix_options)
        me.Hd_hmatrix = me.Hd_hmatrix_nonsym.sym()

        me.shifted_inverse_interpolator = hpro.HMatrixShiftedInverseInterpolator(
            me.Hd_hmatrix, me.HR_hmatrix, B_fac=me.HR_hmatrix_factorized_inverse,
            **me.shifted_inverse_options)

        if use_psf_now:
            me.current_preconditioning_type = 'psf'

