import numpy as np
import dolfin as dl
import typing as typ
import scipy.sparse as sps
from dataclasses import dataclass
from functools import cached_property
from collections import namedtuple

from .assertion_helpers import *
from .bilaplacian_regularization_lumped import BilaplacianRegularization
from .localpsf_cg1_lumped import PSFObjectFenicsWrapper, make_psf_fenics
from .newtoncg import newtoncg_ls
import hlibpro_python_wrapper as hpro

@dataclass
class InverseProblemObjective:
    misfit: typ.Any
    regularization: BilaplacianRegularization
    regularization_parameter: float

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


@dataclass
class InverseProblemPSFHessianPreconditioner:
    IP: InverseProblemObjective
    Vh: dl.FunctionSpace
    mass_lumps: np.ndarray
    bct: hpro.BlockClusterTree
    areg_min: float # minimum range limit for regularization parameter
    areg_max: float # maximum range limit for regularization parameter

    use_regularization_preconditioning_initially: bool=False
    psf_options: typ.Dict[str, typ.Any] = None
    Hd_hmatrix_options: typ.Dict[str, typ.Any] = None
    shifted_inverse_interpolator_options: typ.Dict[str, typ.Any] = None

    psf_object: PSFObjectFenicsWrapper=None
    Hd_kernel_hmatrix: hpro.HMatrix = None
    Hd_hmatrix_nonsym: hpro.HMatrix = None
    Hd_hmatrix: hpro.HMatrix=None
    shifted_inverse_interpolator: hpro.HMatrixShiftedInverseInterpolator=None

    def __post_init__(me):
        assert_gt(me.areg_min, 0.0)
        assert_ge(me.areg_max, me.areg_min)
        me.psf_options = dict() if me.psf_options is None else me.psf_options
        me.Hd_hmatrix_options = dict() if me.Hd_hmatrix_options is None else me.Hd_hmatrix_options
        me.shifted_inverse_interpolator_options = dict() if me.shifted_inverse_interpolator_options is None else me.shifted_inverse_interpolator_options

    @cached_property
    def N(me) -> int:
        return me.IP.N

    def solve_hessian_preconditioner(me, b: np.ndarray, display=False) -> np.ndarray:
        if me.shifted_inverse_interpolator is None:
            if me.use_regularization_preconditioning_initially:
                return me.IP.solve_regularization_hessian(b)
            else:
                return b
        else:
            return me.shifted_inverse_interpolator.solve_shifted_deflated_preconditioner(
                b, me.IP.regularization_parameter, display=display)

    @cached_property
    def HR0_hmatrix(me) -> hpro.HMatrix:
        return me.IP.regularization.Cov.make_invC_hmatrix(me.bct, 1.0) # areg0=gamma0=1.0

    def build_hessian_preconditioner(me):
        me.psf_object = make_psf_fenics(
            me.IP.apply_misfit_gauss_newton_hessian,
            me.IP.apply_misfit_gauss_newton_hessian,
            me.Vh, me.Vh, me.mass_lumps, me.mass_lumps, **me.psf_options)

        print('Building hmatrix')
        me.Hd_hmatrix_nonsym, me.Hdkernel_hmatrix = me.psf_object.construct_hmatrices(me.bct, **me.Hd_hmatrix_options)

        me.Hd_hmatrix = me.Hd_hmatrix_nonsym.sym()

        me.shifted_inverse_interpolator = hpro.deflate_negative_eigs_then_make_shifted_hmatrix_inverse_interpolator(
            me.Hd_hmatrix, me.HR0_hmatrix,
            me.areg_min, me.areg_max,
            **me.shifted_inverse_interpolator_options)


def nonlinear_morozov_psf(
        IP: InverseProblemObjective,
        noise_datanorm: float,
        Vh: dl.FunctionSpace,
        mass_lumps: np.ndarray,
        bct: hpro.BlockClusterTree,
        morozov_rtol: float = 1e-2,
        morozov_factor: float = 5.0,
        psf_build_iter: int = 3,
        newton_rtol: float = 1e-6,
        num_gn_first: int = 5,
        num_gn_rest: int = 2,
        shifted_inverse_options: typ.Dict[str, typ.Any] = None,
        psf_options: typ.Dict[str, typ.Any] = None,
        deflation_options: typ.Dict[str, typ.Any] = None,
        Hd_hmatrix_options: typ.Dict[str, typ.Any] = None,
) -> typ.Tuple[float,  # optimal morozov regularization parmaeter
               typ.List[float],  # all regularization parameters
               typ.List[float],  # all morozov discrepancies
               hpro.HMatrixShiftedInverseInterpolator,
               PSFObjectFenicsWrapper]:
    shifted_inverse_options = dict() if shifted_inverse_options is None else shifted_inverse_options
    psf_options = dict() if psf_options is None else psf_options
    Hd_hmatrix_options = dict() if Hd_hmatrix_options is None else Hd_hmatrix_options
    deflation_options = dict() if deflation_options is None else deflation_options

    gradnorm_ini = np.linalg.norm(IP.gradient())

    preconditioner_build_iters_L: typ.List[typ.Tuple[int, ...]] = [tuple([])]
    if psf_build_iter is not None:
        preconditioner_build_iters_L[0] = (psf_build_iter,)

    HR_hmatrix: hpro.HMatrix = IP.regularization.Cov.make_invC_hmatrix(bct, 1.0)  # areg0=gamma0=1.0

    psf_object_L: typ.List[PSFObjectFenicsWrapper] = [None]
    HSII_L: typ.List[hpro.HMatrixShiftedInverseInterpolator] = [None]

    def build_hessian_preconditioner() -> None:
        print('building psf object')
        psf_object = make_psf_fenics(
            IP.apply_misfit_gauss_newton_hessian,
            IP.apply_misfit_gauss_newton_hessian,
            Vh, Vh, mass_lumps, mass_lumps, **psf_options)

        print('Building hmatrix')
        Hd_hmatrix_nonsym, Hdkernel_hmatrix = psf_object.construct_hmatrices(bct, **Hd_hmatrix_options)
        Hd_hmatrix = Hd_hmatrix_nonsym.sym()

        HSII = hpro.HMatrixShiftedInverseInterpolator(Hd_hmatrix, HR_hmatrix, **shifted_inverse_options)
        HSII.insert_new_mu(IP.regularization_parameter)
        HSII.deflate_more(-IP.regularization_parameter * 0.1, **deflation_options)

        psf_object_L[0] = psf_object
        HSII_L[0] = HSII

    def solve_hessian_preconditioner(b: np.ndarray) -> np.ndarray:
        HSII = HSII_L[0]
        if HSII is not None:
            return HSII.solve_shifted_deflated_preconditioner(
                b, IP.regularization_parameter)
        else:
            return b

    num_gn_iter_L: typ.List[int] = [num_gn_first]

    def ncg_solve() -> None:
        num_gn_iter = num_gn_iter_L[0]
        preconditioner_build_iters = preconditioner_build_iters_L[0]
        print('preconditioner_build_iters=', preconditioner_build_iters)
        print('solving optimization problem with a_reg=', IP.regularization_parameter)
        newtoncg_ls(
            IP.get_optimization_variable,
            IP.set_optimization_variable,
            IP.cost_triple,
            IP.gradient,
            IP.apply_hessian,
            IP.apply_gauss_newton_hessian,
            build_hessian_preconditioner,
            lambda X, Y: None,
            solve_hessian_preconditioner,
            preconditioner_build_iters=preconditioner_build_iters,
            rtol=newton_rtol,
            forcing_sequence_power=1.0,
            num_gn_iter=num_gn_iter,
            gradnorm_ini=gradnorm_ini)

    morozov_aregs: typ.List[float] = list()
    morozov_discrepancies: typ.List[float] = list()

    def get_morozov_discrepancy():
        ncg_solve()
        misfit_datanorm = np.sqrt(2.0 * IP.misfit_cost())
        print('areg=', IP.regularization_parameter,
              ', noise_datanorm=', noise_datanorm,
              ', misfit_datanorm=', misfit_datanorm)
        morozov_aregs.append(IP.regularization_parameter)
        morozov_discrepancies.append(misfit_datanorm)
        return misfit_datanorm

    print('Initial guess.')
    misfit_datanorm = get_morozov_discrepancy()
    if np.abs(misfit_datanorm - noise_datanorm) <= morozov_rtol * np.abs(noise_datanorm):
        return IP.regularization_parameter, morozov_aregs, morozov_discrepancies, HSII_L[0], psf_object_L[0]

    num_gn_iter_L[0] = num_gn_rest
    preconditioner_build_iters_L[0] = tuple([])

    if noise_datanorm < misfit_datanorm:
        print('initial a_reg too big. decreasing via geometric search')
        while noise_datanorm < misfit_datanorm:
            bracket_max = IP.regularization_parameter
            misfit_max = misfit_datanorm
            IP.update_regularization_parameter(bracket_max / morozov_factor)
            HSII_L[0].insert_new_mu(IP.regularization_parameter)
            HSII_L[0].deflate_more(-IP.regularization_parameter * 0.1, **deflation_options)
            misfit_datanorm = get_morozov_discrepancy()
        bracket_min = IP.regularization_parameter
        misfit_min = misfit_datanorm
    else:
        print('initial a_reg too small. increasing via geometric search')
        while noise_datanorm >= misfit_datanorm:
            bracket_min = IP.regularization_parameter
            misfit_min = misfit_datanorm
            IP.update_regularization_parameter(bracket_min * morozov_factor)
            HSII_L[0].insert_new_mu(IP.regularization_parameter)
            misfit_datanorm = get_morozov_discrepancy()
        bracket_max = IP.regularization_parameter
        misfit_max = misfit_datanorm

    if np.abs(misfit_min - noise_datanorm) <= morozov_rtol * np.abs(noise_datanorm):
        return bracket_min, morozov_aregs, morozov_discrepancies, HSII_L[0], psf_object_L[0]

    if np.abs(misfit_max - noise_datanorm) <= morozov_rtol * np.abs(noise_datanorm):
        return bracket_max, morozov_aregs, morozov_discrepancies, HSII_L[0], psf_object_L[0]

    def solve_linear_1d(y, x0, y0, x1, y1):
        '''solves for x in y = m * x + b'''
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m*x0
        assert(np.abs(m * x0 + b - y0) <= 1e-8 * np.abs(y0))
        assert(np.abs(m * x1 + b - y1) <= 1e-8 * np.abs(y1))
        x = (y - b) / m
        assert(np.abs(m * x + b - y) <= 1e-8 * np.abs(y))
        return x

    print('Bracketing search.')
    bracket_mid = np.exp(solve_linear_1d(np.log(noise_datanorm),
                                         np.log(bracket_min), np.log(misfit_min),
                                         np.log(bracket_max), np.log(misfit_max)))
    IP.update_regularization_parameter(bracket_mid)
    misfit_mid = get_morozov_discrepancy()
    print('(bracket_min, bracket_mid, bracket_max)=', (bracket_min, bracket_mid, bracket_max))
    print('(misfit_min, misfit_mid, misfit_max)=', (misfit_min, misfit_mid, misfit_max))
    while np.abs(misfit_mid - noise_datanorm) > morozov_rtol * np.abs(noise_datanorm):
        if misfit_mid < noise_datanorm:
            misfit_min = misfit_mid
        else:
            misfit_max = misfit_mid
        bracket_mid = np.exp(solve_linear_1d(np.log(noise_datanorm),
                                             (np.log(bracket_min), np.log(misfit_min)),
                                             (np.log(bracket_max), np.log(misfit_max))))
        IP.update_regularization_parameter(bracket_mid)
        misfit_mid = get_morozov_discrepancy()
        print('(bracket_min, bracket_mid, bracket_max)=', (bracket_min, bracket_mid, bracket_max))
        print('(misfit_min, misfit_mid, misfit_max)=', (misfit_min, misfit_mid, misfit_max))

    HSII_L[0].insert_new_mu(bracket_mid)
    return bracket_mid, morozov_aregs, morozov_discrepancies, HSII_L[0], psf_object_L[0]