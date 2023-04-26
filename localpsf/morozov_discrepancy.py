import numpy as np
import dolfin as dl
import typing as typ
from scipy.optimize import root_scalar
from typing import Callable

from .assertion_helpers import *
from .inverse_problem_objective import InverseProblemObjective
from .localpsf_cg1_lumped import PSFObjectFenicsWrapper, make_psf_fenics
from .newtoncg import newtoncg_ls
import hlibpro_python_wrapper as hpro


def nonlinear_morozov_psf(
        IP: InverseProblemObjective,
        noise_datanorm: float,
        Vh: dl.FunctionSpace,
        mass_lumps: np.ndarray,
        bct: hpro.BlockClusterTree,
        gradnorm_ini: float=None,
        morozov_rtol: float = 1e-3,
        morozov_factor: float = 10.0,
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

    gradnorm_ini = np.linalg.norm(IP.gradient()) if gradnorm_ini is None else gradnorm_ini

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
            bracket_min = bracket_mid
            misfit_min = misfit_mid
        else:
            bracket_max = bracket_mid
            misfit_max = misfit_mid
        bracket_mid = np.exp(solve_linear_1d(np.log(noise_datanorm),
                                             np.log(bracket_min), np.log(misfit_min),
                                             np.log(bracket_max), np.log(misfit_max)))
        print('bracket_mid=', bracket_mid)
        IP.update_regularization_parameter(bracket_mid)
        misfit_mid = get_morozov_discrepancy()
        print('(bracket_min, bracket_mid, bracket_max)=', (bracket_min, bracket_mid, bracket_max))
        print('(misfit_min, misfit_mid, misfit_max)=', (misfit_min, misfit_mid, misfit_max))

    HSII_L[0].insert_new_mu(bracket_mid)
    return bracket_mid, morozov_aregs, morozov_discrepancies, HSII_L[0], psf_object_L[0]


######## OLD BELOW HERE ########

def compute_morozov_regularization_parameter(solve_inverse_problem : Callable,
                                             compute_morozov_discrepancy : Callable,
                                             noise_level,
                                             a_reg_min=1e-5, a_reg_max=1e0,
                                             rtol=1e-2):
    def f(log_a_reg):
        u = solve_inverse_problem(np.exp(log_a_reg))
        discrepancy = compute_morozov_discrepancy(u)
        print('a_reg=', np.exp(log_a_reg), ', morozov discrepancy=', discrepancy, ', noise level=', noise_level)
        log_residual = np.log(discrepancy) - np.log(noise_level)
        return log_residual

    print('Computing regularization parameter via Morozov discrepancy principle')
    sol = root_scalar(f, bracket=[np.log(a_reg_min), np.log(a_reg_max)], rtol=rtol)
    a_reg_morozov = np.exp(sol.root)
    print('a_reg_morozov=', a_reg_morozov)
    return a_reg_morozov