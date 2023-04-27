import numpy as np
import dolfin as dl
import typing as typ
from scipy.optimize import root_scalar
from typing import Callable

from .assertion_helpers import *
from .newtoncg import newtoncg_ls


def upward_geometric_search(
        y: float,
        f: typ.Callable[[float], float],
        x0: float,
        y0: float=None,
        scaling_factor: float=10.0,
        display: bool=False,
        maxiter: int=20
) -> typ.Tuple[float, float, float, float]: # x_lo, y_lo, x_hi, y_hi
    '''Increases x until the function f(x) goes from less thay y to greater than y
    I.e., y_lo = f(x_lo) < y <= f(x_hi) = y_hi
    Initial guess y0 = f(x0)
    Must have f(x0) < y.

    In:
        f = lambda x: x**2
        x = 203.1
        y = f(x)
        x0 = 0.1
        scaling_factor=5.0
        x_lo, y_lo, x_hi, y_hi = upward_geometric_search(y, f, x0, scaling_factor=scaling_factor, display=True)
        print('Result: (x_lo, y_lo)=', (x_lo, y_lo), ', (x, y)=', (x, y), ', (x_hi, y_hi)=', (x_hi, y_hi))
        print('x_lo < x_true=', x_lo < x_true)
        print('x_true <= x_hi=', x_true <= x_hi)
    Out:
        Upward geometric search. y= 41249.61 , (x0, y0)= (0.1, 0.010000000000000002)
        (x_lo, y_lo)= (0.1, 0.010000000000000002) , y= 41249.61 , (x_hi, y_hi)= (0.5, 0.25)
        (x_lo, y_lo)= (0.5, 0.25) , y= 41249.61 , (x_hi, y_hi)= (2.5, 6.25)
        (x_lo, y_lo)= (2.5, 6.25) , y= 41249.61 , (x_hi, y_hi)= (12.5, 156.25)
        (x_lo, y_lo)= (12.5, 156.25) , y= 41249.61 , (x_hi, y_hi)= (62.5, 3906.25)
        (x_lo, y_lo)= (62.5, 3906.25) , y= 41249.61 , (x_hi, y_hi)= (312.5, 97656.25)
        Result: (x_lo, y_lo)= (62.5, 3906.25) , (x, y)= (203.1, 41249.61) , (x_hi, y_hi)= (312.5, 97656.25)
        x_lo < x_true= True
        x_true <= x_hi= True
    '''
    assert(x0 > 0.0)
    y0 = f(x0) if y0 is None else y0
    assert(y0 < y)

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    printmaybe('Upward geometric search. y=', y, ', (x0, y0)=', (x0, y0))

    x_lo = x0
    y_lo = y0
    x_hi = x_lo
    y_hi = y_lo
    iter = 0
    while y_hi < y:
        iter += 1
        assert(iter <= maxiter)
        x_lo = x_hi
        y_lo = y_hi
        x_hi = x_hi * scaling_factor
        y_hi = f(x_hi)
        printmaybe('(x_lo, y_lo)=', (x_lo, y_lo), ', y=', y, ', (x_hi, y_hi)=', (x_hi, y_hi))

    return x_lo, y_lo, x_hi, y_hi

def downward_geometric_search(
        y: float,
        f: typ.Callable[[float], float],
        x0: float,
        y0: float=None,
        scaling_factor: float=10.0,
        display: bool=False,
        maxiter: int=20
) -> typ.Tuple[float, float, float, float]: # x_lo, y_lo, x_hi, y_hi
    '''Decreases x until the function f(x) goes from greater thay y to less than y
    I.e., y_lo = f(x_lo) < y <= f(x_hi) = y_hi
    Initial guess y0 = f(x0)
    Must have y <= g(x0).

    In:
        f = lambda x: x**2
        x = 0.341
        y = f(x)
        x0 = 100.0
        scaling_factor=5.0
        x_lo, y_lo, x_hi, y_hi = downward_geometric_search(y, f, x0, scaling_factor=scaling_factor, display=True)
        print('Result: (x_lo, y_lo)=', (x_lo, y_lo), ', (x, y)=', (x, y), ', (x_hi, y_hi)=', (x_hi, y_hi))
        print('x_lo < x_true=', x_lo < x_true)
        print('x_true <= x_hi=', x_true <= x_hi)
    Out:
        Downward geometric search. y= 0.11628100000000002 , (x0, y0)= (100.0, 10000.0)
        (x_lo, y_lo)= (20.0, 400.0) , y= 0.11628100000000002 , (x_hi, y_hi)= (100.0, 10000.0)
        (x_lo, y_lo)= (4.0, 16.0) , y= 0.11628100000000002 , (x_hi, y_hi)= (20.0, 400.0)
        (x_lo, y_lo)= (0.8, 0.6400000000000001) , y= 0.11628100000000002 , (x_hi, y_hi)= (4.0, 16.0)
        (x_lo, y_lo)= (0.16, 0.0256) , y= 0.11628100000000002 , (x_hi, y_hi)= (0.8, 0.6400000000000001)
        Result: (x_lo, y_lo)= (0.16, 0.0256) , (x, y)= (0.341, 0.11628100000000002) , (x_hi, y_hi)= (0.8, 0.6400000000000001)
        x_lo < x_true= True
        x_true <= x_hi= True
    '''
    assert(x0 > 0.0)
    y0 = f(x0) if y0 is None else y0
    assert(y <= y0)

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    printmaybe('Downward geometric search. y=', y, ', (x0, y0)=', (x0, y0))

    x_hi = x0
    y_hi = y0
    x_lo = x_hi
    y_lo = y_hi
    iter = 0
    while y <= y_lo:
        iter += 1
        assert(iter <= maxiter)
        x_hi = x_lo
        y_hi = y_lo
        x_lo = x_hi / scaling_factor
        y_lo = f(x_lo)
        printmaybe('(x_lo, y_lo)=', (x_lo, y_lo), ', y=', y, ', (x_hi, y_hi)=', (x_hi, y_hi))

    return x_lo, y_lo, x_hi, y_hi


def solve_linear_1d(y, x0, y0, x1, y1):
    '''solves for x in y = m * x + b'''
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m*x0
    assert(np.abs(m * x0 + b - y0) <= 1e-8 * np.abs(y0))
    assert(np.abs(m * x1 + b - y1) <= 1e-8 * np.abs(y1))
    x = (y - b) / m
    assert(np.abs(m * x + b - y) <= 1e-8 * np.abs(y))
    return x


def logarithmic_bracket_search(
        y: float,
        f: typ.Callable[[float], float],
        x_lo: float,
        x_hi: float,
        y_lo: float=None,
        y_hi: float=None,
        rtol: float=1e-3,
        maxiter: int=100,
        display: bool=False
) -> typ.Tuple[float, float]: # x, y
    '''Finds zero of function f(x) that goes from negative to positive in bracket.
    I.e., y_lo = f(x_lo) < y = f(x)  <= f(x_hi)
    and x_lo < x <= x_hi

    In:
        f = lambda x: x**2 + 3.2*x
        y = 100.0
        x_lo = 1.2
        x_hi = 35.4
        logarithmic_bracket_search(y, f, x_lo, x_hi, display=True)
    Out:
        Bracketing search.
        (x_lo, x_mid, x_hi)= (1.2, 7.199007624070433, 35.4)
        (y_lo, y_mid, y_hi)= (5.279999999999999, 74.8625351684496, 1366.4399999999998)
        y= 100.0
        (x_lo, x_mid, x_hi)= (7.199007624070433, 8.437774255585728, 35.4)
        (y_lo, y_mid, y_hi)= (74.8625351684496, 98.19691200609961, 1366.4399999999998)
        y= 100.0
        relerr= 0.01803087993900391
        (x_lo, x_mid, x_hi)= (8.437774255585728, 8.521805649426556, 35.4)
        (y_lo, y_mid, y_hi)= (98.19691200609961, 99.89094960476334, 1366.4399999999998)
        y= 100.0
        relerr= 0.001090503952366646
        (x_lo, x_mid, x_hi)= (8.521805649426556, 8.526869041929116, 35.4)
        (y_lo, y_mid, y_hi)= (99.89094960476334, 99.99347659238232, 1366.4399999999998)
        y= 100.0
        relerr= 6.52340761767789e-05
        Out[30]: (8.526869041929116, 99.99347659238232)
    '''
    y_lo = f(x_lo) if y_lo is None else y_lo
    assert(y_lo > 0.0)
    assert(y_lo < y)
    y_hi = f(x_hi) if y_hi is None else y_hi
    assert(y <= y_hi)
    assert(x_lo < x_hi)
    assert(rtol > 0.0)

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    printmaybe('Bracketing search.')
    x_mid = np.exp(solve_linear_1d(np.log(y),
                                   np.log(x_lo), np.log(y_lo),
                                   np.log(x_hi), np.log(y_hi)))
    y_mid = f(x_mid)
    printmaybe('(x_lo, x_mid, x_hi)=', (x_lo, x_mid, x_hi))
    printmaybe('(y_lo, y_mid, y_hi)=', (y_lo, y_mid, y_hi))
    printmaybe('y=', y)

    iter = 0
    while np.abs(y_mid - y) > rtol * np.abs(y):
        if y_mid < y:
            x_lo = x_mid
            y_lo = y_mid
        else:
            x_hi = x_mid
            y_hi = y_mid
        x_mid = np.exp(solve_linear_1d(np.log(y),
                                       np.log(x_lo), np.log(y_lo),
                                       np.log(x_hi), np.log(y_hi)))
        y_mid = f(x_mid)
        printmaybe('(x_lo, x_mid, x_hi)=', (x_lo, x_mid, x_hi))
        printmaybe('(y_lo, y_mid, y_hi)=', (y_lo, y_mid, y_hi))
        printmaybe('y=', y)
        relerr = np.abs(y_mid - y) / np.abs(y)
        printmaybe('relerr=', relerr)
        iter += 1
        if iter > maxiter:
            print('maximum iterations performed without achieving tolerance.')
            print('maxiter=', maxiter, ' rtol=', rtol, ', relerr=', relerr)
            break

    return x_mid, y_mid


def nonlinear_morozov(
        regularization_parameter_initial_guess: float,
        #
        get_optimization_variable:      typ.Callable[[], np.ndarray],
        set_optimization_variable:      typ.Callable[[np.ndarray], None],
        #
        set_regularization_parameter:   typ.Callable[[float], None],
        #
        cost_triple:                    typ.Callable[[], typ.Tuple[float, float, float]], # J, Jd, Jr
        gradient:                       typ.Callable[[], np.ndarray], # Gradient at current point
        apply_hessian:                  typ.Callable[[np.ndarray], np.ndarray], # p -> H@p
        apply_gauss_newton_hessian:     typ.Callable[[np.ndarray], np.ndarray], # p -> Hgn @ p
        #
        build_hessian_preconditioner:   typ.Callable[[], None],
        solve_hessian_preconditioner:   typ.Callable[[np.ndarray], np.ndarray],
        #
        noise_datanorm: float,
        gradnorm_ini: float=None,
        morozov_rtol: float = 1e-3,
        morozov_factor: float = 10.0,
        psf_build_iter: int = 3,
        newton_rtol: float = 1e-6,
        num_gn_first: int = 5,
        num_gn_rest: int = 2,
        display: bool=False,
) -> typ.Tuple[float,       # optimal morozov regularization parmaeter
               np.ndarray,  # all regularization parameters
               np.ndarray]: # all morozov discrepancies
    gradnorm_ini = np.linalg.norm(gradient()) if gradnorm_ini is None else gradnorm_ini

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    areg_L = [regularization_parameter_initial_guess]

    def get_areg() -> float:
        return areg_L[0]

    def set_areg(new_areg: float) -> None:
        areg_L[0] = new_areg
        set_regularization_parameter(new_areg)

    preconditioner_build_iters_L: typ.List[typ.Tuple[int, ...]] = [tuple([])]
    if psf_build_iter is not None:
        preconditioner_build_iters_L[0] = (psf_build_iter,)

    num_gn_iter_L: typ.List[int] = [num_gn_first]

    def ncg_solve() -> None:
        num_gn_iter = num_gn_iter_L[0]
        preconditioner_build_iters = preconditioner_build_iters_L[0]
        printmaybe('preconditioner_build_iters=', preconditioner_build_iters)
        printmaybe('solving optimization problem with a_reg=', get_areg())
        newtoncg_ls(
            get_optimization_variable,
            set_optimization_variable,
            cost_triple,
            gradient,
            apply_hessian,
            apply_gauss_newton_hessian,
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

    def compute_morozov_discrepancy() -> float: # misfit_datanorm
        ncg_solve()
        Jd = cost_triple()[1]
        misfit_datanorm = np.sqrt(2.0 * Jd)
        areg = get_areg()
        printmaybe(
            'areg=', areg,
            ', noise_datanorm=', noise_datanorm,
            ', misfit_datanorm=', misfit_datanorm)
        morozov_aregs.append(areg)
        morozov_discrepancies.append(misfit_datanorm)
        return misfit_datanorm

    print('Initial Guess.')
    areg = get_areg()
    misfit_datanorm = compute_morozov_discrepancy()
    if np.abs(misfit_datanorm - noise_datanorm) <= morozov_rtol * np.abs(noise_datanorm):
        return get_areg(), np.array(morozov_aregs), np.array(morozov_discrepancies)

    num_gn_iter_L[0] = num_gn_rest
    preconditioner_build_iters_L[0] = tuple([])

    def f(x: float) -> float:
        set_areg(x)
        return compute_morozov_discrepancy()

    if noise_datanorm < misfit_datanorm:
        printmaybe('Initial a_reg too big. decreasing via geometric search')
        bracket_min, misfit_min, bracket_max, misfit_max = downward_geometric_search(
            noise_datanorm, f, areg, y0=misfit_datanorm, scaling_factor=morozov_factor, display=display)
        areg = bracket_min
        misfit_datanorm = misfit_min
    else:
        printmaybe('Initial a_reg too small. increasing via geometric search')
        bracket_min, misfit_min, bracket_max, misfit_max = upward_geometric_search(
            noise_datanorm, f, areg, y0=misfit_datanorm, scaling_factor=morozov_factor, display=display)
        areg = bracket_max
        misfit_datanorm = misfit_max

    if np.abs(misfit_datanorm - noise_datanorm) <= morozov_rtol * np.abs(noise_datanorm):
        return areg, np.array(morozov_aregs), np.array(morozov_discrepancies)

    printmaybe('Bracketing search.')
    areg, misfit_datanorm = logarithmic_bracket_search(
        noise_datanorm, f, bracket_min, bracket_max,
        y_lo=misfit_min, y_hi=misfit_max,
        rtol=morozov_rtol, display=display)

    return areg, np.array(morozov_aregs), np.array(morozov_discrepancies)


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