import numpy as np
from scipy.optimize import root_scalar
from typing import Callable



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