import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix
from localpsf.visualization import column_error_plot
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter

import scipy.sparse.linalg as spla


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 3e-2} # {'mesh_h': 2e-2}

num_batches = 10
num_neighbors = 10
tau = 2.5 # 4
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
num_random_error_matvecs = 50


########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**nondefault_HIP_options)

#

PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                               num_batches, num_batches,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma,
                               sigma_min=sigma_min)

#

PCK.col_batches.visualize_impulse_response_batch(0)

#

A_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=1e-3)


Phi_pch = extras['A_kernel_hmatrix']
A_pch_nonsym = extras['A_hmatrix_nonsym']

#

Phi_pch.visualize('Phi_pch')
A_pch_nonsym.visualize('A_pch_nonsym')
A_pch.visualize('A_pch')

#

Phi_col_rel_errs_fro, Phi_col_norms_fro, Phi_col_errs_fro, Phi_relative_err_fro \
    = estimate_column_errors_randomized(HIP.apply_iM_Hd_iM_numpy,
                                        lambda x: Phi_pch * x,
                                        Phi_pch.shape[1],
                                        num_random_error_matvecs)

print('Phi_relative_err_fro=', Phi_relative_err_fro)

_, _, _, A_relative_err_fro \
    = estimate_column_errors_randomized(HIP.apply_Hd_numpy,
                                        lambda x: A_pch_nonsym * x,
                                        A_pch_nonsym.shape[1],
                                        num_random_error_matvecs)

print('A_relative_err_fro=', A_relative_err_fro)

_, _, _, Asym_relative_err_fro \
    = estimate_column_errors_randomized(HIP.apply_Hd_numpy,
                                        lambda x: A_pch * x,
                                        A_pch.shape[1],
                                        num_random_error_matvecs)

print('Asym_relative_err_fro=', Asym_relative_err_fro)

#

column_error_plot(Phi_col_rel_errs_fro, PCK.V_in, PCK.col_batches.sample_points)

#

Phi_linop = spla.LinearOperator(Phi_pch.shape, matvec=lambda x: HIP.apply_iM_Hd_iM_numpy(x))
err_Phi_linop = spla.LinearOperator(Phi_pch.shape, matvec=lambda x: HIP.apply_iM_Hd_iM_numpy(x) - Phi_pch*x)

aa_Phi,_ = spla.eigsh(Phi_linop, k=1, which='LM')
ee_Phi,_ = spla.eigsh(err_Phi_linop, k=1, which='LM')

Phi_relative_error_induced2 = np.max(np.abs(ee_Phi)) / np.max(np.abs(aa_Phi))
print('Phi_relative_error_induced2=', Phi_relative_error_induced2)

#

A_linop = spla.LinearOperator(Phi_pch.shape, matvec=lambda x: HIP.apply_Hd_numpy(x))
err_A_linop = spla.LinearOperator(Phi_pch.shape, matvec=lambda x: HIP.apply_Hd_numpy(x) - A_pch_nonsym*x)

aa_A,_ = spla.eigsh(A_linop, k=1, which='LM')
ee_A,_ = spla.eigsh(err_A_linop, k=1, which='LM')

A_relative_err_induced2 = np.max(np.abs(ee_A)) / np.max(np.abs(aa_A))
print('A_relative_err_induced2=', A_relative_err_induced2)

#

err_Asym_linop = spla.LinearOperator(Phi_pch.shape, matvec=lambda x: HIP.apply_Hd_numpy(x) - A_pch*x)

ee_Asym,_ = spla.eigsh(err_Asym_linop, k=1, which='LM')

Asym_relative_err_induced2 = np.max(np.abs(ee_Asym)) / np.max(np.abs(aa_A))
print('Asym_relative_err_induced2=', Asym_relative_err_induced2)

#

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, A_pch.bct)

def solve_inverse_problem(a_reg):
    HIP.regularization_parameter = a_reg

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    H_hmatrix = A_pch + a_reg * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix.as_linear_operator(),
                                          tol=1e-10, maxiter=500)

    u0 = dl.Function(HIP.V)
    u0.vector()[:] = u0_numpy

    return u0.vector()  # dolfin Vector


a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
                                                         HIP.morozov_discrepancy,
                                                         HIP.noise_Mnorm)

#

HIP.regularization_parameter = a_reg_morozov

g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
H_hmatrix = A_pch + a_reg_morozov * R0_hmatrix
iH_hmatrix = H_hmatrix.inv()

u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                      M=iH_hmatrix.as_linear_operator(),
                                      tol=1e-11, maxiter=500)

_, _, residuals_PCH, errors_PCH = custom_cg(HIP.H_linop, -g0_numpy,
                                            x_true=u0_numpy,
                                            M=iH_hmatrix.as_linear_operator(),
                                            tol=1e-10, maxiter=500)

_, _, residuals_None, errors_None = custom_cg(HIP.H_linop, -g0_numpy,
                                              x_true=u0_numpy,
                                              tol=1e-10, maxiter=500)

_, _, residuals_Reg, errors_Reg = custom_cg(HIP.H_linop, -g0_numpy,
                                            x_true=u0_numpy,
                                            M=HIP.solve_R_linop,
                                            tol=1e-10, maxiter=1000)

plt.figure()
plt.semilogy(errors_PCH)
plt.semilogy(errors_None)
plt.semilogy(errors_Reg)
plt.xlabel('Iteration')
plt.ylabel('relative l2 error')
plt.title('Relative error vs. Krylov iteration')
plt.legend(['PCH', 'None', 'Reg'])

#

