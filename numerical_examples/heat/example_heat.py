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


########    CONSTRUCT KERNEL    ########

PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                               num_batches, num_batches,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma,
                               sigma_min=sigma_min)


########    VISUALIZE IMPULSE RESPONSE BATCHES    ########

PCK.col_batches.visualize_impulse_response_batch(0)


########    CREATE HMATRICES    ########

A_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=1e-3)

Phi_pch = extras['A_kernel_hmatrix']
A_pch_nonsym = extras['A_hmatrix_nonsym']


########    VISUALIZE HMATRICES    ########

Phi_pch.visualize('Phi_pch')
A_pch_nonsym.visualize('A_pch_nonsym')
A_pch.visualize('A_pch')


########    ESTIMATE ERRORS IN FROBENIUS NORM    ########

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


########    COLUMN ERROR PLOT    ########

column_error_plot(Phi_col_rel_errs_fro, PCK.V_in, PCK.col_batches.sample_points)


########    ESTIMATE ERRORS IN INDUCED2 NORM    ########

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


########    SET UP REGULARIZATION OPERATOR AND INVERSE PROBLEM SOLVER    ########

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


########    FIND REGULARIZATION PARAMETER VIA MOROZOV DISCREPANCY PRINCIPLE    ########

a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
                                                         HIP.morozov_discrepancy,
                                                         HIP.noise_Mnorm)


########    SOLVE INVERSE PROBLEM VIA CG WITH DIFFERENT PRECONDITIONERS    ########

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


########    COMPUTE PRECONDITIONED SPECTRUM    ########

num_eigs = 1000

delta_hmatrix_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x: HIP.apply_H_numpy(x) - H_hmatrix * x)
ee_hmatrix, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=H_hmatrix.as_linear_operator(),
                           Minv=iH_hmatrix.as_linear_operator(), which='LM')
abs_ee_hmatrix = np.sort(np.abs(ee_hmatrix))[::-1]

delta_reg_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x: HIP.apply_H_numpy(x) - HIP.apply_R_numpy(x))
ee_reg, _ = spla.eigsh(delta_reg_linop, k=num_eigs, M=HIP.R_linop, Minv=HIP.solve_R_linop, which='LM')
abs_ee_reg = np.sort(np.abs(ee_reg))[::-1]

plt.figure()
plt.semilogy(abs_ee_reg)
plt.semilogy(abs_ee_hmatrix)
plt.xlabel('k')
plt.ylabel(r'$\lambda_k$')
plt.title('preconditioned spectrum')
plt.legend(['Reg', 'PCH sym'])


########    PRECONDITIONED CONDITION NUMBER    ########

biggest_eig_pch = spla.eigsh(HIP.H_linop, k=1, M=H_hmatrix.as_linear_operator(),
                             Minv=iH_hmatrix.as_linear_operator(), which='LM')[0]

smallest_eig_pch = spla.eigsh(HIP.H_linop, k=1, M=H_hmatrix.as_linear_operator(),
                              Minv=iH_hmatrix.as_linear_operator(), which='SM')[0]

cond_pch = np.abs(biggest_eig_pch) / np.abs(smallest_eig_pch)
print('cond_pch=', cond_pch)

#

biggest_eig_none = spla.eigsh(HIP.H_linop, k=1, which='LM')[0]

smallest_eig_none = spla.eigsh(HIP.H_linop, k=1, which='SM')[0]

cond_none = np.abs(biggest_eig_none) / np.abs(smallest_eig_none)
print('cond_none=', cond_none)

#

biggest_eig_reg = spla.eigsh(HIP.H_linop, k=1, M=HIP.R_linop, Minv=HIP.solve_R_linop, which='LM')[0]

# smallest_eig_reg = spla.eigsh(HIP.H_linop, k=1, M=HIP.R_linop, Minv=HIP.solve_R_linop, which='SM')[0]
smallest_eig_reg = 1.0

cond_reg = np.abs(biggest_eig_reg) / np.abs(smallest_eig_reg)
print('cond_reg=', cond_reg)