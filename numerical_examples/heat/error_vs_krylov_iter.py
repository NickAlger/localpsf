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


save_data = True
save_figures = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'error_vs_krylov_iter'
save_dir.mkdir(parents=True, exist_ok=True)


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 2.5e-2} # {'mesh_h': 2e-2}

tau = 2.5
all_num_batches = [1,3,6,9]
num_neighbors = 10
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
hmatrix_tol=1e-4


########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**nondefault_HIP_options)

########    CONSTRUCT KERNEL    ########

PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                               10, 10,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma)

########    CREATE HMATRICES    ########

A_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)

# _, eeA_max, _ = spla.svds(A_pch_nonsym.as_linear_operator(), k=3, which='LM')
# eA_max = np.max(eeA_max)

# cutoff = -1e-2*eA_max

# A_pch = A_pch_nonsym.spd(rtol=1e-3, atol=1e-5)
A_pch = A_pch_nonsym.spd()

# min_reg_param = 1e-3
# eeR_min, _ = spla.eigsh(min_reg_param * HIP.R0_scipy, k=3, which='SM')
# eR_min = np.min(eeR_min)
#
# A_pch = A_pch_nonsym.spd(cutoff=0.8*eR_min)


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

_, _, residuals_None, errors_None = custom_cg(HIP.H_linop, -g0_numpy,
                                              x_true=u0_numpy,
                                              tol=1e-10, maxiter=500)

_, _, residuals_Reg, errors_Reg = custom_cg(HIP.H_linop, -g0_numpy,
                                            x_true=u0_numpy,
                                            M=HIP.solve_R_linop,
                                            tol=1e-10, maxiter=1000)


all_residuals_PCH = list()
all_errors_PCH = list()
for num_batches in all_num_batches:
    print('num_batches=', num_batches)
    PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                                   num_batches, num_batches,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma)

    A_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)

    A_pch = A_pch_nonsym.spd()

    H_hmatrix = A_pch + a_reg_morozov * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    _, _, residuals_PCH, errors_PCH = custom_cg(HIP.H_linop, -g0_numpy,
                                                x_true=u0_numpy,
                                                M=iH_hmatrix.as_linear_operator(),
                                                tol=1e-10, maxiter=500)

    all_residuals_PCH.append(residuals_PCH)
    all_errors_PCH.append(errors_PCH)


########    SAVE DATA    ########

all_num_batches = np.array(all_num_batches)
residuals_None = np.array(residuals_None)
errors_None = np.array(errors_None)
residuals_Reg = np.array(residuals_Reg)
errors_Reg = np.array(errors_Reg)
all_residuals_PCH = [np.array(r) for r in all_residuals_PCH]
all_errors_PCH = [np.array(e) for e in all_errors_PCH]

if save_data:
    np.savetxt(save_dir / 'all_num_batches.txt', all_num_batches)
    np.savetxt(save_dir / 'residuals_None.txt', residuals_None)
    np.savetxt(save_dir / 'errors_None.txt', errors_None)
    np.savetxt(save_dir / 'residuals_Reg.txt', residuals_Reg)
    np.savetxt(save_dir / 'errors_Reg.txt', errors_Reg)
    for k in range(len(all_num_batches)):
        np.savetxt(save_dir / ('all_residuals_PCH' + str(all_num_batches[k]) + '.txt'), all_residuals_PCH[k])
        np.savetxt(save_dir / ('all_errors_PCH' + str(all_num_batches[k]) + '.txt'), all_errors_PCH[k])


########    MAKE FIGURES    ########

plt.figure()
plt.semilogy(errors_Reg)
plt.semilogy(errors_None)
legend = ['Reg', 'None']
for k in range(len(all_num_batches)):
    plt.semilogy(all_errors_PCH[k])
    legend.append('PCH ' + str(all_num_batches[k]))

plt.xlabel('Iteration')
plt.ylabel('relative l2 error')
plt.title('Relative error vs. Krylov iteration')
plt.legend(legend)

if save_figures:
    plt.savefig(save_dir / 'error_vs_krylov_iter.pdf', bbox_inches='tight', dpi=100)


plt.figure()
plt.semilogy(residuals_Reg)
plt.semilogy(residuals_None)
legend = ['Reg', 'None']
for k in range(len(all_num_batches)):
    plt.semilogy(all_residuals_PCH[k])
    legend.append('PCH ' + str(all_num_batches[k]))

plt.xlabel('Iteration')
plt.ylabel('relative residual')
plt.title('Relative residual vs. Krylov iteration')
plt.legend(legend)

if save_figures:
    plt.savefig(save_dir / 'relative_residual_vs_krylov_iter.pdf', bbox_inches='tight', dpi=100)

