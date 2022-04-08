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


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 2.5e-2} # {'mesh_h': 2e-2}

num_neighbors = 10
tau = 2.5 # 4
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid


save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'integration_time'
save_dir.mkdir(parents=True, exist_ok=True)


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 2.5e-2} # {'mesh_h': 2e-2}
all_integration_times = [1e-2, 1e-3, 1e-4]
num_neighbors = 10
tau = 2.5 # 4
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
hmatrix_rtol = 1e-4

err_tol = 1e-1
max_batches = 50
num_eigs0 = 100
max_eigs = 2000


########    SET UP HEAT INVERSE PROBLEM    ########

integration_time = all_integration_times[0]
HIP = HeatInverseProblem(final_time=integration_time, **nondefault_HIP_options)


########    LOW RANK APPROXIMATION    ########

print('low rank approximation')
spectral_drop = 1.0
num_eigs = num_eigs0
while num_eigs <= max_eigs:
    Hd_eigs, _ = spla.eigsh(HIP.Hd_linop, k=num_eigs, which='LM')
    Hd_eigs = np.sort(np.abs(Hd_eigs))[::-1]

    spectral_drop = Hd_eigs[-1] / Hd_eigs[0]
    print('num_eigs=', max_eigs, ', spectral_drop=', spectral_drop)
    if spectral_drop < err_tol:
        break
    else:
        num_eigs = 2*num_eigs

numerical_rank = np.argwhere(Hd_eigs / Hd_eigs[0] < err_tol)[0,0] + 1
print('integration_time=', integration_time, ', err_tol=', err_tol, ', numerical_rank=', numerical_rank)


########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########

PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                               0, 0,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma)

all_num_batches = list()
all_num_sample_points = list()
all_induced2_errors = list()
while PCK.col_batches.num_batches < max_batches:
    PCK.col_batches.add_one_sample_point_batch()
    all_num_batches.append(PCK.col_batches.num_batches)
    all_num_sample_points.append(PCK.col_batches.num_sample_points)

    ########    CREATE HMATRIX    ########

    A_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=False, hmatrix_tol=hmatrix_rtol)

    ########    ESTIMATE ERRORS IN INDUCED2 NORM    ########

    err_A_linop = spla.LinearOperator(A_pch.shape, matvec=lambda x: HIP.apply_Hd_numpy(x) - A_pch * x)

    ee, _ = spla.eigsh(err_A_linop, k=1, which='LM')

    relative_err_induced2 = np.max(np.abs(ee)) / np.max(np.abs(Hd_eigs))
    print('num_batches=', PCK.col_batches.num_batches,
          ', num_sample_points=', PCK.col_batches.num_sample_points,
          ', relative_err_induced2=', relative_err_induced2)

    all_induced2_errors.append(relative_err_induced2)

    if relative_err_induced2 < err_tol:
        break


required_num_batches = PCK.col_batches.num_batches
required_num_points = PCK.col_batches.num_sample_points
print('integration_time=', integration_time, ', err_tol=', err_tol, ', required_num_batches=', required_num_batches, ', required_num_points=', required_num_points)

plt.figure()
PCK.col_batches.visualize_impulse_response_batch(0)





















all_Hd_hmatrix = list()
all_extras = list()
for num_batches in all_batch_sizes:
    PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                                   num_batches, num_batches,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma)

    Hd_hmatrix, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=1e-3)

    all_Hd_hmatrix.append(Hd_hmatrix)
    all_extras.append(extras)


########    COMPUTE REGULARIZATION PARAMETER VIA MOROZOV DISCREPANCY PRINCIPLE    ########

best_Hd_hmatrix = all_Hd_hmatrix[-1]

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, best_Hd_hmatrix.bct)

def solve_inverse_problem(a_reg):
    HIP.regularization_parameter = a_reg

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    H_hmatrix = Hd_hmatrix + a_reg * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix.as_linear_operator(),
                                          tol=1e-10, maxiter=500)

    u0 = dl.Function(HIP.V)
    u0.vector()[:] = u0_numpy

    return u0.vector() # dolfin Vector

a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
                                                         HIP.morozov_discrepancy,
                                                         HIP.noise_Mnorm)

HIP.regularization_parameter = a_reg_morozov


########    COMPUTE PRECONDITIONED SPECTRUM    ########

print('PCH preconditioning')
all_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
all_abs_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
for kk_batch in range(len(all_batch_sizes)):
    print('batch size=', all_batch_sizes[kk_batch])
    Hd_hmatrix = all_Hd_hmatrix[kk_batch]
    H_hmatrix = Hd_hmatrix + a_reg_morozov * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()
    delta_hmatrix_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x: HIP.apply_H_numpy(x) - H_hmatrix * x)
    ee_hmatrix, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=H_hmatrix.as_linear_operator(),
                               Minv=iH_hmatrix.as_linear_operator(), which='LM')
    abs_ee_hmatrix = np.sort(np.abs(ee_hmatrix))[::-1]

    all_ee_hmatrix[kk_batch, :] = ee_hmatrix
    all_abs_ee_hmatrix[kk_batch, :] = abs_ee_hmatrix

print('reg preconditioning')
delta_reg_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x: HIP.apply_H_numpy(x) - HIP.apply_R_numpy(x))
ee_reg, _ = spla.eigsh(delta_reg_linop, k=num_eigs, M=HIP.R_linop, Minv=HIP.solve_R_linop, which='LM')
abs_ee_reg = np.sort(np.abs(ee_reg))[::-1]


########    SAVE DATA    ########

all_batch_sizes = np.array(all_batch_sizes)
abs_ee_hmatrix = np.array(abs_ee_hmatrix)
abs_ee_reg = np.array(abs_ee_reg)

if save_data:
    np.savetxt(save_dir / 'all_batch_sizes.txt', all_batch_sizes)
    np.savetxt(save_dir / 'abs_ee_hmatrix.txt', abs_ee_hmatrix)
    np.savetxt(save_dir / 'abs_ee_reg.txt', abs_ee_reg)


########    MAKE FIGURE    ########

plt.figure()
plt.semilogy(abs_ee_reg)
for abs_ee_hmatrix in all_abs_ee_hmatrix:
    plt.semilogy(abs_ee_hmatrix)

plt.title(r'Absolute values of eigenvalues of $P^{-1}H-I$')
plt.xlabel(r'$i$')
plt.ylabel(r'$|\lambda_i|$')
plt.legend(['Reg'] + ['PCH' + str(nB) for nB in all_batch_sizes])

plt.show()

if save_figures:
    plt.savefig(save_dir / 'preconditioned_spectrum.pdf', bbox_inches='tight', dpi=100)

