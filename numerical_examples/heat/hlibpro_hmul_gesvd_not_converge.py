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
all_integration_times = [3e-2, 3e-3, 3e-4]
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
                               gamma=gamma,
                               sigma_min=sigma_min)

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



