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

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'error_vs_num_batches'
save_dir.mkdir(parents=True, exist_ok=True)


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 2.5e-2} # {'mesh_h': 2e-2}

all_tau = [1.0, 2.0, 3.0, 4.0, 5.0]
num_neighbors = 10
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
hmatrix_tol=1e-4
num_random_error_matvecs = 50


########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**nondefault_HIP_options)


########    CONSTRUCT KERNEL    ########

all_num_sample_points = list()
all_num_batches = list()
all_fro_errors = list()
all_induced2_errors = list()

for tau in all_tau:
    print('tau=', tau)
    all_num_sample_points.append(list())
    all_num_batches.append(list())
    all_fro_errors.append(list())
    all_induced2_errors.append(list())

    PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                                   0, 0,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma,
                                   sigma_min=sigma_min)

    while PCK.col_batches.num_sample_points < HIP.V.dim():
        PCK.col_batches.add_one_sample_point_batch()
        all_num_batches[-1].append(PCK.col_batches.num_batches)
        all_num_sample_points[-1].append(PCK.col_batches.num_sample_points)
        print('tau=', tau, ', num_batches=', PCK.col_batches.num_batches, ', num_sample_points=', PCK.col_batches.num_sample_points)


        ########    CREATE HMATRICES    ########

        A_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=False, hmatrix_tol=hmatrix_tol)


        ########    ESTIMATE ERRORS IN FROBENIUS NORM    ########

        _, _, _, relative_err_fro \
            = estimate_column_errors_randomized(HIP.apply_Hd_numpy,
                                                lambda x: A_pch * x,
                                                A_pch.shape[1],
                                                num_random_error_matvecs)

        print('relative_err_fro=', relative_err_fro)

        all_fro_errors[-1].append(relative_err_fro)


        ########    ESTIMATE ERRORS IN INDUCED2 NORM    ########

        A_linop = spla.LinearOperator(A_pch.shape, matvec=lambda x: HIP.apply_Hd_numpy(x))
        err_A_linop = spla.LinearOperator(A_pch.shape, matvec=lambda x: HIP.apply_Hd_numpy(x) - A_pch * x)

        aa, _ = spla.eigsh(A_linop, k=1, which='LM')
        ee, _ = spla.eigsh(err_A_linop, k=1, which='LM')

        relative_err_induced2 = np.max(np.abs(ee)) / np.max(np.abs(aa))
        print('relative_err_induced2=', relative_err_induced2)

        all_induced2_errors[-1].append(relative_err_induced2)


########    SAVE DATA    ########

all_num_sample_points = [np.array(x) for x in all_num_sample_points]
all_num_batches = [np.array(x) for x in all_num_batches]
all_fro_errors = [np.array(x) for x in all_fro_errors]
all_induced2_errors = [np.array(x) for x in all_induced2_errors]

if save_data:
    np.savetxt(save_dir / 'all_tau.txt', all_tau)
    for k in range(len(all_tau)):
        np.savetxt(save_dir / ('num_sample_points_'+str(k)+'.txt'), all_num_sample_points[k])
        np.savetxt(save_dir / ('num_batches_' + str(k) + '.txt'), all_num_batches[k])
        np.savetxt(save_dir / ('fro_errors_' + str(k) + '.txt'), all_fro_errors[k])
        np.savetxt(save_dir / ('induced2_errors_' + str(k) + '.txt'), all_induced2_errors[k])


########    MAKE FIGURES    ########

plt.figure()
legend = list()
for k in range(len(all_tau)):
    plt.semilogy(all_num_batches[k], all_induced2_errors[k])
    legend.append('tau='+str(all_tau[k]))

plt.title(r'Relative error (induced-2 norm) vs. number of batches')
plt.xlabel(r'Number of batches')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||_2}{||H_d||_2}$')
plt.legend(legend)
plt.show()

if save_figures:
    plt.savefig(save_dir / 'error_vs_num_batches_induced2.pdf', bbox_inches='tight', dpi=100)

#

plt.figure()
legend = list()
for k in range(len(all_tau)):
    plt.semilogy(all_num_batches[k], all_fro_errors[k])
    legend.append('tau=' + str(all_tau[k]))

plt.title(r'Relative error (Frobenius norm) vs. number of batches')
plt.xlabel(r'Number of batches')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||_\mathrm{Fro}}{||H_d||_\mathrm{Fro}}$')
plt.legend(legend)
plt.show()

if save_figures:
    plt.savefig(save_dir / 'error_vs_num_batches_fro.pdf', bbox_inches='tight', dpi=100)

#

plt.figure()
legend = list()
for k in range(len(all_tau)):
    plt.semilogy(all_num_sample_points[k], all_induced2_errors[k])
    legend.append('tau='+str(all_tau[k]))

plt.title(r'Relative error (induced-2 norm) vs. number of sample points')
plt.xlabel(r'Number of sample points')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||_2}{||H_d||_2}$')
plt.legend(legend)
plt.show()

if save_figures:
    plt.savefig(save_dir / 'error_vs_num_sample_points_induced2.pdf', bbox_inches='tight', dpi=100)

#

plt.figure()
legend = list()
for k in range(len(all_tau)):
    plt.semilogy(all_num_sample_points[k], all_fro_errors[k])
    legend.append('tau='+str(all_tau[k]))

plt.title(r'Relative error (Frobenius norm) vs. number of sample points')
plt.xlabel(r'Number of sample points')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||_\mathrm{Fro}}{||H_d||_\mathrm{Fro}}$')
plt.legend(legend)
plt.show()

if save_figures:
    plt.savefig(save_dir / 'error_vs_num_sample_points_fro.pdf', bbox_inches='tight', dpi=100)

#

