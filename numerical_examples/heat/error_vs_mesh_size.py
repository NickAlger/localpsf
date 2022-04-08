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

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'error_vs_mesh_size'
save_dir.mkdir(parents=True, exist_ok=True)


########    OPTIONS    ########

nondefault_HIP_options = dict()

mesh_hh =  np.logspace(np.log10(0.0075), np.log10(0.1), 7)[::-1] #[1e-1, 3e-2] # [1e-2]

tau = 2.5
num_neighbors = 10
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
hmatrix_tol=1e-4
num_random_error_matvecs = 50
error_cutoff = 1e-1

all_num_dofs = list()
required_num_sample_points = list()
required_num_batches = list()

for mesh_h in mesh_hh:
    print('mesh_h=', mesh_h)

    ########    SET UP HEAT INVERSE PROBLEM    ########

    HIP = HeatInverseProblem(mesh_h=mesh_h, **nondefault_HIP_options)

    all_num_dofs.append(HIP.V.dim())

    ########    CONSTRUCT KERNEL    ########

    PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                                   0, 0,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma)


    relative_err_fro = np.inf
    while relative_err_fro > error_cutoff:
        PCK.col_batches.add_one_sample_point_batch()



        ########    CREATE HMATRICES    ########

        Hd_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=False, hmatrix_tol=hmatrix_tol)


        ########    ESTIMATE ERRORS IN FROBENIUS NORM    ########

        _, _, _, relative_err_fro \
            = estimate_column_errors_randomized(HIP.apply_Hd_numpy,
                                                lambda x: Hd_pch * x,
                                                Hd_pch.shape[1],
                                                num_random_error_matvecs)
        print('mesh_h=', mesh_h, ', relative_err_fro=', relative_err_fro, ', num_batches=', PCK.col_batches.num_batches, ', num_sample_points=', PCK.col_batches.num_sample_points)

    required_num_sample_points.append(PCK.col_batches.num_sample_points)
    required_num_batches.append(PCK.col_batches.num_batches)


########    SAVE DATA    ########

mesh_hh = np.array(mesh_hh)
all_num_dofs = np.array(all_num_dofs)
required_num_batches = np.array(required_num_batches)
required_num_sample_points = np.array(required_num_sample_points)

if save_data:
    np.savetxt(save_dir / 'mesh_hh.txt', mesh_hh)
    np.savetxt(save_dir / 'all_num_dofs.txt', all_num_dofs)
    np.savetxt(save_dir / 'required_num_batches.txt', required_num_batches)
    np.savetxt(save_dir / 'required_num_sample_points.txt', required_num_sample_points)


########    MAKE FIGURES    ########

plt.figure()
plt.semilogx(mesh_hh, required_num_batches)
plt.semilogx(mesh_hh, required_num_batches, '.')
plt.xlabel(r'mesh size $h$')
plt.ylabel('number of batches')
plt.title('Number of batches to achieve relative error '+str(error_cutoff))
plt.show()

if save_figures:
    plt.savefig(save_dir / 'error_vs_mesh_size.pdf', bbox_inches='tight', dpi=100)

