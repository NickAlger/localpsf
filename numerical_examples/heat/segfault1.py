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

# mesh_hh = np.logspace(-2, -1, 5)[::-1] #[1e-1, 3e-2]

tau = 2.5
num_neighbors = 10
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
hmatrix_tol=1e-4
num_random_error_matvecs = 50
error_cutoff = 1e-1


mesh_h = 1e-2
num_batches = 2

print('mesh_h=', mesh_h)


########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(mesh_h=mesh_h, **nondefault_HIP_options)


########    CONSTRUCT KERNEL    ########

PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                               0, 0,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma,
                               sigma_min=sigma_min)

PCK.col_batches.add_one_sample_point_batch()
PCK.col_batches.add_one_sample_point_batch()
PCK.col_batches.add_one_sample_point_batch()
print('mesh_h=', mesh_h, ', num_batches=', PCK.col_batches.num_batches, ', num_sample_points=', PCK.col_batches.num_sample_points)

np.savetxt('mesh_vertices.txt', PCK.col_batches.mesh_vertices)
np.savetxt('mesh_cells.txt', PCK.col_batches.mesh_cells)

np.savetxt('points_batch0.txt', PCK.col_batches.point_batches[0])
np.savetxt('mu_batch0.txt', PCK.col_batches.mu_batches[0])
np.savetxt('Sigma_batch0.txt', PCK.col_batches.Sigma_batches[0].reshape((-1,4)))
np.savetxt('phi_batch0.txt', PCK.col_batches.phi_batches[0])

np.savetxt('points_batch1.txt', PCK.col_batches.point_batches[1])
np.savetxt('mu_batch1.txt', PCK.col_batches.mu_batches[1])
np.savetxt('Sigma_batch1.txt', PCK.col_batches.Sigma_batches[1].reshape((-1,4)))
np.savetxt('phi_batch1.txt', PCK.col_batches.phi_batches[1])

np.savetxt('points_batch2.txt', PCK.col_batches.point_batches[2])
np.savetxt('mu_batch2.txt', PCK.col_batches.mu_batches[2])
np.savetxt('Sigma_batch2.txt', PCK.col_batches.Sigma_batches[2].reshape((-1,4)))
np.savetxt('phi_batch2.txt', PCK.col_batches.phi_batches[2])

np.savetxt('dof_coords.txt', PCK.row_coords)


# Hd_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=False, hmatrix_tol=hmatrix_tol) # SEGFAULT
Hd_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=False, hmatrix_tol=1e-3) # SEGFAULT

# Phi = extras['A_kernel_hmatrix']
# Phi.visualize('bbb')
