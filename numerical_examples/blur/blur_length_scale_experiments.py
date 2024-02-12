import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dolfin as dl
import localpsf.localpsf_cg1_lumped as lpsf1
from localpsf.blur import *
import scipy.linalg as sla
from functools import partial
from nalger_helper_functions import plot_ellipse
import hlibpro_python_wrapper as hpro
from tqdm.auto import tqdm


max_batches = 100
nx = 63
a = 1.0 # <-- How bumpy? Use 1.0, which is maximum bumpiness without negative numbers
all_length_scalings = [1.0 / (t**2) for t in [1.0, 2.0, 3.0]]
# length_scaling = 1./(3.0**2) #1.0

all_all_num_batches = []
all_all_num_impulses = []
all_all_fro_errors = []
for length_scaling in all_length_scalings:
    print('length_scaling=', length_scaling)
    Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

    Ker_reordered = Ker[:, kdtree_sort_inds][kdtree_sort_inds,:]
    dof_coords_reordered = dof_coords[kdtree_sort_inds,:]

    #

    psf_object = lpsf1.make_psf_fenics(
        lambda X: H @ X,
        lambda X: H.T @ X,
        Vh, Vh,
        mass_lumps, mass_lumps,
        num_initial_batches=0,
        tau=3.0, display=True,
        num_neighbors=10
    )

    all_num_batches = []
    all_num_impulses = []
    all_fro_errors = []
    for num_batches in range(max_batches):
        psf_object.add_impulse_response_batch()
        num_batches = psf_object.psf_object.impulse_response_batches.num_batches
        num_impulses = psf_object.psf_object.impulse_response_batches.num_sample_points

        Ker_psf = psf_object.psf_object.psf_kernel.cpp_object.eval_integral_kernel_block(dof_coords.T, dof_coords.T)
        err_psf = np.linalg.norm(Ker_psf - Ker) / np.linalg.norm(Ker)
        print('err_psf=', err_psf)
        print('num_batches=', num_batches)
        print('num_impulses=', num_impulses)

        all_fro_errors.append(err_psf)
        all_num_batches.append(num_batches)
        all_num_impulses.append(num_impulses)

    all_all_num_batches.append(all_num_batches)
    all_all_num_impulses.append(all_num_impulses)
    all_all_fro_errors.append(all_fro_errors)

    #### Plot two impulse responses individually, and kernel with the corresponding columns indicated

    p = np.array([0.2, 0.45])

    ii = nearest_ind_func(dof_coords, p)
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.figure(figsize=(3.8, 3.8))
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title('length_scaling=', length_scaling)

