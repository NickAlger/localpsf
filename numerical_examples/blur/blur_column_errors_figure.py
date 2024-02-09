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


nx = 63
length_scaling = 1.0 #0.0625
a = 1.0
Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

all_num_batches = [5, 10, 20]
all_Ker_psf = []
all_sample_points = []
all_fro_errors = []

psf_object = lpsf1.make_psf_fenics(
    lambda X: H @ X,
    lambda X: H.T @ X,
    Vh, Vh,
    mass_lumps, mass_lumps,
    num_initial_batches=0,
    tau=3.0, display=True,
    num_neighbors=10
)

for num_batches in all_num_batches:
    while psf_object.num_batches < num_batches:
        psf_object.add_impulse_response_batch()

    Ker_psf = psf_object.psf_object.psf_kernel.cpp_object.eval_integral_kernel_block(dof_coords.T, dof_coords.T)
    err_psf = np.linalg.norm(Ker_psf - Ker) / np.linalg.norm(Ker)
    print('err_psf=', err_psf)

    all_fro_errors.append(err_psf)
    all_Ker_psf.append(Ker_psf)
    all_sample_points.append(psf_object.psf_object.impulse_response_batches.sample_points.copy())

np.savetxt('blur_all_num_batches.txt', all_num_batches)
np.savetxt('blur_all_fro_errors.txt', all_fro_errors)

#

zero_one_func = dl.Function(Vh)
zero_one_func.vector()[1] = 1.0
plt.figure()
cm_zero_one = dl.plot(zero_one_func, cmap='binary')

for num_batches, Ker_psf, sp in zip(all_num_batches, all_Ker_psf, all_sample_points):
    ker_err_func = dl.Function(Vh)
    ker_err_vec = np.linalg.norm(Ker_psf - Ker, axis=0) / np.linalg.norm(Ker, axis=0)
    # ker_err_vec = np.linalg.norm(Ker_psf - Ker, axis=0) / (np.linalg.norm(Ker) / np.sqrt(Ker.shape[1]))
    ker_err_vec[np.linalg.norm(Ker, axis=0) == 0] = 0.0
    ker_err_func.vector()[:] = ker_err_vec

    plt.figure()
    # cm = dl.plot(ker_err_func, cmap='binary', markersize=5)
    cm = dl.plot(ker_err_func, cmap='binary', markersize=5, vmin=0.0, vmax=1.0)
    plt.plot(sp[:,0], sp[:,1], '.', c='k')
    # cm.set_clim(0.0, 1.0)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_aspect('equal')
    plt.savefig('frog_column_errors_psf' + str(num_batches) + '.png', dpi=300, bbox_inches='tight')

    # plt.colorbar(cm)
    plt.colorbar(cm_zero_one) # workaround for fenics bug
    plt.savefig('frog_column_errors_psf' + str(num_batches) + '_colorbar.png', dpi=300, bbox_inches='tight')

for nb, e in zip(all_num_batches, all_fro_errors):
    print('num_batches=', nb, ', fro_err=', e)