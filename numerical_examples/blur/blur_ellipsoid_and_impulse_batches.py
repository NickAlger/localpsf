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
length_scaling = 1.0
a = 1.0
Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

#### Create PSF approximation

psf_object = lpsf1.make_psf_fenics(
    lambda X: H @ X,
    lambda X: H.T @ X,
    Vh, Vh,
    mass_lumps, mass_lumps,
    tau = 3.0, display=True,
)

#### Plot ellipsoid batch without impulses

bb = [0,1,2]

for b in bb:
    IRB = psf_object.psf_object.impulse_response_batches
    fig = plt.figure(figsize=(4,4))

    phi = psf_object.impulse_response_batch(b)

    start = IRB.batch2point_start[b]
    stop = IRB.batch2point_stop[b]
    pp = IRB.sample_points[start:stop, :]
    mu_batch = IRB.sample_mu[start:stop, :]
    Sigma_batch = IRB.sample_Sigma[start:stop, :, :]

    for k in range(mu_batch.shape[0]):
        plot_ellipse(mu_batch[k, :], Sigma_batch[k, :, :], n_std_tau=IRB.tau,
                     facecolor='none', edgecolor='k', linewidth=1)

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.savefig('frog_ellipsoid_batch' + str(b) + '.png', bbox_inches='tight', dpi=400)

#### Plot ellipsoid batch with impulses

for b in bb:
    IRB = psf_object.psf_object.impulse_response_batches
    fig = plt.figure(figsize=(4,4))

    phi = psf_object.impulse_response_batch(b)

    start = IRB.batch2point_start[b]
    stop = IRB.batch2point_stop[b]
    pp = IRB.sample_points[start:stop, :]
    mu_batch = IRB.sample_mu[start:stop, :]
    Sigma_batch = IRB.sample_Sigma[start:stop, :, :]

    cm = dl.plot(phi, cmap='binary')

    for k in range(mu_batch.shape[0]):
        plot_ellipse(mu_batch[k, :], Sigma_batch[k, :, :], n_std_tau=IRB.tau,
                     facecolor='none', edgecolor='k', linewidth=1)

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.savefig('frog_impulse_batch' + str(b) + '.png', bbox_inches='tight', dpi=400)
