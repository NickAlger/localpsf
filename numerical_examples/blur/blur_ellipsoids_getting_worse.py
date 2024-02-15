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
aa = [1.0, 20.0, 27.0]

aa_ricker = [0.0, 0.23, 0.249]

#
# phi_vec = phi.vector()[:] / np.sum(phi.vector()[:] * mass_lumps)
# mu_x = np.sum(mass_lumps * phi_vec * dof_coords[:,0]) # E_phi[x]
# mu_y = np.sum(mass_lumps * phi_vec * dof_coords[:,1]) # E_phi[y]
# Sigma_xx = np.sum(mass_lumps * phi_vec * (dof_coords[:,0] - mu_x)**2) # E_phi[x - mu_x]
# Sigma_xy = np.sum(mass_lumps * phi_vec * (dof_coords[:,0] - mu_x)*(dof_coords[:,1] - mu_y)) # E_phi[x - mu_x]
# Sigma_yy = np.sum(mass_lumps * phi_vec * (dof_coords[:,1] - mu_y)**2) # E_phi[x - mu_x]
#
# mu = np.array([mu_x, mu_y])
# Sigma = np.array([[Sigma_xx, Sigma_xy],[Sigma_xy, Sigma_yy]])

#

for a in aa_ricker:
    Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = ricker_setup(nx, length_scaling, a)

    p0 = np.array([0.5, 0.5])
    ii = np.argmin(np.linalg.norm(dof_coords - p0.reshape((1, -1)), axis=1))
    p = dof_coords[ii,:]

    phi = dl.Function(Vh)
    # phi.vector()[:] = phi_function(dof_coords, p)
    phi_vec = Ker[:, ii].copy()
    phi.vector()[:] = phi_vec

    #

    psf_object = lpsf1.make_psf_fenics(
        lambda X: H @ X,
        lambda X: H.T @ X,
        Vh, Vh,
        mass_lumps, mass_lumps,
        num_initial_batches=0,
        tau = 3.0, display=True,
        num_neighbors = 10
    )

    #

    num_pts_1d = 500
    coords_1d = np.array([np.linspace(0, 1, num_pts_1d), p[1]*np.ones(num_pts_1d)]).T # vertical line

    rr_1d = phi_function(coords_1d, p)

    plt.figure()
    plt.plot(np.zeros(rr_1d.size), '--', c='gray')
    plt.plot(rr_1d, c='k', linewidth=2)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.xlim(0, len(coords_1d))

    # plt.title('rr_1d')

    plt.savefig('ricker_1d_a=' + str(a) + '.png', bbox_inches='tight', dpi=400)

    vmax = np.max(np.abs(phi.vector()[:]))

    plt.figure()
    # cm = dl.plot(phi, cmap='binary')
    cm = dl.plot(phi, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # plt.title('frog1')
    plt.plot(coords_1d[:,0], coords_1d[:,1], c='k', linewidth=1)
    plt.xlim(0.3, 0.7)
    plt.ylim(0.1, 0.9)

    mu_p = np.array([psf_object.mu(0)(p), psf_object.mu(1)(p)])
    Sigma_p = np.array([[psf_object.Sigma(0,0)(p), psf_object.Sigma(0,1)(p)],
                        [psf_object.Sigma(1,0)(p), psf_object.Sigma(1,1)(p)]])

    tau=3.0
    plot_ellipse(mu_p, Sigma_p, n_std_tau=tau, facecolor='none', edgecolor='k', linewidth=2)

    plt.savefig('ricker_ellipsoid_a=' + str(a) + '.png', bbox_inches='tight', dpi=400)


####

for a in aa:
    Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

    p0 = np.array([0.5, 0.5])
    ii = np.argmin(np.linalg.norm(dof_coords - p0.reshape((1, -1)), axis=1))
    p = dof_coords[ii,:]

    phi = dl.Function(Vh)
    # phi.vector()[:] = phi_function(dof_coords, p)
    phi_vec = Ker[:, ii].copy()
    phi_vec[np.abs(phi_vec) < 1e-1] = 0.0
    phi.vector()[:] = phi_vec

    #

    psf_object = lpsf1.make_psf_fenics(
        lambda X: H @ X,
        lambda X: H.T @ X,
        Vh, Vh,
        mass_lumps, mass_lumps,
        num_initial_batches=0,
        tau = 3.0, display=True,
        num_neighbors = 10
    )

    #

    num_pts_1d = 500
    coords_1d = np.array([np.linspace(0, 1, num_pts_1d), p[1]*np.ones(num_pts_1d)]).T # vertical line

    rr_1d = phi_function(coords_1d, p)

    plt.figure()
    plt.plot(np.zeros(rr_1d.size), '--', c='gray')
    plt.plot(rr_1d, c='k', linewidth=2)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.xlim(0, len(coords_1d))

    # plt.title('rr_1d')

    plt.savefig('frog_1d_a=' + str(a) + '.png', bbox_inches='tight', dpi=400)

    vmax = np.max(np.abs(phi.vector()[:]))

    plt.figure()
    # cm = dl.plot(phi, cmap='binary')
    cm = dl.plot(phi, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # plt.title('frog1')
    plt.plot(coords_1d[:,0], coords_1d[:,1], c='k', linewidth=1)
    plt.xlim(0.3, 0.7)
    plt.ylim(0.1, 0.9)

    mu_p = np.array([psf_object.mu(0)(p), psf_object.mu(1)(p)])
    Sigma_p = np.array([[psf_object.Sigma(0,0)(p), psf_object.Sigma(0,1)(p)],
                        [psf_object.Sigma(1,0)(p), psf_object.Sigma(1,1)(p)]])

    tau=3.0
    plot_ellipse(mu_p, Sigma_p, n_std_tau=tau, facecolor='none', edgecolor='k', linewidth=2)


    plt.savefig('frog_ellipsoid_a=' + str(a) + '.png', bbox_inches='tight', dpi=400)

##

