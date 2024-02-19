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
a = 1.0 # <-- How bumpy? Use 1.0, which is maximum bumpiness without negative numbers
length_scaling = 1.0 / (2.0**2)
all_tau = [2.0, 2.5, 3.0, 3.5, 4.0]
all_num_batches0 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
shape_parameter = 0.5
num_neighbors = 10

Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

Ker_reordered = Ker[:, kdtree_sort_inds][kdtree_sort_inds,:]
dof_coords_reordered = dof_coords[kdtree_sort_inds,:]

all_num_batches     = np.zeros((len(all_tau), len(all_num_batches0)))
all_num_impulses    = np.zeros((len(all_tau), len(all_num_batches0)))
all_fro_errors      = np.zeros((len(all_tau), len(all_num_batches0)))
for ii, tau in enumerate(all_tau):
    psf_object = lpsf1.make_psf_fenics(
        lambda X: H @ X,
        lambda X: H.T @ X,
        Vh, Vh,
        mass_lumps, mass_lumps,
        num_initial_batches=0,
        tau=tau, display=True,
        num_neighbors=num_neighbors,
        shape_parameter=shape_parameter,
    )

    for jj, num_batches0 in enumerate(all_num_batches0):
        while psf_object.psf_object.impulse_response_batches.num_batches < num_batches0:
            new_points = psf_object.add_impulse_response_batch()
            if not new_points:
                break

        num_batches = psf_object.psf_object.impulse_response_batches.num_batches
        num_impulses = psf_object.psf_object.impulse_response_batches.num_sample_points

        Ker_psf = psf_object.psf_object.psf_kernel.cpp_object.eval_integral_kernel_block(dof_coords.T, dof_coords.T)
        err_psf = np.linalg.norm(Ker_psf - Ker) / np.linalg.norm(Ker)
        print('err_psf=', err_psf)
        print('num_batches=', num_batches)
        print('num_impulses=', num_impulses)
        print('tau=', tau)

        all_num_batches[ii,jj] = num_batches
        all_num_impulses[ii,jj] = num_impulses
        all_fro_errors[ii,jj] = err_psf

        np.savetxt('all_tau.txt', all_tau)
        np.savetxt('all_num_batches.txt', all_num_batches)
        np.savetxt('all_num_impulses.txt', all_num_impulses)
        np.savetxt('all_fro_errors.txt', all_fro_errors)

fro_errs_batches_theory = 0.15 / np.array(all_num_batches[-1,:])

plt.figure()
leg = []
for ii, tau in enumerate(all_tau):
    plt.loglog(all_num_batches[ii,:], all_fro_errors[ii,:])
    leg.append(r'$\tau$='+str(tau))
plt.loglog(all_num_batches[-1,:], fro_errs_batches_theory, '--', c='gray')
leg.append(r'$const \times \# batches^{-1}$')
plt.legend(leg)
plt.ylim(None, 1e0)
plt.xlabel('#batches')
plt.ylabel(r'$||\Phi - \widetilde{\Phi}||/||\Phi||$')
plt.title(r'Convergence for different $\tau$')
plt.savefig('frog_tau_comparison_by_batch.pdf', bbox_inches='tight', dpi=300)


fro_errs_impulses_theory = 10*0.15 / np.array(all_num_impulses[-1,:])

plt.figure()
leg = []
for ii, tau in enumerate(all_tau):
    plt.loglog(all_num_impulses[ii,:], all_fro_errors[ii,:])
    leg.append(r'$\tau$='+str(tau))
plt.loglog(all_num_impulses[-1,:], fro_errs_impulses_theory, '--', c='gray')
leg.append(r'$const \times \left(\# impulse~responses\right)^{-1}$')
plt.legend(leg)
plt.ylim(None, 1e0)
plt.xlabel('#impulse responses')
plt.ylabel(r'$||\Phi - \widetilde{\Phi}||/||\Phi||$')
plt.title(r'Convergence for different $\tau$')
plt.savefig('frog_tau_comparison_by_impulse.pdf', bbox_inches='tight', dpi=300)

for ii in range(len(all_tau)):
    np.savetxt('frog_convergence_batches_tau'+str(all_tau[ii])+'.dat', np.vstack([all_num_batches[ii,:], all_fro_errors[ii,:]]).T)
    np.savetxt('frog_convergence_impulses_tau'+str(all_tau[ii])+'.dat', np.vstack([all_num_impulses[ii, :], all_fro_errors[ii, :]]).T)

np.savetxt('frog_convergence_rate_batches.dat', np.vstack([all_num_batches[-1,:], fro_errs_batches_theory]).T)
np.savetxt('frog_convergence_rate_impulses.dat', np.vstack([all_num_impulses[-1,:], fro_errs_impulses_theory]).T)
