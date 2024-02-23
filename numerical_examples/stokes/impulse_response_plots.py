import numpy as np
import dolfin as dl
import scipy.linalg as sla
import matplotlib.pyplot as plt
from localpsf import localpsf_root
from nalger_helper_functions import *

import localpsf.stokes_variational_forms as svf

nearest_ind_func = lambda yy, x: np.argmin(np.linalg.norm(yy - x.reshape((1, -1)), axis=1))

UIP: svf.LinearStokesInverseProblemUnregularized = svf.stokes_inverse_problem_unregularized()

UIP.derivatives.update_parameter(UIP.mtrue_Vh2().vector()[:])

Vh = UIP.function_spaces.Vh2
dof_coords = Vh.tabulate_dof_coordinates()

ny = 11
nx = 11
pp_y = np.linspace(-1e4, 1e4, ny)
pp_x = np.linspace(-1e4, 1e4, nx)
for jj in range(ny):
    py = pp_y[jj]
    for ii in range(nx):
        px = pp_x[ii]
        p = np.array([px, py])
        if np.linalg.norm(p) > 1e4:
            continue

        ind = nearest_ind_func(dof_coords, p)

        delta_p = np.zeros(Vh.dim())
        delta_p[ind] = 1.0

        phi_p = dl.Function(Vh)
        phi_p.vector()[:] = UIP.derivatives.apply_gauss_newton_hessian(delta_p)

        plt.figure(figsize=(8,8))
        dl.plot(phi_p, vmin=0.0, cmap='binary')
        plt.xlim(-1.1e4, 1.1e4)
        plt.ylim(-1.1e4, 1.1e4)
        plt.gca().add_patch(plt.Circle((0, 0), 1e4, color='gray', fill=False, lw=1))
        plt.axis('off')
        plt.savefig('impulse_plots/ice_mountain_fine_mesh_phi_'+str(p[0])+'_'+str(p[1])+'.png', bbox_inches='tight', dpi=300)
