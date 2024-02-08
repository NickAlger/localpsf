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

#### Plot one impulse response with ellipsoid

p0 = np.array([0.6, 0.6])
ii = np.argmin(np.linalg.norm(dof_coords - p0.reshape((1, -1)), axis=1))
p = dof_coords[ii,:]

mu_p = np.array([psf_object.mu(0)(p), psf_object.mu(1)(p)])
Sigma_p = np.array([[psf_object.Sigma(0,0)(p), psf_object.Sigma(0,1)(p)],
                    [psf_object.Sigma(1,0)(p), psf_object.Sigma(1,1)(p)]])

phi = dl.Function(Vh)
phi.vector()[:] = Ker[:, ii].copy()
plt.figure(figsize=(4, 4))
cm = dl.plot(phi, cmap='binary')
plt.axis('off')

tau=3.0
plot_ellipse(mu_p, Sigma_p, n_std_tau=tau, facecolor='none', edgecolor='k', linewidth=1)

xwidth = (tau*1.1)*np.sqrt(Sigma_p[0,0])
ywidth = (tau*1.1)*np.sqrt(Sigma_p[1,1])

plt.xlim([mu_p[0] - xwidth, mu_p[0] + xwidth])
plt.ylim([mu_p[1] - ywidth, mu_p[1] + ywidth])

plt.savefig('frog_one_ellipsoid.png', bbox_inches='tight', dpi=400)
