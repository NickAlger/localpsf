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

#### Plot moments

plt.figure(figsize=(4,4))
dl.plot(psf_object.vol(), cmap='binary', clim=[0.0, None])
# plt.colorbar(cm)
# plt.title(r'$V$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_vol.png', bbox_inches='tight', dpi=400)

plt.figure(figsize=(4,4))
cm = dl.plot(psf_object.mu(0), cmap='binary')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_mu0.png', bbox_inches='tight', dpi=400)

plt.figure(figsize=(4,4))
dl.plot(psf_object.mu(1), cmap='binary')
# plt.title(r'$\mu^2$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_mu1.png', bbox_inches='tight', dpi=400)

plt.figure(figsize=(4,4))
dl.plot(psf_object.Sigma(0,0), cmap='binary')
# plt.title(r'$\Sigma^{11}$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_Sigma00.png', bbox_inches='tight', dpi=400)

plt.figure(figsize=(4,4))
dl.plot(psf_object.Sigma(0,1), cmap='binary')
# plt.title(r'$\Sigma^{12}$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_Sigma01.png', bbox_inches='tight', dpi=400)

plt.figure(figsize=(4,4))
dl.plot(psf_object.Sigma(1,1), cmap='binary')
# plt.title(r'$\Sigma^{22}$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_Sigma11.png', bbox_inches='tight', dpi=400)
