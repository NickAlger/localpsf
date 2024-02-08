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

Ker_reordered = Ker[:, kdtree_sort_inds][kdtree_sort_inds,:]
dof_coords_reordered = dof_coords[kdtree_sort_inds,:]

#### Plot two impulse responses individually, and kernel with the corresponding columns indicated

p1 = np.array([0.2, 0.45])
p2 = np.array([0.6, 0.6])
pp = [p1, p2]

plt.matshow(Ker_reordered, cmap='binary')
plt.ylim([Ker_reordered.shape[0], 0])
plt.gca().set_xticks([])
plt.gca().set_yticks([])

for k in range(len(pp)):
    p = pp[k]
    ii = nearest_ind_func(dof_coords_reordered, p)
    print('k=', k, ', ii=', ii)
    plt.plot([ii, ii], [0, Ker.shape[0]], 'k', linestyle='dotted', linewidth=2.0)

plt.gcf().set_size_inches(8, 8)
plt.savefig('frog_kernel_matrix_a1.png', bbox_inches='tight', dpi=400)

for k in range(len(pp)):
    p = pp[k]
    ii = nearest_ind_func(dof_coords, p)
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.figure(figsize=(3.8, 3.8))
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig('frog_impulse_response' + str(k) + '.png', bbox_inches='tight', dpi=400)
