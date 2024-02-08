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

#### Impulse response grid plot

num_pts_x = 5
num_pts_y = 5
pX, pY = np.meshgrid(np.linspace(0.01,0.99,num_pts_x), np.linspace(0.01,99,num_pts_y)[::-1])
pp = np.vstack([pX.reshape(-1), pY.reshape(-1)]).T

plt.figure(figsize = (num_pts_x,num_pts_y))
gs1 = gridspec.GridSpec(num_pts_x, num_pts_y)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
for k in range(len(pp)):
    p = pp[k]
    ii = nearest_ind_func(dof_coords, p)
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.subplot(gs1[k])
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

plt.savefig('frog_impulse_response_grid.png', bbox_inches='tight', dpi=300)