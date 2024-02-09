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


nx = 49
length_scaling = 1.0 #0.0625
a = 1.0
Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

num_neighbors = 10

psf_object = lpsf1.make_psf_fenics(
    lambda X: H @ X,
    lambda X: H.T @ X,
    Vh, Vh,
    mass_lumps, mass_lumps,
    num_initial_batches=20,#0,
    tau = 3.0, display=True,
    num_neighbors = num_neighbors
)

#

print('Making row and column cluster trees')
ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=32)

print('Making block cluster trees')
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

bct.visualize('frog_bct.eps')