import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import sys
from blur import *
import HODLR as HODLR

save_dir = 'data/'
nx = 63 # 64^2 = 2^12
#nx = 31
a = 1.0 # <-- How bumpy? Use 1.0, which is maximum bumpiness without negative numbers
all_length_scalings = [1.0 / (t**2) for t in [1.0, 2.0, 3.0]]
#all_error_levels = [0.2, 0.1, 0.05]
# length_scaling = 1./(3.0**2) #1.0

np.savetxt(save_dir+'all_length_scalings.txt', all_length_scalings)

#all_all_num_batches = []
#all_all_num_impulses = []
all_all_fro_errors = []
all_all_HODLR_costs = []
# ---- HODLR parameter settings
d = 5
eta = 10.
tau_r = 1.e-16
adaptive_tols = np.logspace(1, -20, base = 0.5, num=300)
I = np.identity((nx+1)**2)
L = 8


for length_scaling in all_length_scalings:
    print('length_scaling=', length_scaling)
    Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

    Ker_reordered = Ker[:, kdtree_sort_inds][kdtree_sort_inds,:]

    # ---- expected to vary from 400 to 1600
    norm_Ker = np.linalg.norm(Ker_reordered)
    print(norm_Ker)
    #### Hierarchical low rank
    Ker_HODLR = HODLR.HODLR(Ker_reordered, I)
    all_fro_errors = []
    all_HODLR_costs = []
    for adaptive_tol in adaptive_tols:
         Ker_HODLR.compress_adaptive(L, d, adaptive_tol, tau_r, eta)
         Ker_HODLR_explicit = Ker_HODLR.form()
         fro_error = np.linalg.norm(Ker_HODLR_explicit - Ker_reordered) / norm_Ker
         HODLR_cost = Ker_HODLR.compression_cost()
         all_fro_errors.append(fro_error)
         all_HODLR_costs.append(HODLR_cost)
         print("fro_error = " , fro_error)
         print("adaptive_tol = " , adaptive_tol)
    np.savetxt(save_dir+'all_fro_errors_L='      +str(length_scaling)+'.txt', all_fro_errors)
    np.savetxt(save_dir+'all_HODLR_costs_L='+str(length_scaling)+'.txt', all_HODLR_costs)
