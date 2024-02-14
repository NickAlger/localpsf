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


max_batches = 100
nx = 63
a = 1.0 # <-- How bumpy? Use 1.0, which is maximum bumpiness without negative numbers
all_length_scalings = [1.0 / (t**2) for t in [1.0, 2.0, 3.0]]
all_error_levels = [0.2, 0.1, 0.05]
# length_scaling = 1./(3.0**2) #1.0

do_PSF = False

np.savetxt('all_length_scalings.txt', all_length_scalings)
np.savetxt('all_error_levels.txt', all_error_levels)

all_all_num_batches = []
all_all_num_impulses = []
all_all_fro_errors = []
all_ss = []
all_all_glr_ranks_fro = []
all_all_glr_ranks_2 = []
for length_scaling in all_length_scalings:
    print('length_scaling=', length_scaling)
    Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

    Ker_reordered = Ker[:, kdtree_sort_inds][kdtree_sort_inds,:]
    dof_coords_reordered = dof_coords[kdtree_sort_inds,:]

    #### Global low rank

    _, ss, _ = np.linalg.svd(Ker)
    all_ss.append(ss)
    norm_ker_fro = np.linalg.norm(Ker)

    all_glr_ranks_fro = []
    for rtol in all_error_levels:
        glr_num_fro = np.sum((norm_ker_fro**2 - np.cumsum(ss**2)) > rtol**2 * norm_ker_fro**2)
        all_glr_ranks_fro.append(glr_num_fro)
        print('rtol=', rtol, ', glr_num_fro=', glr_num_fro)
    all_all_glr_ranks_fro.append(all_glr_ranks_fro)

    all_glr_ranks_2 = []
    for rtol in all_error_levels:
        glr_num_2 = np.sum(ss > rtol * np.max(ss))
        all_glr_ranks_2.append(glr_num_2)
        print('rtol=', rtol, ', glr_num_2=', glr_num_2)
    all_all_glr_ranks_2.append(all_glr_ranks_2)

    np.savetxt('frog_singular_values_L='+str(length_scaling)+'.txt', ss)
    np.savetxt('all_glr_ranks_fro_L='   +str(length_scaling)+'.txt', all_glr_ranks_fro)
    np.savetxt('all_glr_ranks_2_L='     +str(length_scaling)+'.txt', all_glr_ranks_2)

    #### PSF
    if do_PSF:
        psf_object = lpsf1.make_psf_fenics(
            lambda X: H @ X,
            lambda X: H.T @ X,
            Vh, Vh,
            mass_lumps, mass_lumps,
            num_initial_batches=0,
            tau=3.0, display=True,
            num_neighbors=10
        )

        # print('Making row and column cluster trees')
        # ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=32)
        #
        # print('Making block cluster trees')
        # bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

        all_num_batches = []
        all_num_impulses = []
        all_fro_errors = []
        for num_batches in range(max_batches):
            psf_object.add_impulse_response_batch()
            num_batches = psf_object.psf_object.impulse_response_batches.num_batches
            num_impulses = psf_object.psf_object.impulse_response_batches.num_sample_points

            # H_psf_hmatrix, Ker_psf_hmatrix = psf_object.construct_hmatrices(bct)
            #
            # Ker_psf = np.zeros(Ker.shape)
            # for ii in range(Ker_psf.shape[1]):
            #     ei = np.zeros(Ker_psf.shape[1])
            #     ei[ii] = 1.0
            #     Ker_psf[:,ii] = Ker_psf_hmatrix * ei

            Ker_psf = psf_object.psf_object.psf_kernel.cpp_object.eval_integral_kernel_block(dof_coords.T, dof_coords.T)
            err_psf = np.linalg.norm(Ker_psf - Ker) / np.linalg.norm(Ker)
            print('err_psf=', err_psf)
            print('num_batches=', num_batches)
            print('num_impulses=', num_impulses)

            all_fro_errors.append(err_psf)
            all_num_batches.append(num_batches)
            all_num_impulses.append(num_impulses)

            if err_psf < np.min(all_error_levels):
                break

        all_all_num_batches.append(all_num_batches)
        all_all_num_impulses.append(all_num_impulses)
        all_all_fro_errors.append(all_fro_errors)

        np.savetxt('all_num_batches_L='     +str(length_scaling)+'.txt', all_num_batches)
        np.savetxt('all_num_impulses_L='    +str(length_scaling)+'.txt', all_num_impulses)
        np.savetxt('all_fro_errors_L='      +str(length_scaling)+'.txt', all_fro_errors)


