import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix

from localpsf.stokes_inverse_problem_cylinder import *
import dolfin as dl

import scipy.sparse.linalg as spla


save_data = True
save_figures = True
# import os
# root_dir = os.path.abspath(os.curdir)
# rel_save_dir = './error_vs_num_batches/'
# save_dir = os.path.join(root_dir, rel_save_dir)
# os.makedirs(save_dir, exist_ok=True)

stokes_base_dir = get_project_root() / 'numerical_examples' / 'stokes'

save_dir = stokes_base_dir / 'error_vs_num_batches'
save_dir.mkdir(parents=True, exist_ok=True)


# --------- set up the problem
# mfile_name = "meshes/cylinder_coarse"
mfile_name = stokes_base_dir / 'meshes' / 'cylinder_medium'
mesh = dl.Mesh(str(mfile_name)+".xml")
boundary_markers = dl.MeshFunction("size_t", mesh, str(mfile_name)+"_facet_region.xml")


Newton_iterations = 2
nondefault_StokesIP_options = {'mesh' : mesh,'boundary_markers' : boundary_markers,
        'Newton_iterations': Newton_iterations, 
        'misfit_only': True,
        'gauss_newton_approx': True,
        'load_fwd': False,
        'lam': 1.e10,
        'solve_inv': False,
        'gamma': 1.e4,
        'm0': 1.5*7.,
        'mtrue_string': 'm0 - (m0 / 7.)*std::cos(2.*x[0]*pi/Radius)'}

StokesIP = StokesInverseProblemCylinder(**nondefault_StokesIP_options)


########    OPTIONS    ########

all_tau = [3.0]
max_num_batches = 50
num_neighbors   = 10
gamma       = 1.e-5
hmatrix_tol = 1.e-5
num_random_error_matvecs = 200
max_scale_discrepancy=1.e5



########    CONSTRUCT KERNEL    ########

all_num_sample_points = list()
all_num_batches = list()
all_fro_errors = list()
all_induced2_errors = list()

# create random vectors "frozen"
Omega = np.random.randn(StokesIP.N, num_random_error_matvecs)
Y_true = np.zeros((StokesIP.N, num_random_error_matvecs))

print("applying true Hd for PCH error estimation")
for k in tqdm(range(num_random_error_matvecs)):
    Y_true[:, k] = StokesIP.apply_Hd_Vsub_numpy(Omega[:, k])
#    Y[:, k] = apply_A_numpy(omega)

#norm_A = np.linalg.norm(Y_true) / np.sqrt(n_random_error_matvecs)
#norm_A_err = np.linalg.norm(Y_true - Y) / np.sqrt(n_random_error_matvecs)
#
#relative_error_overall = norm_A_err / norm_A
#
#norm_of_each_column_of_A_true = np.linalg.norm(Y_true, axis=1) / np.sqrt(n_random_error_matvecs)
#norm_of_each_column_of_error = np.linalg.norm(Y_true - Y, axis=1) / np.sqrt(n_random_error_matvecs)
#relative_error_of_each_column = norm_of_each_column_of_error / norm_of_each_column_of_A_true
#
#return relative_error_of_each_column, norm_of_each_column_of_A_true, norm_of_each_column_of_error, relative_error_overall


for tau in all_tau:
    print('tau=', tau)
    all_num_sample_points.append(list())
    all_num_batches.append(list())
    all_fro_errors.append(list())
    all_induced2_errors.append(list())

    PCK = ProductConvolutionKernel(StokesIP.V, StokesIP.V, StokesIP.apply_Hd_petsc, StokesIP.apply_Hd_petsc,
                                   0, 0,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma,
                                   max_scale_discrepancy=max_scale_discrepancy,
                                   cols_only=True,
                                   use_lumped_mass_moments=True,
                                   use_lumped_mass_impulses=True)
    np.savetxt("vol.txt", PCK.col_batches.vol)
    for batch_num in range(max_num_batches):
        PCK.col_batches.add_one_sample_point_batch()
        all_num_batches[-1].append(PCK.col_batches.num_batches)
        all_num_sample_points[-1].append(PCK.col_batches.num_sample_points)
        print('tau=', tau, ', num_batches=', PCK.col_batches.num_batches, ', num_sample_points=', PCK.col_batches.num_sample_points)


        ########    CREATE HMATRICES    ########

        # A_pch, extras  = make_hmatrix_from_kernel(PCK, make_positive_definite=False, hmatrix_tol=hmatrix_tol)
        A_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=hmatrix_tol)
        A_pchT         = A_pch.transpose()

        ########    ESTIMATE ERRORS IN FROBENIUS NORM    ########

        #_, _, _, relative_err_fro \
        #    = estimate_column_errors_randomized(StokesIP.apply_Hd_Vsub_numpy,
        #                                        lambda x: A_pch * x,
        #                                        A_pch.shape[1],
        #                                        num_random_error_matvecs)
        Y = np.zeros((StokesIP.N, num_random_error_matvecs))
        for k in tqdm(range(num_random_error_matvecs)):
            omega = (Omega[:,k]).copy()
            Y[:, k] = A_pchT * omega
        
        norm_A = np.linalg.norm(Y_true) / np.sqrt(num_random_error_matvecs)
        norm_A_err = np.linalg.norm(Y_true - Y) / np.sqrt(num_random_error_matvecs)
        
        relative_error_overall = norm_A_err / norm_A
        
        norm_of_each_column_of_A_true = np.linalg.norm(Y_true, axis=1) / np.sqrt(num_random_error_matvecs)
        norm_of_each_column_of_A_true = np.max([np.ones(StokesIP.N)*1.e-8, norm_of_each_column_of_A_true])
        
        norm_of_each_column_of_error = np.linalg.norm(Y_true - Y, axis=1) / np.sqrt(num_random_error_matvecs)

        relative_error_of_each_column = norm_of_each_column_of_error / norm_of_each_column_of_A_true
        
        relative_err_fro = relative_error_overall

        print('relative_err_fro=', relative_err_fro)

        all_fro_errors[-1].append(relative_err_fro)

        vol_fct          = dl.Function(StokesIP.V)
        vol_fct.vector()[:] = np.log10(np.abs(PCK.col_batches.vol))
        log_vol = np.log10(np.abs(PCK.col_batches.vol))
        log_vol[np.isinf(log_vol)] = 0.0
        vol_fct.vector()[:] = log_vol
        relative_err_fct = dl.Function(StokesIP.V)
        relative_err_fct.vector()[:] = relative_error_of_each_column
        err_fct          = dl.Function(StokesIP.V)
        err_fct.vector()[:] = np.log10(np.abs(norm_of_each_column_of_error))

        pp = PCK.col_batches.sample_points
        print("shape pp = ", pp.shape)
        
        cm = dl.plot(relative_err_fct)
        plt.colorbar(cm)
        
        #pp = np.vstack(point_batches)
        plt.plot(pp[:, 0], pp[:, 1], '.r')
        
        plt.title('Hd columns relative error, ' + str(pp.shape[0]) + ' points')
        plt.savefig(str(save_dir)+'/column_error_'+str(batch_num)+'batch.png')
        plt.close()
        
        cm = dl.plot(err_fct)
        plt.colorbar(cm)
        
        #pp = np.vstack(point_batches)
        plt.plot(pp[:, 0], pp[:, 1], '.r')
        
        plt.title('Log Hd columns absolute error, ' + str(pp.shape[0]) + ' points')
        plt.savefig(str(save_dir)+'/column_logabserror_'+str(batch_num)+'batch.png')
        plt.close()
        
        cm = dl.plot(vol_fct)
        plt.colorbar(cm)
        
        #pp = np.vstack(point_batches)
        plt.plot(pp[:, 0], pp[:, 1], '.r')
        
        plt.title('Log Volume function, ' + str(pp.shape[0]) + ' points')
        plt.savefig(str(save_dir)+'/volume_function_'+str(batch_num)+'batch.png')
        plt.close()



        ########    ESTIMATE ERRORS IN INDUCED2 NORM    ########

        A_linop = spla.LinearOperator(A_pch.shape, matvec=lambda x: StokesIP.apply_Hd_Vsub_numpy(x))
        err_A_linop = spla.LinearOperator(A_pch.shape, matvec=lambda x: StokesIP.apply_Hd_Vsub_numpy(x) - A_pch * x)

        aa, _ = spla.eigsh(A_linop, k=1, which='LM')
        ee, _ = spla.eigsh(err_A_linop, k=1, which='LM')

        relative_err_induced2 = np.max(np.abs(ee)) / np.max(np.abs(aa))
        print('relative_err_induced2=', relative_err_induced2)

        all_induced2_errors[-1].append(relative_err_induced2)
        plt.figure()
        PCK.col_batches.visualize_impulse_response_batch(batch_num)
        plt.savefig(str(save_dir) + '/impulseresponse' + str(batch_num)+'.png')
        plt.close()


########    SAVE DATA    ########

all_num_sample_points = [np.array(x) for x in all_num_sample_points]
all_num_batches = [np.array(x) for x in all_num_batches]
all_fro_errors = [np.array(x) for x in all_fro_errors]
all_induced2_errors = [np.array(x) for x in all_induced2_errors]

if save_data:
    np.savetxt(str(save_dir) + '/all_tau.txt', all_tau)
    for k in range(len(all_tau)):
        np.savetxt(str(save_dir) + '/num_sample_points_'+str(k)+'.txt', all_num_sample_points[k])
        np.savetxt(str(save_dir) + '/num_batches_' + str(k) + '.txt', all_num_batches[k])
        np.savetxt(str(save_dir) + '/fro_errors_' + str(k) + '.txt', all_fro_errors[k])
        np.savetxt(str(save_dir) + '/induced2_errors_' + str(k) + '.txt', all_induced2_errors[k])


########    MAKE FIGURES    ########

legend = list()
for k in range(len(all_tau)):
    plt.semilogy(all_num_batches[k], all_induced2_errors[k])
    legend.append('tau='+str(all_tau[k]))

plt.title(r'Relative error (induced-2 norm) vs. number of batches')
plt.xlabel(r'Number of batches')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||_2}{||H_d||_2}$')
plt.legend(legend)

if save_figures:
    plt.savefig(str(save_dir) + '/error_vs_num_batches_induced2.pdf', bbox_inches='tight', dpi=100)
plt.show()
plt.close()
#

legend = list()
for k in range(len(all_tau)):
    plt.semilogy(all_num_batches[k], all_fro_errors[k])
    legend.append('tau=' + str(all_tau[k]))

plt.title(r'Relative error (Frobenius norm) vs. number of batches')
plt.xlabel(r'Number of batches')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||_\mathrm{Fro}}{||H_d||_\mathrm{Fro}}$')
plt.legend(legend)

if save_figures:
    plt.savefig(str(save_dir) + '/error_vs_num_batches_fro.pdf', bbox_inches='tight', dpi=100)
plt.show()
plt.close()
#

legend = list()
for k in range(len(all_tau)):
    plt.semilogy(all_num_sample_points[k], all_induced2_errors[k])
    legend.append('tau='+str(all_tau[k]))

plt.title(r'Relative error (induced-2 norm) vs. number of sample points')
plt.xlabel(r'Number of sample points')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||_2}{||H_d||_2}$')
plt.legend(legend)

if save_figures:
    plt.savefig(str(save_dir) + '/error_vs_num_sample_points_induced2.pdf', bbox_inches='tight', dpi=100)
plt.show()
plt.close()
#

legend = list()
for k in range(len(all_tau)):
    plt.semilogy(all_num_sample_points[k], all_fro_errors[k])
    legend.append('tau='+str(all_tau[k]))

plt.title(r'Relative error (Frobenius norm) vs. number of sample points')
plt.xlabel(r'Number of sample points')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||_\mathrm{Fro}}{||H_d||_\mathrm{Fro}}$')
plt.legend(legend)

if save_figures:
    plt.savefig(str(save_dir) + '/error_vs_num_sample_points_fro.pdf', bbox_inches='tight', dpi=100)
plt.show()
plt.close()
#

