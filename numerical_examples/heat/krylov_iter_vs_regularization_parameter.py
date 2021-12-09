import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix
from localpsf.visualization import column_error_plot
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter

import scipy.sparse.linalg as spla


save_data = True
save_figures = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'krylov_iter_vs_regularization_parameter'
save_dir.mkdir(parents=True, exist_ok=True)


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 2.5e-2} # {'mesh_h': 2e-2}

tau = 2.5
all_num_batches = [1,3,6,9]
krylov_tol = 1e-6
a_reg_min = 1e-5
a_reg_max = 1e0
num_reg = 11
num_neighbors = 10
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
hmatrix_tol=1e-4


########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**nondefault_HIP_options)

########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########

all_Hd_hmatrix = list()
all_extras = list()
for num_batches in all_num_batches:
    ########    CONSTRUCT KERNEL    ########

    PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                                   num_batches, num_batches,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma,
                                   sigma_min=sigma_min)

    ########    CREATE HMATRIX    ########

    Hd_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=hmatrix_tol)
    all_Hd_hmatrix.append(Hd_pch)
    all_extras.append(extras)

best_Hd_hmatrix = all_Hd_hmatrix[-1]


########    SOLVE INVERSE PROBLEM FOR EACH REGULARIZATION PARAMETER    ########

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, best_Hd_hmatrix.bct)

regularization_parameters = np.logspace(np.log10(a_reg_min), np.log10(a_reg_max), num_reg)
u0_reconstructions = list()
for kk_reg in range(len(regularization_parameters)):
    a_reg = regularization_parameters[kk_reg]
    print('a_reg=', a_reg)
    HIP.regularization_parameter = a_reg

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    H_hmatrix = best_Hd_hmatrix + a_reg * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix.as_linear_operator(),
                                          tol=1e-10, maxiter=500)

    u0 = dl.Function(HIP.V)
    u0.vector()[:] = u0_numpy
    u0_reconstructions.append(u0)


########    KRYLOV ITERATIONS TO ACHIEVE GIVEN TOLERANCE    ########

def iterations_to_achieve_tolerance(errors, tol):
    successful_iterations = np.argwhere(np.array(errors) < tol)
    if len(successful_iterations > 0):
        first_successful_iteration = successful_iterations[0, 0] + 1
    else:
        first_successful_iteration = np.nan
    return first_successful_iteration


num_cg_iters_reg = np.zeros(len(regularization_parameters))
num_cg_iters_none = np.zeros(len(regularization_parameters))
all_num_cg_iters_hmatrix = np.zeros((len(regularization_parameters), len(all_num_batches)))
for ii_reg in range(len(regularization_parameters)):
    a_reg = regularization_parameters[ii_reg]
    HIP.regularization_parameter = a_reg

    u0_numpy = u0_reconstructions[ii_reg].vector()[:]
    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))

    _, _, errors_reg = custom_cg(HIP.H_linop, -g0_numpy, M=HIP.solve_R_linop, tol=1e-8, maxiter=1500,
                                 x_true=u0_numpy, track_residuals=False)

    cg_reg_iters = iterations_to_achieve_tolerance(errors_reg, krylov_tol)
    num_cg_iters_reg[ii_reg] = cg_reg_iters

    _, _, errors_none = custom_cg(HIP.H_linop, -g0_numpy, tol=1e-8, maxiter=1500,
                                  x_true=u0_numpy, track_residuals=False)

    cg_none_iters = iterations_to_achieve_tolerance(errors_none, krylov_tol)
    num_cg_iters_none[ii_reg] = cg_none_iters

    for kk_batch in range(len(all_num_batches)):
        Hd_hmatrix = all_Hd_hmatrix[kk_batch]
        H_hmatrix = Hd_hmatrix + a_reg * R0_hmatrix
        iH_hmatrix = H_hmatrix.inv()

        _, _, errors_hmatrix = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix.as_linear_operator(),
                                         tol=1e-8, maxiter=500,
                                         x_true=u0_numpy, track_residuals=False)

        cg_hmatrix_iters = iterations_to_achieve_tolerance(errors_hmatrix, krylov_tol)
        all_num_cg_iters_hmatrix[ii_reg, kk_batch] = cg_hmatrix_iters


########    SAVE DATA    ########

all_num_batches = np.array(all_num_batches)
regularization_parameters = np.array(regularization_parameters)

if save_data:
    np.savetxt(save_dir / 'all_num_batches.txt', all_num_batches)
    np.savetxt(save_dir / 'regularization_parameters.txt', regularization_parameters)
    np.savetxt(save_dir / 'num_cg_iters_none.txt', num_cg_iters_none)
    np.savetxt(save_dir / 'num_cg_iters_reg.txt', num_cg_iters_reg)
    np.savetxt(save_dir / 'all_num_cg_iters_hmatrix.txt', all_num_cg_iters_hmatrix)


########    MAKE FIGURE    ########

plt.figure()
plt.loglog(regularization_parameters, num_cg_iters_none)
plt.loglog(regularization_parameters, num_cg_iters_reg)
for kk_batch in range(len(all_num_batches)):
    plt.loglog(regularization_parameters, all_num_cg_iters_hmatrix[:, kk_batch])

plt.title(r'Conjugate gradient iterations to achieve tolerance $10^{-6}$')
plt.legend(['None', 'Reg'] + ['PCH'+str(nB) for nB in all_num_batches])
plt.xlabel(r'Regularization parameter, $\alpha$')
plt.ylabel(r'$\frac{\|u_0 - u_0^*\|}{\|u_0^*\|}$')

plt.show()

if save_figures:
    plt.savefig(save_dir / 'krylov_iter_vs_regularization_parameter.pdf', bbox_inches='tight', dpi=100)















########    CONSTRUCT KERNEL    ########

PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                               10, 10,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma,
                               sigma_min=sigma_min)

########    CREATE HMATRICES    ########

A_pch, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=hmatrix_tol)


########    SET UP REGULARIZATION OPERATOR AND PARAMETERS   ########

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, A_pch.bct)

regularization_parameters = np.logspace(np.log10(a_reg_min), np.log10(a_reg_max), num_reg)


########    LOOP OVER REG PARAM   ########

for a_reg in regularization_parameters:
    ########    SOLVE INVERSE PROBLEM   ########
    print('a_reg=', a_reg)
    HIP.regularization_parameter = a_reg

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    H_hmatrix = A_pch + a_reg * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix.as_linear_operator(),
                                          tol=1e-10, maxiter=500)



























import numpy as np
import matplotlib.pyplot as plt

import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix


load_results_from_file = False
save_results = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'krylov_iter_vs_regularization_parameter'
save_dir.mkdir(parents=True, exist_ok=True)

data_file = str(save_dir / 'data.npz')
HIP_options_file = save_dir / 'HIP_options.txt'
options_file = save_dir / 'options.txt'
plot_file = str(save_dir / 'krylov_iter_vs_regularization_parameter.pdf')


########    GENERATE OR LOAD RESULTS    ########

if not load_results_from_file:
    ########    OPTIONS    ########

    nondefault_HIP_options = {'mesh_h' : 3e-2}

    hmatrix_rtol = 1e-4
    krylov_tol = 1e-6
    a_reg_min = 1e-5
    a_reg_max = 1e0
    num_reg = 11
    all_batch_sizes = [1,3,6,9]

    options = {'hmatrix_rtol' : hmatrix_rtol,
               'krylov_tol' : krylov_tol,
               'a_reg_min' : a_reg_min,
               'a_reg_max' : a_reg_max,
               'num_reg' : num_reg,
               'all_batch_sizes' : all_batch_sizes}


    ########    SET UP HEAT INVERSE PROBLEM    ########

    HIP = HeatInverseProblem(**nondefault_HIP_options)


    ########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########

    all_Hd_hmatrix = list()
    all_extras = list()
    for k in all_batch_sizes:
        Hd_hmatrix, extras = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, k,
                                                         hmatrix_tol=hmatrix_rtol, make_positive_definite=True,
                                                         return_extras=True, grid_density_multiplier=0.5)
        all_Hd_hmatrix.append(Hd_hmatrix)
        all_extras.append(extras)

    best_Hd_hmatrix = all_Hd_hmatrix[-1]


    ########    SOLVE INVERSE PROBLEM FOR EACH REGULARIZATION PARAMETER    ########

    R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, best_Hd_hmatrix.bct)

    regularization_parameters = np.logspace(np.log10(a_reg_min), np.log10(a_reg_max), num_reg)
    u0_reconstructions = list()
    for kk_reg in range(len(regularization_parameters)):
        a_reg = regularization_parameters[kk_reg]
        print('a_reg=', a_reg)
        HIP.regularization_parameter = a_reg

        g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
        H_hmatrix = best_Hd_hmatrix + a_reg * R0_hmatrix
        iH_hmatrix = H_hmatrix.inv()

        u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                              M=iH_hmatrix.as_linear_operator(),
                                              tol=1e-10, maxiter=500)

        u0 = dl.Function(HIP.V)
        u0.vector()[:] = u0_numpy
        u0_reconstructions.append(u0)


    ########    KRYLOV ITERATIONS TO ACHIEVE GIVEN TOLERANCE    ########

    def iterations_to_achieve_tolerance(errors, tol):
        successful_iterations = np.argwhere(np.array(errors) < tol)
        if len(successful_iterations > 0):
            first_successful_iteration = successful_iterations[0, 0] + 1
        else:
            first_successful_iteration = np.nan
        return first_successful_iteration

    num_cg_iters_reg = np.zeros(len(regularization_parameters))
    num_cg_iters_none = np.zeros(len(regularization_parameters))
    all_num_cg_iters_hmatrix = np.zeros((len(regularization_parameters), len(all_batch_sizes)))
    for ii_reg in range(len(regularization_parameters)):
        a_reg = regularization_parameters[ii_reg]
        HIP.regularization_parameter = a_reg

        u0_numpy = u0_reconstructions[ii_reg].vector()[:]
        g0_numpy = HIP.g_numpy(np.zeros(HIP.N))

        _, _, errors_reg = custom_cg(HIP.H_linop, -g0_numpy, M=HIP.solve_R_linop, tol=1e-8, maxiter=1500,
                                 x_true=u0_numpy, track_residuals=False)

        cg_reg_iters = iterations_to_achieve_tolerance(errors_reg, krylov_tol)
        num_cg_iters_reg[ii_reg] = cg_reg_iters

        _, _, errors_none = custom_cg(HIP.H_linop, -g0_numpy, tol=1e-8, maxiter=1500,
                                      x_true=u0_numpy, track_residuals=False)

        cg_none_iters = iterations_to_achieve_tolerance(errors_none, krylov_tol)
        num_cg_iters_none[ii_reg] = cg_none_iters

        for kk_batch in range(len(all_batch_sizes)):
            Hd_hmatrix = all_Hd_hmatrix[kk_batch]
            H_hmatrix = Hd_hmatrix + a_reg * R0_hmatrix
            iH_hmatrix = H_hmatrix.inv()

            _, _, errors_hmatrix = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix.as_linear_operator(),
                                             tol=1e-8, maxiter=500,
                                              x_true=u0_numpy, track_residuals=False)

            cg_hmatrix_iters = iterations_to_achieve_tolerance(errors_hmatrix, krylov_tol)
            all_num_cg_iters_hmatrix[ii_reg, kk_batch] = cg_hmatrix_iters


        ########    SAVE RESULTS    ########

        if save_results:
            # Save options
            with open(options_file, 'w') as f:
                print(options, file=f)

            # Save HIP options
            with open(HIP_options_file, 'w') as f:
                print(HIP.options, file=f)

            # Save data
            np.savez(data_file,
                     regularization_parameters=regularization_parameters,
                     num_cg_iters_reg=num_cg_iters_reg,
                     num_cg_iters_none=num_cg_iters_none,
                     all_num_cg_iters_hmatrix=all_num_cg_iters_hmatrix,
                     krylov_tol=krylov_tol,
                     all_batch_sizes=all_batch_sizes)
else:
    data = np.load(data_file)
    regularization_parameters = data['regularization_parameters']
    num_cg_iters_none = data['num_cg_iters_none']
    num_cg_iters_reg = data['num_cg_iters_reg']
    all_num_cg_iters_hmatrix = data['all_num_cg_iters_hmatrix']
    all_batch_sizes = data['all_batch_sizes']


########    MAKE FIGURE    ########

plt.figure()
plt.loglog(regularization_parameters, num_cg_iters_none)
plt.loglog(regularization_parameters, num_cg_iters_reg)
for kk_batch in range(len(all_batch_sizes)):
    plt.loglog(regularization_parameters, all_num_cg_iters_hmatrix[:, kk_batch])

plt.title(r'Conjugate gradient iterations to achieve tolerance $10^{-6}$')
plt.legend(['None', 'Reg'] + ['PCH'+str(nB) for nB in all_batch_sizes])
plt.xlabel(r'Regularization parameter, $\alpha$')
plt.ylabel(r'$\frac{\|u_0 - u_0^*\|}{\|u_0^*\|}$')

plt.show()

if save_results:
    plt.savefig(plot_file, bbox_inches='tight', dpi=100)

