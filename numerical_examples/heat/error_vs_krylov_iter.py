import numpy as np
import matplotlib.pyplot as plt

import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter


load_results_from_file = False
save_results = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'error_vs_krylov_iter'
save_dir.mkdir(parents=True, exist_ok=True)

data_file = str(save_dir / 'data.npz')
options_file = save_dir / 'options.txt'
plot_file = str(save_dir / 'error_vs_krylov_iter.pdf')


########    GENERATE OR LOAD RESULTS    ########

if not load_results_from_file:
    ########    OPTIONS    ########

    HIP_options = {'mesh_h': 1.5e-2,
                   'finite_element_order': 1,
                   'final_time': 3e-4,
                   'num_timesteps': 35,
                   'noise_level': 5e-2,
                   'prior_correlation_length': 0.05}

    hmatrix_rtol = 1e-4
    all_batch_sizes = [1,3,6,9]

    options = {'hmatrix_rtol' : hmatrix_rtol,
               'all_batch_sizes' : all_batch_sizes}

    options = options.update(HIP_options)


    ########    SET UP HEAT INVERSE PROBLEM    ########

    HIP = HeatInverseProblem(**HIP_options)


    ########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########

    all_Hd_hmatrix = list()
    all_extras = list()
    for k in all_batch_sizes:
        Hd_hmatrix, extras = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, k,
                                                         hmatrix_tol=hmatrix_rtol, make_positive_definite=True,
                                                         return_extras=True, grid_density_multiplier=0.5)
        all_Hd_hmatrix.append(Hd_hmatrix)
        all_extras.append(extras)


    ########    COMPUTE REGULARIZATION PARAMETER VIA MOROZOV DISCREPANCY PRINCIPLE    ########

    best_Hd_hmatrix = all_Hd_hmatrix[-1]

    R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, best_Hd_hmatrix.bct)

    def solve_inverse_problem(a_reg):
        HIP.regularization_parameter = a_reg

        g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
        H_hmatrix = Hd_hmatrix + a_reg * R0_hmatrix
        iH_hmatrix = H_hmatrix.inv()

        u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                              M=iH_hmatrix.as_linear_operator(),
                                              tol=1e-10, maxiter=500)

        u0 = dl.Function(HIP.V)
        u0.vector()[:] = u0_numpy

        return u0.vector() # dolfin Vector

    a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
                                                             HIP.morozov_discrepancy,
                                                             HIP.noise_Mnorm)


    ########    KRYLOV CONVERGENCE FOR MOROZOV REGULARIZATION PARAMETER    ########

    HIP.regularization_parameter = a_reg_morozov

    Hd_hmatrix = all_Hd_hmatrix[-1]
    H_hmatrix = Hd_hmatrix + a_reg_morozov * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    u0_numpy, _, _ = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix.as_linear_operator(), tol=1e-11,
                               maxiter=1000, track_residuals=True)

    _, _, errors_reg_morozov = custom_cg(HIP.H_linop, -g0_numpy, M=HIP.solve_R_linop, tol=1e-10, maxiter=1000,
                                         x_true=u0_numpy, track_residuals=False)

    _, _, errors_none_morozov = custom_cg(HIP.H_linop, -g0_numpy, tol=1e-10, maxiter=1000,
                                          x_true=u0_numpy, track_residuals=False)

    all_errors_hmatrix_morozov = list()
    for kk_batch in range(len(all_batch_sizes)):
        Hd_hmatrix = all_Hd_hmatrix[kk_batch]
        H_hmatrix = Hd_hmatrix + a_reg_morozov * R0_hmatrix
        iH_hmatrix = H_hmatrix.inv()
        _, _, errors_hmatrix_morozov = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix.as_linear_operator(),
                                                 tol=1e-10, maxiter=1000,
                                                 x_true=u0_numpy, track_residuals=False)
        all_errors_hmatrix_morozov.append(errors_hmatrix_morozov)

    if save_results:
        # Save options
        with open(options_file, 'w') as f:
            print(HIP_options, file=f)

        # Save data
        np.savez(str(save_dir / 'error_vs_krylov_iter.npz'),
                 errors_reg_morozov=errors_reg_morozov,
                 errors_none_morozov=errors_none_morozov,
                 all_errors_hmatrix_morozov=all_errors_hmatrix_morozov,
                 a_reg_morozov=a_reg_morozov)
else:
    data = np.load(data_file)
    errors_reg_morozov = data['errors_reg_morozov']
    errors_none_morozov = data['errors_none_morozov']
    all_errors_hmatrix_morozov = data['all_errors_hmatrix_morozov']
    a_reg_morozov = data['a_reg_morozov']
    all_batch_sizes = data['all_batch_sizes']


########    MAKE FIGURE    ########

plt.figure()
plt.semilogy(errors_reg_morozov)
plt.semilogy(errors_none_morozov)
for errors_hmatrix_morozov in all_errors_hmatrix_morozov:
    plt.semilogy(errors_hmatrix_morozov)
plt.xlabel('Conjugate gradient iteration')
plt.ylabel(r'$\frac{\|u_0 - u_0^*\|}{\|u_0^*\|}$')
plt.title('Convergence of conjugate gradient')
plt.legend(['Reg', 'None'] + ['PCH'+str(nB) for nB in all_batch_sizes])

plt.show()

if save_results:
    plt.savefig(plot_file, bbox_inches='tight', dpi=100)
