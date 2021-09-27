import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
from localpsf.estimate_column_errors_randomized import estimate_column_errors_randomized


load_results_from_file = False
save_results = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'cond_vs_diff'
save_dir.mkdir(parents=True, exist_ok=True)

data_file = str(save_dir / 'data.npz')
HIP_options_file = save_dir / 'HIP_options.txt'
options_file = save_dir / 'options.txt'
plot_file = str(save_dir / 'cond_vs_diff.pdf')


########    GENERATE OR LOAD RESULTS    ########

if not load_results_from_file:
    ########    OPTIONS    ########

    nondefault_HIP_options = {'mesh_h': 3e-2}

    diffusion_time_min = 1e-4
    diffusion_time_max = 5e-3
    num_diffusion_times = 11
    num_batches = 6

    options = {'diffusion_time_min' : diffusion_time_min,
               'diffusion_time_max' : diffusion_time_max,
               'num_diffusion_times' : num_diffusion_times,
               'num_batches' : num_batches}


    ########    GENERATE RESULTS FOR DIFFERENT DIFFUSION TIMES    ########

    all_diffusion_times = np.logspace(np.log10(diffusion_time_min),
                                      np.log10(diffusion_time_max),
                                      num_diffusion_times)
    all_a_reg_morozov = list()

    all_cond_pch = list()
    all_cond_reg = list()
    for T in all_diffusion_times:
        ########    SET UP HEAT INVERSE PROBLEM    ########

        HIP = HeatInverseProblem(**nondefault_HIP_options)
        nondefault_HIP_options['final_time'] = T


        ########    BUILD PC-HMATRIX APPROXIMATIONS    ########

        Hd_hmatrix, extras = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, num_batches,
                                                         make_positive_definite=True,
                                                         return_extras=True)


        ########    COMPUTE REGULARIZATION PARAMETER VIA MOROZOV DISCREPANCY PRINCIPLE    ########

        R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, Hd_hmatrix.bct)

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

            return u0.vector()  # dolfin Vector


        a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
                                                                 HIP.morozov_discrepancy,
                                                                 HIP.noise_Mnorm,
                                                                 a_reg_min=1e-6)

        all_a_reg_morozov.append(a_reg_morozov)

        HIP.regularization_parameter = a_reg_morozov
        H_hmatrix = Hd_hmatrix + a_reg_morozov * R0_hmatrix
        iH_hmatrix = H_hmatrix.inv()


        ########    COMPUTE PRECONDITIONED HESSIAN CONDITION NUMBER    ########

        H_op = HIP.H_linop

        # PCH Preconditioner
        H_pch_op = H_hmatrix.as_linear_operator()
        iH_pch_op = iH_hmatrix.as_linear_operator()

        ee_max_pch, _ = spla.eigsh(H_op, M=H_pch_op, Minv=iH_pch_op, which='LM')
        ee_min_pch, _ = spla.eigsh(H_op, M=H_pch_op, Minv=iH_pch_op, which='SM')

        cond_pch = np.max(ee_max_pch) / np.min(ee_min_pch)

        # Regularization Preconditioner
        R_op = HIP.R_linop
        iR_op = HIP.solve_R_linop

        ee_max_reg, _ = spla.eigsh(H_op, M=R_op, Minv=iR_op, which='LM')
        ee_min_reg, _ = spla.eigsh(H_op, M=R_op, Minv=iR_op, which='SM')

        cond_reg = np.max(ee_max_pch) / np.min(ee_min_pch)

        print('T=', T, ', cond_pch=', cond_pch, ', cond_reg=', cond_reg)

        all_cond_pch.append(cond_pch)
        all_cond_reg.append(cond_reg)


    all_cond_pch = np.array(all_cond_pch)
    all_cond_reg = np.array(all_cond_reg)
    all_a_reg_morozov = np.array(all_a_reg_morozov)


    ########    SAVE RESULTS    ########

    if save_results:
        # Save HIP_options
        with open(HIP_options_file, 'w') as f:
            print(HIP.options, file=f)

        # Save options
        with open(options_file, 'w') as f:
            print(options, file=f)

        # Save data
        np.savez(data_file,
                 all_diffusion_times=all_diffusion_times,
                 all_a_reg_morozov=all_a_reg_morozov,
                 all_cond_pch=all_cond_pch,
                 all_cond_reg=all_cond_reg)
else:
    data = np.load(data_file)
    all_diffusion_times = data['all_diffusion_times']
    all_a_reg_morozov = data['all_a_reg_morozov']
    all_cond_pch = data['all_cond_pch']
    all_cond_reg = data['all_cond_reg']


########    MAKE FIGURE    ########

plt.figure()
plt.plot(all_diffusion_times, all_cond_reg)
plt.plot(all_diffusion_times, all_cond_pch)


plt.title(r'Preconditioned condition number vs. diffusion time')
plt.xlabel(r'Diffusion time')
plt.ylabel(r'Condition number of $P^{-1}H$')
plt.legend(['P = H_{PC}', 'P = R'])

plt.show()

if save_results:
    plt.savefig(plot_file, bbox_inches='tight', dpi=100)
