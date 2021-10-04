import numpy as np
import matplotlib.pyplot as plt

import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
from localpsf.estimate_column_errors_randomized import estimate_column_errors_randomized


load_results_from_file = False
save_results = False

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'error_vs_num_batches'
save_dir.mkdir(parents=True, exist_ok=True)

data_file = str(save_dir / 'data.npz')
HIP_options_file = save_dir / 'HIP_options.txt'
options_file = save_dir / 'options.txt'
plot_file = str(save_dir / 'error_vs_num_batches.pdf')


########    GENERATE OR LOAD RESULTS    ########

if not load_results_from_file:
    ########    OPTIONS    ########

    nondefault_HIP_options = {'mesh_h': 3e-2}

    hmatrix_rtol = 1e-8
    # all_batch_sizes = list(np.arange(9) + 1) #[1,3,6,9]
    all_batch_sizes = [28]
    n_random_error_matvecs = 100
    grid_density_multiplier=0.05
    tau=4
    w_support_rtol=5e-4

    options = {'hmatrix_rtol' : hmatrix_rtol,
               'all_batch_sizes' : all_batch_sizes,
               'n_random_error_matvecs' : n_random_error_matvecs,
               'grid_density_multiplier' : grid_density_multiplier}


    ########    SET UP HEAT INVERSE PROBLEM    ########

    HIP = HeatInverseProblem(**nondefault_HIP_options)


    ########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########

    all_Hd_hmatrix = list()
    all_extras = list()
    for k in all_batch_sizes:
        Hd_hmatrix, extras = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, k,
                                                         hmatrix_tol=hmatrix_rtol, make_positive_definite=True,
                                                         return_extras=True,
                                                         grid_density_multiplier=grid_density_multiplier,
                                                         tau=tau,
                                                         w_support_rtol=w_support_rtol)
        all_Hd_hmatrix.append(Hd_hmatrix)
        all_extras.append(extras)


    ########    COMPUTE RELATIVE ERRORS VIA RANDOMIZED METHOD    ########

    Hd_op = spla.LinearOperator((HIP.N, HIP.N), matvec=HIP.apply_Hd_numpy)
    hh, _ = spla.eigsh(Hd_op, which='LM')
    induced_2norm_Hd = np.max(np.abs(hh))

    all_Hd_rel_err_fro = list()
    all_Hd_rel_err_induced2 = list()
    for k in range(len(all_extras)):
        Hd_pch = all_Hd_hmatrix[k]
        Hd_pch_nonsym = extras['A_hmatrix_nonsym']
        Phi_pch = extras['A_kernel_hmatrix']

        Phi_rel_err_fro, _ = estimate_column_errors_randomized(HIP.apply_iM_Hd_iM_numpy,
                                                              lambda x: Phi_pch * x,
                                                              HIP.V, n_random_error_matvecs)

        print('Phi_rel_err_fro=', Phi_rel_err_fro)

        Hd_rel_err_fro, _ = estimate_column_errors_randomized(HIP.apply_Hd_numpy,
                                                                lambda x: Hd_pch * x,
                                                                HIP.V, n_random_error_matvecs)

        E_fct = lambda x: HIP.apply_Hd_numpy(x) - Hd_pch * x
        E_op = spla.LinearOperator((HIP.N, HIP.N), matvec=E_fct)
        ee, _ = spla.eigsh(E_op, which='LM')
        induced_2norm_E = np.max(np.abs(ee))

        Hd_rel_err_induced2 = induced_2norm_E / induced_2norm_Hd

        print('batch size=', all_batch_sizes[k], ', Hd_rel_err_fro=', Hd_rel_err_fro, ', Hd_rel_err_induced2=', Hd_rel_err_induced2)

        all_Hd_rel_err_fro.append(Hd_rel_err_fro)
        all_Hd_rel_err_induced2.append(Hd_rel_err_induced2)

    all_Hd_rel_err_fro = np.array(all_Hd_rel_err_fro)
    all_Hd_rel_err_induced2 = np.array(all_Hd_rel_err_induced2)


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
                 all_Hd_rel_err_fro=all_Hd_rel_err_fro,
                 all_Hd_rel_err_induced2=all_Hd_rel_err_induced2,
                 all_batch_sizes=all_batch_sizes)
else:
    data = np.load(data_file)
    all_Hd_rel_err_fro = data['all_Hd_rel_err_fro']
    all_Hd_rel_err_induced2 = data['all_Hd_rel_err_induced2']
    all_batch_sizes = data['all_batch_sizes']


########    MAKE FIGURE    ########

plt.figure()
plt.plot(all_batch_sizes, all_Hd_rel_err_induced2)
plt.plot(all_batch_sizes, all_Hd_rel_err_fro)


plt.title(r'Relative error vs. number of batches')
plt.xlabel(r'Number of batches')
plt.ylabel(r'$\frac{||H_d - H_d^\mathrm{PC}||}{||H_d||}$')
plt.legend(['Induced 2-norm', 'Frobenius norm'])

plt.show()

if save_results:
    plt.savefig(plot_file, bbox_inches='tight', dpi=100)
