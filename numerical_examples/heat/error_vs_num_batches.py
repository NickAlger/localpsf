import numpy as np
import matplotlib.pyplot as plt

import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
from localpsf.estimate_column_errors_randomized import estimate_column_errors_randomized


load_results_from_file = False
save_results = True

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

    hmatrix_rtol = 1e-4
    all_batch_sizes = [1,3,6,9]
    n_random_error_matvecs = 100

    options = {'hmatrix_rtol' : hmatrix_rtol,
               'all_batch_sizes' : all_batch_sizes,
               'n_random_error_matvecs' : n_random_error_matvecs}


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


    ########    COMPUTE RELATIVE ERRORS VIA RANDOMIZED METHOD    ########

    all_relative_Phi_errs = list()
    for k in range(len(all_extras)):
        extras = all_extras[k]
        iM_Hd_iM_hmatrix_T = extras['A_kernel_hmatrix'].T

        relative_Phi_err, _ = estimate_column_errors_randomized(HIP.apply_iM_Hd_iM_numpy,
                                                                lambda x: iM_Hd_iM_hmatrix_T * x,
                                                                HIP.V, n_random_error_matvecs)

        Hd_op = spla.LinearOperator((HIP.N, HIP.N), matvec=HIP.apply_iM_Hd_iM_numpy)

        E_fct = lambda x: HIP.apply_iM_Hd_iM_numpy(x) - iM_Hd_iM_hmatrix_T * x
        E_op = spla.LinearOperator((HIP.N, HIP.N), matvec=E_fct)

        hh, _ = spla.eigsh(Hd_op, which='LM')
        ee, _ = spla.eigsh(E_op, which='LM')

        all_relative_Phi_errs.append(relative_Phi_err)

    all_relative_Phi_errs = np.array(all_relative_Phi_errs)


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
                 all_relative_Hd_errs=all_relative_Phi_errs,
                 all_batch_sizes=all_batch_sizes)
else:
    data = np.load(data_file)
    all_relative_Phi_errs = data['all_relative_Phi_errs']
    all_batch_sizes = data['all_batch_sizes']


########    MAKE FIGURE    ########

plt.figure()
plt.plot(all_batch_sizes, all_relative_Phi_errs)

plt.title(r'Error vs. number of batches')
plt.xlabel(r'Number of batches')
plt.ylabel(r'$||\Phi - \Phi_{PC}||_{L^2(\Omega \times \Omega)}}$')

plt.show()

if save_results:
    plt.savefig(plot_file, bbox_inches='tight', dpi=100)
