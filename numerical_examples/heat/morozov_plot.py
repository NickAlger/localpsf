import numpy as np
import matplotlib.pyplot as plt

import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
from localpsf.estimate_column_errors_randomized import estimate_column_errors_randomized
from localpsf import localpsf_root


load_results_from_file = False
save_results = True

save_dir = localpsf_root / 'numerical_examples' / 'heat' / 'morozov_plot'
save_dir.mkdir(parents=True, exist_ok=True)

data_file = str(save_dir / 'data.npz')
HIP_options_file = save_dir / 'HIP_options.txt'
options_file = save_dir / 'options.txt'
plot_file = str(save_dir / 'morozov_plot.pdf')


########    GENERATE OR LOAD RESULTS    ########

if not load_results_from_file:
    ########    OPTIONS    ########

    nondefault_HIP_options = {'mesh_h': 5e-2}

    num_batches = 6
    hmatrix_rtol = 1e-4
    a_reg_min = 1e-5
    a_reg_max = 1e0
    num_reg = 11

    options = {'num_batches' : num_batches,
               'hmatrix_rtol' : hmatrix_rtol,
               'num_reg' : num_reg,
               'a_reg_min' : a_reg_min,
               'a_reg_max' : a_reg_max}


    ########    SET UP HEAT INVERSE PROBLEM    ########

    HIP = HeatInverseProblem(**nondefault_HIP_options)


    ########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########

    Hd_hmatrix, extras = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, num_batches,
                                                     hmatrix_tol=hmatrix_rtol, make_positive_definite=True,
                                                     return_extras=True, grid_density_multiplier=0.5)


    ########    COMPUTE MOROZOV DISCREPANCY FOR MANY REGULARIZATION PARAMETERS    ########

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

        return u0.vector() # dolfin Vector

    regularization_parameters = np.logspace(np.log10(a_reg_min), np.log10(a_reg_max), num_reg)
    morozov_discrepancies = np.zeros(len(regularization_parameters))
    noise_Mnorms = np.ones(len(regularization_parameters)) * HIP.noise_Mnorm
    for kk_reg in range(len(regularization_parameters)):
        a_reg = regularization_parameters[kk_reg]
        u0_vec = solve_inverse_problem(a_reg)
        morozov_discrepancy = HIP.morozov_discrepancy(u0_vec)
        morozov_discrepancies[kk_reg] = morozov_discrepancy
        print('a_reg=', a_reg, ', noise_Mnorm=', noise_Mnorms[kk_reg], ', morozov_discrepancy=', morozov_discrepancy)

    ########    COMPUTE MOROZOV REGULARIZATION PARAMETER    ########

    gtinds = np.argwhere(morozov_discrepancies > HIP.noise_Mnorm)
    ltinds = np.argwhere(morozov_discrepancies <= HIP.noise_Mnorm)

    bracket_min = np.max(regularization_parameters[ltinds])
    bracket_max = np.min(regularization_parameters[gtinds])

    a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
                                                             HIP.morozov_discrepancy,
                                                             HIP.noise_Mnorm,
                                                             a_reg_min=bracket_min,
                                                             a_reg_max=bracket_max,
                                                             rtol=1e-2)

    print('a_reg_morozov=', a_reg_morozov)

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
                 regularization_parameters=regularization_parameters,
                 morozov_discrepancies=morozov_discrepancies,
                 noise_Mnorms=noise_Mnorms,
                 a_reg_morozov=a_reg_morozov)
else:
    data = np.load(data_file)
    regularization_parameters = data['regularization_parameters']
    morozov_discrepancies = data['morozov_discrepancies']
    noise_Mnorms = data['noise_Mnorms']
    a_reg_morozov = data['a_reg_morozov']


########    MAKE FIGURE    ########

plt.figure()
plt.loglog(regularization_parameters, morozov_discrepancies)
plt.loglog(regularization_parameters, noise_Mnorms)
plt.plot(a_reg_morozov, noise_Mnorms[0],'*')
plt.xlabel(r'Regularization parameter, $\alpha$')
plt.title('Morozov discrepancy')
plt.legend(['Morozov discrepancy', 'noise norm'])

plt.show()

if save_results:
    plt.savefig(plot_file, bbox_inches='tight', dpi=100)
