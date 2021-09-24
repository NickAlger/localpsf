import numpy as np
import matplotlib.pyplot as plt

import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter


load_results_from_file = False
save_results = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'preconditioned_spectrum'
save_dir.mkdir(parents=True, exist_ok=True)

data_file = str(save_dir / 'data.npz')
HIP_options_file = save_dir / 'HIP_options.txt'
options_file = save_dir / 'options.txt'
plot_file = str(save_dir / 'preconditioned_spectrum.pdf')


########    GENERATE OR LOAD RESULTS    ########

if not load_results_from_file:
    ########    OPTIONS    ########

    nondefault_HIP_options = {'mesh_h': 3e-2}

    hmatrix_rtol = 1e-4
    all_batch_sizes = [1,3,6,9]
    num_eigs = 1000

    options = {'hmatrix_rtol' : hmatrix_rtol,
               'all_batch_sizes' : all_batch_sizes,
               'num_eigs' : num_eigs}


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


    ########    COMPUTE PRECONDITIONED SPECTRUM    ########

    all_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
    all_abs_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
    for kk_batch in range(len(all_batch_sizes)):
        print('batch size=', all_batch_sizes[kk_batch])
        Hd_hmatrix = all_Hd_hmatrix[kk_batch]
        H_hmatrix = Hd_hmatrix + a_reg_morozov * R0_hmatrix
        iH_hmatrix = H_hmatrix.inv()
        delta_hmatrix_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x: HIP.apply_H_numpy(x) - H_hmatrix * x)
        ee_hmatrix, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=H_hmatrix.as_linear_operator(),
                                   Minv=iH_hmatrix.as_linear_operator(), which='LM')
        abs_ee_hmatrix = np.sort(np.abs(ee_hmatrix))[::-1]

        all_ee_hmatrix[kk_batch, :] = ee_hmatrix
        all_abs_ee_hmatrix[kk_batch, :] = abs_ee_hmatrix

    delta_reg_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x: HIP.apply_H_numpy(x) - HIP.apply_R_numpy(x))
    ee_reg, _ = spla.eigsh(delta_reg_linop, k=num_eigs, M=HIP.R_linop, Minv=HIP.solve_R_linop, which='LM')
    abs_ee_reg = np.sort(np.abs(ee_reg))[::-1]

    if save_results:
        # Save HIP_options
        with open(HIP_options_file, 'w') as f:
            print(HIP.options, file=f)

        # Save options
        with open(options_file, 'w') as f:
            print(options, file=f)

        # Save data
        np.savez(data_file,
                 abs_ee_reg=abs_ee_reg,
                 all_abs_ee_hmatrix=all_abs_ee_hmatrix,
                 a_reg_morozov=a_reg_morozov,
                 all_batch_sizes=all_batch_sizes)
else:
    data = np.load(data_file)
    abs_ee_reg = data['abs_ee_reg']
    all_abs_ee_hmatrix = data['all_abs_ee_hmatrix']
    a_reg_morozov = data['a_reg_morozov']
    all_batch_sizes = data['all_batch_sizes']


########    MAKE FIGURE    ########

plt.figure()
plt.semilogy(abs_ee_reg)
for abs_ee_hmatrix in all_abs_ee_hmatrix:
    plt.semilogy(abs_ee_hmatrix)

plt.title(r'Absolute values of eigenvalues of $P^{-1}H-I$')
plt.xlabel(r'$i$')
plt.ylabel(r'$|\lambda_i|$')
plt.legend(['Reg'] + ['PCH' + str(nB) for nB in all_batch_sizes])

plt.show()

if save_results:
    plt.savefig(plot_file, bbox_inches='tight', dpi=100)
