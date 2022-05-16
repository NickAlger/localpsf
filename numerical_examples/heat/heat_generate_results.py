import numpy as np
import dolfin as dl
import time
import pathlib
from tqdm.auto import tqdm

from nalger_helper_functions import *
import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.visualization import visualize_impulse_response_batch, visualize_weighting_function
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix
from localpsf import localpsf_root


########    OPTIONS    ########

HIP_options = {'mesh_h' : 1.5e-2,
               'finite_element_order' : 1,
               'final_time' : 3e-4,
               'num_timesteps' : 35,
               'noise_level' : 5e-2,
               'mesh_type' : 'circle',
               'conductivity_type' : 'wiggly',
               'initial_condition_type' : 'angel_peak',
               'perform_checks' : True,
               'prior_correlation_length' : 0.05,
               'regularization_parameter' : 1e-1}

hmatrix_rtol = 1e-4
krylov_tol = 1e-6
a_reg_min = 1e-5
a_reg_max = 1e0
num_reg = 11
all_batch_sizes = [1,3,6,9]
n_random_error_matvecs = 50

save_dir = localpsf_root / 'numerical_examples' / 'heat' / time.ctime()
save_dir.mkdir(parents=True, exist_ok=True)

options_file = save_dir / 'options.txt'
with open(options_file, 'w') as f:
    print(HIP_options, file=f)


########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**HIP_options)

# Mesh

plt.figure()
dl.plot(HIP.mesh)
plt.title('mesh')
plt.savefig(str(save_dir / 'mesh.pdf'), bbox_inches='tight', dpi=100)

file = dl.XDMFFile(str(save_dir / 'mesh.xdmf'))
file.write(HIP.mesh)
file.close()

# Conductivity kappa

plt.figure()
cm = dl.plot(HIP.kappa, cmap='gray')
plt.colorbar(cm)
plt.title(r'Thermal conductivity, $\kappa$')
plt.savefig(str(save_dir / 'conductivity_kappa.pdf'), bbox_inches='tight', dpi=100)

file = dl.XDMFFile(str(save_dir / 'conductivity_kappa.xdmf'))
file.write(HIP.kappa)
file.close()

# Initial concentration u0

plt.figure()
dl.plot(HIP.u0_true, cmap='gray')
plt.title(r'Initial temperature, $u_0$')
plt.savefig(str(save_dir / 'initial_temperature_u0.pdf'), bbox_inches='tight', dpi=100)

file = dl.XDMFFile(str(save_dir / 'initial_temperature_u0.xdmf'))
file.write(HIP.u0_true)
file.close()

# Final concentration uT

plt.figure()
dl.plot(HIP.uT_true, cmap='gray')
plt.title(r'Final temperature $u_T$')
plt.savefig(str(save_dir / 'final_temperature_uT.pdf'), bbox_inches='tight', dpi=100)

file = dl.XDMFFile(str(save_dir / 'final_temperature_uT.xdmf'))
file.write(HIP.uT_true)
file.close()

# Noisy observations

plt.figure()
dl.plot(HIP.uT_obs, cmap='gray')
plt.title(r'Noisy observations of $u_T$')
plt.savefig(str(save_dir / 'noisy_observations_of_uT.pdf'), bbox_inches='tight', dpi=100)

file = dl.XDMFFile(str(save_dir / ('noisy_observations_of_uT.xdmf')))
file.write(HIP.uT_obs)
file.close()

# Prior greens function

ps = dl.PointSource(HIP.V, dl.Point([0.5, 0.5]), 1.0)
ps_dual_vec = dl.assemble(dl.TestFunction(HIP.V) * dl.Constant(0.0) * dl.dx)
ps.apply(ps_dual_vec)
prior_greens = dl.Function(HIP.V)
prior_greens.vector()[:] = HIP.solve_R_petsc(ps_dual_vec)

plt.figure()
cm = dl.plot(prior_greens)
plt.colorbar(cm)
plt.title('Prior greens function')
plt.savefig(str(save_dir / 'prior_greens.pdf'), bbox_inches='tight', dpi=100)

file = dl.XDMFFile(str(save_dir / ('prior_greens.xdmf')))
file.write(prior_greens)
file.close()


########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########

all_Hd_hmatrix = list()
all_extras = list()
for k in all_batch_sizes:
    Hd_hmatrix, extras = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, k,
                                                     hmatrix_tol=hmatrix_rtol, make_positive_definite=True,
                                                     return_extras=True, grid_density_multiplier=0.5)
    all_Hd_hmatrix.append(Hd_hmatrix)
    all_extras.append(extras)


# Columns error plots

def estimate_column_errors_randomized(apply_A_true_numpy, apply_A_numpy, function_space_V, n_random_error_matvecs):
    ncol_A = function_space_V.dim()
    Y_true = np.zeros((ncol_A, n_random_error_matvecs))
    Y = np.zeros((ncol_A, n_random_error_matvecs))
    for k in tqdm(range(n_random_error_matvecs)):
        omega = np.random.randn(HIP.N)
        Y_true[:, k] = apply_A_true_numpy(omega)
        Y[:, k] = apply_A_numpy(omega)

    norm_A = np.linalg.norm(Y_true) / np.sqrt(n_random_error_matvecs)
    norm_A_err = np.linalg.norm(Y_true - Y) / np.sqrt(n_random_error_matvecs)

    relative_A_err = norm_A_err / norm_A

    A_norm_vec = np.linalg.norm(Y_true, axis=1) / np.sqrt(n_random_error_matvecs)
    A_err_vec = np.linalg.norm(Y_true - Y, axis=1) / np.sqrt(n_random_error_matvecs)
    A_relative_err_vec = A_err_vec / A_norm_vec
    A_relative_err_fct = dl.Function(function_space_V)
    A_relative_err_fct.vector()[:] = A_relative_err_vec

    return relative_A_err, A_relative_err_fct


def column_error_plot(relative_err_fct, point_batches):
    plt.figure()
    cm = dl.plot(relative_err_fct)
    plt.colorbar(cm)

    num_batches = len(point_batches)
    pp = np.vstack(point_batches)
    plt.plot(pp[:, 0], pp[:, 1], '.r')

    plt.title('Hd columns relative error, ' + str(num_batches) + ' batches')


all_relative_Hd_errs = list()
for k in range(len(all_extras)):
    extras = all_extras[k]
    # iM_Hd_iM_hmatrix_T = all_extras[2]['A_kernel_hmatrix'].T
    iM_Hd_iM_hmatrix_T = extras['A_kernel_hmatrix'].T

    relative_Hd_err, Hd_relative_err_fct = estimate_column_errors_randomized(HIP.apply_iM_Hd_iM_numpy,
                                                                             lambda x: iM_Hd_iM_hmatrix_T * x,
                                                                             HIP.V, n_random_error_matvecs)

    all_relative_Hd_errs.append(relative_Hd_err)

    column_error_plot(Hd_relative_err_fct, extras['point_batches'])

all_relative_Hd_errs = np.array(all_relative_Hd_errs)


########    MOROZOV FINDER    ########




########    REG PARAM SWEEPS    ########

# Morozov discrepancy

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, all_Hd_hmatrix[-1].bct)

regularization_parameters = np.logspace(np.log10(a_reg_min), np.log10(a_reg_max), num_reg)
u0_reconstructions = list()
morozov_discrepancies = np.zeros(len(regularization_parameters))
noise_Mnorms = np.zeros(len(regularization_parameters))
Hd_hmatrix = all_Hd_hmatrix[-1]
for kk_reg in range(len(regularization_parameters)):
    a_reg = regularization_parameters[kk_reg]
    print('a_reg=', a_reg)
    HIP.regularization_parameter = a_reg

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    H_hmatrix = Hd_hmatrix + a_reg * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix.as_linear_operator(),
                                          tol=1e-10, maxiter=500)

    u0 = dl.Function(HIP.V)
    u0.vector()[:] = u0_numpy
    u0_reconstructions.append(u0)

    morozov_discrepancy = HIP.morozov_discrepancy(u0.vector())
    noise_Mnorm = HIP.noise_Mnorm

    morozov_discrepancies[kk_reg] = morozov_discrepancy
    noise_Mnorms[kk_reg] = noise_Mnorm

    print('noise_Mnorm=', noise_Mnorm, ', morozov_discrepancy=', morozov_discrepancy)

    plt.figure()
    cm = dl.plot(u0, cmap='gray')
    plt.colorbar(cm)
    plt.title(r'Reconstructed initial condition $u_0$, for $\alpha=$'+str(a_reg))


a_reg_morozov = inverse_of_monotone_increasing_piecewise_loglinear_function(noise_Mnorms[0],
                                                                            regularization_parameters,
                                                                            morozov_discrepancies)

print('a_reg_morozov=', a_reg_morozov)

plt.figure()
plt.loglog(regularization_parameters, morozov_discrepancies)
plt.loglog(regularization_parameters, noise_Mnorms)
plt.xlabel(r'Regularization parameter, $\alpha$')
plt.title('Morozov discrepancy')
plt.legend(['Morozov discrepancy', 'noise norm'])

plt.savefig(str(save_dir / 'morozov.pdf'), bbox_inches='tight', dpi=100)

np.savez(str(save_dir / 'morozov.npz'),
         regularization_parameters=regularization_parameters,
         morozov_discrepancies=morozov_discrepancies,
         noise_Mnorms=noise_Mnorms,
         a_reg_morozov=a_reg_morozov)

# Krylov iterations to given tolerance

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

plt.figure()
plt.loglog(regularization_parameters, num_cg_iters_none)
plt.loglog(regularization_parameters, num_cg_iters_reg)
for kk_batch in range(len(all_batch_sizes)):
    plt.loglog(regularization_parameters, all_num_cg_iters_hmatrix[:, kk_batch])

plt.title(r'Conjugate gradient iterations to achieve tolerance $10^{-6}$')
plt.legend(['None', 'Reg'] + ['PCH'+str(nB) for nB in all_batch_sizes])
plt.xlabel(r'Regularization parameter, $\alpha$')
plt.ylabel(r'$\frac{\|u_0 - u_0^*\|}{\|u_0^*\|}$')

plt.savefig(str(save_dir / 'krylov_iter_vs_reg_parameter.pdf'), bbox_inches='tight', dpi=100)

np.savez(str(save_dir / 'krylov_iter_vs_reg_parameter.npz'),
         regularization_parameters=regularization_parameters,
         num_cg_iters_reg=num_cg_iters_reg,
         num_cg_iters_none=num_cg_iters_none,
         all_num_cg_iters_hmatrix=all_num_cg_iters_hmatrix,
         krylov_tol=krylov_tol)


########    KRYLOV CONVERGENCE FOR FIXED REG PARAM    ########

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


plt.figure()
plt.semilogy(errors_reg_morozov)
plt.semilogy(errors_none_morozov)
for errors_hmatrix_morozov in all_errors_hmatrix_morozov:
    plt.semilogy(errors_hmatrix_morozov)
plt.xlabel('Conjugate gradient iteration')
plt.ylabel(r'$\frac{\|u_0 - u_0^*\|}{\|u_0^*\|}$')
plt.title('Convergence of conjugate gradient')
plt.legend(['Reg', 'None'] + ['PCH'+str(nB) for nB in all_batch_sizes])
plt.savefig(str(save_dir / 'error_vs_krylov_iter.pdf'), bbox_inches='tight', dpi=100)

np.savez(str(save_dir / 'error_vs_krylov_iter.npz'),
         errors_reg_morozov=errors_reg_morozov,
         errors_none_morozov=errors_none_morozov,
         all_errors_hmatrix_morozov=all_errors_hmatrix_morozov,
         a_reg_morozov=a_reg_morozov)


########    PRECONDITIONED SPECTRUM PLOT    ########

num_eigs = 1000

all_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
all_abs_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
for kk_batch in range(len(all_batch_sizes)):
    Hd_hmatrix = all_Hd_hmatrix[kk_batch]
    H_hmatrix = Hd_hmatrix + a_reg_morozov * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()
    delta_hmatrix_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x : HIP.apply_H_numpy(x) - H_hmatrix * x)
    ee_hmatrix, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=H_hmatrix.as_linear_operator(), Minv=iH_hmatrix.as_linear_operator(), which='LM')
    abs_ee_hmatrix = np.sort(np.abs(ee_hmatrix))[::-1]

    all_ee_hmatrix[kk_batch,:] = ee_hmatrix
    all_abs_ee_hmatrix[kk_batch,:] = abs_ee_hmatrix

delta_reg_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x : HIP.apply_H_numpy(x) - HIP.apply_R_numpy(x))
ee_reg, _ = spla.eigsh(delta_reg_linop, k=num_eigs, M=HIP.R_linop, Minv=HIP.solve_R_linop, which='LM')
abs_ee_reg = np.sort(np.abs(ee_reg))[::-1]

plt.figure()
plt.semilogy(abs_ee_reg)
for abs_ee_hmatrix in all_abs_ee_hmatrix:
    plt.semilogy(abs_ee_hmatrix)

plt.title(r'Absolute values of eigenvalues of $P^{-1}H-I$')
plt.xlabel(r'$i$')
plt.ylabel(r'$|\lambda_i|$')
plt.legend(['Reg'] + ['PCH'+str(nB) for nB in all_batch_sizes])

plt.savefig(str(save_dir / 'preconditioned_spectrum.pdf'), bbox_inches='tight', dpi=100)

np.savez(str(save_dir / 'preconditioned_spectrum.npz'),
         abs_ee_reg=abs_ee_reg,
         all_abs_ee_hmatrix=all_abs_ee_hmatrix,
         a_reg_morozov=a_reg_morozov)


########    PLOT IMPULSE RESPONSES    ########

for k in range(len(all_batch_sizes)):
    extras = all_extras[k]
    ff_batches = extras['ff_batches']
    mu_batches = extras['mu_batches']
    Sigma_batches = extras['Sigma_batches']
    point_batches = extras['point_batches']
    tau = extras['tau']
    for b in range(len(ff_batches)):
        visualize_impulse_response_batch(ff_batches[k], point_batches[k], mu_batches[k], Sigma_batches[k], tau)
        plt.title('Impulse response batch')

plt.figure()
cm = dl.plot(ff_batches[k])

# for c in cm.collections:
#     c.set_edgecolor("face")

plt.savefig(str(save_dir / 'test.png'), bbox_inches='tight', dpi=500)


########    PLOT WEIGHTING FUNCTIONS    ########

vk = 0
vw = 11
visualize_weighting_function(all_extras[vk]['ww'], np.vstack(all_extras[vk]['point_batches']), vw)
