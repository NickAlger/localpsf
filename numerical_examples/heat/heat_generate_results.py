import numpy as np
import dolfin as dl
import time
import pathlib

from nalger_helper_functions import *
import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.impulse_response_batches import visualize_impulse_response_batch
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix


########    OPTIONS    ########

HIP_options = {'mesh_h' : 2e-2,
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
all_batch_sizes = [1,5,10]

save_dir = get_project_root() / 'numerical_examples' / 'heat' / time.ctime()
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
                                                     return_extras=True)
    all_Hd_hmatrix.append(Hd_hmatrix)
    all_extras.append(extras)

# Plot impulse response batches

for k in range(len(all_batch_sizes)):
    extras = all_extras[k]
    ff_batches = extras['ff_batches']
    mu_batches = extras['mu_batches']
    Sigma_batches = extras['Sigma_batches']
    point_batches = extras['point_batches']
    tau = extras['tau']
    for b in range(len(ff_batches)):
        visualize_impulse_response_batch(ff_batches[k], point_batches[k], mu_batches[k], Sigma_batches[k], tau)


# Hd_hmatrix1, extras1 = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, 1,
#                                           hmatrix_tol=hmatrix_rtol, make_positive_definite=True, return_extras=True)
#
# Hd_hmatrix5, extras5 = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, 5,
#                                           hmatrix_tol=hmatrix_rtol, make_positive_definite=True, return_extras=True)
#
# Hd_hmatrix10, extras10 = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, 10,
#                                           hmatrix_tol=hmatrix_rtol, make_positive_definite=True, return_extras=True)


########    REG PARAM SWEEP    ########

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, Hd_hmatrix1.bct)

def iterations_to_achieve_tolerance(errors, tol):
    successful_iterations = np.argwhere(np.array(errors) < tol)
    if len(successful_iterations > 0):
        first_successful_iteration = successful_iterations[0, 0] + 1
    else:
        first_successful_iteration = np.nan
    return first_successful_iteration

regularization_parameters = np.logspace(np.log10(a_reg_min), np.log10(a_reg_max), num_reg)
u0_reconstructions = list()
morozov_discrepancies = list()
noise_Mnorms = list()
all_cg_reg_iters = list()
all_cg_hmatrix1_iters = list()
all_cg_hmatrix5_iters = list()
all_cg_hmatrix10_iters = list()
all_cg_none_iters = list()
for a_reg in list(regularization_parameters):
    print('a_reg=', a_reg)
    HIP.regularization_parameter = a_reg

    H_hmatrix1 = Hd_hmatrix1 + a_reg * R0_hmatrix
    H_hmatrix5 = Hd_hmatrix5 + a_reg * R0_hmatrix
    H_hmatrix10 = Hd_hmatrix10 + a_reg * R0_hmatrix

    iH_hmatrix1 = H_hmatrix1.inv()
    iH_hmatrix5 = H_hmatrix5.inv()
    iH_hmatrix10 = H_hmatrix10.inv()

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix10.as_linear_operator(), tol=1e-10, maxiter=500)

    u0 = dl.Function(HIP.V)
    u0.vector()[:] = u0_numpy
    u0_reconstructions.append(u0)

    morozov_discrepancy = HIP.morozov_discrepancy(u0.vector())
    noise_Mnorm = HIP.noise_Mnorm

    morozov_discrepancies.append(morozov_discrepancy)
    noise_Mnorms.append(noise_Mnorm)

    print('noise_Mnorm=', noise_Mnorm, ', morozov_discrepancy=', morozov_discrepancy)

    plt.figure()
    cm = dl.plot(u0, cmap='gray')
    plt.colorbar(cm)
    plt.title(r'Reconstructed initial condition $u_0$, for $\alpha=$'+str(a_reg))

    _, _, errors_reg = custom_cg(HIP.H_linop, -g0_numpy, M=HIP.solve_R_linop, tol=1e-8, maxiter=1500,
                             x_true=u0_numpy, track_residuals=False)

    cg_reg_iters = iterations_to_achieve_tolerance(errors_reg, krylov_tol)
    all_cg_reg_iters.append(cg_reg_iters)

    _, _, errors_hmatrix1 = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix1.as_linear_operator(), tol=1e-8, maxiter=500,
                                      x_true=u0_numpy, track_residuals=False)

    cg_hmatrix1_iters = iterations_to_achieve_tolerance(errors_hmatrix1, krylov_tol)
    all_cg_hmatrix1_iters.append(cg_hmatrix1_iters)

    _, _, errors_hmatrix5 = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix5.as_linear_operator(), tol=1e-8, maxiter=500,
                                      x_true=u0_numpy, track_residuals=False)

    cg_hmatrix5_iters = iterations_to_achieve_tolerance(errors_hmatrix5, krylov_tol)
    all_cg_hmatrix5_iters.append(cg_hmatrix5_iters)

    _, _, errors_hmatrix10 = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix10.as_linear_operator(), tol=1e-8, maxiter=500,
                                       x_true=u0_numpy, track_residuals=False)

    cg_hmatrix10_iters = iterations_to_achieve_tolerance(errors_hmatrix10, krylov_tol)
    all_cg_hmatrix10_iters.append(cg_hmatrix10_iters)

    _, _, errors_none = custom_cg(HIP.H_linop, -g0_numpy, tol=1e-8, maxiter=1500,
                                  x_true=u0_numpy, track_residuals=False)

    cg_none_iters = iterations_to_achieve_tolerance(errors_none, krylov_tol)
    all_cg_none_iters.append(cg_none_iters)

morozov_discrepancies = np.array(morozov_discrepancies)
noise_Mnorms = np.array(noise_Mnorms)
all_cg_reg_iters = np.array(all_cg_reg_iters)
all_cg_hmatrix1_iters = np.array(all_cg_hmatrix1_iters)
all_cg_hmatrix5_iters = np.array(all_cg_hmatrix5_iters)
all_cg_hmatrix10_iters = np.array(all_cg_hmatrix10_iters)
all_cg_none_iters = np.array(all_cg_none_iters)

# Morozov discrepancy

ind = np.argwhere(morozov_discrepancies - noise_Mnorms > 0)[0,0]
f0 = morozov_discrepancies[ind-1]
f1 = morozov_discrepancies[ind]
x0 = regularization_parameters[ind-1]
x1 = regularization_parameters[ind]

slope_log = (np.log(f1) - np.log(f0)) / (np.log(x1) - np.log(x0))
dx_log = (np.log(noise_Mnorms[0]) - np.log(f0)) / slope_log
x_log = np.log(x0) + dx_log
a_reg_morozov = np.exp(x_log)
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

plt.figure()
plt.loglog(regularization_parameters, all_cg_none_iters)
plt.loglog(regularization_parameters, all_cg_reg_iters)
plt.loglog(regularization_parameters, all_cg_hmatrix1_iters)
plt.loglog(regularization_parameters, all_cg_hmatrix5_iters)
plt.loglog(regularization_parameters, all_cg_hmatrix10_iters)
plt.title(r'Conjugate gradient iterations to achieve tolerance $10^{-6}$')
plt.legend(['None', 'Reg', 'PCH1', 'PCH5', 'PCH10'])
plt.xlabel(r'Regularization parameter, $\alpha$')
plt.ylabel(r'$\frac{\|u_0 - u_0^*\|}{\|u_0^*\|}$')

plt.savefig(str(save_dir / 'krylov_iter_vs_reg_parameter.pdf'), bbox_inches='tight', dpi=100)

np.savez(str(save_dir / 'krylov_iter_vs_reg_parameter.npz'),
         regularization_parameters=regularization_parameters,
         all_cg_reg_iters=all_cg_reg_iters,
         all_cg_hmatrix1_iters=all_cg_hmatrix1_iters,
         all_cg_hmatrix5_iters=all_cg_hmatrix5_iters,
         all_cg_hmatrix10_iters=all_cg_hmatrix10_iters,
         all_cg_none_iters=all_cg_none_iters,
         krylov_tol=krylov_tol)


########    KRYLOV CONVERGENCE FOR FIXED REG PARAM    ########

HIP.regularization_parameter = a_reg_morozov

H_hmatrix1 = Hd_hmatrix1 + a_reg_morozov * R0_hmatrix
H_hmatrix5 = Hd_hmatrix5 + a_reg_morozov * R0_hmatrix
H_hmatrix10 = Hd_hmatrix10 + a_reg_morozov * R0_hmatrix

iH_hmatrix1 = H_hmatrix1.inv()
iH_hmatrix5 = H_hmatrix5.inv()
iH_hmatrix10 = H_hmatrix10.inv()

g0_numpy = HIP.g_numpy(np.zeros(HIP.N))

u0_numpy, _, _ = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix1.as_linear_operator(), tol=1e-11,
                           maxiter=1000, track_residuals=True)

_, _, errors_reg_morozov = custom_cg(HIP.H_linop, -g0_numpy, M=HIP.solve_R_linop, tol=1e-10, maxiter=1000,
                                     x_true=u0_numpy, track_residuals=False)

_, _, errors_hmatrix1_morozov = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix1.as_linear_operator(), tol=1e-10, maxiter=1000,
                                          x_true=u0_numpy, track_residuals=False)

_, _, errors_hmatrix5_morozov = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix5.as_linear_operator(), tol=1e-10, maxiter=1000,
                                          x_true=u0_numpy, track_residuals=False)

_, _, errors_hmatrix10_morozov = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix10.as_linear_operator(), tol=1e-10, maxiter=1000,
                                           x_true=u0_numpy, track_residuals=False)

_, _, errors_none_morozov = custom_cg(HIP.H_linop, -g0_numpy, tol=1e-10, maxiter=1000,
                                      x_true=u0_numpy, track_residuals=False)

plt.figure()
plt.semilogy(errors_reg_morozov)
plt.semilogy(errors_none_morozov)
plt.semilogy(errors_hmatrix1_morozov)
plt.semilogy(errors_hmatrix5_morozov)
plt.semilogy(errors_hmatrix10_morozov)
plt.xlabel('Conjugate gradient iteration')
plt.ylabel(r'$\frac{\|u_0 - u_0^*\|}{\|u_0^*\|}$')
plt.title('Convergence of conjugate gradient')
plt.legend(['Reg', 'None', 'PCH1', 'PCH5', 'PCH10'])

plt.savefig(str(save_dir / 'error_vs_krylov_iter.pdf'), bbox_inches='tight', dpi=100)

np.savez(str(save_dir / 'error_vs_krylov_iter.npz'),
         errors_reg_morozov=errors_reg_morozov,
         errors_hmatrix1_morozov=errors_hmatrix1_morozov,
         errors_hmatrix5_morozov=errors_hmatrix5_morozov,
         errors_hmatrix10_morozov=errors_hmatrix10_morozov,
         errors_none_morozov=errors_none_morozov,
         a_reg_morozov=a_reg_morozov)


########    PRECONDITIONED SPECTRUM PLOTS    ########

k_eig = 1000

delta_hmatrix_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x : HIP.apply_H_numpy(x) - H_hmatrix1 * x)
ee_hmatrix1, _ = spla.eigsh(delta_hmatrix_linop, k=k_eig, M=H_hmatrix1.as_linear_operator(), Minv=iH_hmatrix1.as_linear_operator(), which='LM')
abs_ee_hmatrix1 = np.sort(np.abs(ee_hmatrix1))[::-1]

delta_hmatrix_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x : HIP.apply_H_numpy(x) - H_hmatrix1 * x)
ee_hmatrix5, _ = spla.eigsh(delta_hmatrix_linop, k=k_eig, M=H_hmatrix5.as_linear_operator(), Minv=iH_hmatrix5.as_linear_operator(), which='LM')
abs_ee_hmatrix5 = np.sort(np.abs(ee_hmatrix5))[::-1]

delta_hmatrix_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x : HIP.apply_H_numpy(x) - H_hmatrix1 * x)
ee_hmatrix10, _ = spla.eigsh(delta_hmatrix_linop, k=k_eig, M=H_hmatrix10.as_linear_operator(), Minv=iH_hmatrix10.as_linear_operator(), which='LM')
abs_ee_hmatrix10 = np.sort(np.abs(ee_hmatrix10))[::-1]

delta_reg_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x : HIP.apply_H_numpy(x) - HIP.apply_R_numpy(x))
ee_reg, _ = spla.eigsh(delta_reg_linop, k=k_eig, M=HIP.R_linop, Minv=HIP.solve_R_linop, which='LM')
abs_ee_reg = np.sort(np.abs(ee_reg))[::-1]

plt.figure()
plt.semilogy(abs_ee_reg)
plt.semilogy(abs_ee_hmatrix1)
plt.semilogy(abs_ee_hmatrix5)
plt.semilogy(abs_ee_hmatrix10)
plt.title(r'Absolute values of eigenvalues of $P^{-1}H-I$')
plt.xlabel(r'$i$')
plt.ylabel(r'$|\lambda_i|$')
plt.legend(['Reg', 'PCH1', 'PCH5', 'PCH10'])

plt.savefig(str(save_dir / 'preconditioned_spectrum.pdf'), bbox_inches='tight', dpi=100)

np.savez(str(save_dir / 'preconditioned_spectrum.npz'),
         abs_ee_reg,
         abs_ee_hmatrix1,
         abs_ee_hmatrix5,
         abs_ee_hmatrix10,
         a_reg_morozov=a_reg_morozov)