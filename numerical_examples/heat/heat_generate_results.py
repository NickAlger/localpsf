import numpy as np
import dolfin as dl
import time
import pathlib

from nalger_helper_functions import *
import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix


########    OPTIONS    ########

HIP_options = {'mesh_h' : 3e-2,
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
num_reg = 10

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


########    REG PARAM SWEEP    ########

Hd_hmatrix = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, 5,
                                         hmatrix_tol=hmatrix_rtol, make_positive_definite=True, return_extras=False)

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, Hd_hmatrix.bct)

def iterations_to_achieve_tolerance(errors, tol):
    successful_iterations = np.argwhere(np.array(errors) < tol)
    if len(successful_iterations > 0):
        first_successful_iteration = successful_iterations[0, 0] + 1
    else:
        first_successful_iteration = -1
    return first_successful_iteration

regularization_parameters = np.logspace(np.log10(a_reg_min), np.log10(a_reg_max), num_reg)
u0_reconstructions = list()
morozov_discrepancies = list()
noise_Mnorms = list()
all_cg_reg_iters = list()
all_cg_hmatrix_iters = list()
all_cg_none_iters = list()
for a_reg in list(regularization_parameters):
    print('a_reg=', a_reg)
    HIP.regularization_parameter = a_reg
    H_hmatrix = Hd_hmatrix + a_reg * R0_hmatrix
    # iH_hmatrix = hpro.h_lu(H_hmatrix)
    iH_hmatrix = H_hmatrix.inv()
    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    # u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
    #                                       M=iH_hmatrix.as_linear_operator(inverse=True), tol=1e-10, maxiter=500)
    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix.as_linear_operator(), tol=1e-10, maxiter=500)

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

    _, _, errors_reg = custom_cg(HIP.H_linop, -g0_numpy, M=HIP.solve_R_linop, tol=1e-10, maxiter=500,
                             x_true=u0_numpy, track_residuals=False)

    cg_reg_iters = iterations_to_achieve_tolerance(errors_reg, krylov_tol)
    all_cg_reg_iters.append(cg_reg_iters)

    _, _, errors_hmatrix = custom_cg(HIP.H_linop, -g0_numpy, M=iH_hmatrix.as_linear_operator(), tol=1e-10, maxiter=500,
                             x_true=u0_numpy, track_residuals=False)

    cg_hmatrix_iters = iterations_to_achieve_tolerance(errors_hmatrix, krylov_tol)
    all_cg_hmatrix_iters.append(cg_hmatrix_iters)

    _, _, errors_none = custom_cg(HIP.H_linop, -g0_numpy, tol=1e-10, maxiter=500,
                                  x_true=u0_numpy, track_residuals=False)

    cg_none_iters = iterations_to_achieve_tolerance(errors_none, krylov_tol)
    all_cg_none_iters.append(cg_none_iters)

morozov_discrepancies = np.array(morozov_discrepancies)
noise_Mnorms = np.array(noise_Mnorms)
all_cg_reg_iters = np.array(all_cg_reg_iters)
all_cg_hmatrix_iters = np.array(all_cg_hmatrix_iters)
all_cg_none_iters = np.array(all_cg_none_iters)

# Morozov discrepancy

plt.figure()
plt.loglog(regularization_parameters, morozov_discrepancies)
plt.loglog(regularization_parameters, noise_Mnorms)
plt.xlabel(r'Regularization parameter, $\alpha$')
plt.title('Morozov discrepancy')
plt.legend(['Morozov discrepancy', 'noise norm'])

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

# Krylov iterations to given tolerance

plt.figure()
plt.loglog(regularization_parameters, all_cg_reg_iters)
plt.loglog(regularization_parameters, all_cg_hmatrix_iters)
plt.loglog(regularization_parameters, all_cg_none_iters)
plt.title(r'Conjugate gradient iterations to achieve tolerance $10^{-6}$')
plt.legend(['Regularization', 'PC-Hmatrix', 'None'])
plt.xlabel(r'Regularization parameter, $\alpha$')
plt.ylabel(r'$\frac{\|u_0 - u_0^*\|}{\|u_0^*\|}$')


