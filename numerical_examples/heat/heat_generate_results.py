import numpy as np
import dolfin as dl
import time
import pathlib

from nalger_helper_functions import *
from localpsf.heat_inverse_problem import *


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


########    REG PARAM SWEEP USING CONVENTIONAL REG-CG AND TIGHT TOLERANCE    ########

regularization_parameters = np.logspace(-4,0,10)
u0_reconstructions = list()
morozov_discrepancies = list()
noise_Mnorms = list()
for a_reg in list(regularization_parameters):
    print('a_reg=', a_reg)
    HIP.regularization_parameter = a_reg
    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    u0_numpy, info, residuals = custom_cg(HIP.H_linop, g0_numpy, M=HIP.solve_R_linop, tol=1e-10, maxiter=500)
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

morozov_discrepancies = np.array(morozov_discrepancies)
noise_Mnorms = np.array(noise_Mnorms)


########    COMPUTE MOROZ DISCREPANCIES    ########

morozov_discrepancies = np.zeros(range(len(regularization_parameters)))
for k in range(len(regularization_parameters)):
    a_reg = regularization_parameters[k]
    u0 = u0_reconstructions[k]
    Jd = HIP.misfit_objective(u0.vector())
    Jr = HIP.regularization_objective(u0.vector())