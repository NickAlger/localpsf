import numpy as np
import dolfin as dl
import time
import pathlib
from tqdm.auto import tqdm

from nalger_helper_functions import *
import hlibpro_python_wrapper as hpro
from localpsf.heat_inverse_problem import *
from localpsf.impulse_response_batches import visualize_impulse_response_batch
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix


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
hmatrix_num_batches = 5
n_random_error_matvecs = 50

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


########    BUILD PC-HMATRIX APPROXIMATION    ########

Hd_hmatrix, extras = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, hmatrix_num_batches,
                                                 hmatrix_tol=hmatrix_rtol, make_positive_definite=True,
                                                 return_extras=True, grid_density_multiplier=0.5)


########    SOLVE INVERSE PROBLEM    ########