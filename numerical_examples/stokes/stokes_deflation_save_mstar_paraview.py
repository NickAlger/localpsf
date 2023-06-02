import numpy as np
import dolfin as dl
import scipy.linalg as sla
import matplotlib.pyplot as plt
from localpsf import localpsf_root
from nalger_helper_functions import *

from localpsf.stokes_variational_forms import make_stokes_universe


######################    OPTIONS    ######################

random_seed = 0
np.random.seed(random_seed)

all_noise_levels = list(np.logspace(np.log10(1e-2), np.log10(2.5e-1), 5)[::-1])
print('all_noise_levels=', all_noise_levels)

primary_noise_level_target = 0.05
primary_ind = np.argmin(np.abs(np.array(all_noise_levels) - primary_noise_level_target))
primary_noise_level = all_noise_levels[primary_ind]
print('primary_noise_level=', primary_noise_level)

num_batches_ncg_table: int = 5

shared_psf_options = {
    'display' : True,
    'smoothing_width_in': 0.0,
    'smoothing_width_out' : 0.0
}

psf_preconditioner_build_iters=(3,)

newton_rtol = 1e-6

forcing_sequence_power = 0.5
num_gn_iter = 5

fig_dpi = 200

save_dir = localpsf_root / 'numerical_examples' / 'stokes' / 'data' / 'gauss_rbf_e6'
save_dir.mkdir(parents=True, exist_ok=True)
save_dir_str = str(save_dir)

SU = make_stokes_universe(
    display=True,
    run_finite_difference_checks=False,
    check_gauss_newton_hessian=False)

N = SU.unregularized_inverse_problem.function_spaces.Vh2.dim()

all_noise_vecs = list(np.loadtxt(save_dir_str + "/all_noise_vecs"))
all_noise_levels2 = list(np.loadtxt(save_dir_str + "/all_noise_levels"))
all_areg_morozov = list(np.loadtxt(save_dir_str + "/all_areg_morozov"))

assert(np.linalg.norm(np.array(all_noise_levels2) - np.array(all_noise_levels))
       <= 1e-10 * np.linalg.norm(np.array(all_noise_levels)))


all_mstar = []

for ii in range(len(all_noise_levels)):
    noise_level = all_noise_levels[ii]
    noise_vec = all_noise_vecs[ii]
    areg_morozov = all_areg_morozov[ii]

    print('noise level=', noise_level, ', areg_morozov=', areg_morozov)

    SU.unregularized_inverse_problem.update_noise(noise_vec)

    psf_options = shared_psf_options.copy()
    psf_options['num_initial_batches'] = num_batches_ncg_table

    SU.objective.set_optimization_variable(SU.objective.regularization.mu)
    SU.psf_preconditioner.current_preconditioning_type = 'none'
    SU.psf_preconditioner.psf_options = psf_options

    SU.solve_inverse_problem(
        areg_morozov,
        forcing_sequence_power=forcing_sequence_power,
        preconditioner_build_iters=psf_preconditioner_build_iters,
        num_gn_iter=num_gn_iter,
        newton_rtol=newton_rtol,
        display=True)

    mstar = SU.unregularized_inverse_problem.m_Vh2()
    all_mstar.append(mstar)

    plt.figure()
    cm = dl.plot(mstar, cmap='gray')
    plt.colorbar(cm)
    plt.title('Optimal m, noise level=' + str(noise_level))
    plt.savefig(save_dir_str + '/mstar_'+str(noise_level)+'.png', dpi=fig_dpi, bbox_inches='tight')

    dl.File(save_dir_str + "/mstar_"+str(noise_level)+".pvd") << mstar


