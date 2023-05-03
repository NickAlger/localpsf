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

newton_rtol = 1e-8

forcing_sequence_power = 0.5
num_gn_iter = 5

run_finite_difference_checks = False
check_gauss_newton_hessian = False

recompute_morozov_aregs = True
recompute_ncg_convergence_comparison = True

all_num_batches = [1, 5, 25] # number of batches used for spectral comparisons

fig_dpi = 200

save_dir = localpsf_root / 'numerical_examples' / 'stokes' / 'data' / 'stokes_deflation2'
save_dir.mkdir(parents=True, exist_ok=True)
save_dir_str = str(save_dir)

options_string = 'all_noise_levels=' + str(all_noise_levels)
options_string += '\n' + 'primary_noise_level=' + str(primary_noise_level)
options_string += '\n' + 'num_batches_ncg_table=' + str(num_batches_ncg_table)
options_string += '\n' + 'shared_psf_options=' + str(shared_psf_options)
options_string += '\n' + 'psf_preconditioner_build_iters=' + str(psf_preconditioner_build_iters)
options_string += '\n' + 'forcing_sequence_power=' + str(forcing_sequence_power)
options_string += '\n' + 'num_gn_iter=' + str(num_gn_iter)
options_string += '\n' + 'run_finite_difference_checks=' + str(run_finite_difference_checks)
options_string += '\n' + 'check_gauss_newton_hessian=' + str(check_gauss_newton_hessian)
options_string += '\n' + 'all_num_batches=' + str(all_num_batches)
options_string += '\n' + 'fig_dpi=' + str(fig_dpi)
options_string += '\n' + 'random_seed=' + str(random_seed)
options_string += '\n' + 'recompute_morozov_aregs=' + str(recompute_morozov_aregs)
options_string += '\n' + 'recompute_ncg_convergence_comparison=' + str(recompute_ncg_convergence_comparison)

with open(save_dir_str + "/options_used.txt", 'w') as fout:
    fout.write(options_string)

######################    SET UP STOKES INVERSE PROBLEM    ######################

SU = make_stokes_universe(
    display=True,
    run_finite_difference_checks=run_finite_difference_checks,
    check_gauss_newton_hessian=check_gauss_newton_hessian)

N = SU.unregularized_inverse_problem.function_spaces.Vh2.dim()

# Save mesh
dl.File(save_dir_str + "/ice_mountain_mesh3D.xml") << SU.unregularized_inverse_problem.meshes.ice_mesh_3d
dl.File(save_dir_str + "/ice_mountain_mesh_base3D.xml") << SU.unregularized_inverse_problem.meshes.basal_mesh_3d
dl.File(save_dir_str + "/ice_mountain_mesh_base2D.xml") << SU.unregularized_inverse_problem.meshes.basal_mesh_2d

plt.figure()
dl.plot(SU.unregularized_inverse_problem.meshes.basal_mesh_2d)
plt.title('2D Basal Mesh')
plt.savefig(save_dir_str + '/2d_basal_mesh.png', dpi=fig_dpi, bbox_inches='tight')

plt.figure()
cm = dl.plot(SU.unregularized_inverse_problem.mtrue_Vh2(), cmap='gray')
plt.colorbar(cm)
plt.title('True parameter')
plt.savefig(save_dir_str + '/true_parameter.png', dpi=fig_dpi, bbox_inches='tight')

m0_Vh2 = dl.Function(SU.unregularized_inverse_problem.function_spaces.Vh2)
m0_Vh2.vector()[:] = SU.unregularized_inverse_problem.m_Vh2().vector()[:]

plt.figure()
cm = dl.plot(m0_Vh2, cmap='gray')
plt.colorbar(cm)
plt.title('Parameter initial guess')
plt.savefig(save_dir_str + '/parameter_initial_guess.png', dpi=fig_dpi, bbox_inches='tight')

dl.File(save_dir_str + "/mtrue_Vh2.pvd") << SU.unregularized_inverse_problem.mtrue_Vh2()
dl.File(save_dir_str + "/mtrue_Vh3.pvd") << SU.unregularized_inverse_problem.mtrue_Vh2()

dl.File(save_dir_str + "/true_velocity.pvd") << SU.unregularized_inverse_problem.velocity_true()
dl.File(save_dir_str + "/true_pressure.pvd") << SU.unregularized_inverse_problem.pressure_true()


######################    COMPUTE MOROZOV REGULARIZATION PARAMETER FOR DIFFERENT NOISE LEVELS    ######################

if recompute_morozov_aregs:
    all_noise_vecs = []
    all_areg_morozov = []
    for noise_level in all_noise_levels:
        noise_vec = SU.unregularized_inverse_problem.generate_multiplicative_noise(noise_level)
        SU.unregularized_inverse_problem.update_noise(noise_vec)
        all_noise_vecs.append(noise_vec)

        noise_datanorm = SU.unregularized_inverse_problem.noise_datanorm()
        print('noise_level=', noise_level, 'noise_datanorm=', noise_datanorm)

        if all_areg_morozov:
            areg0 = all_areg_morozov[-1]
        else:
            areg0 = SU.areg_ini

        areg_morozov, seen_aregs, seen_discrepancies = SU.compute_morozov_areg(
            areg0, display=True)

        all_areg_morozov.append(areg_morozov)

        print('noise_level=', noise_level, ', areg_morozov=', areg_morozov)
        inds = np.argsort(seen_aregs)

        plt.figure()
        plt.loglog(seen_aregs[inds], seen_discrepancies[inds])
        plt.loglog(areg_morozov, noise_datanorm, '*r')
        plt.xlabel('regularization parameter')
        plt.ylabel('Morozov discrepancy')
        plt.title('noise_level=' + str(noise_level))
        plt.savefig(save_dir_str + '/morozov_'+str(noise_level)+'.png', dpi=fig_dpi, bbox_inches='tight')

    plt.figure()
    plt.loglog(all_noise_levels, all_areg_morozov)
    plt.xlabel('noise level')
    plt.ylabel('Morozov regularization parameter')
    plt.title('Morozov regularization parameter vs. noise level')
    plt.savefig(save_dir_str + '/morozov_vs_noise.png', dpi=fig_dpi, bbox_inches='tight')

    np.savetxt(save_dir_str + "/all_noise_vecs", np.array(all_noise_vecs))
    np.savetxt(save_dir_str + "/all_noise_levels", np.array(all_noise_levels))
    np.savetxt(save_dir_str + "/all_areg_morozov", np.array(all_areg_morozov))
else:
    all_noise_vecs = list(np.loadtxt(save_dir_str + "/all_noise_vecs"))
    all_noise_levels2 = list(np.loadtxt(save_dir_str + "/all_noise_levels"))
    all_areg_morozov = list(np.loadtxt(save_dir_str + "/all_areg_morozov"))

    assert(np.linalg.norm(np.array(all_noise_levels2) - np.array(all_noise_levels))
           <= 1e-10 * np.linalg.norm(np.array(all_noise_levels)))


######################    SOLVE INVERSE PROBLEM WITH DIFFERENT PRECONDITIONERS    ######################

if recompute_ncg_convergence_comparison:
    noise_level = all_noise_levels[primary_ind]
    noise_vec = all_noise_vecs[primary_ind]
    areg_morozov = all_areg_morozov[primary_ind]

    SU.unregularized_inverse_problem.update_noise(noise_vec)

    print('Solving inverse problem for noise_level=', noise_level, ', areg_morozov=', areg_morozov)

    psf_options = shared_psf_options.copy()
    psf_options['num_initial_batches'] = num_batches_ncg_table

    SU.objective.set_optimization_variable(SU.objective.regularization.mu)
    SU.psf_preconditioner.current_preconditioning_type = 'none'
    SU.psf_preconditioner.psf_options = psf_options

    # PSF preconditioning
    ncg_info_psf = SU.solve_inverse_problem(
        areg_morozov,
        forcing_sequence_power=forcing_sequence_power,
        preconditioner_build_iters=psf_preconditioner_build_iters,
        num_gn_iter=num_gn_iter,
        newton_rtol=newton_rtol,
        display=True)

    with open(save_dir_str + "/ncg_convergence_psf.txt", 'w') as fout:
        fout.write(ncg_info_psf.string())

    # No preconditioning
    SU.objective.set_optimization_variable(SU.objective.regularization.mu)
    SU.psf_preconditioner.current_preconditioning_type = 'none'

    ncg_info_none = SU.solve_inverse_problem(
        areg_morozov,
        forcing_sequence_power=forcing_sequence_power,
        preconditioner_build_iters=tuple(),
        num_gn_iter=num_gn_iter,
        newton_rtol=newton_rtol,
        display=True)

    with open(save_dir_str + "/ncg_convergence_none.txt", 'w') as fout:
        fout.write(ncg_info_none.string())

    # Regularization preconditioning
    SU.objective.set_optimization_variable(SU.objective.regularization.mu)
    SU.psf_preconditioner.current_preconditioning_type = 'reg'

    ncg_info_reg = SU.solve_inverse_problem(
        areg_morozov,
        forcing_sequence_power=forcing_sequence_power,
        preconditioner_build_iters=tuple(),
        num_gn_iter=num_gn_iter,
        newton_rtol=newton_rtol,
        display=True)

    with open(save_dir_str + "/ncg_convergence_reg.txt", 'w') as fout:
        fout.write(ncg_info_reg.string())


######################    SPECTRAL PLOTS, PCG CONVERGENCE, CONDITION NUMBERS    ######################

print('Computing spectral plots, PCG convergence, condition numbers')

num_noises = len(all_noise_levels)
num_psfs = len(all_num_batches)

all_mstar = []

all_ee_none     = np.zeros((num_noises, N))
all_ee_reg      = np.zeros((num_noises, N))
all_ee_psf      = np.zeros((num_noises, num_psfs, N))

all_ee_none_gn  = np.zeros((num_noises, N))
all_ee_reg_gn   = np.zeros((num_noises, N))
all_ee_psf_gn   = np.zeros((num_noises, num_psfs, N))

all_cond_none   = np.zeros(num_noises)
all_cond_reg    = np.zeros(num_noises)
all_cond_psf    = np.zeros((num_noises, num_psfs))

all_cond_none_gn    = np.zeros(num_noises)
all_cond_reg_gn     = np.zeros(num_noises)
all_cond_psf_gn     = np.zeros((num_noises, num_psfs))

all_errs_none = []
all_errs_reg = []
all_errs_psf = []

b = np.random.randn(N)

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

    H_dense = build_dense_matrix_from_matvecs(
        lambda x: SU.objective.apply_hessian(x, areg_morozov), N)

    Hgn_dense = build_dense_matrix_from_matvecs(
        lambda x: SU.objective.apply_gauss_newton_hessian(x, areg_morozov), N)

    HR_dense = build_dense_matrix_from_matvecs(
        lambda x: SU.objective.apply_regularization_hessian(x, areg_morozov), N)

    ee_none = sla.eigh(H_dense)[0][::-1]
    ee_reg = sla.eigh(H_dense, HR_dense)[0][::-1]

    ee_none_gn = sla.eigh(Hgn_dense)[0][::-1]
    ee_reg_gn = sla.eigh(H_dense, HR_dense)[0][::-1]

    all_ee_none[ii,:] = ee_none
    all_ee_reg[ii,:] = ee_reg

    all_ee_none_gn[ii, :] = ee_none_gn
    all_ee_reg_gn[ii,:] = ee_reg_gn

    all_inv_Hpsf_dense = []
    for jj in range(len(all_num_batches)):
        num_batches = all_num_batches[jj]
        print('num_batches=', num_batches)

        psf_options = shared_psf_options.copy()
        psf_options['num_initial_batches'] = num_batches

        SU.psf_preconditioner.psf_options = psf_options
        SU.psf_preconditioner.build_hessian_preconditioner()
        SU.psf_preconditioner.shifted_inverse_interpolator.display = False
        SU.psf_preconditioner.shifted_inverse_interpolator.insert_new_mu(areg_morozov)
        SU.psf_preconditioner.update_deflation(areg_morozov)

        inv_Hpsf_dense = build_dense_matrix_from_matvecs(
            lambda x: SU.psf_preconditioner.solve_hessian_preconditioner(x, areg_morozov), N)

        all_inv_Hpsf_dense.append(inv_Hpsf_dense)

        Hpsf_dense = np.linalg.inv(inv_Hpsf_dense)

        ee_psf = sla.eigh(H_dense, Hpsf_dense)[0][::-1]
        ee_psf_gn = sla.eigh(Hgn_dense, Hpsf_dense)[0][::-1]

        all_ee_psf[ii,jj,:] = ee_psf
        all_ee_psf_gn[ii, jj, :] = ee_psf

    plt.figure()
    plt.semilogy(ee_none[::-1])
    plt.semilogy(ee_reg[::-1])
    legend = ['None', 'Reg']
    for jj in range(len(all_num_batches)):
        plt.semilogy(all_ee_psf[ii,jj,:])
        legend.append('PSF '+str(all_num_batches[jj]))
        plt.legend(legend)
    plt.ylabel('generalized eigenvalue')
    plt.title('generalized Hessian eigenvalues, noise level=' + str(noise_level))
    plt.savefig(save_dir_str + '/preconditioned_spectrum_noise'+str(noise_level)+'.png', dpi=fig_dpi, bbox_inches='tight')

    print('noise_level=', noise_level)

    cond_none = np.max(ee_none) / np.min(ee_none)
    cond_reg = np.max(ee_reg) / np.min(ee_reg)
    all_cond_none[ii] = cond_none
    all_cond_reg[ii] = cond_reg
    print('cond_none=', cond_none)
    print('cond_reg=', cond_reg)
    for jj in range(len(all_num_batches)):
        cond_psf = np.max(all_ee_psf[ii,jj,:]) / np.min(all_ee_psf[ii,jj,:])
        print('cond_psf=', cond_psf, ', num_batches=', all_num_batches[jj])
        all_cond_psf[ii, jj] = cond_psf

    cond_none_gn = np.max(ee_none_gn) / np.min(ee_none_gn)
    cond_reg_gn = np.max(ee_reg_gn) / np.min(ee_reg_gn)
    all_cond_none_gn[ii] = cond_none_gn
    all_cond_reg_gn[ii] = cond_reg_gn
    print('cond_none_gn=', cond_none_gn)
    print('cond_reg_gn=', cond_reg_gn)
    for jj in range(len(all_num_batches)):
        cond_psf_gn = np.max(all_ee_psf_gn[ii,jj,:]) / np.min(all_ee_psf_gn[ii,jj,:])
        print('cond_psf_gn=', cond_psf_gn, ', num_batches=', all_num_batches[jj])
        all_cond_psf_gn[ii, jj] = cond_psf_gn

    x_true = custom_cg(H_dense, b, M=all_inv_Hpsf_dense[-1], tol=1e-13)[0]

    _, _, _, errs_none = custom_cg(H_dense, b, x_true=x_true, tol=1e-12)
    _, _, _, errs_reg = custom_cg(H_dense, b, M=np.linalg.inv(HR_dense), x_true=x_true, tol=1e-12)
    all_errs_none.append(errs_none)
    all_errs_reg.append(errs_reg)
    all_errs_psf.append([])
    for jj in range(len(all_num_batches)):
        _, _, _, errs_psf = custom_cg(H_dense, b, M=all_inv_Hpsf_dense[jj], x_true=x_true, tol=1e-12)
        all_errs_psf[-1].append(errs_psf)

    plt.figure()
    plt.semilogy(errs_none)
    plt.semilogy(errs_reg)
    legend = ['None', 'Reg']
    for jj in range(len(all_num_batches)):
        plt.semilogy(all_errs_psf[-1][jj])
        legend.append('PSF '+str(all_num_batches[jj]))
    plt.legend(legend)
    plt.xlabel('Iteration')
    plt.ylabel('Relative error')
    plt.title('PCG convergence, noise level=' + str(noise_level))
    plt.savefig(save_dir_str + '/pcg_noise' + str(noise_level) + '.png', dpi=fig_dpi, bbox_inches='tight')


for ii, noise_level in enumerate(all_noise_levels):
    np.savetxt(save_dir_str + "/errs_none_"+str(noise_level), np.array(all_errs_none[ii]))

for ii, noise_level in enumerate(all_noise_levels):
    np.savetxt(save_dir_str + "/errs_reg_" + str(noise_level), np.array(all_errs_reg[ii]))

for ii, noise_level in enumerate(all_noise_levels):
    for jj , nb in enumerate(all_num_batches):
        np.savetxt(save_dir_str + "/errs_psf"+str(nb)+"_noise" + str(noise_level), np.array(all_errs_psf[ii][jj]))


for ii, noise_level in enumerate(all_noise_levels):
    np.savetxt(save_dir_str + "/ee_none_"+str(noise_level), np.array(all_ee_none[ii,:]))

for ii, noise_level in enumerate(all_noise_levels):
    np.savetxt(save_dir_str + "/ee_reg_"+str(noise_level), np.array(all_ee_reg[ii,:]))

for ii, noise_level in enumerate(all_noise_levels):
    for jj , nb in enumerate(all_num_batches):
        np.savetxt(save_dir_str + "/ee_psf"+str(nb)+"_noise" + str(noise_level), np.array(all_ee_psf[ii,jj,:]))


np.savetxt(save_dir_str + "/all_cond_none", np.array(all_cond_none))
np.savetxt(save_dir_str + "/all_cond_reg", np.array(all_cond_reg))

for jj, nb in enumerate(all_num_batches):
    np.savetxt(save_dir_str + "/all_cond_psf" + str(nb), np.array(all_cond_psf[:,jj,:]))

