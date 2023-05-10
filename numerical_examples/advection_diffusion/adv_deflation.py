import dolfin as dl
import ufl
import math
import numpy as np
import typing as typ
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import cached_property

import localpsf.advection_diffusion as adv
from localpsf import localpsf_root

from scipy.spatial import KDTree
import scipy.sparse.linalg as spla
import scipy.linalg as sla
# from localpsf.bilaplacian_regularization import BiLaplacianRegularization
from localpsf.bilaplacian_regularization_lumped import BilaplacianRegularization, BiLaplacianCovariance, make_bilaplacian_covariance
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
from localpsf.inverse_problem_objective import PSFHessianPreconditioner
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
import nalger_helper_functions as nhf

import hlibpro_python_wrapper as hpro
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel

fig_dpi = 200

save_dir = localpsf_root / 'numerical_examples' / 'advection_diffusion' / 'data'
save_dir.mkdir(parents=True, exist_ok=True)
save_dir_str = str(save_dir)


#### INITIAL CONDITION AND OBSERVATIONS PLOTS ####

noise_level = 0.1 # 0.05
kappa=1e-4 # 1e-3
t_final = 1.0 #2.0 # 1.0 # 0.5
num_checkers = 8


ADV = adv.make_adv_universe(noise_level, kappa, t_final,
                            num_checkers_x=num_checkers,
                            num_checkers_y=num_checkers)

#

plt.figure()
cm = dl.plot(ADV.true_initial_condition_func, cmap='gray')
plt.colorbar(cm)
plt.axis('off')
plt.title('True initial condition')
plt.savefig(save_dir_str + '/true_initial_condition.png', dpi=fig_dpi, bbox_inches='tight')

dl.File(save_dir_str + "/true_initial_condition.pvd") << ADV.true_initial_condition_func

#

plt.figure()
cm = dl.plot(ADV.obs_func, cmap='gray')
plt.colorbar(cm)
plt.axis('off')
plt.title('Noisy observations at time T='+str(t_final))
plt.savefig(save_dir_str + '/noisy_observations.png', dpi=fig_dpi, bbox_inches='tight')

dl.File(save_dir_str + "/noisy_observations.pvd") << ADV.obs_func


#### IMPULSE RESPONSE PLOTS ####

# all_kappa = [1e-4]
# all_t_final = [2.0]
all_kappa = list(np.logspace(-4, -3, 3))
all_t_final = [0.5, 1.0, 2.0]

# p = np.array([0.292, 0.842])
# p = np.array([0.3, 0.85])
p = np.array([0.3, 0.8]) # <-- Good upper left
# p = np.array([0.75, 0.25])

for kp in all_kappa:
    for tf  in all_t_final:
        ADV = adv.make_adv_universe(noise_level, kp, tf,
                                    num_checkers_x=num_checkers,
                                    num_checkers_y=num_checkers)

        phi = dl.Function(ADV.Vh)
        phi.vector()[:] = ADV.get_misfit_hessian_impulse_response(p)

        plt.figure()
        cm = dl.plot(phi)
        plt.colorbar(cm)
        plt.axis('off')
        plt.plot(p[0], p[1], '*r')
        plt.title('Impulse response, T='+str(tf)+', kappa='+str(kp))
        plt.savefig(save_dir_str + '/impulse_response_T'+str(tf)+'_k'+str(kp)+'.png', dpi=fig_dpi, bbox_inches='tight')

        dl.File(save_dir_str + '/impulse_response_T'+str(tf)+'_k'+str(kp)+'.pvd') << phi


#### MOROZOV DISCREPANCY ####

ADV = adv.make_adv_universe(noise_level, kappa, t_final,
                            num_checkers_x=num_checkers,
                            num_checkers_y=num_checkers)

def compute_morozov_discrepancy(areg: float) -> float:
    m0 = np.zeros(ADV.N)
    g0 = ADV.gradient(m0, areg)

    H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg))
    result = nhf.custom_cg(H_linop, -g0, display=True, maxiter=2000, tol=1e-4)

    mstar_vec = m0 + result[0]
    predicted_obs = ADV.parameter_to_observable_map(mstar_vec)

    discrepancy = np.linalg.norm(predicted_obs - ADV.obs)
    noise_discrepancy = np.linalg.norm(ADV.noise)

    print('areg=', areg, ', discrepancy=', discrepancy, ', noise_discrepancy=', noise_discrepancy)
    return discrepancy


areg_morozov, seen_aregs, seen_discrepancies = compute_morozov_regularization_parameter(
    1e-5, compute_morozov_discrepancy, ADV.noise_norm, display=True)

print('areg_morozov=', areg_morozov)

inds = np.argsort(seen_aregs)

plt.figure()
plt.loglog(seen_aregs[inds], seen_discrepancies[inds])
plt.loglog(areg_morozov, ADV.noise_norm, '*r')
plt.xlabel('regularization parameter')
plt.ylabel('Morozov discrepancy')
plt.title('noise_level=' + str(noise_level))
plt.savefig(save_dir_str + '/morozov_' + str(noise_level) + '.png', dpi=fig_dpi, bbox_inches='tight')


#### SOLVE TO TIGHT TOLERANCE AND PLOT RECONSTRUCTION ####

m0 = np.zeros(ADV.N)
g0 = ADV.gradient(m0, areg_morozov)

H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg_morozov))
result = nhf.custom_cg(H_linop, -g0, display=True, maxiter=3000, tol=1e-13)

mstar_vec = m0 + result[0]
mstar = dl.Function(ADV.Vh)
mstar.vector()[:] = mstar_vec.copy()

plt.figure()
cm = dl.plot(mstar, cmap='gray')
plt.colorbar(cm)
plt.axis('off')
plt.title('Reconstructed initial condition')
plt.savefig(save_dir_str + '/reconstructed_initial_condition.png', dpi=fig_dpi, bbox_inches='tight')

dl.File(save_dir_str + '/reconstructed_initial_condition.pvd') << mstar

#### KRYLOV CONVERGENCE PLOTS ####

M_reg_matvec = lambda x: ADV.regularization.solve_hessian(x, m0, areg_morozov)
M_reg_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=M_reg_matvec)

_, _, _, errs_none = nhf.custom_cg(H_linop, -g0, x_true=mstar_vec, display=True, maxiter=3000, tol=1e-12)
_, _, _, errs_reg = nhf.custom_cg(H_linop, -g0, M=M_reg_linop, x_true=mstar_vec, display=True, maxiter=3000, tol=1e-12)

#

print('Making row and column cluster trees')
ct = hpro.build_cluster_tree_from_pointcloud(ADV.mesh_and_function_space.dof_coords, cluster_size_cutoff=50)

print('Making block cluster trees')
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

HR_hmatrix = ADV.regularization.Cov.make_invC_hmatrix(bct, 1.0)

all_num_batches = [1, 5, 25]
all_errs_psf = []
for num_batches in all_num_batches:
    psf_preconditioner = PSFHessianPreconditioner(
        ADV.apply_misfit_hessian, ADV.Vh, ADV.mesh_and_function_space.mass_lumps,
        HR_hmatrix, display=True)

    psf_preconditioner.psf_options['num_initial_batches'] = num_batches
    psf_preconditioner.build_hessian_preconditioner()

    psf_preconditioner.shifted_inverse_interpolator.insert_new_mu(areg_morozov)
    psf_preconditioner.update_deflation(areg_morozov)

    P_linop = spla.LinearOperator(
        (ADV.N, ADV.N),
        matvec=lambda x: psf_preconditioner.solve_hessian_preconditioner(x, areg_morozov))

    _, _, _, errs_psf = nhf.custom_cg(H_linop, -g0, M=P_linop, x_true=mstar_vec, display=True, maxiter=3000, tol=1e-12)
    all_errs_psf.append(errs_psf)

plt.figure()
plt.semilogy(errs_none)
plt.semilogy(errs_reg)
legend = ['None', 'Reg']
for jj in range(len(all_num_batches)):
    plt.semilogy(all_errs_psf[jj])
    legend.append('PSF ' + str(all_num_batches[jj]))
plt.legend(legend)
plt.xlabel('Iteration')
plt.ylabel('Relative error')
plt.title('PCG convergence, noise level=' + str(noise_level))
plt.savefig(save_dir_str + '/pcg' + str(noise_level) + '.png', dpi=fig_dpi, bbox_inches='tight')

# # # #

print('Making row and column cluster trees')
ct = hpro.build_cluster_tree_from_pointcloud(ADV.mesh_and_function_space.dof_coords, cluster_size_cutoff=50)

print('Making block cluster trees')
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

HR_hmatrix = ADV.regularization.Cov.make_invC_hmatrix(bct, 1.0)

psf_preconditioner = PSFHessianPreconditioner(
    ADV.apply_misfit_hessian, ADV.Vh, ADV.mesh_and_function_space.mass_lumps,
    HR_hmatrix, display=True)

psf_preconditioner.psf_options['num_initial_batches'] = 25
psf_preconditioner.build_hessian_preconditioner()

m0 = np.zeros(ADV.N) # np.random.randn(ADV.N) # np.zeros(ADV.N)
areg = 3e-5

J0 = ADV.cost(m0, areg)
print('J0=', J0)
g0 = ADV.gradient(m0, areg)

H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg))
result = nhf.custom_cg(H_linop, -g0, display=True, maxiter=1000, tol=1e-5)

plt.figure()
mstar = dl.Function(ADV.Vh)
mstar.vector()[:] = m0 + result[0]
cm = dl.plot(mstar, cmap='gray')
plt.colorbar(cm)
plt.title('reconstruction, areg='+str(areg))

predicted_obs = ADV.parameter_to_observable_map(mstar.vector()[:])

discrepancy = np.linalg.norm(predicted_obs - ADV.obs)
noise_discrepancy = np.linalg.norm(ADV.noise)

print('discrepancy=', discrepancy)
print('noise_discrepancy=', noise_discrepancy)

#

print('Making row and column cluster trees')
ct = hpro.build_cluster_tree_from_pointcloud(ADV.mesh_and_function_space.dof_coords, cluster_size_cutoff=50)

print('Making block cluster trees')
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

HR_hmatrix = ADV.regularization.Cov.make_invC_hmatrix(bct, 1.0)

psf_preconditioner = PSFHessianPreconditioner(
    ADV.apply_misfit_hessian, ADV.Vh, ADV.mesh_and_function_space.mass_lumps,
    HR_hmatrix, display=True)

psf_preconditioner.build_hessian_preconditioner()

psf_preconditioner.shifted_inverse_interpolator.insert_new_mu(areg)
psf_preconditioner.update_deflation(areg)

# areg = 1e-5 #2e-5 # psf5: 132 iter
P_linop = spla.LinearOperator(
    (ADV.N, ADV.N),
    matvec=lambda x: psf_preconditioner.solve_hessian_preconditioner(x, areg))
H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg))
result2 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=2000, tol=1e-8) # 73 iter

plt.figure()
mstar2 = dl.Function(ADV.Vh)
mstar2.vector()[:] = m0 + result2[0]
cm = dl.plot(mstar2)
plt.colorbar(cm)

#

psf_preconditioner.current_preconditioning_type = 'reg'
result3 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=2000, tol=1e-8) # 932 iter

#

psf_preconditioner.current_preconditioning_type = 'none'
result3 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=2000, tol=1e-8) # 388 iter

#

predicted_obs = ADV.parameter_to_observable_map(mstar2.vector()[:])

discrepancy = np.linalg.norm(predicted_obs - ADV.obs)
noise_discrepancy = np.linalg.norm(ADV.noise)

print('discrepancy=', discrepancy)
print('noise_discrepancy=', noise_discrepancy)


# more (25) batches

# psf_preconditioner.psf_options['num_initial_batches'] = 50 # # 34 iter, areg=3.3e-5, kappa=1e-4, t_final=1.0
psf_preconditioner.psf_options['num_initial_batches'] = 25 # 42 iter, areg=3.3e-5, kappa=1e-4, t_final=1.0
# psf_preconditioner.psf_options['num_initial_batches'] = 5 # 104 iter, areg=3.3e-5, kappa=1e-4, t_final=1.0
# psf_preconditioner.psf_options['num_initial_batches'] = 1 # 196 iter, areg=3.3e-5, kappa=1e-4, t_final=1.0

psf_preconditioner.build_hessian_preconditioner()

psf_preconditioner.shifted_inverse_interpolator.insert_new_mu(areg)
psf_preconditioner.update_deflation(areg)

areg = 3e-5 #3.3e-5
P_linop = spla.LinearOperator(
    (ADV.N, ADV.N),
    matvec=lambda x: psf_preconditioner.solve_hessian_preconditioner(x, areg))
H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg))
result = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=1000, tol=1e-8) # 23 iter


mstar = dl.Function(ADV.Vh)
mstar.vector()[:] = m0 + result[0]
plt.figure()
cm = dl.plot(mstar, cmap='gray')
plt.colorbar(cm)

predicted_obs = ADV.parameter_to_observable_map(mstar.vector()[:])

discrepancy = np.linalg.norm(predicted_obs - ADV.obs)
noise_discrepancy = np.linalg.norm(ADV.noise)

print('areg=', areg)
print('discrepancy=', discrepancy)
print('noise_discrepancy=', noise_discrepancy)

#

psf_preconditioner.current_preconditioning_type = 'reg'
result3 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=1000, tol=1e-8) # 623 iter
# sharper image, 2e-5 areg: 830 iter

#

psf_preconditioner.current_preconditioning_type = 'none'
result3 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=1000, tol=1e-8) # 562 iter
# sharper image, 2e-5 areg: 525 iter

#

psf_preconditioner.current_preconditioning_type = 'psf'
result3 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=1000, tol=1e-8) #196 psf1 # 104 psf5 # 48 iter, PSF25
# sharper image, 2e-5 areg: 132 iter psf5
# sharper image, 2e-5 areg: 54 iter psf25

#
# ADP = AdvectionDiffusionProblem(kappa, t_final, gamma)
#
# cmap = 'gray'  # 'binary_r' #'gist_gray' #'binary' # 'afmhot'
#
# plt.figure(figsize=(5, 5))
# cm = dl.plot(ADP.utrue_final, cmap=cmap)
# plt.axis('off')
# cm.set_clim(0.0, 1.0)
# # cm.extend = 'both'
# plt.colorbar(cm, fraction=0.046, pad=0.04)
# plt.gca().set_aspect('equal')
# plt.show()
#
# plt.set_cmap('viridis')
#
# plt.figure(figsize=(5, 5))
# p = np.array([0.25, 0.75])
# impulse_response0 = dl.Function(Vh)
# impulse_response0.vector()[:] = ADP.get_impulse_response(find_nearest_dof(p))
# cm = dl.plot(impulse_response0)
# plt.colorbar(cm)
# plt.title('impulse response near ' + str(p))
#
#
# ic_func = checkerboard_function(num_checkers_x, num_checkers_y, Vh)
# true_initial_condition = ic_func.vector()
#
# utrue_initial = dl.Function(Vh)
# utrue_initial.vector()[:] = true_initial_condition
#
# cmap = 'gray' #'binary_r' #'gist_gray' #'binary' # 'afmhot'
#
# plt.figure(figsize=(5,5))
# cm = dl.plot(utrue_initial, cmap=cmap)
# plt.axis('off')
# plt.colorbar(cm,fraction=0.046, pad=0.04)
# plt.gca().set_aspect('equal')
# plt.show()
#
