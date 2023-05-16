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
from localpsf.inverse_problem_objective import PSFHessianPreconditioner
import nalger_helper_functions as nhf

import hlibpro_python_wrapper as hpro
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel


def compute_threshold_crossings(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    '''out[ii]=k where k is first index at which y[k] < threshold[ii].

    In:
        compute_threshold_crossings(np.logspace(-1, -5, 20), np.array([10, 1e-2, 1e-4, 1e-20]))
    Out:
        array([ 0,  5, 15])
    '''
    assert(len(y.shape) == 1)
    assert(len(thresholds.shape) == 1)
    crossings = np.ones(len(thresholds), dtype=int) * np.nan
    for ii, threshold in enumerate(thresholds):
        good_inds = np.argwhere(y < threshold).reshape(-1)
        if len(good_inds) > 0:
            crossing = np.min(good_inds)
            crossings[ii] = crossing
        # print('good_inds=', good_inds)
        # print('y=', y)
        # print('threshold=', threshold)
        # print('crossing=', crossing)
    return crossings

fig_dpi = 200

save_dir = localpsf_root / 'numerical_examples' / 'advection_diffusion' / 'data'
save_dir.mkdir(parents=True, exist_ok=True)
save_dir_str = str(save_dir)

num_checkers = 8

all_noise_levels = [0.05, 0.1, 0.2]
all_kappa = list(np.logspace(-4, -3, 5))
all_t_final = [0.5, 1.0, 2.0]
all_num_batches = [1, 5, 25]

krylov_thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

exact_rtol = 1e-13
tight_rtol = 1e-11
morozov_rtol = 1e-3
maxiter_krylov = 4000

impulse_points = np.array([[0.1, 0.8],
                           [0.75, 0.25],
                           [0.3, 0.6]])

np.savetxt(save_dir_str + '/all_noise_levels.txt', np.array(all_noise_levels))
np.savetxt(save_dir_str + '/all_kappa.txt', np.array(all_kappa))
np.savetxt(save_dir_str + '/all_t_final.txt', np.array(all_t_final))
np.savetxt(save_dir_str + '/all_num_batches.txt', np.array(all_num_batches))
np.savetxt(save_dir_str + '/krylov_thresholds.txt', np.array(krylov_thresholds))
np.savetxt(save_dir_str + '/impulse_points.txt', impulse_points)


#### MAKE INITIAL CONDITION PLOT ####

ADV = adv.make_adv_universe(all_noise_levels[0], all_kappa[0], all_t_final[0],
                            num_checkers_x=num_checkers,
                            num_checkers_y=num_checkers)

plt.figure()
cm = dl.plot(ADV.true_initial_condition_func, cmap='gray')
plt.colorbar(cm)
plt.axis('off')
plt.title('True initial condition')
plt.savefig(save_dir_str + '/true_initial_condition.png', dpi=fig_dpi, bbox_inches='tight')

dl.File(save_dir_str + "/true_initial_condition.pvd") << ADV.true_initial_condition_func
plt.close()


#### MAKE VELOCITY PLOT ####

ADV.wind_object.plot_velocity()
plt.title('Wind velocity')
plt.savefig(save_dir_str + '/wind_velocity.png', dpi=fig_dpi, bbox_inches='tight')

dl.File(save_dir_str + "/wind_velocity.pvd") << ADV.wind_object.velocity
plt.close()

#### INITIALIZE DATA ARRAYS ####

newton_krylov_iters_none = np.ones((len(all_noise_levels),
                                    len(all_kappa),
                                    len(all_t_final),
                                    len(krylov_thresholds)), dtype=int) * np.nan

newton_krylov_iters_reg = np.ones((len(all_noise_levels),
                                   len(all_kappa),
                                   len(all_t_final),
                                   len(krylov_thresholds)), dtype=int) * np.nan

newton_krylov_iters_psf = np.ones((len(all_num_batches),
                                   len(all_noise_levels),
                                   len(all_kappa),
                                   len(all_t_final),
                                   len(krylov_thresholds)), dtype=int) * np.nan

####

randn_krylov_iters_none = np.ones((len(all_noise_levels),
                                   len(all_kappa),
                                   len(all_t_final),
                                   len(krylov_thresholds)), dtype=int) * np.nan

randn_krylov_iters_reg = np.ones((len(all_noise_levels),
                                  len(all_kappa),
                                  len(all_t_final),
                                  len(krylov_thresholds)), dtype=int) * np.nan

randn_krylov_iters_psf = np.ones((len(all_num_batches),
                                  len(all_noise_levels),
                                  len(all_kappa),
                                  len(all_t_final),
                                  len(krylov_thresholds)), dtype=int) * np.nan

####

geigs_none = np.ones((len(all_noise_levels),
                      len(all_kappa),
                      len(all_t_final),
                      ADV.N)) * np.nan

geigs_reg = np.ones((len(all_noise_levels),
                     len(all_kappa),
                     len(all_t_final),
                     ADV.N)) * np.nan

geigs_psf = np.ones((len(all_num_batches),
                     len(all_noise_levels),
                     len(all_kappa),
                     len(all_t_final),
                     ADV.N)) * np.nan

####

all_areg_morozov = np.ones((len(all_noise_levels),
                            len(all_kappa),
                            len(all_t_final))) * np.nan

all_noise_norms = np.ones((len(all_noise_levels),
                           len(all_kappa),
                           len(all_t_final))) * np.nan

####

print('Making row and column cluster trees')
ct = hpro.build_cluster_tree_from_pointcloud(ADV.mesh_and_function_space.dof_coords, cluster_size_cutoff=50)

print('Making block cluster trees')
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

HR_hmatrix = ADV.regularization.Cov.make_invC_hmatrix(bct, 1.0)

####

for ii, ns in enumerate(all_noise_levels):
    for jj, kp in enumerate(all_kappa):
        for kk, tf in enumerate(all_t_final):
            plt.close('all')

            ns_str = '_N=' + np.format_float_scientific(ns, precision=1, exp_digits=1)
            tf_str = '_T=' +np.format_float_scientific(tf, precision=1, exp_digits=1)
            kp_str = '_K=' +np.format_float_scientific(kp, precision=1, exp_digits=1)
            id_str = tf_str + kp_str + ns_str

            ADV = adv.make_adv_universe(ns, kp, tf,
                                        num_checkers_x=num_checkers,
                                        num_checkers_y=num_checkers)

            #### IMPULSE RESPONSE PLOTS ####
            if ii == 0: # impulse responses don't depend on noise
                print('making impulse response pictures')
                plt.figure()
                for ll in range(impulse_points.shape[0]):
                    p_str = '_P' + str(tuple(impulse_points[ll, :]))
                    phi = dl.Function(ADV.Vh)
                    phi.vector()[:] = ADV.get_misfit_hessian_impulse_response(impulse_points[ll,:])

                    cm = dl.plot(phi)
                    plt.colorbar(cm)
                    plt.axis('off')
                    plt.plot(impulse_points[ll,0], impulse_points[ll,1], '*r')
                    plt.title('Impulse response, T='+str(tf)+', kappa='+str(kp))
                    plt.savefig(save_dir_str + '/impulse_response'+p_str+id_str+'.png',
                                dpi=fig_dpi, bbox_inches='tight')

                    dl.File(save_dir_str + '/impulse_response'+p_str+id_str+'.pvd') << phi
                    plt.clf()
                plt.close()

            #### PLOT OBSERVATIONS ####
            plt.figure()
            cm = dl.plot(ADV.obs_func, cmap='gray')
            plt.colorbar(cm)
            plt.axis('off')
            plt.title('noisy observations')
            plt.savefig(save_dir_str + '/noisy_observations'+id_str+'.png', dpi=fig_dpi, bbox_inches='tight')

            dl.File(save_dir_str + '/noisy_observations'+id_str+'.pvd') << ADV.obs_func
            plt.close()

            #### COMPUTE MOROZOV REGULARIZATION PARAMETER ####
            areg_initial_guess = 1e-5
            areg_morozov, seen_aregs, seen_discrepancies = ADV.compute_morozov_regularization_parameter(
                areg_initial_guess, morozov_options={'morozov_rtol' : morozov_rtol})

            inds = np.argsort(seen_aregs)
            seen_aregs = seen_aregs[inds]
            seen_discrepancies = seen_discrepancies[inds]

            print('areg_morozov=', areg_morozov)
            all_areg_morozov[ii,jj,kk] = areg_morozov
            all_noise_norms[ii,jj,kk] = ADV.noise_norm

            np.save(save_dir_str + '/all_areg_morozov', all_areg_morozov)
            np.savetxt(save_dir_str + '/seen_aregs' + id_str + '.txt', seen_aregs)
            np.savetxt(save_dir_str + '/seen_discrepancies' + id_str + '.txt', seen_discrepancies)

            plt.figure()
            plt.loglog(seen_aregs, seen_discrepancies)
            plt.loglog(areg_morozov, ADV.noise_norm, '*r')
            plt.xlabel('regularization parameter')
            plt.ylabel('Morozov discrepancy')
            plt.title('noise_level=' + str(ns))
            plt.savefig(save_dir_str + '/morozov' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')
            plt.close()

            #### SOLVE TO TIGHT TOLERANCE AND PLOT RECONSTRUCTION ####
            print('Solving inverse problem to tight tol with areg_morozov')
            m0 = np.zeros(ADV.N)
            g0 = ADV.gradient(m0, areg_morozov)

            H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg_morozov))
            p_newton = nhf.custom_cg(H_linop, -g0, display=True, maxiter=maxiter_krylov, tol=exact_rtol)[0]

            mstar_vec = m0 + p_newton
            mstar = dl.Function(ADV.Vh)
            mstar.vector()[:] = mstar_vec.copy()

            plt.figure()
            cm = dl.plot(mstar, cmap='gray')
            plt.colorbar(cm)
            plt.axis('off')
            plt.title('Reconstructed initial condition')
            plt.savefig(save_dir_str + '/reconstructed_initial_condition'+id_str+'.png', dpi=fig_dpi, bbox_inches='tight')

            dl.File(save_dir_str + '/reconstructed_initial_condition'+id_str+'.pvd') << mstar
            plt.close()

            #

            print('Solving inverse problem to tight tol with areg_morozov')
            b_randn = np.random.randn(ADV.N)
            x_randn = nhf.custom_cg(H_linop, b_randn, display=True, maxiter=maxiter_krylov, tol=exact_rtol)[0]

            #### KRYLOV CONVERGENCE AND PRECONDITIONED SPECTRUM PLOTS ####
            print('Making Krylov convergence plots')
            M_reg_matvec = lambda x: ADV.regularization.solve_hessian(x, m0, areg_morozov)
            M_reg_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=M_reg_matvec)

            _, _, _, newton_errs_none = nhf.custom_cg(H_linop, -g0, x_true=p_newton, display=True, maxiter=maxiter_krylov, tol=tight_rtol)
            _, _, _, newton_errs_reg = nhf.custom_cg(H_linop, -g0, M=M_reg_linop, x_true=p_newton, display=True, maxiter=maxiter_krylov, tol=tight_rtol)

            np.savetxt(save_dir_str + '/newton_errs_none' + id_str + '.txt', newton_errs_none)
            np.savetxt(save_dir_str + '/newton_errs_reg' + id_str + '.txt', newton_errs_reg)

            newton_crossings_none = compute_threshold_crossings(np.array(newton_errs_none), np.array(krylov_thresholds))
            newton_crossings_reg = compute_threshold_crossings(np.array(newton_errs_reg), np.array(krylov_thresholds))

            newton_krylov_iters_none[ii, jj, kk] = newton_crossings_none
            newton_krylov_iters_reg[ii, jj, kk] = newton_crossings_reg

            np.save(save_dir_str + '/newton_krylov_iters_none', newton_krylov_iters_none)
            np.save(save_dir_str + '/newton_krylov_iters_reg', newton_krylov_iters_reg)

            print('krylov_thresholds=', krylov_thresholds)
            print('newton_crossings_none=', newton_crossings_none)
            print('newton_crossings_reg=', newton_crossings_reg)

            #

            _, _, _, randn_errs_none = nhf.custom_cg(H_linop, b_randn, x_true=x_randn, display=True, maxiter=maxiter_krylov, tol=tight_rtol)
            _, _, _, randn_errs_reg = nhf.custom_cg(H_linop, b_randn, M=M_reg_linop, x_true=x_randn, display=True, maxiter=maxiter_krylov, tol=tight_rtol)

            np.savetxt(save_dir_str + '/randn_errs_none' + id_str + '.txt', randn_errs_none)
            np.savetxt(save_dir_str + '/randn_errs_reg' + id_str + '.txt', randn_errs_reg)

            randn_crossings_none = compute_threshold_crossings(np.array(randn_errs_none), np.array(krylov_thresholds))
            randn_crossings_reg = compute_threshold_crossings(np.array(randn_errs_reg), np.array(krylov_thresholds))

            randn_krylov_iters_none[ii, jj, kk, :] = randn_crossings_none
            randn_krylov_iters_reg[ii, jj, kk, :] = randn_crossings_reg

            np.save(save_dir_str + '/randn_krylov_iters_none', randn_krylov_iters_none)
            np.save(save_dir_str + '/randn_krylov_iters_reg', randn_krylov_iters_reg)

            print('krylov_thresholds=', krylov_thresholds)
            print('randn_crossings_none=', randn_crossings_none)
            print('randn_crossings_reg=', randn_crossings_reg)

            #

            print('Building H_dense')
            H_dense = nhf.build_dense_matrix_from_matvecs(H_linop.matvec, H_linop.shape[1])

            print('computing ee_none')
            ee_none = sla.eigh(H_dense, eigvals_only=True)[::-1]

            geigs_none[ii,jj,kk,:] = ee_none
            np.save(save_dir_str + '/geigs_none', geigs_none)

            print('Building invHR_dense')
            invHR_dense = nhf.build_dense_matrix_from_matvecs(M_reg_linop.matvec, M_reg_linop.shape[1])

            print('Computing HR_dense')
            HR_dense = np.linalg.inv(invHR_dense)
            del invHR_dense

            print('computing ee_reg')
            ee_reg = sla.eigh(H_dense, HR_dense, eigvals_only=True)[::-1]
            del HR_dense

            geigs_reg[ii, jj, kk, :] = ee_reg
            np.save(save_dir_str + '/geigs_reg', geigs_reg)

            #

            all_newton_errs_psf = []
            all_randn_errs_psf = []
            all_ee_psf = []
            for ll, num_batches in enumerate(all_num_batches):
                bt_str = '_B=' + np.format_float_scientific(num_batches, precision=1, exp_digits=1)
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

                _, _, _, newton_errs_psf = nhf.custom_cg(H_linop, -g0, M=P_linop, x_true=p_newton, display=True, maxiter=maxiter_krylov, tol=tight_rtol)
                all_newton_errs_psf.append(newton_errs_psf)

                np.savetxt(save_dir_str + '/newton_errs_psf' + bt_str + id_str + '.txt', newton_errs_psf)
                newton_crossings_psf = compute_threshold_crossings(np.array(newton_errs_psf), np.array(krylov_thresholds))
                newton_krylov_iters_psf[ll, ii, jj, kk, :] = newton_crossings_psf
                np.save(save_dir_str + '/newton_krylov_iters_psf', newton_krylov_iters_psf)

                print('num_batches=', num_batches)
                print('krylov_thresholds=', krylov_thresholds)
                print('newton_crossings_psf=', newton_crossings_psf)

                #

                _, _, _, randn_errs_psf = nhf.custom_cg(H_linop, b_randn, M=P_linop, x_true=x_randn, display=True, maxiter=maxiter_krylov, tol=tight_rtol)
                all_randn_errs_psf.append(randn_errs_psf)

                np.savetxt(save_dir_str + '/randn_errs_psf' + bt_str + id_str + '.txt', randn_errs_psf)
                randn_crossings_psf = compute_threshold_crossings(np.array(randn_errs_psf), np.array(krylov_thresholds))
                randn_krylov_iters_psf[ll, ii, jj, kk, :] = randn_crossings_psf
                np.save(save_dir_str + '/randn_krylov_iters_psf', randn_krylov_iters_psf)

                print('num_batches=', num_batches)
                print('krylov_thresholds=', krylov_thresholds)
                print('randn_crossings_psf=', randn_crossings_psf)

                #

                apply_PSF = lambda x: psf_preconditioner.shifted_inverse_interpolator.apply_shifted_deflated(x, areg_morozov)

                print('Building PSF_dense')
                PSF_dense = nhf.build_dense_matrix_from_matvecs(apply_PSF, P_linop.shape[1])

                print('computing ee_psf')
                ee_psf = sla.eigh(H_dense, PSF_dense, eigvals_only=True)[::-1]
                del PSF_dense

                all_ee_psf.append(ee_psf)

                geigs_psf[ll, ii, jj, kk, :] = ee_psf
                np.save(save_dir_str + '/geigs_psf', geigs_psf)

            del H_dense

            plt.figure()
            plt.semilogy(newton_errs_none)
            plt.semilogy(newton_errs_reg)
            legend = ['None', 'Reg']
            for jj in range(len(all_num_batches)):
                plt.semilogy(all_newton_errs_psf[jj])
                legend.append('PSF ' + str(all_num_batches[jj]))
            plt.legend(legend)
            plt.xlabel('Iteration')
            plt.ylabel('Relative error')
            plt.title('PCG convergence (Newton system) ' + id_str)
            plt.savefig(save_dir_str + '/pcg_convergence_newton' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.semilogy(randn_errs_none)
            plt.semilogy(randn_errs_reg)
            legend = ['None', 'Reg']
            for jj in range(len(all_num_batches)):
                plt.semilogy(all_randn_errs_psf[jj])
                legend.append('PSF ' + str(all_num_batches[jj]))
            plt.legend(legend)
            plt.xlabel('Iteration')
            plt.ylabel('Relative error')
            plt.title('PCG convergence (randn RHS) ' + id_str)
            plt.savefig(save_dir_str + '/pcg_convergence_randn' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.semilogy(ee_none)
            plt.semilogy(ee_reg)
            legend = ['None', 'Reg']
            for jj in range(len(all_num_batches)):
                plt.semilogy(all_ee_psf[jj])
                legend.append('PSF ' + str(all_num_batches[jj]))
            plt.legend(legend)
            plt.xlabel('index')
            plt.ylabel('eigenvalue')
            plt.title('generalized eigenvalues ' + id_str)
            plt.savefig(save_dir_str + '/generalized_eigenvalues' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')
            plt.close()



