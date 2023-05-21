import dolfin as dl
import ufl
import math
import numpy as np
import typing as typ
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import cached_property
import sys

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


np.random.seed(0)

fig_dpi = 200

save_dir = localpsf_root / 'numerical_examples' / 'advection_diffusion' / 'data'
save_dir.mkdir(parents=True, exist_ok=True)
save_dir_str = str(save_dir)

num_checkers = 8

exact_rtol = 1e-13
tight_rtol = 1e-11
morozov_rtol = 1e-3
maxiter_krylov = 4000

all_noise_levels = [0.05]  # [0.05, 0.1, 0.2]
all_kappa = list(np.logspace(-4, -3, 5))
all_t_final = [0.5, 1.0, 2.0]
all_num_batches = [1, 5, 25]

krylov_thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

impulse_points = np.array([[0.1, 0.8],
                           [0.75, 0.25],
                           [0.3, 0.6]])

def save2(file, data):
    if data is not None:
        np.save(file, data)

def savetxt2(file, data):
    if data is not None:
        np.savetxt(file, data)

def load2(file):
    try:
        return np.load(file, allow_pickle=True)
    except:
        return None

def loadtxt2(file):
    try:
        return np.loadtxt(file)
    except:
        return None

@dataclass
class ADVData:
    all_noise_levels: np.ndarray = None
    all_kappa: np.ndarray = None
    all_t_final: np.ndarray = None
    all_num_batches: np.ndarray = None
    krylov_thresholds: np.ndarray = None
    impulse_points: np.ndarray = None

    newton_krylov_iters_none: np.ndarray = None
    newton_krylov_iters_reg: np.ndarray = None
    newton_krylov_iters_psf: np.ndarray = None

    randn_krylov_iters_none: np.ndarray = None
    randn_krylov_iters_reg: np.ndarray = None
    randn_krylov_iters_psf: np.ndarray = None

    geigs_none: np.ndarray = None
    geigs_reg: np.ndarray = None
    geigs_psf: np.ndarray = None

    all_areg_morozov: np.ndarray = None
    all_noise_norms: np.ndarray = None

    all_gradients: np.ndarray = None
    all_newton_dirs: np.ndarray = None

    all_randn_b: np.ndarray = None
    all_randn_x: np.ndarray = None

    all_newton_errs_none: np.ndarray = None
    all_newton_errs_reg: np.ndarray = None
    all_newton_errs_psf: np.ndarray = None

    all_randn_errs_none: np.ndarray = None
    all_randn_errs_reg: np.ndarray = None
    all_randn_errs_psf: np.ndarray = None

    def save(me):
        savetxt2(save_dir_str + '/all_noise_levels.txt', me.all_noise_levels)
        savetxt2(save_dir_str + '/all_kappa.txt', me.all_kappa)
        savetxt2(save_dir_str + '/all_t_final.txt', me.all_t_final)
        savetxt2(save_dir_str + '/all_num_batches.txt', me.all_num_batches)
        savetxt2(save_dir_str + '/krylov_thresholds.txt', me.krylov_thresholds)
        savetxt2(save_dir_str + '/impulse_points.txt', me.impulse_points)

        save2(save_dir_str + '/newton_krylov_iters_none', me.newton_krylov_iters_none)
        save2(save_dir_str + '/newton_krylov_iters_reg', me.newton_krylov_iters_reg)
        save2(save_dir_str + '/newton_krylov_iters_psf', me.newton_krylov_iters_psf)

        save2(save_dir_str + '/randn_krylov_iters_none', me.randn_krylov_iters_none)
        save2(save_dir_str + '/randn_krylov_iters_reg', me.randn_krylov_iters_reg)
        save2(save_dir_str + '/randn_krylov_iters_psf', me.randn_krylov_iters_psf)

        save2(save_dir_str + '/geigs_none', me.geigs_none)
        save2(save_dir_str + '/geigs_reg', me.geigs_reg)
        save2(save_dir_str + '/geigs_psf', me.geigs_psf)

        save2(save_dir_str + '/all_areg_morozov', me.all_areg_morozov)
        save2(save_dir_str + '/all_noise_norms', me.all_noise_norms)

        save2(save_dir_str + '/all_gradients', me.all_gradients)
        save2(save_dir_str + '/all_newton_dirs', me.all_newton_dirs)

        save2(save_dir_str + '/all_randn_b', me.all_randn_b)
        save2(save_dir_str + '/all_randn_x', me.all_randn_x)

        save2(save_dir_str + '/all_newton_errs_none', me.all_newton_errs_none)
        save2(save_dir_str + '/all_newton_errs_reg', me.all_newton_errs_reg)
        save2(save_dir_str + '/all_newton_errs_psf', me.all_newton_errs_psf)

        save2(save_dir_str + '/all_randn_errs_none', me.all_randn_errs_none)
        save2(save_dir_str + '/all_randn_errs_reg', me.all_randn_errs_reg)
        save2(save_dir_str + '/all_randn_errs_psf', me.all_randn_errs_psf)

    def load(me):
        me.all_noise_levels = loadtxt2(save_dir_str + '/all_noise_levels.txt').reshape(-1)
        me.all_kappa = loadtxt2(save_dir_str + '/all_kappa.txt').reshape(-1)
        me.all_t_final = loadtxt2(save_dir_str + '/all_t_final.txt').reshape(-1)
        me.all_num_batches = loadtxt2(save_dir_str + '/all_num_batches.txt').reshape(-1)
        me.all_num_batches = np.array(np.rint(me.all_num_batches), dtype=int)

        me.krylov_thresholds = loadtxt2(save_dir_str + '/krylov_thresholds.txt').reshape(-1)
        me.impulse_points = loadtxt2(save_dir_str + '/impulse_points.txt')
        if len(me.impulse_points.shape) == 1:
            me.impulse_points = me.impulse_points.reshape((1,-1))

        me.newton_krylov_iters_none = load2(save_dir_str + '/newton_krylov_iters_none.npy')
        me.newton_krylov_iters_reg = load2(save_dir_str + '/newton_krylov_iters_reg.npy')
        me.newton_krylov_iters_psf = load2(save_dir_str + '/newton_krylov_iters_psf.npy')

        me.randn_krylov_iters_none = load2(save_dir_str + '/randn_krylov_iters_none.npy')
        me.randn_krylov_iters_reg = load2(save_dir_str + '/randn_krylov_iters_reg.npy')
        me.randn_krylov_iters_psf = load2(save_dir_str + '/randn_krylov_iters_psf.npy')

        me.geigs_none = load2(save_dir_str + '/geigs_none.npy')
        me.geigs_reg = load2(save_dir_str + '/geigs_reg.npy')
        me.geigs_psf = load2(save_dir_str + '/geigs_psf.npy')

        me.all_areg_morozov = load2(save_dir_str + '/all_areg_morozov.npy')
        me.all_noise_norms = load2(save_dir_str + '/all_noise_norms.npy')

        me.all_gradients = load2(save_dir_str + '/all_gradients.npy')
        me.all_newton_dirs = load2(save_dir_str + '/all_newton_dirs.npy')

        me.all_randn_b = load2(save_dir_str + '/all_randn_b.npy')
        me.all_randn_x = load2(save_dir_str + '/all_randn_x.npy')

        me.all_newton_errs_none = load2(save_dir_str + '/all_newton_errs_none.npy')
        me.all_newton_errs_reg = load2(save_dir_str + '/all_newton_errs_reg.npy')
        me.all_newton_errs_psf = load2(save_dir_str + '/all_newton_errs_psf.npy')

        me.all_randn_errs_none = load2(save_dir_str + '/all_randn_errs_none.npy')
        me.all_randn_errs_reg = load2(save_dir_str + '/all_randn_errs_reg.npy')
        me.all_randn_errs_psf = load2(save_dir_str + '/all_randn_errs_psf.npy')


def initialize_stuff():
    print()
    print('------------------    adv_initial_stuff()    -----------------------')

    S = ADVData()
    S.all_noise_levels = all_noise_levels
    S.all_kappa = all_kappa
    S.all_t_final = all_t_final
    S.all_num_batches = all_num_batches
    S.krylov_thresholds = krylov_thresholds
    S.impulse_points = impulse_points
    S.save()

    #### MAKE INITIAL CONDITION PLOT ####

    np.random.seed(0)
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

    S.newton_krylov_iters_none = -1 * np.ones((len(all_noise_levels),
                                               len(all_kappa),
                                               len(all_t_final),
                                               len(krylov_thresholds)), dtype=int)

    S.newton_krylov_iters_reg = -1 * np.ones((len(all_noise_levels),
                                              len(all_kappa),
                                              len(all_t_final),
                                              len(krylov_thresholds)), dtype=int)

    S.newton_krylov_iters_psf = -1 * np.ones((len(all_num_batches),
                                              len(all_noise_levels),
                                              len(all_kappa),
                                              len(all_t_final),
                                              len(krylov_thresholds)), dtype=int)

    ####

    S.randn_krylov_iters_none = -1 * np.ones((len(all_noise_levels),
                                              len(all_kappa),
                                              len(all_t_final),
                                              len(krylov_thresholds)), dtype=int)

    S.randn_krylov_iters_reg = -1 * np.ones((len(all_noise_levels),
                                             len(all_kappa),
                                             len(all_t_final),
                                             len(krylov_thresholds)), dtype=int)

    S.randn_krylov_iters_psf = -1 * np.ones((len(all_num_batches),
                                             len(all_noise_levels),
                                             len(all_kappa),
                                             len(all_t_final),
                                             len(krylov_thresholds)), dtype=int)

    ####

    S.geigs_none = np.ones((len(all_noise_levels),
                            len(all_kappa),
                            len(all_t_final),
                            ADV.N)) * np.nan

    S.geigs_reg = np.ones((len(all_noise_levels),
                           len(all_kappa),
                           len(all_t_final),
                           ADV.N)) * np.nan

    S.geigs_psf = np.ones((len(all_num_batches),
                           len(all_noise_levels),
                           len(all_kappa),
                           len(all_t_final),
                           ADV.N)) * np.nan

    ####

    S.all_areg_morozov = np.ones((len(all_noise_levels),
                                  len(all_kappa),
                                  len(all_t_final))) * np.nan

    S.all_noise_norms = np.ones((len(all_noise_levels),
                                 len(all_kappa),
                                 len(all_t_final))) * np.nan

    ####

    S.all_gradients = np.ones((len(all_noise_levels),
                               len(all_kappa),
                               len(all_t_final),
                               ADV.N)) * np.nan

    S.all_newton_dirs = np.ones((len(all_noise_levels),
                                  len(all_kappa),
                                  len(all_t_final),
                                  ADV.N)) * np.nan

    S.all_randn_b = np.ones((len(all_noise_levels),
                             len(all_kappa),
                             len(all_t_final),
                             ADV.N)) * np.nan

    S.all_randn_x = np.ones((len(all_noise_levels),
                             len(all_kappa),
                             len(all_t_final),
                             ADV.N)) * np.nan

    ####

    S.all_newton_errs_none = np.empty((len(all_noise_levels),
                                       len(all_kappa),
                                       len(all_t_final)), dtype=object)

    S.all_newton_errs_reg = np.empty((len(all_noise_levels),
                                      len(all_kappa),
                                      len(all_t_final)), dtype=object)

    S.all_newton_errs_psf = np.empty((len(all_num_batches),
                                      len(all_noise_levels),
                                      len(all_kappa),
                                      len(all_t_final)), dtype=object)
    ####

    S.all_randn_errs_none = np.empty((len(all_noise_levels),
                                      len(all_kappa),
                                      len(all_t_final)), dtype=object)

    S.all_randn_errs_reg = np.empty((len(all_noise_levels),
                                     len(all_kappa),
                                     len(all_t_final)), dtype=object)

    S.all_randn_errs_psf = np.empty((len(all_num_batches),
                                     len(all_noise_levels),
                                     len(all_kappa),
                                     len(all_t_final)), dtype=object)

    ####

    S.save()


def do_one_run_firstpart(ii, jj, kk):
    S = ADVData()
    S.load()

    ns = S.all_noise_levels[ii]
    kp = S.all_kappa[jj]
    tf = S.all_t_final[kk]

    print()
    print('------------------    do_one_run_firstpart    -----------------------')
    print('ns=', ns)
    print('kp=', kp)
    print('tf=', tf)

    ns_str = '_N=' + np.format_float_scientific(ns, precision=1, exp_digits=1)
    tf_str = '_T=' + np.format_float_scientific(tf, precision=1, exp_digits=1)
    kp_str = '_K=' + np.format_float_scientific(kp, precision=1, exp_digits=1)
    id_str = tf_str + kp_str + ns_str

    np.random.seed(0)
    ADV = adv.make_adv_universe(ns, kp, tf,
                                num_checkers_x=num_checkers,
                                num_checkers_y=num_checkers)

    #### IMPULSE RESPONSE PLOTS ####
    if ii == 0:  # impulse responses don't depend on noise
        print('making impulse response pictures')
        plt.figure()
        for ll in range(S.impulse_points.shape[0]):
            p_str = '_P' + str(tuple(S.impulse_points[ll, :]))
            phi = dl.Function(ADV.Vh)
            phi.vector()[:] = ADV.get_misfit_hessian_impulse_response(S.impulse_points[ll, :])

            cm = dl.plot(phi)
            plt.colorbar(cm)
            plt.axis('off')
            plt.plot(S.impulse_points[ll, 0], S.impulse_points[ll, 1], '*r')
            plt.title('Impulse response, T=' + str(tf) + ', kappa=' + str(kp))
            plt.savefig(save_dir_str + '/impulse_response' + p_str + id_str + '.png',
                        dpi=fig_dpi, bbox_inches='tight')

            dl.File(save_dir_str + '/impulse_response' + p_str + id_str + '.pvd') << phi
            plt.clf()
        plt.close()

    S.save()

    #### PLOT OBSERVATIONS ####
    plt.figure()
    cm = dl.plot(ADV.obs_func, cmap='gray')
    plt.colorbar(cm)
    plt.axis('off')
    plt.title('noisy observations')
    plt.savefig(save_dir_str + '/noisy_observations' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')

    dl.File(save_dir_str + '/noisy_observations' + id_str + '.pvd') << ADV.obs_func
    plt.close()

    #### COMPUTE MOROZOV REGULARIZATION PARAMETER ####
    areg_initial_guess = 1e-5
    areg_morozov, seen_aregs, seen_discrepancies = ADV.compute_morozov_regularization_parameter(
        areg_initial_guess, morozov_options={'morozov_rtol': morozov_rtol})

    inds = np.argsort(seen_aregs)
    seen_aregs = seen_aregs[inds]
    seen_discrepancies = seen_discrepancies[inds]

    print('areg_morozov=', areg_morozov)
    S.all_areg_morozov[ii, jj, kk] = areg_morozov
    S.all_noise_norms[ii, jj, kk] = ADV.noise_norm
    S.save()

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

    S.all_gradients[ii,jj,kk,:] = g0
    S.save()

    H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg_morozov))
    p_newton = nhf.custom_cg(H_linop, -g0, display=True, maxiter=maxiter_krylov, tol=exact_rtol)[0]

    S.all_newton_dirs[ii,jj,kk,:] = p_newton
    S.save()

    mstar_vec = m0 + p_newton
    mstar = dl.Function(ADV.Vh)
    mstar.vector()[:] = mstar_vec.copy()

    plt.figure()
    cm = dl.plot(mstar, cmap='gray')
    plt.colorbar(cm)
    plt.axis('off')
    plt.title('Reconstructed initial condition')
    plt.savefig(save_dir_str + '/reconstructed_initial_condition' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')

    dl.File(save_dir_str + '/reconstructed_initial_condition' + id_str + '.pvd') << mstar
    plt.close()

    #

    print('Solving inverse problem to tight tol with areg_morozov')
    b_randn = np.random.randn(ADV.N)
    x_randn = nhf.custom_cg(H_linop, b_randn, display=True, maxiter=maxiter_krylov, tol=exact_rtol)[0]

    S.all_randn_b[ii,jj,kk,:] = b_randn
    S.all_randn_x[ii, jj, kk, :] = x_randn
    S.save()

    #### KRYLOV CONVERGENCE AND PRECONDITIONED SPECTRUM PLOTS ####
    print('Making Krylov convergence plots')
    M_reg_matvec = lambda x: ADV.regularization.solve_hessian(x, m0, areg_morozov)
    M_reg_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=M_reg_matvec)

    _, _, _, newton_errs_none = nhf.custom_cg(H_linop, -g0, x_true=p_newton, display=True, maxiter=maxiter_krylov,
                                              tol=tight_rtol)
    _, _, _, newton_errs_reg = nhf.custom_cg(H_linop, -g0, M=M_reg_linop, x_true=p_newton, display=True,
                                             maxiter=maxiter_krylov, tol=tight_rtol)

    np.savetxt(save_dir_str + '/newton_errs_none' + id_str + '.txt', newton_errs_none)
    np.savetxt(save_dir_str + '/newton_errs_reg' + id_str + '.txt', newton_errs_reg)

    S.all_newton_errs_none[ii, jj, kk] = newton_errs_none
    S.all_newton_errs_reg[ii, jj, kk] = newton_errs_reg
    S.newton_krylov_iters_none[ii, jj, kk] = nhf.threshold_crossings(newton_errs_none, S.krylov_thresholds)
    S.newton_krylov_iters_reg[ii, jj, kk] = nhf.threshold_crossings(newton_errs_reg, S.krylov_thresholds)
    S.save()

    print('ns=', ns)
    print('kp=', kp)
    print('tf=', tf)
    print('krylov_thresholds=', S.krylov_thresholds)
    print('newton_crossings_none=', S.newton_krylov_iters_none[ii, jj, kk])
    print('newton_crossings_reg=', S.newton_krylov_iters_reg[ii, jj, kk])

    #

    _, _, _, randn_errs_none = nhf.custom_cg(H_linop, b_randn, x_true=x_randn, display=True, maxiter=maxiter_krylov,
                                             tol=tight_rtol)
    _, _, _, randn_errs_reg = nhf.custom_cg(H_linop, b_randn, M=M_reg_linop, x_true=x_randn, display=True,
                                            maxiter=maxiter_krylov, tol=tight_rtol)

    np.savetxt(save_dir_str + '/randn_errs_none' + id_str + '.txt', randn_errs_none)
    np.savetxt(save_dir_str + '/randn_errs_reg' + id_str + '.txt', randn_errs_reg)

    S.all_randn_errs_none[ii, jj, kk] = randn_errs_none
    S.all_randn_errs_reg[ii, jj, kk] = randn_errs_reg
    S.randn_krylov_iters_none[ii, jj, kk, :] = nhf.threshold_crossings(randn_errs_none, S.krylov_thresholds)
    S.randn_krylov_iters_reg[ii, jj, kk, :] = nhf.threshold_crossings(randn_errs_reg, S.krylov_thresholds)
    S.save()

    print('krylov_thresholds=', S.krylov_thresholds)
    print('randn_crossings_none=', S.randn_krylov_iters_none[ii, jj, kk, :])
    print('randn_crossings_reg=', S.randn_krylov_iters_reg[ii, jj, kk, :])

    #

    print('Building H_dense')
    H_dense = nhf.build_dense_matrix_from_matvecs(H_linop.matvec, H_linop.shape[1])
    np.save('H_dense_tmp', H_dense)

    print('computing ee_none')
    ee_none = sla.eigh(H_dense, eigvals_only=True)[::-1]

    S.geigs_none[ii, jj, kk, :] = ee_none
    S.save()

    print('Building invHR_dense')
    invHR_dense = nhf.build_dense_matrix_from_matvecs(M_reg_linop.matvec, M_reg_linop.shape[1])

    print('Computing HR_dense')
    HR_dense = np.linalg.inv(invHR_dense)
    del invHR_dense

    print('computing ee_reg')
    ee_reg = sla.eigh(H_dense, HR_dense, eigvals_only=True)[::-1]
    del HR_dense

    S.geigs_reg[ii, jj, kk, :] = ee_reg
    S.save()


def do_one_run_psf(ll, ii, jj, kk): # ll is num_batches index
    S = ADVData()
    S.load()

    ns = S.all_noise_levels[ii]
    kp = S.all_kappa[jj]
    tf = S.all_t_final[kk]
    nb = S.all_num_batches[ll]

    np.random.seed(0)
    ADV = adv.make_adv_universe(ns, kp, tf,
                                num_checkers_x=num_checkers,
                                num_checkers_y=num_checkers)

    print()
    print('------------------    do_one_run_psf    -----------------------')
    print('ns=', ns)
    print('kp=', kp)
    print('tf=', tf)
    print('nb=', nb)

    ns_str = '_N=' + np.format_float_scientific(ns, precision=1, exp_digits=1)
    tf_str = '_T=' + np.format_float_scientific(tf, precision=1, exp_digits=1)
    kp_str = '_K=' + np.format_float_scientific(kp, precision=1, exp_digits=1)
    bt_str = '_B=' + np.format_float_scientific(nb, precision=1, exp_digits=1)
    extended_id_str = bt_str + tf_str + kp_str + ns_str

    print('Making row and column cluster trees')
    ct = hpro.build_cluster_tree_from_pointcloud(ADV.mesh_and_function_space.dof_coords,
                                                 cluster_size_cutoff=32)

    print('Making block cluster trees')
    bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=1.0)

    HR_hmatrix = ADV.regularization.Cov.make_invC_hmatrix(bct, 1.0)

    ####

    areg_morozov = S.all_areg_morozov[ii, jj, kk]
    H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg_morozov))
    g0 = S.all_gradients[ii,jj,kk,:]
    p_newton = S.all_newton_dirs[ii,jj,kk,:]

    psf_preconditioner = PSFHessianPreconditioner(
        ADV.apply_misfit_hessian, ADV.Vh, ADV.mesh_and_function_space.mass_lumps,
        HR_hmatrix, display=True)

    psf_preconditioner.psf_options['num_initial_batches'] = nb
    psf_preconditioner.build_hessian_preconditioner()

    psf_preconditioner.shifted_inverse_interpolator.insert_new_mu(areg_morozov)
    psf_preconditioner.update_deflation(areg_morozov)

    P_linop = spla.LinearOperator(
        (ADV.N, ADV.N),
        matvec=lambda x: psf_preconditioner.solve_hessian_preconditioner(x, areg_morozov))

    _, _, _, newton_errs_psf = nhf.custom_cg(H_linop, -g0, M=P_linop, x_true=p_newton, display=True,
                                             maxiter=maxiter_krylov, tol=tight_rtol)

    np.savetxt(save_dir_str + '/newton_errs_psf' + extended_id_str + '.txt', newton_errs_psf)

    S.all_newton_errs_psf[ll, ii, jj, kk] = newton_errs_psf
    S.newton_krylov_iters_psf[ll, ii, jj, kk, :] = nhf.threshold_crossings(newton_errs_psf, S.krylov_thresholds)
    S.save()

    print('ns=', ns)
    print('kp=', kp)
    print('tf=', tf)
    print('nb=', nb)
    print('krylov_thresholds=', S.krylov_thresholds)
    print('newton_crossings_psf=', S.newton_krylov_iters_psf[ll, ii, jj, kk, :])

    #
    b_randn = S.all_randn_b[ii, jj, kk, :]
    x_randn = S.all_randn_x[ii, jj, kk, :]

    _, _, _, randn_errs_psf = nhf.custom_cg(H_linop, b_randn, M=P_linop, x_true=x_randn, display=True,
                                            maxiter=maxiter_krylov, tol=tight_rtol)

    np.savetxt(save_dir_str + '/randn_errs_psf' + extended_id_str + '.txt', randn_errs_psf)

    S.all_randn_errs_psf[ll, ii, jj, kk] = randn_errs_psf
    S.randn_krylov_iters_psf[ll, ii, jj, kk, :] = nhf.threshold_crossings(randn_errs_psf, S.krylov_thresholds)
    S.save()

    print('ns=', ns)
    print('kp=', kp)
    print('tf=', tf)
    print('nb=', nb)
    print('krylov_thresholds=', S.krylov_thresholds)
    print('randn_crossings_psf=', S.randn_krylov_iters_psf[ll, ii, jj, kk, :])

    #

    H_dense = np.load('H_dense_tmp.npy')

    apply_PSF = lambda x: psf_preconditioner.shifted_inverse_interpolator.apply_shifted_deflated(x, areg_morozov)

    print('Building PSF_dense')
    PSF_dense = nhf.build_dense_matrix_from_matvecs(apply_PSF, P_linop.shape[1])

    print('computing ee_psf')
    ee_psf = sla.eigh(H_dense, PSF_dense, eigvals_only=True)[::-1]
    del PSF_dense

    S.geigs_psf[ll, ii, jj, kk, :] = ee_psf
    S.save()

    ####

def do_one_run_lastpart(ii, jj, kk):
    S = ADVData()
    S.load()

    ns = S.all_noise_levels[ii]
    kp = S.all_kappa[jj]
    tf = S.all_t_final[kk]

    ns_str = '_N=' + np.format_float_scientific(ns, precision=1, exp_digits=1)
    tf_str = '_T=' + np.format_float_scientific(tf, precision=1, exp_digits=1)
    kp_str = '_K=' + np.format_float_scientific(kp, precision=1, exp_digits=1)
    id_str = tf_str + kp_str + ns_str

    print()
    print('------------------    do_one_run_lastpart    -----------------------')
    print('ns=', ns)
    print('kp=', kp)
    print('tf=', tf)

    plt.figure()
    plt.semilogy(S.all_newton_errs_none[ii, jj, kk])
    plt.semilogy(S.all_newton_errs_reg[ii, jj, kk])
    legend = ['None', 'Reg']
    for ll in range(len(S.all_num_batches)):
        plt.semilogy(S.all_newton_errs_psf[ll, ii, jj, kk])
        legend.append('PSF ' + str(S.all_num_batches[ll]))
    plt.legend(legend)
    plt.xlabel('Iteration')
    plt.ylabel('Relative error')
    plt.title('PCG convergence (Newton system) ' + id_str)
    plt.savefig(save_dir_str + '/pcg_convergence_newton' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.semilogy(S.all_randn_errs_none[ii, jj, kk])
    plt.semilogy(S.all_randn_errs_reg[ii, jj, kk])
    legend = ['None', 'Reg']
    for ll in range(len(S.all_num_batches)):
        plt.semilogy(S.all_randn_errs_psf[ll, ii, jj, kk])
        legend.append('PSF ' + str(S.all_num_batches[ll]))
    plt.legend(legend)
    plt.xlabel('Iteration')
    plt.ylabel('Relative error')
    plt.title('PCG convergence (randn RHS) ' + id_str)
    plt.savefig(save_dir_str + '/pcg_convergence_randn' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.semilogy(S.geigs_none[ii, jj, kk, :])
    plt.semilogy(S.geigs_reg[ii, jj, kk, :])
    legend = ['None', 'Reg']
    for ll in range(len(S.all_num_batches)):
        plt.semilogy(S.geigs_psf[ll, ii, jj, kk, :])
        legend.append('PSF ' + str(S.all_num_batches[ll]))
    plt.legend(legend)
    plt.xlabel('index')
    plt.ylabel('eigenvalue')
    plt.title('generalized eigenvalues ' + id_str)
    plt.savefig(save_dir_str + '/generalized_eigenvalues' + id_str + '.png', dpi=fig_dpi, bbox_inches='tight')
    plt.close()

####

def make_kappa_plot(ii, kk):
    S = ADVData()
    S.load()

    ns = S.all_noise_levels[ii]
    tf = S.all_t_final[kk]

    ns_str = '_N=' + np.format_float_scientific(ns, precision=1, exp_digits=1)
    tf_str = '_T=' + np.format_float_scientific(tf, precision=1, exp_digits=1)
    id_str_reduced = tf_str + ns_str

    threshold_ind = 5
    krylov_tol= S.krylov_thresholds[threshold_ind]
    print('krylov_tol=', krylov_tol)
    plt.figure()
    plt.loglog(S.all_kappa, S.newton_krylov_iters_none[ii,:,kk,threshold_ind])
    plt.loglog(S.all_kappa, S.newton_krylov_iters_reg[ii,:,kk,threshold_ind])
    legend = ['None', 'Reg']
    for ll in range(len(S.all_num_batches)):
        plt.loglog(S.all_kappa, S.newton_krylov_iters_psf[ll,ii,:,kk,threshold_ind])
        legend.append('PSF ' + str(S.all_num_batches[ll]))
    plt.legend(legend)
    plt.xlabel('kappa')
    plt.ylabel('num krylov iters')
    plt.title('number of krylov iters vs kappa, newton, krylov_tol=' + str(krylov_tol))
    plt.savefig(save_dir_str + '/krylov_vs_kappa_newton' + id_str_reduced + '.png', dpi=fig_dpi, bbox_inches='tight')

    plt.figure()
    plt.loglog(S.all_kappa, S.randn_krylov_iters_none[ii,:,kk,threshold_ind])
    plt.loglog(S.all_kappa, S.randn_krylov_iters_reg[ii,:,kk,threshold_ind])
    legend = ['None', 'Reg']
    for ll in range(len(S.all_num_batches)):
        plt.loglog(S.all_kappa, S.randn_krylov_iters_psf[ll,ii,:,kk,threshold_ind])
        legend.append('PSF ' + str(S.all_num_batches[ll]))
    plt.legend(legend)
    plt.xlabel('kappa')
    plt.ylabel('num krylov iters')
    plt.title('number of krylov iters vs kappa, randn, krylov_tol=' + str(krylov_tol))
    plt.savefig(save_dir_str + '/krylov_vs_kappa_randn' + id_str_reduced + '.png', dpi=fig_dpi, bbox_inches='tight')


def make_tf_plot(ii, jj):
    S = ADVData()
    S.load()

    ns = S.all_noise_levels[ii]
    kp = S.all_kappa[jj]

    ns_str = '_N=' + np.format_float_scientific(ns, precision=1, exp_digits=1)
    kp_str = '_T=' + np.format_float_scientific(kp, precision=1, exp_digits=1)
    id_str_reduced2 = kp_str + ns_str

    threshold_ind = 5
    krylov_tol= S.krylov_thresholds[threshold_ind]
    print('krylov_tol=', krylov_tol)
    plt.figure()
    plt.loglog(S.all_t_final, S.newton_krylov_iters_none[ii,jj,:,threshold_ind])
    plt.loglog(S.all_t_final, S.newton_krylov_iters_reg[ii,jj,:,threshold_ind])
    legend = ['None', 'Reg']
    for ll in range(len(S.all_num_batches)):
        plt.loglog(S.all_t_final, S.newton_krylov_iters_psf[ll,ii,jj,:,threshold_ind])
        legend.append('PSF ' + str(S.all_num_batches[ll]))
    plt.legend(legend)
    plt.xlabel('Tf')
    plt.ylabel('num krylov iters')
    plt.title('number of krylov iters vs Tf, newton, krylov_tol=' + str(krylov_tol))
    plt.savefig(save_dir_str + '/krylov_vs_tf_newton' + id_str_reduced2 + '.png', dpi=fig_dpi, bbox_inches='tight')

    plt.figure()
    plt.loglog(S.all_t_final, S.randn_krylov_iters_none[ii,jj,:,threshold_ind])
    plt.loglog(S.all_t_final, S.randn_krylov_iters_reg[ii,jj,:,threshold_ind])
    legend = ['None', 'Reg']
    for ll in range(len(S.all_num_batches)):
        plt.loglog(S.all_t_final, S.randn_krylov_iters_psf[ll,ii,jj,:,threshold_ind])
        legend.append('PSF ' + str(S.all_num_batches[ll]))
    plt.legend(legend)
    plt.xlabel('Tf')
    plt.ylabel('num krylov iters')
    plt.title('number of krylov iters vs Tf, randn, krylov_tol=' + str(krylov_tol))
    plt.savefig(save_dir_str + '/krylov_vs_tf_randn' + id_str_reduced2 + '.png', dpi=fig_dpi, bbox_inches='tight')




args = sys.argv[1:]
ii_in = int(args[0]) # noise
jj_in = int(args[1]) # kappa
kk_in = int(args[2]) # t_final
ll_in = int(args[3]) # num_batches
run_type = str(args[4])

print('noise_level: ii_in=', ii_in)
print('kappa: jj_in=', jj_in)
print('t_final: kk_in=', kk_in)
print('num_batches: ll_in=', ll_in)
print('run_type=', run_type)

if run_type.lower() == 'init':
    initialize_stuff()
elif run_type.lower() == 'first':
    do_one_run_firstpart(ii_in, jj_in, kk_in)
elif run_type.lower() == 'psf':
    do_one_run_psf(ll_in, ii_in, jj_in, kk_in)
elif run_type.lower() == 'last':
    do_one_run_lastpart(ii_in, jj_in, kk_in)
elif run_type.lower() == 'kappa':
    make_kappa_plot(ii_in, kk_in)
elif run_type.lower() == 'tf':
    make_tf_plot(ii_in, jj_in)

print()
print('-----------------------------------------')
geigs_psf = load2(save_dir_str + '/geigs_psf.npy')
incomplete_runs = np.argwhere(np.isnan(geigs_psf[:,:,:,:,0]))
print('ll, ii, jj, kk')
print('nb, ns, kp, tf')
print('incomplete_runs:')
print(incomplete_runs)
print('-----------------------------------------')
print()