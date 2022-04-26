import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix
from localpsf.visualization import column_error_plot
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter

import scipy.sparse.linalg as spla


save_data = True
save_figures = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'error_vs_krylov_iter'
save_dir.mkdir(parents=True, exist_ok=True)


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 2.5e-2} # {'mesh_h': 2e-2}

tau = 2.5
all_num_batches = [1,3,6,9]
num_neighbors = 10
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
hmatrix_tol=1e-4


########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**nondefault_HIP_options)

########    CONSTRUCT KERNEL    ########

PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                               10, 10,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma)

########    CREATE HMATRICES    ########

A_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)

# _, eeA_max, _ = spla.svds(A_pch_nonsym.as_linear_operator(), k=3, which='LM')
# eA_max = np.max(eeA_max)

# cutoff = -1e-2*eA_max

# A_pch = A_pch_nonsym.spd(rtol=1e-3, atol=1e-5)
A_pch = A_pch_nonsym.spd()

# min_reg_param = 1e-3
# eeR_min, _ = spla.eigsh(min_reg_param * HIP.R0_scipy, k=3, which='SM')
# eR_min = np.min(eeR_min)
#
# A_pch = A_pch_nonsym.spd(cutoff=0.8*eR_min)


########    SET UP REGULARIZATION OPERATOR AND INVERSE PROBLEM SOLVER    ########

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, A_pch.bct)

def solve_inverse_problem(a_reg):
    HIP.regularization_parameter = a_reg

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    H_hmatrix = A_pch + a_reg * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix.as_linear_operator(),
                                          tol=1e-10, maxiter=500)

    u0 = dl.Function(HIP.V)
    u0.vector()[:] = u0_numpy

    return u0.vector()  # dolfin Vector


########    FIND REGULARIZATION PARAMETER VIA MOROZOV DISCREPANCY PRINCIPLE    ########

a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
                                                         HIP.morozov_discrepancy,
                                                         HIP.noise_Mnorm)



########    SOLVE INVERSE PROBLEM VIA CG WITH DIFFERENT PRECONDITIONERS    ########

HIP.regularization_parameter = a_reg_morozov

g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
H_hmatrix = A_pch + a_reg_morozov * R0_hmatrix
iH_hmatrix = H_hmatrix.inv()

u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                      M=iH_hmatrix.as_linear_operator(),
                                      tol=1e-11, maxiter=500)

_, _, residuals_None, errors_None = custom_cg(HIP.H_linop, -g0_numpy,
                                              x_true=u0_numpy,
                                              tol=1e-10, maxiter=500)

_, _, residuals_Reg, errors_Reg = custom_cg(HIP.H_linop, -g0_numpy,
                                            x_true=u0_numpy,
                                            M=HIP.solve_R_linop,
                                            tol=1e-10, maxiter=1000)


all_residuals_PCH = list()
all_errors_PCH = list()
for num_batches in all_num_batches:
    print('num_batches=', num_batches)
    PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                                   num_batches, num_batches,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma)

    A_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)

    A_pch = A_pch_nonsym.spd()

    H_hmatrix = A_pch + a_reg_morozov * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    _, _, residuals_PCH, errors_PCH = custom_cg(HIP.H_linop, -g0_numpy,
                                                x_true=u0_numpy,
                                                M=iH_hmatrix.as_linear_operator(),
                                                tol=1e-10, maxiter=500)

    all_residuals_PCH.append(residuals_PCH)
    all_errors_PCH.append(errors_PCH)


########    SAVE DATA    ########

all_num_batches = np.array(all_num_batches)
residuals_None = np.array(residuals_None)
errors_None = np.array(errors_None)
residuals_Reg = np.array(residuals_Reg)
errors_Reg = np.array(errors_Reg)
all_residuals_PCH = [np.array(r) for r in all_residuals_PCH]
all_errors_PCH = [np.array(e) for e in all_errors_PCH]

if save_data:
    np.savetxt(save_dir / 'all_num_batches.txt', all_num_batches)
    np.savetxt(save_dir / 'residuals_None.txt', residuals_None)
    np.savetxt(save_dir / 'errors_None.txt', errors_None)
    np.savetxt(save_dir / 'residuals_Reg.txt', residuals_Reg)
    np.savetxt(save_dir / 'errors_Reg.txt', errors_Reg)
    for k in range(len(all_num_batches)):
        np.savetxt(save_dir / ('all_residuals_PCH' + str(all_num_batches[k]) + '.txt'), all_residuals_PCH[k])
        np.savetxt(save_dir / ('all_errors_PCH' + str(all_num_batches[k]) + '.txt'), all_errors_PCH[k])


########    MAKE FIGURES    ########

plt.figure()
plt.semilogy(errors_Reg)
plt.semilogy(errors_None)
legend = ['Reg', 'None']
for k in range(len(all_num_batches)):
    plt.semilogy(all_errors_PCH[k])
    legend.append('PCH ' + str(all_num_batches[k]))

plt.xlabel('Iteration')
plt.ylabel('relative l2 error')
plt.title('Relative error vs. Krylov iteration')
plt.legend(legend)

if save_figures:
    plt.savefig(save_dir / 'error_vs_krylov_iter.pdf', bbox_inches='tight', dpi=100)


plt.figure()
plt.semilogy(residuals_Reg)
plt.semilogy(residuals_None)
legend = ['Reg', 'None']
for k in range(len(all_num_batches)):
    plt.semilogy(all_residuals_PCH[k])
    legend.append('PCH ' + str(all_num_batches[k]))

plt.xlabel('Iteration')
plt.ylabel('relative residual')
plt.title('Relative residual vs. Krylov iteration')
plt.legend(legend)

if save_figures:
    plt.savefig(save_dir / 'relative_residual_vs_krylov_iter.pdf', bbox_inches='tight', dpi=100)


# ###################################################################################################
# ###################################################################################################
# ###################################################################################################
# ###################################################################################################
#
# ##############3
#
# class LinopWithCounter(spla.LinearOperator):
#     def __init__(me, linop):
#         me.matvec_counter = 0
#         me.rmatvec_counter = 0
#         me.linop = linop
#         me.shape = linop.shape
#         me.dtype = linop.dtype
#
#     def matvec(me, X):
#         me.matvec_counter += 1
#         print('matvec_counter=', me.matvec_counter)
#         return me.linop.matvec(X)
#         # if len(X.shape)==1:
#         #     X = X.reshape((-1,1))
#         # Y = np.zeros((me.shape[0], X.shape[1]))
#         # for k in range(X.shape[1]):
#         #     me.matvec_counter += 1
#         #     print('matvec_counter=', me.matvec_counter)
#         #     Y[:,k] = me.linop.matvec(X[:,k])
#         # return Y
#
#     def rmatvec(me, X):
#         if len(X.shape) == 1:
#             X = X.reshape((-1, 1))
#         Y = np.zeros((me.shape[1], X.shape[1]))
#         for k in range(X.shape[1]):
#             me.rmatvec_counter += 1
#             print('rmatvec_counter=', me.rmatvec_counter)
#             Y[:, k] = me.linop.rmatvec(X[:, k])
#         return Y
#
#     def reset_counters(me):
#         me.matvec_counter = 0
#         me.rmatvec_counter = 0
#
# ####    Experimenting with negative eigenvalue stuff
#
#
#
# A_pch_sym = A_pch_nonsym.sym()
#
# R_hmatrix = a_reg_morozov * R0_hmatrix
#
# A_plus_R = A_pch_sym + R_hmatrix
#
# iA_plus_R = A_plus_R.inv()
#
# iR_hmatrix = R_hmatrix.inv()
#
# iR0_hmatrix = R0_hmatrix.inv()
#
# # Hd_dense = build_dense_matrix_from_matvecs(A_pch_sym.matvec, A_pch_sym.shape[1])
# Hd_dense = build_dense_matrix_from_matvecs(A_pch.matvec, A_pch.shape[1])
# R0_dense = build_dense_matrix_from_matvecs(R0_hmatrix.matvec, R0_hmatrix.shape[1])
#
# import scipy.linalg as sla
# ee_true, uu_true = sla.eigh(Hd_dense, R0_dense)
#
# plt.figure()
# plt.plot(ee_true)
# plt.plot(-1e-4 * np.ones(len(ee_true)) / 2.)
#
# # rr_true = np.sum(uu_true * np.dot(R0_dense, uu_true), axis=0) # (1,1,...,1)
#
# shift = -1.1*np.min(ee_true)
#
# plt.figure()
# plt.plot(1./(ee_true + shift),'.')
# plt.plot(1./(1. + shift),'.')
#
# ####
#
# default_rtol = 1e-6
# default_atol = 1e-6
#
#
# def make_hmatrix_spd_hackbusch_kress_2007(A_hmatrix, k=5, rtol=default_rtol, atol=default_atol, display_progress=True):
#     # Hackbusch, Wolfgang, and Wendy Kress. "A projection method for the computation of inner eigenvalues using high degree rational operators." Computing 81.4 (2007): 259-268.
#     if display_progress:
#         print('making hmatrix spd')
#         print('symmetrizing')
#     A_hmatrix = A_hmatrix.sym()
#
#     if display_progress:
#         print('getting largest eigenvalue with Lanczos')
#     ee, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='LM')
#     if display_progress:
#         print('largest eigenvalue='+str(np.max(ee)))
#
#     if display_progress:
#         print('Setting up operator T = (A - (b+a) I) / (b-a)')
#     b = np.max(ee) * 1.5
#     a = 0.0
#     T = A_hmatrix.copy()
#     T = (T * 2.0).add_identity(s=-(b + a)) * (1.0 / (b - a))
#
#     if display_progress:
#         print('computing T^(2^k)')
#     for ii in range(k):
#         if display_progress:
#             print('computing T^(2^'+str(ii+1)+') = T^(2^'+str(ii)+') * T^(2^'+str(ii)+')')
#         T = hpro.h_mul(T, T,
#                        rtol=rtol, atol=atol,
#                        display_progress=display_progress)
#
#     if display_progress:
#         print('computing non-negative spectral projector Pi = I / (I + T^(2^k))')
#     Pi = T.add_identity().inv(rtol=rtol, atol=atol, display_progress=display_progress)
#
#     if display_progress:
#         print('computing A_plus = Pi * A')
#     A_plus = hpro.h_mul(Pi, A_hmatrix,
#                         rtol=T_rtol, atol=T_atol,
#                         display_progress=display_progress).sym()
#
#     return A_plus
#
# A_plus = make_hmatrix_spd_hackbusch_kress_2007(A_pch_sym)
#
# A_plus_dense = build_dense_matrix_from_matvecs(A_plus.matvec, A_plus.shape[1])
#
# ee_plus, uu_plus = np.linalg.eigh(A_plus_dense)
#
# print('np.max(ee_plus)=', np.max(ee_plus))
# print('np.min(ee_plus)=', np.min(ee_plus))
#
# plt.figure()
# plt.plot(ee_plus)
#
# A_dense = build_dense_matrix_from_matvecs(A_pch_sym.matvec, A_pch_sym.shape[1])
#
# ee, uu = np.linalg.eigh(A_dense)
#
# print('np.max(ee)=', np.max(ee))
# print('np.min(ee)=', np.min(ee))
#
# plt.plot(ee)
#
#
# ########
#
# nrow, ncol = A_pch_sym.shape
#
# # A1 = A_pch_sym.low_rank_update(np.ones((nrow,1)), 1e-9*np.ones((1,ncol)))
#
# A1 = A_pch_sym
# A1.visualize('A1')
#
# ee, _ = spla.eigsh(A1.as_linear_operator(), k=1, which='LM')
# b = np.max(ee) * 1.5
# a = 0.0
#
# T_rtol = 1e-5
# T_atol = 1e-5
#
# T1 = (A1*2.0).add_identity(-(b+a))*(1.0/(b-a))
# T1.visualize('T1')
#
# T2 = hpro.h_mul(T1, T1, rtol=T_rtol, atol=T_atol, display_progress=True)
# T2.visualize('T2')
#
# z = np.random.randn(T2.shape[1])
#
# y2_true = T1 * (T1 * z)
# y2 = T2 * z
#
# err_T2 = np.linalg.norm(y2 - y2_true) / np.linalg.norm(y2_true)
# print('err_T2=', err_T2)
#
# #
#
# T4 = hpro.h_mul(T2, T2, rtol=T_rtol, atol=T_atol, display_progress=True)
# T4.visualize('T4')
#
# y4_true = T1 * (T1 * y2_true)
# y4 = T4 * z
#
# err_T4 = np.linalg.norm(y4 - y4_true) / np.linalg.norm(y4_true)
# print('err_T4=', err_T4)
#
# #
#
# T8 = hpro.h_mul(T4, T4, rtol=T_rtol, atol=T_atol, display_progress=True)
# T8.visualize('T8')
#
# y8_true = T1 * (T1 * (T1 * (T1 * y4_true)))
# y8 = T8 * z
#
# err_T8 = np.linalg.norm(y8 - y8_true) / np.linalg.norm(y8_true)
# print('err_T8=', err_T8)
#
# #
#
# T16 = hpro.h_mul(T8, T8, rtol=T_rtol, atol=T_atol, display_progress=True)
# T16.visualize('T16')
#
# y16_true = T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * y8_true)))))))
# y16 = T16 * z
#
# err_T16 = np.linalg.norm(y16 - y16_true) / np.linalg.norm(y16_true)
# print('err_T16=', err_T16)
#
# #
#
# T32 = hpro.h_mul(T16, T16, rtol=T_rtol, atol=T_atol, display_progress=True)
# T32.visualize('T32')
#
# y32_true = T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * (T1 * y16_true)))))))))))))))
# y32 = T32 * z
#
# err_T32 = np.linalg.norm(y32 - y32_true) / np.linalg.norm(y32_true)
# print('err_T32=', err_T32)
#
# #
#
# T64 = hpro.h_mul(T32, T32, rtol=T_rtol, atol=T_atol, display_progress=True)
# T64.visualize('T64')
#
# y64_true = z
# for k in range(64):
#     y64_true = T1 * y64_true
#
# y64 = T64 * z
#
# err_T64 = np.linalg.norm(y64 - y64_true) / np.linalg.norm(y64_true)
# print('err_T64=', err_T64)
#
# #
#
# A_plus_projector = T64.add_identity().inv(rtol=T_rtol, atol=T_atol)
# A_plus_projector.visualize('A_plus_projector')
#
# A_plus = hpro.h_mul(A_plus_projector, A1, rtol=T_rtol, atol=T_atol, display_progress=True)
# A_plus = A_plus.sym()
#
# A_plus_dense = build_dense_matrix_from_matvecs(A_plus.matvec, A_plus.shape[1])
#
# ee_plus, uu_plus = np.linalg.eigh(A_plus_dense)
#
# print('np.max(ee_plus)=', np.max(ee_plus))
# print('np.min(ee_plus)=', np.min(ee_plus))
#
# plt.figure()
# plt.plot(ee_plus)
#
# A_dense = build_dense_matrix_from_matvecs(A1.matvec, A1.shape[1])
#
# ee, uu = np.linalg.eigh(A_dense)
#
# print('np.max(ee)=', np.max(ee))
# print('np.min(ee)=', np.min(ee))
#
# plt.plot(ee)
#
# plt.figure()
# plt.plot(np.log10(np.abs(ee_plus)))
# plt.plot(np.log10(np.abs(ee)))
#
# ####
#
# A_shift = A_pch_sym.add_identity(s=shift)
#
# iA_shift = A_shift.inv()
#
# ee_eigsh, uu_eigsh = spla.eigsh(LinopWithCounter(iA_shift.as_linear_operator()),
#                                 k=3, which='SA',
#                                 M=iR0_hmatrix.as_linear_operator(),
#                                 Minv=R0_hmatrix.as_linear_operator())
#
# ##
#
# ee_true, uu_true = sla.eigh(Hd_dense + a_reg_morozov * R0_dense,
#                             b=a_reg_morozov * R0_dense)
#
# iee_true2, iuu_true2 = spla.eigsh(LinopWithCounter(iA_plus_R.as_linear_operator()),
#     # np.linalg.inv(Hd_dense + a_reg_morozov * R0_dense),
#                                 k=1, which='LM', tol=1e-8,
#                                 M=np.linalg.inv(a_reg_morozov * R0_dense),
#                                 Minv=a_reg_morozov * R0_dense)
#
# ee_true2 = 1./iee_true2[::-1]
#
# iee_eigsh, iuu_eigsh = spla.eigsh(LinopWithCounter(iA_plus_R.as_linear_operator()),
#                                 k=1, which='LM', tol=1e-8,
#                                 M=iR_hmatrix.as_linear_operator(),
#                                 Minv=R_hmatrix.as_linear_operator())
#
# ee_eigsh = 1./iee_eigsh[::-1]
#
#
#
# ####
#
#
# k=0
# U_eigsh = dl.Function(HIP.V)
# U_eigsh.vector()[:] = uu_true[:,k]
# plt.figure()
# cm = dl.plot(U_eigsh)
# plt.colorbar(cm)
# plt.title(str(k)+'th worst eigenvector')
#
# ee_eigsh, uu_eigsh = spla.eigsh(Hd_dense,
#                                 k=5, which='SA',
#                                 M=R0_dense)
#
#
# ee_lobpcg, uu_lobpcg = spla.lobpcg(Hd_dense,
#                                    np.random.randn(Hd_dense.shape[1],10),
#                                    B=R0_dense,
#                                    largest=False, maxiter=500)
#
# # ee_lobpcg, uu_lobpcg = spla.lobpcg(np.linalg.inv(Hd_dense + R_dense),
# #                                    np.random.randn(A_plus_R.shape[1],1),
# #                                    B=np.linalg.inv(R_dense), largest=True, maxiter=500)
#
#
#
#
# ee_lobpcg, uu_lobpcg = spla.lobpcg(LinopWithCounter(A_pch_sym.as_linear_operator()),
#                                    np.ones((A_pch_sym.shape[1],1)),
#                                    # uu_eigsh,
#                                    # np.random.randn(A_plus_R.shape[1],1),
#                                    B=R0_hmatrix.as_linear_operator(),
#                                    largest=False,
#                                    tol=1e-12,
#                                    maxiter=500)
#
#
# ee_lobpcg, uu_lobpcg = spla.lobpcg(LinopWithCounter(A_plus_R.as_linear_operator()),
#                                    np.ones((A_plus_R.shape[1],10)),
#                                    # uu_eigsh,
#                                    # np.random.randn(A_plus_R.shape[1],1),
#                                    B=R_hmatrix.as_linear_operator(),
#                                    largest=False,
#                                    tol=1e-8,
#                                    maxiter=20)
#
# res_lobpcg = A_plus_R.matvec(uu_lobpcg[:,0]) - ee_lobpcg[0] * R_hmatrix.matvec(uu_lobpcg[:,0])
# err_lobpcg = np.linalg.norm(res_lobpcg) / np.linalg.norm(A_plus_R.matvec(uu_lobpcg[:,0]))
# print('err_lobpcg=', err_lobpcg)
#
# ee_eigsh, uu_eigsh = spla.eigsh(LinopWithCounter(iA_plus_R.as_linear_operator()),
#                                 k=5, which='SA',
#                                 M=iR_hmatrix.as_linear_operator(),
#                                 Minv=R_hmatrix.as_linear_operator())
#
# U_eigsh = dl.Function(HIP.V)
# U_eigsh.vector()[:] = uu_eigsh[:,0]
# cm = dl.plot(U_eigsh)
# plt.colorbar(cm)
#
# # A_plus_R.matvec(uu_eigsh[:,0]) / R_hmatrix.matvec(uu_eigsh[:,0])
#
# res_eigsh = A_plus_R.matvec(uu_eigsh[:,0]) - ee_eigsh[0] * R_hmatrix.matvec(uu_eigsh[:,0])
# err_eigsh = np.linalg.norm(res_eigsh) / np.linalg.norm(A_plus_R.matvec(uu_eigsh[:,0]))
# print('err_eigsh=', err_eigsh)
#
# ee_pre, uu_pre = spla.eigsh(LinopWithCounter(iA_plus_R.as_linear_operator()),
#                             k=1, which='LM',
#                             M=iR_hmatrix.as_linear_operator(),
#                             Minv=R_hmatrix.as_linear_operator())
#
# ee_pre, uu_pre = spla.eigsh(LinopWithCounter(A_pch_sym.as_linear_operator()),
#                             k=1, which='SA',
#                             M=R_hmatrix.as_linear_operator(),
#                             Minv=iR_hmatrix.as_linear_operator())
#
# # ee_pre, uu_pre = spla.eigsh(A_plus_R.as_linear_operator(), k=3, which='SA',
# #                             M=R_hmatrix.as_linear_operator(),
# #                             Minv=iR_hmatrix.as_linear_operator())
#
# # A_pch = A_pch_nonsym.spd(cutoff=cutoff)
# #
# # H_hmatrix = A_pch + a_reg_morozov * R0_hmatrix
# # iH_hmatrix = H_hmatrix.inv()