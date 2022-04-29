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
import scipy.linalg as sla


save_data = True
save_figures = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'error_vs_krylov_iter'
save_dir.mkdir(parents=True, exist_ok=True)


#

class LinopWithCounter:
    def __init__(me, linop):
        me.counter = 0
        me.linop = linop
        me.shape = linop.shape
        me.dtype = linop.dtype

    def matvec(me, x):
        me.counter+=1
        print('counter=', me.counter)
        return me.linop.matvec(x)

#

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

A_pch = A_pch_nonsym.spd(k=1)

# _, eeA_max, _ = spla.svds(A_pch_nonsym.as_linear_operator(), k=3, which='LM')
# eA_max = np.max(eeA_max)

# cutoff = -1e-2*eA_max

####

# A_hmatrix = A_pch.sym()
# k=2
# N = 2**k
# beta = np.pow(2, -N)
# alpha = ()

#########################################
#
# A_hmatrix = A_pch_nonsym
#
# a_factor=2.0
# b_factor=0.0
# k=2
# rtol = hpro.default_rtol
# atol = hpro.default_atol
# # rtol = 1e-5 # 1e-14
# # atol = 1e-8 # 1e-14
# display_progress=True
#
# if display_progress:
#     print('making hmatrix spd')
#     print('symmetrizing')
# A_hmatrix = A_hmatrix.sym()
#
# if display_progress:
#     print('getting largest eigenvalue with Lanczos')
# ee_LM, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='LM')
# lambda_max = np.max(ee_LM)
# if display_progress:
#     print('lambda_max=', lambda_max)
#
# if display_progress:
#     print('getting smallest eigenvalue with Lanczos')
# ee_SA, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='SA')
# lambda_min = np.min(ee_SA)
# if display_progress:
#     print('lambda_min=', lambda_min)
#
# # B = (A_hmatrix * (0.5/np.abs(lambda_min))).add_identity().inv(rtol=rtol, atol=atol)
# # # B = (A_hmatrix * (0.5/np.abs(lambda_min)))
# # # B.visualize('BBB')
#
# # ee_B, _ = spla.eigsh(B.as_linear_operator(), k=1, which='LA')
# # ee_B
#
# if display_progress:
#     print('Setting up operator T = (2*A - (b+a) I) / (b-a)')
# a_minus = -1.5 * np.abs(lambda_min)
# b_minus = 0.0
# a_plus = 0.0
# b_plus = 1.5 * np.abs(lambda_max)
#
# # a = 1.0
# # b = 4.0
# # a = lambda_min * a_factor
# # b = lambda_min * b_factor
#
# if display_progress:
#     plus_scaling_at_lambda_min = 1. / (1. + ((2.0 * lambda_min - (b_plus + a_plus)) / (b_plus - a_plus)) ** (2 ** k))
#     print('plus_scaling_at_lambda_min=', plus_scaling_at_lambda_min)
#
#     plus_scaling_at_zero = 1. / (1. + ((2.0*0.0 - (b_plus + a_plus)) / (b_plus - a_plus)) ** (2 ** k))
#     print('plus_scaling_at_zero=', plus_scaling_at_zero)
#
#     plus_scaling_at_lambda_max = 1. / (1. + ((2.0 * lambda_max - (b_plus + a_plus)) / (b_plus - a_plus)) ** (2 ** k))
#     print('plus_scaling_at_lambda_max=', plus_scaling_at_lambda_max)
#
#     minus_scaling_at_lambda_min = 1. / (1. + ((2.0 * lambda_min - (b_minus + a_minus)) / (b_minus - a_minus)) ** (2 ** k))
#     print('minus_scaling_at_lambda_min=', minus_scaling_at_lambda_min)
#
#     minus_scaling_at_zero = 1. / (1. + ((2.0 * 0.0 - (b_minus + a_minus)) / (b_minus - a_minus)) ** (2 ** k))
#     print('minus_scaling_at_zero=', minus_scaling_at_zero)
#
#     minus_scaling_at_lambda_max = 1. / (1. + ((2.0 * lambda_max - (b_minus + a_minus)) / (b_minus - a_minus)) ** (2 ** k))
#     print('minus_scaling_at_lambda_max=', minus_scaling_at_lambda_max)
#     # scaling_at_zero = 1. / (1. + ((b + a) / (b - a)) ** (2 ** k))
#     # print('scaling_at_zero=', scaling_at_zero)
#
# A_dense = build_dense_matrix_from_matvecs(A_hmatrix.matvec, A_hmatrix.shape[1])
# T_plus_dense =  (2.0*A_dense - (b_plus +a_plus) *np.eye(A_dense.shape[0])) / (b_plus  - a_plus)
# T_minus_dense = (2.0*A_dense - (b_minus+a_minus)*np.eye(A_dense.shape[0])) / (b_minus - a_minus)
#
# for ii in range(k):
#     T_plus_dense = T_plus_dense @ T_plus_dense
#     T_minus_dense = T_minus_dense @ T_minus_dense
#
# Pi_numerator_dense = T_minus_dense - T_plus_dense
#
# Pi_denominator_dense = T_minus_dense + T_plus_dense + 2.0 * np.eye(T_minus_dense.shape[0])
#
# Pi_dense = np.linalg.solve(Pi_denominator_dense, Pi_numerator_dense)
#
# A_plus_dense = Pi_dense @ A_dense
#
# ee_Pi_dense, uu_Pi_dense = sla.eigh(Pi_dense)
#
# plt.figure()
# plt.plot(ee_Pi_dense)
#
# ee_A_dense, uu_A_dense = sla.eigh(A_dense)
#
# ee_A_plus_dense, uu_A_plus_dense = sla.eigh(A_plus_dense)
#
# ray_plus_dense = np.sum(uu_A_dense * (A_plus_dense @ uu_A_dense), axis=0)
#
# plt.figure()
# plt.plot(ee_A_dense)
# plt.plot(ee_A_plus_dense)
# plt.plot(ray_plus_dense)
#
# # if display_progress:
# #     print('inverting spectral projector denominator: Pi_denom^-1')
# # iPi_denominator = Pi_denominator.inv(rtol=rtol, atol=atol)
# #
# # if display_progress:
# #     print('computing spectral projector: Pi_num * Pi_denom^-1')
# # Pi = Pi_numerator * iPi_denominator
#
# # T = A_hmatrix.copy()
#
#
#
#
#
# ############################33
# ################################3
# ####################################
#
# k_plus = 2
# k_minus = 2
# rtol = hpro.default_rtol
# atol = hpro.default_atol
#
# T_plus = A_hmatrix.copy()
# T_plus = (T_plus * 2.0).add_identity(s=-(b_plus + a_plus)) * (1.0 / (b_plus - a_plus))
#
# T_minus = A_hmatrix.copy()
# T_minus = (T_minus * 2.0).add_identity(s=-(b_minus + a_minus)) * (1.0 / (b_minus - a_minus))
#
# if display_progress:
#     print('computing T_plus^(2^k)')
# for ii in range(k_plus):
#     if display_progress:
#         print('computing T_plus^(2^' + str(ii + 1) + ') = T_plus^(2^' + str(ii) + ') * T_plus^(2^' + str(ii) + ')')
#     T_plus = hpro.h_mul(T_plus, T_plus,
#               rtol=rtol, atol=atol,
#               display_progress=display_progress)
#
# if display_progress:
#     print('computing T_minus^(2^k)')
# for ii in range(k_minus):
#     if display_progress:
#         print('computing T_minus^(2^' + str(ii + 1) + ') = T_minus^(2^' + str(ii) + ') * T_minus^(2^' + str(ii) + ')')
#     T_minus = hpro.h_mul(T_minus, T_minus,
#               rtol=rtol, atol=atol,
#               display_progress=display_progress)
#
# # if display_progress:
# #     print('computing Pi_plus = (I + T_plus^(2^k))^-1')
# # Pi_plus = T_plus.add_identity().inv(rtol=rtol, atol=atol)
# #
# # if display_progress:
# #     print('computing Pi_minus = (I + T_minus^(2^k))^-1')
# # Pi_minus = T_minus.add_identity().inv(rtol=rtol, atol=atol)
# #
# # if display_progress:
# #     print('computing Pi_num = Pi_plus - Pi_minus')
# # Pi_num = Pi_plus - Pi_minus
# #
# # if display_progress:
# #     print('computing iPi_denom = (Pi_plus + Pi_minus)^(-1)')
# # iPi_denom = (Pi_plus + Pi_minus).inv(rtol=rtol, atol=atol)
# #
# # if display_progress:
# #     print('computing Pi = Pi_num * iPi_denom')
# # Pi = hpro.h_mul(Pi_num, iPi_denom,
# #                 rtol=rtol, atol=atol,
# #                 display_progress=display_progress)
# #
# # Pi_num_dense = build_dense_matrix_from_matvecs(Pi_num.matvec, Pi_num.shape[1])
# # iPi_denom_dense = build_dense_matrix_from_matvecs(iPi_denom.matvec, iPi_denom.shape[1])
# # Pi_dense = build_dense_matrix_from_matvecs(Pi.matvec, Pi.shape[1])
# #
# # ee_Pi_dense, uu_Pi_dense = sla.eigh(Pi_dense)
#
# if display_progress:
#     print('computing spectral projector numerator Pi_num = T_minus^(2^k) - T_plus^(2^k)')
# # Pi_numerator = T_minus - T_plus
# Pi_numerator = T_plus - T_minus
#
# if display_progress:
#     print('computing spectral projector denominator Pi_denom = 2I + T_minus^(2^k) + T_plus^(2^k)')
# Pi_denominator = (T_minus + T_plus).add_identity(s=2.0)
#
# if display_progress:
#     print('inverting spectral projector denominator: Pi_denom^-1')
# iPi_denominator = Pi_denominator.inv(rtol=rtol, atol=atol)
#
# if display_progress:
#     print('computing spectral projector: Pi_num * Pi_denom^-1')
# Pi = hpro.h_mul(Pi_numerator, iPi_denominator,
#                 rtol=rtol, atol=atol,
#                 display_progress=display_progress)
#
# if display_progress:
#     print('computing A_plus = Pi * A')
# A_plus = hpro.h_mul(Pi, A_hmatrix,
#                      rtol=rtol, atol=atol,
#                      display_progress=display_progress).sym()
#
# # A_plus = A_hmatrix + (A_minus * -1.0)
# # A_plus = A_hmatrix + (A_minus * -2.0)
#
# # T_dense = build_dense_matrix_from_matvecs(T.matvec, T.shape[1])
# Pi_dense = build_dense_matrix_from_matvecs(Pi.matvec, Pi.shape[1])
# ee_Pi, uu_Pi = sla.eigh(Pi_dense)
#
# A_plus_dense = build_dense_matrix_from_matvecs(A_plus.matvec, A_plus.shape[1])
# ee_plus, uu_plus = sla.eigh(A_plus_dense)
# print('np.min(ee_plus)=', np.min(ee_plus))
#
# ray_plus_dense = np.sum(uu_A_dense * (A_plus_dense @ uu_A_dense), axis=0)
#
# plt.figure()
# # plt.plot(ee_plus)
# plt.plot(ray_plus_dense)
# plt.plot(ee_A_dense)
#
#
# plt.figure()
# plt.plot(ee_A_dense,ee_Pi)
#
# ####################################
# ################################3
# ############################33
#
# rtol = hpro.default_rtol
# atol = hpro.default_atol
#
# A_hmatrix = A_hmatrix.sym()
#
# ee_LM, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='LM')
# lambda_max = np.max(ee_LM)
#
# ee_SA, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='SA')
# lambda_min = np.min(ee_SA)
#
# B_hmatrix = (A_hmatrix * (0.5 / np.abs(lambda_min))).add_identity().inv(rtol=1e-4, atol=1e-6)
#
# # B_dense = build_dense_matrix_from_matvecs(B_hmatrix.matvec, B_hmatrix.shape[1])
# # ee_B, uu_B = sla.eigh(B_dense)
#
# aX = 1.0
# bX = 3.0
# kX = 3
#
# TX = B_hmatrix.copy()
# TX = (TX * 2.0).add_identity(s=-(bX + aX)) * (1.0 / (bX - aX))
#
# for ii in range(kX):
#     TX = hpro.h_mul(TX, TX,
#               rtol=rtol, atol=atol,
#               display_progress=display_progress)
#
# ###
#
# rtol = hpro.default_rtol
# atol = hpro.default_atol
#
# A_hmatrix = A_hmatrix.sym()
#
# ee_LM, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='LM')
# lambda_max = np.max(ee_LM)
#
# a = 0.0
# b = 1.5 * lambda_max
# k = 6
#
# T = A_hmatrix.copy()
# T = (T * 2.0).add_identity(s=-(b + a)) * (1.0 / (b - a))
#
# for ii in range(k):
#     T = hpro.h_mul(T, T,
#                    rtol=rtol, atol=atol,
#                    display_progress=display_progress)
#
# Pi_plus = T.add_identity().inv(rtol=rtol, atol=atol)
#
# Pi = (Pi_plus * 2.0).add_identity(s=-1.0)
#
# A_plus = hpro.h_mul(Pi, A_hmatrix,
#                     rtol=rtol, atol=atol,
#                     display_progress=display_progress).sym()
#
# A_dense = build_dense_matrix_from_matvecs(A_hmatrix.matvec, A_hmatrix.shape[1])
# ee, uu = sla.eigh(A_dense)
#
# A_plus_dense = build_dense_matrix_from_matvecs(A_plus.matvec, A_plus.shape[1])
# ee_plus, uu_plus = sla.eigh(A_plus_dense)
#
# ray_plus = np.sum(uu * (A_plus_dense @ uu), axis=0)
#
# plt.figure()
# plt.plot(ee)
# plt.plot(ee_plus)
# plt.plot(ray_plus)
#
# ###
#
# rtol = hpro.default_rtol
# atol = hpro.default_atol
#
# A_hmatrix = A_hmatrix.sym()
#
# ee_SA, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='SA')
# lambda_min = np.min(ee_SA)
#
# a = 1.5*lambda_min
# b = 0.0
# k = 1
#
# T = A_hmatrix.copy()
# T = (T * 2.0).add_identity(s=-(b + a)) * (1.0 / (b - a))
#
# for ii in range(k):
#     T = hpro.h_mul(T, T,
#                    rtol=rtol, atol=atol,
#                    display_progress=display_progress)
#
# Pi_plus = T.add_identity().inv(rtol=rtol, atol=atol)
#
# Pi = (Pi_plus * (-2.0)).add_identity()
#
# A_plus = hpro.h_mul(Pi, A_hmatrix,
#                     rtol=rtol, atol=atol,
#                     display_progress=display_progress).sym()
#
# # A_dense = build_dense_matrix_from_matvecs(A_hmatrix.matvec, A_hmatrix.shape[1])
# # ee, uu = sla.eigh(A_dense)
#
# A_plus_dense = build_dense_matrix_from_matvecs(A_plus.matvec, A_plus.shape[1])
# ee_plus, uu_plus = sla.eigh(A_plus_dense)
#
# ray_plus = np.sum(uu * (A_plus_dense @ uu), axis=0)
#
# plt.figure()
# plt.plot(ee)
# plt.plot(ee_plus)
# plt.plot(ray_plus)
#
# # plt.figure()
# # plt.plot(ray_plus / ee)
#
# # eeX, uuX = sla.eigh(A_dense, A_plus_dense)
# #
# # plt.figure()
# # plt.plot(eeX)
#
# ####
#
#
# eeX, uuX = spla.eigsh(LinopWithCounter(B_hmatrix.as_linear_operator()), k=10, which='LM')
#
#
# B_dense = build_dense_matrix_from_matvecs(B_hmatrix.matvec, B_hmatrix.shape[1])
# ee_B, uu_B = sla.eigh(B_dense)
#
# k1 = 4
# # a1 = np.abs(lambda_min)
# a1 = 0.1*lambda_max
# b1 = 2.0*lambda_max
#
# T1_scaling_at_zero = 1. / (1. + ((2.0 * 0.0 - (b1 + a1)) / (b1 - a1)) ** (2 ** k1))
# print('T1_scaling_at_zero=', T1_scaling_at_zero)
#
# T1 = A_hmatrix.copy()
# T1 = (T1 * 2.0).add_identity(s=-(b1 + a1)) * (1.0 / (b1 - a1))
#
# if display_progress:
#     print('computing T_plus^(2^k)')
# for ii in range(k_plus):
#     if display_progress:
#         print('computing T_plus^(2^' + str(ii + 1) + ') = T_plus^(2^' + str(ii) + ') * T_plus^(2^' + str(ii) + ')')
#     T_plus = hpro.h_mul(T_plus, T_plus,
#               rtol=rtol, atol=atol,
#               display_progress=display_progress)
#
# if display_progress:
#     print('computing T_minus^(2^k)')
# for ii in range(k_minus):
#     if display_progress:
#         print('computing T_minus^(2^' + str(ii + 1) + ') = T_minus^(2^' + str(ii) + ') * T_minus^(2^' + str(ii) + ')')
#     T_minus = hpro.h_mul(T_minus, T_minus,
#               rtol=rtol, atol=atol,
#               display_progress=display_progress)
#
# #
# #
# #
# #
#
# A_pch0 = A_pch_nonsym.sym()
# A0_dense = build_dense_matrix_from_matvecs(A_pch0.matvec, A_pch0.shape[1])
# ee0, uu0 = sla.eigh(A0_dense)
# plt.plot(ee0)
#
# #
#
# ee_T, _ = sla.eigh(T_dense)
#
# #
#
# A2 = hpro.h_mul(A_hmatrix, A_hmatrix,
#                 rtol=rtol, atol=atol,
#                 display_progress=display_progress)
#
# A4 = hpro.h_mul(A2, A2,
#                 rtol=rtol, atol=atol,
#                 display_progress=display_progress)
#
# A8 = hpro.h_mul(A4, A4,
#                 rtol=rtol, atol=atol,
#                 display_progress=display_progress)
#
# ee2, uu2 = spla.eigsh(LinopWithCounter(A2.as_linear_operator()), k=20, which='SA')
# ee4, uu4 = spla.eigsh(LinopWithCounter(A4.as_linear_operator()), k=20, which='SA')
# ee8, uu8 = spla.eigsh(LinopWithCounter(A8.as_linear_operator()), k=20, which='SA')
#
# #
#
# rtol=1e-4
# atol=1e-6
#
# if display_progress:
#     print('getting smallest eigenvalue with Lanczos')
# ee_SA, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='SA')
# lambda_min = np.min(ee_SA)
# if display_progress:
#     print('lambda_min=', lambda_min)
#
# B = (A_hmatrix * (0.5/np.abs(lambda_min))).add_identity().inv(rtol=rtol, atol=atol)
#
# ee_B, _ = spla.eigsh(B.as_linear_operator(), k=1, which='LA')
# print('ee_B=', ee_B)
#
# B2 = hpro.h_mul(B, B,
#                 rtol=rtol, atol=atol,
#                 display_progress=display_progress)
#
# B4 = hpro.h_mul(B2, B2,
#                 rtol=rtol, atol=atol,
#                 display_progress=display_progress)
#
# B8 = hpro.h_mul(B4, B4,
#                 rtol=rtol, atol=atol,
#                 display_progress=display_progress)
#
# ee1, uu1 = spla.eigsh(LinopWithCounter(B.as_linear_operator()), k=100, which='LM')
# ee2, uu2 = spla.eigsh(LinopWithCounter(B2.as_linear_operator()), k=100, which='LM')
# ee4, uu4 = spla.eigsh(LinopWithCounter(B4.as_linear_operator()), k=100, which='LM')
# ee8, uu8 = spla.eigsh(LinopWithCounter(B8.as_linear_operator()), k=100, which='LM')
#
# #########################################
#
# # A_pch = A_pch_nonsym.spd(rtol=1e-3, atol=1e-5)
# A_pch0 = A_pch_nonsym.sym()
# A_pch3 = A_pch_nonsym.spd(k=3, a_factor=0.0)
# A_pch5 = A_pch_nonsym.spd(k=5, a_factor=0.0)
# A_pch7 = A_pch_nonsym.spd(k=7, a_factor=0.0)
# A_pch7 = A_pch_nonsym.spd(k=7, a_factor=0.0, rtol=1e-3, atol=1e-3)
#
# A_pch = A_pch7
#
# A0_dense = build_dense_matrix_from_matvecs(A_pch0.matvec, A_pch0.shape[1])
# A3_dense = build_dense_matrix_from_matvecs(A_pch3.matvec, A_pch3.shape[1])
# A5_dense = build_dense_matrix_from_matvecs(A_pch5.matvec, A_pch5.shape[1])
# A7_dense = build_dense_matrix_from_matvecs(A_pch7.matvec, A_pch7.shape[1])
#
#
# ee0, uu0 = sla.eigh(A0_dense)
# ee3, uu3 = sla.eigh(A3_dense)
# ee5, uu5 = sla.eigh(A5_dense)
# ee7, uu7 = sla.eigh(A7_dense)
#
# print('np.min(ee0)=', np.min(ee0))
# print('np.min(ee3)=', np.min(ee3))
# print('np.min(ee5)=', np.min(ee5))
# print('np.min(ee7)=', np.min(ee7))
#
# plt.figure()
# plt.plot(ee0)
# plt.plot(ee3)
# plt.plot(ee5)
# plt.plot(ee7)
# plt.legend(['ee0', 'ee3', 'ee5', 'ee7'])
#
# # min_reg_param = 1e-3
# # eeR_min, _ = spla.eigsh(min_reg_param * HIP.R0_scipy, k=3, which='SM')
# # eR_min = np.min(eeR_min)
# #
# # A_pch = A_pch_nonsym.spd(cutoff=0.8*eR_min)


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

