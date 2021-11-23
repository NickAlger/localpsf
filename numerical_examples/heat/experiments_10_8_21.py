import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix, build_product_convolution_kernel, ImpulseResponseBatches, ProductConvolutionKernelRBF
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
from localpsf.estimate_column_errors_randomized import *

import scipy.sparse.linalg as spla

########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 2e-2} # {'mesh_h': 3e-2}

num_batches = 10
num_neighbors = 10
tau = 2.5 # 4
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid

########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**nondefault_HIP_options)

# IRB = ImpulseResponseBatches(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc)
#
# inds = IRB.add_one_sample_point_batch()
#
# IRB.cpp_object.interpolation_points_and_values(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
# IRB.cpp_object.interpolation_points_and_values(np.random.randn(2), np.random.randn(2))

#

PCK = ProductConvolutionKernelRBF(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                                  num_batches, num_batches,
                                  tau_rows=tau, tau_cols=tau,
                                  num_neighbors_rows=num_neighbors,
                                  num_neighbors_cols=num_neighbors,
                                  symmetric=True,
                                  gamma=gamma,
                                  sigma_min=sigma_min)

# PCK = ProductConvolutionKernelRBF(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
#                                   0, num_batches,
#                                   tau_rows=tau, tau_cols=tau,
#                                   num_neighbors_rows=num_neighbors,
#                                   num_neighbors_cols=num_neighbors,
#                                   symmetric=False,
#                                   gamma=gamma)

PCK.cpp_object.eval_integral_kernel(np.array([0.5, 0.5]), np.array([0.5, 0.5]))

#

PCK.col_batches.visualize_impulse_response_batch(0)

#

ct = hpro.build_cluster_tree_from_pointcloud(PCK.col_coords, cluster_size_cutoff=50)
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

# PCK.gamma = 1e-4
Phi_pch = PCK.build_hmatrix(bct, tol=1e-5)

v = np.random.randn(Phi_pch.shape[1])
z1 = Phi_pch * v
z2 = HIP.apply_iM_Hd_iM_numpy(v)

err_hmatrix_vs_true = np.linalg.norm(z1 - z2) / np.linalg.norm(z2)
print('err_hmatrix_vs_true=', err_hmatrix_vs_true)

Phi_pch.visualize('Phi_pch1')

#

err_pch_fro, e_fct = estimate_column_errors_randomized(HIP.apply_iM_Hd_iM_numpy,
                                                        lambda x: Phi_pch * x,
                                                        HIP.V, 50)

print('err_pch_fro=', err_pch_fro)

column_error_plot(e_fct, PCK.col_batches.sample_points)

#

z1 = np.random.randn(Phi_pch.shape[0])
z2 = np.random.randn(Phi_pch.shape[1])

ip1 = np.dot(Phi_pch * z1, z2)
ip2 = np.dot(z1, Phi_pch * z2)
err_sym = np.abs(ip1 - ip2) / (np.abs(ip1) + np.abs(ip2))
print('err_sym', err_sym)

#

Phi_pch_plus = Phi_pch.spd()

err_pch_plus_fro, e_plus_fct = estimate_column_errors_randomized(HIP.apply_iM_Hd_iM_numpy,
                                                                 lambda x: Phi_pch_plus * x,
                                                                 HIP.V, 50)

print('err_pch_plus_fro=', err_pch_plus_fro)

column_error_plot(e_plus_fct, PCK.col_batches.sample_points)
plt.title('column errors spd')


#

Phi_linop = spla.LinearOperator(Phi_pch.shape, matvec=lambda x: HIP.apply_iM_Hd_iM_numpy(x))
err_linop = spla.LinearOperator(Phi_pch.shape, matvec=lambda x: HIP.apply_iM_Hd_iM_numpy(x) - Phi_pch*x)

aa,_ = spla.eigsh(Phi_linop, k=10, which='LM')
ee,_ = spla.eigsh(err_linop, k=10, which='LM')

err_pch_induced2 = np.max(np.abs(ee)) / np.max(np.abs(aa))
print('err_pch_induced2=', err_pch_induced2)

#

err_plus_linop = spla.LinearOperator(Phi_pch.shape, matvec=lambda x: HIP.apply_iM_Hd_iM_numpy(x) - Phi_pch_plus*x)

ee_plus,_ = spla.eigsh(err_plus_linop, k=10, which='LM')

err_pch_plus_induced2 = np.max(np.abs(ee_plus)) / np.max(np.abs(aa))
print('err_pch_plus_induced2=', err_pch_plus_induced2)

#

dof_coords_in = np.array(HIP.V.tabulate_dof_coordinates().T, order='F')
dof_coords_out = dof_coords_in

t = time()
# A = PCK[:100, :100]
A = PCK(dof_coords_out[:,:100], dof_coords_in[:,:100])
dt_A = time() - t
print('dt_A=', dt_A)

t = time()
Phi_pc = PCK[:,:]
# Phi_pc = PCK(dof_coords_out, dof_coords_in)
dt_build_pc = time() - t
print('dt_build_pc=', dt_build_pc)

z0 = np.dot(Phi_pc, v)

#



# N = HIP.V.dim()
# Phi_pc = np.zeros((N,N))
# for ii in tqdm(range(N)):
#     for jj in range(N):
#         x = dof_coords_in[ii,:].copy()
#         y = dof_coords_out[jj,:].copy()
#         Phi_pc[ii,jj] = PCK(y,x)

t = time()
Phi = build_dense_matrix_from_matvecs(HIP.apply_iM_Hd_iM_numpy, HIP.V.dim())
dt_build_dense = time() - t
print('dt_build_dense', dt_build_dense)

#

err_fro = np.linalg.norm(Phi_pc - Phi) / np.linalg.norm(Phi)
print('err_fro=', err_fro)

err_induced2 = np.linalg.norm(Phi_pc - Phi,2) / np.linalg.norm(Phi,2)
print('err_induced2=', err_induced2)

#

Phi_pc_sym = 0.5*(Phi_pc.T+Phi_pc)

err_fro_sym = np.linalg.norm(Phi_pc_sym - Phi) / np.linalg.norm(Phi)
print('err_fro_sym=', err_fro_sym)

err_induced2_sym = np.linalg.norm(Phi_pc_sym - Phi,2) / np.linalg.norm(Phi,2)
print('err_induced2_sym=', err_induced2_sym)


# x = np.array([0.513, 0.467])
# ii=1
# x = dof_coords_in[:,ii].copy()
# v = dl.Function(PCK.V_out)
# for jj in tqdm(range(Phi.shape[0])):
#     y = dof_coords_out[:,jj].copy()
#     v.vector()[jj] = PCK(y, x)
#
# plt.figure()
# cm = dl.plot(v)
# plt.colorbar(cm)
# plt.title('v')
#
# v_true = dl.Function(PCK.V_out)
# v_true.vector()[:] = Phi[:,ii].copy()
#
# plt.figure()
# cm = dl.plot(v_true)
# plt.colorbar(cm)
# plt.title('v_true')
#
# dl.norm(v.vector() - v_true.vector()) / dl.norm(v_true.vector())

#

e_fct = dl.Function(PCK.V_out)
e_fct.vector()[:] = np.linalg.norm(Phi_pc - Phi, axis=0) / np.linalg.norm(Phi, axis=0)
# e_fct.vector()[:] = np.linalg.norm(0.5*(Phi_pc.T+Phi_pc) - Phi, axis=0) / np.linalg.norm(Phi, axis=0)

plt.figure()
cm = dl.plot(e_fct)
plt.colorbar(cm)

####

PCK.visualize_impulse_response_batch(0)
PCK.visualize_weighting_function(0)
PCK.visualize_weighting_function(10)

Phi_pc = PCK.build_dense_integral_kernel()

v = np.random.randn(HIP.V.dim())
y = HIP.apply_iM_Hd_iM_numpy(v)
y_pc = np.dot(Phi_pc, v)
matvec_err = np.linalg.norm(y - y_pc)/np.linalg.norm(y)
print('matvec_err=', matvec_err)

Phi = build_dense_matrix_from_matvecs(HIP.apply_iM_Hd_iM_numpy, HIP.V.dim())

fro_err = np.linalg.norm(Phi - Phi_pc) / np.linalg.norm(Phi)
print('fro_err=', fro_err)

e_fct = dl.Function(HIP.V)
e_fct.vector()[:] = np.linalg.norm(Phi_pc - Phi, axis=0) / np.linalg.norm(Phi, axis=0)
plt.figure()
cm = dl.plot(e_fct)
plt.colorbar(cm)

# tt = np.linspace(-1., 1., 20)
# X, Y = np.meshgrid(tt, tt)
#
# pp = np.stack([X.reshape(-1), Y.reshape(-1)]).T.copy()
#
# ll = PCK.eval_one_batch_of_convolution_kernels_at_points(pp.copy(), 0, reflect=False)

########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########
#
# all_Hd_hmatrix = list()
# all_extras = list()
# for k in all_batch_sizes:
#     Hd_hmatrix, extras = product_convolution_hmatrix(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, k,
#                                                      hmatrix_tol=hmatrix_rtol, make_positive_definite=True,
#                                                      return_extras=True, grid_density_multiplier=0.25)
#     all_Hd_hmatrix.append(Hd_hmatrix)
#     all_extras.append(extras)

# k = 0
# pk = extras['point_batches'][0][k, :]
# Fk = extras['FF'][k].translate(pk)
# Fk.plot(title='Fk')
#
# plt.figure()
# Wk = extras['WW'][k]
# Wk.plot(title='Wk')
#
# fk = extras['ff_batches'][0]
# plt.figure()
# cm = dl.plot(fk)
# plt.title('fk')
# plt.colorbar(cm)
#
# V = fk.function_space()
# X = V.tabulate_dof_coordinates()
#
# fk2 = dl.Function(V)
# fk2.vector()[:] = Fk(X)
# fk2.set_allow_extrapolation(True)
#
# plt.figure()
# cm = dl.plot(fk2)
# plt.colorbar(cm)
# plt.title('fk2')
#
# p = np.array([0.96, 0.51])
# e = (fk(p) - fk2(p)) / fk(p)
# print('e=', e)
#
# N = Fk.gridpoints.shape[0]
# ff3 = np.zeros(N)
# for ii in range(N):
#     ff3[ii] = fk(Fk.gridpoints[ii,:])
#
# from nalger_helper_functions import *
#
# F3_arr = ff3.reshape(Fk.shape)
# F3 = BoxFunction(Fk.min, Fk.max, F3_arr)
# F3.plot(title='F3')
#
# (F3 - Fk).plot()
#
# ########    COMPUTE REGULARIZATION PARAMETER VIA MOROZOV DISCREPANCY PRINCIPLE    ########
#
# best_Hd_hmatrix = all_Hd_hmatrix[-1]
#
# R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, best_Hd_hmatrix.bct)
#
# def solve_inverse_problem(a_reg):
#     HIP.regularization_parameter = a_reg
#
#     g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
#     H_hmatrix = Hd_hmatrix + a_reg * R0_hmatrix
#     iH_hmatrix = H_hmatrix.inv()
#
#     u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
#                                           M=iH_hmatrix.as_linear_operator(),
#                                           tol=1e-10, maxiter=500)
#
#     u0 = dl.Function(HIP.V)
#     u0.vector()[:] = u0_numpy
#
#     return u0.vector() # dolfin Vector
#
# a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
#                                                          HIP.morozov_discrepancy,
#                                                          HIP.noise_Mnorm)
#
#
# ########    COMPUTE PRECONDITIONED SPECTRUM    ########
#
# all_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
# all_abs_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
# for kk_batch in range(len(all_batch_sizes)):
#     print('batch size=', all_batch_sizes[kk_batch])
#     Hd_hmatrix = all_Hd_hmatrix[kk_batch]
#     H_hmatrix = Hd_hmatrix + a_reg_morozov * R0_hmatrix
#     iH_hmatrix = H_hmatrix.inv()
#     delta_hmatrix_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x: HIP.apply_H_numpy(x) - H_hmatrix * x)
#     ee_hmatrix, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=H_hmatrix.as_linear_operator(),
#                                Minv=iH_hmatrix.as_linear_operator(), which='LM')
#     abs_ee_hmatrix = np.sort(np.abs(ee_hmatrix))[::-1]
#
#     all_ee_hmatrix[kk_batch, :] = ee_hmatrix
#     all_abs_ee_hmatrix[kk_batch, :] = abs_ee_hmatrix
#
# delta_reg_linop = spla.LinearOperator((HIP.N, HIP.N), matvec=lambda x: HIP.apply_H_numpy(x) - HIP.apply_R_numpy(x))
# ee_reg, _ = spla.eigsh(delta_reg_linop, k=num_eigs, M=HIP.R_linop, Minv=HIP.solve_R_linop, which='LM')
# abs_ee_reg = np.sort(np.abs(ee_reg))[::-1]
#
# if save_results:
#     # Save HIP_options
#     with open(HIP_options_file, 'w') as f:
#         print(HIP.options, file=f)
#
#     # Save options
#     with open(options_file, 'w') as f:
#         print(options, file=f)
#
#     # Save data
#     np.savez(data_file,
#              abs_ee_reg=abs_ee_reg,
#              all_abs_ee_hmatrix=all_abs_ee_hmatrix,
#              a_reg_morozov=a_reg_morozov,
#              all_batch_sizes=all_batch_sizes)


#
# ########    MAKE FIGURE    ########
#
# plt.figure()
# plt.semilogy(abs_ee_reg)
# for abs_ee_hmatrix in all_abs_ee_hmatrix:
#     plt.semilogy(abs_ee_hmatrix)
#
# plt.title(r'Absolute values of eigenvalues of $P^{-1}H-I$')
# plt.xlabel(r'$i$')
# plt.ylabel(r'$|\lambda_i|$')
# plt.legend(['Reg'] + ['PCH' + str(nB) for nB in all_batch_sizes])
#
# plt.show()
#
# if save_results:
#     plt.savefig(plot_file, bbox_inches='tight', dpi=100)
