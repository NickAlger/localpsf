import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_hmatrix import product_convolution_hmatrix, build_product_convolution_kernel
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 3e-2}

num_batches = 9
num_neighbors = 10
tau = 2.5 #4

########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**nondefault_HIP_options)

PCK = build_product_convolution_kernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc, num_batches,
                                       tau=tau, num_neighbors=num_neighbors)

x = np.array([0.513, 0.467])
y = x
PCK(y,x)

x = np.array([0.513, 0.467])
y = x + 1e-10
PCK(y,x)


x = np.array([0.5, 0.5])
y = np.array([0.5, 0.5])
PCK(y,x)

x = np.array([0.5, 0.5])
y = np.array([0.5, 0.5]) - 1e-10
PCK(y,x)

x = PCK.all_points_list[0]
y = PCK.all_points_list[0]
PCK(y,x)

x = PCK.all_points_list[0]
y = PCK.all_points_list[0] + 1e-10
PCK(y,x)

x = PCK.all_points_list[0]
y = PCK.all_points_list[1]
PCK(y,x)


dof_coords_in = np.array(HIP.V.tabulate_dof_coordinates().T, order='F')
dof_coords_out = dof_coords_in

t = time()
A = PCK(dof_coords_out[:,:100], dof_coords_in[:,:100])
dt_A = time() - t
print('dt_A=', dt_A)

t = time()
Phi_pc = PCK(dof_coords_out, dof_coords_in)
dt_build_pc = time() - t
print('dt_build_pc=', dt_build_pc)

N = HIP.V.dim()
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
ii=1
x = dof_coords_in[:,ii].copy()
v = dl.Function(PCK.V_out)
for jj in tqdm(range(Phi.shape[0])):
    y = dof_coords_out[:,jj].copy()
    v.vector()[jj] = PCK(y, x)

plt.figure()
cm = dl.plot(v)
plt.colorbar(cm)
plt.title('v')

v_true = dl.Function(PCK.V_out)
v_true.vector()[:] = Phi[:,ii].copy()

plt.figure()
cm = dl.plot(v_true)
plt.colorbar(cm)
plt.title('v_true')

dl.norm(v.vector() - v_true.vector()) / dl.norm(v_true.vector())

#

e_fct = dl.Function(PCK.V_out)
# e_fct.vector()[:] = np.linalg.norm(Phi_pc - Phi, axis=0) / np.linalg.norm(Phi, axis=0)
e_fct.vector()[:] = np.linalg.norm(0.5*(Phi_pc.T+Phi_pc) - Phi, axis=0) / np.linalg.norm(Phi, axis=0)

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
