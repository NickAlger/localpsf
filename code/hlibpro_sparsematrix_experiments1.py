import numpy as np
import scipy.sparse as sps
import fenics
from scipy.io import savemat
from time import time
import hlibpro_experiments1 as hpro

default_rtol = 1e-6
default_atol = 1e-16

def h_add(A_hmatrix, B_hmatrix, alpha=1.0, beta=1.0, rtol=default_rtol, atol=default_atol):
    # C = A + alpha * B to tolerance given by truncation accuracy object acc
    acc = hpro.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    C_hmatrix = B_hmatrix.copy()
    hpro.add(alpha, A_hmatrix, beta, C_hmatrix, acc)
    return C_hmatrix

def h_scale(A_hmatrix, alpha):
    # C = alpha * A
    C_hmatrix = A_hmatrix.copy()
    C_hmatrix.scale(alpha)
    return C_hmatrix

def h_mul(A_hmatrix, B_hmatrix, alpha=1.0, rtol=default_rtol, atol=default_atol, display_progress=True):
    # C = A * B
    acc = hpro.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    C_hmatrix = A_hmatrix.copy()
    # C_hmatrix = B_hmatrix.copy()
    C_hmatrix.set_nonsym()
    A_hmatrix.set_nonsym()
    B_hmatrix.set_nonsym()
    if display_progress:
        hpro.multiply_with_progress_bar(alpha, hpro.apply_normal, A_hmatrix,
                                        hpro.apply_normal, B_hmatrix, 0.0, C_hmatrix, acc)
    else:
        hpro.multiply_without_progress_bar(alpha, hpro.apply_normal, A_hmatrix,
                                           hpro.apply_normal, B_hmatrix, 0.0, C_hmatrix, acc)

    # A_hmatrix.set_symmetric()
    # B_hmatrix.set_symmetric()
    return C_hmatrix

def h_factorized_inverse(A_hmatrix, rtol=default_rtol, atol=default_atol, display_progress=True):
    acc = hpro.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    iA_factors = A_hmatrix.copy()
    iA_hmatrix = hpro.factorize_inv_with_progress_bar(iA_factors, acc)
    return iA_hmatrix, iA_factors

mesh = fenics.UnitSquareMesh(85, 95)
V = fenics.FunctionSpace(mesh, 'CG', 1)
dof_coords = V.tabulate_dof_coordinates()

u_trial = fenics.TrialFunction(V)
v_test = fenics.TestFunction(V)

stiffness_form = fenics.inner(fenics.grad(u_trial), fenics.grad(v_test))*fenics.dx
mass_form = u_trial*v_test*fenics.dx

K = fenics.assemble(stiffness_form)
M = fenics.assemble(mass_form)

def convert_fenics_csr_matrix_to_scipy_csr_matrix(A_fenics):
    ai, aj, av = fenics.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy

K_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(K).tocsc()
M_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(M).tocsc()
A_csc = K_csc + M_csc

stiffness_fname = 'sparse_stiffness_matrix.mat'
savemat(stiffness_fname, {'K': K_csc})

mass_fname = 'sparse_mass_matrix.mat'
savemat(mass_fname, {'M': M_csc})

#

hpro.initialize_hlibpro()
ct = hpro.build_cluster_tree_from_dof_coords(dof_coords, 60)
bct = hpro.build_block_cluster_tree(ct, ct, 2.0)
hpro.visualize_cluster_tree(ct, "sparsemat_cluster_tree_from_python")
hpro.visualize_block_cluster_tree(bct, "sparsemat_block_cluster_tree_from_python")

K_hmatrix = hpro.build_hmatrix_from_sparse_matfile (stiffness_fname, ct, ct, bct)
hpro.visualize_hmatrix(K_hmatrix, "stiffness_hmatrix_from_python")

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(K_hmatrix, ct, ct, x)

y2 = K_csc * x
err_hmat_stiffness = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_hmat_stiffness=', err_hmat_stiffness)

three_K_hmatrix = h_scale(K_hmatrix, 3.0)
three_y = hpro.h_matvec(three_K_hmatrix, ct, ct, x)

err_h_scale = np.linalg.norm(three_y - 3.0*y)
print('err_h_scale=', err_h_scale)

M_hmatrix = hpro.build_hmatrix_from_sparse_matfile (mass_fname, ct, ct, bct)
hpro.visualize_hmatrix(M_hmatrix, "mass_hmatrix_from_python")

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(M_hmatrix, ct, ct, x)

y2 = M_csc * x
err_hmat_mass = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_hmat_mass=', err_hmat_mass)

########    ADD    ########

A_hmatrix = h_add(K_hmatrix, M_hmatrix)

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(A_hmatrix, ct, ct, x)

y2 = A_csc * x
err_h_add = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_h_add=', err_h_add)

########    MULT    ########

KM_hmatrix = h_mul(K_hmatrix, M_hmatrix, rtol=1e-15)

x = np.random.randn(dof_coords.shape[0])
# x = np.ones(dof_coords.shape[0])
y = hpro.h_matvec(KM_hmatrix, ct, ct, x)

y2 = hpro.h_matvec(K_hmatrix, ct, ct, hpro.h_matvec(M_hmatrix, ct, ct, x))
y3 = K_csc * (M_csc * x)
err_H_mul = np.linalg.norm(y2 - y)/np.linalg.norm(y2)
print('err_H_mul=', err_H_mul)

########    FACTORIZE    ########

iA_hmatrix, iA_factors = h_factorized_inverse(A_hmatrix, rtol=1e-9)

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(A_hmatrix, ct, ct, x)

x2 = hpro.h_factorized_inverse_matvec(iA_hmatrix, ct, ct, y)

err_hfac = np.linalg.norm(x - x2)/np.linalg.norm(x2)
print('err_hfac=', err_hfac)

# inv_A = hpro.hmatrix_factorized_inverse_destructive(A_hmatrix, 1e-4)
# x2 = hpro.hmatrix_factorized_inverse_matvec(inv_A, ct, ct, y)
#
# err_hfac = np.linalg.norm(x - x2)/np.linalg.norm(x2)
# print('err_hfac=', err_hfac)
#
#
# ######
#
# hpro.bem1d(100)
#
# min_pt = np.array([-1.2, 0.5])
# max_pt = np.array([2.1, 3.3])
# deltas = max_pt - min_pt
# nx = 50
# ny = 43
# VG = np.random.randn(nx, ny)
#
# cc = min_pt.reshape((1,-1)) + np.random.rand(62683,2) * deltas.reshape((1,-1))
#
# # plt.plot(cc[:,0], cc[:,1], '.')
#
# t = time()
# ve = hpro.grid_interpolate(cc, min_pt[0], max_pt[0], min_pt[1], max_pt[1], VG)
# dt_cpp = time() - t
# print('dt_cpp=', dt_cpp)
# print(ve)
#
# t = time()
# ve1 = hpro.grid_interpolate_vectorized(cc, min_pt[0], max_pt[0], min_pt[1], max_pt[1], VG)
# dt_cpp_vectorized = time() - t
# print('dt_cpp_vectorized=', dt_cpp_vectorized)
# print(ve1)
#
# xx = np.linspace(min_pt[0], max_pt[0], nx)
# yy = np.linspace(min_pt[1], max_pt[1], ny)
# RGI = RegularGridInterpolator((xx, yy), VG, method='linear', bounds_error=False, fill_value=0.0)
# ve2 = RGI(cc)
# print(ve2)
#
# t = time()
# err_interp = np.linalg.norm(ve - ve2)
# dt_scipy = time() - t
# print('dt_scipy=', dt_scipy)
# print('err_interp=', err_interp)
#
# ####
#
# nx = 85
# ny = 97
#
# xmin = -1
# xmax = 1.2
# ymin = -1.5
# ymax = 1.4
# xx = np.linspace(xmin, xmax, nx)
# yy = np.linspace(ymin, ymax, ny)
# X, Y = np.meshgrid(xx, yy, indexing='xy')
# V = np.exp(-(X**2 + Y**2)/0.05)
#
# # plt.matshow(V)
# xx2 = np.linspace(xmin-0.1, xmax+0.1, 35)
# yy2 = np.linspace(ymin-0.1, ymax-0.1, 47)
# X2, Y2 = np.meshgrid(xx2, yy2, indexing='xy')
#
# dof_coords = np.array([X2.reshape(-1), Y2.reshape(-1)]).T
# # dof_coords = np.random.randn(1000,2)
#
# rhs_b = np.random.randn(dof_coords.shape[0])
#
# hpro.Custom_bem1d(dof_coords, xmin, xmax, ymin, ymax, V, rhs_b)
#
# #####
#
# num_batch_points = 3
# eta_array = np.random.randn(nx, ny)
# ww_arrays = [np.random.randn(nx, ny) for _ in range(3)]
# pp = 0.25 * np.random.randn(num_batch_points, 2)
# mus = pp + 0.1 * np.random.randn(num_batch_points, 2)
# Sigmas = [0.3*np.eye(2), 0.5*np.eye(2), np.eye(2)]
# tau = 1.5
#
# pcb = hpro.ProductConvolutionOneBatch(eta_array, ww_arrays, pp, mus, Sigmas, tau, xmin, xmax, ymin, ymax)
#
# num_eval_pts = 53
# ss = np.random.randn(num_eval_pts,2)
# tt = np.random.randn(num_eval_pts,2)
#
# t = time()
# vv = pcb.compute_entries(tt, ss)
# dt_entries_cpp = time() - t
# print('dt_entries_cpp=', dt_entries_cpp)
#
#
# ####
#
# from eval_product_convolution import *
#
# eval_eta = RegularGridInterpolator((xx, yy), eta_array, method='linear', bounds_error=False, fill_value=0.0)
# eval_w0 = RegularGridInterpolator((xx, yy), ww_arrays[0], method='linear', bounds_error=False, fill_value=0.0)
# eval_w1 = RegularGridInterpolator((xx, yy), ww_arrays[1], method='linear', bounds_error=False, fill_value=0.0)
# eval_w2 = RegularGridInterpolator((xx, yy), ww_arrays[2], method='linear', bounds_error=False, fill_value=0.0)
# eval_weighting_functions_w = [[eval_w0, eval_w1, eval_w2]]
#
# eval_eta_batches = [eval_eta]
# mean_batches = [mus]
# sample_point_batches = [pp]
# covariance_batches_Sigma = [np.array(Sigmas)]
#
# PC = BatchProductConvolution(eval_eta_batches, eval_weighting_functions_w,
#                              sample_point_batches, mean_batches, covariance_batches_Sigma,
#                              tau)
#
# t = time()
# vv2 = PC._compute_product_convolution_entries_one_batch(tt, ss, 0)
# dt_entries_py = time() - t
# print('dt_entries_py=', dt_entries_py)
#
# err_ProductConvolutionOneBatch = np.linalg.norm(vv - vv2)
# print('err_ProductConvolutionOneBatch=', err_ProductConvolutionOneBatch)
#
# ####
#
# nb1 = 3
# nb2 = 4
#
# eta_array_batches = [np.random.randn(nx, ny), np.random.randn(nx, ny)]
# ww_array_batches = [[np.random.randn(nx, ny) for _ in range(nb1)], [np.random.randn(nx, ny) for _ in range(nb2)]]
# pp_batches = [0.25 * np.random.randn(nb1, 2), 0.25 * np.random.randn(nb2, 2)]
# mus_batches = [pp_batches[0] + 0.1 * np.random.randn(nb1, 2), pp_batches[1] + 0.1 * np.random.randn(nb2, 2)]
# Sigmas_batches = [[0.3*np.eye(2), 0.5*np.eye(2), np.eye(2)], [0.3*np.eye(2), 0.5*np.eye(2), np.eye(2), 0.8*np.eye(2)]]
# tau = 1.2
#
# pcb_multi = hpro.ProductConvolutionMultipleBatches(eta_array_batches,
#                                                                    ww_array_batches,
#                                                                    pp_batches,
#                                                                    mus_batches,
#                                                                    Sigmas_batches,
#                                                                    tau, xmin, xmax, ymin, ymax)
#
#
# t = time()
# vv_multi = pcb_multi.compute_entries(tt, ss)
# dt_entries_multi_cpp = time() - t
# print('dt_entries_multi_cpp=', dt_entries_multi_cpp)
#
# #
#
# eval_eta0 = RegularGridInterpolator((xx, yy), eta_array_batches[0], method='linear', bounds_error=False, fill_value=0.0)
# eval_eta1 = RegularGridInterpolator((xx, yy), eta_array_batches[1], method='linear', bounds_error=False, fill_value=0.0)
# eval_eta_batches = [eval_eta0, eval_eta1]
#
# eval_w00 = RegularGridInterpolator((xx, yy), ww_array_batches[0][0], method='linear', bounds_error=False, fill_value=0.0)
# eval_w01 = RegularGridInterpolator((xx, yy), ww_array_batches[0][1], method='linear', bounds_error=False, fill_value=0.0)
# eval_w02 = RegularGridInterpolator((xx, yy), ww_array_batches[0][2], method='linear', bounds_error=False, fill_value=0.0)
#
# eval_w10 = RegularGridInterpolator((xx, yy), ww_array_batches[1][0], method='linear', bounds_error=False, fill_value=0.0)
# eval_w11 = RegularGridInterpolator((xx, yy), ww_array_batches[1][1], method='linear', bounds_error=False, fill_value=0.0)
# eval_w12 = RegularGridInterpolator((xx, yy), ww_array_batches[1][2], method='linear', bounds_error=False, fill_value=0.0)
# eval_w13 = RegularGridInterpolator((xx, yy), ww_array_batches[1][3], method='linear', bounds_error=False, fill_value=0.0)
#
# eval_ww_batches = [[eval_w00, eval_w01, eval_w02], [eval_w10, eval_w11, eval_w12, eval_w13]]
#
# Sigmas_tensor_batches = [np.array(Sigma_batch) for Sigma_batch in Sigmas_batches]
#
# PC_multi = BatchProductConvolution(eval_eta_batches,
#                                    eval_ww_batches,
#                                    pp_batches,
#                                    mus_batches,
#                                    Sigmas_tensor_batches,
#                                    tau)
#
# t = time()
# vv_multi2 = PC_multi.compute_product_convolution_entries(tt, ss)
# dt_entries_multi_py = time() - t
# print('dt_entries_multi_py=', dt_entries_multi_py)
#
# err_PCmulti = np.linalg.norm(vv_multi - vv_multi2)
# print('err_PCmulti=', err_PCmulti)
#
# ####
#
#
#
# ####
#
#
