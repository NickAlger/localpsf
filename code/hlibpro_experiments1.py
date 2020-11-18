import numpy as np
from scipy.interpolate import RegularGridInterpolator
from time import time
import hlibpro_experiments1

hlibpro_experiments1.bem1d(100)

min_pt = np.array([-1.2, 0.5])
max_pt = np.array([2.1, 3.3])
deltas = max_pt - min_pt
nx = 50
ny = 43
VG = np.random.randn(nx, ny)

cc = min_pt.reshape((1,-1)) + np.random.rand(62683,2) * deltas.reshape((1,-1))

# plt.plot(cc[:,0], cc[:,1], '.')

t = time()
ve = hlibpro_experiments1.grid_interpolate(cc, min_pt[0], max_pt[0], min_pt[1], max_pt[1], VG)
dt_cpp = time() - t
print('dt_cpp=', dt_cpp)
print(ve)

t = time()
ve1 = hlibpro_experiments1.grid_interpolate_vectorized(cc, min_pt[0], max_pt[0], min_pt[1], max_pt[1], VG)
dt_cpp_vectorized = time() - t
print('dt_cpp_vectorized=', dt_cpp_vectorized)
print(ve1)

xx = np.linspace(min_pt[0], max_pt[0], nx)
yy = np.linspace(min_pt[1], max_pt[1], ny)
RGI = RegularGridInterpolator((xx, yy), VG, method='linear', bounds_error=False, fill_value=0.0)
ve2 = RGI(cc)
print(ve2)

t = time()
err_interp = np.linalg.norm(ve - ve2)
dt_scipy = time() - t
print('dt_scipy=', dt_scipy)
print('err_interp=', err_interp)

####

nx = 85
ny = 97

xmin = -1
xmax = 1.2
ymin = -1.5
ymax = 1.4
xx = np.linspace(xmin, xmax, nx)
yy = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(xx, yy, indexing='xy')
V = np.exp(-(X**2 + Y**2)/0.05)

# plt.matshow(V)
xx2 = np.linspace(xmin-0.1, xmax+0.1, 35)
yy2 = np.linspace(ymin-0.1, ymax-0.1, 47)
X2, Y2 = np.meshgrid(xx2, yy2, indexing='xy')

dof_coords = np.array([X2.reshape(-1), Y2.reshape(-1)]).T
# dof_coords = np.random.randn(1000,2)

rhs_b = np.random.randn(dof_coords.shape[0])

hlibpro_experiments1.Custom_bem1d(dof_coords, xmin, xmax, ymin, ymax, V, rhs_b)

#####

num_batch_points = 3
eta_array = np.random.randn(nx, ny)
ww_arrays = [np.random.randn(nx, ny) for _ in range(3)]
pp = 0.25 * np.random.randn(num_batch_points, 2)
mus = pp + 0.1 * np.random.randn(num_batch_points, 2)
Sigmas = [0.3*np.eye(2), 0.5*np.eye(2), np.eye(2)]
tau = 1.5e10

pcb = hlibpro_experiments1.ProductConvolutionOneBatch(eta_array, ww_arrays, pp, mus, Sigmas, tau, xmin, xmax, ymin, ymax)

num_eval_pts = 53
ss = np.random.randn(num_eval_pts,2)
tt = np.random.randn(num_eval_pts,2)

t = time()
vv = pcb.compute_entries(tt, ss)
dt_entries_cpp = time() - t
print('dt_entries_cpp=', dt_entries_cpp)


####

from eval_product_convolution import *

eval_eta = RegularGridInterpolator((xx, yy), eta_array, method='linear', bounds_error=False, fill_value=0.0)
eval_w0 = RegularGridInterpolator((xx, yy), ww_arrays[0], method='linear', bounds_error=False, fill_value=0.0)
eval_w1 = RegularGridInterpolator((xx, yy), ww_arrays[1], method='linear', bounds_error=False, fill_value=0.0)
eval_w2 = RegularGridInterpolator((xx, yy), ww_arrays[2], method='linear', bounds_error=False, fill_value=0.0)
eval_weighting_functions_w = [[eval_w0, eval_w1, eval_w2]]

eval_eta_batches = [eval_eta]
mean_batches = [mus]
sample_point_batches = [pp]
covariance_batches_Sigma = [np.array(Sigmas)]

PC = BatchProductConvolution(eval_eta_batches, eval_weighting_functions_w,
                             sample_point_batches, mean_batches, covariance_batches_Sigma,
                             tau)

t = time()
vv2 = PC._compute_product_convolution_entries_one_batch(tt, ss, 0)
dt_entries_py = time() - t
print('dt_entries_py=', dt_entries_py)

err_ProductConvolutionOneBatch = np.linalg.norm(vv - vv2)
print('err_ProductConvolutionOneBatch=', err_ProductConvolutionOneBatch)

#####





#
# eval_eta1 = lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1])
# eval_eta2 = lambda x: x[:, 0] ** 2 + 2.3 * x[:, 1] * x[:, 0]
# eval_eta3 = lambda x: np.linalg.norm(x, axis=1)
#
# eval_dirac_comb_responses_eta = [eval_eta1, eval_eta2, eval_eta3]
# sample_point_batches_xx = [np.random.rand(5, 2), np.random.rand(1, 2), np.random.rand(2, 2)]
#
# w1_1 = lambda x: np.sin(x[:, 0] + x[:, 1])
# w1_2 = lambda x: np.sin(x[:, 0] - x[:, 1])
# w1_3 = lambda x: (x[:, 0] + 0.3 * x[:, 1]) ** 2
# w1_4 = lambda x: x[:, 0]
# w1_5 = lambda x: x[:, 1]
#
# w2_1 = lambda x: 0.1 * x[:, 0] + x[:, 1] - 4.
#
# w3_1 = lambda x: np.exp(x[:, 0]) + x[:, 1]
# w3_2 = lambda x: np.log(np.abs(1. + x[:, 0] * x[:, 1]))
#
# eval_weighting_functions_w = [[w1_1, w1_2, w1_3, w1_4, w1_5], [w2_1], [w3_1, w3_2]]
#
# mean_batches_mu = [np.random.rand(5, 2), np.random.rand(1, 2), np.random.rand(2, 2)]
#
#
# def random_spd_matrix():
#     U, ss, _ = np.linalg.svd(np.random.randn(2, 2))
#     return np.dot(U, np.dot(np.diag(ss), U.T))
#
#
# Sigma1 = np.zeros((5, 2, 2))
# for k in range(5):
#     Sigma1[k, :, :] = random_spd_matrix()
#
# Sigma2 = np.zeros((1, 2, 2))
# for k in range(1):
#     Sigma2[k, :, :] = random_spd_matrix()
#
# Sigma3 = np.zeros((2, 2, 2))
# for k in range(2):
#     Sigma3[k, :, :] = random_spd_matrix()
#
# covariance_batches_Sigma = [Sigma1, Sigma2, Sigma3]
#
# num_standard_deviations_tau = 0.5
#
# PC = BatchProductConvolution(eval_dirac_comb_responses_eta, eval_weighting_functions_w,
#                              sample_point_batches_xx, mean_batches_mu, covariance_batches_Sigma,
#                              num_standard_deviations_tau)
#
# xx = np.random.rand(5000, 2)
# yy = np.random.rand(5000, 2)
# t = time()
# hha1 = PC._compute_product_convolution_entries_one_batch_slow(yy, xx, 0)
# dt1 = time() - t
# print('dt1=', dt1)
# t = time()
# hha2 = PC._compute_product_convolution_entries_one_batch(yy, xx, 0)
# dt2 = time() - t
# print('dt2=', dt2)
# err1 = np.linalg.norm(hha1 - hha2)
# print('err1=', err1)
#
# hhb1 = PC._compute_product_convolution_entries_one_batch_slow(yy, xx, 1)
# hhb2 = PC._compute_product_convolution_entries_one_batch(yy, xx, 1)
# err2 = np.linalg.norm(hhb1 - hhb2)
# print('err2=', err2)
#
# hhc1 = PC._compute_product_convolution_entries_one_batch_slow(yy, xx, 2)
# hhc2 = PC._compute_product_convolution_entries_one_batch(yy, xx, 2)
# err3 = np.linalg.norm(hhc1 - hhc2)
# print('err3=', err3)
#
# h = PC.compute_product_convolution_entries(yy, xx)
# h2 = hha1 + hhb1 + hhc1
# err_total = np.linalg.norm(h - h2)
# print('err_total=', err_total)