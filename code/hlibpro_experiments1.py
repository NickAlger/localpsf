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
tau = 1.5

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

####

nb1 = 3
nb2 = 4

eta_array_batches = [np.random.randn(nx, ny), np.random.randn(nx, ny)]
ww_array_batches = [[np.random.randn(nx, ny) for _ in range(nb1)], [np.random.randn(nx, ny) for _ in range(nb2)]]
pp_batches = [0.25 * np.random.randn(nb1, 2), 0.25 * np.random.randn(nb2, 2)]
mus_batches = [pp_batches[0] + 0.1 * np.random.randn(nb1, 2), pp_batches[1] + 0.1 * np.random.randn(nb2, 2)]
Sigmas_batches = [[0.3*np.eye(2), 0.5*np.eye(2), np.eye(2)], [0.3*np.eye(2), 0.5*np.eye(2), np.eye(2), 0.8*np.eye(2)]]
tau = 1.2

pcb_multi = hlibpro_experiments1.ProductConvolutionMultipleBatches(eta_array_batches,
                                                                   ww_array_batches,
                                                                   pp_batches,
                                                                   mus_batches,
                                                                   Sigmas_batches,
                                                                   tau, xmin, xmax, ymin, ymax)


t = time()
vv_multi = pcb_multi.compute_entries(tt, ss)
dt_entries_multi_cpp = time() - t
print('dt_entries_multi_cpp=', dt_entries_multi_cpp)

#

eval_eta0 = RegularGridInterpolator((xx, yy), eta_array_batches[0], method='linear', bounds_error=False, fill_value=0.0)
eval_eta1 = RegularGridInterpolator((xx, yy), eta_array_batches[1], method='linear', bounds_error=False, fill_value=0.0)
eval_eta_batches = [eval_eta0, eval_eta1]

eval_w00 = RegularGridInterpolator((xx, yy), ww_array_batches[0][0], method='linear', bounds_error=False, fill_value=0.0)
eval_w01 = RegularGridInterpolator((xx, yy), ww_array_batches[0][1], method='linear', bounds_error=False, fill_value=0.0)
eval_w02 = RegularGridInterpolator((xx, yy), ww_array_batches[0][2], method='linear', bounds_error=False, fill_value=0.0)

eval_w10 = RegularGridInterpolator((xx, yy), ww_array_batches[1][0], method='linear', bounds_error=False, fill_value=0.0)
eval_w11 = RegularGridInterpolator((xx, yy), ww_array_batches[1][1], method='linear', bounds_error=False, fill_value=0.0)
eval_w12 = RegularGridInterpolator((xx, yy), ww_array_batches[1][2], method='linear', bounds_error=False, fill_value=0.0)
eval_w13 = RegularGridInterpolator((xx, yy), ww_array_batches[1][3], method='linear', bounds_error=False, fill_value=0.0)

eval_ww_batches = [[eval_w00, eval_w01, eval_w02], [eval_w10, eval_w11, eval_w12, eval_w13]]

Sigmas_tensor_batches = [np.array(Sigma_batch) for Sigma_batch in Sigmas_batches]

PC_multi = BatchProductConvolution(eval_eta_batches,
                                   eval_ww_batches,
                                   pp_batches,
                                   mus_batches,
                                   Sigmas_tensor_batches,
                                   tau)

t = time()
vv_multi2 = PC_multi.compute_product_convolution_entries(tt, ss)
dt_entries_multi_py = time() - t
print('dt_entries_multi_py=', dt_entries_multi_py)

err_PCmulti = np.linalg.norm(vv_multi - vv_multi2)
print('err_PCmulti=', err_PCmulti)

####

hlibpro_experiments1.initialize_hlibpro()
ct = hlibpro_experiments1.build_cluster_tree_from_dof_coords(dof_coords, 60)
bct = hlibpro_experiments1.build_block_cluster_tree(ct, ct, 2.0)
hlibpro_experiments1.visualize_cluster_tree(ct, "cluster_tree_from_python")
hlibpro_experiments1.visualize_block_cluster_tree(bct, "block_cluster_tree_from_python")