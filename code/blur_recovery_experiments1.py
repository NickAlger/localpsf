import numpy as np
# from dolfin import *
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy import signal
from numpy import fft
from gaussblur import GaussBlurDerivatives, gauss, grad_gauss, periodic_displacement, make_regular_grid_2d, make_periodic_2d_laplacian, circular_translate, make_blur_operator, make_gradblur_tensor
from cross import py_ica
from covariance_estimation import compute_col_moments, distance_from_boundary_function, mark_points_near_boundary, compute_boundary_normal_std

n=50
# n = 150
ns0 = 5 # for show
ns = 1*ns0
n_diag_basis = int(0.65*(ns * ns))
# oscillation_width = 0.05
oscillation_width = 0.0
m=5

def true_kernel_function(C, p, qq):
    # xi = periodic_displacement(p,qq)
    xi = p - qq
    ndx = np.linalg.norm(xi, axis=0)
    return np.cos(oscillation_width * np.pi * ndx) * gauss(C, xi)

def approx_kernel_function(C, p, qq):
    return gauss(C, p - qq)
    # return gauss(C, periodic_displacement(p, qq))

def grad_approx_kernel_function(C, p, qq):
    return grad_gauss(C, p - qq)
    # return grad_gauss(C, periodic_displacement(p, qq))


qq = make_regular_grid_2d(n)
boundary_inds = np.where(np.any(1 - np.abs(qq) < 1e-5, axis=0))[0]
dd = distance_from_boundary_function(qq, boundary_inds)

sigmas = 0.02+(0.07 + 0.04 * np.sin(np.pi * qq[0,:])*np.cos(np.pi * qq[1,:])).reshape([1,-1])
plt.matshow(sigmas.reshape((n,n)))
plt.title('local covariance length')
CC_true = np.einsum('mn,ij->nij', sigmas ** 2, np.eye(2))

B_true = make_blur_operator(true_kernel_function, CC_true, qq)
# B = (B + B.T)/2.

mass_matrix = (4./((n-1)*(n-1))) * np.eye(n*n)

def apply_B(v):
    return np.dot(B_true, v)

def apply_Bt(v):
    return np.dot(B_true.T, v)

def apply_W(v):
    return np.dot(mass_matrix, v)

col_volumes, col_means, col_vars = compute_col_moments(apply_Bt, apply_W, qq)

boundary_normal_std = compute_boundary_normal_std(apply_Bt, apply_W, qq, boundary_inds, col_volumes)

near_boundary = mark_points_near_boundary(qq, boundary_inds, boundary_normal_std)

near_boundary2 = mark_points_near_boundary(qq, boundary_inds, sigmas.reshape(-1))

true_boundary_normal_std = sigmas.reshape(-1)[boundary_inds]

err_boundary_width = np.linalg.norm(true_boundary_normal_std - boundary_normal_std) / np.linalg.norm(true_boundary_normal_std)
print('err_boundary_width=', err_boundary_width)

z = np.zeros(n*n)
z[boundary_inds] = (true_boundary_normal_std - boundary_normal_std) / true_boundary_normal_std
plt.matshow(z.reshape((n,n)))

boundary_indicator_function = np.zeros(n*n)
boundary_indicator_function[boundary_inds] = 1.
z = apply_B(boundary_indicator_function)
# z = apply_B(boundary_indicator_function / col_volumes)
plt.matshow(z.reshape((n,n)))

####

qq00 = np.mod(qq + np.array([0.5,0.5]).reshape((2,1)), 2.)
qq01 = np.mod(qq + np.array([-0.5,0.5]).reshape((2,1)), 2.)
qq10 = np.mod(qq + np.array([0.5,-0.5]).reshape((2,1)), 2.)
qq11 = np.mod(qq + np.array([-0.5,-0.5]).reshape((2,1)), 2.)

col_volumes00, col_means00, col_vars00 = compute_col_moments(apply_Bt, apply_W, qq00)
col_volumes01, col_means01, col_vars01 = compute_col_moments(apply_Bt, apply_W, qq01)
col_volumes10, col_means10, col_vars10 = compute_col_moments(apply_Bt, apply_W, qq10)
col_volumes11, col_means11, col_vars11 = compute_col_moments(apply_Bt, apply_W, qq11)

plt.matshow(np.linalg.norm((col_vars00.swapaxes(0,2) - CC_true).reshape((-1,4)), axis=1).reshape((n,n)))
plt.matshow(np.linalg.norm((col_vars01.swapaxes(0,2) - CC_true).reshape((-1,4)), axis=1).reshape((n,n)))
plt.matshow(np.linalg.norm((col_vars10.swapaxes(0,2) - CC_true).reshape((-1,4)), axis=1).reshape((n,n)))
plt.matshow(np.linalg.norm((col_vars11.swapaxes(0,2) - CC_true).reshape((-1,4)), axis=1).reshape((n,n)))

qq00_bool = np.all(np.abs(qq00 -1.) <= 0.55, axis=0)
qq01_bool = np.all(np.abs(qq01 -1.) <= 0.55, axis=0)
qq10_bool = np.all(np.abs(qq10 -1.) <= 0.55, axis=0)
qq11_bool = np.all(np.abs(qq11 -1.) <= 0.55, axis=0)
z = np.zeros(n*n)
z[qq00_bool] = 1.
plt.matshow(z.reshape((n,n)))
z = np.zeros(n*n)
z[qq01_bool] = 1.
plt.matshow(z.reshape((n,n)))
z = np.zeros(n*n)
z[qq10_bool] = 1.
plt.matshow(z.reshape((n,n)))
z = np.zeros(n*n)
z[qq11_bool] = 1.
plt.matshow(z.reshape((n,n)))

col_vars_combined = np.zeros(col_vars00.shape)
col_vars_combined[:,:,qq00_bool] = col_vars00[:,:,qq00_bool]
col_vars_combined[:,:,qq01_bool] = col_vars01[:,:,qq01_bool]
col_vars_combined[:,:,qq10_bool] = col_vars10[:,:,qq10_bool]
col_vars_combined[:,:,qq11_bool] = col_vars11[:,:,qq11_bool]
CC_approx = col_vars_combined.swapaxes(0,2)

plt.matshow(np.linalg.norm((CC_true).reshape((-1,4)), axis=1).reshape((n,n)))
plt.matshow(np.linalg.norm((CC_approx - CC_true).reshape((-1,4)), axis=1).reshape((n,n)))

# col_volumes2, col_means2, col_vars2 = compute_col_moments(apply_Bt, apply_W, qq2)
# plt.matshow(np.linalg.norm((col_vars2.swapaxes(0,2) - CC_true).reshape((-1,4)), axis=1).reshape((n,n)))
#
# qq_center_bool = np.all(np.abs(qq) < 0.5, axis=0)
# col_vars_combined = col_vars2.copy()
# col_vars_combined[:,:,qq_center_bool] = col_vars[:,:,qq_center_bool]
# plt.matshow(np.linalg.norm((col_vars_combined.swapaxes(0,2) - CC_true).reshape((-1,4)), axis=1).reshape((n,n)))

#### Boundary estimation

boundary_nodes_bool = np.any(np.abs(qq + 1) < 1e-5, axis=0)
# boundary_nodes_bool = np.any(np.abs(qq + 1) < 1e-5, axis=0) | np.any(np.abs(qq - 1) < 1e-5, axis=0)
num_boundary_nodes = np.sum(boundary_nodes_bool)
boundary_source = np.zeros(n*n)
boundary_source[boundary_nodes_bool] = 1.

boundary_response = apply_Bt(apply_W(boundary_source))


####

def make_SVIR(M):
    n = int(np.sqrt(M.shape[0]))
    SVIR = np.zeros((n*n, n, n))
    for i in range(n):
        print('i=', i)
        for j in range(n):
            SVIR[:, i, j] = circular_translate(M[:, n*i + j], i, j)
    SVIR = SVIR.reshape((n*n, -1))
    return SVIR

SVIR = make_SVIR(B_true)

U,ss,Vt = np.linalg.svd(SVIR)

U, V = py_ica(SVIR, 1e-7)
SVIR_cross = np.dot(U, V)
err_cross = np.linalg.norm(SVIR - SVIR_cross)/np.linalg.norm(SVIR)
print('err_cross=', err_cross)

Q, _, _ = np.linalg.svd(V.T,full_matrices=False)

plt.matshow(Q[:,2].reshape((n,n)))

##########################

#### DIAGONAL STRIPES OF B_true

k=30
dk = np.concatenate([np.diagonal(B_true, k), np.diagonal(B_true, k - N)])
plt.matshow(dk.reshape((n,n)))


# _,ss,_ = np.linalg.svd(Hgn)

U,ssG,Vt = np.linalg.svd(G)
Vt_tensor = Vt.reshape((-1,G_tensor.shape[1],G_tensor.shape[2],G_tensor.shape[3]))

norm_res = np.linalg.norm(res)
print('norm_res=', norm_res)
CC_next = CC - 2.5e-10*g
B_next = make_blur_operator(approx_kernel_function, CC_next, qq)
y_next = np.dot(B_next, w1)
res_next = y_true - y_next
norm_res_next = np.linalg.norm(res_next)
print('norm_res_next=', norm_res_next)

##

##

Reg0 = make_periodic_2d_laplacian(n)
Reg = Reg0 + 0.1*np.eye(N)

H = Hd + areg*Reg


ii_sample = np.linspace(0,n,ns, dtype=int, endpoint=False)
def get_sample_sub(i,j):
    return n * ii_sample[i] + ii_sample[j]

ii_sample_display = np.linspace(0, n, ns0, dtype=int, endpoint=False)
def get_sample_sub_display(i,j):
    return n * ii_sample_display[i] + ii_sample_display[j]

Z = np.zeros((N, ns0, ns0))
print('making Z')
for i in range(ns0):
    for j in range(ns0):
        Z[:, i, j] = Hd[:, get_sample_sub_display(i, j)]

Zsum = np.sum(np.sum(Z, axis=-1), axis=-1)
plt.matshow(Zsum.reshape((n,n)))
plt.title('matvec on dirac comb')


PHI = np.zeros((N, ns, ns))
print('making PHI')
for i in range(ns):
    for j in range(ns):
        di = ii_sample[i]
        dj = ii_sample[j]
        PHI[:, i, j] = circular_translate(Hd[:, get_sample_sub(i, j)], di, dj)
PHI = PHI.reshape((N, -1))

Q0,_,_ = np.linalg.svd(Reg)
Q = Q0[:, ::-1][:, :n_diag_basis]
# Q0,_,_ = np.linalg.svd(Hd)
# Q = Q0[:, :n_diag_basis]

sub_xx = []
for i in range(ns):
    for j in range(ns):
        sub_xx.append(get_sample_sub(i,j))
sub_xx = np.array(sub_xx)

SVIR2 = np.dot(PHI, np.linalg.lstsq(Q[sub_xx,:].T, Q.T, rcond=None)[0])

SVIR = np.zeros((N,n,n))
Hd3 = Hd.reshape((N,n,n))
print('making SVIR')
for i in range(n):
    for j in range(n):
        SVIR[:, i, j] = circular_translate(Hd3[:, i,j], i, j)
SVIR = SVIR.reshape((N,N))

ERR = np.linalg.norm(SVIR - SVIR2)/np.linalg.norm(SVIR)
print('ERR=', ERR)

W_BASIS_ERR = np.linalg.norm(np.dot(Q, np.dot(Q.T, SVIR.T)) - SVIR.T) / np.linalg.norm(SVIR.T)
print('W_BASIS_ERR=', W_BASIS_ERR)

PHI_orth, _, _ = np.linalg.svd(PHI, full_matrices=False)

PHI_BASIS_ERR = np.linalg.norm(np.dot(PHI_orth, np.dot(PHI_orth.T, SVIR)) - SVIR)/np.linalg.norm(SVIR)
print('PHI_BASIS_ERR=', PHI_BASIS_ERR)

Hd2 = np.zeros((N,n,n))
SVIR3 = SVIR2.reshape((N,n,n))
print('making Hd2')
for i in range(n):
    for j in range(n):
        Hd2[:, i, j] = circular_translate(SVIR3[:, i,j], -i, -j)
Hd2 = Hd2.reshape((N,N))
Hd2 = (Hd2 + Hd2.T)/2.

np.linalg.norm(Hd2 - Hd)/np.linalg.norm(Hd)

_,ssH,_ = np.linalg.svd(Hd)

ee = np.cumsum((ssH**2))/(np.sum(ssH**2))
equivalent_rank = np.min(np.where(ee > (1.-ERR))[0])
print('equivalent_rank=', equivalent_rank)

# plt.figure()
# plt.semilogy(ssH)
# plt.semilogy

H2 = Hd2 + areg*Reg

# A = np.linalg.solve(H2, H)
# _,ssA,_ = np.linalg.svd(A)
# plt.figure()
# plt.semilogy(ssA)

x0 = np.dot(Hd,w1)

y = np.dot(Hd,x0)

x_true = np.linalg.solve(H, y)

maxit = 5

x1 = spla.cg(H, y, M=Reg, maxiter=maxit)[0]
res1 = np.linalg.norm(np.dot(H, x1) - y)/np.linalg.norm(y)
print('res1=', res1)

x2 = spla.cg(H, y, M=H2, maxiter=maxit)[0]
res2 = np.linalg.norm(np.dot(H, x2) - y)/np.linalg.norm(y)
print('res2=', res2)

# plt.matshow(x1.reshape((n,n)))
# plt.matshow(x2.reshape((n,n)))

#

def gauss_exp(A, xi):
    eta = np.linalg.solve(A, xi)
    return np.exp(-(1./2.) * np.sum(eta * xi, axis=0))

def gauss_c0(A):
    d = A.shape[0]
    return np.power(2. * np.pi, -d / 2.)

def gauss_const(A):
    return gauss_c0(A) * np.power(np.linalg.det(A), -1./2.)

def gauss(A, xi):
    return gauss_const(A) * gauss_exp(A, xi)

def dgauss_exp(A,xi, dA):
    eta = np.linalg.solve(A, xi)
    return (1./2.) * np.sum(eta * np.dot(dA, eta), axis=0) * gauss_exp(A, xi)

def dgauss_const(A, dA):
    return -(1. / 2.) * gauss_c0(A) * np.power(np.linalg.det(A), -3./2.) * np.linalg.det(A) * np.trace(np.linalg.solve(A, dA))

def dgauss(A,xi, dA):
    return dgauss_const(A, dA) * gauss_exp(A,xi) + gauss_const(A) * dgauss_exp(A,xi, dA)

def grad_gauss_exp(A,xi):
    eta = np.linalg.solve(A, xi)
    return (1./2.) * np.einsum('n,in,jn->nij', gauss_exp(A, xi), eta, eta)

def grad_gauss_const(A):
    return -(1. / 2.) * gauss_c0(A) * np.power(np.linalg.det(A), -3./2.) * np.linalg.det(A) * np.linalg.inv(A).T

def grad_gauss(A,xi):
    return np.einsum('n,ij->nij', gauss_exp(A, xi), grad_gauss_const(A)) + gauss_const(A) * grad_gauss_exp(A, xi)

d=2
num_xi=25
M = np.random.randn(d,d)
M = np.dot(M, M.T)
xi = np.random.randn(d,num_xi)
dM = np.random.randn(d,d)
dM = np.dot(dM, dM.T)
s = 1e-6

J0 = gauss_exp(M,xi)
J1 = gauss_exp(M + s*dM, xi)
dJ_diff = (J1-J0)/s
dJ = dgauss_exp(M,xi,dM)
err_dgauss_exp = np.linalg.norm(dJ - dJ_diff)
print('s=', s, ', err_dgauss_exp=', err_dgauss_exp)

K0 = gauss_const(M)
K1 = gauss_const(M + s*dM)
dK_diff = (K1-K0)/s
dK = dgauss_const(M,dM)
err_dgauss_const = np.linalg.norm(dK - dK_diff)
print('s=', s, ', err_dgauss_const=', err_dgauss_const)

L0 = gauss(M,xi)
L1 = gauss(M + s*dM, xi)
dL_diff = (L1-L0)/s
dL = dgauss(M,xi,dM)
err_dgauss = np.linalg.norm(dL - dL_diff)
print('s=', s, ', err_dgauss=', err_dgauss)

err_grad_gauss_exp = np.linalg.norm(np.dot(grad_gauss_exp(M,xi).reshape((num_xi,-1)), dM.reshape(-1)) - dJ)
print('err_grad_gauss_exp=', err_grad_gauss_exp)

err_grad_gauss_const = np.linalg.norm(np.dot(grad_gauss_const(M).reshape((-1)), dM.reshape(-1)) - dK)
print('err_grad_gauss_const=', err_grad_gauss_const)

err_grad_gauss = np.linalg.norm(np.dot(grad_gauss(M,xi).reshape((num_xi,-1)), dM.reshape(-1)) - dL)
print('err_grad_gauss=', err_grad_gauss)



