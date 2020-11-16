import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh
from brent_minimize import brent_minimize
from numba import njit
from time import time

@njit
def K(x, dd, v):
    return 1. - np.sum(v * ((dd * x * (1. - x)) / (x + dd * (1. - x))) * v)

def do_ellipsoids_intersect(A, B, a, b):
    dd, Phi = eigh(A, B, eigvals_only=False)
    v = np.dot(Phi.T, a - b)
    xmin, fval, iter, funcalls = brent_minimize(K, 0., 1., args=(dd, v))
    return (fval >= 0)



# d=3
#
# Ua, ssa, _ = np.linalg.svd(np.random.randn(d, d))
# Ub, ssb, _ = np.linalg.svd(np.random.randn(d, d))
#
# A = np.dot(Ua, np.dot(np.diag(ssa), Ua.T))
# B = np.dot(Ub, np.dot(np.diag(ssb), Ub.T))
#
# a = np.random.randn(d)
# b = np.random.randn(d)
#
# t = time()
# do_ellipsoids_intersect(A, B, a, b)
# dt = time() - t
# print('dt=', dt)
#
# dd, Phi = eigh(A, B, eigvals_only=False)
# v = np.dot(Phi.T, a - b)
#
# N=100
# AA = np.zeros((d,d,N))
# BB = np.zeros((d,d,N))
# for k in range(N):
#     Ua, ssa, _ = np.linalg.svd(np.random.randn(d, d))
#     Ub, ssb, _ = np.linalg.svd(np.random.randn(d, d))
#
#     AA[:,:,k] = np.dot(Ua, np.dot(np.diag(ssa), Ua.T))
#     BB[:,:,k] = np.dot(Ub, np.dot(np.diag(ssb), Ub.T))
#
# aa = np.random.randn(d,N)
# bb = np.random.randn(d,N)
#
# xx = np.random.rand(N)
# t = time()
# for k in range(N):
#     do_ellipsoids_intersect(AA[:, :, k], BB[:, :, k], aa[:, k], bb[:, k])
#     # K(xx[k],dd,v)
# dt = time() - t
# print('dt=', dt)
