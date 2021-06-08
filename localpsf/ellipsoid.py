import numpy as np
from numba import jit

from .brent_minimize import brent_minimize


@jit(nopython=True)
def points_which_are_not_in_ellipsoid_numba(Sigma, mu, pp, tau):
    d = Sigma.shape[-1]
    zz = (pp.reshape((-1,d)) - mu.reshape((1,d))).T
    return (np.sum(zz * np.linalg.solve(Sigma, zz), axis=0) > tau**2)


@jit(nopython=True)
def ellipsoids_intersect(Sigma_a, Sigma_b, mu_a, mu_b, tau):
    lambdas, Phi, v_squared = ellipsoids_intersect_helper(Sigma_a, Sigma_b, mu_a, mu_b)
    xmin, fval, iter, funcalls = brent_minimize(K_fct, 0., 1., args=(lambdas, v_squared, tau))
    return np.bool_(fval >= 0)


@jit(nopython=True)
def ellipsoids_intersect_helper(Sigma_a, Sigma_b, mu_a, mu_b):
    lambdas, Phi = geigh_numpy(Sigma_a, Sigma_b)
    v_squared = np.dot(Phi.T, mu_a - mu_b) ** 2
    return lambdas, Phi, v_squared


@jit(nopython=True)
def K_fct(s, lambdas, v_squared, tau):
    return 1. - (1. / tau**2) * np.sum(v_squared * ((s * (1. - s)) / (1. + s * (lambdas - 1.))))


@jit(nopython=True)
def K_fct_vectorized(ss, lambdas, v_squared, tau):
    KK = np.empty(len(ss))
    for ii in range(len(ss)):
        KK[ii] = K_fct(ss[ii], lambdas, v_squared, tau)
    return KK


@jit(nopython=True)
def geigh_numpy(A, B):
    # Generalized eigenvalue decomposition A*Phi = B*Phi*Lambda
    # Phi.T * B * Phi = I
    lambdas0, Phi0 = np.linalg.eig(np.linalg.solve(B, A))
    lambdas = lambdas0.real
    Rayleigh_matrix = np.dot(np.dot(Phi0.real.T, B), Phi0.real)
    scaling_factors = np.sqrt(np.diag(Rayleigh_matrix))
    Phi = Phi0.real / scaling_factors.reshape((1,-1))
    return lambdas, Phi