import numpy as np
from brent_minimize import brent_minimize
from numba import jit


# @jit(nopython=True)
def extract_columns_from_many_vectors(uu, S):
    U_S = np.vstack([u[np.array(S)] for u in uu])
    return U_S


# @jit(nopython=True)
def get_chi_b_inds(Sigma, mu, tau, zeta, Sb, S):
    chi_b_ii = list()
    chi_b_kk = list()
    for j in Sb:
        # i = i_of_j[j]
        i = np.where(np.array(S) == j)[0][0]
        kk_good_mask = points_which_are_not_in_ellipsoid(Sigma[j, :, :], mu[j, :], zeta[i, :, :], tau)
        kk_good = np.where(kk_good_mask)[0]
        chi_b_ii = chi_b_ii + [i for _ in range(len(kk_good))]
        chi_b_kk = chi_b_kk + list(kk_good)
    return chi_b_ii, chi_b_kk


def point_minkowski_sum(pp, qq):
    d = pp.shape[1]
    N = pp.shape[0]
    M = qq.shape[0]
    pp_plus_qq = np.zeros((N, M, d))
    point_minkowski_sum_helper(pp_plus_qq, pp, qq)
    return pp_plus_qq

@jit(nopython=True)
def point_minkowski_sum_helper(pp_plus_qq, pp, qq):
    d = pp.shape[1]
    N = pp.shape[0]
    M = qq.shape[0]
    for ii in range(N):
        for kk in range(M):
            pp_plus_qq[ii,kk] = pp[ii,:] + qq[kk,:]
    return pp_plus_qq


@jit(nopython=True)
def choose_sample_points_batch(mu, Sigma, num_standard_deviations_tau, candidate_inds, randomize=True):
    N = len(candidate_inds)
    mu_candidates = mu[candidate_inds, :]
    Sigma_candidates = Sigma[candidate_inds, :, :]

    if randomize:
        perm_inds = np.random.permutation(N)
    else:
        perm_inds = np.arange(N)
    mu_perm = mu_candidates[perm_inds, :]
    Sigma_perm = Sigma_candidates[perm_inds, :, :]

    P_inds_perm = list(range(N))
    X_inds_perm = list()
    while P_inds_perm:
    # for _ in range(1):
        p_ind = P_inds_perm.pop()
        p_is_acceptable = True
        for x_ind in X_inds_perm:
            if ellipsoids_intersect(Sigma_perm[x_ind, :, :], Sigma_perm[p_ind, :, :], mu_perm[x_ind, :],
                                    mu_perm[p_ind, :], num_standard_deviations_tau):
                p_is_acceptable = False
                break
        if p_is_acceptable:
            X_inds_perm.append(p_ind)
            P_inds_perm_far_enough = points_which_are_not_in_ellipsoid_numba(Sigma_perm[p_ind, :, :], mu_perm[p_ind, :],
                                                                             mu_perm[np.array(P_inds_perm), :], num_standard_deviations_tau)
            P_inds_perm = list(np.array(P_inds_perm)[P_inds_perm_far_enough])

    X_inds = list(candidate_inds[perm_inds[np.array(X_inds_perm)]])
    return X_inds


def points_which_are_not_in_ellipsoid(Sigma, mu, pp, tau):
    d = Sigma.shape[-1]
    zz = (pp.reshape((-1,d)) - mu.reshape((1,d))).T
    return (np.sum(zz * np.linalg.solve(Sigma, zz), axis=0) > tau**2)


@jit(nopython=True)
def points_which_are_not_in_ellipsoid_numba(Sigma, mu, pp, tau):
    d = Sigma.shape[-1]
    zz = (pp.reshape((-1,d)) - mu.reshape((1,d))).T
    return (np.sum(zz * np.linalg.solve(Sigma, zz), axis=0) > tau**2)

# @jit(nopython=True)
def point_is_in_ellipsoid(Sigma, mu, p, tau):
    return np.sum(points_which_are_not_in_ellipsoid(Sigma, mu, p.reshape((1,-1)), tau)) == 0

@jit(nopython=True)
def K(s, lambdas, v_squared, tau):
    return 1. - (1. / tau**2) * np.sum(v_squared * ((s * (1. - s)) / (1. + s * (lambdas - 1.))))


@jit(nopython=True)
def ellipsoids_intersect(Sigma_a, Sigma_b, mu_a, mu_b, tau):
    lambdas, Phi = geigh_numpy(Sigma_a, Sigma_b)
    v_squared = np.dot(Phi.T, mu_a - mu_b)**2
    xmin, fval, iter, funcalls = brent_minimize(K, 0., 1., args=(lambdas, v_squared, tau))
    return np.bool_(fval >= 0)


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


@jit(nopython=True)
def slicer(X):
    return X[np.array([0,2]),:]

# X = np.random.randn(3,4)
# y = slicer(X)

@jit(nopython=True)
def reshaper(X):
    return X.reshape(-1)

# X = np.random.randn(3,4,5)
# z = reshaper(X[1,:,:])