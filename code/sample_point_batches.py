import numpy as np
import dolfin as dl
from numba import jit

from ellipsoid import points_which_are_not_in_ellipsoid_numba, ellipsoids_intersect


def choose_sample_points_batches(num_batches, mu_function, Sigma_function, tau, ):
    pass


@jit(nopython=True)
def choose_one_sample_points_batch(mu, Sigma, num_standard_deviations_tau, candidate_inds, randomize=True):
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



