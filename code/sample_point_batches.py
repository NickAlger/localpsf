import numpy as np
import dolfin as dl
from numba import jit

from ellipsoid import points_which_are_not_in_ellipsoid_numba, ellipsoids_intersect
from nalger_helper_functions import *


def choose_sample_point_batches(num_batches, mu_function, Sigma_function, tau, max_candidate_points=None):
    V = mu_function.function_space()

    dof_coords = V.tabulate_dof_coordinates()
    mu_array = dlfct2array(mu_function)
    Sigma_array = dlfct2array(Sigma_function)

    if max_candidate_points is None:
        candidate_inds = np.arange(V.dim())
    else:
        candidate_inds = np.random.permutation(V.dim())[:max_candidate_points]

    point_batches = list()
    mu_batches = list()
    Sigma_batches = list()
    for b in range(num_batches):
        qq = dof_coords[candidate_inds, :]
        dd = np.inf * np.ones(len(candidate_inds))
        for pp in point_batches:
            for k in range(pp.shape[0]):
                pk = pp[k, :].reshape((1, -1))
                ddk = np.linalg.norm(pk - qq, axis=1)
                dd = np.min([dd, ddk], axis=0)
        candidate_inds_ordered_by_distance = np.array(candidate_inds)[np.argsort(dd)]

        new_inds = choose_one_sample_point_batch(mu_array, Sigma_array, tau,
                                                  candidate_inds_ordered_by_distance, randomize=False)

        new_points = dof_coords[new_inds, :]
        new_mu = mu_array[new_inds, :]
        new_Sigma = Sigma_array[new_inds, :]

        point_batches.append(new_points)
        mu_batches.append(new_mu)
        Sigma_batches.append(new_Sigma)

        candidate_inds = list(np.setdiff1d(candidate_inds, new_inds))

    sample_points = np.vstack(point_batches)
    sample_mu = np.vstack(mu_batches)
    sample_Sigma = np.vstack(Sigma_batches)
    batch_lengths = [pp.shape[0] for pp in point_batches]

    return sample_points, sample_mu, sample_Sigma, batch_lengths


@jit(nopython=True)
def choose_one_sample_point_batch(mu, Sigma, num_standard_deviations_tau, candidate_inds, randomize=True):
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



