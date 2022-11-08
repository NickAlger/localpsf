import numpy as np
import dolfin as dl
from numba import jit
from tqdm.auto import tqdm

from nalger_helper_functions import *
from .ellipsoid import points_which_are_not_in_ellipsoid_numba, ellipsoids_intersect


def choose_sample_point_batches(num_batches, V, mu_function, Sigma_function, tau, max_candidate_points=None):
    '''Chooses several batches of sample points. Uses a greedy algorithm to choose as many sample points as possible per patch,
     under the constraint that that the supports of impulse responses for a given batch do not overlap.

    Parameters
    ----------
    num_batches: nonnegative int. Number of sample point batches
    mu_function: fenics Function. Vector-valued. spatially varying mean function
    Sigma_function: fenics Function. Matrix-valued. spatially varying covariance function
    tau: positive float. Number of standard deviations for ellipsoid.
        ellipsoid = {x : (x-mu)^T Sigma^-1 (x-mu) <= tau^2}
    max_candidate_points: (optional) positive int. Maximum number of candidate points to pick sample points points from.

    Returns
    -------
    point_batches: list of numpy arrays. Batches of sample points
        point_batches[b].shape=(num_sample_points_in_batch_b, spatial_dimension).
    mu_batches: list of numpy arrays. Impulse response means at each sample point
        mu_batches[b].shape=(num_sample_points_in_batch_b, spatial_dimension).
    Sigma_batches: list of numpy arrays. Impulse response covariance matrices at each sample point
        Sigma_batches[b].shape=(num_sample_points, spatial_dimension, spatial_dimension).

    '''
    # V = mu_function.function_space()

    # dof_coords = np.unique(V.tabulate_dof_coordinates(), axis=0)
    dof_coords = V.tabulate_dof_coordinates()
    mu_array = dlfct2array(mu_function)
    Sigma_array = dlfct2array(Sigma_function)

    if max_candidate_points is None:
        candidate_inds = np.arange(dof_coords.shape[0])
    else:
        candidate_inds = np.random.permutation(V.dim())[:max_candidate_points]

    point_batches = list()
    mu_batches = list()
    Sigma_batches = list()
    print('Choosing sample point batches')
    for b in tqdm(range(num_batches)):
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

    return point_batches, mu_batches, Sigma_batches


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
            if ellipsoids_intersect(Sigma_perm[x_ind, :, :], Sigma_perm[p_ind, :, :],
                                    mu_perm[x_ind, :], mu_perm[p_ind, :],
                                    num_standard_deviations_tau):
                p_is_acceptable = False
                break
        if p_is_acceptable:
            X_inds_perm.append(p_ind)
            P_inds_perm_far_enough = points_which_are_not_in_ellipsoid_numba(Sigma_perm[p_ind, :, :], mu_perm[p_ind, :],
                                                                             mu_perm[np.array(P_inds_perm), :], num_standard_deviations_tau)
            P_inds_perm = list(np.array(P_inds_perm)[P_inds_perm_far_enough])

    X_inds = list(candidate_inds[perm_inds[np.array(X_inds_perm)]])
    return X_inds



