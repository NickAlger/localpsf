import numpy as np
import scipy.linalg as sla
from tqdm.auto import tqdm
import dolfin as dl


def eval_radial_gaussian_kernels_at_point(qq, ss, p):
    dd_squared = np.sum((qq - p.reshape((1, -1)))**2, axis=1).reshape(-1)
    return np.exp(-0.5 * dd_squared / (ss**2))


def invert_dictionary(d):
    return {d[key] : key for key in d.keys()}


def pointcloud_nearest_neighbor_distances(pp):
    N, d = pp.shape
    nearest_neighbor_distances = np.zeros(N)
    for k in range(N):
        p = pp[k,:].reshape((1, d))
        qq = np.array([pp[:k], pp[k+1:]]).reshape((N-1, d))
        nearest_neighbor_distances[k] = np.min(np.linalg.norm(qq - p, axis=1))
    return nearest_neighbor_distances


def point_is_in_ellipsoid(p, mu, Sigma, tau):
    d = mu - p
    return np.dot(d, np.linalg.solve(Sigma, d)) <= tau**2


class PSFInterpolator:
    def __init__(me, impulse_response_batches, sample_points_batches, mu_fenics_Function, Sigma_fenics_Function,
                 ellipsoid_tau, drop_tol=1e-2, gaussian_tau=2.0):
        me.impulse_response_batches = impulse_response_batches
        me.sample_points_batches = sample_points_batches
        me.mu = mu_fenics_Function
        me.Sigma = Sigma_fenics_Function
        me.drop_tol = drop_tol
        me.ellipsoid_tau = ellipsoid_tau
        me.gaussian_tau = gaussian_tau

        me.V = me.impulse_response_batches[0].function_space()
        me.dof_coords = me.V.tabulate_dof_coordinates()
        me.num_dofs = me.V.dim()

        me.bbt = me.V.mesh().bounding_box_tree()
        me.bad_bbt_entity = me.bbt.compute_first_entity_collision(dl.Point(*(np.inf for _ in range(me.d))))

        me.num_batches = len(me.sample_points_batches)
        me.sample_points = np.array(me.sample_points_batches) # stack point arrays on top of each other
        me.num_sample_pts, me.d = me.sample_points.shape


        print('forming global_to_local_ind_map')
        # me.global_to_local_ind_map used as follows:
        # b, k = me.global_to_local_ind_map[s]
        #   => me.sample_points[s,:] = me.sample_points_batches[b][k,:]
        me.global_to_local_ind_map = dict()
        s = 0
        for b in tqdm(range(me.num_batches)):
            batch_size = me.sample_points_batches[b].shape[0]
            for k in range(batch_size):
                me.global_to_local_ind_map[s] = (b,k)
                s = s + 1

        me.local_to_global_ind_map = invert_dictionary(me.global_to_local_ind_map)


        print('computing gaussian_kernel_sigmas')
        me.gaussian_kernel_sigmas = me.gaussian_tau * pointcloud_nearest_neighbor_distances(me.sample_points)


        print('Constructing full_interpolation_matrix')
        me.full_interpolation_matrix = np.zeros((me.num_sample_pts, me.num_sample_pts))
        for s in tqdm(range(me.num_sample_pts)):
            q = me.sample_points[s,:]
            me.full_interpolation_matrix[:,s] = eval_radial_gaussian_kernels_at_point(me.sample_points,
                                                                                      me.gaussian_kernel_sigmas, q)


        print('Tabulating relevant_sample_point_inds')
        # For each degree of freedom in the function space, which sample points are relevant to that point?
        me.relevant_sample_point_inds = list()
        for ii in tqdm(range(me.num_dofs)):
            q = me.dof_coords[ii,:]
            gg = eval_radial_gaussian_kernels_at_point(me.sample_points, me.gaussian_kernel_sigmas, q)
            me.relevant_sample_point_inds.append(np.argwhere(gg > me.drop_tol))


        print('Forming mu_at_sample_points')
        me.mu_at_sample_points = np.zeros((me.num_sample_pts, me.d))
        for s in tqdm(range(me.num_sample_pts)):
            p = me.sample_points[s,:]
            me.mu_at_sample_points[s,:] = me.mu(dl.Point(*tuple(p)))


        print('Forming Sigma_at_sample_points')
        me.Sigma_at_sample_points = np.zeros((me.num_sample_pts, me.d, me.d))
        for s in tqdm(range(me.num_sample_pts)):
            p = me.sample_points[s,:]
            me.Sigma_at_sample_points[s,:,:] = me.Sigma(dl.Point(*tuple(p))).reshape((me.d, me.d))


    def eval_convolution_kernel(me, dof_ind_ii, shift_z):
        ss_good = list()
        for s in me.relevant_sample_point_inds[dof_ind_ii]:
            p = me.sample_points[s,:]
            mu = me.mu_at_sample_points[s,:]
            Sigma = me.Sigma_at_sample_points[s,:,:]
            y = p + shift_z
            if point_is_in_ellipsoid(y, mu, Sigma, me.ellipsoid_tau):
                if me.point_is_in_mesh(y):
                    ss_good.append(s)


    def point_is_in_mesh(me, z_numpy):
        z_fenics = dl.Point(*tuple(z_numpy))
        if me.bbt.compute_first_entity_collision(z_fenics) == me.bad_bbt_entity:
            return False
        else:
            return True