import numpy as np
import typing as typ
import scipy.sparse as sps
from dataclasses import dataclass
from functools import cached_property
import dolfin as dl
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm.auto import tqdm

from .assertion_helpers import *
from .smoothing_matrix import make_smoothing_matrix
from .impulse_response_moments import impulse_response_moments, impulse_response_moments_simplified
from .sample_point_batches import choose_one_sample_point_batch

from nalger_helper_functions import make_mass_matrix, dlfct2array, plot_ellipse
import hlibpro_python_wrapper as hpro


class ImpulseResponseBatches:
    def __init__(me, V_in, V_out,
                 apply_A, apply_At,
                 solve_M_in_moments,
                 solve_M_in_impulses,
                 solve_M_out_impulses,
                 num_initial_batches=5,
                 tau=3.0,
                 max_candidate_points=None,
                 num_neighbors=8,
                 max_scale_discrepancy=1e5):
        me.V_in = V_in
        me.V_out = V_out
        me.apply_A = apply_A
        me.apply_At = apply_At
        me.solve_M_in_moments = solve_M_in_moments
        me.solve_M_in_impulses = solve_M_in_impulses
        me.solve_M_out_impulses = solve_M_out_impulses
        me.max_candidate_points = max_candidate_points
        me.max_scale_discrepancy = max_scale_discrepancy

        print('Computing impulse response moments')
        me.vol, me.mu, me.Sigma0, me.bad_inds = impulse_response_moments(me.V_in, me.V_out,
                                                                         me.apply_At, me.solve_M_in_moments,
                                                                         max_scale_discrepancy=max_scale_discrepancy)

        me.vol[me.bad_inds] = 0.0

        print('Preparing sample point batch stuff')
        me.dof_coords_in = me.V_in.tabulate_dof_coordinates()
        me.dof_coords_out = me.V_out.tabulate_dof_coordinates()

        N_out, d_out = me.dof_coords_out.shape

        dof_coords_out_kdtree = KDTree(me.dof_coords_out)
        closest_distances, _ = dof_coords_out_kdtree.query(me.dof_coords_out, 2)
        sigma_mins = closest_distances[:,1].reshape((N_out,1)) # shape=(N,1)

        me.vertex2dof_out = dl.vertex_to_dof_map(me.V_out)
        me.dof2vertex_out = dl.dof_to_vertex_map(me.V_out)

        eee0, PP = np.linalg.eigh(me.Sigma0) # eee0.shape=(N,d), PP.shape=(N,d,d)
        eee = np.max([np.ones((1,d_out))*sigma_mins**2, eee0], axis=0)
        me.Sigma = np.einsum('nij,nj,nkj->nik', PP, eee, PP)

        mesh_vertex_vol   = [me.vol[k]       for k in me.vertex2dof_out]
        mesh_vertex_mu    = [me.mu[k,:]      for k in me.vertex2dof_out]
        mesh_vertex_Sigma = [me.Sigma[k,:,:] for k in me.vertex2dof_out]

        print('Preparing c++ object')
        me.mesh_out = me.V_out.mesh()
        me.mesh_vertices = np.array(me.mesh_out.coordinates().T, order='F')
        me.mesh_cells = np.array(me.mesh_out.cells().T, order='F')

        me.cpp_object = hpro.hpro_cpp.ImpulseResponseBatches( me.mesh_vertices,
                                                              me.mesh_cells,
                                                              mesh_vertex_vol,
                                                              mesh_vertex_mu,
                                                              mesh_vertex_Sigma,
                                                              num_neighbors,
                                                              tau )

        me.candidate_inds = np.argwhere(np.logical_not(me.bad_inds)).reshape(-1)

        if me.max_candidate_points is not None:
            me.candidate_inds = np.random.permutation(len(me.candidate_inds))[:max_candidate_points]

        print('Building initial sample point batches')
        for ii in tqdm(range(num_initial_batches)):
            me.add_one_sample_point_batch()

    def add_one_sample_point_batch(me):
        qq = np.array(me.dof_coords_in[me.candidate_inds, :].T, order='F')
        if me.num_sample_points > 0:
            dd = me.cpp_object.kdtree.query(qq,1)[1][0]
            candidate_inds_ordered_by_distance = np.array(me.candidate_inds)[np.argsort(dd)]
        else:
            candidate_inds_ordered_by_distance = np.array(me.candidate_inds)

        if len(candidate_inds_ordered_by_distance) < 1:
            print('no points left to choose')
            return np.array([])

        new_inds = choose_one_sample_point_batch(me.mu, me.Sigma, me.tau,
                                                 candidate_inds_ordered_by_distance, randomize=False)

        new_points = me.dof_coords_in[new_inds, :]
        new_vol = me.vol[new_inds]

        phi = get_one_dirac_comb_response(new_points, me.V_in, me.V_out, me.apply_A,
                                          me.solve_M_in_impulses, me.solve_M_out_impulses,
                                          scale_factors=1./new_vol)

        phi_vertex = phi.vector()[me.vertex2dof_out].copy()

        me.cpp_object.add_batch(me.dof2vertex_out[new_inds],
                                phi_vertex,
                                True)

        me.candidate_inds = list(np.setdiff1d(me.candidate_inds, new_inds))

        return new_inds

    def visualize_impulse_response_batch(me, b):
        if (0 <= b) and (b < me.num_batches):
            phi = dl.Function(me.V_out)
            phi.vector()[me.vertex2dof_out] = me.psi_vertex_batches[b]

            start = me.batch2point_start[b]
            stop = me.batch2point_stop[b]
            pp = me.sample_points[start:stop, :]
            mu_batch = me.sample_mu[start:stop, :]
            Sigma_batch = me.sample_Sigma[start:stop, :, :]

            plt.figure()

            cm = dl.plot(phi)
            plt.colorbar(cm)

            plt.scatter(pp[:, 0], pp[:, 1], c='r', s=2)
            plt.scatter(mu_batch[:, 0], mu_batch[:, 1], c='k', s=2)

            for k in range(mu_batch.shape[0]):
                plot_ellipse(mu_batch[k, :], Sigma_batch[k, :, :], n_std_tau=me.tau,
                             facecolor='none', edgecolor='k', linewidth=1)

            plt.title('Impulse response batch '+str(b))
        else:
            print('bad batch number. num_batches=', me.num_batches, ', b=', b)

    @property
    def mesh_vertex_vol(me):
        return np.array(me.cpp_object.mesh_vertex_vol)

    @property
    def mesh_vertex_mu(me):
        return np.array(me.cpp_object.mesh_vertex_mu)

    @property
    def mesh_vertex_Sigma(me):
        return np.array(me.cpp_object.mesh_vertex_Sigma)

    @property
    def sample_points(me):
        return np.array(me.cpp_object.sample_points)

    @property
    def sample_vol(me):
        return np.array(me.cpp_object.sample_vol)

    @property
    def sample_mu(me):
        return np.array(me.cpp_object.sample_mu)

    @property
    def sample_Sigma(me):
        return np.array(me.cpp_object.sample_Sigma)

    @property
    def point2batch(me):
        return np.array(me.cpp_object.point2batch)

    @property
    def batch2point_start(me):
        return np.array(me.cpp_object.batch2point_start)

    @property
    def batch2point_stop(me):
        return np.array(me.cpp_object.batch2point_stop)

    @property
    def psi_vertex_batches(me):
        return np.array(me.cpp_object.psi_batches)

    @property
    def tau(me):
        return me.cpp_object.tau

    @tau.setter
    def tau(me, new_tau):
        me.cpp_object.tau = new_tau

    @property
    def num_neighbors(me):
        return me.cpp_object.num_neighbors

    @num_neighbors.setter
    def num_neighbors(me, new_num_neighbors):
        me.cpp_object.num_neighbors = new_num_neighbors

    @property
    def num_sample_points(me):
        return me.cpp_object.num_pts()

    @property
    def num_batches(me):
        return me.cpp_object.num_batches()


def get_one_dirac_comb_response(points_pp, V_in, V_out, apply_A, solve_M_in, solve_M_out, scale_factors=None):
    dirac_comb_dual_vector = make_dirac_comb_dual_vector(points_pp, V_in, scale_factors=scale_factors)
    dirac_comb_response = dl.Function(V_out)
    dirac_comb_response.vector()[:] = solve_M_out(apply_A(solve_M_in(dirac_comb_dual_vector)))
    return dirac_comb_response


def make_dirac_comb_dual_vector(pp, V, scale_factors=None):
    num_pts, d = pp.shape
    if scale_factors is None:
        scale_factors = np.ones(num_pts)

    dirac_comb_dual_vector = dl.assemble(dl.Constant(0.0) * dl.TestFunction(V) * dl.dx)
    for k in range(num_pts):
            ps = dl.PointSource(V, dl.Point(pp[k,:]), scale_factors[k])
            ps.apply(dirac_comb_dual_vector)
    return dirac_comb_dual_vector
