import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from nalger_helper_functions import *


class LocalPSFGrid:
    def __init__(me, ww_weighting_functions_unstructured_mesh,
                 ff_impulse_responses_unstructured_mesh, pp_sample_point_batches,
                 mu_batches, Sigma_batches, tau,
                 grid_density_multiplier=1.0, run_checks=True,
                 max_postprocessing_neighbors=10, num_plots=10):
        me.ww_fenics = ww_weighting_functions_unstructured_mesh
        me.ff_fenics = ff_impulse_responses_unstructured_mesh
        me.pp_batches = pp_sample_point_batches
        me.mu_batches = mu_batches
        me.Sigma_batches = Sigma_batches
        me.tau = tau
        me.grid_density_multiplier = 1.0
        me.max_postprocessing_neighbors = max_postprocessing_neighbors
        me.num_plots = num_plots

        me.V = me.ww_fenics[0].function_space()
        me.dof_coords = me.V.tabulate_dof_coordinates()
        me.d = me.V.mesh().geometric_dimension()

        me.pp = np.vstack(me.pp_batches)
        me.all_mu = np.vstack(me.mu_batches)
        me.all_Sigma = np.vstack(me.Sigma_batches)
        me.batch_lengths = [point_batch.shape[0] for point_batch in me.pp_batches]
        me.num_batches = len(me.batch_lengths)
        me.num_pts = me.pp.shape[0]

        me.pp_cKDTree = cKDTree(me.pp)

        print('getting initial boxes')
        me.ww_min0, me.ww_max0 = get_initial_weighting_function_boxes(me.ww_fenics)
        me.ff_min0, me.ff_max0 = get_initial_impulse_response_boxes(me.all_mu, me.all_Sigma, me.tau)

        print('making conforming boxgrids')
        me.ww_min, me.ww_max, me.ww_grid_shapes, \
        me.ff_min, me.ff_max, me.ff_grid_shapes = \
            make_conforming_boxgrids(me.ww_min0, me.ww_max0, me.ff_min0, me.ff_max0,
                                     me.pp, me.dof_coords, grid_density_multiplier)

        me.hh = (me.ww_max - me.ww_min) / (me.ww_grid_shapes - 1.)

        print('making meshgrids') # not sure about whether to build these
        me.ff_meshgrids = [make_regular_grid(me.ff_min[ii, :], me.ff_max[ii, :], me.ff_grid_shapes[ii, :])[1]
                           for ii in range(me.num_pts)]
        me.ww_meshgrids = [make_regular_grid(me.ww_min[ii, :], me.ww_max[ii, :], me.ww_grid_shapes[ii, :])[1]
                           for ii in range(me.num_pts)]

        print('making weighting function grid transfer operators:')
        me.ww_G2F = make_grid_to_function_transfer_operators(me.V, me.ww_min, me.ww_max)
        me.ww_F2G = make_function_to_grid_transfer_operators(me.V, me.ww_min, me.ww_max, me.ww_grid_shapes)

        print('making impulse response grid transfer operators:')
        me.ff_G2F = make_grid_to_function_transfer_operators(me.V, me.ff_min, me.ff_max)
        me.ff_F2G = make_function_to_grid_transfer_operators(me.V, me.ff_min, me.ff_max,
                                                             me.ff_grid_shapes,
                                                             me.all_mu, me.all_Sigma, me.tau)

        print('computing weighting function grid functions')
        me.ww_grid = list()
        for ii in tqdm(range(me.num_pts)):
            F2G = me.ww_F2G[ii]
            w_grid = F2G(me.ww_fenics[ii], outside_domain_fill_value=0.0, use_extrapolation=True)
            me.ww_grid.append(w_grid)

        print('computing initial impulse response grid functions')
        me.ff_grid = list()
        for ii in tqdm(range(me.num_pts)):
            b, k = ind2sub_batches(ii, me.batch_lengths)
            f = me.ff_fenics[b]
            F2G = me.ff_F2G[ii]
            me.ff_grid.append(F2G(f, outside_domain_fill_value=np.nan))

        print('postprocessing impulse responses to fill in boundary nans')
        ff_grid_new = list()
        ff_min_new = list()
        ff_max_new = list()
        ff_grid_shapes_new = list()
        for _ in range(2):
            for ii in tqdm(range(len(me.ff_grid))):
                if np.any(np.isnan(me.ff_grid[ii])):
                    new_f, new_f_min, new_f_max, new_f_grid_shape = \
                        me.compute_impulse_response_neighbor_extension(ii,make_plots=False)
                    ff_grid_new.append(new_f)
                    ff_min_new.append(new_f_min)
                    ff_max_new.append(new_f_max)
                    ff_grid_shapes_new.append(new_f_grid_shape)
            me.ff_grid = ff_grid_new
            me.ff_min = ff_min_new
            me.ff_max = ff_max_new
            me.ff_grid_shapes = ff_grid_shapes_new

        # me.ff_grid = list()
        # for ii in tqdm(range(me.num_pts)):
        #     f_grid = postprocess_impulse_response(ii, me.pp, me.ff_min, me.ff_max, me.ff_grid1,
        #                                           max_neighbors=max_postprocessing_neighbors)
        #     me.ff_grid.append(f_grid)

        print('making convolution grid and transfer operators')
        me.cc_min = me.ww_min + me.ff_min - me.pp
        me.cc_max = me.ww_max + me.ff_max - me.pp
        me.cc_grid_shapes = np.round(((me.cc_max - me.cc_min) / me.hh) + 1.).astype(int)

        me.cc_G2F = make_grid_to_function_transfer_operators(me.V, me.cc_min, me.cc_max)
        me.cc_F2G = make_function_to_grid_transfer_operators(me.V, me.cc_min, me.cc_max, me.cc_grid_shapes)

        print('making adjoint convolution grids and functions')
        me.ffstar_min = me.pp - me.ff_min
        me.ffstar_max = me.pp - me.ff_max
        me.ffstar_grid_shapes = me.ff_grid_shapes
        me.ffstar_grid = list()
        for ii in range(me.num_pts):
            F = me.ff_grid[ii]
            flipped_F = F[[slice(None, None, -1) for _ in F.shape]]
            me.ffstar_grid.append(flipped_F)

        if run_checks:
            print('running checks')
            me.check_conforming_error()
            me.make_several_plots(me.num_plots)

    def eval_wi(me, ii, eval_points, fill_value=0.0):
        return grid_interpolate(me.ww_min[ii,:], me.ww_max[ii,:], me.ww_grid[ii], eval_points, fill_value=fill_value)

    def eval_fi(me, ii, eval_points, fill_value=0.0):
        return grid_interpolate(me.ff_min[ii,:], me.ff_max[ii,:], me.ff_grid[ii], eval_points, fill_value=fill_value)

    def matvec(me, u_fenics): # product-convolution on each patch
        Au_fenics = dl.Function(me.V)
        ui_fenics = dl.Function(me.V)
        for ii in range(me.num_pts):
            ui_fenics.vector()[:] = u_fenics.vector() * me.ww_fenics[ii].vector()
            Ui = me.ww_F2G[ii](ui_fenics)
            Fi = me.ff_grid[ii]

            AUi, min_i, max_i = conforming_grid_convolution(Ui, me.ww_min[ii,:], me.ww_max[ii,:],
                                                            Fi, me.ff_min[ii,:], me.ff_max[ii,:], p2=me.pp[ii,:])

            if ((np.linalg.norm(min_i - me.cc_min[ii]) > 1e-10)
                    or (np.linalg.norm(max_i - me.cc_max[ii]) > 1e-10)):
                print('min_i=', min_i, ', me.cc_min[ii]=', me.cc_min[ii])
                print('max_i=', max_i, ', me.cc_max[ii]=', me.cc_max[ii])
                raise RuntimeError('convolution grid consistency problem')

            Aui = me.cc_G2F[ii](AUi)
            Au_fenics.vector()[:] = Au_fenics.vector() + Aui.vector()

        return Au_fenics

    def get_neighbors(me, ii, num_neighbors0=10, make_plots=True):
        dd, neighbor_inds0 = me.pp_cKDTree.query(me.pp[ii, :], num_neighbors0)
        dd = dd.reshape(-1)
        neighbor_inds0 = neighbor_inds0.reshape(-1)
        nearest_neighbor_distance = dd[1]

        W0_nbrs = np.zeros(tuple([num_neighbors0]) + tuple(me.ww_grid_shapes[ii]))
        for k in range(len(neighbor_inds0)):
            W0_nbrs[k, :] = me.ww_F2G[ii](me.ww_fenics[neighbor_inds0[k]], outside_domain_fill_value=np.nan)

        coords_ww0 = np.vstack([X.reshape(-1) for X in me.ww_meshgrids[ii]]).T

        in_annulus = np.logical_and(np.linalg.norm(coords_ww0 - me.pp[ii, :], axis=1) <= nearest_neighbor_distance,
                                    np.linalg.norm(coords_ww0 - me.pp[ii, :], axis=1) >= nearest_neighbor_distance / 2.)
        in_domain = np.logical_not(np.isnan(W0_nbrs[0, :])).reshape(-1)
        annulus_mask = np.logical_and(in_annulus, in_domain).reshape(me.ww_grid_shapes[ii])

        relevant_nbrs = np.unique((np.argmax(W0_nbrs[1:], axis=0) + 1)[annulus_mask].reshape(-1))
        neighbor_inds = np.concatenate([[ii], neighbor_inds0[relevant_nbrs]])

        if make_plots:
            plt.figure()
            plt.pcolor(me.ww_meshgrids[ii][0], me.ww_meshgrids[ii][1],
                       (in_domain * 0.5).reshape(me.ww_grid_shapes[ii]).astype(float))
            plt.pcolor(me.ww_meshgrids[ii][0], me.ww_meshgrids[ii][1], (annulus_mask).astype(float))

            print('neighbor_inds=', neighbor_inds)
            me.plot_w_box(ii, which='final')
            plt.plot(me.pp[neighbor_inds, 0], me.pp[neighbor_inds, 1], '.r')

        return neighbor_inds, nearest_neighbor_distance

    def compute_impulse_response_neighbor_extension(me, ii, num_neighbors0=10, make_plots=True):

        neighbor_inds, nearest_neighbor_distance = me.get_neighbors(ii, num_neighbors0=num_neighbors0, make_plots=make_plots)

        num_neighbors = len(neighbor_inds)

        # Get new bigger box for f_i that will fit shifted neighbor f_j's
        new_f_min0 = np.min(me.ff_min[neighbor_inds, :] - me.pp[neighbor_inds, :] + me.pp[ii, :], axis=0)
        new_f_max0 = np.max(me.ff_max[neighbor_inds, :] - me.pp[neighbor_inds, :] + me.pp[ii, :], axis=0)
        new_f_min, new_f_max, new_f_grid_shape = conforming_box(new_f_min0, new_f_max0, me.pp[ii, :], me.hh[ii, :])

        new_ff_nbrs = np.zeros(tuple([num_neighbors]) + tuple(new_f_grid_shape))
        for k in range(num_neighbors):
            jj = neighbor_inds[k]
            F2G = Function2Grid(me.V,
                                new_f_min - me.pp[ii,:] + me.pp[jj,:],
                                new_f_max - me.pp[ii,:] + me.pp[jj,:],
                                new_f_grid_shape,
                                mu=me.all_mu[jj, :], Sigma=me.all_Sigma[jj, :, :], tau=me.tau)
            b, _ = ind2sub_batches(jj, me.batch_lengths)
            new_ff_nbrs[k,:] = F2G(me.ff_fenics[b], outside_domain_fill_value=np.nan)

        _, meshgrids_new = make_regular_grid(new_f_min, new_f_max, new_f_grid_shape)
        coords_ff_new = np.vstack([X.reshape(-1) for X in meshgrids_new]).T

        if make_plots:
            for k in range(num_neighbors):
                plt.figure(figsize=(12,8))
                plt.subplot(121)
                plt.pcolor(meshgrids_new[0], meshgrids_new[1], new_ff_nbrs[k,:])
                plt.colorbar()
                plt.plot(me.pp[neighbor_inds, 0], me.pp[neighbor_inds, 1], '.k')
                plt.plot(me.pp[neighbor_inds[k], 0], me.pp[neighbor_inds[k], 1], '.r')

                jj = neighbor_inds[k]
                plt.subplot(122)
                plt.pcolor(me.ff_meshgrids[jj][0], me.ff_meshgrids[jj][1], me.ff_grid1[jj])
                plt.colorbar()
                plt.plot(me.pp[neighbor_inds, 0], me.pp[neighbor_inds, 1], '.k')
                plt.plot(me.pp[jj, 0], me.pp[jj, 1], '.r')

        nan_mask = np.isnan(new_ff_nbrs[0,:]).reshape(-1)
        valid_mask = np.logical_not(nan_mask)

        nan_inds = np.argwhere(nan_mask).reshape(-1)
        valid_inds = np.argwhere(valid_mask).reshape(-1)

        nan_coords = coords_ff_new[nan_inds, :]
        valid_coords = coords_ff_new[valid_inds, :]

        T = cKDTree(valid_coords)
        _, closest_valid_inds = T.query(nan_coords, k=1)

        closest_valid_coords = coords_ff_new[valid_inds[closest_valid_inds], :]

        nan_to_domain_vectors = closest_valid_coords - nan_coords
        unit_nan_to_domain_vectors = nan_to_domain_vectors / np.linalg.norm(nan_to_domain_vectors, axis=1).reshape((-1, 1))

        gauss_vectors = me.pp[ii, :].reshape((1, -1)) + nearest_neighbor_distance * unit_nan_to_domain_vectors

        if make_plots:
            plt.figure()
            plt.pcolor(meshgrids_new[0], meshgrids_new[1], new_ff_nbrs[0, :])
            plt.plot(me.pp[neighbor_inds, 0], me.pp[neighbor_inds, 1], '.r')

            for jj in range(nan_coords.shape[0]):
                plt.plot(nan_coords[jj, 0], nan_coords[jj, 1], '.k')
                plt.plot(np.array([nan_coords[jj, 0], closest_valid_coords[jj, 0]]),
                         np.array([nan_coords[jj, 1], closest_valid_coords[jj, 1]]))

            # plt.figure()
            plt.plot(gauss_vectors[:, 0], gauss_vectors[:, 1], 'g.')
            plt.plot(me.pp[ii, 0], me.pp[ii, 1], '.r')


        W0_gauss = np.zeros((num_neighbors, np.prod(new_f_grid_shape)))
        for k in range(num_neighbors):
            W0_gauss[k, nan_inds] = me.eval_wi(neighbor_inds[k], gauss_vectors)
        W0_gauss = W0_gauss.reshape(tuple([num_neighbors]) + tuple(new_f_grid_shape))

        if make_plots:
            for k in range(num_neighbors):
                plt.figure()
                plt.pcolor(meshgrids_new[0], meshgrids_new[1], W0_gauss[k, :])
                plt.title('W_gauss[' + str(k) + ',:]')

                plt.plot(me.pp[neighbor_inds, 0], me.pp[neighbor_inds, 1], '.k')
                ii_nbr = neighbor_inds[k]
                plt.plot(me.pp[ii_nbr, 0], me.pp[ii_nbr, 1], '.r')
                plt.colorbar()

        W1_gauss = W0_gauss.reshape((num_neighbors, -1))
        W1_gauss[np.isnan(W1_gauss)] = 0.
        W1_gauss[0, valid_inds] = 1.
        W1_gauss = W1_gauss.reshape(tuple([num_neighbors]) + tuple(new_f_grid_shape))

        W1_gauss = np.logical_not(np.isnan(new_ff_nbrs)) * W1_gauss * (W1_gauss > 0)

        W2_gauss = W1_gauss / np.sum(W1_gauss, axis=0).reshape(tuple([1]) + W1_gauss.shape[1:])
        W2_gauss[np.isnan(W2_gauss)] = 0.

        if make_plots:
            for k in range(num_neighbors):
                plt.figure()
                plt.pcolor(meshgrids_new[0], meshgrids_new[1], W2_gauss[k,:])
                plt.colorbar()
                plt.title('W2_gauss['+str(k)+',:]')

            plt.figure()
            plt.pcolor(meshgrids_new[0], meshgrids_new[1], np.sum(W2_gauss, axis=0))
            plt.colorbar()
            plt.title('np.sum(W2_gauss, axis=0)')

        still_nan_mask = np.all(np.isnan(new_ff_nbrs), axis=0)

        new_ff_nbrs2 = new_ff_nbrs
        new_ff_nbrs2[np.isnan(new_ff_nbrs2)] = 0.

        new_f = np.sum(W2_gauss * new_ff_nbrs2, axis=0)
        new_f[still_nan_mask] = np.nan

        if make_plots:
            plt.figure()
            plt.pcolor(meshgrids_new[0], meshgrids_new[1], new_f)
            plt.colorbar()
            plt.title('new_f')

        return new_f, new_f_min, new_f_max, new_f_grid_shape


















    # def rmatvec(me, u_fenics):
    #     U = me.cc_F2G(u_fenics)

    def make_several_plots(me, num_plots):
        for ii in np.random.permutation(me.num_pts)[:num_plots]:
            me.plot_w_box(ii)
            me.plot_f_box(ii)
            me.make_w_transfer_plot(ii)
            me.make_f_boundary_extension_plot(ii)

    def check_conforming_error(me):
        hh_w = (me.ww_max - me.ww_min) / (me.ww_grid_shapes - 1)
        hh_f = (me.ff_max - me.ff_min) / (me.ff_grid_shapes - 1)
        hh_c = (me.cc_max - me.cc_min) / (me.cc_grid_shapes - 1)

        conforming_error_0a = np.linalg.norm(hh_w - hh_f)
        print('conforming_error_0a=', conforming_error_0a)

        conforming_error_0b = np.linalg.norm(hh_w - hh_c)
        print('conforming_error_0b=', conforming_error_0b)

        w_spacing1 = (me.ww_max - me.pp) / hh_w
        w_spacing2 = (me.ww_min - me.pp) / hh_w

        conforming_error_1 = np.linalg.norm(np.round(w_spacing1) - w_spacing1)
        conforming_error_2 = np.linalg.norm(np.round(w_spacing2) - w_spacing2)

        print('conforming_error_1=', conforming_error_1)
        print('conforming_error_2=', conforming_error_2)

        ff_spacing1 = (me.ff_max - me.pp) / hh_f
        ff_spacing2 = (me.ff_min - me.pp) / hh_f

        conforming_error_3 = np.linalg.norm(np.round(ff_spacing1) - ff_spacing1)
        conforming_error_4 = np.linalg.norm(np.round(ff_spacing2) - ff_spacing2)

        print('conforming_error_3=', conforming_error_3)
        print('conforming_error_4=', conforming_error_4)

        cc_spacing1 = (me.cc_max - me.pp) / hh_c
        cc_spacing2 = (me.cc_min - me.pp) / hh_c

        conforming_error_5 = np.linalg.norm(np.round(cc_spacing1) - cc_spacing1)
        conforming_error_6 = np.linalg.norm(np.round(cc_spacing2) - cc_spacing2)

        print('conforming_error_5=', conforming_error_5)
        print('conforming_error_6=', conforming_error_6)

    def plot_w_box(me, ii, which='both'):
        plt.figure()
        cm = dl.plot(me.ww_fenics[ii])
        plt.colorbar(cm)

        if which == 'initial':
            plot_rectangle(me.ww_min0[ii, :], me.ww_max0[ii, :])
        elif which == 'final':
            plot_rectangle(me.ww_min[ii, :], me.ww_max[ii, :], edgecolor='r')
        else:
            plot_rectangle(me.ww_min0[ii, :], me.ww_max0[ii, :])
            plot_rectangle(me.ww_min[ii, :], me.ww_max[ii, :], edgecolor='r')

        plt.plot(me.pp[ii, 0], me.pp[ii, 1], '.k')
        plt.title('weighting function ' + str(ii))

    def plot_f_box(me, ii, which='both'):
        plt.figure()
        b, k = ind2sub_batches(ii, me.batch_lengths)
        cm = dl.plot(me.ff_fenics[b])
        plt.colorbar(cm)

        if which == 'initial':
            plot_rectangle(me.ff_min0[ii, :], me.ff_max0[ii, :])
        elif which == 'final':
            plot_rectangle(me.ff_min[ii, :], me.ff_max[ii, :], edgecolor='r')
        else:
            plot_rectangle(me.ff_min0[ii, :], me.ff_max0[ii, :])
            plot_rectangle(me.ff_min[ii, :], me.ff_max[ii, :], edgecolor='r')

        plot_ellipse(me.all_mu[ii, :], me.all_Sigma[ii, :, :], me.tau)
        plt.plot(me.pp[ii, 0], me.pp[ii, 1], '.k')
        plt.title('impulse response ' + str(ii))

    def make_w_transfer_plot(me, ii):
        plt.figure(figsize=(12, 4), dpi=100, facecolor='w', edgecolor='k')

        w = me.ww_fenics[ii]
        plt.subplot(131)
        cm = dl.plot(w)
        plt.colorbar(cm)
        xlims = plt.xlim()
        ylims = plt.ylim()
        plt.title('weighting function ' + str(ii))

        _, (XX, YY) = make_regular_grid(me.ww_min[ii, :], me.ww_max[ii, :], me.ww_grid_shapes[ii])

        W = me.ww_grid[ii]
        plt.subplot(132)
        plt.pcolor(XX, YY, W)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.title('grid weighting function ' + str(ii))
        plt.colorbar()

        w2 = me.ww_G2F[ii](W)
        plt.subplot(133)
        cm = dl.plot(w2)
        plt.colorbar(cm)
        plt.title('backtransferred weighting function ' + str(ii))

    def make_f_transfer_plot(me, ii):
        plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

        b, _ = ind2sub_batches(ii, me.batch_lengths)
        f = me.ff_fenics[b]
        plt.subplot(131)
        cm = dl.plot(f)
        plt.colorbar(cm)
        xlims = plt.xlim()
        ylims = plt.ylim()
        plot_rectangle(me.ff_min[ii,:], me.ff_max[ii,:])
        plt.title('impulse_response ' + str(ii))

        _, (XX, YY) = make_regular_grid(me.ff_min[ii, :], me.ff_max[ii, :], me.ff_grid_shapes[ii])

        F = me.ff_grid1[ii]
        plt.subplot(132)
        plt.pcolor(XX, YY, F)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.title('grid impulse response ' + str(ii))
        plt.colorbar()

        f2 = me.ff_G2F[ii](F)
        plt.subplot(133)
        cm = dl.plot(f2)
        plt.colorbar(cm)
        plt.title('backtransferred impulse response ' + str(ii))

    def make_f_boundary_extension_plot(me, ii):
        f0 = me.ff_grid1[ii]
        f = me.ff_grid[ii]

        _, (Xi, Yi) = make_regular_grid(me.ff_min[ii, :], me.ff_max[ii, :], f0.shape)

        plt.figure(figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')

        plt.subplot(121)
        plt.pcolor(Xi, Yi, f0)
        plt.colorbar()
        plt.title('impulse response ' + str(ii) + ' without extension')

        plt.subplot(122)
        plt.pcolor(Xi, Yi, f)
        plt.plot(me.pp[ii, 0], me.pp[ii, 1], '.k')
        plt.colorbar()
        plt.title('impulse response ' + str(ii) + ' with extension')



def get_initial_weighting_function_boxes(weighting_functions, support_rtol=2e-2):
    num_pts = len(weighting_functions)
    V = weighting_functions[0].function_space()
    d = V.mesh().geometric_dimension()
    dof_coords = V.tabulate_dof_coordinates()

    ww_min0 = np.zeros((num_pts, d))
    ww_max0 = np.zeros((num_pts, d))
    for ii in range(num_pts):
        wi = weighting_functions[ii]
        wi_min, wi_max = function_support_box(wi.vector()[:], dof_coords, support_rtol=support_rtol)
        ww_min0[ii ,:] = wi_min
        ww_max0[ii ,:] = wi_max

    return ww_min0, ww_max0


def get_initial_impulse_response_boxes(all_mu, all_Sigma, tau):
    num_pts, d = all_mu.shape

    ff_min0 = np.zeros((num_pts, d))
    ff_max0 = np.zeros((num_pts, d))
    for ii in range(num_pts):
        min_pt, max_pt = ellipsoid_bounding_box(all_mu[ii, :], all_Sigma[ii, :, :], tau)
        ff_min0[ii, :] = min_pt
        ff_max0[ii, :] = max_pt

    return ff_min0, ff_max0


def box_h(box_min, box_max, dof_coords):
    box_mask = point_is_in_box(dof_coords, box_min, box_max)
    points_in_box = dof_coords[box_mask, :]

    T = cKDTree(points_in_box)
    dd, _ = T.query(points_in_box, k=2)
    h = np.min(dd[:,1])
    return h


def many_conforming_boxes(all_min0, all_max0, hh, pp):
    all_min = np.zeros(all_min0.shape)
    all_max = np.zeros(all_max0.shape)
    all_grid_shapes = []
    for k in range(all_min0.shape[0]):
        min_k, max_k, shape_k = conforming_box(all_min0[k,:], all_max0[k,:], pp[k,:], hh[k])
        all_min[k,:] = min_k
        all_max[k,:] = max_k
        all_grid_shapes.append(shape_k)
    all_grid_shapes = np.array(all_grid_shapes)
    return all_min, all_max, all_grid_shapes


def make_conforming_boxgrids(ww_min0, ww_max0, ff_min0, ff_max0, pp, dof_coords, grid_density_multiplier):
    num_pts = pp.shape[0]

    hh0_w = np.array([box_h(ww_min0[k,:], ww_max0[k,:], dof_coords)
                      for k in range(num_pts)])

    hh0_f = np.array([box_h(ff_min0[k, :], ff_max0[k, :], dof_coords)
                           for k in range(num_pts)])

    hh = np.min([hh0_w, hh0_f], axis=0) / grid_density_multiplier

    ww_min, ww_max, ww_grid_shapes = many_conforming_boxes(ww_min0, ww_max0, hh, pp)
    ff_min, ff_max, ff_grid_shapes = many_conforming_boxes(ff_min0, ff_max0, hh, pp)

    return ww_min, ww_max, ww_grid_shapes, ff_min, ff_max, ff_grid_shapes


class Function2Grid:
    def __init__(me, V, box_min, box_max, grid_shape,
                 mu=None, Sigma=None, tau=None):
        me.F2G = FenicsFunctionToRegularGridInterpolator(V, box_min, box_max, grid_shape)
        me.mu = mu
        me.Sigma = Sigma
        me.tau = tau

    def __call__(me, u_fenics, outside_domain_fill_value=0.0, use_extrapolation=False):
        U_array = me.F2G.interpolate(u_fenics, mu=me.mu, Sigma=me.Sigma, tau=me.tau,
                                     use_extrapolation=use_extrapolation,
                                     outside_domain_default_value=outside_domain_fill_value)
        return U_array


class Grid2Function:
    def __init__(me, V, box_min, box_max, fill_value=0.0):
        me.V = V
        dof_coords = V.tabulate_dof_coordinates()
        me.box_min = box_min
        me.box_max = box_max
        me.fill_value = fill_value
        box_mask = point_is_in_box(dof_coords, me.box_min, me.box_max)
        me.inds_of_points_in_box = np.argwhere(box_mask).reshape(-1)
        me.points_in_box = dof_coords[box_mask, :]

    def __call__(me, U_array):
        u = dl.Function(me.V)
        uu = grid_interpolate(me.box_min, me.box_max, U_array, me.points_in_box)
        uu[np.isnan(uu)] = me.fill_value
        u.vector()[me.inds_of_points_in_box] = uu
        return u


def make_grid_to_function_transfer_operators(V, all_min, all_max):
    num_pts = all_min.shape[0]
    print('making Grid2Function transfer operators')
    all_G2F = list()
    for ii in tqdm(range(num_pts)):
        G2F = Grid2Function(V, all_min[ii, :], all_max[ii, :])
        all_G2F.append(G2F)
    return all_G2F


def make_function_to_grid_transfer_operators(V, all_min, all_max, grid_shapes,
                                             all_mu=None, all_Sigma=None, tau=None):
    num_pts = all_min.shape[0]
    print('making Function2Grid transfer operators')
    all_F2G = list()
    for ii in tqdm(range(num_pts)):
        if all_mu is None:
            F2G = Function2Grid(V, all_min[ii, :], all_max[ii, :], grid_shapes[ii, :])
        else:
            F2G = Function2Grid(V, all_min[ii, :], all_max[ii, :], grid_shapes[ii, :],
                                mu=all_mu[ii, :], Sigma=all_Sigma[ii, :, :], tau=tau)
        all_F2G.append(F2G)
    return all_F2G


def postprocess_impulse_response(ii, pp, box_mins, box_maxes,
                                 impulse_responses_grid0,
                                 # all_mu, all_Sigma, tau,
                                 max_neighbors=10):
    PSI_i = impulse_responses_grid0[ii].copy()
    pi = pp[ii,:]
    min_i = box_mins[ii,:]
    max_i = box_maxes[ii,:]

    dd = np.linalg.norm(pp - pi.reshape((1,-1)), axis=1)
    inds = list(np.argsort(dd)[1:]) # first ind is the point itself
    for jj in inds[:max_neighbors]:
        if not np.any(np.isnan(PSI_i)):
            break

        PSI_j = impulse_responses_grid0[jj]
        pj = pp[jj,:]
        min_j = box_mins[jj,:]
        max_j = box_maxes[jj,:]

        _, _, PSI_i = \
            combine_grid_functions([min_i, min_j + pi - pj],
                                   [max_i, max_j + pi - pj],
                                   [PSI_i, PSI_j], expand_box=False)

    PSI_i[np.isnan(PSI_i)] = 0.0
    return PSI_i
