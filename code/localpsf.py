import numpy as np
import fenics
import matplotlib.pyplot as plt
from plot_ellipse import plot_ellipse
from localpsf_helpers import *
from poisson_interpolation import PoissonSquaredInterpolation
from eval_product_convolution import BatchProductConvolution
from make_fenics_amg_solver import make_fenics_amg_solver
from mass_matrix import make_mass_matrix
from fenics_function_fast_grid_evaluator import FenicsFunctionFastGridEvaluator
import hlibpro_python_wrapper as hpro
import scipy.sparse as sps

from time import time


def make_dirac_comb_dual_vector(points_pp, function_space_V):
    pp = points_pp
    V = function_space_V
    num_pts, d = pp.shape
    dirac_comb_dual_vector = fenics.assemble(fenics.Constant(0.0) * fenics.TestFunction(V) * fenics.dx)
    for k in range(num_pts):
        ps = fenics.PointSource(V, fenics.Point(pp[k,:]), 1.0)
        ps.apply(dirac_comb_dual_vector)
    return dirac_comb_dual_vector


def get_hessian_dirac_comb_response(points_pp, function_space_V, apply_hessian_H, solve_mass_matrix_M):
    apply_H = apply_hessian_H
    solve_M = solve_mass_matrix_M

    dirac_comb_dual_vector = make_dirac_comb_dual_vector(points_pp, function_space_V)
    dirac_comb_response = fenics.Function(function_space_V)
    dirac_comb_response.vector()[:] = solve_M(apply_H(solve_M(dirac_comb_dual_vector)))
    return dirac_comb_response


def compute_spatially_varying_volume(function_space_V, apply_hessian_transpose_Ht, solve_mass_matrix_M):
    V = function_space_V
    apply_Ht = apply_hessian_transpose_Ht
    solve_M = solve_mass_matrix_M

    vol = fenics.Function(V)
    constant_fct = fenics.interpolate(fenics.Constant(1.0), V)
    vol.vector()[:] = solve_M(apply_Ht(constant_fct.vector()))
    return vol


def compute_spatially_varying_mean(function_space_V, apply_hessain_transpose_Ht,
                                   solve_mass_matrix_M, volume_function_vol):
    V = function_space_V
    apply_Ht = apply_hessain_transpose_Ht
    solve_M = solve_mass_matrix_M
    vol = volume_function_vol

    d = V.mesh().geometric_dimension()
    V_vec = fenics.VectorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())

    mu = fenics.Function(V_vec)
    for k in range(d):
        linear_fct = fenics.interpolate(fenics.Expression('x[k]', element=V.ufl_element(), k=k), V)
        mu_k = fenics.Function(V)
        mu_k.vector()[:] = solve_M(apply_Ht(linear_fct.vector()))
        mu_k = fenics.project(mu_k / vol, V)
        fenics.assign(mu.sub(k), mu_k)

    mu.set_allow_extrapolation(True)
    return mu


def get_spatially_varying_covariance(function_space_V, apply_hessian_transpose_Ht, solve_mass_matrix_M,
                                     volume_function_vol, mean_function_mu):
    V = function_space_V
    apply_Ht = apply_hessian_transpose_Ht
    solve_M = solve_mass_matrix_M
    vol = volume_function_vol
    mu = mean_function_mu

    d = V.mesh().geometric_dimension()
    V_mat = fenics.TensorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())

    Sigma = fenics.Function(V_mat)
    for k in range(d):
        for j in range(k + 1):
            quadratic_fct = fenics.interpolate(fenics.Expression('x[k]*x[j]', element=V.ufl_element(), k=k, j=j), V)
            Sigma_kj = fenics.Function(V)
            Sigma_kj.vector()[:] = solve_M(apply_Ht(quadratic_fct.vector()))
            Sigma_kj = fenics.project(Sigma_kj / vol - mu.sub(k) * mu.sub(j), V)
            fenics.assign(Sigma.sub(k + d * j), Sigma_kj)
            fenics.assign(Sigma.sub(j + d * k), Sigma_kj)

    Sigma.set_allow_extrapolation(True)
    return Sigma


def get_boundary_function(function_space_V, apply_hessian_transpose_Ht, solve_mass_matrix_M):
    V = function_space_V
    apply_Ht = apply_hessian_transpose_Ht
    solve_M = solve_mass_matrix_M

    boundary_source = fenics.Function(V)
    boundary_source.vector()[:] = solve_M(fenics.assemble(fenics.TestFunction(V) * fenics.ds))

    boundary_function = fenics.Function(V)
    boundary_function.vector()[:] = solve_M(apply_Ht(boundary_source.vector()))
    return boundary_function


def get_boundary_inds(function_space_V):
    boundary_form = fenics.TestFunction(function_space_V) * fenics.Constant(1.0) * fenics.ds
    return np.argwhere(fenics.assemble(boundary_form)[:] > 1e-12).reshape(-1)


##


def remove_points_near_boundary(candidate_points, eval_boundary_function, boundary_epsilon):
    near_boundary_inds = (abs(eval_boundary_function(candidate_points)) > boundary_epsilon)
    return candidate_points[near_boundary_inds, :]

##


class FenicsRegularGridInterpolator2D:
    # https://github.com/NickAlger/helper_functions/blob/master/interpolate_fenics_function_onto_regular_grid.ipynb
    def __init__(me, function_space_V, grid_oversampling_parameter=2.0):
        me.V = function_space_V
        me.mesh = me.V.mesh()
        me.eta = grid_oversampling_parameter

        me.X = me.V.tabulate_dof_coordinates()
        min_point0 = np.min(me.X, axis=0)
        max_point0 = np.max(me.X, axis=0)
        mid_point = (min_point0 + max_point0) / 2.
        delta =  (max_point0 - min_point0) / 2.
        me.min_point = mid_point - (1. + 1e-14) * delta
        me.max_point = mid_point + (1. + 1e-14) * delta

        me.nn = (me.eta * (me.max_point - me.min_point) / me.mesh.hmin()).astype(int)
        me.bounding_box_mesh = fenics.RectangleMesh(fenics.Point(me.min_point),
                                                    fenics.Point(me.max_point),
                                                    me.nn[0], me.nn[1])

        me.V_grid = fenics.FunctionSpace(me.bounding_box_mesh, 'CG', 1)
        me.X_grid = me.V_grid.tabulate_dof_coordinates()

        me.sort_inds = np.lexsort(me.X_grid[:,::-1].T)

        me.xx = np.linspace(me.min_point[0], me.max_point[0], me.nn[0] + 1)
        me.yy = np.linspace(me.min_point[1], me.max_point[1], me.nn[1] + 1)

    def interpolate_function(me, u):
        u.set_allow_extrapolation(True)
        u_grid = fenics.interpolate(u, me.V_grid)
        U = u_grid.vector()[me.sort_inds].reshape(me.nn + 1)
        return U

    def make_fast_grid_function(me, u):
        U_grid = me.interpolate_function(u)
        return FastGridFunction2D(U_grid, me.min_point, me.max_point)


class FastGridFunction2D:
    def __init__(me, U_grid, min_point, max_point):
        me.U_grid = U_grid
        me.min_point = min_point
        me.max_point = max_point
        me.xmin = me.min_point[0]
        me.xmax = me.max_point[0]
        me.ymin = me.min_point[1]
        me.ymax = me.max_point[1]

    def __call__(me, eval_coords):
        return hpro.hpro_cpp.grid_interpolate_vectorized(eval_coords, me.xmin, me.xmax, me.ymin, me.ymax, me.U_grid)


##



class LocalPSF:
    def __init__(me, apply_hessian_H, apply_hessian_transpose_Ht, function_space_V, error_epsilon=1e-2,
                 num_standard_deviations_tau=3, max_batches=5, verbose=True):
        me.apply_H = apply_hessian_H
        me.apply_Ht = apply_hessian_transpose_Ht
        me.V = function_space_V
        me.error_epsilon = error_epsilon
        me.tau = num_standard_deviations_tau
        me.max_batches = max_batches

        me.X = me.V.tabulate_dof_coordinates()
        me.N, me.d = me.X.shape

        me.FRGI = FenicsRegularGridInterpolator2D(me.V)

        print('making mass matrix and solver')
        me.M = make_mass_matrix(me.V)
        me.solve_M = make_fenics_amg_solver(me.M)

        print('getting spatially varying volume')
        me.vol = compute_spatially_varying_volume(me.V, me.apply_Ht, me.solve_M)

        print('getting spatially varying mean')
        me.mu = compute_spatially_varying_mean(me.V, me.apply_Ht, me.solve_M, me.vol)

        print('getting spatially varying covariance')
        me.Sigma = get_spatially_varying_covariance(me.V, me.apply_Ht, me.solve_M, me.vol, me.mu)

        print('constructing fast evaluators')
        me.eval_vol = me.FRGI.make_fast_grid_function(me.vol)
        me.eval_mu = FenicsFunctionFastGridEvaluator(me.mu)
        eval_Sigma0 = FenicsFunctionFastGridEvaluator(me.Sigma)
        me.eval_Sigma = lambda pp : eval_Sigma0(pp).reshape((-1, me.d, me.d))
        print('done')

        print('getting nodes on boundary')
        me.boundary_inds = get_boundary_inds(me.V) # indices for nodes exactly on the boundary
        print('done')

        all_mu = me.eval_mu(me.X)
        all_Sigma = me.eval_Sigma(me.X)

        print('computing inds of points far from boundary')
        me.inds_of_points_far_from_boundary = list()
        for k in range(me.N):
            far_bp = points_which_are_not_in_ellipsoid_numba(all_Sigma[k,:,:], all_mu[k,:],
                                                             me.X[me.boundary_inds,:], me.tau)
            if np.all(far_bp):
                me.inds_of_points_far_from_boundary.append(k)
        print('done')

        me.inds_of_points_far_from_boundary = np.arange(me.N) # Testing

        me.far_from_boundary_function = fenics.Function(me.V)
        me.far_from_boundary_function.vector()[me.inds_of_points_far_from_boundary] = 1.0

        me.candidate_points = me.X[me.inds_of_points_far_from_boundary, :]
        me.candidate_mu = all_mu[me.inds_of_points_far_from_boundary, :]
        me.candidate_Sigma = all_Sigma[me.inds_of_points_far_from_boundary, :, :]
        me.candidate_inds = list(range(me.candidate_points.shape[0]))

        me.point_batches = list()
        me.dirac_comb_responses = list()
        me.eval_dirac_comb_responses = list()
        me.eval_weighting_functions_by_batch = list()
        me.mu_batches = list()
        me.Sigma_batches = list()
        me.PSI = PoissonSquaredInterpolation(me.V)
        me.weighting_functions = me.PSI.weighting_functions
        me.BPC = BatchProductConvolution(me.eval_dirac_comb_responses, me.eval_weighting_functions_by_batch,
                                         me.point_batches, me.mu_batches, me.Sigma_batches, me.tau)
        for k in range(max_batches):
            me.add_new_batch()

        all_points = list()
        for pp in me.point_batches:
            for k in range(pp.shape[0]):
                all_points.append(pp[k,:])
        me.PSI.add_points(all_points)

        # eval_ww_flat = [FenicsFunctionFastGridEvaluator(w) for w in me.weighting_functions]
        eval_ww_flat = [me.FRGI.make_fast_grid_function(w) for w in me.weighting_functions]
        me.put_flat_list_into_batched_list_of_lists(eval_ww_flat, me.eval_weighting_functions_by_batch)

        me.BPC_cpp = me.build_BPC_cpp()

    def add_new_batch(me):
        qq = me.candidate_points[me.candidate_inds, :]
        dd = np.inf * np.ones(len(me.candidate_inds))
        for pp in me.point_batches:
            for k in range(pp.shape[0]):
                pk = pp[k,:].reshape((1,-1))
                ddk = np.linalg.norm(pk - qq, axis=1)
                dd = np.min([dd, ddk], axis=0)
        candidate_inds_ordered_by_distance = np.array(me.candidate_inds)[np.argsort(dd)]

        new_inds = choose_sample_points_batch(me.candidate_mu, me.candidate_Sigma,
                                              me.tau, candidate_inds_ordered_by_distance, randomize=False)

        # new_inds = choose_sample_points_batch(me.candidate_mu, me.candidate_Sigma,
        #                                       me.tau, np.array(me.candidate_inds))
        new_points = me.candidate_points[new_inds, :]
        me.point_batches.append(new_points)
        me.candidate_inds = list(np.setdiff1d(me.candidate_inds, new_inds))

        me.mu_batches.append(me.eval_mu(new_points))
        me.Sigma_batches.append(me.eval_Sigma(new_points).reshape((-1,me.d,me.d)))

        new_dirac_comb_response = get_hessian_dirac_comb_response(new_points, me.V, me.apply_H, me.solve_M)
        me.dirac_comb_responses.append(new_dirac_comb_response)

        me.eval_dirac_comb_responses.append(me.FRGI.make_fast_grid_function(new_dirac_comb_response))
        # me.eval_dirac_comb_responses.append(FenicsFunctionFastGridEvaluator(new_dirac_comb_response))

        # num_new_pts = new_points.shape[0]
        # me.PSI.add_points([new_points[k, :] for k in range(num_new_pts)])
        #
        # eval_ww_flat = [FenicsFunctionFastGridEvaluator(w) for w in me.weighting_functions]
        # me.put_flat_list_into_batched_list_of_lists(eval_ww_flat, me.eval_weighting_functions_by_batch)


    def put_flat_list_into_batched_list_of_lists(me, flat_list, batched_list_of_lists):
        batched_list_of_lists.clear()
        ii = 0
        for pp in me.point_batches:
            this_batch = list()
            for k in range(pp.shape[0]):
                this_batch.append(flat_list[ii])
                ii = ii+1
            batched_list_of_lists.append(this_batch)
        return batched_list_of_lists

    @property
    def num_batches(me):
        return len(me.point_batches)

    @property
    def r(me):
        return len(me.weighting_functions)

    @property
    def points(me):
        return np.array(me.PSI.points)

    def evaluate_approximate_hessian_entries_at_points_yy_xx(me, yy, xx):
        return me.BPC_cpp.compute_entries(yy, xx)

    def old_evaluate_approximate_hessian_entries_at_points_yy_xx(me, yy, xx):
        return me.BPC.compute_product_convolution_entries(yy, xx)

    def make_plots(me):
        if me.d != 2:
            print('d=', me.d, ', can only make plots for d=2')
            return

        plt.figure()
        fenics.plot(me.far_from_boundary_function)
        plt.title('far from boundary region')

        plt.figure()
        cm = fenics.plot(me.vol)
        plt.colorbar(cm)
        plt.title('volume function')

        plt.figure()
        cm = fenics.plot(me.mu.sub(0))
        plt.colorbar(cm)
        plt.title('mean in x direction')

        plt.figure()
        cm = fenics.plot(me.mu.sub(1))
        plt.colorbar(cm)
        plt.title('mean in y direction')

        for k in range(me.num_batches):
            me.plot_impulse_response_batch(k)

        w_inds = np.random.permutation(len(me.weighting_functions))[:5]
        for k in w_inds:
            me.plot_weighting_function(k)

    def plot_weighting_function(me, w_ind):
        plt.figure()
        cm = fenics.plot(me.weighting_functions[w_ind])
        plt.colorbar(cm)
        plt.plot(me.points[:, 0], me.points[:, 1], '.k')
        plt.plot(me.points[w_ind,0], me.points[w_ind,1], '.r')
        plt.title('weighting function ' + str(w_ind))

    def plot_impulse_response_batch(me, batch_ind):
        plt.figure()

        pp_batch = me.point_batches[batch_ind]
        Hdc = me.dirac_comb_responses[batch_ind]

        cmap = fenics.plot(Hdc)
        plt.colorbar(cmap)

        plt.title('Hessian response 3-sigma support, batch '+str(batch_ind))

        for k in range(pp_batch.shape[0]):
            p = pp_batch[k, :]
            mu_p = me.mu(p)
            C_p = me.Sigma(p).reshape((me.d, me.d))
            plot_ellipse(plt.gca(), mu_p, C_p, me.tau)

    def integrate_function_over_Omega(me, f_vec):
        f = fenics.Function(me.V)
        f.vector()[:] = f_vec
        return fenics.assemble(f * fenics.dx)

    def compute_mean_of_function(me, f_vec):
        vol = me.integrate_function_over_Omega(f_vec)
        f_vec_hat = f_vec / vol
        mu_f = np.zeros(me.d)
        for k in range(me.d):
           mu_f[k] =  me.integrate_function_over_Omega(me.X[:,k] * f_vec_hat)
        return mu_f

    def compute_covariance_of_function(me, f_vec):
        V = me.integrate_function_over_Omega(f_vec)
        f_vec_hat = f_vec / V
        mu_f = me.compute_mean_of_function(f_vec)
        Sigma_f = np.zeros((me.d, me.d))
        for k in range(me.d):
            for j in range(me.d):
                Sigma_f[k,j] = me.integrate_function_over_Omega(f_vec_hat * (me.X[:,k] - mu_f[k]) * (me.X[:,j] - mu_f[j]))
        return Sigma_f

    def check_volume_mean_covariance_at_random_point(me):
        k = np.random.randint(me.N)
        delta_pk = np.zeros(me.N)
        delta_pk[k] = 1.
        vk = me.solve_M(me.apply_H(me.solve_M(delta_pk)))

        V_vk = me.vol[k]
        V_vk_true = me.integrate_function_over_Omega(vk)
        err_vol = np.linalg.norm(V_vk - V_vk_true)/np.linalg.norm(V_vk_true)

        mu_vk = me.mu[k,:]
        mu_vk_true = me.compute_mean_of_function(vk)
        err_mean = np.linalg.norm(mu_vk - mu_vk_true)/np.linalg.norm(mu_vk_true)

        Sigma_vk = me.Sigma[k, :, :]
        Sigma_vk_true = me.compute_covariance_of_function(vk)
        err_cov = np.linalg.norm(Sigma_vk - Sigma_vk_true) / np.linalg.norm(Sigma_vk_true)

        print('k=', k, ', err_vol=', err_vol, ', err_mean=', err_mean, ', err_cov=', err_cov)
        return err_vol, err_mean, err_cov

    def build_BPC_cpp(me):
        eta_array_batches = [eta.U_grid for eta in me.eval_dirac_comb_responses]

        ww_array_batches = list()
        for eval_ww_batch in me.eval_weighting_functions_by_batch:
            ww_array_batches.append([eval_w.U_grid for eval_w in eval_ww_batch])

        Sigma_batches_recursive_list = list()
        for Sigma_batch in me.Sigma_batches:
            Sigma_batches_recursive_list.append([Sigma_batch[k, :, :] for k in range(Sigma_batch.shape[0])])

        grid_xmin = me.FRGI.min_point[0]
        grid_xmax = me.FRGI.max_point[0]
        grid_ymin = me.FRGI.min_point[1]
        grid_ymax = me.FRGI.max_point[1]

        BPC_cpp = hpro.hpro_cpp.ProductConvolutionMultipleBatches(eta_array_batches,
                                                                  ww_array_batches,
                                                                  me.point_batches,
                                                                  me.mu_batches,
                                                                  Sigma_batches_recursive_list,
                                                                  me.tau, grid_xmin, grid_xmax, grid_ymin, grid_ymax)
        return BPC_cpp

    def build_hmatrix(me, block_cluster_tree, tol=1e-6, symmetrize=False):
        BPC_coefffn = hpro.hpro_cpp.ProductConvolutionCoeffFn(me.BPC_cpp, me.X)
        hmatrix_cpp_object = hpro.hpro_cpp.build_hmatrix_from_coefffn(BPC_coefffn, block_cluster_tree, tol)
        A_hmatrix = hpro.HMatrixWrapper(hmatrix_cpp_object, block_cluster_tree)

        if symmetrize:
            A_hmatrix = A_hmatrix.sym()

        M_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(me.M)
        M_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_csc, block_cluster_tree)

        A_hmatrix = M_hmatrix * (A_hmatrix * M_hmatrix)

        return A_hmatrix


def convert_fenics_csr_matrix_to_scipy_csr_matrix(A_fenics):
    ai, aj, av = fenics.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy

