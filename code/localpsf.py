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


##


def remove_points_near_boundary(candidate_points, eval_boundary_function, boundary_epsilon):
    near_boundary_inds = (abs(eval_boundary_function(candidate_points)) > boundary_epsilon)
    return candidate_points[near_boundary_inds, :]

##



class LocalPSF:
    def __init__(me, apply_hessian_H, apply_hessian_transpose_Ht, function_space_V, error_epsilon=1e-2,
                 boundary_tol=5e-1, num_standard_deviations_tau=3, max_batches=5, verbose=True,
                 mass_matrix_M=None, solve_mass_matrix_M=None):
        me.apply_H = apply_hessian_H
        me.apply_Ht = apply_hessian_transpose_Ht
        me.V = function_space_V
        me.error_epsilon = error_epsilon
        me.boundary_tol = boundary_tol
        me.tau = num_standard_deviations_tau
        me.max_batches = max_batches

        me.X = me.V.tabulate_dof_coordinates()
        me.N, me.d = me.X.shape

        if (mass_matrix_M == None) or (solve_mass_matrix_M == None):
            print('making mass matrix and solver')
            me.M = make_mass_matrix(me.V)
            me.solve_M = make_fenics_amg_solver(me.M)
            # me.M, M_solver = make_mass_matrix_and_solver(function_space_V)
            # me.solve_M = lambda x: M_solver.solve(x)
        else:
            me.M = mass_matrix_M
            me.solve_M = solve_mass_matrix_M


        print('getting boundary function')
        me.boundary_function = get_boundary_function(me.V, me.apply_Ht, me.solve_M)

        print('getting spatially varying volume')
        me.vol = compute_spatially_varying_volume(me.V, me.apply_Ht, me.solve_M)

        print('getting spatially varying mean')
        me.mu = compute_spatially_varying_mean(me.V, me.apply_Ht, me.solve_M, me.vol)

        print('getting spatially varying covariance')
        me.Sigma = get_spatially_varying_covariance(me.V, me.apply_Ht, me.solve_M, me.vol, me.mu)

        print('constructing fast evaluators')
        me.eval_boundary_function = FenicsFunctionFastGridEvaluator(me.boundary_function)
        me.eval_vol = FenicsFunctionFastGridEvaluator(me.vol)
        me.eval_mu = FenicsFunctionFastGridEvaluator(me.mu)
        me.eval_Sigma = FenicsFunctionFastGridEvaluator(me.Sigma)
        print('done')

        interior_inds = (abs(me.eval_boundary_function(me.X)) < me.boundary_tol)
        me.candidate_points = me.X[interior_inds, :]
        me.candidate_mu = me.eval_mu(me.candidate_points)
        me.candidate_Sigma = me.eval_Sigma(me.candidate_points).reshape((-1, me.d, me.d))
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

        eval_ww_flat = [FenicsFunctionFastGridEvaluator(w) for w in me.weighting_functions]
        me.put_flat_list_into_batched_list_of_lists(eval_ww_flat, me.eval_weighting_functions_by_batch)

    def add_new_batch(me):
        qq = me.X[me.candidate_inds, :]
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
        me.eval_dirac_comb_responses.append(FenicsFunctionFastGridEvaluator(new_dirac_comb_response))

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


    def evaluate_approximate_hessian_entries_at_points_yy_xx(me, yy, xx):
        return me.BPC.compute_product_convolution_entries(yy, xx)

    def plot_boundary_region(me):
        if me.d != 2:
            print('d=', me.d, ', can only plot boundary region for d=2')
            return

        xx_boundary = me.X[me.Omega_B, 0]
        yy_boundary = me.X[me.Omega_B, 1]

        xx_interior = me.X[me.Omega_I, 0]
        yy_interior = me.X[me.Omega_I, 1]

        plt.figure()
        plt.scatter(xx_boundary, yy_boundary, c='r')
        plt.scatter(xx_interior, yy_interior, c='b')

        plt.legend(['boundary points', 'interior points'])

    def integrate_function_over_Omega(me, f_vec):
        return np.dot(f_vec, me.ones_dual_vector)

    def compute_mean_of_function(me, f_vec):
        V = me.integrate_function_over_Omega(f_vec)
        f_vec_hat = f_vec / V
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


