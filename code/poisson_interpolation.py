import numpy as np
import fenics
import scipy.linalg as sla
from neumann_poisson_solver import NeumannPoissonSolver


class PoissonSquaredInterpolation:
    def __init__(me, function_space_V, initial_points=None):
        me.V = function_space_V
        me.N = me.V.dim()

        u_trial = fenics.TrialFunction(me.V)
        v_test = fenics.TestFunction(me.V)

        mass_form = u_trial * v_test * fenics.dx
        me.M = fenics.assemble(mass_form)

        me.constant_one_function = fenics.interpolate(fenics.Constant(1.0), me.V)

        me.NPPSS = NeumannPoissonSolver(me.V)
        me.solve_neumann_poisson = me.NPPSS.solve
        me.solve_neumann_point_source = me.NPPSS.solve_point_source

        me.points = list()
        me.impulse_responses = list()
        me.solve_S = lambda x: np.nan
        me.eta = np.zeros(me.num_pts)
        me.mu = np.nan
        me.smooth_basis = list()
        me.weighting_functions = list()

        if initial_points is not None:
            me.add_points(initial_points)

    def add_points(me, new_points):
        me.points = me.points + new_points

        new_impulse_responses = list()
        for p in new_points:
            new_impulse_responses.append(me.solve_neumann_point_source(p, point_type='coords').vector())
        me.impulse_responses = me.impulse_responses + new_impulse_responses

        S = np.zeros((me.num_pts, me.num_pts))
        for i in range(me.num_pts):
            for j in range(me.num_pts):
                S[i, j] = me.impulse_responses[i].inner(me.M * me.impulse_responses[j])
        me.solve_S = make_dense_lu_solver(S)
        me.eta = me.solve_S(np.ones(me.num_pts))
        me.mu = np.dot(np.ones(me.num_pts), me.eta)

        new_smooth_basis_vectors = list()
        for u in new_impulse_responses:
            new_smooth_basis_vectors.append(-me.solve_neumann_poisson(me.M * u).vector())
        me.smooth_basis = me.smooth_basis + new_smooth_basis_vectors

        me.compute_weighting_functions()

    def compute_weighting_functions(me):
        me.weighting_functions.clear()
        I = np.eye(me.num_pts)
        for k in range(me.num_pts):
            ek = I[:,k]
            w = fenics.Function(me.V)
            w.vector()[:] = me.interpolate_values(ek)
            me.weighting_functions.append(w)

    @property
    def num_pts(me):
        return len(me.points)

    def interpolate_values(me, values_at_points_y):
        y = values_at_points_y
        alpha = (1. / me.mu) * np.dot(me.eta, y)
        p = -me.solve_S(y - alpha * np.ones(me.num_pts))
        # print('np.sort(np.abs(p))=', np.sort(np.abs(p)))
        good_inds = np.argwhere(np.abs(p) > 1e-3 * np.max(np.abs(p)))
        p2 = np.zeros(p.shape)
        p2[good_inds] = p[good_inds]
        print('len(good_inds)=', len(good_inds))
        u = alpha * me.constant_one_function.vector()
        for k in range(len(me.smooth_basis)):
            u = u + me.smooth_basis[k] * p2[k]
        return u


def make_dense_lu_solver(M):
    M_lu, M_pivot = sla.lu_factor(M)
    solve_M = lambda b: sla.lu_solve((M_lu, M_pivot), b)
    return solve_M
