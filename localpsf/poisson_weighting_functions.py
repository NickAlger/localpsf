import numpy as np
import dolfin as dl
# import scipy.linalg as sla
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from neumann_poisson_solver import NeumannPoissonSolver

from nalger_helper_functions import make_dense_lu_solver, NeumannPoissonSolver


def make_poisson_weighting_functions(function_space_V, points_pp):
    print('Making Poisson weighting functions')
    initial_points = [points_pp[k,:] for k in range(points_pp.shape[0])]
    PSI = PoissonSquaredInterpolation(function_space_V, initial_points=initial_points)
    ww = PSI.weighting_functions
    return ww


class PoissonSquaredInterpolation:
    def __init__(me, function_space_V, initial_points=None):
        me.V = function_space_V
        me.N = me.V.dim()

        u_trial = dl.TrialFunction(me.V)
        v_test = dl.TestFunction(me.V)

        mass_form = u_trial * v_test * dl.dx
        me.M = dl.assemble(mass_form)

        me.constant_one_function = dl.interpolate(dl.Constant(1.0), me.V)

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

        print('Poisson weighting functions: computing Poisson impulse responses')
        new_impulse_responses = list()
        for p in tqdm(new_points):
            new_impulse_responses.append(me.solve_neumann_point_source(p, point_type='coords').vector())
        me.impulse_responses = me.impulse_responses + new_impulse_responses

        print('Poisson weighting functions: computing S')
        S = np.zeros((me.num_pts, me.num_pts))
        for i in tqdm(range(me.num_pts)):
            for j in range(me.num_pts):
                S[i, j] = me.impulse_responses[i].inner(me.M * me.impulse_responses[j])
        me.solve_S = make_dense_lu_solver(S)
        me.eta = me.solve_S(np.ones(me.num_pts))
        me.mu = np.dot(np.ones(me.num_pts), me.eta)

        print('Poisson weighting functions: computing Poisson squared impulse responses')
        new_smooth_basis_vectors = list()
        for u in tqdm(new_impulse_responses):
            new_smooth_basis_vectors.append(-me.solve_neumann_poisson(me.M * u).vector())
        me.smooth_basis = me.smooth_basis + new_smooth_basis_vectors

        me.compute_weighting_functions()

    def compute_weighting_functions(me):
        print('Poisson weighting functions: forming weighting functions')
        me.weighting_functions.clear()
        I = np.eye(me.num_pts)
        for k in tqdm(range(me.num_pts)):
            ek = I[:,k]
            w = dl.Function(me.V)
            w.vector()[:] = me.interpolate_values(ek)
            me.weighting_functions.append(w)

    @property
    def num_pts(me):
        return len(me.points)

    def interpolate_values(me, values_at_points_y):
        y = values_at_points_y
        alpha = (1. / me.mu) * np.dot(me.eta, y)
        p = -me.solve_S(y - alpha * np.ones(me.num_pts))
        u = alpha * me.constant_one_function.vector()
        for k in range(len(me.smooth_basis)):
            u = u + me.smooth_basis[k] * p[k]
        return u


