import fenics
import numpy as np
from scipy.interpolate import griddata

run_test = False

class FenicsFunctionExtendByZeroEvaluator:
    def __init__(me, f_vec, V):
        me.f = fenics.Function(V)
        me.f.vector()[:] = f_vec.copy()
        me.bbt = V.mesh().bounding_box_tree()
        me.d = V.mesh().geometric_dimension()

        me.maximum_entity_bound = 2**30

    def __call__(me, points_pp):
        if len(points_pp.shape) == 1:
            N = 1
        else: # len(points_pp.shape) == 2
            N, _ = points_pp.shape
        points_pp = points_pp.reshape((N, me.d))

        inds_of_points_in_mesh = []
        for k in range(N):
            pk = fenics.Point(points_pp[k,:])
            if me.bbt.compute_first_entity_collision(pk) < me.maximum_entity_bound:
                inds_of_points_in_mesh.append(k)

        ff = np.zeros(N)
        for k in inds_of_points_in_mesh:
            pk = fenics.Point(points_pp[k, :])
            ff[k] = me.f(pk)
        return ff


if run_test:
    mesh = fenics.UnitSquareMesh(10,10)
    V = fenics.FunctionSpace(mesh, 'CG', 2)

    u = fenics.Function(V)
    u.vector()[:] = np.random.randn(V.dim())

    V_evaluator = FenicsFunctionExtendByZeroEvaluator(u.vector()[:], V)

    p = np.array([0.5,0.5])
    up_true = u(fenics.Point(p))
    up = V_evaluator(p)
    err_V_evaluator_single = np.abs(up - up_true) / np.abs(up_true)
    print('err_V_evaluator_single=', err_V_evaluator_single)

    N = 1000
    pp = 3 * np.random.rand(N,2) - 1.0 # random points in [-1, 2]^2

    upp_true = np.zeros(N)
    for k in range(N):
        pk = pp[k,:]
        if np.all(pk >= 0) and np.all(pk <= 1):
            upp_true[k] = u(fenics.Point(pk))

    upp = V_evaluator(pp)

    err_V_evaluator_many = np.linalg.norm(upp - upp_true)/np.linalg.norm(upp_true)
    print('err_V_evaluator=', err_V_evaluator_many)
