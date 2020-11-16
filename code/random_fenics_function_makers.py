import fenics
import numpy as np
from fenics_function_smoother import FenicsFunctionSmoother


def randn_fenics_function(function_space_V):
    x = fenics.Function(function_space_V)
    x.vector()[:] = np.random.randn(function_space_V.dim())
    return x


def rand_fenics_function(function_space_V):
    x = fenics.Function(function_space_V)
    x.vector()[:] = np.random.rand(function_space_V.dim())
    return x


class FenicsSmoothFunctionMaker:
    def __init__(me, function_space_V, smoothing_time=None):
        me.V = function_space_V
        if smoothing_time == None:
            me.function_smoother = FenicsFunctionSmoother(me.V)
        else:
            me.function_smoother = FenicsFunctionSmoother(me.V, smoothing_time=smoothing_time)

    def random_smooth_function(me, normalize=False, random_type='randn'):
        if random_type == 'randn':
            u = randn_fenics_function(me.V)
        else:
            u = rand_fenics_function(me.V)

        me.function_smoother.smooth(u)
        if normalize:
            u = u / fenics.norm(u)
        return u


def random_smooth_partition_of_unity(num_functions, function_space_V, temperature=1, normalize=False, random_type='randn'):
    sfm = FenicsSmoothFunctionMaker(function_space_V)
    ww0 = [fenics.exp(temperature * sfm.random_smooth_function(normalize=normalize, random_type=random_type))
           for _ in range(num_functions)]

    Z = fenics.Function(function_space_V)
    for k in range(len(ww0)):
        Z = Z + ww0[k]
    iZ = fenics.Constant(1.0) / Z

    ww = []
    for w0 in ww0:
        ww.append(fenics.project(w0 * iZ, function_space_V))

    return ww


def random_spd_matrix(d):
    U,ss,_ = np.linalg.svd(np.random.randn(d,d))
    A = np.dot(U, np.dot(np.diag(ss**3), U.T))
    return A
