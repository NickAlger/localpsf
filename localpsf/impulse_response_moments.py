import numpy as np
import dolfin as dl
import typing as typ
import scipy.sparse as sps
from scipy.spatial import KDTree



def impulse_response_moments(V_in, V_out, apply_At, solve_M_in,
                             max_scale_discrepancy=1e5):
    '''Computes spatially varying volume, mean, and covariance of impulse response function for operator
        A : V_in -> V_out

    Parameters
    ----------
    V_in: fenics FunctionSpace.
    V_out: fenis FunctionSpace.
    apply_At: callable. u -> A^T u. Maps fenics Vector to fenics Vector.
    solve_M_in: callable. u -> M_in^-1 u. Maps fenics Vector to fenics Vector.
        Solver for input space mass matrix

    Returns
    -------
    vol: scalar-valued fenics Function. spatially varying volume of impulse response
    mu: vector-valued fenics Function. spatially varying mean of impulse response
    Sigma: tensor-valued fenics Function. spatially varying covariance of impulse response

    '''
    d = V_in.mesh().geometric_dimension()
    N = V_in.dim()

    print('getting spatially varying volume')
    vol = np.zeros(N)
    constant_fct = dl.interpolate(dl.Constant(1.0), V_out)
    vol[:] = solve_M_in(apply_At(constant_fct.vector()))[:]

    min_vol = np.max(vol) / max_scale_discrepancy
    bad_inds = (vol < min_vol)
    rvol = vol.copy()
    rvol[bad_inds] = min_vol

    print('getting spatially varying mean')
    mu = np.zeros((N,d))
    for k in range(d):
        linear_fct = dl.interpolate(dl.Expression('x[k]', element=V_out.ufl_element(), k=k), V_out)
        mu_k_base = solve_M_in(apply_At(linear_fct.vector()))[:]
        mu[:,k] = mu_k_base / rvol

    print('getting spatially varying covariance')
    Sigma = np.zeros((N,d,d))
    for k in range(d):
        for j in range(k + 1):
            quadratic_fct = dl.interpolate(dl.Expression('x[k]*x[j]', element=V_out.ufl_element(), k=k, j=j), V_out)
            Sigma_kj_base = solve_M_in(apply_At(quadratic_fct.vector()))[:]
            Sigma[:,k,j] = Sigma_kj_base / rvol - mu[:,k]*mu[:,j]
            Sigma[:,j,k] = Sigma[:,k,j]

    return vol, mu, Sigma, bad_inds
