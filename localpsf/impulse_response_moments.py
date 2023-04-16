import numpy as np
import dolfin as dl
import typing as typ
import scipy.sparse as sps
from scipy.spatial import KDTree


def impulse_response_moments_simplified(
        apply_At: typ.Callable[[np.ndarray], np.ndarray],  # V_out -> V_in
        dof_coords_out: np.ndarray, # shape=(ndof_out, gdim_out)
        mass_lumps_in: np.ndarray, # shape=(ndof_in,)
        stable_division_rtol: float=1.0e-10,
        display=False,
        ) -> typ.Tuple[np.ndarray, # vol, shape=(ndof_in,), dtype=float
                       np.ndarray, # mean, shape=(ndof_in, gdim_out), dtype=float
                       np.ndarray]: # covariance, shape=(ndof_in, gdim_out, gdim_out), dtype=float
    '''Computes spatially varying impulse response volume, mean, and covariance for A : V_in -> V_out
    '''
    assert(0.0 <= stable_division_rtol)
    assert(stable_division_rtol < 1.0)
    ndof_in = len(mass_lumps_in)
    assert(mass_lumps_in.shape == (ndof_in,))
    gdim_out, ndof_out = dof_coords_out.shape

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    printmaybe('getting spatially varying volume')
    constant_fct = np.ones(ndof_out)
    vol = apply_At(constant_fct) / mass_lumps_in
    assert(vol.shape == (ndof_in,))

    min_abs_vol = np.max(np.abs(vol)) * stable_division_rtol
    unstable_vols = (np.abs(vol) < min_abs_vol)
    rvol = vol.copy()
    rvol[unstable_vols] = min_abs_vol
    printmaybe('stable_division_rtol=', stable_division_rtol, ', num_unstable / ndof_in=', np.sum(unstable_vols), ' / ', ndof_in)

    printmaybe('getting spatially varying mean')
    mu = np.zeros((ndof_in, gdim_out))
    for k in range(gdim_out):
        linear_fct = dof_coords_out[k,:]
        mu_k_base = (apply_At(linear_fct) / mass_lumps_in)
        assert(mu_k_base.shape == (ndof_in,))
        mu[:, k] = mu_k_base / rvol

    printmaybe('getting spatially varying covariance')
    Sigma = np.zeros((ndof_in, gdim_out, gdim_out))
    for k in range(gdim_out):
        for j in range(k + 1):
            quadratic_fct = dof_coords_out[k,:] * dof_coords_out[j,:]
            Sigma_kj_base = apply_At(quadratic_fct) / mass_lumps_in
            assert(Sigma_kj_base.shape == (ndof_in,))
            Sigma[:,k,j] = Sigma_kj_base / rvol - mu[:,k]*mu[:,j]
            Sigma[:,j,k] = Sigma[:,k,j]

    return vol, mu, Sigma


def find_bad_moments(
        vol: np.ndarray, # shape=(ndof_in,)
        mu: np.ndarray, # shape=(ndof_in, gdim_out)
        Sigma: np.ndarray, # shape=(ndof_in, gdim_out, gdim_out)
        dof_coords_in: np.ndarray, # shape=(ndof_in, gdim_in)
        dof_coords_out: np.ndarray, # shape=(ndof_out, gdim_out)
        min_vol_rtol: float=1e-5,
        max_aspect_ratio: float=20.0, # maximum aspect ratio of an ellipsoid
        display: bool=False,
        ) -> typ.Tuple[np.ndarray, # bad_vols, shape=(ndof_in,), dtype=bool
                       np.ndarray, # tiny_Sigmas, shape=(ndof_in,), dtype=bool
                       np.ndarray]: # bad_aspect_Sigma, shape=(ndof_in,), dtype=bool
    assert(0.0 <= min_vol_rtol)
    assert(1.0 < min_vol_rtol)
    assert(max_aspect_ratio >= 1.0)
    ndof_in, gdim_in = dof_coords_out.shape
    ndof_out, gdim_out = dof_coords_out.shape
    assert(Sigma.shape == (ndof_in, gdim_out, gdim_out))

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    printmaybe('postprocessing impulse response moments')
    printmaybe('finding small volumes. min_vol_rtol=', min_vol_rtol)
    if np.all(vol  <= 0.0):
        printmaybe('All impulse response volumes are negative!')
        bad_vols = (vol  <= 0.0)
    else:
        min_vol = np.max(vol) * min_vol_rtol
        bad_vols = (vol < min_vol)
    printmaybe('min_vol_rtol=', min_vol_rtol, ', num_bad_vols / ndof_in=', np.sum(bad_vols), ' / ', ndof_in)

    printmaybe('building dof_coords_out_kdtree')
    dof_coords_out_kdtree = KDTree(dof_coords_out)

    printmaybe('finding nearest neighbors to each out dof')
    closest_distances_out, closest_inds_out = dof_coords_out_kdtree.query(dof_coords_out, 2)
    hh_out = closest_distances_out[:, 1].reshape((ndof_out, 1))
    min_length = np.min(hh_out)

    printmaybe('computing eigenvalue decompositions of all ellipsoid covariances')
    eee0, PP = np.linalg.eigh(Sigma)  # eee0.shape=(N,d), PP.shape=(N,d,d)

    printmaybe('finding ellipsoids that have tiny or negative primary axis lengths')
    tiny_Sigmas = np.any(eee0 <= 0.5*min_length**2, axis=1)
    printmaybe('num_tiny_Sigmas / ndof_in =', np.sum(tiny_Sigmas), ' / ', ndof_in)

    printmaybe('finding ellipsoids that have aspect ratios greater than ', max_aspect_ratio)
    squared_aspect_ratios = np.max(np.abs(eee0), axis=1) / np.min(np.abs(eee0), axis=1)
    bad_aspect_Sigmas = squared_aspect_ratios > max_aspect_ratio**2
    printmaybe('num_bad_aspect_Sigmas / ndof_in =', np.sum(bad_aspect_Sigmas), ' / ', ndof_in)

    return bad_vols, tiny_Sigmas, bad_aspect_Sigmas


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
