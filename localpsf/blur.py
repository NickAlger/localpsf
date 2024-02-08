import numpy as np
import dolfin as dl
from localpsf.geometric_sort import geometric_sort


# Frog function
def frog_function(
    xx: np.ndarray, # shape=(N, 2), N=nx*ny
    vol: float,
    mu: np.ndarray, # shape=(2,)
    Sigma: np.ndarray, # shape=(2, 2)
    a: float, # how much negative to include. a=0: gaussian
    use_bump: bool = True,
) -> np.ndarray: # shape=(N,)
    assert(xx.shape[1] == 2)
    N = xx.shape[0]
    assert(Sigma.shape == (2, 2))
    assert(mu.shape == (2,))

    # theta = np.pi * (mu[0] + mu[1])
    theta = (np.pi / 2.0) * (mu[0] + mu[1])
    Rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    bump = mu[0]*(1 - mu[0])*mu[1]*(1-mu[1]) if use_bump else 0.0

    pp = (xx - mu.reshape((1,2))) @ Rot_matrix.T #@ Rot_matrix.T
    inv_Sigma = np.linalg.inv(Sigma)
    t_squared_over_sigma_squared = np.einsum('ai,ai->a', pp, np.einsum('ij,aj->ai', inv_Sigma, pp))
    G = vol / (2.0 * np.pi * np.sqrt(np.linalg.det(Sigma))) * np.exp(-0.5 * t_squared_over_sigma_squared)

    cos_x_over_sigma = np.cos(pp[:,0] / (np.sqrt(Sigma[0,0]) / 2.0))
    sin_y_over_sigma = np.sin(pp[:,1] / (np.sqrt(Sigma[1,1]) / 2.0))
    return bump * (1.0 + a * cos_x_over_sigma * sin_y_over_sigma) * G


def frog_setup(
        nx:             int     = 63,
        length_scaling: float   = 1.0,
        a:              float   = 1.0,
):
    ny = nx
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh = dl.FunctionSpace(mesh, 'CG', 1)
    dof_coords = Vh.tabulate_dof_coordinates()

    vol = 1.0

    Sigma = length_scaling * np.array([[0.01, 0.0], [0.0, 0.0025]]) # 0.25*np.array([[0.01, 0.0], [0.0, 0.0025]])

    phi_function = lambda yy, x: frog_function(yy, vol, x, Sigma, a)

    Ker = np.zeros((Vh.dim(), Vh.dim()))
    for ii in range(Vh.dim()):
        Ker[:,ii] = phi_function(dof_coords, dof_coords[ii,:])

    mass_lumps = dl.assemble(dl.Constant(1.0) * dl.TestFunction(Vh) * dl.dx)[:].copy()

    H = mass_lumps.reshape((-1, 1)) * Ker * mass_lumps.reshape((1, -1))

    kdtree_sort_inds = geometric_sort(dof_coords)

    return Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds

nearest_ind_func = lambda yy, x: np.argmin(np.linalg.norm(yy - x.reshape((1, -1)), axis=1))
