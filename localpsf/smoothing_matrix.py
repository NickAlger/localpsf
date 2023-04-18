import numpy as np
import typing as typ
import scipy.sparse as sps
from scipy.spatial import KDTree

from .assertion_helpers import *


def make_smoothing_matrix(dof_coords: np.ndarray,  # shape=(ndof, gdim)
                          mass_lumps: np.ndarray,  # shape=(ndof,)
                          num_inner_dofs: int=5,  # Gaussian stdev chosen so this many gidpoints are within distance stdev*width_factor
                          width_factor: float=0.5,
                          workers=1,  # Number of workers to use for parallel processing. If -1 is given all CPU threads are used.
                          display=False,
                          ) -> sps.csr_matrix:
    '''Make matrix that smooths a function by local convolution with Gaussian. Gaussian width varies based on local grid.

    In:
        import numpy as np
        import matplotlib.pyplot as plt
        ndofs = 2000
        xx = np.random.randn(ndofs, 2)
        mass_lumps = np.ones(ndofs)
        S1 = make_smoothing_matrix(xx, mass_lumps, num_dofs_in_stdev=1, display=True)
        S5 = make_smoothing_matrix(xx, mass_lumps, num_dofs_in_stdev=5, display=True)
        S20 = make_smoothing_matrix(xx, mass_lumps, num_dofs_in_stdev=10, display=True)
        u = np.random.randn(ndofs)
        plt.figure()
        plt.scatter(xx[:,0], xx[:,1], c=u)
        plt.title('unsmoothed')
        plt.figure()
        plt.scatter(xx[:,0], xx[:,1], c=S1 @ u)
        plt.title('smoothed (1)')
        plt.figure()
        plt.scatter(xx[:,0], xx[:,1], c=S5 @ u)
        plt.title('smoothed (5)')
        plt.figure()
        plt.scatter(xx[:,0], xx[:,1], c=S20 @ u)
        plt.title('smoothed (20)')
    Outputs pictures noise with progressively more smoothing
    '''
    ndof, gdim = dof_coords.shape
    assert_equal(mass_lumps.shape, (ndof,))
    assert_ge(num_inner_dofs, 1)

    num_dofs_in_neighborhood = num_inner_dofs * 5 # could do this more rigorously

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    printmaybe('making smoothing matrix')
    printmaybe('Gaussian: ', num_inner_dofs, ' gidpoints are within distance stdev / ', width_factor)

    printmaybe('building dof_coords kdtree')
    kdtree = KDTree(dof_coords)

    printmaybe('finding', num_dofs_in_neighborhood, ' nearest neighbors to each out dof')
    distances, inds = kdtree.query(dof_coords, num_dofs_in_neighborhood + 1, workers=workers) # +1 to include self
    lengthscales = distances[:, num_inner_dofs + 1]

    gauss = np.exp(-0.5 * (distances / (width_factor*lengthscales).reshape((-1,1)))**2 ) # shape=(ndof, num_neighbors + 1)

    row_inds = np.outer(np.arange(ndof), np.ones(num_dofs_in_neighborhood + 1)).reshape(-1) # e.g., [0,0,0,0,1,1,1,1,2,2,2,2...]
    col_inds = inds.reshape(-1)
    data = gauss.reshape(-1)

    gauss_matrix0 = sps.csr_matrix((data, (row_inds, col_inds)), shape=(ndof, ndof)) # Rows of G are un-normalized gaussian functions evaluated at gridpoints
    normalization_constants = gauss_matrix0 @ mass_lumps
    gauss_matrix = sps.diags(1.0/normalization_constants, 0, shape=(ndof,ndof)).tocsr() @ gauss_matrix0 # normalize rows

    mass_matrix = sps.diags(mass_lumps, 0, shape=(ndof, ndof)).tocsr()
    smoothing_matrix = gauss_matrix @ mass_matrix

    return smoothing_matrix