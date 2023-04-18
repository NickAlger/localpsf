import numpy as np
import typing as typ
import scipy.sparse as sps
from scipy.spatial import KDTree


def local_mesh_size_maxmin(dof_coords: np.ndarray,  # shape=(ndof, gdim),
                           kdtree: KDTree=None,
                           num_neighbors: int=5) -> np.ndarray:
    '''local mesh size near each mesh dof location.
    '''
    assert(num_neighbors > 0)
    ndof, gdim = dof_coords.shape
    num_neighbors = np.min([num_neighbors, ndof]) # in case there are more neighbors than gridpoints
    if kdtree is None:
        kdtree = KDTree(dof_coords)

    distances, inds = kdtree.query(dof_coords, num_neighbors)
    if num_neighbors == 1:
        return distances.reshape(-1)

    d_min = distances[:,1].reshape(-1) # nearest distance to any other dof from current dof
    d_maxmin = np.max(d_min[inds], axis=1).reshape(-1) # maximum nearest distance among neighbors
    return d_maxmin


def make_smoothing_matrix(dof_coords: np.ndarray,  # shape=(ndof, gdim)
                          mass_lumps: np.ndarray,  # shape=(ndof,)
                          num_neighbors: int=25, # could probably be chosen more rigirously without
                          width_factor: float=1.0, # Gaussian stdev = width_factor*local_mesh_size
                          kdtree: KDTree=None,
                          display=False,
                          ) -> sps.csr_matrix:
    '''Make matrix that smooths a function by local convolution with Gaussian. Gaussian width varies based on local grid.

    In:
        import numpy as np
        import matplotlib.pyplot as plt
        ndofs = 2000
        xx = np.random.randn(ndofs, 2)
        mass_lumps = np.ones(ndofs)
        S1 = make_smoothing_matrix(xx, mass_lumps, width_factor=1.0, display=True)
        S2 = make_smoothing_matrix(xx, mass_lumps, width_factor=2.0, display=True)
        S5 = make_smoothing_matrix(xx, mass_lumps, width_factor=5.0, display=True)
        u = np.random.randn(ndofs)

        plt.figure()
        plt.scatter(xx[:,0], xx[:,1], c=u)
        plt.title('unsmoothed')
        plt.colorbar()

        plt.figure()
        plt.scatter(xx[:,0], xx[:,1], c=S1 @ u)
        plt.title('smoothed (1)')
        plt.colorbar()

        plt.figure()
        plt.scatter(xx[:,0], xx[:,1], c=S2 @ u)
        plt.title('smoothed (2)')
        plt.colorbar()

        plt.figure()
        plt.scatter(xx[:,0], xx[:,1], c=S5 @ u)
        plt.title('smoothed (5)')
        plt.colorbar()
    Outputs pictures noise with progressively more smoothing
    '''
    ndof, gdim = dof_coords.shape
    assert(mass_lumps.shape == (ndof,))
    assert(num_neighbors >= 1)

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    if kdtree is None:
        printmaybe('building dof_coords kdtree')
        kdtree = KDTree(dof_coords)

    printmaybe('making smoothing matrix')
    printmaybe('Gaussian: ', num_neighbors, ' gidpoints are within distance stdev / ', width_factor)

    printmaybe('finding', num_neighbors, ' nearest neighbors to each out dof')
    distances, inds = kdtree.query(dof_coords, num_neighbors + 1) # +1 to include self
    local_mesh_size = local_mesh_size_maxmin(dof_coords, kdtree=kdtree)

    gauss = np.exp(-0.5 * (distances / (width_factor*local_mesh_size).reshape((-1,1)))**2 ) # shape=(ndof, num_neighbors + 1)

    row_inds = np.outer(np.arange(ndof), np.ones(num_neighbors + 1)).reshape(-1) # e.g., [0,0,0,0,1,1,1,1,2,2,2,2...]
    col_inds = inds.reshape(-1)
    data = gauss.reshape(-1)

    gauss_matrix0 = sps.csr_matrix((data, (row_inds, col_inds)), shape=(ndof, ndof)) # Rows of G are un-normalized gaussian functions evaluated at gridpoints
    normalization_constants = gauss_matrix0 @ mass_lumps
    gauss_matrix = sps.diags(1.0/normalization_constants, 0, shape=(ndof,ndof)).tocsr() @ gauss_matrix0 # normalize rows

    mass_matrix = sps.diags(mass_lumps, 0, shape=(ndof, ndof)).tocsr()
    smoothing_matrix = gauss_matrix @ mass_matrix

    return smoothing_matrix