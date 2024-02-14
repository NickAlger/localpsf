import numpy as np
import typing as typ


def geometric_sort_helper(
    start:  int,
    stop:   int,
    depth:  int,
    dim:    int,
    points:     np.ndarray, # shape=(num_points, spatial_dimension)
    sort_inds:  np.ndarray, # shape=(num_pts,), dtype=int
) -> None:
    num_pts_local: int = stop - start
    if num_pts_local >= 2:
        axis: int = depth % dim
        local_sort_inds = np.argsort(points[start:stop, axis])
        points[start:stop, :] = points[start+local_sort_inds, :]
        sort_inds[start:stop] = sort_inds[start+local_sort_inds]

        mid: int = start + int(num_pts_local / 2)
        geometric_sort_helper(start, mid,  depth + 1, dim, points, sort_inds)
        geometric_sort_helper(mid,   stop, depth + 1, dim, points, sort_inds)


def geometric_sort(
    points: np.ndarray, # shape=(num_pts, spatial_dimension)
) -> np.ndarray: # sort_inds, len=num_pts
    num_pts, dim = points.shape
    sort_inds: np.ndarray = np.arange(num_pts)
    geometric_sort_helper(0, num_pts, 0, dim, points.copy(), sort_inds) # modifies sort_inds
    return sort_inds


# Example:
run_example = False
if run_example:
    import dolfin as dl
    import matplotlib.pyplot as plt

    mesh = dl.UnitSquareMesh(50,50)
    Vh = dl.FunctionSpace(mesh, 'CG', 1)
    points = Vh.tabulate_dof_coordinates()

    sort_inds = geometric_sort(points)
    sorted_points = points[sort_inds, :]

    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(points[:,0], points[:,1], c=np.arange(Vh.dim()))
    plt.title('point index, unsorted')

    plt.subplot(1,2,2)
    plt.scatter(sorted_points[:,0], sorted_points[:,1], c=np.arange(Vh.dim()))
    plt.title('point index, sorted')

    sigma = 0.25
    N = Vh.dim()
    A = np.zeros((N, N))
    for ii in range(N):
        A[:,ii] = np.exp(-0.5*(np.linalg.norm(points - points[ii,:].reshape((1,-1)), axis=1) / sigma**2)**2)

    A_sorted = A[:, sort_inds][sort_inds, :]

    plt.matshow(A)
    plt.title('A, unsorted')

    plt.matshow(A_sorted)
    plt.title('A, sorted')

    m = int(Vh.dim()/2)
    _, ss, _ = np.linalg.svd(A[:m, m:])
    _, ss_sorted, _ = np.linalg.svd(A_sorted[:m, m:])

    plt.figure()
    plt.semilogy(ss)
    plt.semilogy(ss_sorted)
    plt.legend(['unsorted, sorted'])
    plt.title('off diagonal block singular values')

    plt.show()