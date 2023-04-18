import numpy as np
import dolfin as dl
import typing as typ
import scipy.sparse as sps
from scipy.spatial import KDTree
from dataclasses import dataclass
from functools import cached_property
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from assertion_helpers import *
from .sample_point_batches import choose_one_sample_point_batch
from .smoothing_matrix import make_smoothing_matrix

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import make_mass_matrix, dlfct2array, plot_ellipse


@dataclass(frozen=True)
class CG1Space:
    vertices: np.ndarray # shape=(ndof, gdim), dtype=float
    cells: np.ndarray # shape=(ndof, gdim+1), dtype=float
    mass_lumps: np.ndarray # shape=(ndof,), dtype=int

    def __post_init__(me):
        assert_equal(me.vertices.shape, (me.ndof, me.gdim))
        assert_equal(me.vertices.dtype, float)
        assert_equal(me.cells.shape, (me.ndof, me.gdim+1))
        assert_equal(me.cells.dtype, int)
        assert_equal(me.mass_lumps.shape, (me.ndof,))
        assert_equal(me.mass_lumps.dtype, float)
        assert(np.all(me.mass_lumps > 0.0))

    @cached_property
    def ndof(me) -> int:
        return me.vertices.shape[0]

    @cached_property
    def gdim(me) -> int:
        return me.vertices.shape[1]

    @cached_property
    def kdtree(me) -> KDTree:
        return KDTree(me.vertices)


@dataclass(frozen=True)
class ImpulseMoments:
    vol: np.ndarray # shape=(ndof_in,), dtype=float
    mu: np.ndarray # shape=(ndof_in, gdim_out), dtype=float
    Sigma: np.ndarray # shape=(ndof_in, gdim_out, gdim_out), dtype=float

    def __post_init__(me):
        assert_equal(me.vol.shape, (me.ndof_in,))
        assert_equal(me.mu.shape, (me.ndof_in, me.gdim_out))
        assert_equal(me.Sigma.shape, (me.ndof_in, me.gdim_out, me.gdim_out))
        assert_equal(me.vol.dtype, float)
        assert_equal(me.mu.dtype, float)
        assert_equal(me.Sigma.dtype, float)

    @cached_property
    def ndof_in(me) -> int:
        return len(me.vol)

    @cached_property
    def gdim_out(me) -> int:
        return me.mu.shape[1]


def compute_impulse_response_moments(
        apply_At: typ.Callable[[np.ndarray], np.ndarray],  # V_out -> V_in
        V_in: CG1Space,
        V_out: CG1Space,
        stable_division_rtol: float=1.0e-10,
        display=False,
        ) -> ImpulseMoments:
    '''Computes spatially varying impulse response volume, mean, and covariance for A : V_in -> V_out
    '''
    assert_le(0.0, stable_division_rtol)
    assert_lt(stable_division_rtol, 1.0)

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    printmaybe('getting spatially varying volume')
    constant_fct = np.ones(V_out.ndof)
    vol = apply_At(constant_fct) / V_in.mass_lumps
    assert_equal(vol.shape, (V_in.ndof,))

    min_abs_vol = np.max(np.abs(vol)) * stable_division_rtol
    unstable_vols = (np.abs(vol) < min_abs_vol)
    rvol = vol.copy()
    rvol[unstable_vols] = min_abs_vol
    printmaybe('stable_division_rtol=', stable_division_rtol, ', num_unstable / ndof_in=', np.sum(unstable_vols), ' / ', ndof_in)

    printmaybe('getting spatially varying mean')
    mu = np.zeros((V_in.ndof, V_out.gdim))
    for k in range(V_out.gdim):
        linear_fct = V_out.vertices[k,:]
        mu_k_base = (apply_At(linear_fct) / V_in.mass_lumps)
        assert_equal(mu_k_base.shape, (V_in.ndof,))
        mu[:, k] = mu_k_base / rvol

    printmaybe('getting spatially varying covariance')
    Sigma = np.zeros((V_in.ndof, V_out.gdim, V_out.gdim))
    for k in range(V_out.gdim):
        for j in range(k + 1):
            quadratic_fct = V_out.vertices[k,:] * V_out.vertices[j,:]
            Sigma_kj_base = apply_At(quadratic_fct) / V_in.mass_lumps
            assert_equal(Sigma_kj_base.shape, (V_in.ndof,))
            Sigma[:,k,j] = Sigma_kj_base / rvol - mu[:,k]*mu[:,j]
            Sigma[:,j,k] = Sigma[:,k,j]

    impulse_moments = ImpulseMoments(vol, mu, Sigma)
    assert_equal(impulse_moments.ndof_in, V_in.ndof)
    assert_equal(impulse_moments.gdim_out, V_out.gdim)

    return impulse_moments


def find_bad_moments(
        IM: ImpulseMoments,
        V_in: CG1Space,
        V_out: CG1Space,
        min_vol_rtol: float=1e-5,
        max_aspect_ratio: float=20.0, # maximum aspect ratio of an ellipsoid
        display: bool=False,
        ) -> typ.Tuple[np.ndarray, # bad_vols, shape=(ndof_in,), dtype=bool
                       np.ndarray, # tiny_Sigmas, shape=(ndof_in,), dtype=bool
                       np.ndarray]: # bad_aspect_Sigma, shape=(ndof_in,), dtype=bool
    assert_equal(IM.ndof_in, V_in.ndof)
    assert_equal(IM.gdim_out, V_out.gdim)
    assert_le(0.0, min_vol_rtol)
    assert_lt(1.0, min_vol_rtol)
    assert_ge(max_aspect_ratio, 1.0)

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    printmaybe('postprocessing impulse response moments')
    printmaybe('finding small volumes. min_vol_rtol=', min_vol_rtol)
    if np.all(IM.vol  <= 0.0):
        printmaybe('All impulse response volumes are negative!')
        bad_vols = (IM.vol  <= 0.0)
    else:
        min_vol = np.max(IM.vol) * min_vol_rtol
        bad_vols = (IM.vol < min_vol)
    printmaybe('min_vol_rtol=', min_vol_rtol, ', num_bad_vols / ndof_in=', np.sum(bad_vols), ' / ', V_in.ndof)

    printmaybe('finding nearest neighbors to each out dof')
    closest_distances_out, closest_inds_out = V_out.kdtree.query(V_out.vertices, 2)
    hh_out = closest_distances_out[:, 1].reshape((V_out.ndof, 1))
    min_length = np.min(hh_out)

    printmaybe('computing eigenvalue decompositions of all ellipsoid covariances')
    eee0, PP = np.linalg.eigh(IM.Sigma)  # eee0.shape=(N,d), PP.shape=(N,d,d)

    printmaybe('finding ellipsoids that have tiny or negative primary axis lengths')
    tiny_Sigmas = np.any(eee0 <= 0.5*min_length**2, axis=1)
    printmaybe('num_tiny_Sigmas / ndof_in =', np.sum(tiny_Sigmas), ' / ', V_in.ndof)

    printmaybe('finding ellipsoids that have aspect ratios greater than ', max_aspect_ratio)
    squared_aspect_ratios = np.max(np.abs(eee0), axis=1) / np.min(np.abs(eee0), axis=1)
    bad_aspect_Sigmas = squared_aspect_ratios > max_aspect_ratio**2
    printmaybe('num_bad_aspect_Sigmas / ndof_in =', np.sum(bad_aspect_Sigmas), ' / ', V_in.ndof)

    return bad_vols, tiny_Sigmas, bad_aspect_Sigmas


@dataclass(frozen=True)
class ImpulseResponseBatches:
    apply_A: typ.Callable[[np.ndarray], np.ndarray]     # ndof_in -> ndof_out
    IM: ImpulseMoments
    V_in: CG1Space
    V_out: CG1Space

    candidate_inds: typ.List[int]
    cpp_object: hpro.hpro_cpp.ImpulseResponseBatches

    def __post_init__(me):
        assert_equal(me.IM.ndof_in, me.V_in.ndof)
        assert_equal(me.IM.gdim_out, me.V_out.gdim)

    def add_one_sample_point_batch(me) -> typ.List[int]:
        qq = np.array(me.V_in.vertices[me.candidate_inds, :].T, order='F')
        if me.num_sample_points > 0:
            dd = me.cpp_object.kdtree.query(qq,1)[1][0]
            candidate_inds_ordered_by_distance = np.array(me.candidate_inds)[np.argsort(dd)]
        else:
            candidate_inds_ordered_by_distance = np.array(me.candidate_inds)

        if len(candidate_inds_ordered_by_distance) < 1:
            print('no points left to choose')
            return []

        new_inds = choose_one_sample_point_batch(me.IM.mu, me.IM.Sigma, me.tau,
                                                 candidate_inds_ordered_by_distance, randomize=False)

        dirac_comb_dual_vector = np.zeros(me.V_in.ndof)
        dirac_comb_dual_vector[new_inds] = 1.0 / me.IM.vol[new_inds]

        phi = me.apply_A(dirac_comb_dual_vector / me.V_in.mass_lumps) / me.V_out.mass_lumps

        me.cpp_object.add_batch(new_inds, phi, True)

        new_candidate_inds = list(np.setdiff1d(me.candidate_inds, new_inds))
        me.candidate_inds.clear()
        me.candidate_inds += new_candidate_inds

        return new_inds

    @property
    def sample_points(me):
        return np.array(me.cpp_object.sample_points)

    @property
    def sample_vol(me):
        return np.array(me.cpp_object.sample_vol)

    @property
    def sample_mu(me):
        return np.array(me.cpp_object.sample_mu)

    @property
    def sample_Sigma(me):
        return np.array(me.cpp_object.sample_Sigma)

    @property
    def point2batch(me):
        return np.array(me.cpp_object.point2batch)

    @property
    def batch2point_start(me):
        return np.array(me.cpp_object.batch2point_start)

    @property
    def batch2point_stop(me):
        return np.array(me.cpp_object.batch2point_stop)

    @property
    def psi_batches(me):
        return np.array(me.cpp_object.psi_batches)

    @property
    def tau(me):
        return me.cpp_object.tau

    def update_tau(me, new_tau: float):
        assert_gt(new_tau, 0.0)
        me.cpp_object.tau = new_tau

    @property
    def num_neighbors(me):
        return me.cpp_object.num_neighbors

    def update_num_neighbors(me, new_num_neighbors: int):
        assert_gt(new_num_neighbors, 0)
        me.cpp_object.num_neighbors = new_num_neighbors

    @property
    def num_sample_points(me):
        return me.cpp_object.num_pts()

    @property
    def num_batches(me):
        return me.cpp_object.num_batches()


def make_impulse_response_batches_simplified(
        apply_A: typ.Callable[[np.ndarray], np.ndarray], # ndof_in -> ndof_out
        IM: ImpulseMoments,
        bad_inds: np.ndarray, # shape=(ndof_in,), dtype=bool
        V_in: CG1Space,
        V_out: CG1Space,
        num_initial_batches: int = 5,
        tau: float = 3.0,
        num_neighbors: int = 10,
        max_candidate_points: int = None
) -> ImpulseResponseBatches:
    assert_equal(IM.ndof_in, V_in.ndof)
    assert_equal(IM.gdim_out, V_out.gdim)
    assert_equal(bad_inds.shape, V_in.ndof)
    assert_equal(bad_inds.dtype, bool)
    assert_gt(tau, 0.0)
    assert_gt(num_neighbors, 0)
    assert_gt(max_candidate_points, 0)

    print('Preparing c++ object')
    cpp_object = hpro.hpro_cpp.ImpulseResponseBatches(
        V_in.vertices, V_in.cells,
        IM.vol, IM.mu, IM.Sigma,
        num_neighbors, tau)

    candidate_inds = list(np.argwhere(np.logical_not(bad_inds)).reshape(-1))

    if max_candidate_points is not None:
        candidate_inds = list(np.random.permutation(len(candidate_inds))[:max_candidate_points])

    IRB = ImpulseResponseBatches(apply_A, IM, V_in, V_out, candidate_inds, cpp_object)

    print('Building initial sample point batches')
    for _ in tqdm(range(num_initial_batches)):
        IRB.add_one_sample_point_batch()

    return IRB


def visualize_impulse_response_batch(
        IRB: ImpulseResponseBatches,
        b: int,
        V_out_fenics: dl.FunctionSpace
) -> None:
    assert_equal(IRB.V_out.ndof, V_out_fenics.dim())
    if (0 <= b) and (b < IRB.num_batches):
        phi = dl.Function(V_out_fenics)
        phi.vector()[:] = IRB.psi_batches[b]

        start = IRB.batch2point_start[b]
        stop = IRB.batch2point_stop[b]
        pp = IRB.sample_points[start:stop, :]
        mu_batch = IRB.sample_mu[start:stop, :]
        Sigma_batch = IRB.sample_Sigma[start:stop, :, :]

        plt.figure()

        cm = dl.plot(phi)
        plt.colorbar(cm)

        plt.scatter(pp[:, 0], pp[:, 1], c='r', s=2)
        plt.scatter(mu_batch[:, 0], mu_batch[:, 1], c='k', s=2)

        for k in range(mu_batch.shape[0]):
            plot_ellipse(mu_batch[k, :], Sigma_batch[k, :, :], n_std_tau=IRB.tau,
                         facecolor='none', edgecolor='k', linewidth=1)

        plt.title('Impulse response batch '+str(b))
    else:
        print('bad batch number. num_batches=', IRB.num_batches, ', b=', b)


@dataclass(frozen=True)
class PSFKernel:
    IRB: ImpulseResponseBatches
    cpp_object: hpro.hpro_cpp.ProductConvolutionKernelRBFColsOnly

    def __post_init__(me):
        assert_equal(me.col_coords.shape, (me.V_in.ndof, me.V_in.gdim))
        assert_equal(me.row_coords.shape, (me.V_out.ndof, me.V_out.gdim))

    def __call__(me, yy, xx):
        if len(xx.shape) == 1 and len(yy.shape) == 1:
            return me.cpp_object.eval_integral_kernel(yy, xx)
        else:
            return me.cpp_object.eval_integral_kernel_block(yy, xx)

    def __getitem__(me, ii_jj):
        ii, jj = ii_jj
        yy = np.array(me.row_coords[ii,:].T, order='F')
        xx = np.array(me.col_coords[jj, :].T, order='F')
        return me.__call__(yy, xx)

    def build_hmatrix(me, bct, tol=1e-6):
        hmatrix_cpp_object = hpro.hpro_cpp.build_hmatrix_from_coefffn( me.cpp_object, bct.cpp_object, tol )
        return hpro.HMatrix(hmatrix_cpp_object, bct)

    @cached_property
    def V_in(me) -> CG1Space:
        return me.IRB.V_in

    @cached_property
    def V_out(me) -> CG1Space:
        return me.IRB.V_out

    @cached_property
    def row_coords(me) -> np.ndarray:
        return np.array(me.cpp_object.row_coords)

    @cached_property
    def col_coords(me) -> np.ndarray:
        return np.ndarray(me.cpp_object.col_coords)

    @property
    def mean_shift(me):
        return me.cpp_object.mean_shift

    @mean_shift.setter
    def mean_shift(me, new_mean_shift_bool : bool):
        me.cpp_object.mean_shift = new_mean_shift_bool

    @property
    def vol_preconditioning(me):
        return me.cpp_object.vol_preconditioning

    @vol_preconditioning.setter
    def vol_preconditioning(me, new_vol_preconditioning_bool : bool):
        me.cpp_object.vol_preconditioning = new_vol_preconditioning_bool

    @property
    def tau(me):
        return me.IRB.tau

    def update_tau(me, new_tau: float):
        assert_gt(new_tau, 0.0)
        me.IRB.update_tau(new_tau)

    @property
    def num_neighbors(me):
        return me.IRB.num_neighbors

    def update_num_neighbors(me, new_num_neighbors: int):
        assert_gt(new_num_neighbors, 0)
        me.IRB.update_num_neighbors(new_num_neighbors)

    @cached_property
    def shape(me) -> typ.Tuple[int, int]:
        return (me.V_out.ndof, me.V_in.ndof)


def make_product_convolution_kernel(col_batches: ImpulseResponseBatches) -> PSFKernel:
    cpp_object = hpro.hpro_cpp.ProductConvolutionKernelRBFColsOnly(
        col_batches.cpp_object, col_batches.V_in.vertices, col_batches.V_out.vertices)
    return PSFKernel(col_batches, cpp_object)


def apply_sandwiched_operator(x: np.ndarray,
                              apply_B: typ.Callable[[np.ndarray], np.ndarray],
                              A: sps.csr_matrix, C: sps.csr_matrix) -> np.ndarray:
    return A @ apply_B(C @ x)


@dataclass(frozen=True)
class PSFObject:
    psf_kernel: PSFKernel
    unmodified_impulse_moments: ImpulseMoments
    bad_vols: np.ndarray
    tiny_Sigmas: np.ndarray
    bad_aspect_Sigmas: np.ndarray

    smoothing_matrix_in: sps.csr_matrix
    smoothing_matrix_out: sps.csr_matrix

    apply_operator: typ.Callable[[np.ndarray], np.ndarray]  # smoothed operator
    apply_operator_transpose: typ.Callable[[np.ndarray], np.ndarray]  # smoothed operator

    def __post_init__(me):
        assert_equal(me.unmodified_impulse_moments.ndof_in, me.V_in.ndof)
        assert_equal(me.unmodified_impulse_moments.gdim_out, me.V_out.gdim)
        assert_equal(me.bad_vols.shape, (me.V_in.ndof,))
        assert_equal(me.tiny_Sigmas.shape, (me.V_in.ndof,))
        assert_equal(me.bad_aspect_Sigmas.shape, (me.V_in.ndof,))
        assert_equal(me.bad_vols.dtype, bool)
        assert_equal(me.tiny_Sigmas.dtype, bool)
        assert_equal(me.bad_aspect_Sigmas.dtype, bool)

    def apply_smoothed_operator(me, u: np.ndarray) -> np.ndarray:
        return me.smoothing_matrix_out.T @ me.apply_operator(me.smoothing_matrix_in @ u)

    def apply_smoothed_operator_transpose(me, u: np.ndarray) -> np.ndarray:
        return me.smoothing_matrix_in.T @ me.apply_operator_transpose(me.smoothing_matrix_out @ u)

    @cached_property
    def impulse_response_batches(me) -> ImpulseResponseBatches:
        return me.psf_kernel.IRB

    @cached_property
    def V_in(me) -> CG1Space:
        return me.impulse_response_batches.V_in

    @cached_property
    def V_out(me) -> CG1Space:
        return me.impulse_response_batches.V_out

    def add_batch(me, num_batches: int=1, recompute: bool=True):
        for _ in range(num_batches):
            me.impulse_response_batches.add_one_sample_point_batch()

        if recompute:
            me.compute_hmatrices()

    def compute_hmatrices(me, bct: hpro.BlockClusterTree, hmatrix_rtol: float=1e-7):
        assert_gt(hmatrix_rtol, 0.0)
        assert_lt(hmatrix_rtol, 1.0)
        assert_gt(me.impulse_response_batches.num_batches, 0)

        print('Building kernel hmatrix')
        kernel_hmatrix = me.psf_kernel.build_hmatrix(bct, tol=hmatrix_rtol)

        print('Computing hmatrix = diag(mass_lumps_out) @ kernel_hmatrix * diag(mass_lumps_in)')
        hmatrix = kernel_hmatrix.copy()
        hmatrix.mul_diag_left(me.V_out.mass_lumps)
        hmatrix.mul_diag_right(me.V_in.mass_lumps)

        return hmatrix, kernel_hmatrix


def make_psf(
        apply_operator: typ.Callable[[np.ndarray], np.ndarray],
        apply_operator_transpose: typ.Callable[[np.ndarray], np.ndarray],
        V_in: CG1Space,
        V_out: CG1Space,
        min_vol_rtol: float=1e-5,
        max_aspect_ratio: float=20.0,
        smoother_num_dofs_in_stdev: int=5,
        num_initial_batches: int = 5,
        tau: float = 3.0,
        num_neighbors: int = 10,
        max_candidate_points: int = None,
        display: bool=False,
) -> PSFObject:
    assert_ge(num_initial_batches, 0)
    assert_gt(min_vol_rtol, 0.0)
    assert_gt(smoother_num_dofs_in_stdev, 1)
    assert_gt(tau, 0.0)
    assert_gt(num_neighbors, 0)

    if max_candidate_points is None:
        max_candidate_points = V_in.ndof
    else:
        assert_gt(max_candidate_points, 0)
    max_candidate_points = np.min([max_candidate_points, V_in.ndof])

    S_in: sps.csr_matrix = make_smoothing_matrix(
        V_in.vertices, V_in.mass_lumps,
        smoother_num_dofs_in_stdev=smoother_num_dofs_in_stdev,
        display=display)

    S_out: sps.csr_matrix = make_smoothing_matrix(
        V_out.vertices, V_out.mass_lumps,
        smoother_num_dofs_in_stdev=smoother_num_dofs_in_stdev,
        display=display)

    apply_A_smooth = lambda u: S_out.T @ (apply_operator(S_in @ u))
    apply_AT_smooth = lambda u: S_in.T @ (apply_operator_transpose(S_out @ u))

    IM: ImpulseMoments = compute_impulse_response_moments(
        apply_AT_smooth, V_in, V_out, display=display)

    bad_vols, tiny_Sigmas, bad_aspect_Sigmas = find_bad_moments(
        IM, V_in, V_out, min_vol_rtol=min_vol_rtol,
        max_aspect_ratio=max_aspect_ratio, display=display)

    # Modify bad moments
    modified_vol = IM.vol.copy()
    modified_vol[bad_vols] = 0.0

    modified_mu = IM.mu.copy()

    bad_Sigmas = np.logical_and(tiny_Sigmas, bad_aspect_Sigmas)
    modified_Sigma = IM.Sigma.copy()
    modified_Sigma[bad_Sigmas,:,:] = np.eye(V_out.gdim).reshape((1, V_out.gdim, V_out.gdim))

    bad_inds = np.logical_and(bad_vols, bad_Sigmas)

    modified_IM = ImpulseMoments(modified_vol, modified_mu, modified_Sigma)

    col_batches: ImpulseResponseBatches = make_impulse_response_batches_simplified(
        apply_A_smooth, modified_IM, bad_inds, V_in, V_out,
        num_initial_batches=num_initial_batches, tau=tau,
        num_neighbors=num_neighbors, max_candidate_points=max_candidate_points)

    psf_kernel: PSFKernel = make_product_convolution_kernel(col_batches)

    psf_object = PSFObject(
        psf_kernel, IM, bad_vols, tiny_Sigmas, bad_aspect_Sigmas,
        S_in, S_out, apply_operator, apply_operator_transpose)

    return psf_object


def mesh_vertices_and_cells_in_CG1_dof_order(V: dl.FunctionSpace) -> typ.Tuple[np.ndarray, np.ndarray]:
    mesh = V.mesh()
    dof_coords = V.tabulate_dof_coordinates()
    vertex2dof = dl.vertex_to_dof_map(V)
    dof2vertex = dl.dof_to_vertex_map(V)

    vertices_bad_order = mesh.coordinates()
    cells_bad_order = mesh.cells()

    vertices_good_order = vertices_bad_order[dof2vertex, :]

    assert_lt(np.linalg.norm(dof_coords - vertices_good_order),
              1e-10 * np.linalg.norm(dof_coords))

    cells_good_order = vertex2dof[cells_bad_order]

    vv1 = vertices_good_order[cells_good_order,:]
    vv2 = vertices_bad_order[cells_bad_order, :]

    assert_lt(np.linalg.norm(vv2 - vv1), 1e-10 * np.linalg.norm(vv2))

    return vertices_good_order, cells_good_order


def make_psf_fenics(
        apply_operator: typ.Callable[[np.ndarray], np.ndarray],
        apply_operator_transpose: typ.Callable[[np.ndarray], np.ndarray],
        V_in: CG1Space,
        V_out: CG1Space,
        min_vol_rtol: float=1e-5,
        max_aspect_ratio: float=20.0,
        smoother_num_dofs_in_stdev: int=5,
        num_initial_batches: int = 5,
        tau: float = 3.0,
        num_neighbors: int = 10,
        max_candidate_points: int = None,
        display: bool=False,
) -> PSFObject:
    pass