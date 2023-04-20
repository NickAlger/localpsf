import numpy as np
import dolfin as dl
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from functools import cached_property
import typing as typ

from .assertion_helpers import *
from nalger_helper_functions import csr_fenics2scipy, csr_scipy2fenics
import hlibpro_python_wrapper as hpro

vec2vec_numpy = typ.Callable[[np.ndarray], np.ndarray]
vec2vec_petsc = typ.Callable[[dl.Vector], dl.Vector]

_RTOL = 1e-10

@dataclass(frozen=True)
class BiLaplacianCovariance:
    '''Bilaplacian covariance operator. Lumped mass matrix. Sparse direct factorization

    C: covariance matrix
    inv(C) = A @ inv(M) @ A
    A = gamma*K + delta * M
    K = finite element stiffness matrix (laplacian-like)
    M = diag(mass_lumps)

    correlation_length = sqrt(8(p - d/2)) sqrt( gamma/delta )
    d: spatial dimension
    p=2: power of A (here there are 2 A's)
    at distance correlation_length away from a source,
    the covariance is 0.1 of its max

    Expensive factorizations performed based on initial gamma0,
    then operations for gamma are performed via appropriate rescaling
    A0 = gamma0*K + delta0*M
    A = (gamma / gamma0) * A0

    C = sqrtC_left_factor  @ sqrtC_right_factor
      = (inv(A) @ sqrt(M)) @ (sqrt(M) @ inv(A))
    '''
    gamma0: float
    correlation_length: float
    K: sps.csr_matrix # stiffness matrix
    mass_lumps: np.ndarray

    def __post_init__(me):
        assert_gt(me.gamma0, 0.0)
        assert_gt(me.correlation_length, 0.0)
        assert_equal(me.K.shape, (me.N, me.N))
        assert_equal(me.mass_lumps.shape, (me.N,))
        assert(np.all(me.mass_lumps > 0.0))

        u = np.random.randn(me.N)
        v = np.random.randn(me.N)
        t1 = np.dot(me.K @ u, v)
        t2 = np.dot(u, me.K @ v)
        assert_le(np.abs(t2-t1), _RTOL * (np.abs(t1)+np.abs(t2)))

        x = np.random.randn(me.N)
        ggg = np.random.rand()

        y1 = me.apply_sqrtC_left_factor(me.apply_sqrtC_right_factor(x, ggg), ggg)
        y2 = me.apply_C(x, ggg)
        assert_le(np.linalg.norm(y2 - y1), _RTOL * np.linalg.norm(y2))

        z1 = me.solve_sqrtC_right_factor(me.solve_sqrtC_left_factor(x, ggg), ggg)
        z2 = me.solve_C(x, ggg)
        assert_le(np.linalg.norm(z2 - z1), _RTOL * np.linalg.norm(z2))

    @property
    def shape(me) -> typ.Tuple[int, int]:
        return me.K.shape

    @property
    def N(me) -> int:
        return me.shape[1]

    def delta(me, gamma: float) -> float:
        assert_gt(gamma, 0)
        return 4. * gamma / (me.correlation_length ** 2)

    @cached_property
    def delta0(me) -> float:
        return me.delta(me.gamma0)

    @cached_property # lumped mass matrix
    def M(me) -> sps.csr_matrix:
        return sps.diags(me.mass_lumps, 0).tocsr()

    @cached_property  # lumped mass matrix
    def invM(me) -> sps.csr_matrix:
        return sps.diags(1.0 / me.mass_lumps, 0).tocsr()

    @cached_property
    def A0(me) -> sps.csr_matrix:
        return me.gamma0 * me.K + me.delta0 * me.M

    def A(me, gamma: float) -> sps.csr_matrix: # A with lumped mass matrix
        assert_gt(gamma, 0)
        return (gamma / me.gamma0) * me.A0
        # return gamma * me.K + me.delta(gamma) * me.ML

    def apply_A(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        assert_gt(gamma, 0)
        return gamma * (me.K @ x) + me.delta(gamma) * (me.M @ x)

    def apply_M(me, x: np.ndarray) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        return x * me.mass_lumps

    def solve_M(me, x: np.ndarray) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        return x / me.mass_lumps

    @cached_property
    def sqrt_mass_lumps(me):
        return np.sqrt(me.mass_lumps)

    def apply_sqrtM(me, x: np.ndarray) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        return x * me.sqrt_mass_lumps

    def solve_sqrtM(me, x: np.ndarray) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        return x / me.sqrt_mass_lumps

    @cached_property
    def solve_A0(me) -> vec2vec_numpy: # lumped mass shifted Bilaplacian
        return spla.factorized(me.A0)

    def solve_A(me, x: np.ndarray, gamma: float) -> np.ndarray: # lumped mass shifted Bilaplacian
        assert_equal(x.shape, (me.N,))
        assert_gt(gamma, 0)
        return (me.gamma0 / gamma) * me.solve_A0(x)

    def apply_C(me, x: np.ndarray, gamma: float) -> np.ndarray: # lumped mass covariance matrix
        assert_equal(x.shape, (me.N,))
        assert_gt(gamma, 0)
        return me.solve_A(me.mass_lumps * me.solve_A(x, gamma), gamma)

    def solve_C(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        assert_gt(gamma, 0)
        return me.apply_A(me.solve_M(me.apply_A(x, gamma)), gamma)
        # return (gamma / me.gamma0)**2 * (me.AL0 @ me.solve_ML(me.AL0 @ x))

    @cached_property
    def invC0(me) -> sps.csr_matrix:
        return me.A0.T @ (me.invM @ me.A0)

    def invC(me, gamma: float) -> sps.csr_matrix:
        assert_gt(gamma, 0)
        return (gamma / me.gamma0)**2 * me.invC0

    def make_invC_hmatrix(me, bct: hpro.BlockClusterTree, gamma: float) -> hpro.HMatrix:
        assert_gt(gamma, 0)
        invC_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(me.invC(gamma), bct)

        x = np.random.randn(me.N)
        y = me.solve_C(x, gamma)
        y2 = invC_hmatrix.matvec(x)
        relerr_invCL_hmatrix = np.linalg.norm(y2 - y) / np.linalg.norm(y)
        print('relerr_invCL_hmatrix=', relerr_invCL_hmatrix)

        return invC_hmatrix

    def apply_sqrtC_left_factor(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        assert_gt(gamma, 0)
        return me.solve_A(me.apply_sqrtM(x), gamma)

    def apply_sqrtC_right_factor(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        assert_gt(gamma, 0)
        return me.apply_sqrtM(me.solve_A(x, gamma))

    def solve_sqrtC_left_factor(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        assert_gt(gamma, 0)
        return me.solve_sqrtM(me.apply_A(x, gamma))

    def solve_sqrtC_right_factor(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        assert_gt(gamma, 0)
        return me.apply_A(me.solve_sqrtM(x), gamma)


def make_bilaplacian_covariance(
        gamma: float,
        correlation_length : float,
        Vh: dl.FunctionSpace,
        mass_lumps: np.ndarray,
        robin_bc: bool=True) -> BiLaplacianCovariance:
    assert_equal(mass_lumps.shape, (Vh.dim(),))
    stiffness_form = dl.inner(dl.grad(dl.TrialFunction(Vh)), dl.grad(dl.TestFunction(Vh))) * dl.dx
    robin_form = dl.inner(dl.TrialFunction(Vh), dl.TestFunction(Vh)) * dl.ds

    # Stiffness matrix without Robin BCs
    K = csr_fenics2scipy(dl.assemble(stiffness_form))
    if robin_bc:
        # Robin BC matrix
        B0 = csr_fenics2scipy(dl.assemble(robin_form))
        K = K + B0 * 2. / (1.42 * correlation_length)

    return BiLaplacianCovariance(gamma, correlation_length, K, mass_lumps)


@dataclass(frozen=True)
class BilaplacianRegularization:
    '''cost(m, a_reg) = -0.5 a_reg (m - mu).T C0^-1 (m - mu))'''
    Cov: BiLaplacianCovariance
    mu: np.ndarray # mean

    def __post__init__(me):
        assert_equal(me.Cov.shape, (me.N, me.N))
        assert_equal(me.mu.shape, (me.N,))

    def gamma(me, a_reg: float) -> float:
        assert_gt(a_reg, 0.0)
        return np.sqrt(a_reg)

    def a_reg(me, gamma: float) -> float:
        assert_gt(gamma, 0.0)
        return gamma**2

    @cached_property
    def N(me):
        return len(me.mu)

    def draw_sample(me, gamma: float) -> np.ndarray:
        assert_gt(gamma, 0.0)
        return me.mu + me.Cov.apply_sqrtC_left_factor(np.random.randn(me.N), gamma)

    def cost(me, m: np.ndarray, a_reg: float) -> float:
        assert_equal(m.shape, (me.N,))
        assert_gt(a_reg, 0.0)
        p = m - me.mu
        return 0.5 * np.dot(p, me.Cov.solve_C(p, me.gamma(a_reg)))

    def gradient(me, m: np.ndarray, a_reg: float) -> np.ndarray:
        assert_equal(m.shape, (me.N,))
        assert_gt(a_reg, 0.0)
        p = m - me.mu
        return me.Cov.solve_C(p, me.gamma(a_reg))

    def apply_hessian(me, x: np.ndarray, m: np.ndarray, a_reg: float) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        assert_equal(m.shape, (me.N,))
        assert_gt(a_reg, 0.0)
        return me.Cov.solve_C(x, me.gamma(a_reg))

    def solve_hessian(me, x: np.ndarray, m: np.ndarray, a_reg: float) -> np.ndarray:
        assert_equal(x.shape, (me.N,))
        assert_equal(m.shape, (me.N,))
        assert_gt(a_reg, 0.0)
        return me.Cov.apply_C(x, me.gamma(a_reg))


