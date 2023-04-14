import numpy as np
import dolfin as dl
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from functools import cached_property
import typing as typ

from nalger_helper_functions import csr_fenics2scipy, csr_scipy2fenics
import hlibpro_python_wrapper as hpro

vec2vec_numpy = typ.Callable[[np.ndarray], np.ndarray]
vec2vec_petsc = typ.Callable[[dl.Vector], dl.Vector]

_RTOL = 1e-10

@dataclass(frozen=True)
class BiLaplacianCovarianceScipy:
    # ==== SET UP PRIOR DISTRIBUTION ====
    # Recall from Daon Stadler 2017
    # "The covariance function of the free-space
    # operator has a characteristic length of
    # sqrt(8(p - d / 2)) sqrt( gamma/delta )
    # meaning that that distance away from a source
    # x, the covariance decays to 0.1 of its
    # maximal value
    # d - dimension of problem
    # A^(-p) - operator, A laplacian-like
    gamma0: float
    correlation_length: float
    K: sps.csr_matrix # stiffness matrix
    M: sps.csr_matrix # mass matrix
    mass_lumps: np.ndarray

    def __post_init__(me):
        assert(me.gamma0 > 0.0)
        assert(me.correlation_length > 0.0)
        assert(me.K.shape == (me.N, me.N))
        assert(me.M.shape == (me.N, me.N))
        assert(me.mass_lumps.shape == (me.N,))
        u = np.random.randn(me.N)
        v = np.random.randn(me.N)
        t1 = np.dot(me.K @ u, v)
        t2 = np.dot(u, me.K @ v)
        assert(np.abs(t2-t1) <= _RTOL * (np.abs(t1)+np.abs(t2)))
        t1 = np.dot(me.M @ u, v)
        t2 = np.dot(u, me.M @ v)
        assert (np.abs(t2 - t1) <= _RTOL * (np.abs(t1) + np.abs(t2)))
        x = np.random.randn(me.N)
        x2 = me.solve_M(me.M @ x)
        assert(np.linalg.norm(x2 - x) < _RTOL * np.linalg.norm(x))
        ggg = np.random.rand()
        x3 = me.solve_C(me.apply_C(x, ggg), ggg)
        assert(np.linalg.norm(x3 - x) < _RTOL * np.linalg.norm(x))
        y1 = me.apply_sqrtCL_left_factor(me.apply_sqrtCL_right_factor(x, ggg), ggg)
        y2 = me.apply_CL(x, ggg)
        assert(np.linalg.norm(y2 - y1) < _RTOL * np.linalg.norm(y2))
        z1 = me.solve_sqrtCL_right_factor(me.solve_sqrtCL_left_factor(x, ggg), ggg)
        z2 = me.solve_CL(x, ggg)
        assert (np.linalg.norm(z2 - z1) < _RTOL * np.linalg.norm(z2))

    @property
    def shape(me) -> typ.Tuple[int, int]:
        return me.M.shape

    @property
    def N(me) -> int:
        return me.shape[1]

    def delta(me, gamma: float) -> float:
        assert(gamma > 0)
        return 4. * gamma / (me.correlation_length ** 2)

    @cached_property
    def delta0(me) -> float:
        return me.delta(me.gamma0)

    @cached_property # lumped mass matrix
    def ML(me) -> sps.csr_matrix:
        return sps.diags(me.mass_lumps, 0).tocsr()

    @cached_property  # lumped mass matrix
    def invML(me) -> sps.csr_matrix:
        return sps.diags(1.0 / me.mass_lumps, 0).tocsr()

    @cached_property
    def A0(me) -> sps.csr_matrix:
        return me.gamma0 * me.K + me.delta0 * me.M

    def A(me, gamma: float) -> sps.csr_matrix:
        assert(gamma > 0)
        return (gamma / me.gamma0) * me.A0
        # return gamma * me.K + me.delta(gamma) * me.M

    def apply_A(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert(x.shape == (me.N,))
        assert (gamma > 0)
        return gamma * (me.K @ x) + me.delta(gamma) * (me.M @ x)

    @cached_property
    def AL0(me) -> sps.csr_matrix:
        return me.gamma0 * me.K + me.delta0 * me.ML

    def AL(me, gamma: float) -> sps.csr_matrix: # A with lumped mass matrix
        assert (gamma > 0)
        return (gamma / me.gamma0) * me.AL0
        # return gamma * me.K + me.delta(gamma) * me.ML

    def apply_AL(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert(x.shape == (me.N,))
        assert (gamma > 0)
        return gamma * (me.K @ x) + me.delta(gamma) * (me.ML @ x)

    @cached_property
    def solve_M(me) -> vec2vec_numpy:
        return spla.factorized(me.M)

    def apply_ML(me, x: np.ndarray) -> np.ndarray:
        assert (x.shape == (me.N,))
        return x * me.mass_lumps

    def solve_ML(me, x: np.ndarray) -> np.ndarray:
        assert (x.shape == (me.N,))
        return x / me.mass_lumps

    @cached_property
    def sqrt_mass_lumps(me):
        return np.sqrt(me.mass_lumps)

    def apply_sqrtML(me, x: np.ndarray) -> np.ndarray:
        assert (x.shape == (me.N,))
        return x * me.sqrt_mass_lumps

    def solve_sqrtML(me, x: np.ndarray) -> np.ndarray:
        assert (x.shape == (me.N,))
        return x / me.sqrt_mass_lumps

    @cached_property
    def solve_A0(me) -> vec2vec_numpy: # shifted Bilaplacian
        return spla.factorized(me.A0)

    def solve_A(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return (me.gamma0 / gamma) * me.solve_A0(x)

    @cached_property
    def solve_AL0(me) -> vec2vec_numpy: # lumped mass shifted Bilaplacian
        return spla.factorized(me.AL0)

    def solve_AL(me, x: np.ndarray, gamma: float) -> np.ndarray: # lumped mass shifted Bilaplacian
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return (me.gamma0 / gamma) * me.solve_AL0(x)

    def apply_C(me, x: np.ndarray, gamma: float) -> np.ndarray: # covariance matrix
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return me.solve_A(me.M @ me.solve_A(x, gamma), gamma)

    def apply_CL(me, x: np.ndarray, gamma: float) -> np.ndarray: # lumped mass covariance matrix
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return me.solve_AL(me.mass_lumps * me.solve_AL(x, gamma), gamma)

    def solve_C(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert(x.shape == (me.N,))
        assert(gamma > 0)
        return me.apply_A(me.solve_M(me.apply_A(x, gamma)), gamma)
        # return (gamma / me.gamma0)**2 * (me.A0 @ me.solve_M(me.A0 @ x))

    def solve_CL(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return me.apply_AL(me.solve_ML(me.apply_AL(x, gamma)), gamma)
        # return (gamma / me.gamma0)**2 * (me.AL0 @ me.solve_ML(me.AL0 @ x))

    @cached_property
    def invCL0(me) -> sps.csr_matrix:
        return me.AL0.T @ (me.invML @ me.AL0)

    def invCL(me, gamma: float) -> sps.csr_matrix:
        assert (gamma > 0)
        return (gamma / me.gamma0)**2 * me.invCL0

    def invC_array(me, gamma: float) -> np.ndarray:
        assert (gamma > 0)
        print('Warning: building dense inverse covariance matrix. May be very expensive.')
        A0_array = me.A0.toarray()
        M_array = me.M.toarray()
        return (gamma / me.gamma0)**2 * A0_array.T @ (np.linalg.inv(M_array) @ A0_array)

    def make_invC_hmatrix(me, bct: hpro.BlockClusterTree, gamma: float,
                          rtol: float=1e-8, atol: float=1e-14, force_spd: bool=False) -> hpro.HMatrix:
        assert (gamma > 0)
        K_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(me.K, bct)
        M_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(me.M, bct)
        iM_hmatrix = M_hmatrix.inv(rtol=rtol, overwrite=False)
        K_iM_K_hmatrix = hpro.h_mul(K_hmatrix, hpro.h_mul(iM_hmatrix, K_hmatrix, rtol=rtol, atol=atol),
                                    rtol=rtol, atol=atol)
        if force_spd:
            K_iM_K_hmatrix = hpro.rational_positive_definite_approximation_low_rank_method(
                K_iM_K_hmatrix, cutoff=0.0, overwrite=True)

        delta_local = me.delta(gamma)
        tmp = hpro.h_add(K_hmatrix, K_iM_K_hmatrix, alpha=2.0 * gamma * delta_local, beta=gamma ** 2,
                         rtol=rtol, atol=atol, overwrite_B=True)
        invC_hmatrix = hpro.h_add(M_hmatrix, tmp, alpha=delta_local ** 2, beta=1.0,
                                  rtol=rtol, atol=atol, overwrite_B=True)

        x = np.random.randn(me.N)
        y = me.solve_C(x, gamma)
        y2 = invC_hmatrix.matvec(x)
        relerr_invC_hmatrix = np.linalg.norm(y2 - y) / np.linalg.norm(y)
        print('rtol=', rtol, ', relerr_invC_hmatrix=', relerr_invC_hmatrix)

        return invC_hmatrix  # gamma^2 K invM K + 2 gamma delta K + delta^2 M = (gamma K + delta M) invM (gamma K + delta M)

    def make_invCL_hmatrix(me, bct: hpro.BlockClusterTree, gamma: float) -> hpro.HMatrix:
        assert (gamma > 0)
        invCL_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(me.invCL(gamma), bct)

        x = np.random.randn(me.N)
        y = me.solve_CL(x, gamma)
        y2 = invCL_hmatrix.matvec(x)
        relerr_invCL_hmatrix = np.linalg.norm(y2 - y) / np.linalg.norm(y)
        print('relerr_invCL_hmatrix=', relerr_invCL_hmatrix)

        return invCL_hmatrix

    def apply_sqrtCL_left_factor(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return me.solve_AL(me.apply_sqrtML(x), gamma)

    def apply_sqrtCL_right_factor(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return me.apply_sqrtML(me.solve_AL(x, gamma))

    def solve_sqrtCL_left_factor(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return me.solve_sqrtML(me.apply_AL(x, gamma))

    def solve_sqrtCL_right_factor(me, x: np.ndarray, gamma: float) -> np.ndarray:
        assert (x.shape == (me.N,))
        assert (gamma > 0)
        return me.apply_AL(me.solve_sqrtML(x), gamma)


def get_mass_lumps(M0: sps.csr_matrix, mass_lumping_method: str='rowsum') -> np.ndarray:
    if mass_lumping_method.lower() in {'consistent', 'rowsum'}:
        mass_lumps = M0 @ np.ones(M0.shape[1])
    elif mass_lumping_method.lower() in {'diagonal', 'diag'}:
        mass_lumps = M0.diagonal()
    else:
        raise RuntimeError('mass lumping method ' + mass_lumping_method + ' not recognized')
    return mass_lumps


def make_bilaplacian_covariance_scipy(
        gamma: float,
        correlation_length : float,
        Vh: dl.FunctionSpace,
        mass_lumping_method: str='rowsum',
        robin_bc: bool=True) -> BiLaplacianCovarianceScipy:
    stiffness_form = dl.inner(dl.grad(dl.TrialFunction(Vh)), dl.grad(dl.TestFunction(Vh))) * dl.dx
    robin_form = dl.inner(dl.TrialFunction(Vh), dl.TestFunction(Vh)) * dl.ds
    mass_form = dl.inner(dl.TrialFunction(Vh), dl.TestFunction(Vh)) * dl.dx

    # Stiffness matrix without Robin BCs
    K = csr_fenics2scipy(dl.assemble(stiffness_form))
    if robin_bc:
        # Robin BC matrix
        B0 = csr_fenics2scipy(dl.assemble(robin_form))
        K = K + B0 * 2. / (1.42 * correlation_length)

    # Mass matrix
    M = csr_fenics2scipy(dl.assemble(mass_form))
    mass_lumps = get_mass_lumps(M, mass_lumping_method)

    return BiLaplacianCovarianceScipy(gamma, correlation_length, K, M, mass_lumps)


@dataclass
class BilaplacianRegularizationScipy:
    Cov: BiLaplacianCovarianceScipy # covariance operator
    mu: np.ndarray # mean
    m: np.ndarray # current parameter value
    gamma: float # regularization parameter
    lumped_mass: bool=True

    def __post__init__(me):
        assert(me.gamma > 0)
        assert(me.Cov.shape == (me.N, me.N))
        assert(me.mu.shape == (me.N,))
        assert(me.m.shape == (me.N,))

    @cached_property
    def N(me):
        return me.Cov.N

    def draw_sample(me) -> np.ndarray:
        if me.lumped_mass:
            return me.mu + me.Cov.apply_sqrtCL_left_factor(np.random.randn(me.N), me.gamma)
        else:
            raise NotImplementedError

    def update_m(me, new_m: np.ndarray) -> None:
        assert(new_m.shape == (me.N,))
        me.m[:] = new_m

    def update_gamma(me, new_gamma: float) -> None:
        assert(new_gamma > 0)
        me.gamma = new_gamma

    def cost(me) -> float:
        p = me.m - me.mu
        if me.lumped_mass:
            return 0.5 * np.dot(p, me.Cov.solve_CL(p, me.gamma))
        else:
            return 0.5 * np.dot(p, me.Cov.solve_C(p, me.gamma))

    def grad(me) -> np.ndarray:
        p = me.m - me.mu
        if me.lumped_mass:
            return me.Cov.solve_CL(p, me.gamma)
        else:
            return me.Cov.solve_C(p, me.gamma)

    def apply_hess(me, x: np.ndarray) -> np.ndarray:
        assert(x.shape == (me.N,))
        if me.lumped_mass:
            return me.Cov.solve_CL(x, me.gamma)
        else:
            return me.Cov.solve_C(x, me.gamma)

    def solve_hess(me, x: np.ndarray) -> np.ndarray:
        assert(x.shape == (me.N,))
        if me.lumped_mass:
            return me.Cov.apply_CL(x, me.gamma)
        else:
            return me.Cov.apply_C(x, me.gamma)

    def apply_hess_linop(me) -> spla.LinearOperator:
        return spla.LinearOperator((me.N, me.N), matvec=me.apply_hessian)

    def solve_hess_linop(me) -> spla.LinearOperator:
        return spla.LinearOperator((me.N, me.N), matvec=me.solve_hessian)

    def make_hess_hmatrix(me, bct: hpro.BlockClusterTree, **kwargs) -> hpro.HMatrix:
        if me.lumped_mass:
            return me.Cov.make_invCL_hmatrix(bct, me.gamma, **kwargs)
        else:
            return me.Cov.make_invC_hmatrix(bct, me.gamma, **kwargs)




class BiLaplacianRegularization:
    # ==== SET UP PRIOR DISTRIBUTION ====
    # Recall from Daon Stadler 2017
    # "The covariance function of the free-space
    # operator has a characteristic length of
    # sqrt(8(p - d / 2)) sqrt( gamma/delta )
    # meaning that that distance away from a source
    # x, the covariance decays to 0.1 of its
    # maximal value
    # d - dimension of problem
    # A^(-p) - operator, A laplacian-like
    def __init__(me,
                 gamma              : float,
                 correlation_length : float,
                 parameter          : dl.Function,
                 prior_mean         : dl.Function,
                 robin_bc=True,
                 lumping_type='consistent',
                 ):
        me.parameter = parameter
        me.prior_mean = prior_mean
        me.robin_bc = robin_bc

        me.V = me.prior_mean.function_space()

        me.stiffness_form = dl.inner(dl.grad(dl.TrialFunction(me.V)), dl.grad(dl.TestFunction(me.V))) * dl.dx
        me.robin_form = dl.inner(dl.TrialFunction(me.V), dl.TestFunction(me.V)) * dl.ds
        me.mass_form = dl.inner(dl.TrialFunction(me.V), dl.TestFunction(me.V)) * dl.dx

        # Stiffness matrix without Robin BCs
        me.K0_petsc = dl.assemble(me.stiffness_form)
        me.K0_scipy = csr_fenics2scipy(me.K_petsc)

        # Robin BC matrix
        me.B0_petsc = dl.assemble(me.robin_form)
        me.B0_scipy = csr_fenics2scipy(me.B_petsc)

        # Mass matrix
        me.M0_petsc = dl.assemble(me.mass_form)
        me.M0_scipy = csr_fenics2scipy(me.M_petsc)

        me.solve_M_numpy = spla.factorized(me.M_scipy)

        # Lumped mass matrix
        me.ML_petsc = dl.assemble(me.mass_form)
        me.mass_lumps_petsc = dl.Vector()
        me.ML_petsc.init_vector(me.mass_lumps_petsc, 1)
        if lumping_type.lower() == 'consistent':
            me.mass_lumps_petsc[:] = dl.assemble(dl.TestFunction(me.V) * dl.dx) # consistent lumps mass matrix
        else: # lumping_type.lower() == 'diagonal':
            me.ML_petsc.get_diagonal(me.mass_lumps_petsc) # diagonal lumps mass matrix
        me.ML_petsc.zero()
        me.ML_petsc.set_diagonal(me.mass_lumps_petsc)

        me.mass_lumps_numpy = me.mass_lumps_petsc[:]
        me.ML_scipy = sps.dia_matrix(me.mass_lumps_numpy).tocsr()
        me.iML_scipy = sps.diags([1.0 / me.mass_lumps_numpy], [0]).tocsr()
        me.iML_petsc = csr_scipy2fenics(me.iML_scipy)

        # Regularization square root
        me.Rsqrt_form = None
        me.Rsqrt_petsc = None
        me.Rsqrt_scipy = None
        me.solve_Rsqrt_numpy = None
        me.R_lumped_scipy = None
        me.solve_R_lumped_scipy = None
        me.gamma = gamma
        me.correlation_length = correlation_length
        me.update_R_stuff()

    def update_R_stuff(me):
        me.Rsqrt_form = (dl.Constant(me.gamma) * me.stiffness_form
                         + dl.Constant(me.delta) * me.mass_form
                         + dl.Constant(me.robin_coeff) * me.robin_form)

        me.Rsqrt_petsc = dl.assemble(me.Rsqrt_form)
        me.Rsqrt_scipy = csr_fenics2scipy(me.Rsqrt_petsc)

        me.solve_Rsqrt_numpy = spla.factorized(me.Rsqrt_scipy)

        # Regularization operator with lumped mass approximation for inverse mass matrix
        me.R_lumped_scipy = me.Rsqrt_scipy @ (me.iML_scipy @ me.Rsqrt_scipy)
        me.solve_R_lumped_scipy = spla.factorized(me.R_lumped_scipy)

    def update_gamma(me, new_gamma):
        me.gamma = new_gamma
        me.update_R_stuff()

    def update_correlation_length(me, new_correlation_length):
        me.correlation_length = new_correlation_length
        me.update_R_stuff()

    @property
    def delta(me):
        return 4. * me.gamma / (me.correlation_length ** 2)

    @property
    def robin_coeff(me):
        if me.robin_bc:
            return me.gamma * np.sqrt(me.delta / me.gamma) / 1.42
            # return me.gamma * np.sqrt((4. * me.gamma / (me.correlation_length ** 2)) / me.gamma) / 1.42
            # return me.gamma * np.sqrt((4. / (me.correlation_length ** 2)) ) / 1.42
            # return me.gamma * 2. / (1.42 * me.correlation_length)
        else:
            return 0.0

    def solve_ML_numpy(me, u_numpy):
        return u_numpy / me.mass_lumps_numpy

    def apply_R_numpy(me, u_numpy):
        return me.Rsqrt_scipy @ me.solve_M_numpy(me.Rsqrt_scipy @ u_numpy)

    def solve_R_numpy(me, v_numpy):
        return me.solve_Rsqrt_numpy(me.M_scipy @ me.solve_Rsqrt_numpy(v_numpy))

    def make_M_hmatrix(me, bct):
        return hpro.build_hmatrix_from_scipy_sparse_matrix(me.M_scipy, bct)

    def make_Rsqrt_hmatrix(me, bct):
        return hpro.build_hmatrix_from_scipy_sparse_matrix(me.Rsqrt_scipy, bct)

    def make_R_lumped_hmatrix(me, bct):
        return hpro.build_hmatrix_from_scipy_sparse_matrix(me.R_lumped_scipy, bct)

    def make_R_hmatrix(me, bct, rtol=1e-7, atol=1e-14):
        Rsqrt_hmatrix = me.make_Rsqrt_hmatrix(bct)
        M_hmatrix = me.make_M_hmatrix(bct)
        iM_hmatrix = M_hmatrix.inv(rtol=rtol, atol=atol)
        R_hmatrix = hpro.h_mul(Rsqrt_hmatrix, hpro.h_mul(iM_hmatrix, Rsqrt_hmatrix, rtol=rtol, atol=atol), rtol=rtol, atol=atol).sym()
        return R_hmatrix

    def apply_R_petsc(me, u_petsc):
        return me.numpy2petsc(me.apply_R_numpy(me.petsc2numpy(u_petsc)))

    def solve_R_petsc(me, u_petsc):
        return me.numpy2petsc(me.solve_R_numpy(me.petsc2numpy(u_petsc)))

    def solve_Rsqrt_petsc(me, u_petsc):
        return me.numpy2petsc(me.solve_Rsqrt_numpy(me.petsc2numpy(u_petsc)))

    def solve_ML_petsc(me, u_petsc):
        return me.numpy2petsc(me.solve_ML_numpy(me.petsc2numpy(u_petsc)))

    def solve_M_petsc(me, u_petsc):
        return me.numpy2petsc(me.solve_M_numpy(me.petsc2numpy(u_petsc)))

    def cost(me):
        delta_m_numpy = me.parameter.vector()[:] - me.prior_mean.vector()[:]
        return 0.5 * np.dot(delta_m_numpy, me.apply_R_numpy(delta_m_numpy))

    def gradient_petsc(me):
        delta_m_petsc = me.parameter.vector() - me.prior_mean.vector()
        return me.apply_R_petsc(delta_m_petsc)

    def gradient_numpy(me):
        return me.gradient_petsc()[:]

    def apply_hessian_petsc(me, z_petsc):
        return me.apply_R_petsc(z_petsc)

    def apply_hessian_numpy(me, z_numpy):
        return me.apply_R_numpy(z_numpy)

    def solve_hessian_petsc(me, z_petsc):
        return me.solve_R_petsc(z_petsc)

    def solve_hessian_numpy(me, z_numpy):
        return me.solve_R_numpy(z_numpy)

    def petsc2numpy(me, u_petsc):
        u_numpy = u_petsc[:]
        return u_numpy

    def numpy2petsc(me, u_numpy):
        u_petsc = dl.Function(me.V).vector()
        u_petsc[:] = u_numpy
        return u_petsc