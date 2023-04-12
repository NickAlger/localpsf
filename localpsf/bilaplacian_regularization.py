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
    gamma: float
    correlation_length: float
    K: sps.csr_matrix # stiffness matrix
    M: sps.csr_matrix # mass matrix

    def __post_init__(me):
        RTOL = 1e-10
        assert(np.linalg.norm(me.K - me.K.T) < RTOL * np.linalg.norm(me.K))
        assert (np.linalg.norm(me.M - me.M.T) < RTOL * np.linalg.norm(me.M))
        x = np.random.randn(me.N)
        x2 = me.solve_M(me.M @ x)
        assert(np.linalg.norm(x2 - x) < RTOL * np.linalg.norm(x))
        x3 = me.solve_covariance(me.apply_covariance(x))
        assert(np.linalg.norm(x3 - x) < RTOL * np.linalg.norm(x))

    @cached_property
    def shape(me) -> typ.Tuple[int, int]:
        return me.M.shape

    @cached_property
    def N(me) -> int:
        return me.shape[1]

    @cached_property
    def delta(me) -> float:
        return 4. * me.gamma / (me.correlation_length ** 2)

    @cached_property
    def A(me) -> sps.csr_matrix:
        return me.gamma * me.K + me.delta * me.M

    @cached_property
    def solve_M(me) -> vec2vec_numpy:
        return spla.factorized(me.M)

    @cached_property
    def solve_A(me) -> vec2vec_numpy:
        return spla.factorized(me.A)

    def apply_covariance(me, x: np.ndarray) -> np.ndarray:
        return me.solve_A(me.M @ me.solve_A(x))

    def solve_covariance(me, x: np.ndarray) -> np.ndarray:
        return me.A @ me.solve_M(me.A @ x)

    def make_K_hmatrix(me, bct) -> hpro.HMatrix:
        return hpro.build_hmatrix_from_scipy_sparse_matrix(me.K, bct)

    def make_M_hmatrix(me, bct) -> hpro.HMatrix:
        return hpro.build_hmatrix_from_scipy_sparse_matrix(me.M, bct)

    def make_precision_hmatrix(me, bct, rtol=1e-8, atol=1e-14, force_spd=True) -> hpro.HMatrix:
        K_hmatrix = me.make_K_hmatrix(bct)
        M_hmatrix = me.make_M_hmatrix(bct)
        iM_hmatrix = M_hmatrix.inv(rtol=rtol, overwrite=False)
        K_iM_K_hmatrix = hpro.h_mul(K_hmatrix, hpro.h_mul(iM_hmatrix, K_hmatrix, rtol=rtol, atol=atol),
                                    rtol=rtol, atol=atol)
        if force_spd:
            K_iM_K_hmatrix = hpro.rational_positive_definite_approximation_low_rank_method(me, cutoff=0.0, overwrite=True)

        tmp = hpro.h_add(me.K, K_iM_K_hmatrix, alpha=2.0*me.gamma*me.delta, beta=me.gamma**2,
                         rtol=rtol, atol=atol, overwrite_B=True)
        invC_hmatrix = hpro.h_add(me.M, tmp, alpha=me.delta**2, beta=1.0,
                                  rtol=rtol, atol=atol, overwrite_B=True)

        x = np.random.randn(me.N)
        y = me.solve_covariance(x)
        y2 = invC_hmatrix.matvec(x)
        relerr_invC_hmatrix = np.linalg.norm(y2 - y) / np.linalg.norm(y)
        print('rtol=', rtol, ', relerr_invC_hmatrix=', relerr_invC_hmatrix)

        return invC_hmatrix # gamma^2 K invM K + 2 gamma delta K + delta^2 M = (gamma K + delta M) invM (gamma K + delta M)

    def update_gamma(me, new_gamma: float) -> 'BiLaplacianCovarianceScipy':
        assert(new_gamma >= 0.0)
        return BiLaplacianCovarianceScipy(new_gamma, me.correlation_length, me.K, me.M)

    def update_correlation_length(me, new_correlation_length):
        assert(new_correlation_length > 0.0)
        return BiLaplacianCovarianceScipy(me.gamma, new_correlation_length, me.K, me.M)

    def lump_mass_matrix(me, mass_lumping_type: str) -> 'BiLaplacianCovarianceScipy':
        ML = lump_mass(me.M, mass_lumping_type)
        return BiLaplacianCovarianceScipy(me.gamma, me.correlation_length, me.K, ML)


def lump_mass(M0: sps.csr_matrix, mass_lumping_type: str) -> sps.csr_matrix:
    if mass_lumping_type.lower() in {'consistent', 'rowsum'}:
        mass_lumps = M0 @ np.ones(M0.shape[1])
        M = sps.diags(mass_lumps, 0).tocsr()
    elif mass_lumping_type.lower() == {'diagonal', 'diag'}:
        M = sps.diags(M0.diagonal(), 0).tocsr()
    else:
        raise RuntimeError('mass lumping type ' + mass_lumping_type + ' not recognized')
    return M

def make_bilaplacian_covariance_scipy(
        gamma: float,
        correlation_length : float,
        Vh: dl.FunctionSpace,
        mass_lumping_type: str=None,
        robin_bc: bool=True) -> BiLaplacianCovarianceScipy:
    stiffness_form = dl.inner(dl.grad(dl.TrialFunction(Vh)), dl.grad(dl.TestFunction(Vh))) * dl.dx
    robin_form = dl.inner(dl.TrialFunction(Vh), dl.TestFunction(Vh)) * dl.ds
    mass_form = dl.inner(dl.TrialFunction(Vh), dl.TestFunction(Vh)) * dl.dx

    # Stiffness matrix without Robin BCs
    K = csr_fenics2scipy(dl.assemble(stiffness_form))
    if robin_bc:
        # Robin BC matrix
        B0 = csr_fenics2scipy(dl.assemble(robin_form))
        K = K + B0 * gamma * 2. / (1.42 * correlation_length)

    # Mass matrix
    M = csr_fenics2scipy(dl.assemble(mass_form))
    if mass_lumping_type is not None:
        M = lump_mass(M, mass_lumping_type)

    return BiLaplacianCovarianceScipy(gamma, correlation_length, K, M)


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