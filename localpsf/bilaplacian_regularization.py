import numpy as np
import dolfin as dl
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from nalger_helper_functions import csr_fenics2scipy, csr_scipy2fenics
import hlibpro_python_wrapper as hpro


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
                 robin_bc=True):
        me.parameter = parameter
        me.prior_mean = prior_mean
        me.robin_bc = robin_bc

        me.V = me.prior_mean.function_space()

        me.stiffness_form = dl.inner(dl.grad(dl.TrialFunction(me.V)), dl.grad(dl.TestFunction(me.V))) * dl.dx
        me.mass_form = dl.inner(dl.TrialFunction(me.V), dl.TestFunction(me.V)) * dl.dx
        me.robin_form = dl.inner(dl.TrialFunction(me.V), dl.TestFunction(me.V)) * dl.ds

        # Mass matrix
        me.M_petsc = dl.assemble(me.mass_form)
        me.M_scipy = csr_fenics2scipy(me.M_petsc)

        me.solve_M_numpy = spla.factorized(me.M_scipy)

        # Lumped mass matrix
        me.ML_petsc = dl.assemble(me.mass_form)
        me.mass_lumps_petsc = dl.Vector()
        me.ML_petsc.init_vector(me.mass_lumps_petsc, 1)
        me.ML_petsc.get_diagonal(me.mass_lumps_petsc)
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