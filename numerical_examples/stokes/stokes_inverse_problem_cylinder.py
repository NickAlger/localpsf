import numpy as np
import dolfin as dl
from ufl import lhs, rhs, replace
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from pathlib import Path
import os

import meshio

from nalger_helper_functions import *
import hlibpro_python_wrapper as hpro

from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix

# For Tucker's computer:
# import sys
# sys.path.append('/home/tucker/software/ucm-ice/')
# sys.path.append('/home/tucker/software/hippylib/')
# End For Tucker

# For Nick's computer:
import meshio
import sys
sys.path.append('../../../hippylib/')
import hippylib as hp
sys.path.append('../../../ucm-ice/')
# from iceModel import *
# End For Nick

import hippylib as hp
from iceModel import *
from scipy.spatial import KDTree
import warnings



"""
   compute y = D^-1 x,
   where D is a diagonal matrix (dolfin Vector datatype)
         x is a Vector          (dolfin Vector datatype)
         y is a Vector          (dolfin Vector datatype)
"""
def diagSolve(A, x):
    y      = dl.Vector(x)
    APETSc = dl.as_backend_type(A).vec()
    xPETSc = dl.as_backend_type(x).vec()
    yPETSc = dl.as_backend_type(y).vec()
    yPETSc.pointwiseDivide(xPETSc, APETSc)
    y = dl.Vector(dl.PETScVector(yPETSc))
    return y

def permut(p, q):
    KDT = KDTree(p)
    p2q = KDT.query(q)[1]
    q2p = np.argsort(p2q)
    if np.linalg.norm(q - p[p2q,:]) > 1.e-10:
        warnings.warn("is q a permutation of p?")
    if str(p) == str(q):
        print("p to q mapping is identitiy")
    return p2q, q2p


class BasalBoundary(dl.SubDomain):
    def inside(me, x, on_boundary):
        return dl.near(x[2], 0.) and on_boundary

class BasalBoundarySub(dl.SubDomain):
    def inside(me, x, on_boundary):
        return dl.near(x[2], 0.)



class TopBoundary(dl.SubDomain):
    def __init__(me, Height):
        me.Height = Height
        dl.SubDomain.__init__(me)
    def inside(me, x, on_boundary):
        return dl.near(x[1], me.Height) and on_boundary

def Tang(u, n):
    return u - dl.outer(n, n)*u

class proj_op:
    """
    This class is implemented in order
    to compute the action of operator op
    on a subspace
    Proj: full space --> subspace
    op:   full space --> full space
    proj_op: subspace --> subspace
    the action of proj_op will be the
    action of P op P^T
    """
    def __init__(self, op, Proj):
        self.op = op
        self.Proj = Proj
        # intermediate full space help vectors
        self.xfull = dl.Vector()
        self.yfull = dl.Vector()
        self.Proj.init_vector(self.xfull, 1)
        self.Proj.init_vector(self.yfull, 1)

    def init_vector(self, x, dim):
        self.Proj.init_vector(x, 0)

    def mult(self, x, y):
        self.Proj.transpmult(x, self.xfull)
        self.op.mult(self.xfull, self.yfull)
        self.Proj.mult(self.yfull, y)


class BiLaplacianRegularizationOperator:
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
    def __init__(me, gamma, correlation_length, V, robin_bc=True):
        me.gamma = gamma
        me.correlation_length = correlation_length
        me.V = V
        me.robin_bc = robin_bc

        me.delta = 4. * me.gamma / (me.correlation_length ** 2)
        if me.robin_bc:
            me.robin_coeff = me.gamma * np.sqrt(me.delta / me.gamma) / 1.42
        else:
            me.robin_coeff = 0.0

        me.stiffness_form = dl.inner(dl.grad(dl.TrialFunction(V)), dl.grad(dl.TestFunction(V))) * dl.dx
        me.mass_form = dl.inner(dl.TrialFunction(V), dl.TestFunction(V)) * dl.dx
        me.robin_form = dl.inner(dl.TrialFunction(V), dl.TestFunction(V)) * dl.ds

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
        me.Rsqrt_form = (dl.Constant(me.gamma) * me.stiffness_form
                         + dl.Constant(me.delta) * me.mass_form
                         + dl.Constant(me.robin_coeff) * me.robin_form)

        me.Rsqrt_petsc = dl.assemble(me.Rsqrt_form)
        me.Rsqrt_scipy = csr_fenics2scipy(me.Rsqrt_petsc)

        me.solve_Rsqrt_numpy = spla.factorized(me.Rsqrt_scipy)

        # Regularization operator with lumped mass approximation for inverse mass matrix
        me.R_lumped_scipy = me.Rsqrt_scipy @ (me.iML_scipy @ me.Rsqrt_scipy)
        me.solve_R_lumped_scipy = spla.factorized(me.R_lumped_scipy)

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

    def petsc2numpy(me, u_petsc):
        u_numpy = u_petsc[:]
        return u_numpy

    def numpy2petsc(me, u_numpy):
        u_petsc = dl.Function(me.V).vector()
        u_petsc[:] = u_numpy
        return u_petsc


def stokes_mesh_setup(mesh):
    boundary_mesh = dl.BoundaryMesh(mesh, "exterior", True)
    basal_mesh3D = dl.SubMesh(boundary_mesh, BasalBoundarySub())
    r0 = 0.05
    sig = 0.4
    valleys = 4
    valley_depth = 0.35
    bump_height = 0.2
    min_thickness = 0.08 / 8.
    avg_thickness = 0.2 / 8.
    theta = -np.pi / 2.
    max_thickness = avg_thickness + (avg_thickness - min_thickness)
    A_thickness = max_thickness - avg_thickness

    dilitation = 1.e4
    Length = 1.
    Width = 1.
    Length *= 2 * dilitation
    Width *= 2 * dilitation
    Radius = dilitation

    coords = mesh.coordinates()
    bcoords = boundary_mesh.coordinates()
    subbcoords = basal_mesh3D.coordinates()
    coord_sets = [coords, bcoords, subbcoords]

    def topography(r, t):
        zero = np.zeros(r.shape)
        R0 = r0 * np.ones(r.shape)
        return bump_height * np.exp(-(r / sig) ** 2) * (
                    1. + valley_depth * np.sin(valleys * t - theta) * np.fmax(zero, (r - R0) / sig))

    def depth(r, t):
        zero = np.zeros(r.shape)
        R0 = r0 * np.ones(r.shape)
        return min_thickness - A_thickness * np.sin(valleys * t - theta) * np.exp(
            -(r / sig) ** 2) * np.fmax(zero, (r - R0) / sig)

    for k in range(len(coord_sets)):
        for i in range(len(coord_sets[k])):
            x, y, z = coord_sets[k][i]
            r = np.sqrt(x ** 2 + y ** 2)
            t = np.arctan2(y, x)
            coord_sets[k][i, 2] = depth(r, t) * z + topography(r, t)
            coord_sets[k][i] *= dilitation

    # ------ generate 2D mesh from 3D boundary subset mesh
    coords = basal_mesh3D.coordinates()[:, :2]
    cells = [("triangle", basal_mesh3D.cells())]
    mesh2D = meshio.Mesh(coords, cells)
    mesh2D.write("mesh2D.xml")
    basal_mesh2D = dl.Mesh("mesh2D.xml")

    return mesh, boundary_mesh, basal_mesh3D, basal_mesh2D, Radius


def function_space_prolongate_numpy(x_numpy, dim_Yh, inds_Xh_in_Yh):
    y_numpy = np.zeros(dim_Yh)
    y_numpy[inds_Xh_in_Yh] = x_numpy
    return y_numpy

def function_space_restrict_numpy(y_numpy, inds_Xh_in_Yh):
    return y_numpy[inds_Xh_in_Yh].copy()

def function_space_prolongate_petsc(x_petsc, Yh, inds_Xh_in_Yh):
    y_petsc = dl.Function(Yh).vector()
    y_petsc[:] = function_space_prolongate_numpy(x_petsc[:], Yh.dim(), inds_Xh_in_Yh)
    return y_petsc

def function_space_restrict_petsc(y_petsc, Xh, inds_Xh_in_Yh):
    x_petsc = dl.Function(Xh).vector()
    x_petsc[:] = function_space_restrict_numpy(y_petsc[:], inds_Xh_in_Yh)
    return x_petsc

def make_prolongation_and_restriction_operators(Vsmall, Vbig, inds_Vsmall_in_Vbig):
    Vbig_to_Vsmall_numpy = lambda vb: function_space_restrict_numpy(vb, inds_Vsmall_in_Vbig)
    Vsmall_to_Vbig_numpy = lambda vs: function_space_prolongate_numpy(vs, Vbig.dim(), inds_Vsmall_in_Vbig)

    Vbig_to_Vsmall_petsc = lambda vb: function_space_restrict_petsc(vb, Vsmall, inds_Vsmall_in_Vbig)
    Vsmall_to_Vbig_petsc = lambda vs: function_space_prolongate_petsc(vs, Vbig, inds_Vsmall_in_Vbig)

    return Vbig_to_Vsmall_numpy, Vsmall_to_Vbig_numpy, Vbig_to_Vsmall_petsc, Vsmall_to_Vbig_petsc


class StokesInverseProblemCylinder:
    def __init__(me,
                 mesh,
                 boundary_markers,
                 noise_level=1e-2,
                 prior_correlation_length=0.05,
                 make_plots=True,
                 save_plots=True,
                 load_fwd=False,
                 # use lam = 1.e10 for coarse mesh
                 # might need to use larger value of lambda for finer mesh.
                 lam=1.e14,   # no-outflow penalty parameter to enforce u^T n = 0 on basal boundary 
                 gamma=6.e1,  
                 m0_constant_value = 7.,
                 mtrue_string = 'm0 - (m0 / 7.)*std::cos((x[0]*x[0]+x[1]*x[1])*pi/(Radius*Radius))',
                 rel_correlation_length = 0.1,
                 noise_type = 'relative_local',
                 reg_robin_bc=True
                 ):
        ########    INITIALIZE OPTIONS    ########
        me.boundary_markers = boundary_markers
        me.noise_level = noise_level
        me.prior_correlation_length = prior_correlation_length
        me.built_R_hmatrix = False
        me.R_hmatrix      = None
        me.make_plots = make_plots
        me.save_plots = save_plots
        me.reg_robin_bc = reg_robin_bc
        me.m0_constant_value = m0_constant_value
        me.lam = lam

        ########    MESH    ########
        me.mesh, me.boundary_mesh, me.basal_mesh3D, me.basal_mesh2D, me.Radius = stokes_mesh_setup(mesh)

        ########    FUNCTION SPACE    ########
        P1 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        TH = P2 * P1
        me.Zh = dl.FunctionSpace(mesh, TH)                        # Zh:  state, also adjoint
        me.Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)             # Wh:  parameter, full 3D domain
        me.Vh3 = dl.FunctionSpace(me.basal_mesh3D, 'Lagrange', 1) # Vh3: parameter, 2d basal manifold, 3d coords
        me.Vh2 = dl.FunctionSpace(me.basal_mesh2D, 'Lagrange', 1) # Vh2: parameter, 2d basal flat space, 2d coords
        me.Xh = [me.Zh, me.Wh, me.Zh]                             # Xh:  (state, parameter full 3D domain, adjoint)

        me.m_Wh = dl.Function(me.Wh)
        me.m_Vh3 = dl.Function(me.Vh3)
        me.m_Vh2 = dl.Function(me.Vh2)
        me.u = dl.Function(me.Zh)
        me.p = dl.Function(me.Zh)

        ########    TRANSFER OPERATORS BETWEEN PARAMETER FUNCTION SPACES    ########
        pp_Vh3 = me.Vh3.tabulate_dof_coordinates()
        pp_Vh2 = me.Vh2.tabulate_dof_coordinates()
        # me.perm_Vh3_to_Vh2, me.perm_Vh2_to_Vh3 = permut(pp_Vh3[:,:2], pp_Vh2)

        pp_Wh = me.Wh.tabulate_dof_coordinates()

        KDT_Wh = KDTree(pp_Wh)
        me.inds_Vh3_in_Wh = KDT_Wh.query(pp_Vh3)[1]
        if np.linalg.norm(pp_Vh3 - pp_Wh[me.inds_Vh3_in_Wh, :]) / np.linalg.norm(pp_Vh3) > 1.e-12:
            warnings.warn('problem with basal function space inclusion')

        pp_Vh3_2D = pp_Vh3[:,:2]

        KDT_Vh3_2D = KDTree(pp_Vh3_2D)
        me.inds_Vh2_in_Vh3 = KDT_Vh3_2D.query(pp_Vh2)[1]
        if np.linalg.norm(pp_Vh2 - pp_Vh3[me.inds_Vh2_in_Vh3, :2]) / np.linalg.norm(pp_Vh2) > 1.e-12:
            warnings.warn('inconsistency between manifold basal mesh and flat basal mesh')

        me.inds_Vh2_in_Wh = me.inds_Vh3_in_Wh[me.inds_Vh2_in_Vh3]

        me.Wh_to_Vh3_numpy, me.Vh3_to_Wh_numpy, me.Wh_to_Vh3_petsc, me.Vh3_to_Wh_petsc = \
            make_prolongation_and_restriction_operators(me.Vh3, me.Wh, me.inds_Vh3_in_Wh)

        me.Vh3_to_Vh2_numpy, me.Vh2_to_Vh3_numpy, me.Vh3_to_Vh2_petsc, me.Vh2_to_Vh3_petsc = \
            make_prolongation_and_restriction_operators(me.Vh2, me.Vh3, me.inds_Vh2_in_Vh3)

        me.Wh_to_Vh2_numpy, me.Vh2_to_Wh_numpy, me.Wh_to_Vh2_petsc, me.Vh2_to_Wh_petsc = \
            make_prolongation_and_restriction_operators(me.Vh2, me.Wh, me.inds_Vh2_in_Wh)

        # ==== SET UP FORWARD MODEL ====
        # Forcing term
        grav  = 9.81           # acceleration due to gravity
        rho   = 910.0          # volumetric mass density of ice
        me.stokes_forcing = dl.Constant( (0., 0., -rho*grav) )

        ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)
        me.ds = ds
        me.ds_base = me.ds(1)
        me.ds_top = me.ds(2)
        me.normal = dl.FacetNormal(me.mesh)
        # Strongly enforced Dirichlet conditions. The no outflow condition will be enforced weakly, via a penalty parameter.
        me.bcs  = []
        me.bcs0 = []

        # Define the Nonlinear Stokes varfs
        # rheology
        me.stokes_n = 1.0 #3.0
        me.stokes_A = dl.Constant(2.140373e-7) # dl.Constant(1.e-16)
        me.smooth_strain = dl.Constant(1e-6)

        me.velocity, me.pressure = dl.split(me.u)
        me.strain = dl.sym(dl.grad(me.velocity))
        me.normEu12 = 0.5 * dl.inner(me.strain, me.strain) + me.smooth_strain

        me.tangent_velocity = (me.velocity - dl.outer(me.normal, me.normal)*me.velocity)
        me.stokes_exponent = ((1. + me.stokes_n) / (2. * me.stokes_n))

        if me.stokes_n == 1.0:
            me.stokes_energy_t1 = me.stokes_A ** (-1.) * me.normEu12 * dl.dx
            # me.stokes_energy_t1 = me.stokes_A ** (-1. / me.stokes_n) * ((2. * me.stokes_n) / (1. + me.stokes_n)) * me.normEu12 * dl.dx
            me.linear_forward_problem = True
        else:
            me.stokes_energy_t1 = (me.stokes_A ** (-1. / me.stokes_n) * ((2. * me.stokes_n) / (1. + me.stokes_n)) *
                                   (me.normEu12 ** ((1. + me.stokes_n) / (2. * me.stokes_n))) * dl.dx)
            me.linear_forward_problem = False

        me.stokes_energy_t2 = -dl.inner(me.stokes_forcing, me.velocity) * dl.dx
        me.stokes_energy_t3 = dl.Constant(.5) * dl.inner(dl.exp(me.m_Wh) * me.tangent_velocity, me.tangent_velocity) * me.ds_base
        me.stokes_energy_t4 = me.lam * dl.inner(me.velocity, me.normal) ** 2 * me.ds_base

        me.stokes_energy_form = me.stokes_energy_t1 + me.stokes_energy_t2 + me.stokes_energy_t3 + me.stokes_energy_t4
        me.stokes_constraint_form = dl.inner(-dl.div(me.velocity), me.pressure) * dl.dx

        # me.stokes_lagrangian_form

        me.stokes_energy_gradient = replace(dl.derivative(me.stokes_energy_form, me.u, dl.TestFunction(me.Zh)),
                                            {me.u:dl.TrialFunction(me.Zh)})
        me.stokes_energy_gradient_lhs = lhs(me.stokes_energy_gradient)
        me.stokes_energy_gradient_rhs = rhs(me.stokes_energy_gradient)

        me.stokes_constraint_gradient = replace(dl.derivative(me.stokes_constraint_form, me.u, dl.TestFunction(me.Zh)),
                                                {me.u:dl.TrialFunction(me.Zh)})
        me.stokes_constraint_gradient_lhs = lhs(me.stokes_constraint_gradient)
        me.stokes_constraint_gradient_rhs = rhs(me.stokes_constraint_gradient)

        me.stokes_coefficient_matrix_scipy = None            # Created in me.solve_forward()
        me.stokes_coefficient_matrix_factorized_solve = None # Created in me.solve_forward()

        # nonlinearStokesFunctional = NonlinearStokesForm(rheology_n, rheology_A, normal, ds(1), f, lam=lam)
        
        # Create one-hot vector on pressure dofs
        # constraint_vec = dl.interpolate(dl.Constant((0.,0., 0., 1.)), me.Zh).vector()
        #
        #
        # me.pde = EnergyFunctionalPDEVariationalProblem(me.Xh, nonlinearStokesFunctional, constraint_vec, bc, bc0)
        # me.pde.fwd_solver.parameters["rel_tolerance"] = 1.e-8
        # me.pde.fwd_solver.parameters["print_level"] = 1
        # me.pde.fwd_solver.parameters["LS"]["max_backtracking_iter"] = 20
        # me.pde.fwd_solver.solver = dl.PETScLUSolver("mumps")

        ########    TRUE BASAL FRICTION FIELD (INVERSION PARAMETER)    ########

        me.prior_mean_val = me.m0_constant_value
        me.prior_mean_Vh3_petsc = dl.interpolate(dl.Constant(me.prior_mean_val), me.Vh3).vector()
        me.prior_mean_Vh2_petsc = me.Vh3_to_Vh2_petsc(me.prior_mean_Vh3_petsc)
        me.m_func = dl.Expression(mtrue_string, element=me.Wh.ufl_element(), m0=me.prior_mean_val,Radius=me.Radius)
        me.mtrue   = dl.interpolate(me.m_func, me.Wh).vector()

        me.rel_correlation_length = rel_correlation_length # 0.1
        me.correlation_Length = me.Radius * me.rel_correlation_length

        me.REGOP = BiLaplacianRegularizationOperator(gamma, me.correlation_Length, me.Vh2, robin_bc=me.reg_robin_bc)


        ########    TRUE SURFACE VELOCITY FIELD AND NOISY OBSERVATIONS    ########
        me.set_m_without_solving(me.Wh_to_Vh2_numpy(me.mtrue[:]))
        me.solve_forward()
        me.utrue_fnc = dl.Function(me.Zh)
        me.utrue_fnc.vector()[:] = me.u.vector()[:].copy()

        me.outflow = np.sqrt(dl.assemble(dl.inner(me.utrue_fnc.sub(0), me.normal)**2.*ds(1)))


        # construct the form which describes the continuous observations
        # of the tangential component of the velocity field at the upper surface boundary
        component_observed = dl.interpolate(dl.Constant((1., 1., 0., 0.)), me.Zh)
        utest, ptest = dl.TestFunctions(me.Zh)
        utrial, ptrial = dl.TrialFunctions(me.Zh)
        form = dl.inner(Tang(utrial, me.normal), Tang(utest, me.normal))*me.ds_top
        me.misfit = hp.ContinuousStateObservation(me.Zh, me.ds_top, bc, form=form)
        # ds(1) basal boundary, ds(2) top boundary
        
        # construct the noisy observations
        me.misfit.d = me.pde.generate_state()
        me.misfit.d.zero()
        me.misfit.d.axpy(1.0, me.utrue)
        u_func = hp.vector2Function(me.utrue, me.Zh)

        # # eta = gaussian noise
        # ||eta||_X = c = noise_level * ||d||_X

        me.data_before_noise = np.zeros(me.misfit.d[:].shape)
        me.data_before_noise[:] = me.misfit.d[:]
        norm_data = me.data_norm_numpy(me.data_before_noise)
        if noise_type == 'relative_global':
            noise0 = np.random.randn(len(me.data_before_noise))  # has data-norm not equal 1
            noise1 = noise0 / me.data_norm_numpy(noise0)         # has data-norm=1
            me.noise = (noise_level * norm_data) * noise1        # has ||noise||_data-norm = noise_level * ||d||_data-norm
        elif noise_type == 'relative_local':
            noise0 = noise_level * np.random.randn(len(me.data_before_noise))
            me.noise = noise0 * np.abs(me.data_before_noise)
        else:
            raise RuntimeError('Invalid noise_type. valid types are relative_local, relative global')
        me.noise_datanorm = me.data_norm_numpy(me.noise)

        me.data_after_noise =  me.data_before_noise + me.noise
        me.misfit.d[:] = me.data_after_noise


        relative_noise_l2norm = np.linalg.norm(me.data_before_noise - me.data_after_noise) / np.linalg.norm(me.data_before_noise)
        print('me.noise_level=', me.noise_level, ', relative_noise_l2norm=', relative_noise_l2norm)

        relative_noise_datanorm = me.data_norm_numpy(me.data_before_noise - me.data_after_noise) / me.data_norm_numpy(me.data_before_noise)
        print('me.noise_level=', me.noise_level, ', relative_noise_datanorm=', relative_noise_datanorm)

        # misfit.noise_variance = (noise_scaling*0.05)**2.
        me.misfit.noise_variance = 1.0

        me.priorVsub = None
        me.prior = None
        me.m_func = None
        me.model = None
        me.REGOP = None

        me.set_gamma(gamma)

        me.x = [dl.Function(me.Zh).vector(), dl.Function(me.Wh).vector(), dl.Function(me.Zh).vector()]
        me.Hd = None
        me.H = None

        me.m0_Vh2_numpy = dl.interpolate(dl.Constant(me.m0_constant_value), me.Vh2).vector()[:]
        me.set_optimization_variable(me.m0_Vh2_numpy)



        # === MAP Point reconstruction ====
        #m = mtrue.copy()
        
        me.mtrue = me.mtrue.copy()

        me.HR_hmatrix = None
        me.Hd_pch = None
        me.H_pch = None
        me.iH_pch = None

    @property
    def gamma(me):
        return me.REGOP.alpha

    def reset_m(me):
        me.set_optimization_variable(me.m0_Vh2_numpy)


    # def build_preconditioner(me, num_neighbors=10, num_batches=6, tau=3.0, hmatrix_tol=1e-5):
    #     print('building PCH preconditioner')
    #     PCK = ProductConvolutionKernel(me.Vbase2D, me.Vbase2D, self.IP.apply_Hd_petsc, self.IP.apply_Hd_petsc,
    #                                    num_batches, num_batches,
    #                                    tau_rows=tau, tau_cols=tau,
    #                                    num_neighbors_rows=num_neighbors,
    #                                    num_neighbors_cols=num_neighbors)
    #     Hd_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)
    #
    #     # Rebuild reg hmatrix with same block cluster tree as PCH data misfit hmatrix
    #     print('Building Regularization H-Matrix')
    #     R_scipy = self.parameters["Rscipy"]
    #     R_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(R_scipy, Hd_pch_nonsym.bct)
    #
    #     # ----- build spd approximation of Hd, with cutoff given by a multiple of the minimum eigenvalue of the regularization operator
    #     Hd_pch = Hd_pch_nonsym.spd()
    #     H_pch = Hd_pch + R_hmatrix
    #
    #     preconditioner_hmatrix = H_pch.inv()


    def get_optimization_variable(me):
        return me.Wh_to_Vh2_numpy(me.x[1][:])

    def solve_forward(me):
        print('Assembling Stokes forward linear system')
        stokes_LHS11_petsc = dl.assemble(me.stokes_energy_gradient_lhs)
        stokes_LHS11_scipy = csr_fenics2scipy(stokes_LHS11_petsc)
        stokes_RHS1_petsc = dl.assemble(me.stokes_energy_gradient_rhs)
        stokes_RHS1_numpy = stokes_RHS1_petsc[:]

        stokes_LHS21_petsc = dl.assemble(me.stokes_constraint_gradient_lhs)
        stokes_LHS21_scipy = csr_fenics2scipy(stokes_LHS21_petsc)
        stokes_RHS2_petsc = dl.assemble(me.stokes_constraint_gradient_rhs)
        stokes_RHS2_numpy = stokes_RHS2_petsc[:]

        me.stokes_coefficient_matrix_scipy = sps.bmat([[stokes_LHS11_scipy, stokes_LHS21_scipy.T],
                                                       [stokes_LHS11_scipy, None]]).tocsr()

        me.stokes_RHS_numpy = np.concatenate([stokes_RHS1_numpy, stokes_RHS2_numpy])

        print('Factorizing Stokes coefficient matrix')
        me.stokes_coefficient_matrix_factorized_solve = spla.factorized(me.stokes_coefficient_matrix_scipy)

        print('Solving forward problem')
        me.u.vector()[:] = me.stokes_coefficient_matrix_factorized_solve(me.stokes_RHS_numpy)

    def set_m_without_solving(me, new_m_Vh2_numpy):
        me.m_Vh2.vector()[:] = new_m_Vh2_numpy
        me.m_Vh3.vector()[:] = me.Vh2_to_Vh3_numpy(new_m_Vh2_numpy)
        me.m_Wh.vector()[:] = me.Vh2_to_Wh_numpy(new_m_Vh2_numpy)

    def set_optimization_variable(me, new_m_Vh2_numpy, reset_state=True):
        me.set_m_without_solving(new_m_Vh2_numpy)
        me.solve_forward()



        if reset_state:
            me.x = [dl.Function(me.Zh).vector(), dl.Function(me.Wh).vector(), dl.Function(me.Zh).vector()]
        me.x[1][:] = me.Vh2_to_Wh_numpy(new_m_Vh2_numpy)
        me.model.solveFwd(me.x[0], me.x)
        me.model.solveAdj(me.x[2], me.x)
        # me.model.setPointForHessianEvaluations(me.x, gauss_newton_approx=True)
        me.model.setPointForHessianEvaluations(me.x, gauss_newton_approx=False)
        me.Hd   = hp.ReducedHessian(me.model, misfit_only = True)
        me.H    = hp.ReducedHessian(me.model, misfit_only = False)

    def misfit_cost(me):
        return me.model.misfit.cost(me.x)
        # return me.model.cost(me.x)

    def regularization_cost(me):
        m_numpy = me.get_optimization_variable()
        m0_numpy = me.prior_mean_Vh2_petsc[:]
        dm_numpy = m_numpy - m0_numpy
        return 0.5 * np.dot(dm_numpy, me.REGOP.apply_R_numpy(dm_numpy))


    def cost(me):
        misfit_cost = me.misfit_cost()
        reg_cost = me.regularization_cost()
        total_cost = misfit_cost + reg_cost
        return [total_cost, reg_cost, misfit_cost]

    def misfit_gradient(me):
        g_misfit_Wh_petsc = dl.Function(me.Wh).vector()
        me.model.evalGradientParameter(me.x, g_misfit_Wh_petsc, misfit_only=True)
        g_misfit_Vh2_numpy = me.Wh_to_Vh2_petsc(g_misfit_Wh_petsc)[:]
        return g_misfit_Vh2_numpy

    def regularization_gradient(me):
        m_numpy = me.get_optimization_variable()
        m0_numpy = me.prior_mean_Vh2_petsc[:]
        dm_numpy = m_numpy - m0_numpy
        return me.REGOP.apply_R_numpy(dm_numpy)

    def gradient(me):
        return me.misfit_gradient() + me.regularization_gradient()

    def apply_misfit_hessian_helper(me, u_Vh2_numpy):
        u_Wh_petsc = dl.Function(me.Wh).vector()
        u_Wh_petsc[:] = me.Vh2_to_Wh_numpy(u_Vh2_numpy)

        v_Wh_petsc = dl.Function(me.Wh).vector()
        me.Hd.mult(u_Wh_petsc, v_Wh_petsc)
        v_Vh2_numpy = me.Wh_to_Vh2_petsc(v_Wh_petsc)[:]

        return v_Vh2_numpy

    def apply_misfit_hessian(me, u_Vh2_numpy): # v = H * u
        old_gn_bool = me.Hd.gauss_newton_approx
        me.Hd.gauss_newton_approx = False

        v_Vh2_numpy = me.apply_misfit_hessian_helper(u_Vh2_numpy)

        me.Hd.gauss_newton_approx = old_gn_bool
        return v_Vh2_numpy

    def apply_misfit_gauss_newton_hessian(me, u_Vh2_numpy): # v = H * u
        old_gn_bool = me.Hd.gauss_newton_approx
        me.Hd.gauss_newton_approx = True

        v_Vh2_numpy = me.apply_misfit_hessian_helper(u_Vh2_numpy)

        me.Hd.gauss_newton_approx = old_gn_bool
        return v_Vh2_numpy

    def apply_regularization_hessian(me, u_Vh2_numpy):
        return me.REGOP.apply_R_numpy(u_Vh2_numpy)

    def apply_hessian(me, u_Vh2_numpy):
        return me.apply_misfit_hessian(u_Vh2_numpy) + me.apply_regularization_hessian(u_Vh2_numpy)

    def apply_gauss_newton_hessian(me, u_Vh2_numpy):
        return me.apply_misfit_gauss_newton_hessian(u_Vh2_numpy) + me.apply_regularization_hessian(u_Vh2_numpy)


    def set_gamma(me, new_gamma):
        # me.gamma  = new_gamma
        # me.delta = 4.*me.gamma / (me.correlation_Length**2)
        me.REGOP = BiLaplacianRegularizationOperator(new_gamma, me.correlation_Length, me.Vh2, robin_bc=me.reg_robin_bc)

        me.priorVsub  = hp.BiLaplacianPrior(me.Vh3, new_gamma, me.REGOP.delta, mean=me.prior_mean_Vh3_petsc, robin_bc=me.REGOP.robin_bc)
        me.prior      = ManifoldPrior(me.Wh, me.Vh3, me.boundary_mesh, me.priorVsub)

        me.model = hp.Model(me.pde, me.prior, me.misfit)

    def data_inner_product(me, x, y): # <x, y>_datanorm = x^T W y
        W = me.misfit.W
        Wx = dl.Vector(W.mpi_comm())
        W.init_vector(Wx, 0)
        W.mult(x, Wx)
        ytWx = Wx.inner(y)
        return ytWx

    def data_norm(me, x): # ||x|| = sqrt(<x,x>)
        return np.sqrt(me.data_inner_product(x, x.copy()))

    def data_inner_product_numpy(me, x_numpy, y_numpy):
        x = me.misfit.d.copy()
        x[:] = x_numpy
        y = me.misfit.d.copy()
        y[:] = y_numpy
        return me.data_inner_product(x, y)

    def data_norm_numpy(me, x_numpy):
        return np.sqrt(me.data_inner_product_numpy(x_numpy, x_numpy))


    # # ---- critical function !
    # # ---- needs to be called
    # def set_parameter(me, x):
    #     if isinstance(x, list):
    #         me.x = x
    #     else:
    #         me.x = [me.model.problem.generate_state(), x, me.model.problem.generate_state()]
    #         me.model.solveFwd(me.x[0], me.x)
    #         me.model.solveAdj(me.x[2], me.x)
    #     me.model.setPointForHessianEvaluations(me.x, gauss_newton_approx = True)
    #     me.Hd   = hp.ReducedHessian(me.model, misfit_only = True)
    #     me.H    = hp.ReducedHessian(me.model, misfit_only = False)
    #     me.Hd.gauss_newton_approx = True
    #     me.H.gauss_newton_approx  = True
    #     me.Hd_proj = proj_op(me.Hd, me.prior.P)
    #     me.H_proj  = proj_op(me.H , me.prior.P)


    @property
    def options(me):
        return {'mesh_h': me.mesh_h,
                'finite_element_order': me.finite_element_order,
                'noise_level': me.noise_level,
                'prior_correlation_length': me.prior_correlation_length}

    def __hash__(me):
        return hash(tuple(me.options.items()))


def get_project_root():
    return Path(__file__).parent.parent


class NonlinearStokesForm:
    def __init__(self, n, A, normal, ds_base, f, lam=0.):
        # Rheology
        self.n = n  # 1
        self.A = A  # A = 2.140373e-7

        # Basal Boundary
        self.normal = normal
        self.ds_base = ds_base

        # Forcing term
        self.f = f

        # Smooth strain
        self.eps = dl.Constant(1e-6)  #

        # penalty parameter for Dirichlet condition
        self.lam = dl.Constant(0.5 * lam)  #

    def _epsilon(self, velocity):
        return dl.sym(dl.grad(velocity))

    def _tang(self, velocity):
        return (velocity - dl.outer(self.normal, self.normal) * velocity)

    def energy_fun(self, u, m):
        velocity, _ = dl.split(u)
        normEu12 = 0.5 * dl.inner(self._epsilon(velocity), self._epsilon(velocity)) + self.eps

        return self.A ** (-1. / self.n) * ((2. * self.n) / (1. + self.n)) * (
                    normEu12 ** ((1. + self.n) / (2. * self.n))) * dl.dx \
               - dl.inner(self.f, velocity) * dl.dx \
               + dl.Constant(.5) * dl.inner(dl.exp(m) * self._tang(velocity), self._tang(velocity)) * self.ds_base \
               + self.lam * dl.inner(velocity, self.normal) ** 2 * self.ds_base

    def constraint(self, u):
        vel, pressure = dl.split(u)
        return dl.inner(-dl.div(vel), pressure) * dl.dx

    # why is the constraint added to the variational form
    # this should not be of no consequence if the constraint
    # is satisfied exactly, but is it needed at all?
    def varf_handler(self, u, m, p):
        return dl.derivative(self.energy_fun(u, m) + self.constraint(u), u, p)  # + self.constraint(u)