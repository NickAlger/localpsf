import numpy as np
import dolfin as dl
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from pathlib import Path
import os

import meshio

from nalger_helper_functions import *

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
                 m0 = 7., 
                 mtrue_string = 'm0 - (m0 / 7.)*std::cos((x[0]*x[0]+x[1]*x[1])*pi/(Radius*Radius))',
                 rel_correlation_Length = 0.05,
                 noise_type = 'relative_local',
                 robin_bc=True
                 ):
        ########    INITIALIZE OPTIONS    ########
        me.boundary_markers = boundary_markers
        me.noise_level = noise_level
        me.prior_correlation_length = prior_correlation_length
        me.built_R_hmatrix = False
        me.R_hmatrix      = None
        me.make_plots = make_plots
        me.save_plots = save_plots
        me.robin_bc = robin_bc
        
        # forcing term
        grav  = 9.81           # acceleration due to gravity
        rho   = 910.0          # volumetric mass density of ice

        # rheology
        n = 3.0
        A = dl.Constant(1.e-16)

        ########    MESH    ########
        me.boundary_mesh = dl.BoundaryMesh(mesh, "exterior", True)
        submesh_bottom = dl.SubMesh(me.boundary_mesh, BasalBoundarySub())
        me.r0            = 0.05
        me.sig           = 0.4
        me.valleys       = 4
        me.valley_depth  = 0.35
        me.bump_height   = 0.2
        me.min_thickness = 0.08 / 8.
        me.avg_thickness = 0.2 / 8.
        me.theta         = -np.pi/2.
        me.max_thickness = me.avg_thickness + (me.avg_thickness - me.min_thickness) 
        me.A_thickness   = me.max_thickness - me.avg_thickness
        
        dilitation = 1.e4
        Length = 1.
        Width  = 1.
        Length *= 2*dilitation
        Width  *= 2*dilitation
        Radius  = dilitation

        coords     = mesh.coordinates()
        bcoords    = me.boundary_mesh.coordinates()
        subbcoords = submesh_bottom.coordinates()
        coord_sets = [coords, bcoords, subbcoords]

        for k in range(len(coord_sets)):
            for i in range(len(coord_sets[k])):
                x,y,z = coord_sets[k][i]
                r     = np.sqrt(x**2 + y**2)
                t     = np.arctan2(y, x)
                coord_sets[k][i,2] = me.depth(r,t)*z + me.topography(r,t)
                coord_sets[k][i]  *= dilitation


        ########    FUNCTION SPACE    ########
        P1 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        TH = P2 * P1
        Vh2 = dl.FunctionSpace(mesh, TH) # product space for state + adjoint
        me.Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
        me.Vbase3D = dl.FunctionSpace(submesh_bottom, 'Lagrange', 1)
        Vh = [Vh2, me.Vh1, Vh2] # state, parameter, adjoint
        me.Vh = Vh

        #------ generate 2D mesh from 3D boundary subset mesh
        coords = submesh_bottom.coordinates()[:,:2]
        cells = [("triangle", submesh_bottom.cells())]
        mesh2D = meshio.Mesh(coords, cells)
        mesh2D.write("mesh2D.xml")
        me.mesh = dl.Mesh("mesh2D.xml")

        me.Vbase2D = dl.FunctionSpace(me.mesh, 'Lagrange', 1)
        pp_Vbase3d = me.Vbase3D.tabulate_dof_coordinates()
        pp_Vbase2d = me.Vbase2D.tabulate_dof_coordinates()
        me.perm_Vbase3d_to_Vbase2d, me.perm_Vbase2d_to_Vbase3d = permut(pp_Vbase3d[:,:2], pp_Vbase2d)

        #

        pp_Vh1 = me.Vh1.tabulate_dof_coordinates()

        KDT = KDTree(pp_Vh1)
        me.basal_inds = KDT.query(pp_Vbase3d)[1]
        if np.linalg.norm(pp_Vbase3d - pp_Vh1[me.basal_inds, :]) > 1.e-10:
            warnings.warn('problem with basal_inds')

        #


        me.V  = me.Vbase2D
        test  = dl.TestFunction(me.V)
        trial = dl.TrialFunction(me.V)
        
        me.M = dl.assemble(test*trial*dl.dx(me.V.mesh()))
        me.Mdiag = dl.Vector(dl.PETScVector(dl.as_backend_type(me.M).mat().getDiagonal()))

        me.dof_coords = me.V.tabulate_dof_coordinates()
        me.N = me.V.dim()
        me.d = me.V.mesh().geometric_dimension()


        # ==== SET UP FORWARD MODEL ====
        # Forcing term
        f=dl.Constant( (0., 0., -rho*grav) )

        ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)
        me.ds = ds
        normal = dl.FacetNormal(mesh)
        # Strongly enforced Dirichlet conditions. The no outflow condition will be enforced weakly, via a penalty parameter.
        bc  = []
        bc0 = []

        # hp.PDEVariationalProblem

        # Define the Nonlinear Stokes varfs
        nonlinearStokesFunctional = NonlinearStokesForm(n, A, normal, ds(1), f, lam=lam)
        
        # Create one-hot vector on pressure dofs
        constraint_vec = dl.interpolate(dl.Constant((0.,0., 0., 1.)), Vh2).vector()
        
        me.pde = EnergyFunctionalPDEVariationalProblem(Vh, nonlinearStokesFunctional, constraint_vec, bc, bc0)
        me.pde.fwd_solver.parameters["rel_tolerance"] = 1.e-8
        me.pde.fwd_solver.parameters["print_level"] = 1
        me.pde.fwd_solver.parameters["LS"]["max_backtracking_iter"] = 20
        me.pde.fwd_solver.solver = dl.PETScLUSolver("mumps")
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

        me.prior_mean_val = m0
        me.prior_mean     = dl.interpolate(dl.Constant(me.prior_mean_val), me.Vbase3D).vector()
        me.m_func = dl.Expression(mtrue_string,\
                                  element=Vh[hp.PARAMETER].ufl_element(), m0=me.prior_mean_val,Radius=Radius)
        mtrue   = dl.interpolate(me.m_func, Vh[hp.PARAMETER]).vector()

        me.rel_correlation_Length = rel_correlation_Length # 0.1
        me.correlation_Length = Radius * me.rel_correlation_Length

        ########    TRUE BASAL FRICTION FIELD (INVERSION PARAMETER)    ########


        ########    TRUE SURFACE VELOCITY FIELD AND NOISY OBSERVATIONS    ########
        me.utrue = me.pde.generate_state()
        if not load_fwd:
            me.pde.solveFwd(me.utrue, [me.utrue, mtrue, None])
            np.savetxt("utrue.txt", me.utrue.get_local())
        else:
            me.utrue.set_local(np.loadtxt("utrue.txt"))
        utrue_fnc = dl.Function(Vh[hp.STATE], me.utrue)
        me.outflow = np.sqrt(dl.assemble(dl.inner(utrue_fnc.sub(0), normal)**2.*ds(1)))


        # construct the form which describes the continuous observations
        # of the tangential component of the velocity field at the upper surface boundary
        component_observed = dl.interpolate(dl.Constant((1., 1., 0., 0.)), Vh[hp.STATE])
        utest, ptest = dl.TestFunctions(Vh[hp.STATE])
        utrial, ptrial = dl.TrialFunctions(Vh[hp.STATE])
        form = dl.inner(Tang(utrial, normal), Tang(utest, normal))*ds(2)
        me.misfit = hp.ContinuousStateObservation(Vh[hp.STATE], ds(2), bc, form=form)
        # ds(1) basal boundary, ds(2) top boundary
        
        # construct the noisy observations
        me.misfit.d = me.pde.generate_state()
        me.misfit.d.zero()
        me.misfit.d.axpy(1.0, me.utrue)
        u_func = hp.vector2Function(me.utrue, Vh[hp.STATE])

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

        # is this consistent with the form defined above?
        # are we using the (x,y) components or the tangential components? Consistency!
        # noise_scaling = np.sqrt(dl.assemble(dl.inner(u_func, component_observed)**2.*ds(2)) / \
        #                         dl.assemble(dl.Constant(1.0)*ds(2)))
        # misfit.cost()
        # before_noise_data = misfit.d[:]
        # if me.noise_level > 0.:
        #     hp.parRandom.normal_perturb(me.noise_level*noise_scaling, misfit.d)
        # after_noise_data = misfit.d[:]

        # D = [I 0] <-- basal      (m)
        #     [0 0] <-- non-basal  (N-m)

        # ||randn||_I = sqrt(N)
        # ||randn||_D = sqrt(m)
        # ||randn||_D = sqrt(m)/sqrt(N) ||randn||_I
        # N = 8*m => sqrt(m)/sqrt(N) = 1/sqrt(8) =approx= 0.35
        # ||randn||_D = 0.35 * ||randn||_I

        relative_noise_l2norm = np.linalg.norm(me.data_before_noise - me.data_after_noise) / np.linalg.norm(me.data_before_noise)
        print('me.noise_level=', me.noise_level, ', relative_noise_l2norm=', relative_noise_l2norm)

        relative_noise_datanorm = me.data_norm_numpy(me.data_before_noise - me.data_after_noise) / me.data_norm_numpy(me.data_before_noise)
        print('me.noise_level=', me.noise_level, ', relative_noise_datanorm=', relative_noise_datanorm)

        # misfit.noise_variance = (noise_scaling*0.05)**2.
        me.misfit.noise_variance = 1.0


        me.gamma = None #smaller gamma, smaller regularization
        # me.delta = None
        me.priorVsub = None
        me.prior = None
        me.m_func = None
        me.model = None

        me.set_gamma(gamma)

        # === MAP Point reconstruction ====
        #m = mtrue.copy()
        
        me.mtrue = mtrue.copy()
        me.x = None
        me.Hd = None
        me.H = None
        me.Hd_proj = None
        me.H_proj  = None

        # Regularization
        ubase2d_trial = dl.TrialFunction(me.Vbase2D)
        vbase2d_test = dl.TestFunction(me.Vbase2D)

        me.kbase2d_form = dl.inner(dl.grad(ubase2d_trial), dl.grad(vbase2d_test)) * dl.dx
        me.mbase2d_form = dl.inner(ubase2d_trial, vbase2d_test) * dl.dx
        me.robinbase2d_form = dl.inner(ubase2d_trial, vbase2d_test) * dl.ds

        me.Mbase2d_petsc = dl.assemble(me.mbase2d_form)
        me.Mbase2d_scipy = csr_scipy2fenics(me.Mbase2d_petsc)

        me.HRsqrt_petsc = None
        me.HRsqrt_scipy = None

        me.solve_Mbase2d_numpy = None
        me.solve_Rsqrt_numpy = None
        me.HR_scipy = None

        me.HR_hmatrix = None
        me.Hd_pch = None
        me.H_pch = None
        me.iH_pch = None


    def build_preconditioner(me, num_neighbors=10, num_batches=6, tau=3.0, hmatrix_tol=1e-5):
        print('building PCH preconditioner')
        PCK = ProductConvolutionKernel(me.Vbase2D, me.Vbase2D, self.IP.apply_Hd_petsc, self.IP.apply_Hd_petsc,
                                       num_batches, num_batches,
                                       tau_rows=tau, tau_cols=tau,
                                       num_neighbors_rows=num_neighbors,
                                       num_neighbors_cols=num_neighbors)
        Hd_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)

        # Rebuild reg hmatrix with same block cluster tree as PCH data misfit hmatrix
        print('Building Regularization H-Matrix')
        R_scipy = self.parameters["Rscipy"]
        R_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(R_scipy, Hd_pch_nonsym.bct)

        # ----- build spd approximation of Hd, with cutoff given by a multiple of the minimum eigenvalue of the regularization operator
        Hd_pch = Hd_pch_nonsym.spd()
        H_pch = Hd_pch + R_hmatrix

        preconditioner_hmatrix = H_pch.inv()

    def apply_HR_numpy(me, ubase2d_numpy):
        return me.HRsqrt_scipy @ me.solve_Mbase2d_numpy(me.HRsqrt_scipy @ ubase2d_numpy)

    def solve_HR_numpy(me, vbase2d_numpy):
        me.solve_HRsqrt_numpy(me.Mbase2d_scipy @ me.solve_HRsqrt_numpy(vbase2d_numpy))

    def get_optimization_variable(me):
        return me.Vh1_to_Vbase2d_numpy(me.x[1][:])

    def set_optimization_variable(me, new_m2d_numpy):
        me.x[1][:] = me.Vbase2d_to_Vh1_numpy(new_m2d_numpy)
        me.model.solveFwd(me.x[0], me.x)
        me.model.solveAdj(me.x[2], me.x)
        me.model.setPointForHessianEvaluations(me.x, gauss_newton_approx = True)
        me.Hd   = hp.ReducedHessian(me.model, misfit_only = True)
        me.H    = hp.ReducedHessian(me.model, misfit_only = False)
        me.Hd.gauss_newton_approx = True
        me.H.gauss_newton_approx  = True
        me.Hd_proj = proj_op(me.Hd, me.prior.P)
        me.H_proj  = proj_op(me.H , me.prior.P)
        me.update_regularization()

    def update_regularization(me):
        me.Rsqrt_petsc = dl.assemble(me.base2d_bilaplacian_form)
        me.Rsqrt_scipy = csr_fenics2scipy(me.Rsqrt_petsc)
        me.solve_Rsqrt_numpy = spla.factorized(me.Rsqrt_scipy)
        me.solve_Mbase2d_numpy = spla.factorized(me.Mbase2d_scipy)

    def cost(me):
        return me.model.cost(me.x)

    def gradient(me):
        g_Vh1_petsc = dl.Vector()
        g_Vh1_petsc.init(me.x[hp.PARAMETER].size())
        me.model.evalGradientParameter(me.x, g_Vh1_petsc)
        g_Vbase2d_numpy = me.Vh1_to_Vbase2d_numpy(g_Vh1_petsc[:])
        return g_Vbase2d_numpy

    def apply_hessian(me, ubase2d_numpy): # v = H * u
        old_gn_bool = me.H.gauss_newton_approx
        me.H.gauss_newton_approx = False

        u_petsc = dl.Function(me.Vh1).vector()
        u_petsc[:] = me.Vbase2d_to_Vh1_numpy(ubase2d_numpy)
        v_petsc = dl.Function(me.Vh1).vector()
        me.H.mult(u_petsc, v_petsc)
        vbase2d_numpy = me.Vh1_to_Vbase2d_numpy(v_petsc)

        me.H.gauss_newton_approx = old_gn_bool
        return vbase2d_numpy

    def apply_gauss_newton_hessian(me, ubase2d_numpy):  # v = Hgn * u
        old_gn_bool = me.H.gauss_newton_approx
        me.H.gauss_newton_approx = True

        u_petsc = dl.Function(me.Vh1).vector()
        u_petsc[:] = me.Vbase2d_to_Vh1_numpy(ubase2d_numpy)
        v_petsc = dl.Function(me.Vh1).vector()
        me.H.mult(u_petsc, v_petsc)
        vbase2d_numpy = me.Vh1_to_Vbase2d_numpy(v_petsc)

        me.H.gauss_newton_approx = old_gn_bool
        return vbase2d_numpy

    @property
    def delta(me):
        return 4. * me.gamma / (me.correlation_Length ** 2)

    @property
    def robin_coeff(me):
        if me.robin_bc:
            return me.gamma * np.sqrt(me.delta / me.gamma) / 1.42
        else:
            return 0.

    @property
    def base2d_bilaplacian_form(me):
        return dl.Constant(me.gamma) * me.kbase2d_form + dl.Constant(me.delta) * me.mbase2d_form + dl.Constant(me.robin_coeff) * me.robinbase2d_form


    def set_gamma(me, new_gamma):
        me.gamma  = new_gamma
        # me.delta = 4.*me.gamma / (me.correlation_Length**2)

        me.priorVsub  = hp.BiLaplacianPrior(me.Vbase3D, me.gamma, me.delta, mean=me.prior_mean, robin_bc=me.robin_bc)
        me.prior      = ManifoldPrior(me.Vh[hp.PARAMETER], me.Vbase3D, me.boundary_mesh, me.priorVsub)

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


    # ---- critical function !
    # ---- needs to be called
    def set_parameter(me, x):
        if isinstance(x, list):
            me.x = x
        else:
            me.x = [me.model.problem.generate_state(), x, me.model.problem.generate_state()]
            me.model.solveFwd(me.x[0], me.x)
            me.model.solveAdj(me.x[2], me.x)
        me.model.setPointForHessianEvaluations(me.x, gauss_newton_approx = True)
        me.Hd   = hp.ReducedHessian(me.model, misfit_only = True)
        me.H    = hp.ReducedHessian(me.model, misfit_only = False)
        me.Hd.gauss_newton_approx = True
        me.H.gauss_newton_approx  = True
        me.Hd_proj = proj_op(me.Hd, me.prior.P)
        me.H_proj  = proj_op(me.H , me.prior.P)
    """
    pass a [u, p, m] where m is a full space parameter field and project it down
    to be compatible with this class/Hessian action
    """
    def g_numpy(me, x):
        g = dl.Vector()
        g.init(x[hp.PARAMETER].size())
        me.model.evalGradientParameter(x, g)
        # projected Vector
        Pg = dl.Vector()
        me.prior.P.init_vector(Pg, 0)
        me.prior.P.mult(g, Pg)
        return Pg.get_local()

    def apply_Hd(me, x):
        y = dl.Vector(x)
        me.Hd_proj.mult(x, y)
        return y

    def apply_Hd_numpy(me, x):
        xdl = dl.Vector()
        xdl.init(len(x))
        xdl.set_local(x)
        ydl = dl.Vector()
        ydl.init(len(x))
        me.Hd_proj.mult(xdl, ydl)
        return ydl.get_local()

    def apply_H_numpy(me, x):
        xdl = dl.Vector()
        xdl.init(len(x))
        xdl.set_local(x)
        ydl = dl.Vector()
        ydl.init(len(x))
        me.H_proj.mult(xdl, ydl)
        return ydl.get_local()

    def apply_Rinv_petsc(me, x):
        y = dl.Vector(x)
        me.priorVsub.Rsolver.solve(y, x)
        return y

    def apply_Rinv_numpy(me, x):
        xdl = dl.Vector()
        xdl.init(len(x))
        xdl.set_local(x)
        ydl = me.apply_Rinv_petsc(xdl)
        return ydl.get_local()

    def Vh1_to_Vbase3d_numpy(me, u_numpy):
        ubase3d_numpy = np.zeros(me.Vbase3D.dim())
        ubase3d_numpy[:] = u_numpy[me.basal_inds]
        return ubase3d_numpy

    def Vbase3d_to_Vh1_numpy(me, ubase3d_numpy):
        u_numpy = np.zeros(me.Vh1.dim())
        u_numpy[me.basal_inds] = ubase3d_numpy
        return u_numpy

    def Vh1_to_Vbase2d_numpy(me, u_numpy):
        return me.Vbase3d_to_Vbase2d_numpy(me.Vh1_to_Vbase3d_numpy(u_numpy))

    def Vbase2d_to_Vh1_numpy(me, ubase2d_numpy):
        return me.Vbase3d_to_Vh1_numpy(me.Vbase2d_to_Vbase3d_numpy(ubase2d_numpy))

    def Vh1_to_Vbase2d_petsc(me, u_petsc):
        ubase2d_petsc = dl.Function(me.Vbase2D).vector()
        ubase2d_petsc[:] = me.Vh1_to_Vbase2d_numpy(u_petsc[:])
        return ubase2d_petsc

    def Vbase2d_to_Vh1_petsc(me, ubase2d_petsc):
        u_petsc = dl.Function(me.Vh1).vector()
        u_petsc[:] = me.Vbase2d_to_Vh1_numpy(ubase2d_petsc[:])
        return u_petsc

    def Vbase2d_to_Vbase3d_numpy(me, ubase2d_numpy):
        ubase3d_numpy = ubase2d_numpy[me.perm_Vbase2d_to_Vbase3d]
        return ubase3d_numpy

    def Vbase3d_to_Vbase2d_numpy(me, ubase3d_numpy):
        ubase2d_numpy = ubase3d_numpy[me.perm_Vbase3d_to_Vbase2d]
        return ubase2d_numpy

    def Vbase2d_to_Vbase3d_petsc(me, ubase2d_petsc):
        ubase_3d_petsc = dl.Vector(ubase2d_petsc)
        ubase_3d_petsc[:] = me.Vbase2d_to_Vbase3d_numpy(ubase2d_petsc[:])
        return ubase_3d_petsc

    def Vbase3d_to_Vbase2d_petsc(me, ubase3d_petsc):
        ubase_2d_petsc = dl.Vector(ubase3d_petsc)
        ubase_2d_petsc[:] = me.Vbase3d_to_Vbase2d_numpy(ubase3d_petsc[:])
        return ubase_2d_petsc

    def apply_Hd_petsc(me, x):
       Px = dl.Vector(x)
       Px.set_local(x.get_local()[me.perm_Vbase2d_to_Vbase3d])
       y = dl.Vector(x)
       me.Hd_proj.mult(Px, y)
       PTy = dl.Vector(x)
       PTy.set_local(y.get_local()[me.perm_Vbase3d_to_Vbase2d])
       return PTy

    def apply_Minv_petsc(me, x):
        return diagSolve(me.Mdiag, x)

    def topography(me, r, t):
        zero = np.zeros(r.shape)
        R0   = me.r0*np.ones(r.shape)
        return me.bump_height*np.exp(-(r/me.sig)**2)*(1.+me.valley_depth*np.sin(me.valleys*t-me.theta)*np.fmax(zero, (r-R0)/me.sig))
    
    def depth(me, r, t):
         zero = np.zeros(r.shape)
         R0   = me.r0*np.ones(r.shape)
         return me.min_thickness - me.A_thickness*np.sin(me.valleys*t-me.theta)*np.exp(-(r/me.sig)**2)*np.fmax(zero, (r-R0)/me.sig)


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


