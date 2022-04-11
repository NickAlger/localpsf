import numpy as np
import dolfin as dl
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from pathlib import Path
import os


import meshio

import sys
sys.path.append('../../hippylib/')
import hippylib as hp
sys.path.append('/home/nick/repos/ucm-ice')
from iceModel import *


from nalger_helper_functions import *

#import subprocess
#import localpsf.path as path
#path.path()
#import sys
#sys.path.append('/home/tucker/software/ucm-ice/')
#sys.path.append('/home/tucker/software/hippylib/')
#
#import hippylib as hp
#from iceModel import *


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
                 noise_level=5e-2,
                 prior_correlation_length=0.05,
                 regularization_parameter=1e-1,
                 make_plots=True,
                 save_plots=True,
                 Newton_iterations=12,
                 misfit_only=False,
                 gauss_newton_approx=True,
                 load_fwd=False,
                 solve_inv=False,
                 lam=1.e14,
                 gamma=6.e1,
                 m0 = 7., 
                 mtrue_string = 'm0 - (m0 / 7.)*std::cos((x[0]*x[0]+x[1]*x[1])*pi/(Radius*Radius))'
                 ):
        ########    INITIALIZE OPTIONS    ########
        me.boundary_markers = boundary_markers
        me.noise_level = noise_level
        me.prior_correlation_length = prior_correlation_length
        me.regularization_parameter = regularization_parameter

        me.make_plots = make_plots
        me.save_plots = save_plots
        
        # forcing term
        grav  = 9.81           # acceleration due to gravity
        rho   = 910.0          # volumetric mass density of ice

        # rheology
        n = 3.0
        A = dl.Constant(1.e-16)

        ########    MESH    ########
        boundary_mesh = dl.BoundaryMesh(mesh, "exterior", True)
        submesh_bottom = dl.SubMesh(boundary_mesh, BasalBoundarySub())
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
        bcoords    = boundary_mesh.coordinates()
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
        Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
        Vsub = dl.FunctionSpace(submesh_bottom, 'Lagrange', 1)
        Vh = [Vh2, Vh1, Vh2]
        me.Vh = Vh

        coords = submesh_bottom.coordinates()[:,:2]
        cells = [("triangle", submesh_bottom.cells())]
        mesh2D = meshio.Mesh(coords, cells)
        mesh2D.write("mesh2D.xml")
        me.mesh = dl.Mesh("mesh2D.xml")

        Vsub2D = dl.FunctionSpace(me.mesh, 'Lagrange', 1)
        pp = Vsub.tabulate_dof_coordinates()
        qq = Vsub2D.tabulate_dof_coordinates()
        me.permVsub2Vsub2D, me.permVsub2D2Vsub = permut(pp[:,:2], qq)

        me.V = Vsub2D
        test  = dl.TestFunction(me.V)
        trial = dl.TrialFunction(me.V)
        
        me.M = dl.assemble(test*trial*dl.dx(me.V.mesh()))
        me.Mdiag = dl.Vector(dl.PETScVector(dl.as_backend_type(me.M).mat().getDiagonal()))
        me.Vsub = Vsub

        me.dof_coords = me.V.tabulate_dof_coordinates()
        me.N = me.V.dim()
        me.d = me.V.mesh().geometric_dimension()


        # ==== SET UP FORWARD MODEL ====
        # Forcing term
        f=dl.Constant( (0., 0., -rho*grav) )

        ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)
        
        normal = dl.FacetNormal(mesh)
        # Strongly enforced Dirichlet conditions. The no outflow condition will be enforced weakly, via a penalty parameter.
        bc = []
        bc0 = []
        
        # Define the Nonlinear Stokes varfs
        nonlinearStokesFunctional = NonlinearStokesForm(n, A, normal, ds(1), f, lam=lam)
        #nonlinearStokesFunctional = NonlinearStokesForm(n, A, normal, ds(1), f)
        # Create one-hot vector on pressure dofs
        constraint_vec = dl.interpolate(dl.Constant((0.,0., 0., 1.)), Vh2).vector()
        
        #x_ref = np.sqrt(2.)/4.
        #y_ref = np.sqrt(2.)/4.
        #t_ref = np.array([np.pi/4.])
        #r_ref = np.array([dilitation/2.])
        #z_ref = dilitation * (me.depth(r_ref, t_ref) * 0.5 + me.topography(r_ref, t_ref))[0]
        #normbc = NormalDirichletBC(Vh[hp.STATE], dl.Constant(0.0), boundary_markers, 1, ref_point=np.array([x_ref, y_ref, z_ref]))
        #pde = EnergyFunctionalPDEVariationalProblem(Vh, nonlinearStokesFunctional, constraint_vec, bc, bc0, normbc)
        pde = EnergyFunctionalPDEVariationalProblem(Vh, nonlinearStokesFunctional, constraint_vec, bc, bc0)
        pde.fwd_solver.parameters["rel_tolerance"] = 1.e-8
        pde.fwd_solver.parameters["print_level"] = 1
        pde.fwd_solver.parameters["LS"]["max_backtracking_iter"] = 20
        pde.fwd_solver.solver = dl.PETScLUSolver("mumps")
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

        gamma  = gamma
        #gamma = 2.e1 #6.e2
        rel_correlation_Length = 0.1
        correlation_Length = Radius*rel_correlation_Length

        delta = 4.*gamma/(correlation_Length**2)
        
        prior_mean_val = m0

        prior_mean = dl.interpolate(dl.Constant(prior_mean_val), Vsub).vector()
        me.priorVsub  = hp.BiLaplacianPrior(Vsub, gamma, delta, mean=prior_mean, robin_bc=True)
        me.prior      = ManifoldPrior(Vh[hp.PARAMETER], Vsub, boundary_mesh, me.priorVsub)
        ########    TRUE BASAL FRICTION FIELD (INVERSION PARAMETER)    ########
        m_func = dl.Expression(mtrue_string,\
                element=Vh[hp.PARAMETER].ufl_element(), m0=prior_mean_val,Radius=Radius)
        mtrue   = dl.interpolate(m_func, Vh[hp.PARAMETER]).vector()

        ########    TRUE SURFACE VELOCITY FIELD AND NOISY OBSERVATIONS    ########
        utrue = pde.generate_state()
        if not load_fwd:
            pde.solveFwd(utrue, [utrue, mtrue, None])
            np.savetxt("utrue.txt", utrue.get_local())
        else:
            utrue.set_local(np.loadtxt("utrue.txt"))
        me.utrue_fnc = dl.Function(Vh[hp.STATE], utrue)
        me.outflow = np.sqrt(dl.assemble(dl.inner(me.utrue_fnc.sub(0), normal)**2.*ds(1)))


        # construct the form which describes the continuous observations
        # of the tangential component of the velocity field at the upper surface boundary
        component_observed = dl.interpolate(dl.Constant((1., 1., 0., 0.)), Vh[hp.STATE])
        utest, ptest = dl.TestFunctions(Vh[hp.STATE])
        utrial, ptrial = dl.TrialFunctions(Vh[hp.STATE])
        form = dl.inner(Tang(utrial, normal), Tang(utest, normal))*ds(2)
        misfit = hp.ContinuousStateObservation(Vh[hp.STATE], ds(2), bc, form=form)
        
        # construct the noisy observations
        misfit.d = pde.generate_state()
        misfit.d.zero()
        misfit.d.axpy(1.0, utrue)
        u_func = hp.vector2Function(utrue, Vh[hp.STATE])
        # is this consistent with the form defined above?
        # are we using the (x,y) components or the tangential components? Consistency!
        noise_std_dev = me.noise_level*np.sqrt(dl.assemble(dl.inner(u_func, component_observed)**2.*ds(2)) / \
                                               dl.assemble(dl.Constant(1.0)*ds(2)))
        hp.parRandom.normal_perturb(noise_std_dev, misfit.d)
        misfit.noise_variance = noise_std_dev**2
        
        # ==== Define the model ====
        me.model = hp.Model(pde, me.prior, misfit)


        # === MAP Point reconstruction ====
        m = mtrue.copy()
        parameters = hp.ReducedSpaceNewtonCG_ParameterList()
        parameters["rel_tolerance"] = 1.e-6
        parameters["abs_tolerance"] = 1.e-12
        parameters["max_iter"]      = Newton_iterations
        parameters["globalization"] = "LS"
        parameters["cg_coarse_tolerance"] = 0.5

        parameters["GN_iter"] = 4
        solver = hp.ReducedSpaceNewtonCG(me.model, parameters)
        
        if solve_inv:
            me.x = solver.solve([None, m, None])
        else:
            me.x = [utrue, mtrue, utrue]
        stateMAP = hp.vector2Function(me.x[hp.STATE], Vh[hp.STATE])
        mMAP     = hp.vector2Function(me.x[hp.PARAMETER], Vh[hp.PARAMETER])
        if save_plots:
            dl.File("velocityMAP.pvd") << stateMAP.sub(0)
            dl.File("pressureMAP.pvd") << stateMAP.sub(1)
            dl.File("parameterMAP.pvd") << mMAP
        if solver.converged:
            print("\n Converged in ", solver.it, " iterations.")
        else:
            print("\n Not Converged")

        me.model.setPointForHessianEvaluations(me.x, gauss_newton_approx = gauss_newton_approx)
        Hessian = hp.ReducedHessian(me.model, misfit_only = misfit_only)

        me.Hessian_proj = proj_op(Hessian, me.prior.P)

        Hessianfull = hp.ReducedHessian(me.model, misfit_only = False)
        me.Hessianfull_proj = proj_op(Hessianfull, me.prior.P)
    
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

    def apply_Hd_Vsub(me, x):
        y = dl.Vector(x)
        me.Hessian_proj.mult(x, y)
        return y
    
    def apply_H_Vsub_numpy(me, x):
        xdl = dl.Vector()
        xdl.init(len(x))
        xdl.set_local(x)
        ydl = dl.Vector()
        ydl.init(len(x))
        me.Hessianfull_proj.mult(xdl, ydl)
        return ydl.get_local()
    def apply_Hd_Vsub_numpy(me, x):
        xdl = dl.Vector()
        xdl.init(len(x))
        xdl.set_local(x)
        ydl = dl.Vector()
        ydl.init(len(x))
        me.Hessian_proj.mult(xdl, ydl)
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

    def apply_Hd_petsc(me, x):
       Px = dl.Vector(x)
       Px.set_local(x.get_local()[me.permVsub2D2Vsub])
       y = dl.Vector(x)
       me.Hessian_proj.mult(Px, y)
       PTy = dl.Vector(x)
       PTy.set_local(y.get_local()[me.permVsub2Vsub2D])
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
                'final_time': me.final_time,
                'num_timesteps': me.num_timesteps,
                'noise_level': me.noise_level,
                'mesh_type': me.mesh_type,
                'conductivity_type': me.conductivity_type,
                'initial_condition_type': me.initial_condition_type,
                'prior_correlation_length': me.prior_correlation_length,
                'regularization_parameter': me.regularization_parameter}

    def __hash__(me):
        return hash(tuple(me.options.items()))


def get_project_root():
    return Path(__file__).parent.parent


