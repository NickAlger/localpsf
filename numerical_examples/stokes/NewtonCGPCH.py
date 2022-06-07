# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import math
import hippylib as hp
from hippylib.utils.parameterList import ParameterList
from hippylib.modeling.reducedHessian import ReducedHessian
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from cgsolverSteihaugPCH import CGSolverSteihaug
from op_operations import *

import hlibpro_python_wrapper as hpro
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel
import scipy.sparse.linalg as spla
import numpy as np

class NCGConvergenceInfo:
    def __init__(me, extra_info=None):
        me.extra_info=extra_info
        me._cg_iteration = list()
        me._hessian_matvecs = list()
        me._total_cost = list()
        me._misfit_cost = list()
        me._reg_cost = list()
        me._mg_mhat = list()
        me._gradnorm = list()
        me._step_length_alpha = list()
        me._tolcg = list()

        me.labels = ['cg_iteration', 'hessian_matvecs', 'total_cost', 'misfit_cost', 'reg_cost', 'mg_mhat', 'gradnorm', 'step_length_alpha', 'tolcg']

    @property
    def num_newton_iterations(me):
        return len(me.cg_iteration)

    @property
    def cg_iteration(me):
        return np.array(me._cg_iteration)

    @property
    def hessian_matvecs(me):
        return np.array(me._hessian_matvecs)

    @property
    def total_cost(me):
        return np.array(me._total_cost)

    @property
    def misfit_cost(me):
        return np.array(me._misfit_cost)

    @property
    def reg_cost(me):
        return np.array(me._reg_cost)

    @property
    def mg_mhat(me):
        return np.array(me._mg_mhat)

    @property
    def gradnorm(me):
        return np.array(me._gradnorm)

    @property
    def step_length_alpha(me):
        return np.array(me._step_length_alpha)

    @property
    def tolcg(me):
        return np.array(me._tolcg)

    @property
    def data(me):
        return np.array([me.cg_iteration,
                         me.hessian_matvecs,
                         me.total_cost,
                         me.misfit_cost,
                         me.reg_cost,
                         me.mg_mhat,
                         me.gradnorm,
                         me.step_length_alpha,
                         me.tolcg]).T


    def add_iteration(me,
                      cg_iteration,
                      hessian_matvecs,
                      total_cost,
                      misfit_cost,
                      reg_cost,
                      mg_mhat,
                      gradnorm,
                      step_length_alpha,
                      tolcg):
        me._cg_iteration.append(cg_iteration)
        me._hessian_matvecs.append(hessian_matvecs)
        me._total_cost.append(total_cost)
        me._misfit_cost.append(misfit_cost)
        me._reg_cost.append(reg_cost)
        me._mg_mhat.append(mg_mhat)
        me._gradnorm.append(gradnorm)
        me._step_length_alpha.append(step_length_alpha)
        me._tolcg.append(tolcg)

    def print(me, return_string=False):
        print_string = 'Newton CG convergence information:'

        print_string += '\n'

        if me.extra_info is not None:
            print_string += me._extra_info

        print_string += "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:14} {7:14} {8:14}".format(
            "It", "cg_it", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "alpha", "tolcg")

        for k in range(me.num_newton_iterations):
            print_string += '\n'
            print_string += "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e}".format(
                me._cg_iteration[k],
                me._hessian_matvecs[k],
                me._total_cost[k],
                me._misfit_cost[k],
                me._reg_cost[k],
                me._mg_mhat[k],
                me._gradnorm[k],
                me._step_length_alpha[k],
                me._tolcg[k])

        print(print_string)
        if return_string:
            return print_string


    def savetxt(me, filename_prefix):
        np.savetxt(filename_prefix + '_data.txt', me.data)
        np.savetxt(filename_prefix + '_labels.txt', me.labels)




def LS_ParameterList():
    """
    Generate a ParameterList for line search globalization.
    type: :code:`LS_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [10, "Maximum number of backtracking iterations"]
    
    return ParameterList(parameters)

def TR_ParameterList():
    """
    Generate a ParameterList for Trust Region globalization.
    type: :code:`RT_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["eta"] = [0.05, "Reject step if (actual reduction)/(predicted reduction) < eta"]
    
    return ParameterList(parameters)

def ReducedSpaceNewtonCG_ParameterList():
    """
    Generate a ParameterList for ReducedSpaceNewtonCG.
    type: :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-18, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [20, "maximum number of iterations"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["initial_iter"]          = [3, "Number of Gauss Newton iterations before preconditioner is build"]
    parameters["GN_iter"]               = [3, "Number of Gauss Newton iterations after preconditioner is build (before switching to full Newton)"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    parameters["cg_max_iter"]           = [400, "Maximum CG iterations"]
    parameters["LS"]                    = [LS_ParameterList(), "Sublist containing LS globalization parameters"]
    parameters["TR"]                    = [TR_ParameterList(), "Sublist containing TR globalization parameters"]

    # ------ new parameters
    parameters["projector"]             = [None, "Projector (FEniCS format)"]
    parameters["Rscipy"]                = [None, "Regularization (projected space) utilized with the PCH scheme"]
    parameters["Pscipy"]                = [None, "Projector (scipy format)"]
    parameters["PCH_precond"]           = [False, "Utilze PCH preconditioning to solve each Newton system"]
    parameters["IP"]                    = [None, "Inverse problem as written for the localpsf framework"]
    parameters["num_neighbors"]         = [10, "num batches used for impulse response interpolation"]
    parameters["num_batches"]           = [5,  "num of PCH batches"]
    parameters["krylov_recycling"]      = [True, "Update preconditioner with Krylov information after each Newton step"]
    return ParameterList(parameters)
  
    

class ReducedSpaceNewtonCG:
    
    """
    Inexact Newton-CG method to solve constrained optimization problems in the reduced parameter space.
    The Newton system is solved inexactly by early termination of CG iterations via Eisenstat-Walker
    (to prevent oversolving) and Steihaug (to avoid negative curvature) criteria.
    Globalization is performed using one of the following methods:

    - line search (LS) based on the armijo sufficient reduction condition; or
    - trust region (TR) based on the prior preconditioned norm of the update direction.

    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.
       
    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient and the Hessian.
    
    More specifically the model object should implement following methods:
    
       - :code:`generate_vector()` -> generate the object containing state, parameter, adjoint
       - :code:`cost(x)` -> evaluate the cost functional, report regularization part and misfit separately
       - :code:`solveFwd(out, x)` -> solve the possibly non linear forward problem
       - :code:`solveAdj(out, x)` -> solve the linear adjoint problem
       - :code:`evalGradientParameter(x, out)` -> evaluate the gradient of the parameter and compute its norm
       - :code:`setPointForHessianEvaluations(x)` -> set the state to perform hessian evaluations
       - :code:`solveFwdIncremental(out, rhs)` -> solve the linearized forward problem for a given :code:`rhs`
       - :code:`solveAdjIncremental(out, rhs)` -> solve the linear adjoint problem for a given :code:`rhs`
       - :code:`applyC(dm, out)`    --> Compute out :math:`= C_x dm`
       - :code:`applyCt(dp, out)`   --> Compute out = :math:`C_x  dp`
       - :code:`applyWuu(du,out)`   --> Compute out = :math:`(W_{uu})_x  du`
       - :code:`applyWmu(dm, out)`  --> Compute out = :math:`(W_{um})_x  dm`
       - :code:`applyWmu(du, out)`  --> Compute out = :math:`W_{mu}  du`
       - :code:`applyR(dm, out)`    --> Compute out = :math:`R  dm`
       - :code:`applyWmm(dm,out)`   --> Compute out = :math:`W_{mm} dm`
       - :code:`Rsolver()`          --> A solver for the regularization term
       
    Type :code:`help(Model)` for additional information
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, dm) less than tolerance"       #3
                           ]
    
    def __init__(self, model, parameters=ReducedSpaceNewtonCG_ParameterList(), callback = None):
        """
        Initialize the ReducedSpaceNewtonCG.
        Type :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for list of default parameters
        and their descriptions.
        
        Parameters:
        :code:`model` The model object that describes the inverse problem
        :code:`parameters`: (type :code:`ParameterList`, optional) set parameters for inexact Newton CG
        :code:`callback`: (type function handler with signature :code:`callback(it: int, x: list of dl.Vector): --> None`
               optional callback function to be called at the end of each iteration. Takes as input the iteration number, and
               the list of vectors for the state, parameter, adjoint.
        """
        self.model = model
        self.parameters = parameters
        
        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0
        
        self.callback = callback

        self.IP = self.parameters["IP"]
        self.PCHinitialized = None # for which Newton iteration was the PCH approximation first constructed
        self.P  = self.parameters["projector"]
        self.residualCounts = []

    def solve(self, x):
        """

        Input: 
            :code:`x = [u, m, p]` represents the initial guess (u and p may be None). 
            :code:`x` will be overwritten on return.
        """
        if self.model is None:
            raise TypeError("model can not be of type None.")
        
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
            
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
            
        if self.parameters["globalization"] == "LS":
            return self._solve_ls(x)
        elif self.parameters["globalization"] == "TR":
            return self._solve_tr(x)
        else:
            raise ValueError(self.parameters["globalization"])
        
    def _solve_ls(self,x):
        """
        Solve the constrained optimization problem with initial guess :code:`x`.
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        print_level = self.parameters["print_level"]
        initial_iter = self.parameters["initial_iter"]
        GN_iter = self.parameters["GN_iter"]
        krylov_recycling = self.parameters["krylov_recycling"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        cg_max_iter         = self.parameters["cg_max_iter"]
        PCH_precond = self.parameters["PCH_precond"]
        
        c_armijo = self.parameters["LS"]["c_armijo"]
        max_backtracking_iter = self.parameters["LS"]["max_backtracking_iter"]

        convergence_info = NCGConvergenceInfo()

        self.model.solveFwd(x[STATE], x)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        mhat = self.model.generate_vector(PARAMETER)    
        mg = self.model.generate_vector(PARAMETER)
                
        x_star = [None, None, None] + x[3::]
        x_star[STATE]     = self.model.generate_vector(STATE)
        x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
        
        cost_old, _, _ = self.model.cost(x)

        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(x[ADJOINT], x)

            using_gauss_newton = (self.it < (initial_iter + GN_iter))
            print('initial_iter=', initial_iter, ', GN_iter=', GN_iter, ', self.it=', self.it, ', using_gauss_newton=', using_gauss_newton)

            self.model.setPointForHessianEvaluations(x, gauss_newton_approx=using_gauss_newton )
            gradnorm = self.model.evalGradientParameter(x, mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break

            HessApply = ReducedHessian(self.model)
            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))


            if (self.it == initial_iter) and PCH_precond:
                print('building PCH preconditioner')
                self.PCHinitialized = self.it
                # build PCH approximation
                # ----- NEED a mechanism to set the parameter
                # ----- also just pass the entire inverse problem object
                self.IP.set_parameter(x)
                num_neighbors = self.parameters["num_neighbors"]
                num_batches = self.parameters["num_batches"]
                tau = 3.0
                hmatrix_tol = 1.e-5
                PCK = ProductConvolutionKernel(self.IP.V, self.IP.V, self.IP.apply_Hd_petsc, self.IP.apply_Hd_petsc,
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


            if (self.it > initial_iter) and PCH_precond and krylov_recycling:
                X = solver.X
                Y = solver.Y
                if self.P is not None:
                    xtest  = dl.Vector()
                    ytest  = dl.Vector()
                    xtest.init(self.IP.Vh[PARAMETER].dim())
                    ytest.init(self.IP.Vh[PARAMETER].dim())
                    xtestP = dl.Vector()
                    ytestP = dl.Vector()
                    xtestP.init(self.IP.N)
                    ytestP.init(self.IP.N)
                    Xproj = []
                    Yproj = []
                    for i in range(len(X)):
                        xtest[:] = X[i]
                        ytest[:] = Y[i]
                        self.IP.prior.P.mult(xtest, xtestP)
                        self.IP.prior.P.mult(ytest, ytestP)
                        Xproj.append(xtestP[:])
                        Yproj.append(ytestP[:])
                    Xproj = np.array(Xproj).T
                    Yproj = np.array(Yproj).T
                    preconditioner_hmatrix = preconditioner_hmatrix.dfp_update(Yproj, Xproj)
                else:
                    X = np.array(X).T
                    Y = np.array(Y).T
                    preconditioner_hmatrix = preconditioner_hmatrix.dfp_update(Y, X)
            
            solver = CGSolverSteihaug(comm = self.model.prior.R.mpi_comm())
            solver.set_operator(HessApply)

            # if self.parameters["PCH_precond"] and self.total_cg_iter > 0:
            if self.parameters["PCH_precond"] and (self.it >= initial_iter):
                preconditioner_linop = preconditioner_hmatrix.as_linear_operator()
                H_pch_precond = solver_from_dot(preconditioner_linop)
                if self.P is not None:
                    H_pch_fullspace_precond = projected_solver(self.P, H_pch_precond) 
                    solver.set_preconditioner(H_pch_fullspace_precond)
                else:
                    solver.set_preconditioner(H_pch_precond)
            else:
                solver.set_preconditioner(self.model.Rsolver())
            self.it += 1
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["max_iter"] = cg_max_iter
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(mhat, -mg)
            self.total_cg_iter += HessApply.ncalls
            self.residualCounts.append(HessApply.ncalls)
            
            alpha = 1.0
            descent = 0
            n_backtrack = 0
            
            mg_mhat = mg.inner(mhat)
            
            while descent == 0 and n_backtrack < max_backtracking_iter:
                # update m and u
                x_star[PARAMETER].zero()
                x_star[PARAMETER].axpy(1., x[PARAMETER])
                x_star[PARAMETER].axpy(alpha, mhat)
                x_star[STATE].zero()
                x_star[STATE].axpy(1., x[STATE])
                self.model.solveFwd(x_star[STATE], x_star)
                
                cost_new, reg_new, misfit_new = self.model.cost(x_star)
                  
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + alpha * c_armijo * mg_mhat) or (-mg_mhat <= self.parameters["gdm_tolerance"]):
                    cost_old = cost_new
                    descent = 1
                    x[PARAMETER].zero()
                    x[PARAMETER].axpy(1., x_star[PARAMETER])
                    x[STATE].zero()
                    x[STATE].axpy(1., x_star[STATE])
                else:
                    n_backtrack += 1
                    alpha *= 0.5

            convergence_info.add_iteration(self.it, HessApply.ncalls, cost_new, misfit_new, reg_new, mg_mhat, gradnorm, alpha, tolcg)

            if print_level >= 0:
                convergence_info.print()

            if self.callback:
                self.callback(self.it, x)

            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
            
            if -mg_mhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break

        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new
        return x

    def _solve_tr(self,x):
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        cg_max_iter         = self.parameters["cg_max_iter"]
        
        eta_TR = self.parameters["TR"]["eta"]
        delta_TR = None
        
        
        self.model.solveFwd(x[STATE], x)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        mhat = self.model.generate_vector(PARAMETER) 
        R_mhat = self.model.generate_vector(PARAMETER)   

        mg = self.model.generate_vector(PARAMETER)
        
        x_star = [None, None, None] + x[3::]
        x_star[STATE]     = self.model.generate_vector(STATE)
        x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
        
        cost_old, reg_old, misfit_old = self.model.cost(x)
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(x[ADJOINT], x)
            
            self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            gradnorm = self.model.evalGradientParameter(x, mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            

            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            HessApply = ReducedHessian(self.model)
            solver = CGSolverSteihaug(comm = self.model.prior.R.mpi_comm())
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Rsolver())
            if self.it > 1:
                solver.set_TR(delta_TR, self.model.prior.R)
            solver.parameters["rel_tolerance"] = tolcg
            self.parameters["max_iter"]        = cg_max_iter
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(mhat, -mg)
            self.total_cg_iter += HessApply.ncalls

            if self.it == 1:
                self.model.prior.R.mult(mhat,R_mhat)
                mhat_Rnorm = R_mhat.inner(mhat)
                delta_TR = max(math.sqrt(mhat_Rnorm),1)

            x_star[PARAMETER].zero()
            x_star[PARAMETER].axpy(1., x[PARAMETER])
            x_star[PARAMETER].axpy(1., mhat)   #m_star = m +mhat
            x_star[STATE].zero()
            x_star[STATE].axpy(1., x[STATE])      #u_star = u
            self.model.solveFwd(x_star[STATE], x_star)
            cost_star, reg_star, misfit_star = self.model.cost(x_star)
            ACTUAL_RED = cost_old - cost_star
            #Calculate Predicted Reduction
            H_mhat = self.model.generate_vector(PARAMETER)
            H_mhat.zero()
            HessApply.mult(mhat,H_mhat)
            mg_mhat = mg.inner(mhat)
            PRED_RED = -0.5*mhat.inner(H_mhat) - mg_mhat
            # print( "PREDICTED REDUCTION", PRED_RED, "ACTUAL REDUCTION", ACTUAL_RED)
            rho_TR = ACTUAL_RED/PRED_RED


            # Nocedal and Wright Trust Region conditions (page 69)
            if rho_TR < 0.25:
                delta_TR *= 0.5
            elif rho_TR > 0.75 and solver.reasonid == 3:
                delta_TR *= 2.0
            

            # print( "rho_TR", rho_TR, "eta_TR", eta_TR, "rho_TR > eta_TR?", rho_TR > eta_TR , "\n")
            if rho_TR > eta_TR:
                x[PARAMETER].zero()
                x[PARAMETER].axpy(1.0,x_star[PARAMETER])
                x[STATE].zero()
                x[STATE].axpy(1.0,x_star[STATE])
                cost_old = cost_star
                reg_old = reg_star
                misfit_old = misfit_star
                accept_step = True
            else:
                accept_step = False
                
            if self.callback:
                self.callback(self.it, x)
                
            if(print_level >= 0) and (self.it == 1):
                print( "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:14} {7:14} {8:14} {9:11} {10:14}".format(
                      "It", "cg_it", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "TR Radius", "rho_TR", "Accept Step","tolcg") )
                
            if print_level >= 0:
                print( "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e} {9:11} {10:14e}".format(
                        self.it, HessApply.ncalls, cost_old, misfit_old, reg_old, mg_mhat, gradnorm, delta_TR, rho_TR, accept_step,tolcg) )
                

            #TR radius can make this term arbitrarily small and prematurely exit.
            if -mg_mhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_old
        return x



