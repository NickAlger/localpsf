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

# import math
# import hippylib as hp
# from hippylib.utils.parameterList import ParameterList
# from hippylib.modeling.reducedHessian import ReducedHessian
# from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
# from cgsolverSteihaugPCH import CGSolverSteihaug
# from op_operations import *
#
# import hlibpro_python_wrapper as hpro
# from localpsf.product_convolution_kernel import ProductConvolutionKernel
# from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel
# import scipy.sparse.linalg as spla
import numpy as np
from scipy.linalg import blas

from typing import TypeVar
from collections.abc import Callable



class NCGConvergenceInfo:
    def __init__(me, extra_info=None):
        me.extra_info = extra_info
        me._cg_iteration = list()
        me._hessian_matvecs = list()
        me._total_cost = list()
        me._misfit_cost = list()
        me._reg_cost = list()
        me._mg_mhat = list()
        me._gradnorm = list()
        me._step_length_alpha = list()
        me._tolcg = list()

        me.labels = ['cg_iteration', 'hessian_matvecs', 'total_cost', 'misfit_cost', 'reg_cost', 'mg_mhat', 'gradnorm',
                     'step_length_alpha', 'tolcg']

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


scalar_type = float
vector_type = np.ndarray

def numpy_axpy(a: scalar_type, x: vector_type, y: vector_type) -> None: # y <- y + a*x
    y += a*x

def numpy_scal(a: scalar_type, x: vector_type) -> None:
    x *= a


def newtoncg_ls(get_optimization_variable:     Callable[[],                                     vector_type],
                set_optimization_variable:     Callable[[vector_type],                          None],
                cost:                          Callable[[],                                     tuple[scalar_type, scalar_type, scalar_type]],
                gradient:                      Callable[[],                                     vector_type],
                apply_hessian:                 Callable[[vector_type],                          vector_type],
                apply_gauss_newton_hessian:    Callable[[vector_type],                          vector_type],
                build_hessian_preconditioner:  Callable[...,                                    None],
                update_hessian_preconditioner: Callable[[list[vector_type], list[vector_type]], None],
                solve_hessian_preconditioner:  Callable[[vector_type],                          vector_type],
                inner_product:       Callable[[vector_type, vector_type], vector_type]       = np.dot,
                copy_vector:         Callable[[vector_type],              vector_type]       = np.copy,
                axpy:                Callable[[scalar_type, vector_type, vector_type], None] = numpy_axpy, # axpy(a, x, y): y <- y + a*x
                scal:                Callable[[scalar_type, vector_type], None]              = numpy_scal, # scal(a, x): x <- a*x
                callback=None,
                rtol=1e-12,
                atol=1e-14,
                maxiter_newton=25,
                maxiter_cg=400,
                display=True,
                num_initial_iter=3, # Number of initial iterations
                num_gn_iter=7, # Number of Gauss-Newton iterations (including, possibly, initial iterations)
                krylov_recycling=True, # Recycle Krylov information after initial iterations
                krylov_recycling_initial_iter=False, # Recycle Krylov information in initial iterations
                cg_coarse_tolerance=0.5,
                c_armijo=1e-4, # Armijo constant for sufficient reduction
                max_backtracking_iter=10, #Maximum number of backtracking iterations
                preconditioner_args=(),
                preconditioner_kwargs=(),
                gdm_tol=1e-18, # we converge when (g,dm) <= gdm_tolerance
                ):

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    it = 0
    converged = False
    total_cg_iter = 0
    num_hessian_calls = 0
    reason = 'unknown reason'

    residualCounts = []
    convergence_info = NCGConvergenceInfo()

    num_hessian_calls += 1

    cost_old, misfit_old, reg_old = cost()

    while (it < maxiter_newton) and (converged == False):
        using_gauss_newton = (it < num_gn_iter)
        if using_gauss_newton:
            apply_hessian = apply_gauss_newton_hessian
        else:
            apply_hessian = apply_hessian
        printmaybe('it=', it, ', num_initial_iter=', num_initial_iter, ', num_gn_iter=', num_gn_iter,
                   ', using_gauss_newton=', using_gauss_newton)

        g = gradient()
        gradnorm = np.sqrt(inner_product(g, g))

        if it == 0:
            gradnorm_ini = gradnorm
            tol = max(atol, gradnorm_ini * rtol)

        if (gradnorm < tol) and (it > 0):
            converged = True
            reason = "Norm of the gradient less than tolerance"
            break

        tolcg = min(cg_coarse_tolerance, np.sqrt(gradnorm / gradnorm_ini))

        if (it == num_initial_iter):
            printmaybe('building preconditioner')
            build_hessian_preconditioner(*preconditioner_args, **preconditioner_kwargs)

        it += 1

        minus_g = copy_vector(g)
        scal(-1.0, minus_g)

        dm, extras = cgsteihaug(apply_hessian, minus_g,
                                solve_M=solve_hessian_preconditioner,
                                inner_product=inner_product,
                                copy_vector=copy_vector,
                                axpy=axpy,
                                scal=scal,
                                rtol=tolcg,
                                display=True,
                                maxiter=maxiter_cg)

        X = extras['X']
        Y = extras['Y']
        num_hessian_calls = len(X)

        recycle = (((it > num_initial_iter and krylov_recycling)
                    or (it <= num_initial_iter and krylov_recycling_initial_iter))
                   and (X is not None) and (Y is not None))

        if recycle:
            update_hessian_preconditioner(X, Y)

        total_cg_iter += num_hessian_calls
        residualCounts.append(num_hessian_calls)

        alpha = 1.0
        descent = 0
        n_backtrack = 0

        gdm = inner_product(g, dm)

        m = get_optimization_variable()

        mstar = copy_vector(m)
        while descent == 0 and n_backtrack < max_backtracking_iter:
            # mstar = m + alpha*dm
            scal(0.0, mstar) # now mstar = 0
            axpy(1.0, m, mstar) # now mstar = m
            axpy(alpha, dm, mstar) # now mstar = m + alpha*dm

            set_optimization_variable(mstar)
            cost_new, reg_new, misfit_new = cost()

            # Check if armijo conditions are satisfied
            if (cost_new < cost_old + alpha * c_armijo * gdm) or (-gdm <= gdm_tol):
                cost_old = cost_new
                descent = 1
                m = mstar
                set_optimization_variable(mstar)
            else:
                n_backtrack += 1
                alpha *= 0.5

        convergence_info.add_iteration(it, num_hessian_calls, cost_new, misfit_new, reg_new, gdm, gradnorm,
                                       alpha, tolcg)

        if display:
            convergence_info.print()

        if callback:
            callback(it, m)

        if n_backtrack == max_backtracking_iter:
            converged = False
            reason = "Maximum number of backtracking reached"
            break

        if -gdm <= gdm_tol:
            converged = True
            reason = "Norm of (g, dm) less than tolerance"
            break

    final_grad_norm = gradnorm
    final_cost = cost_new
    printmaybe('final_grad_norm=', final_grad_norm, ', final_cost=', final_cost,
               ', converged=', converged, ', total_cg_iter=', total_cg_iter,  'reason=', reason)
    return convergence_info


def cgsteihaug(apply_A: Callable[[vector_type], vector_type],
               b:       vector_type,
               solve_M:             Callable[[vector_type],                           vector_type] = lambda x: x,
               inner_product:       Callable[[vector_type, vector_type],              vector_type] = np.dot,
               axpy:                Callable[[scalar_type, vector_type, vector_type], None]        = numpy_axpy,
               scal:                Callable[[scalar_type, vector_type],              None]        = numpy_scal,
               copy_vector:         Callable[[vector_type],                           vector_type] = np.copy,
               x0=None,
               display=True,
               maxiter=1000,
               rtol=1e-9,
               atol=1e-12):

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    X = []
    Y = []
    def apply_A_wrapper(x: vector_type) -> vector_type:
        y = apply_A(x)
        X.append(x)
        Y.append(y)
        return y

    iter = 0

    r = copy_vector(b)
    if x0 is None:
        x = scal(0.0, copy_vector(b)) # x = 0, r = b
    else:
        x = copy_vector(x0)
        axpy(-1.0, apply_A_wrapper(x), r) # r = b - A*x0

    z = solve_M(r)
    print("||r0|| = {0:1.3e}".format(np.sqrt(inner_product(r, r))))
    print("||B^-1 r0|| = {0:1.3e}".format(np.sqrt(inner_product(z, z))))

    d = copy_vector(z)

    nom0 = inner_product(d, r)
    nom = nom0

    printmaybe(" Iteration : ", 0, " (B r, r) = ", nom)

    rtol2 = nom * rtol * rtol
    atol2 = atol * atol
    r0 = max(rtol2, atol2)

    if nom <= r0:
        converged = True
        reason = "Relative/Absolute residual less than tol"
        final_norm = np.sqrt(nom)
        printmaybe(reason)
        printmaybe("Converged in ", iter, " iterations with final norm ", final_norm)
        extras = {'X': X, 'Y': Y, 'converged': converged, 'reason': reason, 'final_norm': final_norm}
        return x, extras

    Ad = apply_A_wrapper(d)
    den = inner_product(Ad, d)

    if den <= 0.0:
        converged = True
        reason = "Reached a negative direction"
        axpy(1.0, d, x)   # x <- x + d
        axpy(-1.0, Ad, r) # r <- r - Ad
        z = solve_M(r)     # z <- inv(M)*r
        nom = inner_product(r, z)
        final_norm = np.sqrt(nom)
        printmaybe(reason)
        printmaybe("Converged in ", iter, " iterations with final norm ", final_norm)
        extras = {'X': X, 'Y': Y, 'converged': converged, 'reason': reason, 'final_norm': final_norm}
        return x, extras

    # start iteration
    iter = 1
    while True:
        alpha = nom / den
        axpy(alpha, d, x) # x = x + alpha * d
        axpy(-alpha, Ad, r) # r = r - alpha * Ad
        z = solve_M(r)
        betanom = inner_product(r, z)
        printmaybe(" Iteration : ", iter, " (B r, r) = ", betanom)

        if betanom < r0:
            converged = True
            reason = "Relative/Absolute residual less than tol"
            final_norm = np.sqrt(betanom)
            printmaybe(reason)
            printmaybe("Converged in ", iter, " iterations with final norm ", final_norm)
            break

        iter += 1
        if iter > maxiter:
            converged = False
            reason = "Maximum Number of Iterations Reached"
            final_norm = np.sqrt(betanom)
            printmaybe(reason)
            printmaybe("Not Converged. Final residual norm ", final_norm)
            break

        beta = betanom / nom

        # d = z + beta d
        scal(beta, d)
        axpy(1.0, z, d)

        Ad = apply_A_wrapper(d)

        den = inner_product(Ad, d)

        if den <= 0.0:
            converged = True
            reason = "Reached a negative direction"
            final_norm = np.sqrt(nom)
            printmaybe(reason)
            printmaybe("Converged in ", iter, " iterations with final norm ", final_norm)
            break

        nom = betanom

    extras = {'X': X, 'Y': Y, 'converged': converged, 'reason': reason, 'final_norm': final_norm}
    return x, extras

#
# def CGSolverSteihaug_ParameterList():
#     """
#     Generate a :code:`ParameterList` for :code:`CGSolverSteihaug`.
#     Type :code:`CGSolverSteihaug_ParameterList().showMe()` for default values and their descriptions
#     """
#     parameters = {}
#     parameters["rel_tolerance"] = [1e-9, "the relative tolerance for the stopping criterion"]
#     parameters["abs_tolerance"] = [1e-12, "the absolute tolerance for the stopping criterion"]
#     parameters["max_iter"] = [1000, "the maximum number of iterations"]
#     parameters["zero_initial_guess"] = [True,
#                                         "if True we start with a 0 initial guess; if False we use the x as initial guess."]
#     parameters["print_level"] = [0,
#                                  "verbosity level: -1 --> no output on screen; 0 --> only final residual at convergence or reason for not not convergence"]
#     return ParameterList(parameters)
#
#
# class CGSolverSteihaug:
#     """
#     Solve the linear system :math:`A x = b` using preconditioned conjugate gradient ( :math:`B` preconditioner)
#     and the Steihaug stopping criterion:
#
#     - reason of termination 0: we reached the maximum number of iterations (no convergence)
#     - reason of termination 1: we reduced the residual up to the given tolerance (convergence)
#     - reason of termination 2: we reached a negative direction (premature termination due to not spd matrix)
#     - reason of termination 3: we reached the boundary of the trust region
#
#     The stopping criterion is based on either
#
#     - the absolute preconditioned residual norm check: :math:`|| r^* ||_{B^{-1}} < atol`
#     - the relative preconditioned residual norm check: :math:`|| r^* ||_{B^{-1}}/|| r^0 ||_{B^{-1}} < rtol,`
#
#     where :math:`r^* = b - Ax^*` is the residual at convergence and :math:`r^0 = b - Ax^0` is the initial residual.
#
#     The operator :code:`A` is set using the method :code:`set_operator(A)`.
#     :code:`A` must provide the following two methods:
#
#     - :code:`A.mult(x,y)`: `y = Ax`
#     - :code:`A.init_vector(x, dim)`: initialize the vector `x` so that it is compatible with the range `(dim = 0)` or
#       the domain `(dim = 1)` of :code:`A`.
#
#     The preconditioner :code:`B` is set using the method :code:`set_preconditioner(B)`.
#     :code:`B` must provide the following method:
#     - :code:`B.solve(z,r)`: `z` is the action of the preconditioner :code:`B` on the vector `r`
#
#     To solve the linear system :math:`Ax = b` call :code:`self.solve(x,b)`.
#     Here :code:`x` and :code:`b` are assumed to be :code:`dolfin.Vector` objects.
#
#     Type :code:`CGSolverSteihaug_ParameterList().showMe()` for default parameters and their descriptions
#     """
#
#     reason = ["Maximum Number of Iterations Reached",
#               "Relative/Absolute residual less than tol",
#               "Reached a negative direction",
#               "Reached trust region boundary"
#               ]
#
#     def __init__(self, parameters=CGSolverSteihaug_ParameterList(), comm=dl.MPI.comm_world):
#
#         self.parameters = parameters
#
#         self.A = None
#         self.B_solver = None
#         self.B_op = None
#         self.converged = False
#         self.iter = 0
#         self.reasonid = 0
#         self.final_norm = 0
#
#         self.TR_radius_2 = None
#
#         self.update_x = self.update_x_without_TR
#
#         self.r = dl.Vector(comm)
#         self.z = dl.Vector(comm)
#         self.Ad = dl.Vector(comm)
#         self.d = dl.Vector(comm)
#         self.Bx = dl.Vector(comm)
#         # -------- Y = A X store applies of the linear system operator A.
#         self.X = None
#         self.Y = None
#
#     def set_operator(self, A):
#         """
#         Set the operator :math:`A`.
#         """
#         self.A = A
#         self.A.init_vector(self.r, 0)
#         self.A.init_vector(self.z, 0)
#         self.A.init_vector(self.d, 0)
#         self.A.init_vector(self.Ad, 0)
#
#     def set_preconditioner(self, B_solver):
#         """
#         Set the preconditioner :math:`B`.
#         """
#         self.B_solver = B_solver
#
#     def set_TR(self, radius, B_op):
#         assert self.parameters["zero_initial_guess"]
#         self.TR_radius_2 = radius * radius
#         self.update_x = self.update_x_with_TR
#         self.B_op = B_op
#         self.B_op.init_vector(self.Bx, 0)
#
#     def update_x_without_TR(self, x, alpha, d):
#         x.axpy(alpha, d)
#         return False
#
#     def update_x_with_TR(self, x, alpha, d):
#         x_bk = x.copy()
#         x.axpy(alpha, d)
#         self.Bx.zero()
#         self.B_op.mult(x, self.Bx)
#         x_Bnorm2 = self.Bx.inner(x)
#
#         if x_Bnorm2 < self.TR_radius_2:
#             return False
#         else:
#             # Move point to boundary of trust region
#             self.Bx.zero()
#             self.B_op.mult(x_bk, self.Bx)
#             x_Bnorm2 = self.Bx.inner(x_bk)
#             Bd = self.d.copy()
#             Bd.zero()
#             self.B_op.mult(self.d, Bd)
#             d_Bnorm2 = Bd.inner(d)
#             d_Bx = Bd.inner(x_bk)
#             a_tau = alpha * alpha * d_Bnorm2
#             b_tau_half = alpha * d_Bx
#             c_tau = x_Bnorm2 - self.TR_radius_2
#             # Solve quadratic for :code:`tau`
#             tau = (-b_tau_half + math.sqrt(b_tau_half * b_tau_half - a_tau * c_tau)) / a_tau
#             x.zero()
#             x.axpy(1, x_bk)
#             x.axpy(tau * alpha, d)
#
#             return True
#
#     def solve(self, x, b):
#         """
#         Solve the linear system :math:`Ax = b`
#         """
#         self.iter = 0
#         self.converged = False
#         self.reasonid = 0
#
#         betanom = 0.0
#         alpha = 0.0
#         beta = 0.0
#         self.X = []
#         self.Y = []
#         if self.parameters["zero_initial_guess"]:
#             self.r.zero()
#             self.r.axpy(1.0, b)
#             x.zero()
#         else:
#             assert self.TR_radius_2 == None
#             self.A.mult(x, self.r)
#             self.r *= -1.0
#             self.r.axpy(1.0, b)
#
#         self.z.zero()
#         self.B_solver.solve(self.z, self.r)  # z = B^-1 r
#         print("||r0|| = {0:1.3e}".format(math.sqrt(self.r.inner(self.r))))
#         print("||B^-1 r0|| = {0:1.3e}".format(math.sqrt(self.z.inner(self.z))))
#
#         self.d.zero()
#         self.d.axpy(1., self.z);  # d = z
#
#         nom0 = self.d.inner(self.r)
#         nom = nom0
#
#         if self.parameters["print_level"] == 1:
#             print(" Iterartion : ", 0, " (B r, r) = ", nom)
#
#         rtol2 = nom * self.parameters["rel_tolerance"] * self.parameters["rel_tolerance"]
#         atol2 = self.parameters["abs_tolerance"] * self.parameters["abs_tolerance"]
#         r0 = max(rtol2, atol2)
#
#         if nom <= r0:
#             self.converged = True
#             self.reasonid = 1
#             self.final_norm = math.sqrt(nom)
#             if (self.parameters["print_level"] >= 0):
#                 print(self.reason[self.reasonid])
#                 print("Converged in ", self.iter, " iterations with final norm ", self.final_norm)
#             return
#
#         self.A.mult(self.d, self.Ad)
#         den = self.Ad.inner(self.d)
#         # ------- store apply
#         self.X.append(self.d[:])
#         self.Y.append(self.Ad[:])
#
#         if den <= 0.0:
#             self.converged = True
#             self.reasonid = 2
#             x.axpy(1., self.d)
#             self.r.axpy(-1., self.Ad)
#             self.B_solver.solve(self.z, self.r)
#             nom = self.r.inner(self.z)
#             self.final_norm = math.sqrt(nom)
#             if (self.parameters["print_level"] >= 0):
#                 print(self.reason[self.reasonid])
#                 print("Converged in ", self.iter, " iterations with final norm ", self.final_norm)
#             return
#
#         # start iteration
#         self.iter = 1
#         while True:
#             alpha = nom / den
#             TrustBool = self.update_x(x, alpha, self.d)  # x = x + alpha d
#             if TrustBool == True:
#                 self.converged = True
#                 self.reasonid = 3
#                 self.final_norm = math.sqrt(betanom)
#                 if (self.parameters["print_level"] >= 0):
#                     print(self.reason[self.reasonid])
#                     print("Converged in ", self.iter, " iterations with final norm ", self.final_norm)
#                 break
#
#             self.r.axpy(-alpha, self.Ad)  # r = r - alpha A d
#
#             self.B_solver.solve(self.z, self.r)  # z = B^-1 r
#             betanom = self.r.inner(self.z)
#
#             if self.parameters["print_level"] == 1:
#                 print(" Iteration : ", self.iter, " (B r, r) = ", betanom)
#
#             if betanom < r0:
#                 self.converged = True
#                 self.reasonid = 1
#                 self.final_norm = math.sqrt(betanom)
#                 if (self.parameters["print_level"] >= 0):
#                     print(self.reason[self.reasonid])
#                     print("Converged in ", self.iter, " iterations with final norm ", self.final_norm)
#                 break
#
#             self.iter += 1
#             if self.iter > self.parameters["max_iter"]:
#                 self.converged = False
#                 self.reasonid = 0
#                 self.final_norm = math.sqrt(betanom)
#                 if (self.parameters["print_level"] >= 0):
#                     print(self.reason[self.reasonid])
#                     print("Not Converged. Final residual norm ", self.final_norm)
#                 break
#
#             beta = betanom / nom
#             self.d *= beta
#             self.d.axpy(1., self.z)  # d = z + beta d
#
#             self.A.mult(self.d, self.Ad)
#             den = self.d.inner(self.Ad)
#             # ------- store apply
#             self.X.append(self.d[:])
#             self.Y.append(self.Ad[:])
#
#             if den <= 0.0:
#                 self.converged = True
#                 self.reasonid = 2
#                 self.final_norm = math.sqrt(nom)
#                 if (self.parameters["print_level"] >= 0):
#                     print(self.reason[self.reasonid])
#                     print("Converged in ", self.iter, " iterations with final norm ", self.final_norm)
#                 break
#
#             nom = betanom
#
#
# class ReducedSpaceNewtonCG:
#     """
#     Inexact Newton-CG method to solve constrained optimization problems in the reduced parameter space.
#     The Newton system is solved inexactly by early termination of CG iterations via Eisenstat-Walker
#     (to prevent oversolving) and Steihaug (to avoid negative curvature) criteria.
#     Globalization is performed using one of the following methods:
#
#     - line search (LS) based on the armijo sufficient reduction condition; or
#     - trust region (TR) based on the prior preconditioned norm of the update direction.
#
#     The stopping criterion is based on a control on the norm of the gradient and a control of the
#     inner product between the gradient and the Newton direction.
#
#     The user must provide a model that describes the forward problem, cost functionals, and all the
#     derivatives for the gradient and the Hessian.
#
#     More specifically the model object should implement following methods:
#
#        - :code:`generate_vector()` -> generate the object containing state, parameter, adjoint
#        - :code:`cost(x)` -> evaluate the cost functional, report regularization part and misfit separately
#        - :code:`solveFwd(out, x)` -> solve the possibly non linear forward problem
#        - :code:`solveAdj(out, x)` -> solve the linear adjoint problem
#        - :code:`evalGradientParameter(x, out)` -> evaluate the gradient of the parameter and compute its norm
#        - :code:`setPointForHessianEvaluations(x)` -> set the state to perform hessian evaluations
#        - :code:`solveFwdIncremental(out, rhs)` -> solve the linearized forward problem for a given :code:`rhs`
#        - :code:`solveAdjIncremental(out, rhs)` -> solve the linear adjoint problem for a given :code:`rhs`
#        - :code:`applyC(dm, out)`    --> Compute out :math:`= C_x dm`
#        - :code:`applyCt(dp, out)`   --> Compute out = :math:`C_x  dp`
#        - :code:`applyWuu(du,out)`   --> Compute out = :math:`(W_{uu})_x  du`
#        - :code:`applyWmu(dm, out)`  --> Compute out = :math:`(W_{um})_x  dm`
#        - :code:`applyWmu(du, out)`  --> Compute out = :math:`W_{mu}  du`
#        - :code:`applyR(dm, out)`    --> Compute out = :math:`R  dm`
#        - :code:`applyWmm(dm,out)`   --> Compute out = :math:`W_{mm} dm`
#        - :code:`Rsolver()`          --> A solver for the regularization term
#
#     Type :code:`help(Model)` for additional information
#     """
#     termination_reasons = [
#         "Maximum number of Iteration reached",  # 0
#         "Norm of the gradient less than tolerance",  # 1
#         "Maximum number of backtracking reached",  # 2
#         "Norm of (g, dm) less than tolerance"  # 3
#     ]
#
#     def __init__(self, model, parameters=ReducedSpaceNewtonCG_ParameterList(), callback=None):
#         """
#         Initialize the ReducedSpaceNewtonCG.
#         Type :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for list of default parameters
#         and their descriptions.
#
#         Parameters:
#         :code:`model` The model object that describes the inverse problem
#         :code:`parameters`: (type :code:`ParameterList`, optional) set parameters for inexact Newton CG
#         :code:`callback`: (type function handler with signature :code:`callback(it: int, x: list of dl.Vector): --> None`
#                optional callback function to be called at the end of each iteration. Takes as input the iteration number, and
#                the list of vectors for the state, parameter, adjoint.
#         """
#         self.model = model
#         self.parameters = parameters
#
#         self.it = 0
#         self.converged = False
#         self.total_cg_iter = 0
#         self.ncalls = 0
#         self.reason = 0
#         self.final_grad_norm = 0
#
#         self.callback = callback
#
#         self.IP = self.parameters["IP"]
#         self.PCHinitialized = None  # for which Newton iteration was the PCH approximation first constructed
#         self.P = self.parameters["projector"]
#         self.residualCounts = []
#
#     def solve(self, x):
#         """
#
#         Input:
#             :code:`x = [u, m, p]` represents the initial guess (u and p may be None).
#             :code:`x` will be overwritten on return.
#         """
#         if self.model is None:
#             raise TypeError("model can not be of type None.")
#
#         if x[STATE] is None:
#             x[STATE] = self.model.generate_vector(STATE)
#
#         if x[ADJOINT] is None:
#             x[ADJOINT] = self.model.generate_vector(ADJOINT)
#
#         if self.parameters["globalization"] == "LS":
#             return self._solve_ls(x)
#         elif self.parameters["globalization"] == "TR":
#             return self._solve_tr(x)
#         else:
#             raise ValueError(self.parameters["globalization"])
#
#     def _solve_ls(self, x):
#         """
#         Solve the constrained optimization problem with initial guess :code:`x`.
#         """
#         rel_tol = self.parameters["rel_tolerance"]
#         abs_tol = self.parameters["abs_tolerance"]
#         max_iter = self.parameters["max_iter"]
#         print_level = self.parameters["print_level"]
#         initial_iter = self.parameters["initial_iter"]
#         GN_iter = self.parameters["GN_iter"]
#         krylov_recycling = self.parameters["krylov_recycling"]
#         cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
#         cg_max_iter = self.parameters["cg_max_iter"]
#         PCH_precond = self.parameters["PCH_precond"]
#
#         c_armijo = self.parameters["LS"]["c_armijo"]
#         max_backtracking_iter = self.parameters["LS"]["max_backtracking_iter"]
#
#         convergence_info = NCGConvergenceInfo()
#
#         self.model.solveFwd(x[STATE], x)
#
#         self.it = 0
#         self.converged = False
#         self.ncalls += 1
#
#         mhat = self.model.generate_vector(PARAMETER)
#         mg = self.model.generate_vector(PARAMETER)
#
#         x_star = [None, None, None] + x[3::]
#         x_star[STATE] = self.model.generate_vector(STATE)
#         x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
#
#         cost_old, _, _ = self.model.cost(x)
#
#         while (self.it < max_iter) and (self.converged == False):
#             self.model.solveAdj(x[ADJOINT], x)
#
#             using_gauss_newton = (self.it < (initial_iter + GN_iter))
#             print('initial_iter=', initial_iter, ', GN_iter=', GN_iter, ', self.it=', self.it, ', using_gauss_newton=',
#                   using_gauss_newton)
#
#             self.model.setPointForHessianEvaluations(x, gauss_newton_approx=using_gauss_newton)
#             gradnorm = self.model.evalGradientParameter(x, mg)
#
#             if self.it == 0:
#                 gradnorm_ini = gradnorm
#                 tol = max(abs_tol, gradnorm_ini * rel_tol)
#
#             # check if solution is reached
#             if (gradnorm < tol) and (self.it > 0):
#                 self.converged = True
#                 self.reason = 1
#                 break
#
#             HessApply = ReducedHessian(self.model)
#             tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm / gradnorm_ini))
#
#             if (self.it == initial_iter) and PCH_precond:
#                 print('building PCH preconditioner')
#                 self.PCHinitialized = self.it
#                 # build PCH approximation
#                 # ----- NEED a mechanism to set the parameter
#                 # ----- also just pass the entire inverse problem object
#                 self.IP.set_parameter(x)
#                 num_neighbors = self.parameters["num_neighbors"]
#                 num_batches = self.parameters["num_batches"]
#                 tau = 3.0
#                 hmatrix_tol = 1.e-5
#                 PCK = ProductConvolutionKernel(self.IP.V, self.IP.V, self.IP.apply_Hd_petsc, self.IP.apply_Hd_petsc,
#                                                num_batches, num_batches,
#                                                tau_rows=tau, tau_cols=tau,
#                                                num_neighbors_rows=num_neighbors,
#                                                num_neighbors_cols=num_neighbors)
#                 Hd_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)
#
#                 # Rebuild reg hmatrix with same block cluster tree as PCH data misfit hmatrix
#                 print('Building Regularization H-Matrix')
#                 R_scipy = self.parameters["Rscipy"]
#                 R_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(R_scipy, Hd_pch_nonsym.bct)
#
#                 # ----- build spd approximation of Hd, with cutoff given by a multiple of the minimum eigenvalue of the regularization operator
#                 Hd_pch = Hd_pch_nonsym.spd()
#                 H_pch = Hd_pch + R_hmatrix
#
#                 preconditioner_hmatrix = H_pch.inv()
#
#             if (self.it > initial_iter) and PCH_precond and krylov_recycling:
#                 X = solver.X
#                 Y = solver.Y
#                 if self.P is not None:
#                     xtest = dl.Vector()
#                     ytest = dl.Vector()
#                     xtest.init(self.IP.Vh[PARAMETER].dim())
#                     ytest.init(self.IP.Vh[PARAMETER].dim())
#                     xtestP = dl.Vector()
#                     ytestP = dl.Vector()
#                     xtestP.init(self.IP.N)
#                     ytestP.init(self.IP.N)
#                     Xproj = []
#                     Yproj = []
#                     for i in range(len(X)):
#                         xtest[:] = X[i]
#                         ytest[:] = Y[i]
#                         self.IP.prior.P.mult(xtest, xtestP)
#                         self.IP.prior.P.mult(ytest, ytestP)
#                         Xproj.append(xtestP[:])
#                         Yproj.append(ytestP[:])
#                     Xproj = np.array(Xproj).T
#                     Yproj = np.array(Yproj).T
#                     preconditioner_hmatrix = preconditioner_hmatrix.dfp_update(Yproj, Xproj)
#                 else:
#                     X = np.array(X).T
#                     Y = np.array(Y).T
#                     preconditioner_hmatrix = preconditioner_hmatrix.dfp_update(Y, X)
#
#             solver = CGSolverSteihaug(comm=self.model.prior.R.mpi_comm())
#             solver.set_operator(HessApply)
#
#             # if self.parameters["PCH_precond"] and self.total_cg_iter > 0:
#             if self.parameters["PCH_precond"] and (self.it >= initial_iter):
#                 preconditioner_linop = preconditioner_hmatrix.as_linear_operator()
#                 H_pch_precond = solver_from_dot(preconditioner_linop)
#                 if self.P is not None:
#                     H_pch_fullspace_precond = projected_solver(self.P, H_pch_precond)
#                     solver.set_preconditioner(H_pch_fullspace_precond)
#                 else:
#                     solver.set_preconditioner(H_pch_precond)
#             else:
#                 solver.set_preconditioner(self.model.Rsolver())
#             self.it += 1
#             solver.parameters["rel_tolerance"] = tolcg
#             solver.parameters["max_iter"] = cg_max_iter
#             solver.parameters["zero_initial_guess"] = True
#             solver.parameters["print_level"] = print_level - 1
#
#             solver.solve(mhat, -mg)
#             self.total_cg_iter += HessApply.ncalls
#             self.residualCounts.append(HessApply.ncalls)
#
#             alpha = 1.0
#             descent = 0
#             n_backtrack = 0
#
#             mg_mhat = mg.inner(mhat)
#
#             while descent == 0 and n_backtrack < max_backtracking_iter:
#                 # update m and u
#                 x_star[PARAMETER].zero()
#                 x_star[PARAMETER].axpy(1., x[PARAMETER])
#                 x_star[PARAMETER].axpy(alpha, mhat)
#                 x_star[STATE].zero()
#                 x_star[STATE].axpy(1., x[STATE])
#                 self.model.solveFwd(x_star[STATE], x_star)
#
#                 cost_new, reg_new, misfit_new = self.model.cost(x_star)
#
#                 # Check if armijo conditions are satisfied
#                 if (cost_new < cost_old + alpha * c_armijo * mg_mhat) or (-mg_mhat <= self.parameters["gdm_tolerance"]):
#                     cost_old = cost_new
#                     descent = 1
#                     x[PARAMETER].zero()
#                     x[PARAMETER].axpy(1., x_star[PARAMETER])
#                     x[STATE].zero()
#                     x[STATE].axpy(1., x_star[STATE])
#                 else:
#                     n_backtrack += 1
#                     alpha *= 0.5
#
#             convergence_info.add_iteration(self.it, HessApply.ncalls, cost_new, misfit_new, reg_new, mg_mhat, gradnorm,
#                                            alpha, tolcg)
#
#             if print_level >= 0:
#                 convergence_info.print()
#
#             if self.callback:
#                 self.callback(self.it, x)
#
#             if n_backtrack == max_backtracking_iter:
#                 self.converged = False
#                 self.reason = 2
#                 break
#
#             if -mg_mhat <= self.parameters["gdm_tolerance"]:
#                 self.converged = True
#                 self.reason = 3
#                 break
#
#         self.final_grad_norm = gradnorm
#         self.final_cost = cost_new
#         return x
#
#     def _solve_tr(self, x):
#         rel_tol = self.parameters["rel_tolerance"]
#         abs_tol = self.parameters["abs_tolerance"]
#         max_iter = self.parameters["max_iter"]
#         print_level = self.parameters["print_level"]
#         GN_iter = self.parameters["GN_iter"]
#         cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
#         cg_max_iter = self.parameters["cg_max_iter"]
#
#         eta_TR = self.parameters["TR"]["eta"]
#         delta_TR = None
#
#         self.model.solveFwd(x[STATE], x)
#
#         self.it = 0
#         self.converged = False
#         self.ncalls += 1
#
#         mhat = self.model.generate_vector(PARAMETER)
#         R_mhat = self.model.generate_vector(PARAMETER)
#
#         mg = self.model.generate_vector(PARAMETER)
#
#         x_star = [None, None, None] + x[3::]
#         x_star[STATE] = self.model.generate_vector(STATE)
#         x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
#
#         cost_old, reg_old, misfit_old = self.model.cost(x)
#         while (self.it < max_iter) and (self.converged == False):
#             self.model.solveAdj(x[ADJOINT], x)
#
#             self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter))
#             gradnorm = self.model.evalGradientParameter(x, mg)
#
#             if self.it == 0:
#                 gradnorm_ini = gradnorm
#                 tol = max(abs_tol, gradnorm_ini * rel_tol)
#
#             # check if solution is reached
#             if (gradnorm < tol) and (self.it > 0):
#                 self.converged = True
#                 self.reason = 1
#                 break
#
#             self.it += 1
#
#             tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm / gradnorm_ini))
#
#             HessApply = ReducedHessian(self.model)
#             solver = CGSolverSteihaug(comm=self.model.prior.R.mpi_comm())
#             solver.set_operator(HessApply)
#             solver.set_preconditioner(self.model.Rsolver())
#             if self.it > 1:
#                 solver.set_TR(delta_TR, self.model.prior.R)
#             solver.parameters["rel_tolerance"] = tolcg
#             self.parameters["max_iter"] = cg_max_iter
#             solver.parameters["zero_initial_guess"] = True
#             solver.parameters["print_level"] = print_level - 1
#
#             solver.solve(mhat, -mg)
#             self.total_cg_iter += HessApply.ncalls
#
#             if self.it == 1:
#                 self.model.prior.R.mult(mhat, R_mhat)
#                 mhat_Rnorm = R_mhat.inner(mhat)
#                 delta_TR = max(math.sqrt(mhat_Rnorm), 1)
#
#             x_star[PARAMETER].zero()
#             x_star[PARAMETER].axpy(1., x[PARAMETER])
#             x_star[PARAMETER].axpy(1., mhat)  # m_star = m +mhat
#             x_star[STATE].zero()
#             x_star[STATE].axpy(1., x[STATE])  # u_star = u
#             self.model.solveFwd(x_star[STATE], x_star)
#             cost_star, reg_star, misfit_star = self.model.cost(x_star)
#             ACTUAL_RED = cost_old - cost_star
#             # Calculate Predicted Reduction
#             H_mhat = self.model.generate_vector(PARAMETER)
#             H_mhat.zero()
#             HessApply.mult(mhat, H_mhat)
#             mg_mhat = mg.inner(mhat)
#             PRED_RED = -0.5 * mhat.inner(H_mhat) - mg_mhat
#             # print( "PREDICTED REDUCTION", PRED_RED, "ACTUAL REDUCTION", ACTUAL_RED)
#             rho_TR = ACTUAL_RED / PRED_RED
#
#             # Nocedal and Wright Trust Region conditions (page 69)
#             if rho_TR < 0.25:
#                 delta_TR *= 0.5
#             elif rho_TR > 0.75 and solver.reasonid == 3:
#                 delta_TR *= 2.0
#
#             # print( "rho_TR", rho_TR, "eta_TR", eta_TR, "rho_TR > eta_TR?", rho_TR > eta_TR , "\n")
#             if rho_TR > eta_TR:
#                 x[PARAMETER].zero()
#                 x[PARAMETER].axpy(1.0, x_star[PARAMETER])
#                 x[STATE].zero()
#                 x[STATE].axpy(1.0, x_star[STATE])
#                 cost_old = cost_star
#                 reg_old = reg_star
#                 misfit_old = misfit_star
#                 accept_step = True
#             else:
#                 accept_step = False
#
#             if self.callback:
#                 self.callback(self.it, x)
#
#             if (print_level >= 0) and (self.it == 1):
#                 print("\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:14} {7:14} {8:14} {9:11} {10:14}".format(
#                     "It", "cg_it", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "TR Radius", "rho_TR", "Accept Step",
#                     "tolcg"))
#
#             if print_level >= 0:
#                 print("{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e} {9:11} {10:14e}".format(
#                     self.it, HessApply.ncalls, cost_old, misfit_old, reg_old, mg_mhat, gradnorm, delta_TR, rho_TR,
#                     accept_step, tolcg))
#
#             # TR radius can make this term arbitrarily small and prematurely exit.
#             if -mg_mhat <= self.parameters["gdm_tolerance"]:
#                 self.converged = True
#                 self.reason = 3
#                 break
#
#         self.final_grad_norm = gradnorm
#         self.final_cost = cost_old
#         return x
#
#
#
