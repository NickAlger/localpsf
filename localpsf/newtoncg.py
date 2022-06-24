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
            print('using Gauss-Newton Hessian')
            apply_H = apply_gauss_newton_hessian
        else:
            print('using Gauss-Newton Hessian')
            apply_H = apply_hessian
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
            build_hessian_preconditioner()

        it += 1

        minus_g = copy_vector(g)
        scal(-1.0, minus_g)

        dm, extras = cgsteihaug(apply_H, minus_g,
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
               atol=1e-12,
               preconditioned_norm_stopping_criterion=False):

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    X = [] # list [x1, x2, ...]
    Y = []
    def apply_A_wrapper(x: vector_type) -> vector_type:
        y = apply_A(x)
        X.append(x) #OK
        Y.append(y)
        return y

    iter = 0

    r = copy_vector(b)
    if x0 is None:
        x = copy_vector(b)
        scal(0.0, x) # x = 0, r = b
    else:
        x = copy_vector(x0)
        axpy(-1.0, apply_A_wrapper(x), r) # r = b - A*x0

    z = solve_M(r)
    d = copy_vector(z)

    rT_r = inner_product(r, r)
    rT_MM_r = inner_product(z, z)
    rT_M_r = inner_product(d, r)
    print('Residual: r=b-Ax')
    print('Preconditioner: M =approx= A^-1')
    print("||r0|| = {0:1.3e}".format(np.sqrt(rT_r)))
    print("||M r0|| = {0:1.3e}".format(np.sqrt(rT_MM_r)))
    print("(M r0, r0) = {0:1.3e}".format(np.sqrt(rT_M_r)))

    if preconditioned_norm_stopping_criterion:
        rnorm_squared = rT_M_r
        printmaybe(" Iteration : ", 0, " (M r, r) = ", rT_M_r)
    else:
        rnorm_squared = rT_r
        printmaybe(" Iteration : ", 0, " (r, r) = ", rT_r)

    cgtol_squared = max(rnorm_squared * rtol * rtol, atol * atol)

    if rnorm_squared <= cgtol_squared:
        converged = True
        reason = "Relative/Absolute residual less than tol"
        final_norm = np.sqrt(rnorm_squared)
        printmaybe(reason)
        printmaybe("Converged in ", iter, " iterations with final norm ", final_norm)
        extras = {'X': X, 'Y': Y, 'converged': converged, 'reason': reason, 'final_norm': final_norm}
        return x, extras

    A_d = apply_A_wrapper(d)
    dT_A_d = inner_product(A_d, d)

    if dT_A_d <= 0.0:
        converged = True
        reason = "Reached a negative direction"
        axpy(1.0, d, x)   # x <- x + d
        axpy(-1.0, A_d, r) # r <- r - Ad
        if preconditioned_norm_stopping_criterion:
            z = solve_M(r)  # z <- inv(M)*r
            rnorm_squared = inner_product(r, z)
        else:
            rnorm_squared = inner_product(r, r)
        final_norm = np.sqrt(rnorm_squared)
        printmaybe(reason)
        printmaybe("Converged in ", iter, " iterations with final norm ", final_norm)
        extras = {'X': X, 'Y': Y, 'converged': converged, 'reason': reason, 'final_norm': final_norm}
        return x, extras

    # start iteration
    iter = 1
    while True:
        alpha = rT_M_r / dT_A_d
        axpy(alpha, d, x) # x = x + alpha * d
        axpy(-alpha, A_d, r) # r = r - alpha * Ad
        rT_r = inner_product(r, r)
        z = solve_M(r)
        previous_rT_M_r = rT_M_r
        rT_M_r = inner_product(r, z)
        if preconditioned_norm_stopping_criterion:
            rnorm_squared = rT_M_r
            printmaybe(" Iteration : ", iter, " (M r, r) = ", rT_M_r)
        else:
            rnorm_squared = rT_r
            printmaybe(" Iteration : ", iter, " (r, r) = ", rT_r)

        if rnorm_squared < cgtol_squared:
            converged = True
            reason = "Relative/Absolute residual less than tol"
            final_norm = np.sqrt(rnorm_squared)
            printmaybe(reason)
            printmaybe("Converged in ", iter, " iterations with final norm ", final_norm)
            break

        iter += 1
        if iter > maxiter:
            converged = False
            reason = "Maximum Number of Iterations Reached"
            final_norm = np.sqrt(rnorm_squared)
            printmaybe(reason)
            printmaybe("Not Converged. Final residual norm ", final_norm)
            break

        beta = rT_M_r / previous_rT_M_r

        # d = z + beta d
        scal(beta, d)
        axpy(1.0, z, d)

        A_d = apply_A_wrapper(d)

        dT_A_d = inner_product(A_d, d)

        if dT_A_d <= 0.0:
            converged = True
            reason = "Reached a negative direction"
            if preconditioned_norm_stopping_criterion:
                final_norm = np.sqrt(rT_M_r)
            else:
                final_norm = np.sqrt(rT_r)
            printmaybe(reason)
            printmaybe("Converged in ", iter, " iterations with final norm ", final_norm)
            break


    X_arr = np.array(X).T
    Y_arr = np.array(Y).T

    YTX = np.dot(Y_arr.T, X_arr)
    ee, _ = np.linalg.eigh(YTX)
    print('CG Steihaug: YTX eigs=', ee)

    extras = {'X': X, 'Y': Y, 'converged': converged, 'reason': reason, 'final_norm': final_norm}
    return x, extras
