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


scalar_type = float
vector_type = np.ndarray

def numpy_axpy(a: scalar_type, x: vector_type, y: vector_type) -> None: # y <- y + a*x
    y += a*x

def numpy_scal(a: scalar_type, x: vector_type) -> None:
    x *= a

class NCGInfo:
    def __init__(me):
        me.newton_iter = list() # [1,2,3,...]
        me.cg_iter = list() # number of CG iterations in each Newton iteration
        me.cost_calls = list() # number of times cost() is called in each Newton iteration
        me.grad_calls = list() # number of times gradient() is called in each Newton iteration
        me.hess_matvecs = list() # number of Hessian-vector products in the CG solve for each Newton iteration
        me.cost = list() # cost                at the beginning of each Newton iteration
        me.misfit = list()
        me.reg_cost = list()
        me.gdm = list()
        me.gradnorm = list()
        me.step_length = list()
        me.backtrack = list()
        me.cgtol = list()
        me.build_precond = list()
        me.gauss_newton = list()
        me.converged = False
        me.reason = 'unknown reason'

    def start_new_iteration(me, newton_iter):
        me.newton_iter.append(newton_iter)
        me.cg_iter.append(0)
        me.cost_calls.append(0)
        me.grad_calls.append(0)
        me.hess_matvecs.append(0)
        me.cost.append(None)
        me.misfit.append(None)
        me.reg_cost.append(None)
        me.gdm.append(None)
        me.gradnorm.append(None)
        me.step_length.append(None)
        me.backtrack.append(None)
        me.cgtol.append(None)
        me.build_precond.append(False)
        me.gauss_newton.append(True)

    @property
    def cumulative_cg_iterations(me):
        ncg = 0
        for nk in me.cg_iter:
            if nk is not None:
                ncg += nk
        return ncg

    @property
    def cumulative_cost_evaluations(me):
        ncost = 0
        for nk in me.cost_calls:
            if nk is not None:
                ncost += nk
        return ncost

    @property
    def cumulative_gradient_evaluations(me):
        ngrad = 0
        for nk in me.grad_calls:
            if nk is not None:
                ngrad += nk
        return ngrad

    @property
    def cumulative_hessian_matvecs(me):
        nhp = 0
        for nk in me.hess_matvecs:
            if nk is not None:
                nhp += nk
        return nhp

    def print(me) -> None:
        print(me.string())

    def string(me) -> str:
        ps = '\n'
        ps += '====================== Begin Newton CG convergence information ======================\n'
        ps += 'Preconditioned inexact Newton-CG with line search\n'
        ps += 'Hp=-g\n'
        ps += 'u <- u + alpha * p\n'
        ps += 'u: parameter, J: cost, g: gradient, H: Hessian, alpha=step size, p=search direction\n'
        ps += '\n'
        ps += 'it=0    : u=u0         -> J -> g -> build precond (optional) -> cgsolve Hp=-g\n'
        ps += 'it=1    : linesearch u -> J -> g -> build precond (optional) -> cgsolve Hp=-g\n'
        ps += '...\n'
        ps += 'it=last : linesearch u -> J -> g -> Done.\n'
        ps += '\n'
        ps += 'it:      Newton iteration number\n'
        ps += 'nCG:     number of CG iterations in Newton iteration\n'
        ps += 'nJ:      number of cost function evaluations in Newton iteration\n'
        ps += 'nG:      number of gradient evaluations in Newton iteration\n'
        ps += 'nHp:     number of Hessian-vector products in Newton iteration\n'
        ps += 'GN:      True (T) if Gauss-Newton Hessian is used, False (F) if Hessian is used\n'
        ps += 'BP:      True (T) if we built or rebuilt the preconditioner, False (F) otherwise.\n'
        ps += 'cost:    cost, J = Jd + Jr\n'
        ps += 'misfit:  misfit cost, Jd\n'
        ps += 'reg:     regularization cost, Jr\n'
        ps += '(g,p):   inner product between gradient, g, and Newton search direction, p\n'
        ps += '||g||L2: l2 norm of gradient\n'
        ps += 'alpha:   step size\n'
        ps += 'tolcg:   relative tolerance for Hp=-g CG solve (unpreconditioned residual decrease)\n'
        ps += '\n'
        ps += "{0:>2} {1:>3} {2:>2} {3:>2} {4:>3} {5:>2} {6:>2} {7:>12} {8:>7} {9:>7} {10:>8} {11:>7} {12:>7} {13:>7}\n".format(
            "it", "nCG", "nJ", "nG", "nHp", "GN", "BP", "cost", "misfit", "reg", "(g,p)", "||g||L2", "alpha", "tolcg")

        def format_int(x, k):
            if x is None:
                return ("{:" + str(k) + "}").format('')
            else:
                return ("{:" + str(k) + "d}").format(x)

        def format_float(x, k, pl):
            if x is None:
                s = '-'*4
            else:
                s = np.format_float_scientific(x, precision=k, trim='k', pad_left=pl, unique=False)
            return ("{:>" + str(k + 5 + pl) + "}").format(s)

        def format_bool(b, k):
            if b:
                s = 'T'
            else:
                s = 'F'
            return ("{:>" + str(k) + "}").format(s)

        for k in range(len(me.newton_iter)):
            ps += format_int(me.newton_iter[k], 2) + ' '
            ps += format_int(me.cg_iter[k], 3) + ' '
            ps += format_int(me.cost_calls[k], 2) + ' '
            ps += format_int(me.grad_calls[k], 2) + ' '
            ps += format_int(me.hess_matvecs[k], 3) + ' '
            ps += format_bool(me.gauss_newton[k], 2) + ' '
            ps += format_bool(me.build_precond[k], 2) + ' '
            ps += format_float(me.cost[k], 6, 1) + ' '
            ps += format_float(me.misfit[k], 1, 1) + ' '
            ps += format_float(me.reg_cost[k], 1, 1) + ' '
            ps += format_float(me.gdm[k], 1, 2) + ' '
            ps += format_float(me.gradnorm[k], 1, 1) + ' '
            ps += format_float(me.step_length[k], 1, 1) + ' '
            ps += format_float(me.cgtol[k], 1, 1) + '\n'

        ps += '\n'
        ps += 'converged : ' + str(me.converged) + '\n'
        ps += 'reason    : ' + str(me.reason) + '\n'
        ps += 'cumulative CG iterations : ' + str(me.cumulative_cg_iterations) + '\n'
        ps += 'cumulative cost evaluations : ' + str(me.cumulative_cost_evaluations) + '\n'
        ps += 'cumulative gradient evaluations : ' + str(me.cumulative_gradient_evaluations) + '\n'
        ps += 'cumulative Hessian vector products (excluding preconditioner builds) : ' + str(me.cumulative_hessian_matvecs) + '\n'

        ps += '======================= End Newton CG convergence information =======================\n'
        return ps


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
                rtol=1e-8,
                atol=1e-14,
                maxiter_newton=25,
                maxiter_cg=400,
                display=True,
                preconditioner_build_iters=(3,), # iterations at which one builds the preconditioner
                num_gn_iter=7, # Number of Gauss-Newton iterations (including, possibly, initial iterations)
                cg_coarse_tolerance=0.5,
                c_armijo=1e-4, # Armijo constant for sufficient reduction
                max_backtracking_iter=10, #Maximum number of backtracking iterations
                gdm_tol=1e-18, # we converge when (g,dm) <= gdm_tolerance
                forcing_sequence_power = 0.5, # p in min(0.5, ||g||^p)||g||
                gradnorm_ini = None,
                ):
    info = NCGInfo()

    it = 0
    info.start_new_iteration(it)

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    def cost_wrapper() -> tuple[scalar_type, scalar_type, scalar_type]:
        info.cost_calls[-1] += 1
        return cost()

    def gradient_wrapper() -> vector_type:
        info.grad_calls[-1] += 1
        return gradient()

    def apply_hessian_wrapper(u : vector_type) -> vector_type:
        info.hess_matvecs[-1] += 1
        return apply_hessian(u)

    def apply_gauss_newton_hessian_wrapper(u : vector_type) -> vector_type:
        info.hess_matvecs[-1] += 1
        return apply_gauss_newton_hessian(u)

    current_cost, current_misfit, current_reg_cost = cost_wrapper()
    info.cost[-1] = current_cost
    info.misfit[-1] = current_misfit
    info.reg_cost[-1] = current_reg_cost

    while (it < maxiter_newton) and (info.converged == False):
        using_gauss_newton = (it < num_gn_iter)
        info.gauss_newton[-1] = using_gauss_newton
        if using_gauss_newton:
            print('using Gauss-Newton Hessian')
            apply_H = apply_gauss_newton_hessian_wrapper
        else:
            print('using Hessian')
            apply_H = apply_hessian_wrapper
        printmaybe('it=', it, ', preconditioner_build_iters=', preconditioner_build_iters, ', num_gn_iter=', num_gn_iter,
                   ', using_gauss_newton=', using_gauss_newton)

        g = gradient_wrapper()
        gradnorm = np.sqrt(inner_product(g, g))
        info.gradnorm[-1] = gradnorm

        if it == 0:
            if gradnorm_ini is None:
                gradnorm_ini = gradnorm
            tol = max(atol, gradnorm_ini * rtol)

        if (gradnorm < tol) and (it > 0):
            info.converged = True
            info.reason = "Norm of the gradient less than tolerance"
            break

        tolcg = min(cg_coarse_tolerance, np.power(gradnorm / gradnorm_ini, forcing_sequence_power))
        info.cgtol[-1] = tolcg

        if (it in preconditioner_build_iters):
            printmaybe('building preconditioner')
            build_hessian_preconditioner()
            info.build_precond[-1] = True

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

        info.cg_iter[-1] = extras['iter']

        update_hessian_preconditioner(extras['X'], extras['Y'])

        alpha = 1.0
        descent = 0
        n_backtrack = 0

        gdm = inner_product(g, dm)
        info.gdm[-1] = gdm

        m = get_optimization_variable()

        if display:
            info.print()

        if callback:
            callback(it, m)

        it += 1
        info.start_new_iteration(it)

        cost_old = current_cost
        mstar = copy_vector(m)
        while descent == 0 and n_backtrack < max_backtracking_iter:
            # mstar = m + alpha*dm
            scal(0.0, mstar) # now mstar = 0
            axpy(1.0, m, mstar) # now mstar = m
            axpy(alpha, dm, mstar) # now mstar = m + alpha*dm

            set_optimization_variable(mstar)
            cost_new, misfit_new, reg_new = cost_wrapper()

            # Check if armijo conditions are satisfied
            if (cost_new < cost_old + alpha * c_armijo * gdm) or (-gdm <= gdm_tol):
                cost_old = cost_new
                descent = 1
                m = mstar
                set_optimization_variable(mstar)
            else:
                n_backtrack += 1
                alpha *= 0.5

        current_cost = cost_new
        current_misfit = misfit_new
        current_reg_cost = reg_new
        info.cost[-1] = current_cost
        info.misfit[-1] = current_misfit
        info.reg_cost[-1] = current_reg_cost
        info.step_length[-1] = alpha
        info.backtrack[-1] = n_backtrack

        if n_backtrack == max_backtracking_iter:
            info.converged = False
            info.reason = "Maximum number of backtracking reached"
            break

        if -gdm <= gdm_tol:
            info.converged = True
            info.reason = "Norm of (g, dm) less than tolerance"
            break

    if info.converged == False:
        g = gradient_wrapper()
        gradnorm = np.sqrt(inner_product(g, g))
        info.gradnorm[-1] = gradnorm

    if display:
        info.print()

    return info


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
               preconditioned_norm_stopping_criterion=False,
               save_vectors=False):

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    X = [] # list [x1, x2, ...]
    Y = []
    def apply_A_wrapper(x: vector_type) -> vector_type:
        y = apply_A(x)
        if save_vectors:
            X.append(copy_vector(x)) #OK
            Y.append(copy_vector(y))
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


    extras = {'iter': iter, 'X': X, 'Y': Y, 'converged': converged, 'reason': reason, 'final_norm': final_norm}
    return x, extras
