# Copyright (c) 2016-2018, The University of Texas at Austin
# & University of California, Merced.
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

import numpy as np
from collections.abc import Callable


scalar_type = float
vector_type = np.ndarray

def numpy_axpy(a: scalar_type, x: vector_type, y: vector_type) -> None: # y <- y + a*x
    y += a*x

def numpy_scal(a: scalar_type, x: vector_type) -> None:
    x *= a


def constrained_newton(set_optimization_variable: Callable[[vector_type], None],
                       get_optimization_variable: Callable[[],            vector_type],
                       energy:                    Callable[[],            scalar_type],
                       gradient:                  Callable[[],            vector_type],
                       solve_hessian:             Callable[[vector_type], vector_type],
                       constraint_vec: vector_type,
                       inner_product: Callable[[vector_type, vector_type], vector_type]        = np.dot,
                       copy_vector:   Callable[[vector_type],              vector_type]        = np.copy,
                       axpy:          Callable[[scalar_type, vector_type,  vector_type], None] = numpy_axpy, # axpy(a, x, y): y <- y + a*x
                       scal:          Callable[[scalar_type, vector_type], None]               = numpy_scal,  # scal(a, x): x <- a*x
                       constraint_tol=1e-6,
                       max_iter=20, # maximum number of iterations for nonlinear forward solve
                       rtol=1e-6, # we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance
                       atol=1e-9, # we converge when sqrt(g,g) <= abs_tolerance
                       gdu_tol=1e-18, # we converge when (g,du) <= gdu_tolerance
                       c_armijo=1e-4,
                       max_backtrack=10,
                       display=True):
    it = 0

    norm = lambda x: np.sqrt(inner_product(x, x))

    if display:
        print("Solving Nonlinear Problem")

    Fn = energy()
    gn = gradient()
    g0_norm = norm(gn)
    gn_norm = g0_norm
    tol = max(g0_norm * rtol, atol)

    if display:
        print("{0:>3} {1:>15} {2:>15} {3:>15} {4:>15}".format(
            "Nit", "Energy", "||g||", "(g,du)", "alpha"))
        print("{0:3d} {1:15e} {2:15e}  {3:15}   {4:15}".format(
            0, Fn, g0_norm, "    -    ", "    -"))

    converged = False
    reason = "Maximum number of Iteration reached"

    for it in range(max_iter):
        u = get_optimization_variable()

        # Ensure that at the end of the first iteration
        # the linear constraint is satisfied.
        # If the constraint is not satisfied we find the minimum energy
        # to satisfy the constraint
        if it == 0:
            constraint_violation = gn * constraint_vec
            if norm(constraint_violation) > constraint_tol:
                du = solve_hessian(-constraint_violation)
                axpy(1.0, du, u) # u <- 1.0*du + u
                Fn = energy()
                gn = gradient()
                continue

        du = solve_hessian(-gn)
        du_gn = inner_product(du, gn)

        alpha = 1.0
        if (np.abs(du_gn) < gdu_tol):
            converged = True
            reason = "Norm of (g, du) less than tolerance"
            axpy(alpha, du, u) # u <- u + alpha * du
            Fn = energy()
            gn = gradient()
            gn_norm = norm(gn)
            break

        u_backtrack = copy_vector(u)

        # Backtrack
        bk_converged = False
        Fn_tmp = Fn
        for j in range(max_backtrack):
            # u_backtrack = u + alpha * du
            scal(0.0, u_backtrack) # now u_backtrack = 0
            axpy(1.0, u, u_backtrack) # now u_backtrack = u
            axpy(alpha, du, u_backtrack) # now u_backtrack = u + alpha * du

            set_optimization_variable(u_backtrack)
            Fnext = energy()
            if Fnext < Fn + alpha * c_armijo * du_gn:
                Fn = Fnext
                bk_converged = True
                break
            alpha /= 2.

        if not bk_converged:
            reason = "Maximum number of backtracking reached"
            if display:
                print("{0:3d}  {1:15e} {2:15e} {3:15e} {4:15e}".format(it + 1, Fn, gn_norm, du_gn, alpha))
            break

        gn_norm = norm(gn)

        if (gn_norm < tol) or (
                np.abs(Fnext - Fn_tmp) / np.abs(Fn_tmp) < 1e-6):  # NICK 5/23/22. Added extra stopping criterion
            print('np.abs(Fnext - Fn_tmp)/np.abs(Fn_tmp)=', np.abs(Fnext - Fn_tmp) / np.abs(Fn_tmp))
            converged = True
            reason = "Norm of the gradient less than tolerance"

            if display:
                print("{0:3d}  {1:15e} {2:15e} {3:15e} {4:15e}".format(it + 1, Fn, gn_norm, du_gn, alpha))

            break

        if display:
            print("{0:3d}  {1:15e} {2:15e} {3:15e} {4:15e}".format(it + 1, Fn, gn_norm, du_gn, alpha))

    it = it + 1
    if display:
        if reason == "Norm of (g, du) less than tolerance":
            print("{0:3d}   {1:15e} {2:15e} {3:15e} {4:15e}".format(
                it, Fn, gn_norm, du_gn, alpha))
        print(reason)
        if converged:
            print("Newton converged in ", it, "nonlinear iterations.")
        else:
            print("Newton did NOT converge in ", it, "iterations.")

        print("Final norm of the gradient: ", gn_norm)
        print("Value of the cost functional: ", Fn)

    return u, reason

