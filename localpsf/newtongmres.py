import numpy as np
import scipy.linalg as sla
import typing as typ
import numpy.typing as npt

vec = np.ndarray
scalar = float
vec2scalar = typ.Callable[[vec], scalar]
vec2vec = typ.Callable[[vec], vec]
inner_product: typ.Callable[[vec, vec], scalar] = np.dot
copy_vec: typ.Callable[[vec], vec] = np.copy
add_vecs: typ.Callable[[vec, vec], vec] = lambda u, v: u + v
scale_vec: typ.Callable[[scalar, vec], vec] = lambda a, u: a*u
norm: typ.Callable[[vec], scalar] = lambda u: np.sqrt(inner_product(u,u))


class NGMRESInfo:
    def __init__(me):
        me.newton_iter = list() # [1,2,3,...]
        me.gmres_iter = list() # number of GMRES iterations in each Newton iteration
        me.cost_calls = list() # number of times cost() is called in each Newton iteration
        me.grad_calls = list() # number of times gradient() is called in each Newton iteration
        me.hess_matvecs = list() # number of Hessian-vector products in the GMRES solve for each Newton iteration
        me.cost = list() # cost                at the beginning of each Newton iteration
        me.gdx = list()
        me.gradnorm = list()
        me.step_length = list()
        me.backtrack = list()
        me.gmrestol = list()
        me.build_precond = list()
        me.gauss_newton = list()
        me.converged = False
        me.reason = 'unknown reason'

    def start_new_iteration(me, newton_iter):
        me.newton_iter.append(newton_iter)
        me.gmres_iter.append(0)
        me.cost_calls.append(0)
        me.grad_calls.append(0)
        me.hess_matvecs.append(0)
        me.cost.append(None)
        me.gdx.append(None)
        me.gradnorm.append(None)
        me.step_length.append(None)
        me.backtrack.append(None)
        me.gmrestol.append(None)
        me.build_precond.append(False)
        me.gauss_newton.append(True)

    @property
    def cumulative_gmres_iterations(me):
        ncg = 0
        for nk in me.gmres_iter:
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

    def print(me):
        ps = '\n'
        ps += '===================== Begin Newton GMRES convergence information ====================\n'
        ps += 'Preconditioned inexact Newton-GMRES with line search\n'
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
        ps += 'nGMRES:     number of GMRES iterations in Newton iteration\n'
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
        ps += 'tolgmres:   relative tolerance for Hp=-g GMRES solve (unpreconditioned residual decrease)\n'
        ps += '\n'
        ps += "{0:>2} {1:>6} {2:>2} {3:>2} {4:>3} {5:>2} {6:>2} {7:>12} {8:>8} {9:>7} {10:>7} {11:>7}\n".format(
            "it", "nGMRES", "nJ", "nG", "nHp", "GN", "BP", "cost", "(g,p)", "||g||L2", "alpha", "tolgmres")

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
            ps += format_int(me.gmres_iter[k], 6) + ' '
            ps += format_int(me.cost_calls[k], 2) + ' '
            ps += format_int(me.grad_calls[k], 2) + ' '
            ps += format_int(me.hess_matvecs[k], 3) + ' '
            ps += format_bool(me.gauss_newton[k], 2) + ' '
            ps += format_bool(me.build_precond[k], 2) + ' '
            ps += format_float(me.cost[k], 6, 1) + ' '
            ps += format_float(me.gdx[k], 1, 2) + ' '
            ps += format_float(me.gradnorm[k], 1, 1) + ' '
            ps += format_float(me.step_length[k], 1, 1) + ' '
            ps += format_float(me.gmrestol[k], 1, 1) + '\n'

        ps += '\n'
        ps += 'converged : ' + str(me.converged) + '\n'
        ps += 'reason    : ' + str(me.reason) + '\n'
        ps += 'cumulative GMRES iterations : ' + str(me.cumulative_gmres_iterations) + '\n'
        ps += 'cumulative cost evaluations : ' + str(me.cumulative_cost_evaluations) + '\n'
        ps += 'cumulative gradient evaluations : ' + str(me.cumulative_gradient_evaluations) + '\n'
        ps += 'cumulative Hessian vector products (excluding preconditioner builds) : ' + str(me.cumulative_hessian_matvecs) + '\n'

        ps += '======================= End Newton GMRES convergence information =======================\n'
        print(ps)


def newtongmres_ls(set_x: typ.Callable[[vec], None],
                   get_x: typ.Callable[[], vec],
                   cost: typ.Callable[[], scalar],
                   grad: typ.Callable[[], vec],
                   apply_hess: vec2vec,
                   apply_gn_hess: vec2vec,
                   build_hess_preconditioner: typ.Callable[[], None],
                   apply_hess_preconditioner: vec2vec,
                   solve_hess_preconditioner: vec2vec,
                   callback: typ.Callable[[int, vec], None]=None, # callback(k, x)
                   rtol: float=1e-8,
                   atol: float=1e-14,
                   maxiter_newton: int=25,
                   maxiter_gmres: int=200,
                   display: bool=True,
                   preconditioner_build_iters: typ.Collection[int]=(3,), # iterations at which one builds the preconditioner
                   num_gn_iter: int=7, # Number of Gauss-Newton iterations (including, possibly, initial iterations)
                   gmres_coarse_tolerance: float=0.5,
                   c_armijo: float=1e-4, # Armijo constant for sufficient reduction
                   max_backtracking_iter: int=10, #Maximum number of backtracking iterations
                   gdx_tol: float=1e-18, # we converge when (g,dm) <= gdm_tolerance
                   forcing_sequence_power: float = 0.5, # p in min(0.5, ||g||^p)||g||
                  ) -> typ.Tuple[vec, NGMRESInfo]: # solution x, convergence information
    info = NGMRESInfo()

    it = 0
    info.start_new_iteration(it)

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    def cost_wrapper() -> scalar:
        info.cost_calls[-1] += 1
        return cost()

    def grad_wrapper() -> vec:
        info.grad_calls[-1] += 1
        return grad()

    def apply_hess_wrapper(z : vec) -> vec:
        info.hess_matvecs[-1] += 1
        return apply_hess(z)

    def apply_gn_hess_wrapper(z : vec) -> vec:
        info.hess_matvecs[-1] += 1
        return apply_gn_hess(z)

    x: vec = get_x()
    current_cost: scalar = cost_wrapper()
    info.cost[-1] = current_cost

    while (it < maxiter_newton) and (info.converged == False):
        using_gauss_newton: bool = (it < num_gn_iter)
        info.gauss_newton[-1] = using_gauss_newton
        if using_gauss_newton:
            print('using Gauss-Newton Hessian')
            apply_H = apply_gn_hess_wrapper
        else:
            print('using Hessian')
            apply_H = apply_hess_wrapper
        printmaybe('it=', it, ', preconditioner_build_iters=', preconditioner_build_iters, ', num_gn_iter=', num_gn_iter,
                   ', using_gauss_newton=', using_gauss_newton)

        g = grad_wrapper()
        gradnorm = np.sqrt(inner_product(g, g))
        info.gradnorm[-1] = gradnorm

        if it == 0:
            gradnorm_ini = gradnorm
            tol = max(atol, gradnorm_ini * rtol)

        if (gradnorm < tol) and (it > 0):
            info.converged = True
            info.reason = "Norm of the gradient less than tolerance"
            break

        tolgmres = min(gmres_coarse_tolerance, np.power(gradnorm / gradnorm_ini, forcing_sequence_power))
        info.gmrestol[-1] = tolgmres

        if (it in preconditioner_build_iters):
            printmaybe('building preconditioner')
            build_hess_preconditioner()
            info.build_precond[-1] = True

        minus_g = scale_vec(-1.0, g)

        dx, gmres_info = positive_gmres(apply_H, minus_g,
                                        apply_M=apply_hess_preconditioner,
                                        solve_M=solve_hess_preconditioner,
                                        rtol=tolgmres,
                                        display=True,
                                        max_iter=maxiter_gmres)

        info.gmres_iter[-1] = gmres_info['iter']

        alpha: float = 1.0
        descent: bool = False
        n_backtrack: int = 0

        gdx: scalar = inner_product(g, dx)
        info.gdx[-1] = gdx

        if display:
            info.print()

        if callback:
            callback(it, x)

        it += 1
        info.start_new_iteration(it)

        cost_old: scalar = current_cost
        while (not descent) and (n_backtrack < max_backtracking_iter):
            xstar: vec = add_vecs(x, scale_vec(alpha, dx)) # xstar = x + alpha*dx
            set_x(xstar)
            cost_new: scalar = cost_wrapper()

            # Check if armijo conditions are satisfied
            if (cost_new < cost_old + alpha * c_armijo * gdx) or (-gdx <= gdx_tol):
                cost_old = cost_new
                descent = True
                x = xstar
                set_x(x)
            else:
                n_backtrack += 1
                alpha *= 0.5

        current_cost = cost_new
        info.cost[-1] = current_cost
        info.step_length[-1] = alpha
        info.backtrack[-1] = n_backtrack

        if n_backtrack == max_backtracking_iter:
            set_x(x)
            info.converged = False
            info.reason = "Maximum number of backtracking reached"
            break

        if -gdx <= gdx_tol:
            info.converged = True
            info.reason = "Norm of (g, dm) less than tolerance"
            break

    if info.converged == False:
        g: vec = grad_wrapper()
        gradnorm: scalar = norm(g)
        info.gradnorm[-1] = gradnorm

    if display:
        info.print()

    return x, info



def positive_gmres(apply_A: vec2vec,  # Coefficient operator. R^N -> R^N
                   b: vec,  # Right hand side vector. shape=(N,)
                   x0: vec=None,  # Initial guess. shape=(N,)
                   apply_M: vec2vec=None,  # Preconditioner. R^N -> R^N
                   solve_M: vec2vec=None,  # Preconditioner. R^N -> R^N
                   rtol: float=1e-10,
                   max_iter: int=50,
                   terminate_negative_direction: bool=True,
                   display: bool=True,
                   callback: typ.Callable[[vec, scalar, scalar], None]=None, # callback(xk, norm_r, xt_A_x)
                   ) -> typ.Tuple[vec, dict]:
    '''Solves linear system Ax=b using GMRES.
    Modified from Algorithm 6.9 in Saad iterative methods book page 172. No restarts. MGS orthogonalization.
    By default, method terminates when negative direction encountered and returns last positive direction.
    '''
    assert(len(b.shape) == 1)
    N = len(b)

    if solve_M is None or apply_M is None:
        assert(solve_M is None)
        assert(apply_M is None)
        apply_M = lambda x: x
        solve_M = lambda x: x

    def printmaybe(stuff):
        if display:
            print(stuff)

    if callback is None:
        callback = lambda xk, nr, xAx: None

    norm_b = np.linalg.norm(b)

    if x0 is None:
        x0 = np.zeros(N)
        Ax0 = np.zeros(N)
    else:
        assert(x0.shape == (N,))
        Ax0 = apply_A(x0)
    r0 = solve_M(b - Ax0)

    beta = np.linalg.norm(r0)

    V_big = np.zeros((N, max_iter + 1))
    V_big[:,0] = r0 / beta

    norm_r = np.linalg.norm(b - Ax0)
    xt_A_x = np.dot(x0, Ax0)
    callback(x0, norm_r, xt_A_x)
    printmaybe('initial guess: norm_r/norm_b=' + str(norm_r / norm_b) + ', xt_A_x=' + str(xt_A_x))
    it = 0

    converged = False
    reason = 'maximum iterations reached'
    x = x0
    H_big = np.zeros((max_iter + 1, max_iter))
    for jj in range(max_iter):
        it += 1
        w = solve_M(apply_A(V_big[:, jj]))
        H_big[:jj+1, jj] = w @ V_big[:, :jj+1]
        w = w - V_big[:, :jj+1] @ H_big[:jj+1, jj]
        h = np.linalg.norm(w)
        H_big[jj+1, jj] = h
        if h < 1e-15*norm_b:
            reason = 'good breakdown'
            printmaybe('GMRES success: good breakdown')
            break
        V_big[:,jj+1] = w / h

        H = H_big[:jj+2, :jj+1] # shape=(j+1, j)
        V = V_big[:, :jj+1] # shape=(N, j)

        e = np.zeros(jj+2)
        e[0] = 1.0

        y = sla.lstsq(H, beta*e)[0]

        x_prop = x0 + V @ y
        Ax_prop = Ax0 + apply_M(V_big[:, :jj+2] @ (H @ y))
        norm_r = np.linalg.norm(b - Ax_prop)

        xt_A_x = np.dot(x_prop, Ax_prop)

        printmaybe('jj=' + str(jj) + ', norm_r/norm_b=' + str(norm_r / norm_b) + ', xt_A_x=' + str(xt_A_x))
        callback(x_prop, norm_r, xt_A_x)

        if terminate_negative_direction:
            if xt_A_x < 0:
                if jj == 0:
                    x = x_prop
                reason = 'negative direction encountered'
                printmaybe('GMRES terminated: negative direction encountered')
                break

        x = x_prop

        if norm_r < rtol * norm_b:
            converged = True
            reason = 'achieved desired tolerance'
            printmaybe('GMRES success: rtol achieved')
            break

    info = {'iter': it, 'converged': converged, 'reason': reason, 'final_norm': norm_r}
    return x, info



if __name__ == "__main__":
    N = 100
    A = np.random.randn(N,N)
    x_true = np.random.randn(N)
    apply_A = lambda p: A @ p
    b = apply_A(x_true)

    M = A + 0.1 * np.random.randn(*A.shape)
    iM = np.linalg.inv(M)
    apply_M = lambda p: M @ p
    solve_M = lambda p: iM @ p

    xx = []
    all_norm_r = []
    all_xAx = []
    def callback(xk, nr, xAx):
        xx.append(xk)
        all_norm_r.append(nr)
        all_xAx.append(xAx)


    x, info = positive_gmres(apply_A, b, max_iter=100, apply_M=apply_M, solve_M=solve_M, callback=callback)
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    relres = np.linalg.norm(b - apply_A(x)) / np.linalg.norm(b)
    xt_A_x = np.dot(x, apply_A(x))
    print('err=', err, ', relres=', relres, ', xt_A_x=', xt_A_x)

