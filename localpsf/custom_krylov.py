import numpy as np
import scipy.linalg as sla
import typing as typ
import numpy.typing as npt


def positive_gmres(apply_A: typ.Callable[[np.ndarray], np.ndarray],  # R^N -> R^N
                   b: np.ndarray,  # shape=(N,)
                   x0: np.ndarray=None,  # shape=(N,)
                   apply_M: typ.Callable[[np.ndarray], np.ndarray]=None,  # R^N -> R^N
                   solve_M: typ.Callable[[np.ndarray], np.ndarray]=None,  # R^N -> R^N
                   rtol: float=1e-10,
                   max_iter=50,
                   terminate_negative_direction=True,
                   display=True,
                   callback: typ.Callable[[np.ndarray, float, float], None]=None, # callback(xk, norm_r, xt_A_x)
                   ) -> np.ndarray:
    '''Modified from Algorithm 6.9 in Saad iterative methods book page 172. No restarts.
    By default, method terminates when negative direction encountered and returns last positive direction.'''
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

    x = x0
    H_big = np.zeros((max_iter + 1, max_iter))
    for jj in range(max_iter):
        w = solve_M(apply_A(V_big[:, jj]))
        H_big[:jj+1, jj] = w @ V_big[:, :jj+1]
        w = w - V_big[:, :jj+1] @ H_big[:jj+1, jj]
        h = np.linalg.norm(w)
        H_big[jj+1, jj] = h
        if h < 1e-15*norm_b:
            printmaybe('GMRES success: good breakdown')
            break
        V_big[:,jj+1] = w / h

        H = H_big[:jj+2, :jj+1] # shape=(j+1, j)
        V = V_big[:, :jj+1] # shape=(N, j)

        e = np.zeros(jj+2)
        e[0] = 1.0

        y = sla.lstsq(H, beta*e)[0]
        # norm_r = np.linalg.norm(beta*e - H @ y)

        x_prop = x0 + V @ y
        # norm_r_true = np.linalg.norm(solve_M(b - apply_A(x_prop)))
        # print('norm_r=', norm_r)
        # print('norm_r_true=', norm_r_true)

        Ax_prop = Ax0 + apply_M(V_big[:, :jj+2] @ (H @ y))
        norm_r = np.linalg.norm(b - Ax_prop)
        # Ax_true = apply_A(x_prop)
        #
        # print('Ax error:', np.linalg.norm(Ax_prop - Ax_true)/np.linalg.norm(Ax_true))

        xt_A_x = np.dot(x_prop, Ax_prop)
        # xt_A_x_true = np.dot(x_prop, apply_A(x_prop))
        # print('xt_A_x=', xt_A_x)
        # print('xt_A_x_true=', xt_A_x_true)

        printmaybe('jj=' + str(jj) + ', norm_r/norm_b=' + str(norm_r / norm_b) + ', xt_A_x=' + str(xt_A_x))
        callback(x_prop, norm_r, xt_A_x)

        if terminate_negative_direction:
            if xt_A_x < 0:
                if jj == 0:
                    x = x_prop
                printmaybe('GMRES terminated: negative direction encountered')
                break

        x = x_prop

        if norm_r < rtol * norm_b:
            printmaybe('GMRES success: rtol achieved')
            break

    return x



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


    x = positive_gmres(apply_A, b, max_iter=100, apply_M=apply_M, solve_M=solve_M, callback=callback)
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    relres = np.linalg.norm(b - apply_A(x)) / np.linalg.norm(b)
    xt_A_x = np.dot(x, apply_A(x))
    print('err=', err, ', relres=', relres, ', xt_A_x=', xt_A_x)

