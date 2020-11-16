import numpy as np
from numba import njit

_cg = 0.3819660
_mintol = 1e-11

@njit
def brent_minimize(func, xmin, xmax, args=(), tol=1e-7, maxiter=200):
    # Based on version in scipy.optimize
    xa = xmin
    xc = xmax
    xb = (xa + xc) / 2.

    funcalls = 0

    x = w = v = xb
    fw = fv = fx = func(*((x,) + args))
    if (xa < xc):
        a = xa
        b = xc
    else:
        a = xc
        b = xa
    deltax = 0.0
    funcalls += 1
    iter = 0
    while (iter < maxiter):
        tol1 = tol * np.abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        # check for convergence
        if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        # XXX In the first iteration, rat is only bound in the true case
        # of this conditional. This used to cause an UnboundLocalError
        # (gh-4140). It should be set before the if (but to what?).
        if (np.abs(deltax) <= tol1):
            if (x >= xmid):
                deltax = a - x  # do a golden section step
            else:
                deltax = b - x
            rat = _cg * deltax
        else:  # do a parabolic step
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if (tmp2 > 0.0):
                p = -p
            tmp2 = np.abs(tmp2)
            dx_temp = deltax
            deltax = rat
            # check parabolic fit
            if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                    (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                rat = p * 1.0 / tmp2  # if parabolic step is useful.
                u = x + rat
                if ((u - a) < tol2 or (b - u) < tol2):
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if (x >= xmid):
                    deltax = a - x  # if it's not do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax

        if (np.abs(rat) < tol1):  # update by at least tol1
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = func(*((u,) + args))  # calculate new output value
        funcalls += 1

        if (fu > fx):  # if it's bigger than current
            if (u < x):
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if (u >= x):
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu

        iter += 1
    #################################
    # END CORE ALGORITHM
    #################################

    xmin = x
    fval = fx
    iter = iter
    funcalls = funcalls
    return xmin, fval, iter, funcalls