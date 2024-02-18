import numpy as np
import matplotlib.pyplot as plt



def phi(
    yy: np.ndarray, # shape=(N,)
    x: float,
    shape_parameter: float,
):
    return np.exp(-(shape_parameter * (yy - x))**2)

def make_B(
    xx: np.ndarray, # shape=(N,)
    shape_parameter: float
):
    N = len(xx)
    B = np.zeros((N,N))
    for ii in range(N):
        B[:,ii] = phi(xx, xx[ii], shape_parameter)
    return B

def eval_interpolant(
    xx_test: np.ndarray, # shape=(N_test,)
    xx: np.ndarray, # shape=(N,)
    ww: np.ndarray, # shape=(N,)
    shape_parameter: float,
):
    N_test = len(xx_test)
    ff_test = np.zeros(N_test)
    for ii in range(N_test):
        ff_test[ii] = np.sum(ww * phi(xx, xx_test[ii], shape_parameter))
    return ff_test

xx = np.linspace(0, 1, 10)
xx_test = np.linspace(0, 1, 500)
true_f = lambda x: np.cos(2*np.pi * x)

ff = true_f(xx)
ff[int(len(ff)/2)] = 0.0

ff_test_true = true_f(xx_test)

shape_parameters = [1e0, 1e1, 1e2]
for shape_parameter in shape_parameters:
    B = make_B(xx, shape_parameter)
    ww = np.linalg.solve(B, ff)
    ff_test = eval_interpolant(xx_test, xx, ww, shape_parameter)

    plt.figure()
    plt.plot(xx_test, ff_test_true, '--k')
    plt.plot(xx_test, ff_test, 'k')
    plt.scatter(xx, ff, c='k')
    plt.ylim(-1.2, 1.2)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig('rbf_cosine_shape_parameter='+str(shape_parameter)+'.pdf', bbox_inches='tight')

#

N = 10
hh = list(np.logspace(np.log10(0.05), np.log10(5), 30))
shape_parameters0 = np.logspace(0, 2, 5)[::-1]

all_errs = np.zeros((len(hh), len(shape_parameters0)))
for ii, h in enumerate(hh):
    xx = np.linspace(0, h, N)
    xx_test = np.linspace(0, h, 500)
    true_f = lambda x: np.cos(2 * np.pi * x)

    ff = true_f(xx)
    ff_test_true = true_f(xx_test)

    shape_parameters = list(shape_parameters0 / h)
    for jj, shape_parameter in enumerate(shape_parameters):
        B = make_B(xx, shape_parameter)
        ww = np.linalg.solve(B, ff)
        ff_test = eval_interpolant(xx_test, xx, ww, shape_parameter)

        err = np.linalg.norm(ff_test - ff_test_true) / np.linalg.norm(ff_test_true)
        print('h=', h, ', shape_parameter=', shape_parameter, ', err=', err)
        all_errs[ii,jj] = err

plt.figure()
leg = []
for ii in range(all_errs.shape[1]):
    plt.loglog(hh, all_errs[:,ii])
    leg.append(r'$\epsilon_0=$' + np.format_float_scientific(shape_parameters0[ii], precision=2))
plt.xlabel(r'$h$')
plt.ylabel('$||f - \widetilde{f}|| / ||\widetilde{f}||$')
plt.legend(leg)
plt.title(r'RBF error approximating $\cos(2 \pi x)$ on $[0,h]$')
plt.savefig('cosine_RBF_convergence.pdf', bbox_inches='tight')

np.savetxt('cos_rbf_hh.txt', hh)
np.savetxt('cos_rbf_shape_parameters0.txt', shape_parameters0)
np.savetxt('cos_rbf_errs.txt', all_errs)

