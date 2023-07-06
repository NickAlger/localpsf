import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

from nalger_helper_functions import custom_cg

geigs_psf = np.load('data/geigs_psf.npy')
geigs_none = np.load('data/geigs_none.npy')
geigs_reg = np.load('data/geigs_reg.npy')

gg_psf1 = geigs_psf[0,0,2,1,:]
gg_psf5 = geigs_psf[1,0,2,1,:]
gg_psf25 = geigs_psf[2,0,2,1,:]
gg_n = geigs_none[0,2,1,:]
gg_r = geigs_reg[0,2,1,:]

plt.figure()
plt.semilogy(gg_psf1)
plt.semilogy(gg_psf5)
plt.semilogy(gg_psf25)
plt.semilogy(gg_n)
plt.semilogy(gg_r)
plt.legend(['psf1', 'psf5', 'psf25', 'none', 'reg'])
plt.title('generalized eigenvalues')

kappa_n = np.max(gg_n) / np.min(gg_n)
kappa_r = np.max(gg_r) / np.min(gg_r)
print('kappa_n=', kappa_n)
print('kappa_r=', kappa_r)

N = len(gg_n)
b = np.random.randn(N)

Dn = np.diag(gg_n)
Dr = np.diag(gg_r)

x_true_n = np.linalg.solve(Dn,b)
out_n = custom_cg(Dn, b, display=True, x_true=x_true_n, maxiter=1000, tol=1e-15)

x_true_r = np.linalg.solve(Dr,b)
out_r = custom_cg(Dr, b, display=True, x_true=x_true_r, maxiter=1000, tol=1e-15)

plt.figure()
plt.semilogy(out_n[3][::5])
plt.semilogy(out_r[3][::5])
plt.legend(['none', 'reg'])