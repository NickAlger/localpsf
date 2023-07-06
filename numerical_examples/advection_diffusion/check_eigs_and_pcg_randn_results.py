import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

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

N = len(gg_n)
b = np.random.randn(N)

