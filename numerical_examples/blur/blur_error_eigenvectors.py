import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dolfin as dl
import localpsf.localpsf_cg1_lumped as lpsf1
from localpsf.blur import *
import scipy.linalg as sla
from functools import partial
from nalger_helper_functions import plot_ellipse
import hlibpro_python_wrapper as hpro
from tqdm.auto import tqdm


num_batches = 50
nx = 89 #63
a = 1.0 # <-- How bumpy? Use 1.0, which is maximum bumpiness without negative numbers
length_scaling = 1.0 / (2.0**2)
tau = 4.0
num_neighbors=10
rotation_rate = 1.0


Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a, rotation_rate=rotation_rate)

psf_object = lpsf1.make_psf_fenics(
    lambda X: H @ X,
    lambda X: H.T @ X,
    Vh, Vh,
    mass_lumps, mass_lumps,
    num_initial_batches=num_batches,
    tau=tau, display=True,
    num_neighbors=num_neighbors
)

k=510
col_func = dl.Function(Vh)
col_func.vector()[:] = Ker[:,k].copy()
plt.figure()
cm = dl.plot(col_func)
plt.colorbar(cm)
plt.title('col_func, k=' +str(k))

sp = psf_object.psf_object.impulse_response_batches.sample_points.copy()

Ker_psf = psf_object.psf_object.psf_kernel.cpp_object.eval_integral_kernel_block(dof_coords.T, dof_coords.T)

err_psf = np.linalg.norm(Ker - Ker_psf) / np.linalg.norm(Ker_psf)
print('err_psf=', err_psf)

#

ker_err_func = dl.Function(Vh)
ker_err_vec = np.linalg.norm(Ker_psf - Ker, axis=0) / np.linalg.norm(Ker, axis=0)
# ker_err_vec = np.linalg.norm(Ker_psf - Ker, axis=0) / (np.linalg.norm(Ker) / np.sqrt(Ker.shape[1]))
ker_err_vec[np.linalg.norm(Ker, axis=0) == 0] = 0.0
ker_err_func.vector()[:] = ker_err_vec

plt.figure()
# cm = dl.plot(ker_err_func, cmap='binary', markersize=5)
cm = dl.plot(ker_err_func, cmap='binary', markersize=5, vmax=0.02)
plt.plot(sp[:, 0], sp[:, 1], '.', c='k')
plt.colorbar(cm)
# cm.set_clim(0.0, 1.0)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().set_aspect('equal')


#

raise RuntimeError

E_H = H - Ker_psf * np.outer(mass_lumps, mass_lumps)

E = Ker - Ker_psf

E_diag_func = dl.Function(Vh)
E_diag_func.vector()[:] = E.diagonal().copy()
plt.figure()
cm = dl.plot(E_diag_func)
plt.colorbar(cm)
plt.title('E_diag_func')

Ker_diag_func = dl.Function(Vh)
Ker_diag_func.vector()[:] = Ker.diagonal().copy()
plt.figure()
cm = dl.plot(Ker_diag_func)
plt.colorbar(cm)
plt.title('Ker_diag_func')

E_colsum_func = dl.Function(Vh)
E_colsum_func.vector()[:] = np.sum(E, axis=0)
plt.figure()
cm = dl.plot(E_colsum_func)
plt.colorbar(cm)
plt.title('E_colsum_func')

Ker_colsum_func = dl.Function(Vh)
Ker_colsum_func.vector()[:] = np.sum(Ker, axis=0)
plt.figure()
cm = dl.plot(Ker_colsum_func)
plt.colorbar(cm)
plt.title('Ker_colsum_func')

E_rowsum_func = dl.Function(Vh)
E_rowsum_func.vector()[:] = np.sum(E, axis=1)
plt.figure()
cm = dl.plot(E_rowsum_func)
plt.colorbar(cm)
plt.title('E_rowsum_func')

Ker_rowsum_func = dl.Function(Vh)
Ker_rowsum_func.vector()[:] = np.sum(Ker, axis=1)
plt.figure()
cm = dl.plot(Ker_rowsum_func)
plt.colorbar(cm)
plt.title('Ker_rowsum_func')

# Ker2 = Ker / np.sum(Ker, axis=0).reshape((-1,1))
# Ker_psf2 = Ker_psf  / np.sum(Ker_psf, axis=0).reshape((-1,1))
#
# Ker2[np.isnan(Ker2)] = 0.0
#
# E2 = Ker2 - Ker_psf2
# E2[np.isnan(E2)] = 0.0
#
# E_diag_func = dl.Function(Vh)
# E_diag_func.vector()[:] = E2.diagonal().copy()
# plt.figure()
# cm = dl.plot(E_diag_func)
# plt.colorbar(cm)
# plt.title('E_diag_func')

err_psf2 = np.linalg.norm(Ker2 - Ker_psf2) / np.linalg.norm(Ker2)
print('err_psf2=', err_psf2)

U, ss, Vt = np.linalg.svd(E)

ssker = (U.T @ Ker @ Vt.T).diagonal()

plt.figure()
plt.semilogy(ss)
plt.semilogy(ssker)
plt.title('error singular values')

k = 0 #585
uk_vec = np.abs(U[:,k].copy())
uk = dl.Function(Vh)
uk.vector()[:] = uk_vec
s = ss[k]
vk_vec = np.abs(Vt[k,:].copy())
vk = dl.Function(Vh)
vk.vector()[:] = vk_vec

sker = uk_vec.T @ Ker @ vk_vec
print('sker', sker, ', s=', s)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
cm = dl.plot(uk, cmap='binary')
plt.colorbar(cm)
plt.plot(sp[:,0], sp[:,1], '.r', markersize=2)
plt.title('uk, k=' + str(k) + ', sval='+str(s))

plt.subplot(1,2,2)
cm = dl.plot(vk, cmap='binary')
plt.colorbar(cm)
plt.plot(sp[:,0], sp[:,1], '.r', markersize=2)
plt.title('vk, k=' + str(k) + ', sker='+str(sker))

raise RuntimeError

ee0, P0 = np.linalg.eig(E)

sort_inds = np.argsort(np.abs(ee0))[::-1]

ee = ee0[sort_inds]
P = P0[:,sort_inds]

plt.figure()
plt.semilogy(np.abs(ee))

k = 4 #585
evec = dl.Function(Vh)
evec.vector()[:] = np.abs(P[:,k].copy())
eig = ee[k]

plt.figure()
cm = dl.plot(evec, cmap='binary')
plt.colorbar(cm)
plt.plot(sp[:,0], sp[:,1], '.r')
plt.title('Error eigenvector k=' + str(k) + ', eig='+str(eig))