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


nx = 49
length_scaling = 1.0
aa = [1.0, 20.0, 17.0]

for a in aa:
    Ker, phi_function, Vh, H, mass_lumps, dof_coords, kdtree_sort_inds = frog_setup(nx, length_scaling, a)

    p0 = np.array([0.5, 0.5])
    ii = np.argmin(np.linalg.norm(dof_coords - p0.reshape((1, -1)), axis=1))
    p = dof_coords[ii,:]

    phi = dl.Function(Vh)
    # phi.vector()[:] = phi_function(dof_coords, p)
    phi_vec = Ker[:, ii].copy()
    phi_vec[np.abs(phi_vec) < 1e-1] = 0.0
    phi.vector()[:] = phi_vec

    #

    psf_object = lpsf1.make_psf_fenics(
        lambda X: H @ X,
        lambda X: H.T @ X,
        Vh, Vh,
        mass_lumps, mass_lumps,
        num_initial_batches=0,
        tau = 3.0, display=True,
        num_neighbors = 10
    )

    #

    num_pts_1d = 500
    coords_1d = np.array([np.linspace(0, 1, num_pts_1d), p[1]*np.ones(num_pts_1d)]).T # vertical line

    rr_1d = phi_function(coords_1d, p)

    plt.figure()
    plt.plot(rr_1d)
    plt.gca().set_xticks([])
    # plt.title('rr_1d')

    plt.savefig('frog_1d_a=' + str(a) + '.png', bbox_inches='tight', dpi=400)

    plt.figure()
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # plt.title('frog1')
    plt.plot(coords_1d[:,0], coords_1d[:,1], '.r')
    plt.xlim(0.3, 0.7)
    plt.ylim(0.1, 0.9)

    mu_p = np.array([psf_object.mu(0)(p), psf_object.mu(1)(p)])
    Sigma_p = np.array([[psf_object.Sigma(0,0)(p), psf_object.Sigma(0,1)(p)],
                        [psf_object.Sigma(1,0)(p), psf_object.Sigma(1,1)(p)]])

    tau=3.0
    plot_ellipse(mu_p, Sigma_p, n_std_tau=tau, facecolor='none', edgecolor='k', linewidth=1)


    plt.savefig('frog_ellipsoid_a=' + str(a) + '.png', bbox_inches='tight', dpi=400)

raise RuntimeError

#### Impulse response grid plot

num_pts_x = 5
num_pts_y = 5
pX, pY = np.meshgrid(np.linspace(0,1,num_pts_x), np.linspace(0,1,num_pts_y)[::-1])
pp = np.vstack([pX.reshape(-1), pY.reshape(-1)]).T

plt.figure(figsize = (num_pts_x,num_pts_y))
gs1 = gridspec.GridSpec(num_pts_x, num_pts_y)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
for k in range(len(pp)):
    p = pp[k]
    ii = nearest_ind_func(dof_coords, p)
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.subplot(gs1[k])
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

plt.savefig('frog_impulse_response_grid.png', bbox_inches='tight', dpi=300)


#### Plot two impulse responses individually, and kernel with the corresponding columns indicated

p1 = np.array([0.2, 0.45])
p2 = np.array([0.6, 0.6])
pp = [p1, p2]

plt.matshow(Ker, cmap='binary')
plt.ylim([Ker.shape[0], 0])
plt.gca().set_xticks([])
plt.gca().set_yticks([])

for k in range(len(pp)):
    p = pp[k]
    ii = nearest_ind_func(dof_coords, p)
    print('k=', k, ', ii=', ii)
    plt.plot([ii, ii], [0, Ker.shape[0]], 'k', linestyle='dotted', linewidth=2.0)

plt.gcf().set_size_inches(8, 8)
plt.savefig('frog_kernel_matrix_a1.png', bbox_inches='tight', dpi=400)

for k in range(len(pp)):
    p = pp[k]
    ii = nearest_ind_func(dof_coords, p)
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.figure(figsize=(3.8, 3.8))
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig('frog_impulse_response' + str(k) + '.png', bbox_inches='tight', dpi=400)


#### Create PSF approximation

psf_object = lpsf1.make_psf_fenics(
    lambda X: H @ X,
    lambda X: H.T @ X,
    Vh, Vh,
    mass_lumps, mass_lumps,
    tau = 3.0, display=True,
)

#### Plot one impulse response with ellipsoid

p0 = np.array([0.6, 0.6])
ii = np.argmin(np.linalg.norm(dof_coords - p0.reshape((1, -1)), axis=1))
p = dof_coords[ii,:]

mu_p = np.array([psf_object.mu(0)(p), psf_object.mu(1)(p)])
Sigma_p = np.array([[psf_object.Sigma(0,0)(p), psf_object.Sigma(0,1)(p)],
                    [psf_object.Sigma(1,0)(p), psf_object.Sigma(1,1)(p)]])

phi.vector()[:] = Ker[:, ii].copy()
plt.figure(figsize=(4, 4))
cm = dl.plot(phi, cmap='binary')
plt.axis('off')

tau=3.0
plot_ellipse(mu_p, Sigma_p, n_std_tau=tau, facecolor='none', edgecolor='k', linewidth=1)

xwidth = (tau*1.1)*np.sqrt(Sigma_p[0,0])
ywidth = (tau*1.1)*np.sqrt(Sigma_p[1,1])

plt.xlim([mu_p[0] - xwidth, mu_p[0] + xwidth])
plt.ylim([mu_p[1] - ywidth, mu_p[1] + ywidth])

plt.savefig('frog_one_ellipsoid.png', bbox_inches='tight', dpi=400) if save_figures else None

#### Plot moments

plt.figure(figsize=(4,4))
dl.plot(psf_object.vol(), cmap='binary', clim=[0.0, None])
# plt.colorbar(cm)
# plt.title(r'$V$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_vol.png', bbox_inches='tight', dpi=400) if save_figures else None

plt.figure(figsize=(4,4))
cm = dl.plot(psf_object.mu(0), cmap='binary')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_mu0.png', bbox_inches='tight', dpi=400) if save_figures else None

plt.figure(figsize=(4,4))
dl.plot(psf_object.mu(1), cmap='binary')
# plt.title(r'$\mu^2$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_mu1.png', bbox_inches='tight', dpi=400) if save_figures else None

plt.figure(figsize=(4,4))
dl.plot(psf_object.Sigma(0,0), cmap='binary')
# plt.title(r'$\Sigma^{11}$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_Sigma00.png', bbox_inches='tight', dpi=400) if save_figures else None

plt.figure(figsize=(4,4))
dl.plot(psf_object.Sigma(0,1), cmap='binary')
# plt.title(r'$\Sigma^{12}$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_Sigma01.png', bbox_inches='tight', dpi=400) if save_figures else None

plt.figure(figsize=(4,4))
dl.plot(psf_object.Sigma(1,1), cmap='binary')
# plt.title(r'$\Sigma^{22}$')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_Sigma11.png', bbox_inches='tight', dpi=400) if save_figures else None

#### Plot ellipsoid batch without impulses

bb = [0,1,2]

for b in bb:
    IRB = psf_object.psf_object.impulse_response_batches
    fig = plt.figure(figsize=(4,4))

    phi = psf_object.impulse_response_batch(b)

    start = IRB.batch2point_start[b]
    stop = IRB.batch2point_stop[b]
    pp = IRB.sample_points[start:stop, :]
    mu_batch = IRB.sample_mu[start:stop, :]
    Sigma_batch = IRB.sample_Sigma[start:stop, :, :]

    for k in range(mu_batch.shape[0]):
        plot_ellipse(mu_batch[k, :], Sigma_batch[k, :, :], n_std_tau=IRB.tau,
                     facecolor='none', edgecolor='k', linewidth=1)

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.savefig('frog_ellipsoid_batch' + str(b) + '.png', bbox_inches='tight', dpi=400) if save_figures else None

#### Plot ellipsoid batch with impulses

for b in bb:
    IRB = psf_object.psf_object.impulse_response_batches
    fig = plt.figure(figsize=(4,4))

    phi = psf_object.impulse_response_batch(b)

    start = IRB.batch2point_start[b]
    stop = IRB.batch2point_stop[b]
    pp = IRB.sample_points[start:stop, :]
    mu_batch = IRB.sample_mu[start:stop, :]
    Sigma_batch = IRB.sample_Sigma[start:stop, :, :]

    cm = dl.plot(phi, cmap='binary')

    for k in range(mu_batch.shape[0]):
        plot_ellipse(mu_batch[k, :], Sigma_batch[k, :, :], n_std_tau=IRB.tau,
                     facecolor='none', edgecolor='k', linewidth=1)

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.savefig('frog_impulse_batch' + str(b) + '.png', bbox_inches='tight', dpi=400) if save_figures else None


#### Compute error as a function of number of batches

raise RuntimeError

nx = 49
length_scaling = 1.0 #0.0625
a = 1.0
Ker, phi_function, Vh, H, mass_lumps, dof_coords = frog_setup(nx, length_scaling, a)

num_neighbors = 10

psf_object = lpsf1.make_psf_fenics(
    lambda X: H @ X,
    lambda X: H.T @ X,
    Vh, Vh,
    mass_lumps, mass_lumps,
    num_initial_batches=20,#0,
    tau = 3.0, display=True,
    num_neighbors = num_neighbors
)

Ker_psf = psf_object.psf_object.psf_kernel.cpp_object.eval_integral_kernel_block(dof_coords.T, dof_coords.T)
err_psf = np.linalg.norm(Ker_psf - Ker) / np.linalg.norm(Ker)
print('err_psf=', err_psf)

p0 =  np.array([0.6, 0.42])
ii = nearest_ind_func(dof_coords, p0)
p = dof_coords[ii,:]

mu_p = np.array([psf_object.mu(0)(p), psf_object.mu(1)(p)])

q0 =  np.array([0.56, 0.54])
jj = nearest_ind_func(dof_coords, q0)
q = dof_coords[jj,:]

sp = psf_object.psf_object.impulse_response_batches.sample_points

nearest_inds = np.argsort(np.linalg.norm(sp - p.reshape((1,-1)), axis=1))[:num_neighbors]

ker_err_func = dl.Function(Vh)
ker_err_vec = np.linalg.norm(Ker_psf - Ker, axis=0) / np.linalg.norm(Ker, axis=0)
# ker_err_vec = np.linalg.norm(Ker_psf - Ker, axis=0) / (np.linalg.norm(Ker) / np.sqrt(Ker.shape[1]))
ker_err_vec[np.linalg.norm(Ker, axis=0) == 0] = 0.0
ker_err_func.vector()[:] = ker_err_vec
# ker_err_func.vector()[:] = ker_err_vec > 0.03
# cm = dl.plot(ker_err_func, cmap='binary')
# plt.colorbar(cm)

p2 = (0.95)*p + (1.0 - 0.95)*q
q2 = (1.0 - 0.96)*p + (0.96)*q

plt.figure()
plt.plot(sp[:,0], sp[:,1], '.', c='gray', markersize=3)
plt.plot([p[0], q[0]], [p[1], q[1]], 'ok', fillstyle='none', markersize=6)
plt.annotate('', xy=(q2[0],q2[1]), xytext=(p2[0],p2[1]), arrowprops=dict(arrowstyle='->'), c='r')
plt.plot(sp[nearest_inds,0], sp[nearest_inds,1], '.k')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().set_aspect('equal')
plt.savefig('frog_nearest_neighbors.png', dpi=300, bbox_inches='tight')

for k in range(num_neighbors):
    sp_k = sp[nearest_inds[k], :]
    ii = nearest_ind_func(dof_coords, sp_k)
    sample_phi = dl.Function(Vh)
    sample_phi.vector()[:] = Ker[:,ii].copy()
    plt.figure()
    cm = dl.plot(sample_phi, cmap='binary')
    plt.axis('off')

    mu_spk = np.array([psf_object.mu(0)(sp_k), psf_object.mu(1)(sp_k)])
    Sigma_spk = np.array([[psf_object.Sigma(0, 0)(sp_k), psf_object.Sigma(0, 1)(sp_k)],
                          [psf_object.Sigma(1, 0)(sp_k), psf_object.Sigma(1, 1)(sp_k)]])

    tau = 3.0

    xwidth = (tau * 1.1) * np.sqrt(Sigma_spk[0, 0])
    ywidth = (tau * 1.1) * np.sqrt(Sigma_spk[1, 1])

    plt.xlim([mu_spk[0] - xwidth, mu_spk[0] + xwidth])
    plt.ylim([mu_spk[1] - ywidth, mu_spk[1] + ywidth])

    a = mu_spk
    delta = q - mu_p
    b = mu_spk + delta

    a2 = (0.97) * a + (1.0 - 0.97) * b
    b2 = (1.0 - 0.98) * a + (0.98) * b

    plt.plot([a[0], b[0]], [a[1], b[1]], 'ok', markerfacecolor='white', markersize=6)
    plt.annotate('', xy=(b2[0], b2[1]), xytext=(a2[0], a2[1]), arrowprops=dict(arrowstyle='->'))

    coords_string = str(sp_k[0]) + '_' + str(sp_k[1])
    plt.savefig('frog_neighbor_impulse_' + coords_string + '_.png', dpi=300, bbox_inches='tight')

#

print('Making row and column cluster trees')
ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=32)

print('Making block cluster trees')
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

bct.visualize('frog_bct.eps')


raise RuntimeError

#

plt.figure(figsize=(8,3))

phi_true = dl.Function(Vh)
phi_true.vector()[:] = Ker[:,ii].copy()
plt.subplot(1,2,1)
cm = dl.plot(phi_true, cmap='binary')
plt.colorbar(cm)
plt.title('phi')

e_func = dl.Function(Vh)
e_func.vector()[:] = (Ker_psf[:,ii] - Ker[:,ii]).copy()
plt.subplot(1,2,2)
cm = dl.plot(e_func, cmap='binary')
plt.colorbar(cm)
plt.title('phi error')




raise RuntimeError

phi_func = dl.Function(Vh)
phi_func.vector()[:] = Ker[:,ii].copy()
plt.figure()
cm = dl.plot(phi_func, cmap='binary')
plt.colorbar(cm)
plt.title('phi')

phi_psf_func = dl.Function(Vh)
phi_psf_func.vector()[:] = Ker_psf[:,ii].copy()
plt.figure()
cm = dl.plot(phi_psf_func, cmap='binary')
plt.colorbar(cm)
plt.title('phi psf')


raise RuntimeError

all_num_batches = []
all_num_impulses = []
all_psf_errs = []
for k in tqdm(range(Ker.shape[1])):
    if psf_object.psf_object.impulse_response_batches.num_sample_points >= Ker.shape[1]:
        break
    psf_object.add_impulse_response_batch()
    num_batches = psf_object.num_batches
    num_impulses = psf_object.psf_object.impulse_response_batches.num_sample_points
    Ker_psf = psf_object.psf_object.psf_kernel.cpp_object.eval_integral_kernel_block(dof_coords.T, dof_coords.T)
    err_psf = np.linalg.norm(Ker_psf - Ker) / np.linalg.norm(Ker)
    all_num_batches.append(num_batches)
    all_num_impulses.append(num_impulses)
    all_psf_errs.append(err_psf)
    print('num_batches=', num_batches, ', num_impulses=', num_impulses, ', err_psf=', err_psf)

plt.semilogy(all_num_impulses, all_psf_errs)

plt.figure()
ii = nearest_ind_func(dof_coords, np.array([0.5, 0.5]))
e_func = dl.Function(Vh)
e_func.vector()[:] = Ker_psf[:,ii] - Ker[:,ii]
cm = dl.plot(e_func, cmap='binary')
plt.colorbar(cm)

U,ss,Vt = np.linalg.svd(Ker)

for r in range(1, len(ss)+1):
err_glr = np.linalg.norm(U[:,:r] @ np.diag(ss[:r]) @ Vt[:r,:] - Ker) / np.linalg.norm(Ker)
print('err_glr=', err_glr)

U_off, ss_off, Vt_off = np.linalg.svd(Ker[:2048,2048:])