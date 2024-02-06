import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dolfin as dl
import localpsf.localpsf_cg1_lumped as lpsf1
import scipy.linalg as sla
from nalger_helper_functions import plot_ellipse

save_figures = True

nx = 89
ny = 89
mesh = dl.UnitSquareMesh(nx, ny)
Vh = dl.FunctionSpace(mesh, 'CG', 1)
dof_coords = Vh.tabulate_dof_coordinates()

# Frog function
def frog_function(
    xx: np.ndarray, # shape=(N, 2), N=nx*ny
    vol: float,
    mu: np.ndarray, # shape=(2,)
    Sigma: np.ndarray, # shape=(2, 2)
    a: float # how much negative to include. a=0: gaussian
) -> np.ndarray: # shape=(N,)
    assert(xx.shape[1] == 2)
    N = xx.shape[0]
    assert(Sigma.shape == (2, 2))
    assert(mu.shape == (2,))

    theta = np.pi * (mu[0] + mu[1])
    Rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    pp = (xx - mu.reshape((1,2))) @ Rot_matrix.T #@ Rot_matrix.T
    inv_Sigma = np.linalg.inv(Sigma)
    t_squared_over_sigma_squared = np.einsum('ai,ai->a', pp, np.einsum('ij,aj->ai', inv_Sigma, pp))
    G = vol / (2.0 * np.pi * np.sqrt(np.linalg.det(Sigma))) * np.exp(-0.5 * t_squared_over_sigma_squared)

    # V = 2.0 + np.cos(4.0 * np.pi * np.sqrt((mu[0] - 0.5)**2 + (mu[1]-0.5)**2))
    #
    # cos_x_over_sigma = np.cos(pp[:,0] / (np.sqrt(Sigma[0,0]) / 2.0))
    # sin_y_over_sigma = np.sin(pp[:,1] / (np.sqrt(Sigma[1,1]) / 2.0))
    # return V * (1.0 + a * cos_x_over_sigma * sin_y_over_sigma) * G

    cos_x_over_sigma = np.cos(pp[:,0] / (np.sqrt(Sigma[0,0]) / 2.0))
    sin_y_over_sigma = np.sin(pp[:,1] / (np.sqrt(Sigma[1,1]) / 2.0))
    return (1.0 + a * cos_x_over_sigma * sin_y_over_sigma) * G


length_scaling = 1.0 # <--- Tucker: experiment with this
vol = 1.0
mu = np.array([0.5, 0.5])
Sigma = length_scaling * np.array([[0.01, 0.0], [0.0, 0.0025]]) # 0.25*np.array([[0.01, 0.0], [0.0, 0.0025]])
a = 1.0

rr = frog_function(dof_coords, vol, mu, Sigma, a)

f1 = dl.Function(Vh)
f1.vector()[:] = rr

num_pts_1d = 500
coords_1d = np.array([mu[0]*np.ones(num_pts_1d), np.linspace(0, 1, num_pts_1d)]).T # vertical line

plt.figure()
cm = dl.plot(f1)
plt.colorbar(cm)
plt.title('f1')
plt.plot(coords_1d[:,0], coords_1d[:,1], '.r')

rr_1d = frog_function(coords_1d, vol, mu, Sigma, a)

plt.figure()
plt.plot(rr_1d)
plt.title('rr_1d')

Ker = np.zeros((Vh.dim(), Vh.dim()))
for ii in range(Vh.dim()):
    Ker[:,ii] = frog_function(dof_coords, vol, dof_coords[ii,:], Sigma, a)


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
    ii = np.argmin(np.linalg.norm(dof_coords - p.reshape((1, -1)), axis=1))
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.subplot(gs1[k])
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

plt.savefig('frog_impulse_response_grid.png', bbox_inches='tight', dpi=300) if save_figures else None


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
    ii = np.argmin(np.linalg.norm(dof_coords - p.reshape((1, -1)), axis=1))
    print('k=', k, ', ii=', ii)
    plt.plot([ii, ii], [0, Ker.shape[0]], 'k', linestyle='dotted', linewidth=2.0)

plt.gcf().set_size_inches(8, 8)
plt.savefig('frog_kernel_matrix_a1.png', bbox_inches='tight', dpi=400) if save_figures else None

for k in range(len(pp)):
    p = pp[k]
    ii = np.argmin(np.linalg.norm(dof_coords - p.reshape((1, -1)), axis=1))
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.figure(figsize=(3.8, 3.8))
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig('frog_impulse_response' + str(k) + '.png', bbox_inches='tight', dpi=400) if save_figures else None


####

mass_lumps = dl.assemble(dl.Constant(1.0) * dl.TestFunction(Vh) * dl.dx)[:].copy()

H = mass_lumps.reshape((-1, 1)) * Ker * mass_lumps.reshape((1, -1))

apply_H = lambda X: H @ X
apply_Ht = lambda X: H.T @ X

verts, cells = lpsf1.mesh_vertices_and_cells_in_CG1_dof_order(Vh)
Vh_CG1 = lpsf1.CG1Space(verts, cells, mass_lumps)

psf_object = lpsf1.make_psf_fenics(
    apply_H, apply_Ht,
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

# plt.title('Impulse response batch ' + str(b))

# cm = dl.plot(phi)



# plt.title('Impulse response batch ' + str(b))

# cm = dl.plot(phi)

raise RuntimeError

# plt.savefig('frog_impulse_response' + str(k) + '.png', bbox_inches='tight', dpi=400) if save_figures else None

IRB = me.psf_object.impulse_response_batches
fig = plt.figure()

phi = me.impulse_response_batch(b)

start = IRB.batch2point_start[b]
stop = IRB.batch2point_stop[b]
pp = IRB.sample_points[start:stop, :]
mu_batch = IRB.sample_mu[start:stop, :]
Sigma_batch = IRB.sample_Sigma[start:stop, :, :]

cm = dl.plot(phi)
plt.colorbar(cm)

plt.scatter(pp[:, 0], pp[:, 1], c='r', s=2)
plt.scatter(mu_batch[:, 0], mu_batch[:, 1], c='k', s=2)

for k in range(mu_batch.shape[0]):
    plot_ellipse(mu_batch[k, :], Sigma_batch[k, :, :], n_std_tau=IRB.tau,
                 facecolor='none', edgecolor='k', linewidth=1)

plt.title('Impulse response batch ' + str(b))

####

psf_object.visualize_impulse_response_batch(0)

plt.figure()
cm = dl.plot(psf_object.vol())
plt.colorbar(cm)
plt.title('vol')

plt.figure()
cm = dl.plot(psf_object.mu(0))
plt.colorbar(cm)
plt.title('mu(0)')

plt.figure()
cm = dl.plot(psf_object.mu(1))
plt.colorbar(cm)
plt.title('mu(1)')

plt.figure()
cm = dl.plot(psf_object.Sigma(0,0))
plt.colorbar(cm)
plt.title('Sigma(0,0)')

plt.figure()
cm = dl.plot(psf_object.Sigma(0,1))
plt.colorbar(cm)
plt.title('Sigma(0,1)')

plt.figure()
cm = dl.plot(psf_object.Sigma(1,1))
plt.colorbar(cm)
plt.title('Sigma(1,1)')

####

# Frog example