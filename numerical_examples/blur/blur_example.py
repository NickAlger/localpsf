import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dolfin as dl
import localpsf.localpsf_cg1_lumped as lpsf1
import scipy.linalg as sla
from functools import partial
from nalger_helper_functions import plot_ellipse
from tqdm.auto import tqdm

save_figures = True

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

    bump = mu[0]*(1 - mu[0])*mu[1]*(1-mu[1])

    pp = (xx - mu.reshape((1,2))) @ Rot_matrix.T #@ Rot_matrix.T
    inv_Sigma = np.linalg.inv(Sigma)
    t_squared_over_sigma_squared = np.einsum('ai,ai->a', pp, np.einsum('ij,aj->ai', inv_Sigma, pp))
    G = vol / (2.0 * np.pi * np.sqrt(np.linalg.det(Sigma))) * np.exp(-0.5 * t_squared_over_sigma_squared)

    cos_x_over_sigma = np.cos(pp[:,0] / (np.sqrt(Sigma[0,0]) / 2.0))
    sin_y_over_sigma = np.sin(pp[:,1] / (np.sqrt(Sigma[1,1]) / 2.0))
    return bump * (1.0 + a * cos_x_over_sigma * sin_y_over_sigma) * G


def frog_setup(
        nx:             int     = 63,
        length_scaling: float   = 1.0,
        a:              float   = 1.0,
):
    ny = nx
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh = dl.FunctionSpace(mesh, 'CG', 1)
    dof_coords = Vh.tabulate_dof_coordinates()

    vol = 1.0

    Sigma = length_scaling * np.array([[0.01, 0.0], [0.0, 0.0025]]) # 0.25*np.array([[0.01, 0.0], [0.0, 0.0025]])

    phi_function = lambda yy, x: frog_function(yy, vol, x, Sigma, a)

    Ker = np.zeros((Vh.dim(), Vh.dim()))
    for ii in range(Vh.dim()):
        Ker[:,ii] = phi_function(dof_coords, dof_coords[ii,:])

    mass_lumps = dl.assemble(dl.Constant(1.0) * dl.TestFunction(Vh) * dl.dx)[:].copy()

    H = mass_lumps.reshape((-1, 1)) * Ker * mass_lumps.reshape((1, -1))

    return Ker, phi_function, Vh, H, mass_lumps, dof_coords

nearest_ind_func = lambda yy, x: np.argmin(np.linalg.norm(yy - x.reshape((1, -1)), axis=1))

nx = 63
length_scaling = 1.0
a = 1.0
Ker, phi_function, Vh, H, mass_lumps, dof_coords = frog_setup(nx, length_scaling, a)
dof_coords = Vh.tabulate_dof_coordinates()

p = np.array([0.5, 0.5])

frog1 = dl.Function(Vh)
frog1.vector()[:] = phi_function(dof_coords, p)

num_pts_1d = 500
coords_1d = np.array([p[0]*np.ones(num_pts_1d), np.linspace(0, 1, num_pts_1d)]).T # vertical line

plt.figure()
cm = dl.plot(frog1, cmap='binary')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.title('frog1')
plt.plot(coords_1d[:,0], coords_1d[:,1], '.r')

rr_1d = phi_function(coords_1d, p)

plt.figure()
plt.plot(rr_1d)
plt.title('rr_1d')


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
    ii = nearest_ind_func(dof_coords, p)
    print('k=', k, ', ii=', ii)
    plt.plot([ii, ii], [0, Ker.shape[0]], 'k', linestyle='dotted', linewidth=2.0)

plt.gcf().set_size_inches(8, 8)
plt.savefig('frog_kernel_matrix_a1.png', bbox_inches='tight', dpi=400) if save_figures else None

for k in range(len(pp)):
    p = pp[k]
    ii = nearest_ind_func(dof_coords, p)
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.figure(figsize=(3.8, 3.8))
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig('frog_impulse_response' + str(k) + '.png', bbox_inches='tight', dpi=400) if save_figures else None


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

nx = 89
length_scaling = 0.0625
a = 1.0
Ker, phi_function, Vh, H, mass_lumps, dof_coords = frog_setup(nx, length_scaling, a)

psf_object = lpsf1.make_psf_fenics(
    lambda X: H @ X,
    lambda X: H.T @ X,
    Vh, Vh,
    mass_lumps, mass_lumps,
    num_initial_batches=10,#0,
    tau = 3.0, display=True,
)

Ker_psf = psf_object.psf_object.psf_kernel.cpp_object.eval_integral_kernel_block(dof_coords.T, dof_coords.T)
err_psf = np.linalg.norm(Ker_psf - Ker) / np.linalg.norm(Ker)
print('err_psf=', err_psf)

ker_err_func = dl.Function(Vh)
ker_err_vec = np.linalg.norm(Ker_psf - Ker, axis=0) / np.linalg.norm(Ker, axis=0)
# ker_err_vec = np.linalg.norm(Ker_psf - Ker, axis=0) / (np.linalg.norm(Ker) / np.sqrt(Ker.shape[1]))
ker_err_vec[np.linalg.norm(Ker, axis=0) == 0] = 0.0
ker_err_func.vector()[:] = ker_err_vec
# ker_err_func.vector()[:] = ker_err_vec > 0.03
plt.figure()
cm = dl.plot(ker_err_func, cmap='binary')
plt.colorbar(cm)
sp = psf_object.psf_object.impulse_response_batches.sample_points
plt.plot(sp[:,0], sp[:,1], '.r')
plt.title('ker error')


raise RuntimeError

ii = nearest_ind_func(dof_coords, np.array([0.5, 0.2]))
e_func = dl.Function(Vh)
e_func.vector()[:] = (Ker_psf[:,ii] - Ker[:,ii]).copy()
plt.figure()
cm = dl.plot(e_func, cmap='binary')
plt.colorbar(cm)
plt.title('phi error')

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
