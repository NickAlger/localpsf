import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import localpsf.localpsf_cg1_lumped as lpsf1
import scipy.linalg as sla

save_figures = False

nx = 63
ny = 63
mesh = dl.UnitSquareMesh(nx, ny)
Vh = dl.FunctionSpace(mesh, 'CG', 1)
dof_coords = Vh.tabulate_dof_coordinates()

# Ricker-type wavelet, https://en.wikipedia.org/wiki/Ricker_wavelet
def ricker(
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

    rotation_center = np.array([0.5, 0.5])
    radial_vector = (mu - rotation_center)
    theta = 4.0 * np.arctan(radial_vector[1] / radial_vector[0])
    # xprime = (mu - rotation_center) / np.linalg.norm(mu - rotation_center)
    # yprime = np.array([xprime[1], -xprime[0]])
    # Rot_matrix = np.vstack([xprime, yprime]).T
    # theta = (np.pi / 2.0) * mu[0] * mu[1]
    Rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    pp = (xx - mu.reshape((1,2))) @ Rot_matrix.T #@ Rot_matrix.T
    inv_Sigma = np.linalg.inv(Sigma)
    t_squared_over_sigma_squared = np.einsum('ai,ai->a', pp, np.einsum('ij,aj->ai', inv_Sigma, pp))
    G = vol / (2.0 * np.pi * np.sqrt(np.linalg.det(Sigma))) * np.exp(-0.5 * t_squared_over_sigma_squared)

    cos_x_over_sigma = np.cos(pp[:,0] / (np.sqrt(Sigma[0,0]) / 2.0))
    sin_y_over_sigma = np.sin(pp[:,1] / (np.sqrt(Sigma[1,1]) / 2.0))
    return (1.0 + a * cos_x_over_sigma * sin_y_over_sigma) * G


length_scaling = 1.0 # <--- Tucker: experiment with this
vol = 1.0
mu = np.array([0.5, 0.5])
Sigma = length_scaling * np.array([[0.01, 0.0], [0.0, 0.0025]]) # 0.25*np.array([[0.01, 0.0], [0.0, 0.0025]])
a = 1.0

rr = ricker(dof_coords, vol, mu, Sigma, a)

f1 = dl.Function(Vh)
f1.vector()[:] = rr

num_pts_1d = 500
coords_1d = np.array([mu[0]*np.ones(num_pts_1d), np.linspace(0, 1, num_pts_1d)]).T # vertical line

plt.figure()
cm = dl.plot(f1)
plt.colorbar(cm)
plt.title('f1')
plt.plot(coords_1d[:,0], coords_1d[:,1], '.r')

rr_1d = ricker(coords_1d, vol, mu, Sigma, a)

plt.figure()
plt.plot(rr_1d)
plt.title('rr_1d')

Ker = np.zeros((Vh.dim(), Vh.dim()))
for ii in range(Vh.dim()):
    Ker[:,ii] = ricker(dof_coords, vol, dof_coords[ii,:], Sigma, a)


# p1 = np.array([0.2, 0.55])
# p2 = np.array([0.5, 0.65])
# pp = [p1, p2]
pp = [
    np.array([0.2, 0.2]),
    np.array([0.2, 0.4]),
    np.array([0.2, 0.6]),
    np.array([0.2, 0.8]),
    np.array([0.4, 0.2]),
    np.array([0.4, 0.4]),
    np.array([0.4, 0.6]),
    np.array([0.4, 0.8]),
    np.array([0.6, 0.2]),
    np.array([0.6, 0.4]),
    np.array([0.6, 0.6]),
    np.array([0.6, 0.8]),
    np.array([0.8, 0.2]),
    np.array([0.8, 0.4]),
    np.array([0.8, 0.6]),
    np.array([0.8, 0.8]),
]

plt.matshow(Ker, cmap='binary')
plt.ylim([Ker.shape[0], 0])
plt.gca().set_xticks([])
plt.gca().set_yticks([])

for k in range(len(pp)):
    p = pp[k]
    ii = np.argmin(np.linalg.norm(dof_coords - p.reshape((1, -1)), axis=1))
    plt.plot([ii, ii], [0, Ker.shape[0]], 'k', linestyle='dotted')  if save_figures else None

plt.savefig('frog_kernel_matrix_a1.png', bbox_inches='tight', dpi=300)

for k in range(len(pp)):
    p = pp[k]
    ii = np.argmin(np.linalg.norm(dof_coords - p.reshape((1, -1)), axis=1))
    phi = dl.Function(Vh)
    phi.vector()[:] = Ker[:, ii].copy()
    plt.figure()
    cm = dl.plot(phi, cmap='binary')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig('frog_impulse_response' + str(k) + '.png', bbox_inches='tight', dpi=150) if save_figures else None


raise RuntimeError
#####
phi = dl.Function(Vh)
phi.vector()[:] = Ker[:,ii].copy()
plt.figure()
cm = dl.plot(phi, cmap='binary')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_impulse_response' + str(k) +'.png', bbox_inches='tight', dpi=150)

phi = dl.Function(Vh)
phi.vector()[:] = Ker[:,ii2].copy()
plt.figure()
cm = dl.plot(phi, cmap='binary')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig('frog_impulse_response2.png', bbox_inches='tight', dpi=150)


mass_lumps = dl.assemble(dl.Constant(1.0) * dl.TestFunction(Vh) * dl.dx)[:].copy()

# H = mass_lumps.reshape((-1, 1)) * Ker_small * mass_lumps.reshape((1, -1))
H = mass_lumps.reshape((-1, 1)) * Ker * mass_lumps.reshape((1, -1))
# H = mass_lumps.reshape((-1, 1)) * Ker_medium * mass_lumps.reshape((1, -1))

apply_H = lambda X: H @ X
apply_Ht = lambda X: H.T @ X

verts, cells = lpsf1.mesh_vertices_and_cells_in_CG1_dof_order(Vh)
Vh_CG1 = lpsf1.CG1Space(verts, cells, mass_lumps)

psf_object = lpsf1.make_psf_fenics(
    apply_H, apply_Ht,
    Vh, Vh,
    mass_lumps, mass_lumps,
    tau = 3.0, display=True,
    # smoothing_width_in=0.0,
    # smoothing_width_out=0.0
)

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