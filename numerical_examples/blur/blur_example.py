import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import localpsf.localpsf_cg1_lumped as lpsf1
import scipy.linalg as sla


nx = 50
ny = 50
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

    pp = xx - mu.reshape((1,2))
    inv_Sigma = np.linalg.inv(Sigma)
    t_squared_over_sigma_squared = np.einsum('ai,ai->a', pp, np.einsum('ij,aj->ai', inv_Sigma, pp))
    G = vol / (2.0 * np.pi * np.sqrt(np.linalg.det(Sigma))) * np.exp(-0.5 * t_squared_over_sigma_squared)

    # cos_t_over_sigma = np.cos(np.sqrt(t_squared_over_sigma_squared))
    cos_x_over_sigma = np.cos(pp[:,0] / (np.sqrt(Sigma[0,0]) / 2.0))
    sin_y_over_sigma = np.sin(pp[:,1] / (np.sqrt(Sigma[1,1]) / 2.0))
    return (1.0 + a * cos_x_over_sigma * sin_y_over_sigma) * G
    # return (1.0 + a * cos_x_over_sigma) * G
    # return cos_x_over_sigma * G
    # return cos_t_over_sigma * G
    # return (1.0 + a * cos_t_over_sigma) * G
    # return (1.0 - a * t_squared_over_sigma_squared) * G

# xx = np.vstack([X.reshape(-1), Y.reshape(-1)]).T

length_scaling = 0.25 # <--- Tucker: experiment with this
vol = 1.0
mu = np.array([0.5, 0.5])
Sigma = length_scaling * np.array([[0.01, 0.0], [0.0, 0.0025]]) # 0.25*np.array([[0.01, 0.0], [0.0, 0.0025]])
a_small = 0.0
a_medium = 1.1 # 0.1
a_large = 5.0 # <--- Tucker: use a=1

rr_small = ricker(dof_coords, vol, mu, Sigma, a_small)
rr_medium = ricker(dof_coords, vol, mu, Sigma, a_medium)
rr_large = ricker(dof_coords, vol, mu, Sigma, a_large)


f1_small = dl.Function(Vh)
f1_small.vector()[:] = rr_small

f1_medium = dl.Function(Vh)
f1_medium.vector()[:] = rr_medium

f1_large = dl.Function(Vh)
f1_large.vector()[:] = rr_large

plt.figure()
cm = dl.plot(f1_small)
plt.colorbar(cm)
plt.title('f1_small')

plt.figure()
cm = dl.plot(f1_medium)
plt.colorbar(cm)
plt.title('f1_medium')

plt.figure()
cm = dl.plot(f1_large)
plt.colorbar(cm)
plt.title('f1_large')

num_pts_1d = 500
# coords_1d = np.array([np.linspace(0, 1, num_pts_1d), mu[1]*np.ones(num_pts_1d)]).T # horizontal line
# coords_1d = np.array([np.linspace(0, 1, num_pts_1d), np.linspace(0, 1, num_pts_1d)]).T # 45 degree line
coords_1d = np.array([mu[0]*np.ones(num_pts_1d), np.linspace(0, 1, num_pts_1d)]).T # vertical line

plt.figure()
cm = dl.plot(f1_medium)
plt.colorbar(cm)
plt.title('f1_medium')
plt.plot(coords_1d[:,0], coords_1d[:,1], '.r')

rr_1d_small = ricker(coords_1d, vol, mu, Sigma, a_small)
rr_1d_medium = ricker(coords_1d, vol, mu, Sigma, a_medium)
rr_1d_large = ricker(coords_1d, vol, mu, Sigma, a_large)

plt.figure()
# plt.plot(coords_1d[:,0], rr_1d_small)
plt.plot(rr_1d_small)
plt.title('rr_1d_small')

plt.figure()
# plt.plot(coords_1d[:,0], rr_1d_medium)
plt.plot(rr_1d_medium)
plt.title('rr_1d_medium')

plt.figure()
# plt.plot(coords_1d[:,0], rr_1d_large)
plt.plot(rr_1d_large)
plt.title('rr_1d_large')

Ker_small = np.zeros((Vh.dim(), Vh.dim()))
for ii in range(Vh.dim()):
    Ker_small[:,ii] = ricker(dof_coords, vol, dof_coords[ii,:], Sigma, a_small)

Ker_medium = np.zeros((Vh.dim(), Vh.dim()))
for ii in range(Vh.dim()):
    Ker_medium[:,ii] = ricker(dof_coords, vol, dof_coords[ii,:], Sigma, a_medium)

Ker_large = np.zeros((Vh.dim(), Vh.dim()))
for ii in range(Vh.dim()):
    Ker_large[:,ii] = ricker(dof_coords, vol, dof_coords[ii,:], Sigma, a_large)

ii1=713
ii2=1504
plt.matshow(Ker_medium, cmap='binary')
# plt.matshow(np.sqrt(np.abs(Ker_medium)), cmap='binary')
plt.plot([ii1, ii1], [0, Ker_medium.shape[0]], 'r')
plt.plot([ii2, ii2], [0, Ker_medium.shape[0]], 'r')
plt.ylim([Ker_medium.shape[0], 0])
plt.title('true kernel medium')


phi = dl.Function(Vh)
phi.vector()[:] = Ker_medium[:,ii1].copy()
plt.figure()
cm = dl.plot(phi, cmap='binary')
# plt.colorbar(cm)
# plt.title('phi')
plt.plot(dof_coords[ii1,0], dof_coords[ii1,1], '.r')

phi = dl.Function(Vh)
phi.vector()[:] = Ker_medium[:,ii2].copy()
plt.figure()
cm = dl.plot(phi, cmap='binary')
# plt.colorbar(cm)
# plt.title('phi')
plt.plot(dof_coords[ii2,0], dof_coords[ii2,1], '.r')

mass_lumps = dl.assemble(dl.Constant(1.0) * dl.TestFunction(Vh) * dl.dx)[:].copy()

# H = mass_lumps.reshape((-1, 1)) * Ker_small * mass_lumps.reshape((1, -1))
H = mass_lumps.reshape((-1, 1)) * Ker_large * mass_lumps.reshape((1, -1))
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