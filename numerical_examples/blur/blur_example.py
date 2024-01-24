import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import localpsf.localpsf_cg1_lumped as lpsf1


nx = 50
ny = 50
mesh = dl.UnitSquareMesh(nx, ny)
Vh = dl.FunctionSpace(mesh, 'CG', 1)
dof_coords = Vh.tabulate_dof_coordinates()
#
# xx_linear = np.linspace(-1.0, 1.0, nx)
# yy_linear = np.linspace(-1.0, 1.0, ny)
#
# X, Y = np.meshgrid(xx_linear, yy_linear)
# N = len(X.reshape(-1))
#
# plt.figure()
# plt.scatter(X.reshape(-1), Y.reshape(-1))

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

    return (1.0 - a * t_squared_over_sigma_squared) * G

# xx = np.vstack([X.reshape(-1), Y.reshape(-1)]).T

vol = 1.0
mu = np.array([0.5, 0.5])
Sigma = np.array([[0.01, 0.0], [0.0, 0.0025]])
a = 0.3

rr = ricker(dof_coords, vol, mu, Sigma, a)

f1 = dl.Function(Vh)
f1.vector()[:] = rr
cm = dl.plot(f1)
plt.colorbar(cm)


Ker = np.zeros((Vh.dim(), Vh.dim()))
for ii in range(Vh.dim()):
    Ker[:,ii] = ricker(dof_coords, vol, dof_coords[ii,:], Sigma, a)

plt.matshow(Ker)
plt.title('true kernel')

ii=953
phi = dl.Function(Vh)
phi.vector()[:] = Ker[:,ii].copy()
plt.figure()
cm = dl.plot(phi)
plt.colorbar(cm)
plt.title('phi')
plt.plot(dof_coords[ii,0], dof_coords[ii,1], '.r')

mass_lumps = dl.assemble(dl.Constant(1.0) * dl.TestFunction(Vh) * dl.dx)[:].copy()

H = mass_lumps.reshape((-1, 1)) * Ker * mass_lumps.reshape((1, -1))

# HC = dl.Function(Vh)
# HC.vector()[:] = (H.T @ np.ones(Vh.dim()) / mass_lumps).copy()
# HC_abs = dl.Function(Vh)
# HC_abs.vector()[:] = (np.abs(H).T @ np.ones(Vh.dim()) / mass_lumps).copy()

# cm = dl.plot(HC)
# plt.colorbar(cm)
# plt.title('HC')

# plt.figure()
# cm = dl.plot(HC_abs)
# plt.colorbar(cm)
# plt.title('HC_abs')

apply_H = lambda X: H @ X
apply_Ht = lambda X: H.T @ X

verts, cells = lpsf1.mesh_vertices_and_cells_in_CG1_dof_order(Vh)
Vh_CG1 = lpsf1.CG1Space(verts, cells, mass_lumps)

psf_object = lpsf1.make_psf_fenics(
    apply_H, apply_Ht,
    Vh, Vh,
    mass_lumps, mass_lumps,
    tau = 3.0, display=True,
    #smoothing_width_in=1.0,
    #smoothing_width_out=1.0
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