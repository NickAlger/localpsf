import numpy as np
import matplotlib.pyplot as plt
from nalger_helper_functions import plot_ellipse

Hv1 = np.load("imp_res_shallow.npy")
Hv2 = np.load("imp_res_deep.npy")
# phi = Hv1.T / np.max(Hv1) + Hv2.T / np.max(Hv2)
phi1 = Hv1.T / np.max(Hv1.T)
phi2 = Hv2.T / np.max(Hv2.T)

plt.figure()
# plt.imshow(phi, cmap='binary', vmax=np.max(phi)/5.0)
plt.imshow(phi2, cmap='binary', vmax=np.max(phi2))
plt.xlim(200 - 60, 200 + 60)
plt.colorbar()

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.savefig('marmousi_impulse_response1.png', dpi=300, bbox_inches='tight')


raise RuntimeError

# plt.figure()
# # plt.imshow(phi, cmap='binary', vmax=np.max(phi)/5.0)
# plt.imshow(phi2, cmap='binary', vmax=np.max(phi2))
# plt.xlim(200 - 60, 200 + 60)
# plt.colorbar()

#

phi = np.abs(phi)

C = np.ones(phi.shape)
Lx = np.outer(np.ones(  phi.shape[0]),  np.arange(phi.shape[1]))
Ly = np.outer(np.arange(phi.shape[0]),  np.ones(  phi.shape[1]))
Qxx = Lx * Lx
Qxy = Lx * Ly
Qyy = Ly * Ly

# plt.imshow(Ly, cmap='binary')
# plt.colorbar()

vol = np.sum(phi * C)
mu_x = np.sum(phi * Lx / vol)
mu_y = np.sum(phi * Ly / vol)
Sigma_xx = np.sum(phi * Qxx / vol) - mu_x*mu_x
Sigma_xy = np.sum(phi * Qxy / vol) - mu_x*mu_y
Sigma_yy = np.sum(phi * Qyy / vol) - mu_y*mu_y

mu = np.array([mu_x, mu_y])
Sigma = np.array([[Sigma_xx, Sigma_xy], [Sigma_xy, Sigma_yy]])

plt.plot(mu_x, mu_y, '.r')
plot_ellipse(mu, Sigma, 1.0)

np.sqrt(np.linalg.eigh(Sigma)[0])