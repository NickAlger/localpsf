import numpy as np
import matplotlib.pyplot as plt

plt.figure()
Hv = np.load("imp_res_shallow.npy")
plt.imshow(Hv.T, cmap='binary')
plt.colorbar()

plt.figure()
Hv = np.load("imp_res_deep.npy")
# Hv = np.abs(Hv)
plt.imshow(Hv.T, cmap='binary')
plt.colorbar()

C = np.ones(Hv.T.shape)
Lx = np.outer(np.ones(Hv.shape[1]), np.arange(Hv.shape[0]))
Ly = np.outer(np.arange(Hv.shape[1]), np.ones(Hv.shape[0]))
Qxx = Lx * Lx
Qxy = Lx * Ly
Qyy = Ly * Ly

# plt.imshow(Ly, cmap='binary')
# plt.colorbar()

vol = np.sum(Hv.T * C)
mu_x = np.sum(Hv.T * Lx / vol)
mu_y = np.sum(Hv.T * Ly / vol)

plt.plot(mu_x, mu_y, '.r')