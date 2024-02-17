import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(0, 1, 10)
ff = np.cos(2 * np.pi * xx)
ff[int(len(ff)/2)] = 0.0

def phi(
    yy: np.ndarray, # shape=(N,)
    x: float,
    shape_parameter: float,
):
    return np.exp(-shape_parameter * (yy - x)**2)

N = len(xx)

shape_parameter = 1e3
B = np.zeros((N,N))
for ii in range(N):
    B[:,ii] = phi(xx, xx[ii], shape_parameter)

ww = np.linalg.solve(B, ff)

xx_test = np.linspace(0, 1, 500)
N_test = len(xx_test)
ff_test = np.zeros(N_test)
for ii in range(N_test):
    ff_test[ii] = np.sum(ww * phi(xx, xx_test[ii], shape_parameter))

ff_test_true = np.cos(2 * np.pi * xx_test)

plt.figure()
plt.plot(xx_test, ff_test_true)
plt.plot(xx_test, ff_test)
plt.scatter(xx, ff, c='r')
plt.title('shape_parameter=' + str(shape_parameter))