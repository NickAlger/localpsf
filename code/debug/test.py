# import sys
# sys.path.append('/Users/gaol/study/pybind/pybind11/debug')
# sys.path.append('/Users/gaol/study/pybind/hlibpro/hlibpro-2.8.1/lib')

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from time import time
import user
user.bem1d(100)

min_pt = np.array([-1.2, 0.5])
max_pt = np.array([2.1, 3.3])
deltas = max_pt - min_pt
nx = 50
ny = 43
VG = np.random.randn(nx, ny)

cc = min_pt.reshape((1,-1)) + np.random.rand(13709,2) * deltas.reshape((1,-1))

# import matplotlib.pyplot as plt
# plt.plot(cc[:,0], cc[:,1], '.')

t = time()
ve = user.grid_interpolate(cc, min_pt[0], max_pt[0], min_pt[1], max_pt[1], VG)
dt_cpp = time() - t
print('dt_cpp=', dt_cpp)
print(ve)

xx = np.linspace(min_pt[0], max_pt[0], nx)
yy = np.linspace(min_pt[1], max_pt[1], ny)
RGI = RegularGridInterpolator((xx, yy), VG, method='linear', bounds_error=True, fill_value=0.0)
ve2 = RGI(cc)
print(ve2)

t = time()
err_interp = np.linalg.norm(ve - ve2)
dt_scipy = time() - t
print('dt_scipy=', dt_scipy)
print('err_interp=', err_interp)