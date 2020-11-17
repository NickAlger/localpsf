# import sys
# sys.path.append('/Users/gaol/study/pybind/pybind11/debug')
# sys.path.append('/Users/gaol/study/pybind/hlibpro/hlibpro-2.8.1/lib')

import numpy as np
import user
user.bem1d(100)

min_pt = np.array([-1.2, 0.5])
max_pt = np.array([2.1, 3.3])
deltas = max_pt - min_pt
nx = 5
ny = 4
VG = np.random.randn(nx, ny)

cc = min_pt.reshape((1,-1)) + np.random.rand(1370,2) * deltas.reshape((1,-1))

# import matplotlib.pyplot as plt
# plt.plot(cc[:,0], cc[:,1], '.')

ve = user.grid_interpolate(cc, min_pt[0], max_pt[0], min_pt[1], max_pt[1], VG)
print(ve)