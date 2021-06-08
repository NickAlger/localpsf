import numpy as np
from scipy.interpolate import RegularGridInterpolator
from time import time
from code import user

user.bem1d(100)

min_pt = np.array([-1.2, 0.5])
max_pt = np.array([2.1, 3.3])
deltas = max_pt - min_pt
nx = 50
ny = 43
VG = np.random.randn(nx, ny)

cc = min_pt.reshape((1,-1)) + np.random.rand(62683,2) * deltas.reshape((1,-1))

# plt.plot(cc[:,0], cc[:,1], '.')

t = time()
ve = user.grid_interpolate(cc, min_pt[0], max_pt[0], min_pt[1], max_pt[1], VG)
dt_cpp = time() - t
print('dt_cpp=', dt_cpp)
print(ve)

t = time()
ve1 = user.grid_interpolate_vectorized(cc, min_pt[0], max_pt[0], min_pt[1], max_pt[1], VG)
dt_cpp_vectorized = time() - t
print('dt_cpp_vectorized=', dt_cpp_vectorized)
print(ve1)

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

####
xmin = -1
xmax = 1.2
ymin = -1.5
ymax = 1.4
xx = np.linspace(xmin, xmax, 85)
yy = np.linspace(ymin, ymax, 97)
X, Y = np.meshgrid(xx, yy, indexing='xy')
V = np.exp(-(X**2 + Y**2)/0.05)

# plt.matshow(V)
xx2 = np.linspace(xmin-0.1, xmax+0.1, 35)
yy2 = np.linspace(ymin-0.1, ymax-0.1, 47)
X2, Y2 = np.meshgrid(xx2, yy2, indexing='xy')

dof_coords = np.array([X2.reshape(-1), Y2.reshape(-1)]).T
# dof_coords = np.random.randn(1000,2)

rhs_b = np.random.randn(dof_coords.shape[0])

user.Custom_bem1d(dof_coords, xmin, xmax, ymin, ymax, V, rhs_b)

#####

