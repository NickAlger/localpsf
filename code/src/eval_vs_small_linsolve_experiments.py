import numpy as np
import dolfin as dl
from time import time

mesh = dl.UnitSquareMesh(71,85)
bbt = mesh.bounding_box_tree()

V = dl.FunctionSpace(mesh, 'CG', 1)
u = dl.Function(V)

n=100000
m=30
u.set_allow_extrapolation(True)
pts = [dl.Point(np.random.randn(), np.random.randn()) for _ in range(n)]
t = time()
uu = [u(pt) for pt in pts]
dt_eval = time() - t
t = time()
ents = [bbt.compute_first_entity_collision(pt) for pt in pts]
dt_collision = time() - t
MM = [np.random.randn(m,m) for _ in range(n)]
bb = [np.random.randn(m) for _ in range(n)]
t = time()
iMMb = [np.linalg.solve(M,b) for M, b in zip(MM, bb)]
dt_solve = time() - t
t = time()
MMb = [np.dot(M,b) for M, b in zip(MM, bb)]
dt_dot = time() - t
print('n=', n, ', dt_eval=', dt_eval, ', dt_collision=', dt_collision, ', dt_solve=', dt_solve, ', dt_dot=', dt_dot)