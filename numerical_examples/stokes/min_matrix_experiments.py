import numpy as np

N = 10
r = 3

X = np.random.randn(N,r)
B = np.random.randn(N,r)

a = 1e-6
print('a=', a)
A = (1./a) * B @ X.T @ (np.eye(N) - X @ np.linalg.inv(a*np.eye(r) + X.T @ X) @ X.T)

rel_res = np.linalg.norm(A @ X - B) / np.linalg.norm(B)
print('rel_res=', rel_res)

norm_A = np.linalg.norm(A)
print('norm_A=', norm_A)