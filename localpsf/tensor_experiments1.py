import numpy as np

N1 = 4
N2 = 5
N3 = 7
N4 = 2
N5 = 3

r1 = 3
r2 = 4
r3 = 5
r4 = 2
r5 = 2

U1 = np.random.randn(N1, r1)
U2 = np.random.randn(N2, r2)
U3 = np.random.randn(N3, r3)
U4 = np.random.randn(N4, r4)
U5 = np.random.randn(N5, r5)

C = np.random.randn(r1,r2,r3,r4,r5)
A = np.einsum('ia,jb,kc,ld,he,abcde->ijklh', U1, U2, U3, U4, U5, C)

# A = np.random.randn(N1, N2, N3, N4, N5)
print('A.shape=', A.shape)

Q_A = np.linalg.svd(A.swapaxes(0,1).reshape((N2, N1*N3*N4*N5)))[0][:,:r2]

Z = np.random.randn(N1, N1)
ZA = np.dot(Z, A.reshape((N1, N2*N3*N4*N5))).reshape((N1, N2, N3, N4, N5))
Q_ZA = np.linalg.svd(ZA.swapaxes(0,1).reshape((N2, N1*N3*N4*N5)))[0][:,:r2]
# Q_ZA = np.linalg.svd(ZA.reshape((N1*N2, N3*N4*N5)))[0][:,:r1]



# np.einsum('')

Q2_A, R2_A = np.linalg.qr(np.dot(np.linalg.inv(Z), Q_ZA.reshape((r1, N2))).reshape((r1*N2, N1*N2)))

err = np.linalg.norm(Q_A - Q2_A @ Q2_A.T @ Q_A) / np.linalg.norm(Q_A)
print('err', err)