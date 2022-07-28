import numpy as np


####    Make random Tucker tensor A    ####

NN = [7,8,6,5]
rr_tucker = [3,4,5,2]
UU = [np.random.randn(N, r) for N, r in zip(NN, rr_tucker)]
C = np.random.randn(*tuple(rr_tucker))

A = np.einsum('ia,jb,kc,ld,abcd->ijkl', UU[0], UU[1], UU[2], UU[3], C)
print('A.shape=', A.shape)


####    Construct low rank basis, Q, for combined first two modes of A    ####

U, ss, _ = np.linalg.svd(A.reshape((NN[0]*NN[1], NN[2]*NN[3])), full_matrices=False)
good_inds = (ss >  np.max(ss) * 1e-10)
Q = U[:, good_inds]
print('Q.shape=', Q.shape)


####    Form tensor B by contracting random square matrix into first mode of A    ####

Z = np.random.randn(NN[0], NN[0])
B = np.einsum('ai,ijkl->ajkl', Z, A)


####    Construct low rank basis, QB, for combined first two modes of B    ####

UB, ssB, _ = np.linalg.svd(B.reshape((NN[0]*NN[1], NN[2]*NN[3])), full_matrices=False)
good_indsB = (ssB >  np.max(ssB) * 1e-10)
QB = UB[:, good_indsB]
print('QB.shape=', QB.shape)


####    Contract A^-1 into first mode of 3-tensor reshaped QB, turn into a matrix again, and reorthogonalize to get Q2    ####

G = np.einsum('ai,ijk->ajk', np.linalg.inv(Z), QB.reshape((NN[0], NN[1], QB.shape[1])))
Q2, _ = np.linalg.qr(G.reshape((NN[0]*NN[1], QB.shape[1])))

err = np.linalg.norm(Q - np.dot(Q2, np.dot(Q2.T, Q))) / np.linalg.norm(Q)
print('err=', err)
