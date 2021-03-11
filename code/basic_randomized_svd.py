import numpy as np
import scipy.sparse.linalg as spla

def basic_randomized_svd(A_linop, rank, oversampling_p=20):
    Omega = np.random.randn(A_linop.shape[1], rank+oversampling_p)
    Y = np.zeros((A_linop.shape[0], Omega.shape[1]))
    for k in range(Omega.shape[1]):
        Y[:,k] = A_linop.matvec(Omega[:,k])

    Q,ss,_ = np.linalg.svd(Y,0)

    X = np.zeros((Q.shape[1], A_linop.shape[1]))
    for k in range(Q.shape[1]):
        X[k,:] = A_linop.rmatvec(Q[:,k])

    U0,ss,Vt = np.linalg.svd(X,0)
    U = np.dot(Q, U0)
    return U[:rank,:], ss[:rank], Vt[:,:rank]