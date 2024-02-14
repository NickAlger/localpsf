"""
Here is the package description that will appear on PyPI
"""
import numpy as np
import numpy.random
import numpy.linalg as LA
import scipy
import scipy.linalg
import time
__version__ = '0.0.1'


class HODLR:
    """
    This class contains

    1. Methods to sample from an operator and
       create a HODLR compressed representation.
       The sampling is done indirectly via 
       the `peeling' algorithm presented
       in Martinsson `Compressing Rank-Structured
       Matrices Via Randomized Sampling', 
       SIAM Journal on Scientific Computing, 2016.
       Keep in mind that if one has a matrix formed
       explicitly, that one can directly sample
       off-diagonal blocks and reduce the
       computational complexity.
    
    2. A method to perform mat-vec applies 
       with the HODLR compressed representation.
    
    3. A method to factor the HODLR compressed
       representation O(N log^2 N) and then use
       the factorization for efficient O(N log N)
       applies of the inverse of the hierarchically
       compressed operator.
       The factorization scheme is detailed in
       Ambikasaran and Darve `An O(N log N)
       Fast Direct Solver for Partial
       Hierarchically Semi-Separable Matrices',
       Springer Journal of Scientific Computing, 2013. 
       Note that the solves will fail unless 
       N / (2^L) is an integer.
    4. a) a means of generating a symmetric factorization
          A = W W^T of the symmetric nonsingular HODLR matrix.
       b) a means to apply the symmetric factor W.
       c) a means to apply the inverse of the symmetric factor.
       See the works of Ambikasaran and Darve for mathematical
       descriptions.
    """

    def __init__(self, op, B):
    #"""
    #Create a HODLR representation given:
    #    - Op: The operator that we wish to hierarchically compress. This
    #          operator must either be a numpy array or an object that
    #          has a .mult method available to it.
    #    - B: A numpy permutation matrix, so that we have the freedom
    #         to compress the matrix with respect to a particular basis.
    #"""
        self.op = op
        self.compressed = False
        self.op_applies = 0 # number of uncompressed operator applies
        self.Uarray = []
        self.Varray = []
        self.Barray = []
        self.Xarray = []
        self.Darray = []
        self.normarray = [] # array of matrix norm estimates
        self.errorarray = [] # array that contains random error estimates for each OD block in the hierarchy
        self.indicies = []
        self.dim = B.shape[0]
        self.B = B
        self.r0 = [] # to be a vector of initial guesses of the ranks
        self.r = [] # data which will contain a vector of ranks for each level of the compression
        self.L = 0  # number of levels in the hierarchical compression
        self.d = 0
        self.tau = 0.
        self.eta = 0.

        self.factored = False # Flag to mark if the HODLR matrix has been factored
        self.Uarray_upd = None

        self.Uarray_symupd = None
        self.DWarray = None
        self.qrtime = 0.*time.time()
   
    # neglects cost for repeated cost of applying test-matrices
    def compression_cost(self):
        print(self.r)
        cost = 4. * sum(self.r) + 2. * self.d * self.L + self.op.shape[0] / (2. ** self.L)
        return cost 
    def stab_gs(self, Q, y):
        y1 = y.copy()
        for gs_idx in range(len(Q[0,:])):
            q = Q[:, gs_idx]
            y1 = y1 - q * np.inner(q, y1)
            y1 = y1 / np.linalg.norm(y1)
        return y1
    
    # truncated action needed to peel the operator
    # during the compression procedure

    def mult_trunc(self, x, level):
        y = np.zeros(len(x))
        xb = x.copy()
        if level==0:
            return y
        else:
            for l in range(min(len(self.indicies), level)):
                index_set = (self.indicies)[l]
                for j in range(1, len(index_set), 2):
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    y[i1:i2] += np.dot((self.Uarray)[l][j-1], np.multiply((self.Barray)[l][j-1], np.dot((self.Varray)[l][j-1].T, xb[i2:i3])))
                    y[i2:i3] += np.dot((self.Uarray)[l][j], np.multiply((self.Barray)[l][j], np.dot((self.Varray)[l][j].T, xb[i1:i2])))
            return y
    
    def mult_truncT(self, x, level):
        y = np.zeros(len(x))
        xb = x.copy()
        if level==0:
            return y
        else:
            for l in range(min(len(self.indicies), level)):
                index_set = (self.indicies)[l]
                for j in range(len(index_set)):
                    if j % 2 == 1:
                        i1 = index_set[j-1]
                        i2 = index_set[j]
                        i3 = index_set[j+1]
                        y[i1:i2] += np.dot((self.Varray)[l][j], np.multiply((self.Barray)[l][j], np.dot((self.Uarray)[l][j].T, xb[i2:i3])))
                        y[i2:i3] += np.dot((self.Varray)[l][j-1], np.multiply((self.Barray)[l][j-1], np.dot((self.Uarray)[l][j-1].T, xb[i1:i2])))
            return y
        
    def compress_adaptive(self, lvl, d, tau, tau_r, eta):
        if type(lvl) not in [int]:
            raise TypeError("# of levels must be an integer.")
        if type(d) not in [int]:
            raise TypeError("# of oversampling vectors must be an integer.")
        if type(tau) not in [np.float64, float]:
            print(type(tau))
            raise TypeError("absolute tolerance must be a real number.")
        if type(tau_r) not in [np.float64, float]:
            print(type(tau_r))
            raise TypeError("relative tolerance must be a real number.")
        if type(eta) not in [float, np.float64]:
            raise TypeError("statistical parametera eta must be a real number.")
        self.dim = (self.op.shape)[0]
        dim = self.dim
        self.L = lvl
        self.r0 = np.ones((lvl,), dtype=int)
        rnks = self.r0
        self.d = d
        self.tau = tau
        self.eta = eta
          
        for levels in range(1, lvl+1):
            self.indicies = [[] for _ in range(levels)]
            di = dim / (2.0**levels)
            q = int(2**levels - 1)
            (self.indicies)[-1].append(0)
            for j in range(q):
                self.indicies[-1].append(int((j+1)*di))
            self.indicies[-1].append(dim)

            for l in range(2, levels + 1):
                for j in range(len((self.indicies)[-l+1])):
                    if j % 2 == 0:
                        (self.indicies)[-l].append((self.indicies)[-l+1][j])
        

        self.Uarray = [[] for _ in range(lvl)]
        self.Uarray_upd = [[] for _ in range(lvl)]
        self.Uarray_symupd = [[] for _ in range(lvl)]
        self.Varray = [[] for _ in range(lvl)]
        self.Barray = [[] for _ in range(lvl)]
        self.normarray = [[] for _ in range(lvl)]
        self.errorarray = [[] for _ in range(lvl)]

        for j in range(lvl):
            (self.Uarray)[j] = [[] for _ in range(2**(j+1))]
            (self.Uarray_upd)[j] = [[] for _ in range(2**(j+1))]
            (self.Uarray_symupd)[j] = [[] for _ in range(2**(j+1))]
            (self.Varray)[j] = [[] for _ in range(2**(j+1))]
            (self.Barray)[j] = [[] for _ in range(2**(j+1))]
            (self.normarray)[j] = [[] for _ in range(2**(j+1))]
            (self.errorarray)[j] = [[] for _ in range(2**(j+1))]
            for k in range(len((self.Uarray)[j])):
                (self.Uarray)[j][k] = []
                (self.Uarray_upd)[j][k] = []
                (self.Uarray_symupd)[j][k] = []
                (self.Varray)[j][k] = []
                (self.Barray)[j][k] = []
                (self.normarray)[j][k] = 0.0
                (self.errorarray)[j][k] = 0.0


        for l in range(lvl):
            index_set = (self.indicies)[l]
            rnk = rnks[l]
            r = int(rnk)

            Omega1s = np.zeros((dim, r))
            Omega2s = np.zeros((dim, r))
            Omega1t = np.zeros((dim, d))
            Omega2t = np.zeros((dim, d))
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    Omega1s[i1:i2,:] = np.random.randn(i2-i1, r)
                    Omega2s[i2:i3,:] = np.random.randn(i3-i2, r)
                    Omega1t[i1:i2,:] = np.random.randn(i2-i1, d)
                    Omega2t[i2:i3,:] = np.random.randn(i3-i2, d)
            Y1s = np.zeros((dim, r))
            Y2s = np.zeros((dim, r))
            Y1t = np.zeros((dim, d))
            Y2t = np.zeros((dim, d))

            for j in range(r):
                Y1s[:,j] = np.dot(self.op, Omega2s[:,j]) - self.mult_trunc(Omega2s[:,j], l)
                Y2s[:,j] = np.dot(self.op, Omega1s[:,j]) - self.mult_trunc(Omega1s[:,j], l)
            for j in range(d):
                Y1t[:,j] = np.dot(self.op, Omega2t[:,j]) - self.mult_trunc(Omega2t[:,j], l)
                Y2t[:,j] = np.dot(self.op, Omega1t[:,j]) - self.mult_trunc(Omega1t[:,j], l)
            
            for j in range(len(index_set)):
                if j % 2 == 1:
                     i1 = index_set[j-1]
                     i2 = index_set[j]
                     i3 = index_set[j+1]
                     for column in range(len(Y1s[0,:])):
                         (self.normarray)[l][j-1] = max((self.normarray)[l][j-1], LA.norm(Y1s[i1:i2,column],2)/LA.norm(Omega2s[i2:i3,column],2))
                         (self.normarray)[l][j] = max((self.normarray)[l][j], LA.norm(Y2s[i2:i3,column],2)/LA.norm(Omega1s[i1:i2,column],2))
                     

            # || A - Q Q^T A|| <= tau with probability (1-eta^(-d))
            # provided that
            # max_(i=1,2,...d) || (I - Q Q^T) A x_i || <= tau/(eta sqrt(2/pi))
            # where x_i are i.i.d. random vectors and
            # Q is a matrix with orthonormal columns
            # see Jianlin Xia, et al, 2014 -- 'A fast randomized eigensolver with
            # LDL update' 

            epsilon   = tau   / (eta*np.sqrt(2./np.pi))
            epsilon_r = tau_r / (eta*np.sqrt(2./np.pi))
            
            # error estimate
            eps = 0.0
                        
            Y1t_proj = Y1t.copy()
            Y2t_proj = Y2t.copy()

            # compute orthonormal bases for the span of the sampled spaces
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    Ualpha, _ = LA.qr(Y1s[i1:i2,:])
                    Ubeta, _  = LA.qr(Y2s[i2:i3,:])
                    (self.Uarray)[l][j-1] = Ualpha.copy()
                    (self.Uarray)[l][j] = Ubeta.copy()
                    Y1t_proj[i1:i2,:] = Y1t[i1:i2,:] - np.dot(Ualpha, np.dot(np.transpose(Ualpha), Y1t[i1:i2,:]))
                    Y2t_proj[i2:i3,:] = Y2t[i2:i3,:] - np.dot(Ubeta, np.dot(np.transpose(Ubeta), Y2t[i2:i3,:]))
                                    
                    for column in range(d):
                        error12 = LA.norm(Y1t_proj[i1:i2,column],2)
                        error23 = LA.norm(Y2t_proj[i2:i3,column],2)
                        eps = max(eps, error12, error23)
                        (self.errorarray)[l][j-1] = max( self.errorarray[l][j-1], error12)
                        (self.errorarray)[l][j] = max( self.errorarray[l][j], error23 )
            max_rank_it = i2-i1-r
            rank_it = 0
            within_rel_tol = False
            while eps > epsilon and max_rank_it > rank_it and not within_rel_tol:
                rank_it += 1
                s = 1
                r += 1
                eps = 0.0
                Y1t_proj = Y1t.copy()
                Y2t_proj = Y2t.copy()
                # construct two new structured random vectors in order to
                # extract more random test samples
                omega1t = np.zeros(dim)
                omega2t = np.zeros(dim)
                for j in range(len(index_set)):
                    if j % 2 == 1:
                        (self.errorarray)[l][j-1] = 0.
                        (self.errorarray)[l][j] = 0.
                        i1 = index_set[j-1]
                        i2 = index_set[j]
                        i3 = index_set[j+1]
                        start = time.time()
                        (self.Uarray)[l][j-1], _ = LA.qr(np.concatenate(((self.Uarray)[l][j-1], np.array([Y1t[i1:i2,s]]).T), axis=1))
                        (self.Uarray)[l][j], _ = LA.qr(np.concatenate(((self.Uarray)[l][j], np.array([Y2t[i2:i3,s]]).T), axis=1))
                        end = time.time()
                        self.qrtime += end-start
                        omega1t[i1:i2] = np.random.randn(i2-i1)
                        omega2t[i2:i3] = np.random.randn(i3-i2)
                Y1t[:,s] = np.dot(self.op, omega2t) - self.mult_trunc(omega2t, l)
                Y2t[:,s] = np.dot(self.op, omega1t) - self.mult_trunc(omega1t, l)

                # use recent samples of OD blocks to generate
                # better estimates of the matrix norms
                for j in range(len(index_set)):
                    if j % 2 == 1:
                        i1 = index_set[j-1]
                        i2 = index_set[j]
                        i3 = index_set[j+1]
                        (self.normarray)[l][j-1] = max((self.normarray)[l][j-1], LA.norm(Y1t[i1:i2,s],2)/LA.norm(omega2t[i2:i3],2))
                        (self.normarray)[l][j] = max((self.normarray)[l][j], LA.norm(Y2t[i2:i3,s],2)/LA.norm(omega1t[i1:i2],2))
                for j in range(len(index_set)):
                    if j % 2 == 1:
                        i1 = index_set[j-1]
                        i2 = index_set[j]
                        i3 = index_set[j+1]
                        Ualpha = (self.Uarray)[l][j-1].copy()
                        Ubeta = (self.Uarray)[l][j].copy()
                        Y1t_proj[i1:i2,:] = Y1t[i1:i2,:] - np.dot(Ualpha, np.dot(Ualpha.T, Y1t[i1:i2,:]))
                        Y2t_proj[i2:i3,:] = Y2t[i2:i3,:] - np.dot(Ubeta, np.dot(Ubeta.T, Y2t[i2:i3,:]))                     
                        for column in range(d):
                            error12 = LA.norm(Y1t_proj[i1:i2, column])
                            error23 = LA.norm(Y2t_proj[i2:i3, column])
                            eps = max(eps, error12, error23)
                            (self.errorarray)[l][j-1] = max((self.errorarray)[l][j-1], error12)
                            (self.errorarray)[l][j] = max((self.errorarray)[l][j], error23)
                
                within_rel_tol = True
                for j in range(len(index_set)):
                    if j % 2 == 1:
                        if ((self.errorarray)[l][j]) > epsilon_r*((self.normarray)[l][j]) or ((self.errorarray)[l][j-1]) > epsilon_r*((self.normarray)[l][j-1]):
                            within_rel_tol = False 
            rnks[l] = r
            Omega1s = np.zeros((dim, r))
            Omega2s = np.zeros((dim, r))
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    Omega1s[i1:i2,:] = (self.Uarray)[l][j-1]
                    Omega2s[i2:i3,:] = (self.Uarray)[l][j]      
             
            Z1 = np.zeros((dim,r))
            Z2 = np.zeros((dim,r))
            for j in range(r):
                Z1[:,j] = np.dot(self.op.T, Omega2s[:,j]) - self.mult_truncT(Omega2s[:,j], l)
                Z2[:,j] = np.dot(self.op.T, Omega1s[:,j]) - self.mult_truncT(Omega1s[:,j], l)
            
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    Valpha, Bbetaalpha, Ubetahat = LA.svd(Z1[i1:i2,:], full_matrices=False)
                    Vbeta, Balphabeta, Ualphahat = LA.svd(Z2[i2:i3,:], full_matrices=False)
                    (self.Varray)[l][j-1] = Vbeta
                    (self.Varray)[l][j] = Valpha
                    (self.Barray)[l][j-1] = Balphabeta
                    (self.Barray)[l][j] = Bbetaalpha
                    # update the left singular vectors
                    (self.Uarray)[l][j-1] = np.dot(self.Uarray[l][j-1], np.transpose(Ualphahat))
                    (self.Uarray)[l][j] = np.dot(self.Uarray[l][j], np.transpose(Ubetahat))
                    
        # loop over the leaves
        index_set = (self.indicies)[-1]
        
        # create an array to carry diagonal subblocks
        self.Darray = [[] for _ in range(len(index_set)-1)]
        for j in range(len(index_set)-1):
            i1 = index_set[j]
            i2 = index_set[j+1]
            (self.Darray)[j] = self.op[i1:i2, i1:i2]
        self.r = rnks
    
    def compress(self, lvl, r, p):
        self.dim = (self.op.shape)[0]
        dim = self.dim
        self.L = lvl
        self.r = r * np.ones((lvl,), dtype=int)
        self.p = p
        
        # ---- hierarchical partioning of the index set
        for levels in range(1, lvl+1):
            self.indicies = [[] for _ in range(levels)]
            di = self.dim / (2.0**levels)
            q = int(2**levels - 1)
            (self.indicies)[-1].append(0)
            for j in range(q):
                self.indicies[-1].append(int((j+1)*di))
            self.indicies[-1].append(self.dim)

            for l in range(2, levels + 1):
                for j in range(len((self.indicies)[-l+1])):
                    if j % 2 == 0:
                        (self.indicies)[-l].append((self.indicies)[-l+1][j])
        
        # ----
        self.Uarray = [[] for _ in range(lvl)]
        self.Varray = [[] for _ in range(lvl)]
        self.Barray = [[] for _ in range(lvl)]

        for j in range(lvl):
            (self.Uarray)[j] = [[] for _ in range(2**(j+1))]
            (self.Varray)[j] = [[] for _ in range(2**(j+1))]
            (self.Barray)[j] = [[] for _ in range(2**(j+1))]
            for k in range(len((self.Uarray)[j])):
                (self.Uarray)[j][k] = []
                (self.Varray)[j][k] = []
                (self.Barray)[j][k] = []


        for l in range(lvl):
            index_set = (self.indicies)[l]
            
            Omega1s = np.zeros((dim, r))
            Omega2s = np.zeros((dim, r))
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    Omega1s[i1:i2,:] = np.random.randn(i2-i1, r)
                    Omega2s[i2:i3,:] = np.random.randn(i3-i2, r)
            Y1 = np.zeros((dim, r))
            Y2 = np.zeros((dim, r))

            for j in range(r):
                Y1[:,j] = np.dot(self.op, Omega2s[:,j]) - self.mult_trunc(Omega2s[:,j], l)
                Y2[:,j] = np.dot(self.op, Omega1s[:,j]) - self.mult_trunc(Omega1s[:,j], l)
                     

            # compute orthonormal bases for the span of the sampled spaces
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    Ualpha, _ = LA.qr(Y1[i1:i2,:])
                    Ubeta, _  = LA.qr(Y2[i2:i3,:])
                    (self.Uarray)[l][j-1] = Ualpha.copy()
                    (self.Uarray)[l][j] = Ubeta.copy()
                    print(Ualpha.shape)
            Omega1s = np.zeros((dim, r))
            Omega2s = np.zeros((dim, r))
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    Omega1s[i1:i2,:] = (self.Uarray)[l][j-1]
                    Omega2s[i2:i3,:] = (self.Uarray)[l][j]      
             
            Z1 = np.zeros((dim,r))
            Z2 = np.zeros((dim,r))
            for j in range(r):
                Z1[:,j] = np.dot(self.op.T, Omega2s[:,j]) - self.mult_truncT(Omega2s[:,j], l)
                Z2[:,j] = np.dot(self.op.T, Omega1s[:,j]) - self.mult_truncT(Omega1s[:,j], l)
            
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    Valpha, Bbetaalpha, Ubetahat = LA.svd(Z1[i1:i2,:], full_matrices=False)
                    Vbeta, Balphabeta, Ualphahat = LA.svd(Z2[i2:i3,:], full_matrices=False)
                    (self.Varray)[l][j-1] = Vbeta
                    (self.Varray)[l][j] = Valpha
                    (self.Barray)[l][j-1] = Balphabeta
                    (self.Barray)[l][j] = Bbetaalpha
                    # update the left singular vectors
                    (self.Uarray)[l][j-1] = np.dot(self.Uarray[l][j-1], np.transpose(Ualphahat))
                    (self.Uarray)[l][j] = np.dot(self.Uarray[l][j], np.transpose(Ubetahat))
                    
        # loop over the leaves
        index_set = (self.indicies)[-1]
        
        # create an array to carry diagonal subblocks
        self.Darray = [[] for _ in range(len(index_set)-1)]
        for j in range(len(index_set)-1):
            i1 = index_set[j]
            i2 = index_set[j+1]
            (self.Darray)[j] = self.op[i1:i2, i1:i2]
            
                    

    def mult(self, x):
        y = np.zeros(len(x))
        xb = np.zeros(len(x))
        xb = np.dot(self.B, x)   
        for l in range(len(self.indicies)):
            index_set = (self.indicies)[l]
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    y[i1:i2] += np.dot((self.Uarray)[l][j-1], np.multiply((self.Barray)[l][j-1], np.dot((self.Varray)[l][j-1].T, xb[i2:i3])))
                    y[i2:i3] += np.dot((self.Uarray)[l][j], np.multiply((self.Barray)[l][j], np.dot((self.Varray)[l][j].T, xb[i1:i2])))
        
        # include contribution to action from diagonal blocks
        index_set = (self.indicies)[-1]
        for j in range(len(index_set)-1):
            i1 = index_set[j]
            i2 = index_set[j+1]
            y[i1:i2] += np.dot((self.Darray)[j], xb[i1:i2])
        y = np.dot(np.transpose(self.B), y)
        return y

    # form the operator from the data compressed rep
    def form(self):
        mat_form = np.zeros((self.dim, self.dim))
        for l in range(len(self.indicies)):
            index_set = (self.indicies)[l]
            for j in range(len(index_set)):
                if j % 2 == 1:
                    i1 = index_set[j-1]
                    i2 = index_set[j]
                    i3 = index_set[j+1]
                    mat_form[i1:i2,i2:i3] += np.dot(self.Uarray[l][j-1], np.multiply(self.Varray[l][j-1], self.Barray[l][j-1]).T)
                    mat_form[i2:i3,i1:i2] += np.dot(self.Uarray[l][j], np.multiply(self.Varray[l][j], self.Barray[l][j]).T)
        index_set = (self.indicies)[-1]
        for j in range(len(index_set)-1):
            i1 = index_set[j]
            i2 = index_set[j+1]
            mat_form[i1:i2,i1:i2] += (self.Darray)[j]
        mat_form = np.dot(np.dot(self.B.T, mat_form), self.B)
        return mat_form
           

    # function to return the solution x of the system (I + U V^T) x = b
    # employing the sherman-morrison-woodbury formula
    # (I + U V^T)^(-1) = I - U (I + V^T U)^(-1) V^T
    def lowranksolve(self, U, VT, b):
        n, k = U.shape
        VTb = np.dot(VT, b)
        sol = b - np.dot(U, np.linalg.solve(np.identity(k) + np.dot(VT,U), VTb))
        test = False
        if test:
            residual = np.dot((np.identity(n) + np.dot(U, VT)), sol)
            print("low rank solve error = ", LA.norm(residual-b)/LA.norm(b))
        return sol

    def lowranksymmfact(self, U, Sig):
        n, k = U.shape
        Q, R = LA.qr(U)
        #M = LA.cholesky(np.identity(k) + np.dot(np.dot(R, Sig), R.T))
        M = scipy.linalg.sqrtm(np.identity(k) + np.dot(np.dot(R, Sig), R.T))
        #X = LA.solve(R, LA.solve(R, M - np.identity(k)).T).T
        # given the number of solves, it is potentially simpler to form
        # the inverse and apply it
        R_inv = np.linalg.inv(R)
        X = np.dot(R_inv, np.dot(M - np.identity(k), R_inv.T))
        return X

    # function to compute a direct factorization of a HODLR compressed matrix
    # Uarray, Barray, Varray are arrays of the RSVD data of the off-diagonal blocks
    # index_set is a set of indicies that specifies the hierarchical partitioning
    # D is an approximation of the diagonal blocks of A
    # the output is a set of the factors [ A_L, A_(L-1),...,A_0 ]
    # A_L is equal to D
    # A_(L-1), A_(L-2),...,A_0 are block diagonal matrices that are low rank-updates to the identity


################################################################################################


###############################################################################################
    def HODLRfactor(self):
        # the essential part of the factorization scheme is in updating the column space vectors
        # ordered high to low: L-1, L-2,..., 1, 0
        # Note that A_L = D, so it will not be included in Uarray
        # The diagonal blocks of D are not low rank perturbations to the identity
        for l in range(len(self.Uarray)):
            for j in range(len(self.Uarray[l])):
                self.Uarray_upd[l][j] = (self.Uarray[l][j]).copy()


        # Factor A_L from A by performing inverse applies of dense
        # matrices from the leaves on left-column vectors.
        
        index_setL = self.indicies[-1]
        for jL in range(1, len(index_setL)):
            i1L = index_setL[jL-1]
            i2L = index_setL[jL]
            for l in range(len(self.indicies)):
                index_set = self.indicies[l]
                for j in range(len(index_set)):
                    if (j % 2 == 1):
                        i1 = index_set[j-1]
                        i2 = index_set[j]
                        i3 = index_set[j+1]
                        if i1 <= i1L and i2L <= i2:
                            self.Uarray_upd[l][j-1][i1L-i1:i1L-i1+(i2L-i1L),:] = np.linalg.solve((self.Darray)[jL-1], (self.Uarray_upd)[l][j-1][i1L-i1:i1L-i1+(i2L-i1L),:].copy())
                        elif i2 <= i1L and i2L <= i3:
                            self.Uarray_upd[l][j][i1L-i2:i1L-i2+(i2L-i1L),:] = np.linalg.solve((self.Darray)[jL-1], self.Uarray_upd[l][j][i1L-i2:i1L-i2+(i2L-i1L),:].copy())
        # A_L has been now factored from A.
        # one must traverse the hierarchy in order of increasing coarseness
        # note that one does not need to do a factorization step at the coarsest level
        for L_eff in reversed(range(0, len(self.indicies)-1)):
            index_setL = self.indicies[L_eff]
            for jL in range(1, len(index_setL)):
                # each diagonal block for this level L_eff H-matrix
                # is a low rank perturbation of the identity
                # the A[i1L:i2L, i1L:i2L] diagonal subblock of the
                # now modified matrix A.
                # The diagonal subblock 
                # it has a data-sparse rep as I + U V^T 
                i1L = index_setL[jL-1]
                i2L = index_setL[jL]
                kL = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[1] + self.Uarray_upd[L_eff+1][2*(jL-1)+1].shape[1]
                kL1 = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[1]
                #NL = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[0]
                #DUi = np.zeros((2*NL, kL))
                NL1 = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[0]
                NL2 = self.Uarray_upd[L_eff+1][2*(jL-1)+1].shape[0]
                DUi = np.zeros((NL1+NL2, kL))
                # now the effective level L_eff H-matrix
                # has diagonal blocks are low-rank perturbations
                # of the identity. However the perturbation
                # data is naturally stored in at level L_eff+1
                # corresponding to how this data was initially
                # stored in the original compress H-matrix format
                DUi[:NL1,:kL1] = self.Uarray_upd[L_eff+1][2*(jL-1)][:,:]
                DUi[NL1:,kL1:] = self.Uarray_upd[L_eff+1][2*(jL-1)+1][:,:]   
                DVi = np.zeros((NL1+NL2, kL))
                DVi[NL1:,:kL1] = np.dot(self.Varray[L_eff+1][2*(jL-1)], np.diag(self.Barray[L_eff+1][2*(jL-1)]))
                DVi[:NL1,kL1:] = np.dot(self.Varray[L_eff+1][2*(jL-1)+1], np.diag(self.Barray[L_eff+1][2*(jL-1)+1]))
                for l in range(L_eff+1):
                    index_set = self.indicies[l]
                    for j in range(len(index_set)):
                        if (j % 2 == 1):
                            i1 = index_set[j-1]
                            i2 = index_set[j]
                            i3 = index_set[j+1]
                            if i1 <= i1L and i2L <= i2:
                                self.Uarray_upd[l][j-1][i1L-i1:i1L-i1+(i2L-i1L),:] = self.lowranksolve(DUi, np.transpose(DVi), self.Uarray_upd[l][j-1][i1L-i1:i1L-i1+(i2L-i1L),:])
                            elif i2 <= i1L and i2L <= i3:
                                self.Uarray_upd[l][j][i1L-i2:i1L-i2+(i2L-i1L),:] = self.lowranksolve(DUi, np.transpose(DVi), self.Uarray_upd[l][j][i1L-i2:i1L-i2+(i2L-i1L),:])
        self.factored = True
    


        # using the updated column vector matrices now perform a solve on Ax = b, A = A_L A_(L-1)...A_0 b ==> x = A_0^(-1)A_1^(-1)...A_L^(-1) b
        # b should be a numpy array
    def HODLRsolve(self, b):
        if self.factored == False:
            self.HODLRfactor()
        sol = b.copy()
        ########################### contents of the solve ######################################
        index_setL = self.indicies[-1]
        # perform dense solves on leaf nodes
        for jL in range(1, len(index_setL)):
            i1L = index_setL[jL-1]
            i2L = index_setL[jL]
            sol[i1L:i2L] = np.linalg.solve((self.Darray)[jL-1], sol[i1L:i2L])
            
        # for each level that is not a leaf perform solves using SMW solves.....
        # traverse the hierarchy from fine to coarse.
        for L_eff in reversed(range(len(self.indicies)-1)):
            index_setL = self.indicies[L_eff]
            for jL in range(1, len(index_setL)):
                i1L = index_setL[jL-1]
                i2L = index_setL[jL] 
                # the current diagonal block is a low-rank perturbation to the identity
                # obtain information concerning the low-rank perturbation to employ a SMW solve 
                # from which we can factor A_Leff
                kL = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[1] + self.Uarray_upd[L_eff+1][2*(jL-1)+1].shape[1]
                kL1 = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[1]
                #NL = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[0]
                #DUi = np.zeros((2*NL, kL))
                NL1 = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[0]
                NL2 = self.Uarray_upd[L_eff+1][2*(jL-1)+1].shape[0]
                DUi = np.zeros((NL1+NL2, kL))
                DUi[:NL1,:kL1] = self.Uarray_upd[L_eff+1][2*(jL-1)]
                DUi[NL1:,kL1:] = self.Uarray_upd[L_eff+1][2*(jL-1)+1]
                DVi = np.zeros((NL1+NL2, kL))
                DVi[NL1:,:kL1] = np.dot(self.Varray[L_eff+1][2*(jL-1)], np.diag(self.Barray[L_eff+1][2*(jL-1)]))
                DVi[:NL1,kL1:] = np.dot(self.Varray[L_eff+1][2*(jL-1)+1], np.diag(self.Barray[L_eff+1][2*(jL-1)+1]))
                sol[i1L:i2L] = self.lowranksolve(DUi, np.transpose(DVi), sol[i1L:i2L])
 
        # now perform a solve on the coarsest level
        #NL = self.Uarray_upd[0][0].shape[0]
        NL1 = self.Uarray_upd[0][0].shape[0]
        NL2 = self.Uarray_upd[0][1].shape[0]
        kL1 = self.Uarray_upd[0][0].shape[1]
        kL = kL1 + self.Uarray_upd[0][1].shape[1]
        #DUi = np.zeros((2*NL, kL))
        #DVi = np.zeros((2*NL, kL))
        DUi = np.zeros((NL1+NL2, kL))
        DVi = np.zeros((NL1+NL2, kL))
 
        DUi[:NL1,:kL1] = self.Uarray_upd[0][0][:,:]
        DUi[NL1:,kL1:] = self.Uarray_upd[0][1][:,:]
        DVi[NL1:,:kL1] = np.dot(self.Varray[0][0][:,:], np.diag(self.Barray[0][0]))
        DVi[:NL1,kL1:] = np.dot(self.Varray[0][1][:,:], np.diag(self.Barray[0][1]))
        sol = self.lowranksolve(DUi, np.transpose(DVi), sol.copy())
        return sol.copy()
 
 ##############################################################################################
    def HODLRsymmfactor(self):
        for l in range(len(self.Uarray)):
            for j in range(len(self.Uarray[l])):
                self.Uarray_symupd[l][j] = (self.Uarray[l][j]).copy()
        
        self.DWarray = [[] for _ in range(len(self.indicies[-1])-1)]
        for j in range(len(self.Darray)):
            self.DWarray[j] = np.linalg.cholesky(self.Darray[j])
        
        # prepare the new data structure
        self.Xarray = [[] for _ in range(self.L)]
        for j in range(len(self.Xarray)):
            (self.Xarray)[j] = [[] for _ in range(2**j)]
        
        # Factor W_L from A by performing dense solves on the leaves
        index_setL = self.indicies[-1]
        for jL in range(1, len(index_setL)):
            i1L = index_setL[jL-1]
            i2L = index_setL[jL]
            for l in range(len(self.indicies)):
                index_set = (self.indicies)[l]
                for j in range(len(index_set)):
                    if (j % 2 == 1):
                        i1 = index_set[j-1]
                        i2 = index_set[j]
                        i3 = index_set[j+1]
                        if i1 <= i1L and i2L <= i2:
                            self.Uarray_symupd[l][j-1][i1L-i1:i1L-i1+(i2L-i1L),:] = np.linalg.solve((self.DWarray)[jL-1], (self.Uarray_symupd)[l][j-1][i1L-i1:i1L-i1+(i2L-i1L),:])
                        elif i2 <= i1L and i2L <= i3:
                            self.Uarray_symupd[l][j][i1L-i2:i1L-i2+(i2L-i1L),:] = np.linalg.solve((self.DWarray)[jL-1], self.Uarray_symupd[l][j][i1L-i2:i1L-i2+(i2L-i1L),:])
        
        # now that W_L has been factored from A
        # one must traverse the hierarchy from fine to coarse
        # one must generate the X data at all levels
        # but does not need to propagate the inverse data
        # at the finest level of refinement
        
        for L_eff in reversed(range(-1, len(self.indicies)-1)):
            if L_eff >= 0:
                index_setL = self.indicies[L_eff].copy()
            else:
                index_setL = [0, self.dim]
            for jL in range(1, len(index_setL)):
                i1L = index_setL[jL-1]
                i2L = index_setL[jL]
                # the current diagonal block is a low-rank perturbation to the identity
                # obtain information concerning the low-rank perturbation to employ a SMW solve 
                # from which we can factor A_Leff
                kL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1] + self.Uarray_symupd[L_eff+1][2*(jL-1)+1].shape[1]
                kL1 = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1]
                NL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[0]
                # need to perform an application of (I + UV^T)^(-1) on set of column vectors
                DUi = np.zeros((2*NL, kL))
                DUi[:NL,:kL1] = self.Uarray_symupd[L_eff+1][2*(jL-1)][:,:]
                DUi[NL:,kL1:] = self.Uarray_symupd[L_eff+1][2*(jL-1)+1][:,:]
                                
                Sig = np.zeros((kL, kL))
                Sig[:kL1, kL1:] = np.diag(self.Barray[L_eff+1][2*(jL-1)])
                Sig[kL1:, :kL1] = np.diag(self.Barray[L_eff+1][2*(jL-1)+1])
                
                U1 = self.Uarray_symupd[L_eff+1][2*(jL-1)][:,:]
                U2 = self.Uarray_symupd[L_eff+1][2*(jL-1)+1][:,:]
                Q1, R1 = np.linalg.qr(U1)
                Q2, R2 = np.linalg.qr(U2)
                Sig1 = np.diag(self.Barray[L_eff+1][2*(jL-1)])
                Sig2 = np.diag(self.Barray[L_eff+1][2*(jL-1)+1])
                Sig1 = np.dot(np.dot(R1, Sig1), R2.T)
                Sig2 = np.dot(np.dot(R2, Sig2), R1.T)
                self.Uarray_symupd[L_eff+1][2*(jL-1)] = Q1.copy()
                self.Uarray_symupd[L_eff+1][2*(jL-1)+1] = Q2.copy()
                DUi[:NL,:kL1] = Q1.copy()
                DUi[NL:,kL1:] = Q2.copy()
                Sig[:kL1,kL1:] = Sig1.copy()
                Sig[kL1:, :kL1] = Sig2.copy()
                self.Xarray[L_eff+1][jL-1] = self.lowranksymmfact(DUi, Sig)
               
                # absorb X into the V array
                # and propagate inverses
                DVTi = np.dot(self.Xarray[L_eff+1][jL-1], DUi.T)
                #print("cond(I + V^T U) = ", LA.cond(np.identity(kL) + np.dot(DVTi, DUi)))
                if L_eff >= 0:
                    for l in range(L_eff+1):
                        index_set = self.indicies[l]
                        for j in range(len(index_set)):
                            if (j % 2 == 1):
                                i1 = index_set[j-1]
                                i2 = index_set[j]
                                i3 = index_set[j+1]
                                if i1 <= i1L and i2L <= i2:
                                    self.Uarray_symupd[l][j-1][i1L-i1:i1L-i1+(i2L-i1L),:] = self.lowranksolve(DUi, DVTi, self.Uarray_symupd[l][j-1][i1L-i1:i1L-i1+(i2L-i1L),:])
                                elif i2 <= i1L and i2L <= i3:
                                    self.Uarray_symupd[l][j][i1L-i2:i1L-i2+(i2L-i1L),:] = self.lowranksolve(DUi, DVTi, self.Uarray_symupd[l][j][i1L-i2:i1L-i2+(i2L-i1L),:])
 
    
    def HODLRsqrtapply(self, x):
        sol = x.copy()
        # traverse the hierarchy from coarse to fine.
        for L_eff in range(-1, len(self.indicies)-1):
            if L_eff >= 0:
                index_setL = self.indicies[L_eff]
            else:
                index_setL = [0, self.dim]
            for jL in range(1, len(index_setL)):
                i1L = index_setL[jL-1]
                i2L = index_setL[jL] 
                # the current diagonal block is a low-rank perturbation to the identity
                # obtain information concerning the low-rank perturbation to employ a SMW solve 
                # from which we can factor A_Leff
                kL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1] + self.Uarray_symupd[L_eff+1][2*(jL-1)+1].shape[1]
                kL1 = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1]
                NL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[0]
                DUi = np.zeros((2*NL, kL))
                DUi[:NL,:kL1] = self.Uarray_symupd[L_eff+1][2*(jL-1)][:,:]
                DUi[NL:,kL1:] = self.Uarray_symupd[L_eff+1][2*(jL-1)+1][:,:]
                sol[i1L:i2L] += np.dot(DUi, np.dot(self.Xarray[L_eff+1][jL-1], np.dot(DUi.T, sol[i1L:i2L])))
        index_setL = self.indicies[-1]
        for jL in range(1, len(index_setL)):
            i1L = index_setL[jL-1]
            i2L = index_setL[jL]
            sol[i1L:i2L] = np.dot((self.DWarray)[jL-1], sol[i1L:i2L])
        return sol
    
    def HODLRsqrtapplyT(self, x):
        sol = x.copy()
        # traverse the hierarchy from fine to coarse
        index_setL = self.indicies[-1]
        for jL in range(1, len(index_setL)):
            i1L = index_setL[jL-1]
            i2L = index_setL[jL]
            sol[i1L:i2L] = np.dot((self.DWarray)[jL-1].T, sol[i1L:i2L])
    
        for L_eff in reversed(range(-1, len(self.indicies)-1)):
            if L_eff >=0:
                index_setL = self.indicies[L_eff]
            else:
                index_setL = [self.indicies[0][0], self.indicies[0][-1]]
            for jL in range(1, len(index_setL)):
                i1L = index_setL[jL-1]
                i2L = index_setL[jL] 
                kL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1] + self.Uarray_symupd[L_eff+1][2*(jL-1)+1].shape[1]
                kL1 = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1]
                NL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[0]
                DUi = np.zeros((2*NL, kL))
                DUi[:NL,:kL1] = self.Uarray_symupd[L_eff+1][2*(jL-1)].copy()
                DUi[NL:,kL1:] = self.Uarray_symupd[L_eff+1][2*(jL-1)+1].copy()
                sol[i1L:i2L] += np.dot(DUi, np.dot(self.Xarray[L_eff+1][jL-1].T, np.dot(DUi.T, sol[i1L:i2L])))
        return sol
    
    def HODLRinvsqrtapply(self, b):
        sol = b.copy()
        
        # perform dense solves on leaf nodes
        index_setL = self.indicies[-1]
        for jL in range(1, len(index_setL)):
            i1L = index_setL[jL-1]
            i2L = index_setL[jL]
            sol[i1L:i2L] = np.linalg.solve((self.DWarray)[jL-1], sol[i1L:i2L])
        # for each level that is not a leaf perform solves using SMW solves.....
        # traverse the hierarchy from fine to coarse.
        for L_eff in reversed(range(-1, len(self.indicies)-1)):
            if L_eff >= 0:
                index_setL = self.indicies[L_eff]
            else:
                index_setL = [0, self.dim]
            for jL in range(1, len(index_setL)):
                i1L = index_setL[jL-1]
                i2L = index_setL[jL] 
                kL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1] + self.Uarray_symupd[L_eff+1][2*(jL-1)+1].shape[1]
                kL1 = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1]
                NL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[0]
                DUi = np.zeros((2*NL, kL))
                DUi[:NL,:kL1] = self.Uarray_symupd[L_eff+1][2*(jL-1)]
                DUi[NL:,kL1:] = self.Uarray_symupd[L_eff+1][2*(jL-1)+1]
                DVi = np.zeros((2*NL, kL))
                DVTi = np.dot(self.Xarray[L_eff+1][jL-1], DUi.T)
                sol[i1L:i2L] = self.lowranksolve(DUi, DVTi, sol[i1L:i2L])
        return sol.copy()

    def HODLRinvsqrtapplyT(self, b):
        sol = b.copy()
        # for each level that is not a leaf perform solves 
        # employing the SMW identity
        # traverse the hierarchy from coarse to fine.
        for L_eff in range(-1, len(self.indicies)-1):
            if L_eff >= 0:
                index_setL = self.indicies[L_eff]
            else:
                index_setL = [0, self.dim]
            for jL in range(1, len(index_setL)):
                i1L = index_setL[jL-1]
                i2L = index_setL[jL] 
                kL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1] + self.Uarray_symupd[L_eff+1][2*(jL-1)+1].shape[1]
                kL1 = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[1]
                NL = self.Uarray_symupd[L_eff+1][2*(jL-1)].shape[0]
                DUi = np.zeros((2*NL, kL))
                DUi[:NL,:kL1] = self.Uarray_symupd[L_eff+1][2*(jL-1)]
                DUi[NL:,kL1:] = self.Uarray_symupd[L_eff+1][2*(jL-1)+1]
                DVi = np.zeros((2*NL, kL))
                DVTi = np.dot(self.Xarray[L_eff+1][jL-1].T, DUi.T)
                sol[i1L:i2L] = self.lowranksolve(DUi, DVTi, sol[i1L:i2L])
        # perform dense solves on leaf nodes
        index_setL = self.indicies[-1]
        for jL in range(1, len(index_setL)):
            i1L = index_setL[jL-1]
            i2L = index_setL[jL]
            sol[i1L:i2L] = np.linalg.solve((self.DWarray)[jL-1].T, sol[i1L:i2L])
        return sol.copy()
    
    
    #################### a useless algorithm ########################
    #################### only used for testing ######################
    ######## A = A_L A_(L-1) ... A_1 A_0 ###########################
    # the following applies A_0 to a vector and then A_1 to the
    # resultant vector and so on    
    def HODLRfactapply(self, x):
        sol = x.copy()
        
        # traverse the hierarchy from fine to coarse.
        for L_eff in range(-1, len(self.indicies)-1):
            if L_eff < 0:
                index_setL = [0, self.dim]
            else:
                index_setL = self.indicies[L_eff]
            #index_setL = self.indicies[L_eff]
            for jL in range(1, len(index_setL)):
                i1L = index_setL[jL-1]
                i2L = index_setL[jL] 
                # the current diagonal block is a low-rank perturbation to the identity
                # obtain information concerning the low-rank perturbation to employ a SMW solve 
                # from which we can factor A_Leff
                kL = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[1] + self.Uarray_upd[L_eff+1][2*(jL-1)+1].shape[1]
                kL1 = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[1]
                NL = self.Uarray_upd[L_eff+1][2*(jL-1)].shape[0]
                DUi = np.zeros((2*NL, kL))
                DUi[:NL,:kL1] = self.Uarray_upd[L_eff+1][2*(jL-1)]
                DUi[NL:,kL1:] = self.Uarray_upd[L_eff+1][2*(jL-1)+1]
                DVi = np.zeros((2*NL, kL))
                DVi[NL:,:kL1] = np.dot(self.Varray[L_eff+1][2*(jL-1)], np.diag(self.Barray[L_eff+1][2*(jL-1)]))
                DVi[:NL,kL1:] = np.dot(self.Varray[L_eff+1][2*(jL-1)+1], np.diag(self.Barray[L_eff+1][2*(jL-1)+1]))
                sol[i1L:i2L] += np.dot(DUi, np.dot(DVi.T, sol[i1L:i2L]))
        index_setL = self.indicies[-1]
        for jL in range(1, len(index_setL)):
            i1L = index_setL[jL-1]
            i2L = index_setL[jL]
            sol[i1L:i2L] = np.dot((self.Darray)[jL-1], sol[i1L:i2L])
        return sol
