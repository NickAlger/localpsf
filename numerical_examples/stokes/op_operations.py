import dolfin as dl
import numpy as np
class op_axpy:
    """
    This class is simply to compute the action
    Y + alpha X
    where Y and X are matrix-free operators
    and alpha is a scalar
    """
    def __init__(self, opx, alpha, opy):
        self.opx = opx
        self.opy = opy
        self.alpha = alpha
        self.ywork = dl.Vector()
        self.init_vector(self.ywork, 0)
                
    def init_vector(self, x , dim):
        self.opx.init_vector(x, dim)
        
    def mult(self, x, y):        
        self.opy.mult(x, y)
        self.opx.mult(x, self.ywork)
        y.axpy(self.alpha, self.ywork)



class solver_from_dot:
    def __init__(self, M):
        self.M = M
        self.shape = M.shape
    def init_vector(self, x, dim):
        x.init(self.shape[dim])
    def solve(self, x, b):
        x.set_local(self.M.dot(b[:]))


class proj_op:
    """
    This class is implemented in order
    to compute the action of operator op
    on a subspace
    Proj: full space --> subspace
    op:   full space --> full space
    proj_op: subspace --> subspace
    the action of proj_op will be the
    action of P op P^T
    """
    def __init__(self, op, Proj):
        self.op = op
        self.Proj = Proj
        # intermediate full space help vectors
        self.xfull = dl.Vector()
        self.yfull = dl.Vector()
        self.Proj.init_vector(self.xfull, 1)
        self.Proj.init_vector(self.yfull, 1)

    def init_vector(self, x, dim):
        self.Proj.init_vector(x, 0)

    def mult(self, x, y):
        self.Proj.transpmult(x, self.xfull)
        self.op.mult(self.xfull, self.yfull)
        self.Proj.mult(self.yfull, y)

class projT_op:
    """
    This class is implemented in order
    to compute the action of 
    subspace operator op
    on a fullspace
    Proj: full space --> subspace
    op:   subspace --> subspace
    projT_op: full space --> full space
    the action of projT_op will be the
    action of P^T op P
    """
    def __init__(self, op, Proj):
        self.op = op
        self.Proj = Proj
        # intermediate subspace help vectors
        self.xhelp = dl.Vector()
        self.yhelp = dl.Vector()
        self.Proj.init_vector(self.xhelp, 0)
        self.Proj.init_vector(self.yhelp, 0)

    def init_vector(self, x, dim):
        self.Proj.init_vector(x, 1)

    def mult(self, x, y):
        self.Proj.mult(x, self.xhelp)
        self.op.mult(self.xhelp, self.yhelp)
        self.Proj.transpmult(self.yhelp, y)




class op_B:
    """
    This class is implemented in order to
    compute the action of an operator
    with respect ao another basis
    op: unordered space --> unordered space
    B: unordered space --> ordered space space
    op_B: ordered space --> ordered space
    """
    def __init__(self, op, B):
        self.op = op
        self.B = B
        
    def init_vector(self, x, dim):
        self.op.init_vector(x, 0)
    def mult(self, x, y):
        BTx = dl.Vector()
        opBTx = dl.Vector()
        self.op.init_vector(BTx, 0)
        self.op.init_vector(opBTx, 0)

        BTx.set_local( np.dot(B.T, x.get_local()))
        self.op.mult(BTx, opBTx)
        y.set_local( np.dot(B, opBTx.get_local()))


class op_np:
    """
    This class is implemented in order to
    transform a numpy operator, to one
    that has a action with dolfin arrays
    """
    def __init__(self, op, opnp):
        self.op = op
        self.opnp = opnp
        self.n = opnp.shape[0]
    def init_vector(self, x, dim):
        self.op.init_vector(x, dim)
    def mult(self, x, y):
        y.set_local(self.opnp.dot(x.get_local()))

class op_AB:
    """
    This class is implemented to
    obtain the composite action AB
    y = A(B(x))
    """
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def init_vector(self, x, dim):
        if dim == 0:
            self.B.init_vector(x, 0)
        else:
            self.A.init_vector(x, 1)
    def mult(self, x, y):
        help1 = dl.Vector()
        self.A.init_vector(help1, 0)
        self.A.mult(x, help1)
        self.B.mult(help1, y)

class op_Ainv:
    """
    This class is implemented to
    obtain the action Ainv
    x = Ainv(x)
    """
    def __init__(self, A, Asolver):
        self.A       = A
        self.Asolver = Asolver
    def init_vector(self, x, dim):
        if dim == 0:
            self.A.init_vector(x, 0)
        else:
            self.A.init_vector(x, 1)
    def mult(self, y, x):
        self.Asolver.solve(x,y)


##
# Is there a purpose to the projT_op class?
# Does the projected solver method remove the need
# for this class?
##


class projT_op:
    """
    This class is implemented in order
    to compute the action of an operator op
    on a larger space
    Proj: full space --> subspace
    op:   small space --> small space
    projT_op: full space --> full space
    the action of projT_op will be the
    action of P^T op P
    """
    def __init__(self, op, Proj):
        self.op = op
        self.Proj = Proj
    def init_vector(self, x, dim):
        self.Proj.init_vector(x, 1)
    def mult(self, x, y):
        xproj = dl.Vector()
        yproj = dl.Vector()
        self.Proj.init_vector(xproj, 0)
        self.Proj.init_vector(yproj, 0)
        self.Proj.mult(x, xproj)
        self.op.mult(xproj, yproj)
        self.Proj.transpmult(yproj, y)


class projected_solver:
    """
    This class is to be used in the following way.
    The NewtonCG algorithm as implemented in 
    hIPPYlib will iterate over all the dof of the
    parameter even if only a subset of the parameter
    dof actually enter the problem. For instance,
    the parameter field may only enter as a boundary
    condition into the inverse problem but the parameter
    will be defined in the entire computational domain.


    projector Proj, full space --> subspace
    Htilde - approximate hessian action on subspace

    to then obtain an approximate solve of Hx = b
    on the full space given the approximate action
    on the subspace Htilde we do the following
    H x = b
    x \ approx P^T (Htilde^-1) P b

    """
    def __init__(self, Proj, H):
        self.Proj = Proj
        self.H = H
        
    def solve(self, x, b):
        Pb = dl.Vector()
        HinvPb = dl.Vector()
        self.Proj.init_vector(Pb, 0)
        self.Proj.init_vector(HinvPb, 0)
        self.Proj.mult(b, Pb)
        self.H.solve(HinvPb, Pb)
        self.Proj.transpmult(HinvPb, x)


class projected_solverB:
    # projector P, maps to the subspace
    # H - approximate hessian action on the subspace
    # B - permutation matrix on the small space
    def __init__(self, B, Proj, H):
        self.B = B
        self.Proj = Proj
        self.H = H
        
    def solve(self, x, b):
        Pb = dl.Vector()
        HinvPb = dl.Vector()
        self.Proj.init_vector(Pb, 0)
        self.Proj.init_vector(HinvPb, 0)
        self.Proj.mult(b, Pb)
        Pb.set_local(np.dot(self.B, Pb.get_local()))
        self.H.solve(HinvPb, Pb)
        HinvPb.set_local(np.dot(self.B.T, HinvPb.get_local()))
        self.Proj.transpmult(HinvPb, x)

#class identity_preconditioner:
#    def solve(self, x, b):
#        x.set_local(b.get_local())

#class identity_operator:
#    def mult(self, x, b):
#        b.set_local(x.get_local())

class op_I:
    def mult(self, x, b):
        b.set_local(x.get_local())
    def solve(self, x, b):
        x.set_local(b.get_local())


