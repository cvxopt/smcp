# Copyright 2010-2018 M. S. Andersen & L. Vandenberghe
#
# This file is part of SMCP.
#
# SMCP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SMCP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SMCP.  If not, see <http://www.gnu.org/licenses/>.


from cvxopt.base import spmatrix, matrix, sparse, spdiag, sqrt, uniform, normal, mul, setseed, syrk
from cvxopt import blas, base, lapack
from smcp import solvers, misc
from cvxopt.solvers import sdp as cvxoptsdp
import chompack
import random,os

class SDP(object):
    """
    Creates SDP object from data file:

    P = SDP(fname)

    PURPOSE
    Creates SDP object from problem data. Sparse SDPA data format (dat-s files) and
    Python 'pickled' data (pkl files) can be read and written.

    ARGUMENTS
    fname     String, path to data file

    RETURNS
    P         SDP object
    """
    _pname = None    # problem name
    _I = None        # Index list, aggregate sparsity pattern
    _ischordal = None
    _A = None; _b = None;
    _blockstruct = None
    _X0 = None; _y0 = None; _S0 = None

    def __init__(self, filename=None,c=None,G=None,h=None,dims=None):
        """
        Class SDP consructor.

        Example:

           P = SDP('datafile.dat-s')

        """
        if filename:
            fp,ext = os.path.splitext(filename)
            self._pname = fp.split('/')[-1]
            if ext == '.dat-s': self._read_sdpa(filename)
            elif ext == '.pkl': self._load(filename)
            elif os.path.splitext(fp)[1] == '.dat-s':
                self._pname = self._pname[:-6]
                self._read_sdpa(filename)
            elif os.path.splitext(fp)[1] == '.pkl': self._load(filename)
            else: raise NameError('Unknown file extension')
            self._agg_sparsity()


    def __str__(self):
        """Returns SDP object info string"""
        buf = "<SDP: n=%i, m=%i, nnz=%i> %s" % (self.n,self.m,
                                                self.nnz,self._pname)
        return buf

    @property
    def n(self):
        """Order of matrices"""
        if self._A is not None: return int(sqrt(self._A.size[0]))
        else: raise AttributeError("SDP object has not been initialized")

    @property
    def m(self):
        """Number of constraints"""
        if self._A is not None :return self._A.size[1]-1
        else: raise AttributeError("SDP object has not been initialized")

    @property
    def V(self):
        """Sparsity pattern (CVXOPT spmatrix)"""
        I,J = misc.ind2sub(self.n,self.I)
        return spmatrix(0.,I,J,(self.n,self.n))

    @property
    def I(self):
        doc = "Sparsity pattern (CVXOPT integer matrix with absolute indices)"
        if self._I is not None: return self._I
        else: self._agg_sparsity()
        return self._I

    @property
    def nnz(self):
        """Number of nonzeros in lower triangle of sparsity pattern"""
        if self._I is not None: return len(self._I)
        else: self._agg_sparsity()
        return len(self._I)

    @property
    def issparse(self):
        """Returns 'True' if the aggregate sparsity density is below 0.5"""
        return (len(self.I) <= 0.5*(self.n*(self.n+1)/2))

    @property
    def b(self):
        """SDP data: vector b"""
        if self._b is not None: return self._b
        else: raise AttributeError("SDP object has not been initialized")

    @property
    def ischordal(self):
        """Is True if the sparsity pattern is chordal, False otherwise."""
        if self._ischordal == None:
            V = +self.V
            p = chompack.maxcardsearch(V)
            if chompack.peo(V,p): self._ischordal = True
            else: self._ischordal = False
        return self._ischordal

    def get_A(self,i=None):
        """Returns A if i is None, otherwise returns Ai as spmatrix"""
        if i==None and self._A: return self._A
        elif i>=0 and i<=self.m and self._A:
            r = self._A.CCS[1][self._A.CCS[0][i]:self._A.CCS[0][i+1]]
            v = self._A.CCS[2][self._A.CCS[0][i]:self._A.CCS[0][i+1]]
            if isinstance(v,float): I = [r/self.n]; J = [r%self.n]
            else: I,J = misc.ind2sub(self.n,r)
            return spmatrix(v,I,J,(self.n,self.n))
        elif self._A: raise ValueError("index is out of range")
        else: raise AttributeError("SDP object has not been initialized")
    A = property(get_A,doc="SDP data: sparse matrix with columns A0,A1,..,Am")

    def _read_sdpa(self,fname):
        """Reads sparse SDPA data file and assigns data to class properties A and b"""
        fp,ext = os.path.splitext(fname)
        if ext == '.bz2' and not os.path.isfile(fp):
            import bz2
            fc = open(fname,'rb')
            fo = open(fp,'w')
            fo.write(bz2.decompress(fc.read()))
            fc.close()
            fo.close()
            #f = open(fp)
            self._A,self._b,self._blockstruct = misc.sdpa_read(fp,neg=True)
            #f.close()
            os.remove(fp)
        else:
            #f = open(fname)
            self._A,self._b,self._blockstruct = misc.sdpa_read(fname,neg=True)
            #f.close()

    def write_sdpa(self,fname=None,compress=False):
        """Writes problem data to sparse SDPA data file"""
        import bz2
        if not self._blockstruct: self._blockstruct = matrix(self.n,(1,1))

        if fname is None:
            fname = self._pname

        fname += ".dat-s"
        if os.path.isfile(fname):
            raise IOError("file %s already exists" % (fname))
        if compress and os.path.isfile(fname+".bz2"):
            raise IOError("file %s already exists" % (fname+".bz2"))
        misc.sdpa_write(fname,self._A,matrix(self._b),self._blockstruct,neg=True)
        if compress:
            f = open(fname+".bz2", 'wb')
            f.write(bz2.compress(open(fname).read()))
            f.close()
            os.remove(fname)

    def _load(self,fname):
        """Load SDP data from 'pickle' file"""
        import bz2, pickle
        fp,ext = os.path.splitext(fname)
        if ext == '.bz2':
            D = pickle.loads(bz2.decompress(open(fname).read()))
        elif ext == '.pkl':
            f = open(fname,'r')
            D = pickle.load(f)
            f.close()
        else:
            raise IOError("unknown extension '%s' " % (ext))
        self._A = D['A']; self._b = D['b']; self._X0 = D['X0']
        self._y0 = D['y0']; self._S0 = D['S0']; self._pname = D['pname']

    def save(self,fname=None,compress=False):
        """Save SDP data to 'pickle' file"""
        import bz2, pickle
        if fname is None:
            fname = self._pname

        if not compress:
            fname += ".pkl"
        else:
            fname += ".pkl.bz2"

        if not os.path.isfile(fname):
            f = open(fname,'w')
        else: raise IOError("file %s already exists" % (fname))

        D = {'A':self._A,'b':self._b,'X0':self._X0,
             'y0':self._y0,'S0':self._S0,'pname':self._pname}
        if not compress:
            pickle.dump(D,f)
        else:
            f.write(bz2.compress(pickle.dumps(D)))
        f.close()

    def _agg_sparsity(self):
        """Generates aggregate sparsity vector for problem data"""
        if self._A: self._I = matrix(list(set(self._A.I)))
        else: raise AttributeError("SDP object has not been initialized")

    def get_nnz(self,i=None):
        """Returns (m+1)-by-1 vector with number of nonzeros in lower triangle of A0,...,Am"""
        if self._A: cptr = self._A.CCS[0]
        else: raise AttributeError("SDP object has no data")
        if i == None:
            u = matrix(0,(len(cptr)-1,1))
            for i in range(len(cptr)-1):
                u[i] = cptr[i+1] - cptr[i]
            return u
        elif i>=0 and i<= self.m: return cptr[i+1]-cptr[i]
        else: raise ValueError("index out of range")
    nnzs = property(get_nnz,doc="Vector with number of nonzeros in lower triangle of A0,A1,...,Am")

    def get_nzcols(self,i=None):
        """Returns (m+1)-by-1 vector with the number of nonzero columns in A1,...,Am"""
        if self._A: nzc = misc.nzcolumns(self._A)
        else: raise AttributeError("SDP object has no data")
        if i == None: return nzc
        elif i>0 and i<= self.m: return nzc[i-1]
        else: raise ValueError("index out of range")
    nzcols = property(get_nzcols,doc="Vector with number of nonzero columns in A1,..,Am")

    def solve_esd(self,kktsolver='chol',scaling='primal',primalstart=None,dualstart=None,p=None):
        """
        Passes problem data to SMCP self-dual solver.

        ARGUMENTS
        kktsolver String, 'chol' (default) or 'qr'

        scaling   String, 'primal' (default) or 'dual'

        RETURNS
        sol       Dictionary
        """

        return solvers.chordalsolver_esd(self.A,self.b,kktsolver=kktsolver,\
                                             scaling=scaling, \
                                             primalstart=primalstart, \
                                             dualstart=dualstart,p=p)

    def solve_feas(self,kktsolver='chol',scaling='primal',\
                       primalstart=None,dualstart=None):
        """
        Passes problem data to SMCP feasible start solver.

        ARGUMENTS
        kktsolver String, 'chol' (default) or 'qr'

        scaling   String, 'primal' (default) or 'dual'

        RETURNS
        sol       Dictionary
        """

        return solvers.chordalsolver_feas(self.A,self.b,kktsolver=kktsolver,\
                                              scaling=scaling, \
                                              primalstart=primalstart, \
                                              dualstart=dualstart)

    def solve_phase1(self,kktsolver='chol',MM = 1e5):
        """
        Solves primal Phase I problem using the feasible
        start solver.

        Returns primal feasible X.

        """
        from cvxopt import cholmod, amd
        k = 1e-3

        # compute Schur complement matrix
        Id = [i*(self.n+1) for i in range(self.n)]
        As = self._A[:,1:]
        As[Id,:] /= sqrt(2.0)

        M = spmatrix([],[],[],(self.m,self.m))
        base.syrk(As,M,trans='T')
        u = +self.b

        # compute least-norm solution
        F = cholmod.symbolic(M)
        cholmod.numeric(M,F)
        cholmod.solve(F,u)
        x = 0.5*self._A[:,1:]*u
        X0 = spmatrix(x[self.V[:].I],self.V.I,self.V.J,(self.n,self.n))

        # test feasibility
        p = amd.order(self.V)
        #Xc,Nf = chompack.embed(X0,p)
        #E = chompack.project(Xc,spmatrix(1.0,range(self.n),range(self.n)))
        symb = chompack.symbolic(self.V,p)
        Xc = chompack.cspmatrix(symb) + X0

        try:
            # L = chompack.completion(Xc)
            L = Xc.copy()
            chompack.completion(L)
            # least-norm solution is feasible
            return X0,None
        except:
            pass

        # create Phase I SDP object
        trA = matrix(0.0,(self.m+1,1))
        e = matrix(1.0,(self.n,1))
        Aa = self._A[Id,1:]
        base.gemv(Aa,e,trA,trans='T')
        trA[-1] = MM
        P1 = SDP()
        P1._A = misc.phase1_sdp(self._A,trA)
        P1._b = matrix([self.b-k*trA[:self.m],MM])
        P1._agg_sparsity()

        # find feasible starting point for Phase I problem
        tMIN = 0.0
        tMAX = 1.0
        while True:
            t = (tMIN+tMAX)/2.0
            #Xt = chompack.copy(Xc)
            #chompack.axpy(E,Xt,t)
            Xt = Xc.copy() + spmatrix(t,list(range(self.n)),list(range(self.n)))

            try:
                # L = chompack.completion(Xt)
                L = Xt.copy()
                chompack.completion(L)
                tMAX = t
                if tMAX - tMIN < 1e-1:
                    break
            except:
                tMAX *= 2.0
                tMIN = t

        tt = t + 1.0
        U = X0 + spmatrix(tt,list(range(self.n,)),list(range(self.n)))
        trU = sum(U[:][Id])

        Z0 = spdiag([U,spmatrix([tt+k,MM-trU],[0,1],[0,1],(2,2))])
        sol = P1.solve_feas(primalstart = {'x':Z0}, kktsolver = kktsolver)

        s = sol['x'][-2,-2] - k
        if s > 0:
            return None,P1
        else:

            sol.pop('y')
            sol.pop('s')
            X0 = sol.pop('x')[:self.n,:self.n]\
                - spmatrix(s,list(range(self.n)),list(range(self.n)))
            return X0,sol

    def solve_cvxopt(self,primalstart=None,dualstart=None,\
                         options=None,kktsolver=None):
        """
        Passes problem data to CVXOPT sdp solver.

        ARGUMENTS


        RETURNS
        sol       Dictionary
        """

        from cvxopt import solvers
        if options:
            for key in list(options.keys()):
                solvers.options[key] = options[key]

        if primalstart:
            y0,S0 = dualstart['y'],dualstart['s']
            PS = {'x':y0,'s':[matrix(S0[:])]}
        else:
            PS = None

        if dualstart:
            X0 = primalstart['x']
            DS = {'y':None,'zl':[matrix(X0[:])]}
        else: DS = None

        return solvers.conelp(c = -self._b,
                              G = self._A[:,1:],
                              h = matrix(self.get_A(0)[:]),
                              dims = {'l':0,'q':[],'s':[self.n]},
                              primalstart = PS, dualstart = DS,
                              kktsolver = kktsolver)

def mk_rand(V,cone='posdef',seed=0):
    """
    Generates random matrix U with sparsity pattern V.
    - U is positive definite if cone is 'posdef'.
    - U is completable if cone is 'completable'.
    """
    if not (cone=='posdef' or cone=='completable'):
        raise ValueError("cone must be 'posdef' (default) or 'completable' ")

    from cvxopt import amd
    setseed(seed)
    n = V.size[0]

    U = +V
    U.V *= 0.0
    for i in range(n):
        u = normal(n,1)/sqrt(n)
        base.syrk(u,U,beta=1.0,partial=True)

    # test if U is in cone: if not, add multiple of identity
    t = 0.1; Ut = +U
    p = amd.order(Ut)
    # Vc, NF = chompack.embed(U,p)
    symb = chompack.symbolic(U,p)
    Vc = chompack.cspmatrix(symb) + U
    while True:
        # Uc = chompack.project(Vc,Ut)
        Uc = chompack.cspmatrix(symb) + Ut
        try:
            if cone=='posdef':
                # positive definite?
                # Y = chompack.cholesky(Uc)
                Y = Uc.copy()
                chompack.cholesky(Y)
            elif cone=='completable':
                # completable?
                # Y = chompack.completion(Uc)
                Y = Uc.copy()
                chompack.completion(Y)
            # Success: Ut is in cone
            U = +Ut
            break
        except:
            Ut = U + spmatrix(t,range(n),range(n),(n,n))
            t*=2.0
    return U

class band_SDP(SDP):
    """
    Generates SDP problem data with band structure.

    P = band_SDP(n,m,bw,seed=0)

    PURPOSE
    Generates random data for band SDP.

    ARGUMENTS
    n         integer, matrix order

    m         integer, number of constraints

    bw        integer, bandwidth

    seed      integer, seed for random number generator

    RETURNS
    P         SDP object
    """

    def __init__(self,n,m,bw,seed=0):
        SDP.__init__(self)
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._bw = bw
        self._gen_bandsdp(n,m,bw,seed)
        self._pname = "band_n%i_m%i_bw%i" % (n,m,bw)
        self._agg_sparsity()

    @property
    def bw(self):
        return self._bw

    def _gen_bandsdp(self,n,m,bw,seed):
        """Random data generator for SDP with band structure"""
        setseed(seed)

        I = matrix([ i for j in range(n) for i in range(j,min(j+bw+1,n))])
        J = matrix([ j for j in range(n) for i in range(j,min(j+bw+1,n))])
        V = spmatrix(0.,I,J,(n,n))
        Il = misc.sub2ind((n,n),I,J)
        Id = matrix([i for i in range(len(Il)) if I[i]==J[i]])

        # generate random y with norm 1
        y0 = normal(m,1)
        y0 /= blas.nrm2(y0)

        # generate random S0
        S0 = mk_rand(V,cone='posdef',seed=seed)
        X0 = mk_rand(V,cone='completable',seed=seed)

        # generate random A1,...,Am and set A0 = sum Ai*yi + S0
        A_ = normal(len(I),m+1,std=1./len(I))
        u = +S0.V
        blas.gemv(A_[:,1:],y0,u,beta=1.0)
        A_[:,0] = u
        # compute b
        x = +X0.V
        x[Id] *= 0.5
        self._b = matrix(0.,(m,1))
        blas.gemv(A_[:,1:],x,self._b,trans='T',alpha=2.0)
        # form A
        self._A = spmatrix(A_[:],
                     [i for j in range(m+1) for i in Il],
                     [j for j in range(m+1) for i in Il],(n**2,m+1))

        self._X0 = X0; self._y0 = y0; self._S0 = S0


class mtxnorm_SDP(SDP):
    """
    Generates SDP problem data for matrix norm minimization problem.

    P = mtxnorm_SDP(p,q,r,density=1.0,seed=0)

    PURPOSE
    Generates random data for matrix norm minimization problem:

        minimize   || A1*y1 + A2*y2 + ... + Ar*yr + B ||

    with pxq-matrices B, Ai, i = 1,...,r. This problem can be cast
    as an SDP

        minimize     t
        subject to   [  t*I    (A(y)+B)' ] >= 0
                     [ A(y)+B    t*I     ]

    where A(y) = A1*y1 + ... + Ar*yr. The inequality constraint
    is with respect to the cone of positive semidefinite matrices
    of order n = p + q. The parameter 'density' controls the
    sparsity of A1,...,Ar, where density=1 corresponds to dense
    matrices.

    ARGUMENTS
    p         integer, rows in Ai

    q         integer, cols in Ai

    r         integer, number of data matrices

    density   float, between 0 and 1 (default is 1)

    seed      integer, seed for random number generator

    RETURNS
    P         SDP object

    """

    def __init__(self,p,q,r,density=1.,seed=0):
        SDP.__init__(self)

        if type(density) is float:
            if density > 1 or density <= 0: raise ValueError("density must be between 0 and 1")
            if density == 1.0: self._pname = "mtxnorm_p%i_q%i_r%i" % (p,q,r)
            else: self._pname = "mtxnorm_p%i_q%i_r%i_d%i" % (p,q,r,int(density*1000))
        elif type(density) is list:
            self._pname = "mtxnorm_p%i_q%i_r%i_vd" % (p,q,r)
        else:
            raise TypeError("density must be a float between 0 and 1 or a list of m+1 floats")
        if type(seed) is not int: raise ValueError("seed must be an integer")
        self._p = p; self._q = q; self._density = density
        self._gen_mtxnormsdp(p,q,r,density,seed)
        self._agg_sparsity()

    @property
    def n(self):
        return self._p+self._q

    def _gen_mtxnormsdp(self,p,q,r,d,seed):
        """
        Random data generator for matrix norm minimization SDP.
        """
        setseed(seed)
        n = p + q;

        I = matrix([ i for j in range(q) for i in range(q,n)])
        J = matrix([ j for j in range(q) for i in range(q,n)])
        Il = list(misc.sub2ind((n,n),I,J))

        if type(d) is float:
            nz = min(max(1, int(round(d*p*q))), p*q)
            A = sparse([[spmatrix(normal(p*q,1),Il,[0 for i in range(p*q)],(n**2,1))],
                        [spmatrix(normal(nz*r,1),
                                  [i for j in range(r) for i in random.sample(Il,nz)],
                                  [j for j in range(r) for i in range(nz)],(n**2,r))]])
        elif type(d) is list:
            if len(d) == r:
                nz = [min(max(1, int(round(v*p*q))), p*q) for v in d]
                nnz = sum(nz)
                A = sparse([[spmatrix(normal(p*q,1),Il,[0 for i in range(p*q)],(n**2,1))],
                            [spmatrix(normal(nnz,1),
                                      [i for j in range(r) for i in random.sample(Il,nz[j])],
                                      [j for j in range(r) for i in range(nz[j])],(n**2,r))]])
        else: raise TypeError
        self._A = sparse([[A],[spmatrix(-1.,list(range(0,n**2,n+1)),[0 for i in range(n)],(n**2,1))]])

        self._b = matrix(0.,(r+1,1))
        self._b[-1] = -1.


class robustLS_SDP(SDP):
    """
    Creates SDP object from robust LS problem data.
    """
    def __init__(self,Al,b,pname=None):
        SDP.__init__(self)

        self._A,self._b = misc.robustLS_to_sdp(Al,b)
        self._r = len(Al)-1
        if pname is None:
            self._pname = "robustLS_p%i_q%i_r%i" % (Al[0].size[0],Al[0].size[1],self._r)
        elif type(pname) is str: self._pname = pname
        else: raise TypeError("pname must be a string")
        self._agg_sparsity()

def robustLS_toep(q,r,delta=None,seed=0):
    """
    Random data generator for matrix norm minimization SDP.

    Al,b = robustLS_toep(q,r,delta=0.1,seed=0])

    PURPOSE
    Generates random data for a robust least squares problem
    with Toeplitz structure:

      minimize e_wc(x)^2

    where

      e_wc(x)=sup_{norm(u)<=1} norm((Ab+A1*u1+...+Ar*ur)*x - b),

      Ab,A1,...,Ar \in \reals^{p \times q}

    The parameter r determines the number of nonzero diagonals
    in Ab (1 <= r <= p).

    ARGUMENTS
    q         integer

    r         integer

    delta     float

    RETURNS
    Al        list of CVXOPT matrices, [Ab,A1,...,Ar]

    b         CVXOPT matrix

    """
    setseed(seed)

    p = q+r-1
    Alist = [sparse(misc.toeplitz(matrix([normal(r,1),matrix(0.,(p-r,1))],(p,1)),matrix(0.,(q,1))))]
    x = normal(q,1)
    b = normal(p,1,std=0.1)
    b += Alist[0]*x

    if delta is None:
        delta = 1./r

    for i in range(r):
        Alist.append(spmatrix(delta,range(i,min(p,q+i)),range(min(q,p-i)),(p,q)))

    return robustLS_SDP(Alist, b)


class rand_SDP(SDP):
    """
    Creates SDP object with random data
    """
    def __init__(self,V,m,density=1.0,seed=0):
        SDP.__init__(self)
        self._n = V.size[0]
        self._m = m
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._gen_randsdp(V,m,density,seed)
        self._pname = "rand_n%i_m%i" % (self._n,m)
        self._agg_sparsity()

    def _gen_randsdp(self,V,m,d,seed):
        """
        Random data generator
        """
        setseed(seed)
        n = self._n
        V = chompack.tril(V)
        N = len(V)
        I = V.I; J = V.J
        Il = misc.sub2ind((n,n),I,J)
        Id = matrix([i for i in range(len(Il)) if I[i]==J[i]])

        # generate random y with norm 1
        y0 = normal(m,1)
        y0 /= blas.nrm2(y0)

        # generate random S0, X0
        S0 = mk_rand(V,cone='posdef',seed=seed)
        X0 = mk_rand(V,cone='completable',seed=seed)

        # generate random A1,...,Am
        if type(d) is float:
            nz = min(max(1, int(round(d*N))), N)
            A = sparse([[spmatrix(normal(N,1),Il,[0 for i in range(N)],(n**2,1))],
                        [spmatrix(normal(nz*m,1),
                                  [i for j in range(m) for i in random.sample(Il,nz)],
                                  [j for j in range(m) for i in range(nz)],(n**2,m))]])
        elif type(d) is list:
            if len(d) == m:
                nz = [min(max(1, int(round(v*N))), N) for v in d]
                nnz = sum(nz)
                A = sparse([[spmatrix(normal(N,1),Il,[0 for i in range(N)],(n**2,1))],
                            [spmatrix(normal(nnz,1),
                                      [i for j in range(m) for i in random.sample(Il,nz[j])],
                                      [j for j in range(m) for i in range(nz[j])],(n**2,m))]])
        else: raise TypeError

        # compute A0
        u = +S0.V
        for k in range(m):
            base.gemv(A[:,k+1][Il],matrix(y0[k]),u,beta=1.0,trans='N')
        A[Il,0] = u
        self._A = A

        # compute b
        X0[Il[Id]] *= 0.5
        self._b = matrix(0.,(m,1))
        u = matrix(0.)
        for k in range(m):
            base.gemv(A[:,k+1][Il],X0.V,u,trans='T',alpha=2.0)
            self._b[k] = u


def completion(X):
    """
    Returns maximum-determinant positive definite completion of X
    if it exists, and otherwise an exception is raised.
    """
    from cvxopt.amd import order
    n = X.size[0]
    Xt = chompack.tril(X)
    p = order(Xt)
    # Xc,N = chompack.embed(Xt,p)
    # L = chompack.completion(Xc)
    symb = chompack.symbolic(Xt,p)
    L = chompack.cspmatrix(symb) + Xt
    chompack.completion(L)
    Xt = matrix(0.,(n,n))
    Xt[::n+1] = 1.
    # chompack.solve(L, Xt, mode=0)
    # chompack.solve(L, Xt, mode=1)
    chompack.trsm(L, Xt)
    chompack.trsm(L, Xt, trans = 'T')
    return Xt
