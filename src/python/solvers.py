# Copyright 2010-2014 M. S. Andersen & L. Vandenberghe
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

# DEFAULT OPTIONS
options = {'debug'        :  False,
           'maxiters'     :    100,
           'abstol'       :   1e-6,
           'reltol'       :   1e-6,
           'feastol'      :   1e-8,
           'refinement'   :      1,
           'cholmod'      :  False,
           'order'        :  'AMD',
           'tnzcols'      :    0.1,
           'show_progress':   True,
           'dimacs'       :   True,
           'eta'          :   None,
           'delta'        :    0.9,
           'alpha'        :   1e-1,
           'beta'         :    0.7,
           'minstep'      :   1e-8,
           'lifting'      :   True,
           't0'           :   1e-1,
           'equalsteps'   :   True,
           'prediction'   :   True,
           'step'         :   0.98}

from copy import copy as __ccopy
__options = __ccopy(options)

def chordalsolver_feas(A,b,primalstart=None,dualstart=None,
                       scaling='primal',kktsolver='chol',p=None):
    """
    Chordal SDP solver:

    sol = chordalsolver_feas(A,b[,primalstart[,dualstart[,scaling[,kktsolver]]]])

    PURPOSE
    Solves a pair of primal and dual cone programs

         minimize   c'*x        maximize   b'*y
         subject to A*x = b     subject to A'*y + s = c
                    x in C                 s in K.

    Here C is the cone of symmetric matrices with sparsity
    pattern V that have a positive semidefinite completion,
    and K is the cone of positive semidefinite matrices with
    sparsity pattern V.

    ARGUMENTS
    A         CVXOPT spmatrix with doubles
    b         CVXOPT matrix with doubles

    RETURNS
    sol       dictionary

    """

    from cvxopt import matrix,spmatrix,sqrt,log,blas,lapack,base,cholmod
    #from chompack import sparse, project, tril, symmetrize, hessian, \
    #    perm, embed, maxcardsearch, llt, dot, axpy, copy, scal, \
    #    completion, cholesky, solve, partial_inv, logdet
    from chompack import cspmatrix, symbolic, tril, perm, symmetrize, dot, peo,\
        hessian, completion, cholesky, projected_inverse, llt, trsm, maxcardsearch, merge_size_fill
    from smcp import __version, misc

    from sys import platform
    if platform.startswith('win'):
        from time import clock as time
        def cputime():
            return 0.0
    else:
        from time import time
        from resource import getrusage, RUSAGE_SELF
        def cputime():
            return (getrusage(RUSAGE_SELF).ru_utime
                    +getrusage(RUSAGE_SELF).ru_stime)

    # Start time
    T0wall = time()
    T0 = cputime()

    status = 'unknown'
    # Number of constraints
    m = A.size[1]-1
    # Matrix order
    n = int(sqrt(A.size[0]))

    DEBUG = options['debug']
    if not type(DEBUG)==bool:
        raise TypeError("options['debug'] must be a bool")

    ALPHA = options['alpha']
    if type(ALPHA) is not float or (ALPHA >=0.5 or ALPHA <= 0.0):
        raise TypeError("options['alpha'] must be a float"\
            "in the interal (0.0,0.5)")

    BETA = options['beta']
    if type(BETA) is not float or (BETA >=1.0 or BETA <= 0.0):
        raise TypeError("options['beta'] must be a float"\
            "in the interal (0.0,1.0)")

    MINSTEP = options['minstep']
    if type(MINSTEP) is not float or (MINSTEP < 0.0):
        raise TypeError("options['minstep'] must be a nonnegative float")

    DELTA = options['delta']
    if type(DELTA) is not float or (DELTA <= 0.0 or DELTA >= 1.0):
        raise TypeError("options['delta'] must be a float"\
            "in the interal (0.0,1.0)")

    ETA = options['eta']
    if ETA is not None:
        if type(ETA) is not float or ETA <= 0.0:
            raise TypeError("options['eta'] must be a positive float")
        ETATOL = 0.10*ETA

    t = options['t0']
    if type(t) is not float:
        raise TypeError("options['t0'] must be a positive float")
    elif t <= 0.0:
        raise ValueError("options['t0'] must be a positive float")

    LIFTING = options['lifting']
    if type(LIFTING) is not bool:
        raise TypeError("options['lifting'] must be True or False")

    EQUALSTEPS = options['equalsteps']
    if type(EQUALSTEPS) is not bool:
        raise TypeError("options['equalsteps'] must be True or False")

    PREDICTION = options['prediction']
    if type(PREDICTION) is not bool:
        raise TypeError("options['prediction'] must be True or False")

    STEP = options['step']
    if type(STEP) is not float:
        raise TypeError("options['step'] must be a float")
    elif STEP > 1 or STEP <= 0:
        raise ValueError("options['step'] must be between 0 and 1.")

    MAXITERS = options['maxiters']
    if type(MAXITERS) is not int:
        raise TypeError("options['maxiters'] must be a positive "\
            "integer")
    elif MAXITERS < 1:
        raise ValueError("options['maxiters'] must be positive")

    ABSTOL = options['abstol']
    if type(ABSTOL) is not float and type(ABSTOL) is not int:
        raise TypeError("options['abstol'] must be a scalar")

    RELTOL = options['reltol']
    if type(RELTOL) is not float and type(RELTOL) is not int:
        raise TypeError("options['reltol'] must be a scalar")

    if RELTOL <= 0.0 and ABSTOL <= 0.0 :
        raise ValueError("at least one of options['reltol'] and " \
            "options['abstol'] must be positive")

    FEASTOL = options['feastol']
    if (type(FEASTOL) is not float and type(FEASTOL) is not int):
        raise TypeError("options['feastol'] must be a positive "\
            "scalar")
    elif FEASTOL <= 0.0:
        raise ValueError("options['feastol'] must be positive")

    CHOLMOD = options['cholmod']
    if type(CHOLMOD) is not bool:
        raise TypeError("options['cholmod'] must be bool")

    ORDER = options['order']
    if ORDER == "AMD":
        from cvxopt.amd import order
    elif ORDER == "METIS":
        from cvxopt.metis import order
    else:
        raise ValueError("options['order'] must be 'AMD' or 'METIS'")

    show_progress = options['show_progress']
    if type(show_progress) is not bool:
        raise TypeError("options['show_progress'] must be a bool")

    REFINEMENT = options['refinement']
    if (type(REFINEMENT)) is not int:
        raise TypeError("options['refinement'] must be a nonnegative "\
            "integer")
    elif REFINEMENT < 0:
        raise ValueError("options['refinement'] must be nonnegative ")

    TNZCOLS = options['tnzcols']
    if type(TNZCOLS) is not float:
        raise TypeError("tnzcols must be a float between 0.0 and 1.0")
    elif TNZCOLS > 1 or TNZCOLS < 0:
        raise ValueError("tnzcols must be between 0.0 and 1.0")
    else:
        TNZCOLS = int(n*TNZCOLS)

    DIMACS = options['dimacs']
    if type(DIMACS) is not bool:
        raise TypeError("dimacs must be a bool")

    if not (scaling=='primal' or scaling=='dual'):
        raise ValueError("scaling must be 'primal' or 'dual'")

    def iperm(p):
        """
        Returns inverse permutation vector.
        ip = iperm(p)
        """
        ip = matrix(0,(len(p),1))
        ip[p] = matrix(list(range(len(p))))
        return ip

    # solver does not handle sparse b yet:
    b = matrix(b)

    # aggregate sparsity pattern
    LIa = matrix(list(set(A.I)))
    Ia, Ja = misc.ind2sub(n,LIa)
    Va = tril(spmatrix(0.,Ia,Ja,(n,n)))
    del Ia,Ja,LIa

    if kktsolver=='chol':
        ### Find permutation
        Nz = misc.nzcolumns(A)
        pm,Ns = misc.matperm(Nz,TNZCOLS)
        # Permute first part of pm (Decorate-Sort-Undecorate)
        if (Ns < m):
            L = list(zip(A.CCS[0][pm[:m-Ns]+2] - A.CCS[0][pm[:m-Ns]+1],pm[:m-Ns]))
            L.sort(reverse=True)
            pm[0:m-Ns] = matrix([l for _,l in L])
        # Permute second part of pm (Decorate-Sort-Undecorate)
        if ( Ns > 0 ):
            L = list(zip(A.CCS[0][pm[m-Ns:m]+2] - A.CCS[0][pm[m-Ns:m]+1],pm[m-Ns:m]))
            #L = [(A.CCS[0][pm[i]+2]-A.CCS[0][pm[i]+1] + Nz[pm[i]],pm[i]) for i in xrange(m-Ns,m)]
            L.sort(reverse=True)
            pm[m-Ns:m] = matrix([l for _,l in L])
        del Nz
        # Permute b
        b = +b[pm]
    else:
        Ns = 0
        pm = matrix(range(m))

    bmax, ii = max(list(zip(abs(b),list(range(len(b))))))

    if p is None: p = order(Va)

    # make embeddings
    if CHOLMOD: # hack for using cholmod symbolic
        ORDER+="+CHOLMOD"
        Ve = Va + spmatrix([float(i+1) for i in range(n)],list(range(n)),list(range(n)),(n,n))
        F = cholmod.symbolic(Ve,p=p)
        cholmod.numeric(Ve,F)
        f = cholmod.getfactor(F)
        fd = [(v,i) for i,v in enumerate(f[:n**2:n+1])]
        fd.sort()
        ip = matrix([i for _,i in fd])
        p = iperm(ip)
        Vp = +f
        symb = symbolic(Vp)
        Vp.V = matrix([float(v) for v in range(len(Vp))])
        Ve = tril(perm(symmetrize(Vp),ip))
        lp = iperm([int(v) for v in Ve.V])
        CHORDAL = False
    else:
        pmcs = maxcardsearch(Va)
        if peo(Va,pmcs):
            CHORDAL = True
            p = pmcs
            symb = symbolic(Va,pmcs)
        else:
            CHORDAL = False
            symb = symbolic(Va,p)

        Ve = symb.sparsity_pattern(reordered=False, symmetric=False)
        Ve.V = matrix([float(v) for v in range(len(Ve))])
        Vp = tril(perm(symmetrize(Ve),p))
        lp = matrix([int(v) for v in Vp.V])
        symb = symbolic(Vp)
        ip = iperm(p)

    # extract vector space index permutation and Vp subscripts
    LI = misc.sub2ind((n,n),Ve.I,Ve.J)
    Ip,Jp = Vp.I,Vp.J

    # Extract Av and c
    # c = A[LI[lp],0] ## <-- unpacks A[:,0]
    c = A[:,0][LI[lp]] ## <-- does not unpack A[:,0]
    # Av = A[LI[lp],pm+1]
    if len(A) < 0.005*len(Va)*m or n > 15000: # do not unpack A
        Ve.V = matrix([float(v) for v in iperm(lp)])
        IJV = []; tmp = 0
        for j in pm:
            j1,j2 = A.CCS[0][j+1],A.CCS[0][j+2]
            #Ltmp = [(int(Ve[A.CCS[1][k]]),tmp,A.CCS[2][k]) for k in xrange(j1,j2)]
            Ltmp = list(zip([int(k) for k in Ve[A.CCS[1][j1:j2]]],
                       [tmp for i in range(j2-j1)],
                       [k for k in A.CCS[2][j1:j2]]))
            Ltmp.sort()
            IJV += Ltmp
            tmp+=1
        It,Jt,Vt = list(zip(*IJV))
        Av = spmatrix(Vt,It,Jt,(len(Vp),m))
        del It,Jt,Vt,IJV
    else: # unpack A
        Av = A[LI[lp],pm+1]

    # Indices of diagonal elements in Av
    Id = matrix([i for i in range(len(Vp)) if Ip[i] == Jp[i]])

    # Check that number of constraints is leq. number of nonzeros
    if m > len(Vp):
        raise ValueError("more constraints than nonzeros")

    # if Cholesky solver, extract nonzero column indices
    if kktsolver=='chol':
        Kl = []
        for j in range(m-Ns,m):
            k = []
            for l in Av.CCS[1][Av.CCS[0][j]:Av.CCS[0][j+1]]:
                k.append(Ip[l])
                k.append(Jp[l])
            Kl.append(list(set(k)))

    # Convert A0 to chompack
    Vp.V = matrix(c)
    C = cspmatrix(symb) + Vp

    def Amap(X,i=None):
        Xp = X.spmatrix(reordered=False,symmetric=False)
        misc.scal_diag(Xp,Id,0.5)
        v = matrix(0.,(m,1))
        if i is None:
            base.gemv(Av,Xp.V,v,trans='T',alpha=2.0)
            return v
        else:
            base.gemv(Av[:,i],Xp.V,v,trans='T',alpha=2.0)
            return v[0]

    def Aadj(y):
        v = matrix(0.,(len(Vp),1));
        base.gemv(Av,y,v,trans='N')
        Vp.V = v;
        return cspmatrix(symb)+Vp

    def Omega(Lsh,Ls,gap):
        """
        Computes

        Omega(X,S) = phip(X) + phid(S) + n log(<X,S>/n) + n

        where Ls is the Cholesky factor of S and Lsh is the Cholesky
        factor of Shat = X^-1.
        """
        return 2.0*sum(log(Lsh.diag())) - 2.0*sum(log(Ls.diag())) + n*log(gap/n)

    # Norm of b and C
    resy0 = max(1,blas.nrm2(b))
    resx0 = max(1,sqrt(dot(C,C)))

    def kkt_res(x,y,bx,by):
        r = x.copy()
        #hessian(L,Y,r,inv=True,adj=True)
        #hessian(L,Y,r,inv=True)
        hessian(L,Y,r,inv=True,adj=None)
        if scaling=='primal':
            blas.scal(-1.0/t,r.blkval)
        else:
            blas.scal(-t,r.blkval)
        r += Aadj(y)-bx
        return r, Amap(x)-by

    def kkt_qr(L,Y):
        Ac_ = [Aj.copy() for Aj in Ac]
        hessian(L,Y,Ac_)
        for j in range(len(Ac_)):
            Aj = Ac_[j].spmatrix(reordered=False, symmetric=False)
            At[:,j] = Aj.V
        # scale diagonal elements
        At[Id,:] /= sqrt(2.)

        # compute QR factorization of At
        try:
            lapack.geqrf(At,tq)
        except ArithmeticError:
            print("*** Factorization failed")
            status = 'unknown'
            return None

        def solve_(bx,by, tt = None):
            """
            Solves the system:

              [  -kk*H  A' ][ x ] = [ bx ]
              [   A     0  ][ y ]   [ by ]

            """
            if tt is None:
                tt = t
            if scaling=='primal':
                kk = 1.0/tt
            else:
                kk = tt

            # compute x,y
            r1 = bx.copy()
            hessian(L,Y,[r1])
            r1 = r1.spmatrix(reordered=False, symmetric=False).V
            r1[Id] /= sqrt(2.0)
            x = +r1

            lapack.ormqr(At,tq,x,trans='T')
            x[m:] = 0.

            r2 = +by
            lapack.trtrs(At[:m,:],r2,uplo='U',trans='T')
            x[:m] += 0.5*kk*r2
            y = +x[:m]
            lapack.trtrs(At[:m,:],y,uplo='U')

            lapack.ormqr(At,tq,x)
            Vp.V = x-r1
            misc.scal_diag(Vp,Id,sqrt(2.0))
            x = cspmatrix(symb) + Vp
            hessian(L,Y,[x],adj=True)
            blas.scal(1.0/kk,x.blkval)

            if DEBUG:
                r,rr = kkt_res(x,y,bx,by)
                r = sqrt(dot(r,r))/max(1,sqrt(dot(bx,bx)))
                rr = blas.nrm2(rr)/max(1,blas.nrm2(by))
                print("\033[1;33m KKTsolver: %.2e  %.2e \033[0m" % (r,rr))

            return x,y
        return solve_

    def kkt_chol(L,Y):
        # Compute H (lower triangle)
        for j in range(0,m-Ns):
            At = misc.Av_to_spmatrix(Av,Ip,Jp,j,n)
            U = cspmatrix(symb) + At
            #hessian(L,Y,[U]); hessian(L,Y,[U],adj=True)
            hessian(L,Y,U,inv=True,adj=None)
            At = U.spmatrix(reordered=False, symmetric=False)
            misc.scal_diag(At,Id,0.5)
            base.gemv(Av[:,j:m],At.V,tmp,trans='T',alpha=2.0)
            H[j:m,j] = tmp[:m-j]

        for j in range(0,Ns):
            V = matrix(spmatrix(1.,Kl[j],range(len(Kl[j])),(n,len(Kl[j]))))
            trsm(L,V)
            trsm(L,V,trans='T')
            # compute column j of H (lower triangle only)
            kkl = matrix(0,(n,1))
            for k1,k2 in enumerate(Kl[j]): kkl[k2] = k1
            misc.SCMcolumn2(H,Av,V,Ip,Jp,kkl,m-Ns+j)

        # Cholesky factorization: H = L*L'
        try:
            lapack.potrf(H)
        except ArithmeticError:
            print("*** Factorization failed")
            status = 'unknown'
            return None

        def solve_(bx,by, tt = None):
            """
            Solves the system:

              [ -kk*H  A' ][ x ] = [ bx ]
              [   A    0  ][ y ]   [ by ]

            """
            if tt is None:
                tt = t
            if scaling=='primal':
                kk = 1.0/tt
            else:
                kk = tt

            r1 = bx.copy()
            #hessian(L,Y,[r1])
            #hessian(L,Y,[r1],adj=True)
            hessian(L,Y,r1,inv=False,adj=None)
            y = kk*by + Amap(r1)
            lapack.potrs(H,y)

            x = Aadj(y)-bx
            #hessian(L,Y,[x])
            #hessian(L,Y,[x],adj=True)
            hessian(L,Y,[x],inv=False,adj=None)
            blas.scal(1.0/kk,x.blkval)

            if DEBUG:
                r,rr = kkt_res(x,y,bx,by)
                r1 = sqrt(dot(r,r))/max(1,sqrt(dot(bx,bx)))
                r2 = blas.nrm2(rr)/max(1,blas.nrm2(by))
                print("\033[1;33m KKTsolver: %.2e  %.2e \033[0m" % (r1,r2))
            return x,y
        return solve_

    if kktsolver == 'chol':
        factor = kkt_chol
        SolvStr = "Chol."
        # Allocate storage for SCM
        H = matrix(0.,(m,m))
        # Allocate storage for tmp
        if Ns<m: tmp = matrix(0.,(m,1))
    elif kktsolver == 'qr':
        factor = kkt_qr
        SolvStr = "QR"
        # Convert A_1,...,A_m to chompack
        Ac = [cspmatrix(symb)+misc.Av_to_spmatrix(Av,Ip,Jp,j,n) for j in range(m)]
        # Allocate storage for At
        At = matrix(0.,(len(Vp),m))
        # Allocate storage for tq
        tq = matrix(0.,(m,1))

    if LIFTING:
        SolvStr += ",lifting"
    if PREDICTION:
        SolvStr += ",prediction"

    def kktsolver(L,Y):
        return factor(L,Y)

    def print_head():
        print("%-20s Barrier method, %s scaling (%s)" % (__version,scaling,SolvStr))
        print("-------------------------------------------------------------------------------")
        print("SDP var. size:       %i " % (n))
        print("Constraints:         %i (%i|%i)" % (m,m-Ns,Ns))
        if CHORDAL: ChStr = "Chordal"
        else: ChStr = "Nonchordal"
        print("Aggregate sparsity:  %-14s NNZ(tril(V)) = %7i" % (ChStr,len(Va)))
        if not CHORDAL:
            print("Embedding:           %-14s       NNZ(L) = %7i" % (ORDER,len(Vp)))
        print("-------------------------------------------------------------------------------")
        print(" it  pcost       dcost      gap     pres    dres    ntdecr  Omega   pstep dstep")

    def print_exit_info():
        if status=='optimal' or status=='unknown':
            if pcost is not None:
                print("   Primal objective:                % .8e" % (pcost))
            if dcost is not None:
                print("   Dual objective:                  % .8e" % (dcost))
        if gap is not None:
            print("   Gap:                             % .8e" % (gap))
        if relgap is not None:
            print("   Relative gap:                    % .8e" % (relgap))
        if pres is not None:
            print("   Primal infeasibility:            % .8e" % (pres))
        if dres is not None:
            print("   Dual infeasibility:              % .8e" % (dres))
        if not iter==0:
            print("   Iterations:                       %i"   % (iter))
            if not platform.startswith('win'):
                print("   CPU time:                         %.2f" % (Tcpu))
                print("   CPU time per iteration:           %.2f" % (Tcpu/iter))
            print("   Real time:                        %.2f" % (Twall))
            print("   Real time per iteration:          %.2f\n" % (Twall/iter))
        if DIMACS:
            print("   DIMACS:  %.2e %.2e %.2e %.2e %.2e %.2e\n" % tuple(DIMACS_err))

    def linesearch(X, dx, S, ds, a = 1.0):
        N = 8
        # PRIMAL STEP
        gMIN = MINSTEP; gMAX = 1.0
        for i in range(N):
            gam = (gMIN + gMAX)/2.0
            Xt = X.copy() + gam*dx
            try:
                Lt = Xt.copy()
                completion(Lt)
                gMIN = gam
                gam_ = gam
            except:
                gMAX = gam
                gam_ = gMIN
        pstep = gam_
        # DUAL STEP
        gMIN = MINSTEP; gMAX = 1.0
        for i in range(N):
            gam = (gMIN + gMAX)/2.0
            St = S.copy() + gam*ds
            try:
                Lt = St.copy()
                cholesky(Lt)
                gMIN = gam
                gam_ = gam
            except:
                gMAX = gam
                gam_ = gMIN
        dstep = gam_
        return a*pstep,a*dstep

    def linesearch_backtrack(X,dx,S,ds,gstart = 1.0):
        gam = gstart
        while gam > MINSTEP:
            Xt = X.copy() + gam*dx
            St = S.copy() + gam*ds
            try:
                cholesky(St)
                completion(Xt)
                break
            except:
                gam *= BETA
        return gam,gam

    def linesearch_Omega(X,dx,S,ds,eta):
        gMIN = MINSTEP; gMAX = 1.0
        for i in range(8):
            gam = (gMIN + gMAX)/2.0
            Xt = X.copy() + gam*dx
            St = S.copy() +gam*ds
            try:
                Lt = Xt.copy()
                completion(Lt)
                Lst = St.copy()
                cholesky(Lst)
                gapt = dot(St,Xt)
                gam_ = gam
                Ot = Omega(Lt,Lst,gapt)
                if Ot - eta > ETATOL:
                    gMAX = gam
                elif Ot - eta < -ETATOL:
                    gMIN = gam
                else:
                    break
            except:
                gMAX = gam
                gam_ = None
        if gam_: return gam_
        else: return gMIN

    # Starting point
    if primalstart is not None:
        X = cspmatrix(symb) + tril(perm(symmetrize(tril(primalstart['x'])),p))
        if blas.nrm2(b-Amap(X))/resy0 > 1e-8:
            raise ValueError("infeasible primal starting point")
        try:
            L = X.copy()
            completion(L)
        except:
            raise ValueError("infeasible primal starting point")
    else:
        X = None
    if dualstart is not None:
        if 'y' in dualstart and 's' in dualstart:
            y = dualstart['y'][pm]
            S = cspmatrix(tril(perm(symmetrize(tril(dualstart['s'])),p)))
            resx = Aadj(-y) + C - S
            if sqrt(dot(resx,resx))/resx0 > 1e-8:
                raise ValueError("infeasible dual starting point")
        elif 'y' in dualstart and 's' not in dualstart:
            y = dualstart['y'][pm]
            S = Aadj(-y) + C
        try:
            L = S.copy()
            cholesky(L)
        except:
            raise ValueError("infeasible dual starting point")
    else:
        y = None; S = None

    if primalstart is None and dualstart is None:
        # heuristics for finding feasible starting point

        # look for strictly feasible primal point
        Xt = cspmatrix(symb) + spmatrix(1.,list(range(n)),list(range(n)))
        Lt = Xt.copy()
        completion(Lt)
        fI = kktsolver(Lt,Xt)
        X0,nu = fI(cspmatrix(symb),b)
        try:
            L = X0.copy()
            completion(L)
            X = X0
            print("Primal least-norm solution is feasible.")
        except:
            trA = matrix(0.0,(m,1))
            base.gemv(Av[Id,:],matrix(1.,(n,1)) , trA, trans = 'T')
            Xb,nu = fI(cspmatrix(symb),trA)
            Xb -= Xt
            try:
                L = Xb.copy()
                completion(L)
                # Xb is in cone!
                gam = 2.0
                while True:
                    X = X0.copy() + gam*Xb
                    try:
                        Lt = X.copy()
                        cholesky(Lt)
                        break
                    except:
                        gam *= 2
                print("Feasible primal solution found.")
            except:
                try:
                    blas.scal(-1.0,Xb.blkval)
                    L = Xb.copy()
                    completion(L)
                    gam = 2.0
                    while True:
                        X = X0.copy() + gam*Xb
                        try:
                            L = X.copy()
                            cholesky(L)
                            break
                        except:
                            gam *= 2
                    print("Feasible primal solution found.")
                except:
                    X = None

        # look for strictly feasible dual point
        nu,y = fI(C,matrix(0.0,(m,1)))
        S = Aadj(-y) + C
        try:
            L = S.copy()
            cholesky(L)
            print("Dual least-squares solution is feasible.")
        except:
            nu,y = fI(cspmatrix(symb)+spmatrix(-1.,list(range(n)),list(range(n))),\
                          matrix(0.0,(m,1)))
            Sh = Aadj(-y)
            try:
                L = Sh.copy()
                cholesky(L)
                # Sh is in cone!
                gam = 1.4
                while True:
                    S = Aadj(-y)+C
                    try:
                        Lt = S.copy()
                        cholesky(Lt)
                        break
                    except:
                        y *= gam
                print("Feasible dual solution found.")
            except:
                S = None
                y = None

    if y is None and X is None:
        raise ValueError("could not find a feasible starting point"\
            " (solve Phase I problem instead)")
    elif X is None and scaling == 'primal':
        print("Switching to dual scaling.")
        scaling = 'dual'
    elif S is None and scaling == 'dual':
        print("Switching to primal scaling.")
        scaling = 'primal'

    if show_progress:
        print_head()

    # initial gap:
    gap = n/t

    if scaling == 'primal':
        dres = None
        dcost = None
    else: # scaling == 'dual'
        pres = None
        pcost = None
    CENTER = True

    for iter in range(1,MAXITERS+2) :
        ## RESIDUALS AND CONVERGENCE STATISTICS
        if scaling == 'primal':
            resy = b-Amap(X)
            pres = blas.nrm2(resy)/resy0
            pcost = dot(C,X)
        else: # scaling == 'dual'
            resx = Aadj(-y)+C-S
            dres = sqrt(dot(resx,resx))/resx0
            dcost = blas.dot(b,y)

        # compute relgap
        if pcost is not None and pcost < 0.0:
            relgap = gap / -pcost
        elif dcost is not None and dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None

        ## Stopping criteria
        if iter == MAXITERS+1:
            # Max. number of iterations reached
            if show_progress:
                print("Terminated (maximum number of iterations reached).")
            status = 'unknown'
            break
        elif dres is not None and pres is not None:
            if pres < FEASTOL and dres < FEASTOL and (gap < ABSTOL):
                if show_progress:
                    print("Optimal solution found.")
                status = 'optimal'
                break
            elif relgap is not None:
                if pres < FEASTOL and dres < FEASTOL and (relgap < RELTOL):
                    if show_progress:
                        print("Optimal solution found.")
                    status = 'optimal'
                    break

        if scaling == 'primal':
            try:
                L = X.copy()
                completion(L)
            except:
                if show_progress:
                    print("*** Completion failed.")
                status = 'unknown'
                break
            Y = X.copy()
        else: # scaling == 'dual'
            try:
                L = S.copy()
                cholesky(L)
            except:
                if show_progress:
                    print("*** Factorization of S failed.")
                status = 'unknown'
                break
            Y = L.copy()
            projected_inverse(Y)

        f = kktsolver(L,Y)
        if f is None: break

        if CENTER:
            if scaling == 'primal':
                # Centering:
                # [ -1/t Hphip(x)   A^T ][ dx  ] = [ C + 1/t Gphip(x) ]
                # [   A             0   ][ lam ] = [      b-Ax        ]
                #
                Shat = L.copy()
                llt(Shat)
                bx = C.copy() - (1.0/t)*Shat
                by = b-Amap(X)
                dx,lam = f(bx,by)
                # iterative refinement
                for ii in range(REFINEMENT):
                    r1,r2 = kkt_res(dx,lam,bx,by)
                    ddx,dlam = f(r1,r2)
                    dx -= ddx
                    lam -= dlam

                # compute Newton decr.
                du = dx.copy()
                hessian(L,Y,[du],inv=True,adj=True)
                ntdecr = sqrt(dot(du,du))
                del du

                if ntdecr > DELTA:
                    if ntdecr >= 1.0:
                        # centering step: backtracking line search
                        gam = 1.0
                        logdetL = sum(log(L.diag()))
                        tdcdx = t*dot(C,dx)
                        #val = t*dot(C,dx) + ALPHA*ntdecr**2
                        while gam > MINSTEP:
                            Xt = X.copy() + gam*dx
                            val = tdcdx + gam*ALPHA*ntdecr**2
                            try:
                                Lt = Xt.copy()
                                completion(Lt)
                                if gam*val < 2*(logdetL - sum(log(Lt.diag()))):
                                    break
                                else:
                                    gam *= BETA
                            except:
                                gam *= BETA

                        # update X
                        X = Xt
                    else:
                        # update X (full step)
                        X += dx
                        gam = 1.0

                    stype = 'c'

                else:
                    # Lifting
                    if LIFTING:
                        X -= dx
                        dxL = dx
                    else:
                        X += dx
                    y = lam
                    S = Aadj(-y) + C
                    CENTER = False

            else: # scaling == 'dual'
                # Centering:
                # [ -(1/t Hphid(s))^-1   A^T ][ nu ] = [-s ]
                # [        A              0  ][ dy ] = [ b ]
                bx = S.copy()
                blas.scal(-1.0, bx.blkval)
                by = b
                nu,dy = f(bx,by)
                # iterative refinement
                for ii in range(REFINEMENT):
                    r1,r2 = kkt_res(nu,dy,bx,by)
                    dnu,ddy = f(r1,r2)
                    nu -= dnu
                    dy -= ddy

                # compute Newton decr.
                du = Aadj(dy)
                hessian(L,Y,[du])
                ntdecr = sqrt(dot(du,du))
                del du

                if ntdecr > DELTA:
                    if ntdecr >= 1.0:
                        # centering step: backtracking line search
                        gam = 1.0
                        logdetL = sum(log(L.diag()))
                        ddyb = - t*blas.dot(dy,b)
                        #val = ALPHA*ntdecr**2 - t*blas.dot(dy,b)
                        while gam > MINSTEP:
                            yt = y + gam*dy
                            St = Aadj(-yt) + C
                            val = ddyb + gam*ALPHA*ntdecr**2
                            try:
                                Lt = St.copy()
                                cholesky(Lt)
                                if gam*val < 2*(sum(log(Lt.diag())) - logdetL) :
                                    break
                                else:
                                    gam *= BETA
                            except:
                                gam *= BETA

                        # update S,y
                        S = St
                        y = yt
                    else:
                        # update S,y (full step)
                        y += dy
                        S = Aadj(-y) + C
                        gam = 1.0

                    stype = 'c'

                else:
                    # Lifting
                    if LIFTING:
                        y -= dy
                        S = Aadj(-y) + C
                        dyL = dy
                    else:
                        y += dy
                        S = Aadj(-y) + C
                    X = nu
                    CENTER = False

        if not CENTER:

            # Approximate tangent step (primal scaling):
            # [ -1/t Hphip(w)   A^T ][ dx  ] = [  s   ]
            # [   A             0   ][ dy  ] = [ b-Ax ]
            #
            # Approximate tangent step (dual scaling):
            # [ -(1/t Hphid(w))^{-1}   A^T ][ dx  ] = [  s   ]
            # [        A               0   ][ dy  ] = [ b-Ax ]
            bx = S.copy()
            by = b - Amap(X)
            dx,dy = f(bx, by)
            # iterative refinement
            for ii in range(REFINEMENT):
                r1,r2 = kkt_res(dx,dy,bx,by)
                ddx,ddy = f(r1,r2)
                dx -= ddx
                dy -= ddy
            ds = Aadj(-dy)

            if ETA is not None:  # STAY IN OMEGA-NEIGHBORHOOD
                # line search (bisection)
                gam = linesearch_Omega(X,dx,S,ds,ETA)
                # update X,y,S
                X += gam*dx
                y += gam*dy
                S = Aadj(-y) + C
                pstep = gam; dstep = gam   # <-- for iter. info

            else:
                # "EXACT" LINE SEARCH ALONG APPROX. TANGENT DIRECTION
                pstep,dstep = linesearch(X,dx,S,ds,STEP)
                if EQUALSTEPS:
                    pstep = min(pstep,dstep)
                    dstep = pstep

                Xt = X.copy() + pstep*dx
                yt = y + dstep*dy
                St = Aadj(-yt) + C

                if not PREDICTION:
                    # Update X,y,S
                    X = Xt
                    S = St
                    y = yt

                else: # PREDICTION

                    # Compute predicted gap
                    gapt = dot(Xt,St)

                    if LIFTING:
                        # undo lifting
                        if scaling == 'primal':
                            X += dxL
                        else: # scaling == 'dual'
                            blas.axpy(dyL,y)
                            S = Aadj(-y) + C

                    if DEBUG: print(" -- predicted gap: %.2e" %(gapt))
                    # update t
                    t = n/gapt

                    if scaling == 'primal':
                        # step:
                        # [ -1/t Hphip(x)   A^T ][ dx  ] = [ s - (1/t)*shat ]
                        # [    A            0   ][ dy  ] = [ b-Ax           ]
                        bx = S.copy()-(1.0/t)*Shat
                        by = b - Amap(X)
                    else: # scaling == 'dual'
                        # step:
                        # [ -t Hphid(s)^{-1}   A^T ][ dx  ] = [ t*Hphid(s)^{-1}*x - s ]
                        # [    A               0   ][ dy  ] = [ b-Ax                  ]
                        bx = X.copy()
                        #hessian(L,Y,[bx],inv=True,adj=True)
                        #hessian(L,Y,[bx],inv=True)
                        hessian(L,Y,bx,inv=True,adj=None)
                        blas.scal(t,bx.blkval)
                        bx -= S
                        by = b - Amap(X)

                    dx,dy = f(bx, by, t)
                    # iterative refinement
                    for ii in range(REFINEMENT):
                        r1,r2 = kkt_res(dx,dy,bx,by)
                        ddx,ddy = f(r1,r2,t)
                        dx -= ddx
                        dy -= ddy
                    ds = Aadj(-dy)

                    # NTDECR
                    if scaling == 'primal':
                        du = dx.copy()
                        hessian(L,Y,[du],inv=True,adj=True)
                        ntdecr = sqrt(dot(du,du))
                        del du
                    else: # scaling == 'dual'
                        du = Aadj(dy)
                        hessian(L,Y,[du])
                        ntdecr = sqrt(dot(du,du))
                        del du

                    # LINE SEARCH
                    if scaling == 'primal':
                        if ntdecr >= 1.0:
                            # centering step: backtracking line search
                            gam = 1.0
                            logdetL = sum(log(L.diag()))
                            tdcdx = t*dot(C,dx)
                            #val = t*dot(C,dx) + ALPHA*ntdecr**2
                            while gam > MINSTEP:
                                Xt = X.copy() + gam*dx
                                val = tdcdx + gam*ALPHA*ntdecr**2
                                try:
                                    Lt = Xt.copy()
                                    completion(Lt)
                                    if gam*val < 2*(logdetL - sum(log(Lt.diag()))):
                                        break
                                    else:
                                        gam *= BETA
                                except:
                                    gam *= BETA

                            # update X
                            pstep = gam
                            X = Xt
                        else:
                            # (full step)
                            pstep = 1.0
                            X += dx

                        gam = 1.0
                        while True:
                            yt = y + gam*dy
                            St = Aadj(-yt) + C
                            try:
                                Lt = St.copy()
                                cholesky(Lt)
                                break
                            except:
                                gam *= BETA
                        dstep = gam
                        y = yt
                        S = St
                    else:
                        if ntdecr >= 1.0:
                        # centering step: backtracking line search
                            gam = 1.0
                            logdetL = sum(log(L.diag()))
                            ddyb = - t*blas.dot(dy,b)
                            #val = ALPHA*ntdecr**2 - t*blas.dot(dy,b)
                            while gam > MINSTEP:
                                yt = y + gam*dy
                                St = Aadj(-yt) + C
                                val = ddyb + gam*ALPHA*ntdecr**2
                                try:
                                    Lt = St.copy()
                                    cholesky(Lt)
                                    if gam*val < 2*(sum(log(Lt.diag())) - logdetL) :
                                        break
                                    else:
                                        gam *= BETA
                                except:
                                    gam *= BETA

                            # update S,y
                            S = St
                            y = yt
                            dstep = gam
                        else:
                            # update S,y (full step)
                            y += dy
                            S = Aadj(-y) + C
                            dstep = 1.0
                        gam = 1.0
                        while True:
                            Xt = X.copy() + gam*dx
                            try:
                                Lt = Xt.copy()
                                completion(Lt)
                                break
                            except:
                                gam *= BETA
                        pstep = gam
                        X = Xt

            # Compute new gap and Omega(x,s)
            Lt = X.copy()
            completion(Lt)
            Lst = S.copy()
            cholesky(Lst)
            gapt = dot(X,S)
            Ot = Omega(Lt,Lst,gapt)
            if DEBUG: print(" -- actual gap:    %.2e" %(gapt))

            # update gap, t
            gap = min(n/t,gapt)  # Alternatively, set: gap = gapt [XXX]
            t = n/gap

            stype = 'a'

            # update residuals
            pres = blas.nrm2(Amap(X)-b)/resy0

            dres = Aadj(y) + S - C
            dres = sqrt(dot(dres,dres))/resx0

            pcost = dot(C,X)
            dcost = blas.dot(b,y)

            CENTER = True

        # print iter. info
        if gap is None:
            ggap = n/t
        else:
            ggap = gap
        if stype is 'c':
            if dcost is None:
                print("%3i % .4e %-11s %.1e %.1e %7s %.1e %7s %4.2f" %\
                    (iter, pcost, " ", ggap, pres, " ", ntdecr," ", gam))
            elif pcost is None:
                print("%3i %-11s % .4e %.1e %7s %.1e %.1e %13s %4.2f" %\
                    (iter, " ", dcost, ggap, " ", dres, ntdecr," ", gam))
            elif scaling == 'primal':
                print("%3i % .4e % .4e %.1e %.1e %.1e %.1e %7s %4.2f" %\
                    (iter, pcost, dcost, ggap, pres, dres, ntdecr," ", gam))
            else:
                print("%3i % .4e % .4e %.1e %.1e %.1e %.1e %13s %4.2f" %\
                    (iter, pcost, dcost, ggap, pres, dres, ntdecr," ", gam))
        elif stype is  'a':
            print("%3i % .4e % .4e %.1e %.1e %.1e %.1e %.1e %4.2f  %4.2f" %\
                (iter, pcost, dcost, ggap, pres, dres, ntdecr,Ot, pstep, dstep))

    ## END OF ITERATION LOOP

    # Get cputime and realtime
    Tcpu = cputime() - T0
    Twall = time() - T0wall

    if DIMACS and (not X is None) and (not y is None) and (not S is None):
        DIMACS_err = matrix(0.0,(6,1))
        DIMACS_err[0] = blas.nrm2(Amap(X) - b)/(1.0+max(abs(b)))
        DIMACS_err[1] = 0.0
        R = Aadj(y) + S - C
        c_abs = abs(c)
        if len(c_abs) == 0: c_abs = [0.0]
        DIMACS_err[2] = sqrt(dot(R,R))/(1.0+max(c_abs))
        DIMACS_err[3] = 0.0
        DIMACS_err[4] = (pcost - dcost)/(1 + abs(pcost) + abs(dcost))
        DIMACS_err[5] = gap/(1 + abs(pcost) + abs(dcost))
    else:
        DIMACS = False
        DIMACS_err = None

    # Convert to CVXOPT variables
    try:
        X = X.spmatrix(reordered=False, symmetric=False)
        X = perm(symmetrize(X),ip)
    except:
        X = None
    try:
        y = y[iperm(pm)]
        S = S.spmatrix(reordered=False, symmetric=False)
        S = perm(symmetrize(S),ip)
    except:
        y = None
        S = None

    if show_progress:
        print_exit_info()

    return {'status':status,
            'x':X,
            'y':y,
            's':S,
            'primal objective':pcost,
            'dual objective':dcost,
            'gap':gap,
            'relative gap':relgap,
            'primal infeasibility':pres,
            'dual infeasibility':dres,
            'iterations':iter,
            'cputime':Tcpu,
            'time':Twall}

def chordalsolver_esd(A,b,primalstart=None,dualstart=None,
                  scaling='primal',kktsolver='chol',p=None):
    """
    Chordal SDP solver:

    sol = chordalsolver_esd(A,b[,primalstart[,dualstart[,scaling[,kktsolver]]]])

    PURPOSE
    Solves a pair of primal and dual cone programs

         minimize   c'*x        maximize   b'*y
         subject to A*x = b     subject to A'*y + s = c
                    x in C                 s in K.

    Here C is the cone of symmetric matrices with sparsity
    pattern V that have a positive semidefinite completion,
    and K is the cone of positive semidefinite matrices with
    sparsity pattern V.

    ARGUMENTS
    A         CVXOPT spmatrix with doubles
    b         CVXOPT matrix with doubles

    RETURNS
    sol       dictionary

    """

    from cvxopt import matrix,spmatrix,sqrt,log,blas,lapack,base,cholmod
    #from chompack import sparse, project, tril, symmetrize, hessian, \
    #    perm, embed, maxcardsearch, llt, dot, axpy, copy, scal, \
    #    completion, cholesky, solve, partial_inv
    from chompack import cspmatrix, symbolic, tril, perm, symmetrize,dot,peo,\
        hessian, completion, cholesky, projected_inverse, llt, trsm, maxcardsearch
    from smcp import __version, misc


    from sys import platform
    if platform.startswith('win'):
        from time import clock as time
        def cputime():
            return 0.0
    else:
        from time import time
        from resource import getrusage, RUSAGE_SELF
        def cputime():
            return (getrusage(RUSAGE_SELF).ru_utime
                    +getrusage(RUSAGE_SELF).ru_stime)

    BETA = 0.7
    EXPON = 3.0
    STEP = 0.99
    MINSTEP = 1e-12

    # Start time
    T0wall = time()
    T0 = cputime()

    status = 'unknown'
    # Number of constraints
    m = A.size[1]-1
    # Matrix order
    n = int(sqrt(A.size[0]))

    DEBUG = options['debug']
    if not type(DEBUG)==bool:
        raise TypeError("options['debug'] must be a bool")

    MAXITERS = options['maxiters']
    if type(MAXITERS) is not int:
        raise TypeError("options['maxiters'] must be a positive "\
            "integer")
    elif MAXITERS < 1:
        raise ValueError("options['maxiters'] must be positive")

    ABSTOL = options['abstol']
    if type(ABSTOL) is not float and type(ABSTOL) is not int:
        raise TypeError("options['abstol'] must be a scalar")

    RELTOL = options['reltol']
    if type(RELTOL) is not float and type(RELTOL) is not int:
        raise TypeError("options['reltol'] must be a scalar")

    if RELTOL <= 0.0 and ABSTOL <= 0.0 :
        raise ValueError("at least one of options['reltol'] and " \
            "options['abstol'] must be positive")

    FEASTOL = options['feastol']
    if (type(FEASTOL) is not float and type(FEASTOL) is not int):
        raise TypeError("options['feastol'] must be a positive "\
            "scalar")
    elif FEASTOL <= 0.0:
        raise ValueError("options['feastol'] must be positive")

    CHOLMOD = options['cholmod']
    if type(CHOLMOD) is not bool:
        raise TypeError("options['cholmod'] must be bool")

    ORDER = options['order']
    if ORDER == "AMD":
        from cvxopt.amd import order
    elif ORDER == "METIS":
        from cvxopt.metis import order
    else:
        raise ValueError("options['order'] must be 'AMD' or 'METIS'")

    show_progress = options['show_progress']
    if type(show_progress) is not bool:
        raise TypeError("options['show_progress'] must be a bool")

    REFINEMENT = options['refinement']
    if (type(REFINEMENT)) is not int:
        raise TypeError("options['refinement'] must be a nonnegative "\
            "integer")
    elif REFINEMENT < 0:
        raise ValueError("options['refinement'] must be nonnegative ")

    TNZCOLS = options['tnzcols']
    if type(TNZCOLS) is not float:
        raise TypeError("tnzcols must be a float between 0.0 and 1.0")
    elif TNZCOLS > 1 or TNZCOLS < 0:
        raise ValueError("tnzcols must be between 0.0 and 1.0")
    else:
        TNZCOLS = int(n*TNZCOLS)

    DIMACS = options['dimacs']
    if type(DIMACS) is not bool:
        raise TypeError("dimacs must be a bool")

    if not (scaling=='primal' or scaling=='dual'):
        raise ValueError("scaling must be 'primal' or 'dual'")

    def iperm(p):
        """
        Returns inverse permutation vector.
        ip = iperm(p)
        """
        ip = matrix(0,(len(p),1))
        ip[p] = matrix(list(range(len(p))))
        return ip

    # solver does not handle sparse b yet:
    b = matrix(b)

    # aggregate sparsity pattern
    LIa = matrix(list(set(A.I)))
    Ia, Ja = misc.ind2sub(n,LIa)
    Va = tril(spmatrix(0.,Ia,Ja,(n,n)))
    del Ia,Ja,LIa

    if kktsolver=='chol':
        ### Find permutation
        Nz = misc.nzcolumns(A)
        pm,Ns = misc.matperm(Nz,TNZCOLS)
        # Permute first part of pm (Decorate-Sort-Undecorate)
        if (Ns < m):
            L = list(zip(A.CCS[0][pm[:m-Ns]+2] - A.CCS[0][pm[:m-Ns]+1],pm[:m-Ns]))
            L.sort(reverse=True)
            pm[0:m-Ns] = matrix([l for _,l in L])
        # Permute second part of pm (Decorate-Sort-Undecorate)
        if ( Ns > 0 ):
            L = list(zip(A.CCS[0][pm[m-Ns:m]+2] - A.CCS[0][pm[m-Ns:m]+1],pm[m-Ns:m]))
            #L = [(A.CCS[0][pm[i]+2]-A.CCS[0][pm[i]+1] + Nz[pm[i]],pm[i]) for i in xrange(m-Ns,m)]
            L.sort(reverse=True)
            pm[m-Ns:m] = matrix([l for _,l in L])
        del Nz
        # Permute b
        b = +b[pm]
    else:
        Ns = 0
        pm = matrix(range(m))

    bmax, ii = max(list(zip(abs(b),list(range(len(b))))))

    if p is None: p = order(Va)

    # make embeddings
    if CHOLMOD: # hack for using cholmod symbolic
        ORDER+="+CHOLMOD"
        Ve = Va + spmatrix([float(i+1) for i in range(n)],list(range(n)),list(range(n)),(n,n))
        F = cholmod.symbolic(Ve,p=p)
        cholmod.numeric(Ve,F)
        f = cholmod.getfactor(F)
        fd = [(j,i) for i,j in enumerate(f[:n**2:n+1])]
        fd.sort()
        Vp = +f
        symb = symbolic(Vp)
        Vp.V = matrix([float(v) for v in range(len(Vp))])
        ip = matrix([j for _,j in fd])
        p = iperm(ip)
        Ve = tril(perm(symmetrize(Vp),ip))
        lp = iperm([int(v) for v in Ve.V])
        CHORDAL = False
    else:
        pmcs = maxcardsearch(Va)
        if peo(Va,pmcs):
            CHORDAL = True
            p = pmcs
            symb = symbolic(Va,pmcs)
        else:
            CHORDAL = False
            symb = symbolic(Va,p)

        Ve = symb.sparsity_pattern(reordered=False, symmetric=False)
        Ve.V = matrix([float(v) for v in range(len(Ve))])
        Vp = tril(perm(symmetrize(Ve),p))
        lp = matrix([int(v) for v in Vp.V])
        symb = symbolic(Vp)
        ip = iperm(p)

    # extract vector space index permutation and Vp subscripts
    LI = misc.sub2ind((n,n),Ve.I,Ve.J)
    Ip,Jp = Vp.I,Vp.J

    # Extract Av and c
    # c = A[LI[lp],0] ## <-- unpacks A[:,0]
    c = A[:,0][LI[lp]] ## <-- does not unpack A[:,0]
    # Av = A[LI[lp],pm+1]
    if len(A) < 0.005*len(Va)*m or n > 15000: # do not unpack A
        Ve.V = matrix([float(v) for v in iperm(lp)])
        IJV = []; tmp = 0
        for j in pm:
            j1,j2 = A.CCS[0][j+1],A.CCS[0][j+2]
            #Ltmp = [(int(Ve[A.CCS[1][k]]),tmp,A.CCS[2][k]) for k in xrange(j1,j2)]
            Ltmp = list(zip([int(k) for k in Ve[A.CCS[1][j1:j2]]],
                       [tmp for i in range(j2-j1)],
                       [k for k in A.CCS[2][j1:j2]]))
            Ltmp.sort()
            IJV += Ltmp
            tmp+=1
        It,Jt,Vt = list(zip(*IJV))
        Av = spmatrix(Vt,It,Jt,(len(Vp),m))
        del It,Jt,Vt,IJV
    else: # unpack A
        Av = A[LI[lp],pm+1]

    # Indices of diagonal elements in Av
    Id = matrix([i for i in range(len(Vp)) if Ip[i] == Jp[i]])

    # Check that number of constraints is leq. number of nonzeros
    if m > len(Vp):
        raise ValueError("more constraints than nonzeros")

    # if Cholesky solver, extract nonzero column indices
    if kktsolver=='chol':
        Kl = []
        for j in range(m-Ns,m):
            k = []
            for l in Av.CCS[1][Av.CCS[0][j]:Av.CCS[0][j+1]]:
                k.append(Ip[l])
                k.append(Jp[l])
            Kl.append(list(set(k)))

    # Convert A0 to chompack
    Vp.V = matrix(c)
    C = cspmatrix(symb) + Vp

    def Amap(X,i=None):
        Xp = X.spmatrix(reordered=False, symmetric=False)
        misc.scal_diag(Xp,Id,0.5)
        v = matrix(0.,(m,1))
        if i is None:
            base.gemv(Av,Xp.V,v,trans='T',alpha=2.0)
            return v
        else:
            base.gemv(Av[:,i],Xp.V,v,trans='T',alpha=2.0)
            return v[0]

    def Aadj(y):
        v = matrix(0.,(len(Vp),1));
        base.gemv(Av,y,v,trans='N')
        Vp.V = v;
        return cspmatrix(symb) + Vp

    # Starting point
    if primalstart is not None:
        X = cspmatrix(symb) + tril(perm(symmetrize(tril(primalstart['x'])),p))
    else:
        X = cspmatrix(symb) + spmatrix(1.,range(n),range(n))
    if dualstart is not None:
        y = dualstart['y'][pm]
        S = cspmatrix(symb) + tril(perm(symmetrize(tril(dualstart['s'])),p))
    else:
        S = cspmatrix(symb) + spmatrix(1.,range(n),range(n))
        y = matrix(0.,(m,1))

    # Alternative starting point
    if False:
        # * Find X that solves
        #
        #     minimize    ||X||_F
        #     subject to  <A_i,X> = b_i, for all i
        #
        #   by solving the system
        #
        #     [ 0   A ][ y ] = [ b ]
        #     [ A' -I ][ X ]   [ 0 ]
        #
        # * Find S=C-sum_i A_i*y_i  that solves
        #
        #     minimize ||S||_F
        #
        #   by solving
        #
        #     [ 0   A ][ y ] = [ 0 ]
        #     [ A' -I ][-S ]   [ C ]

        if primalstart is None or dualstart is None:
            Avs = +Av
            Avs[Id] /= sqrt(2.0)
            L = spmatrix([],[],[],(m,m))
            base.syrk(Avs,L,trans='T')
            F = cholmod.symbolic(L)
            cholmod.numeric(L,F)
            Ic = cspmatrix(symb) + spmatrix(1.,list(range(n)),list(range(n)))
        if primalstart is None:
            u = +b
            cholmod.solve(F,u)
            X = Aadj(u)
            X = cspmatrix(symb) + Vp
            u = 1.
            while True :
                try:
                    Lt = X.copy()
                    completion(Lt)
                    break
                except ArithmeticError:
                    X += u*Ic
                    u *= 2.

        if dualstart is None:
            y = Amap(C)
            cholmod.solve(F,y)
            # compute s
            S = Aadj(-y) + C
            u = 1.
            while True :
                St = S.copy()
                try:
                    Lt = St.copy()
                    cholesky(Lt)
                    break
                except ArithmeticError:
                    S += u*Ic
                    u *= 2.

    tau   = 1.
    kappa = 1.

    # Initial gap
    gap = dot(X,S)/tau**2
    t = (n+1.)/(gap*tau**2+tau*kappa)

    # Norm of b and C
    resy0 = max(1,blas.nrm2(b))
    resx0 = max(1,sqrt(dot(C,C)))

    if scaling=='primal':
        def bres(sigma, dz=None):
            rby = (1-sigma)*ry

            rbx = rx.copy()
            blas.scal(1-sigma,rbx.blkval)

            rbt = (1-sigma)*rt

            rbs = L.copy()
            llt(rbs)
            blas.scal(sigma/t,rbs.blkval)
            rbs -= S

            rbk = -kappa + sigma/(t*tau)

            if dz:
                rby += b*dz[2] - Amap(dz[1])                 # uddate rby
                rbx += Aadj(dz[0])+dz[3]-dz[2]*C             # update rbx
                rbt += dot(C,dz[1])-blas.dot(b,dz[0])+dz[4]  # update rbt
                rbs -= dz[3]                                 # update rbs
                u = dz[1].copy()
                #hessian(L,Y,[u],inv=True,adj=True)
                #hessian(L,Y,[u],inv=True)
                hessian(L,Y,[u],inv=True,adj=None)
                rbs -= (1.0/t)*u
                rbk -= dz[4] + 1./(t*tau**2)*dz[2]           # update rbk

            return (rby,rbx,rbt,rbs,rbk)
    else: # dual scaling
        def bres(sigma, dz=None):

            rby = (1-sigma)*ry

            rbx = rx.copy()
            blas.scal(1-sigma,rbx.blkval)

            rbt = (1-sigma)*rt

            rbs = Y.copy()
            blas.scal(sigma/t,rbs.blkval)
            rbs -= X

            rbk = -tau + sigma/(t*kappa)

            if dz:
                rby += b*dz[2] - Amap(dz[1])                 # uddate rby
                rbx += -dz[2]*C+dz[3]+Aadj(dz[0])            # update rbx
                rbt += dot(C,dz[1])-blas.dot(b,dz[0])+dz[4]  # update rbt
                rbs -= dz[1]                                 # update rbs
                u = dz[3].copy()
                #hessian(L,Y,[u])
                #hessian(L,Y,[u],adj=True)
                hessian(L,Y,[u],inv=False,adj=None)
                rbs -= (1.0/t)*u
                rbk -= dz[2] + 1./(t*kappa**2)*dz[4]         # update rbk

            return (rby,rbx,rbt,rbs,rbk)

    if scaling=='primal':
        def tres(rbz):
            a = tau**2*t*(rbz[2] + rbz[4])

            rtx = C.copy()
            blas.scal(a,rtx.blkval)
            rtx -= rbz[1]+rbz[3]
            rty = rbz[0] + a*b

            return rtx,rty
    else:
        def tres(rbz):
            a = rbz[4] + 1./(t*kappa**2)*rbz[2]

            rtx = rbz[3].copy()
            #hessian(L,Y,[rtx],inv=True,adj=True)
            #hessian(L,Y,[rtx],inv=True)
            hessian(L,Y,rtx,inv=True,adj=None)
            blas.scal(-t,rtx.blkval)
            rtx += a*C-rbz[1]
            rty = rbz[0] + a*b

            return rtx,rty

    if DEBUG:
        def kkt_res(x,y,bx,by):
            r = x.copy()
            #hessian(L,Y,[r],inv=True,adj=True)
            #hessian(L,Y,[r],inv=True)
            hessian(L,Y,[r],inv=True,adj=None)
            if scaling=='primal':
                blas.scal(-1.0/t,r.blkval)
            else:
                blas.scal(-t,r.blkval)
            r += Aadj(y) - bx
            return r, Amap(x)-by

        def newton_res(sigma):
            rbz = bres(sigma)
            r1 = blas.nrm2(Amap(dX) - dtau*b - rbz[0])
            r2 = Aadj(-dy) + dtau*C - dS - rbz[1]
            r2 = sqrt(dot(r2,r2))
            r3 = abs(blas.dot(b,dy) - dot(C,dX) - dkappa - rbz[2])

            if scaling=='primal':
                r4 = dX.copy()
                #hessian(L,Y,[r4],adj=True,inv=True)
                #hessian(L,Y,[r4],adj=False,inv=True)
                hessian(L,Y,r4,inv=True,adj=None)
                blas.scal(1.0/t,r4.blkval)
                r4 += dS - rbz[3]
                r4 = sqrt(dot(r4,r4))
                r5 = abs(dtau/(t*tau**2) + dkappa - rbz[4])
            else:
                r4 = dS.copy()
                #hessian(L,Y,[r4])
                #hessian(L,Y,[r4],adj=True)
                hessian(L,Y,r4,inv=False,adj=None)
                blas.scal(1.0/t,r4.blkval)
                r4 += dX-rbz[3]
                r4 = sqrt(dot(r4,r4))
                r5 = abs(dkappa/(t*kappa**2) + dtau - rbz[4])
            print("\033[1;32m Newton:   % .2e % .2e % .2e % .2e % .2e \033[0m"
                  % (r1,r2,r3,r4,r5))

    def kkt_qr(L,Y):
        Ac_ = [Aj.copy() for Aj in Ac]
        hessian(L,Y,Ac_)
        for j in range(len(Ac_)):
            Aj = Ac_[j].spmatrix(reordered=False, symmetric=False)
            At[:,j] = Aj.V
        # scale diagonal elements
        At[Id,:] /= sqrt(2.)

        # compute QR factorization of At
        try:
            lapack.geqrf(At,tq)
        except ArithmeticError:
            print("\033[1;31m*** Factorization failed\033[0m")
            status = 'unknown'
            return None

        if scaling=='primal':
            kk = 1.0/t
        else:
            kk = t
        def solve_(bx,by):
            """
            Solves the system:

              [  -kk*H  A' ][ x ] = [ bx ]
              [   A     0  ][ y ]   [ by ]

            """
            # compute x,y
            r1 = bx.copy()
            hessian(L,Y,[r1])
            r1 = r1.spmatrix(reordered=False, symmetric=False)
            r1 = r1.V
            r1[Id] /= sqrt(2.0)
            x = +r1

            lapack.ormqr(At,tq,x,trans='T')
            x[m:] = 0.

            r2 = +by
            lapack.trtrs(At[:m,:],r2,uplo='U',trans='T')
            x[:m] += 0.5*kk*r2
            y = +x[:m]
            lapack.trtrs(At[:m,:],y,uplo='U')

            lapack.ormqr(At,tq,x)
            Vp.V = x-r1
            misc.scal_diag(Vp,Id,sqrt(2.0))
            x = cspmatrix(symb) + Vp
            hessian(L,Y,[x],adj=True)
            blas.scal(1.0/kk,x.blkval)

            if DEBUG:
                r,rr = kkt_res(x,y,bx,by)
                r = sqrt(dot(r,r))/sqrt(dot(bx,bx))
                rr = blas.nrm2(rr)/blas.nrm2(by)
                print("\033[1;33m KKTsolver: %.2e  %.2e \033[0m" % (r,rr))

            return x,y
        return solve_

    def kkt_chol(L,Y):
        # Compute H (lower triangle)
        for j in range(0,m-Ns):
            At = misc.Av_to_spmatrix(Av,Ip,Jp,j,n)
            U = cspmatrix(symb) + At
            #hessian(L,Y,[U]);
            #hessian(L,Y,[U],adj=True)
            hessian(L,Y,U,inv=False,adj=None)
            At = U.spmatrix(reordered=False, symmetric=False)
            misc.scal_diag(At,Id,0.5)
            base.gemv(Av[:,j:m],At.V,tmp,trans='T',alpha=2.0)
            H[j:m,j] = tmp[:m-j]

        for j in range(0,Ns):
            V = matrix(spmatrix(1.,Kl[j],range(len(Kl[j])),(n,len(Kl[j]))))
            trsm(L,V)
            trsm(L,V,trans='T')
            # compute column j of H (lower triangle only)
            kkl = matrix(0,(n,1))
            for k1,k2 in enumerate(Kl[j]): kkl[k2] = k1
            misc.SCMcolumn2(H,Av,V,Ip,Jp,kkl,m-Ns+j)

        # Cholesky factorization: H = L*L'
        try:
            lapack.potrf(H)
        except ArithmeticError:
            print("\033[1;31m*** Factorization failed\033[0m")
            status = 'unknown'
            return None

        if scaling=='primal':
            kk = 1.0/t
        else:
            kk = t
        def solve_(bx,by):
            """
            Solves the system:

              [ -kk*H  A' ][ x ] = [ bx ]
              [   A    0  ][ y ]   [ by ]

            """
            r1 = bx.copy()
            #hessian(L,Y,[r1])
            #hessian(L,Y,[r1],adj=True)
            hessian(L,Y,r1,inv=False,adj=None)
            y = kk*by + Amap(r1)
            lapack.potrs(H,y)

            x = Aadj(y)-bx
            #hessian(L,Y,[x])
            #hessian(L,Y,[x],adj=True)
            hessian(L,Y,[x],inv=False,adj=None)
            blas.scal(1.0/kk,x.blkval)

            if DEBUG:
                r,rr = kkt_res(x,y,bx,by)
                r1 = sqrt(dot(r,r))/sqrt(dot(bx,bx))
                r2 = blas.nrm2(rr)/blas.nrm2(by)
                print("\033[1;33m KKTsolver: %.2e  %.2e \033[0m" % (r1,r2))
            return x,y
        return solve_

    if kktsolver == 'chol':
        factor = kkt_chol
        SolvStr = "Cholesky"
        # Allocate storage for SCM
        H = matrix(0.,(m,m))
        # Allocate storage for tmp
        if Ns<m: tmp = matrix(0.,(m,1))
    elif kktsolver == 'qr':
        factor = kkt_qr
        SolvStr = "QR"
        # Convert A_1,...,A_m to chompack
        Ac = [cspmatrix(symb) + misc.Av_to_spmatrix(Av,Ip,Jp,j,n) for j in range(m)]
        # Allocate storage for At
        At = matrix(0.,(len(Vp),m))
        # Allocate storage for tq
        tq = matrix(0.,(m,1))

    def kktsolver(L,Y):
        return factor(L,Y)

    if scaling=='primal':
        def newton(sigma):
            rbz = bres(sigma)
            rtx,rty = tres(rbz)
            u1,u2 = f(rtx,rty)

            gamma = (-blas.dot(b,u2)+dot(C,u1))/  \
                (1.0/(t*tau**2) + blas.dot(b,v2) - dot(C,v1))

            dy = u2 + gamma*v2
            dX = u1.copy()+gamma*v1

            dkappa = - rbz[2]+blas.dot(b,dy) - dot(C,dX)
            if bmax > 1e-5:
                dtau = (Amap(dX,ii) - rbz[0][ii])/b[ii] # OPTIMIZE!!
            else:
                dtau = (rbz[4] - dkappa)*t*tau**2

            #dS = Aadj(-dy)
            #axpy(C,dS,dtau)
            #axpy(rbz[1],dS,-1)
            dS = dX.copy()
            blas.scal(-1.0/t,dS.blkval)
            #hessian(L,Y,[dS],inv=True,adj=True)
            #hessian(L,Y,[dS],inv=True)
            hessian(L,Y,dS,inv=True,adj=None)
            dS += rbz[3]

            for i in range(REFINEMENT):
                rbz = bres(sigma,dz=(dy,dX,dtau,dS,dkappa))
                rtx,rty = tres(rbz)
                u1,u2 = f(rtx,rty)

                gamma = (-blas.dot(b,u2) + dot(C,u1))/  \
                    (1.0/(t*tau**2) + blas.dot(b,v2) - dot(C,v1))

                ddy = u2 + gamma*v2
                ddX = u1.copy() + gamma*v1

                ddkappa = - rbz[2] + blas.dot(b,ddy) - dot(C,ddX)
                if bmax > 1e-5:
                    ddtau = (Amap(ddX,ii) - rbz[0][ii])/b[ii] # OPTIMIZE!!
                else:
                    ddtau = (rbz[4] - ddkappa)*t*tau**2

                #ddS = Aadj(-ddy)
                #axpy(C,dS,ddtau)
                #axpy(rbz[1],ddS,-1.0)
                ddS = ddX.copy()
                blas.scal(-1.0/t,ddS.blkval)
                #hessian(L,Y,[ddS],inv=True,adj=True)
                #hessian(L,Y,[ddS],inv=True)
                hessian(L,Y,[ddS],inv=True,adj=None)
                ddS += rbz[3]

                dy += ddy
                dX += ddX
                dtau += ddtau
                dS += ddS
                dkappa += ddkappa
            return (dy,dX,dtau,dS,dkappa)
    else:
        def newton(sigma):
            rbz = bres(sigma)
            rtx,rty = tres(rbz)
            u1,u2 = f(rtx,rty)

            gamma = (-blas.dot(b,u2) + dot(C,u1))/  \
                (t*kappa**2 + blas.dot(b,v2) - dot(C,v1))

            dy = u2 + gamma*v2
            dX = u1.copy() + gamma*v1

            dkappa = -rbz[2] + blas.dot(b,dy) - dot(C,dX)
            if bmax > 1e-5:
                dtau = (Amap(dX,ii) - rbz[0][ii])/b[ii] # OPTIMIZE!!
            else:
                dtau = rbz[4] - dkappa/(t*kappa**2)

            #dS = Aadj(-dy)
            #axpy(C,dS,dtau)
            #axpy(rbz[1],dS,-1.0)
            dS = rbz[3].copy() - dX
            blas.scal(t,dS.blkval)
            #hessian(L,Y,[dS],inv=True,adj=True)
            #hessian(L,Y,[dS],inv=True)
            hessian(L,Y,[dS],inv=True,adj=None)

            for i in range(REFINEMENT):
                rbz = bres(sigma,dz=(dy,dX,dtau,dS,dkappa))
                rtx,rty = tres(rbz)
                u1,u2 = f(rtx,rty)

                gamma = (-blas.dot(b,u2) + dot(C,u1))/  \
                (t*kappa**2 + blas.dot(b,v2) - dot(C,v1))

                ddy = u2 + gamma*v2
                ddX = u1.copy() + gamma*v1

                ddkappa = - rbz[2] + blas.dot(b,ddy) - dot(C,ddX)
                if bmax > 1e-5:
                    ddtau = (Amap(ddX,ii) - rbz[0][ii])/b[ii] # OPTIMIZE!!
                else:
                    ddtau = rbz[4] - ddkappa/(t*kappa**2)

                #ddS = Aadj(-ddy)
                #axpy(C,ddS,ddtau)
                #axpy(rbz[1],ddS,-1.0)
                ddS = rbz[3].copy()-ddX
                blas.scal(t,ddS.blkval)
                hessian(L,Y,[ddS],inv=True,adj=True)
                hessian(L,Y,[ddS],inv=True)
                hessian(L,Y,[ddS],inv=True,adj=None)

                dy += ddy
                dX += ddX
                dtau += ddtau
                dS += ddS
                dkappa += ddkappa
            return (dy,dX,dtau,dS,dkappa)


    def print_head():
        print("%-20s Extended self-dual embedding, %s scaling (%s)" % (__version,scaling,SolvStr))
        print("----------------------------------------------------------------------------")
        print("SDP var. size:       %i " % (n))
        print("Constraints:         %i (%i|%i)" % (m,m-Ns,Ns))
        if CHORDAL: ChStr = "Chordal"
        else: ChStr = "Nonchordal"
        print("Aggregate sparsity:  %-14s NNZ(tril(V)) = %7i" % (ChStr,len(Va)))
        if not CHORDAL:
            print("Embedding:           %-14s       NNZ(L) = %7i" % (ORDER,len(Vp)))
        print("----------------------------------------------------------------------------")
        print(" it  pcost       dcost      gap     pres    dres    k/t     step    cputime")

    def print_exit_info():
        if status=='optimal' or status=='unknown':
            if pcost is not None:
                print("   Primal objective:                % .8e" % (pcost))
            if dcost is not None:
                print("   Dual objective:                  % .8e" % (dcost))
        if gap is not None:
            print("   Gap:                             % .8e" % (gap))
        if relgap is not None:
            print("   Relative gap:                    % .8e" % (relgap))
        if pres is not None:
            print("   Primal infeasibility:            % .8e" % (pres))
        if dres is not None:
            print("   Dual infeasibility:              % .8e" % (dres))
        if not iter==0:
            print("   Iterations:                       %i"   % (iter))
            if not platform.startswith('win'):
                print("   CPU time:                         %.2f" % (Tcpu))
                print("   CPU time per iteration:           %.2f" % (Tcpu/iter))
            print("   Real time:                        %.2f" % (Twall))
            print("   Real time per iteration:          %.2f\n" % (Twall/iter))
        if DIMACS:
            print("   DIMACS:  %.2e %.2e %.2e %.2e %.2e %.2e" % tuple([v for v in DIMACS_err]))

    def linesearch(dX,dS,dtau,dkappa):

        """
        Backtracking linesearch:

        Computes step length t such that

           kappa + t*dkappa > 0, tau + t*dtau > 0,
           S + t*dS > 0, X + t*dX > 0
        """
        t = 1.
        while (kappa+t*dkappa <= 0) or (tau+t*dtau <= 0):
            t *= BETA
            if t < MINSTEP: return None

        while True :
            Xt = X.copy() + t*dX
            try:
                Lt = Xt.copy()
                completion(Lt)
                break
            except ArithmeticError:
                t *= BETA
                if t < MINSTEP: return None

        while True :
            St = S.copy() + t*dS
            try:
                Lt = St.copy()
                cholesky(Lt)
                break
            except ArithmeticError:
                t *= BETA
                if t < MINSTEP: return None

        return t

    if show_progress:
        print_head()

    for iter in range(MAXITERS+1) :
        ## RESIDUALS AND CONVERGENCE STATISTICS
        # hry = A*x
        hry = Amap(X)
        # ry = b*tau - A*x
        ry = b*tau - hry

        # hrx = A'*y + s
        hrx = Aadj(y) + S
        # rx = A'*y + s - tau*C
        rx = hrx.copy() - tau*C

        # rt = kappa - b'*y + c'*x
        rt = kappa - blas.dot(b,y) + dot(C,X)

        resy = blas.nrm2(ry)/tau
        resx = sqrt(dot(rx,rx))/tau
        pres = resy/resy0
        dres = resx/resx0

        cx = dot(C,X)
        pcost = cx/tau
        by = blas.dot(b,y) # TODO: handle sparse b
        dcost = by/tau

        # compute gap
        gap = dot(X,S)/tau**2
        # compute relgap
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None

        # compute pinfres
        if -by < 0.0:
            hresx = sqrt(dot(hrx,hrx))
            pinfres = hresx/resx0/by
        else:
            pinfres = None

        # compute dinfres
        if cx < 0.0:
            hresy = blas.nrm2(hry)
            dinfres = hresy/resy0/(-cx)
        else:
            dinfres = None

        if show_progress:
            if not iter == 0:
                print("%3d % .4e % .4e %.1e %.1e %.1e %.1e %.1e %7.1f" % \
                    (iter,pcost,dcost,gap,pres,dres,kappa/tau,step,cputime()-T0))
            else:
                print("%3d % .4e % .4e %.1e %.1e %.1e %.1e         %7.1f" % \
                    (iter,pcost,dcost,gap,pres,dres,kappa/tau,cputime()-T0))

        ## Stopping criteria
        if (dres <= FEASTOL) and (pres <= FEASTOL) and \
                (gap <= ABSTOL or (relgap is not None and relgap <= RELTOL)) :
            # Optimal
            if show_progress:
                print("Optimal solution found.")
            status = 'optimal'
            break
        elif pinfres is not None and pinfres <= FEASTOL:
            # Primal infeasible
            if show_progress:
                print("Certificate of primal infeasibility found.")
            status = 'primal infeasibility'
            X = None
            pcost = None; dcost = 1;
            gap = None; relgap = None
            pres = None; dres = None
            dinfres = None
            break
        elif dinfres is not None and dinfres <= FEASTOL:
            # Dual infeasible
            if show_progress:
                print("Certificate of dual infeasibility found.")
            status = 'dual infeasibility'
            y = None; S = None
            pcost = -1; dcost = None;
            gap = None; relgap = None
            pres = None; dres = None
            pinfres = None
            break
        elif iter == MAXITERS:
            # Max. number of iterations reached
            if show_progress:
                print("\033[1;31mTerminated " \
                    "(maximum number of iterations reached).\033[0m")
            status = 'unknown'
            break

        # compute new t
        t = (n+1)/(gap*tau**2 + kappa*tau)

        if scaling=='primal':
            try:
                L = X.copy()
                completion(L)
            except:
                if show_progress:
                    print("\033[1;31m*** Completion failed\033[0m")
                status = 'unknown'
                break
            Y = X
        else:
            try:
                L = S.copy()
                cholesky(L)
            except:
                if show_progress:
                    print("\033[1;31m*** Factorization of S failed\033[0m")
                status = 'unknown'
                break
            Y = L.copy()
            projected_inverse(Y)

        f = kktsolver(L,Y)
        if f is None: break
        v1,v2 = f(C,b)

        # Solve for aff. scaling direction
        dy,dX,dtau,dS,dkappa = newton(0.)
        if DEBUG: newton_res(0.)
        # Compute step length
        step = linesearch(dX,dS,dtau,dkappa)
        if not step:
            if show_progress: print("\033[1;31mTerminated "\
                    "(small step size detected).\033[0m")
            status = 'unknown'
            break

        # Compute sigma
        St = S.copy() + step*dS
        Xt = X.copy() + step*dX
        taut = tau + step*dtau
        kappat = kappa+step*dkappa
        sigma = ((dot(Xt,St)+taut*kappat)/(gap*tau**2+kappa*tau))**EXPON

        # compute newton direction
        dy,dX,dtau,dS,dkappa = newton(sigma)
        if DEBUG: newton_res(sigma)
        # Compute step length
        step = linesearch(dX,dS,dtau,dkappa)
        if not step:
            if show_progress: print("\033[1;31mTerminated "\
                    "(small step size detected).\033[0m")
            break

        # Update (X,y,S)
        X += STEP*step*dX
        y += STEP*step*dy
        S += STEP*step*dS
        tau += STEP*step*dtau
        kappa += STEP*step*dkappa
        if DEBUG: print("\033[1;31m mu=%.2e  tau=%.2e  kappa=%.2e  sigma=%.2e \033[0m" % (1.0/t,tau,kappa,sigma))


    ## END OF ITERATION LOOP

    # Get cputime and realtime
    Tcpu = cputime() - T0
    Twall = time() - T0wall

    if DIMACS and (not X is None) and (not y is None) and (not S is None):
        DIMACS_err = matrix(0.0,(6,1))
        DIMACS_err[0] = blas.nrm2(Amap(X)/tau - b)/(1+max(abs(b)))
        DIMACS_err[1] = 0.0
        R = Aadj(y) + S; blas.scal(1.0/tau,R.blkval);
        R -= C
        c_abs = abs(c)
        if len(c_abs) == 0: c_abs = [0.0]
        DIMACS_err[2] = sqrt(dot(R,R))/(1+max(c_abs))
        DIMACS_err[3] = 0.0
        DIMACS_err[4] = (pcost - dcost)/(1 + abs(pcost) + abs(dcost))
        DIMACS_err[5] = gap/(1 + abs(pcost) + abs(dcost))
    else:
        DIMACS = False
        DIMACS_err = None

    # Scale and convert to CVXOPT
    if not X is None:
        blas.scal(1.0/tau,X.blkval)
        X = X.spmatrix(reordered=False, symmetric=False)
        X = perm(symmetrize(X),ip)
    if not y is None:
        y = y[iperm(pm)]
        y /= tau
    if not S is None:
        blas.scal(1.0/tau,S.blkval)
        S = S.spmatrix(reordered=False, symmetric=False)
        S = perm(symmetrize(S),ip)

    if show_progress:
        print_exit_info()

    return {'status':status,
            'x':X,
            'y':y,
            's':S,
            'primal objective':pcost,
            'dual objective':dcost,
            'gap':gap,
            'relative gap':relgap,
            'primal infeasibility':pres,
            'dual infeasibility':dres,
            'residual as primal infeasibility certificate':pinfres,
            'residual as dual infeasibility certificate':dinfres,
            'iterations':iter,
            'cputime':Tcpu,
            'time':Twall}



def conelp(c,G,h,dims=None,kktsolver='chol'):
    from smcp import SDP,misc
    from cvxopt import matrix,sparse,spdiag,spmatrix
    import chompack

    Nl = dims['l']
    Nq = dims['q']
    Ns = dims['s']
    if not Nl: Nl = 0

    nblocks = Nl + len(Nq) + len(Ns)

    P = SDP()

    P._n = Nl+sum(Nq)+sum(Ns)
    P._m = G.size[1]

    P._A = spmatrix([],[],[],(P._n**2,P._m+1))
    P._b = -c
    P._blockstruct = []
    if Nl: P._blockstruct.append(-Nl)
    for i in Nq: P._blockstruct.append(i)
    for i in Ns: P._blockstruct.append(i)

    for k in range(P._m+1):
        if not k==0:
            v = G[:,k-1].spmatrix(reordered=False, symmetric=False)
        else:
            v = h.spmatrix(reordered=False, symmetric=False)
        B = []

        ptr = 0
        # lin. constraints
        if Nl:
            u = v[:Nl]
            I = u.I
            B.append(spmatrix(u.V,I,I,(Nl,Nl)))
            ptr += Nl

        # SOC constraints
        for i in range(len(Nq)):
            nq = Nq[i]
            u0 = v[ptr]
            u1 = v[ptr+1:ptr+nq]
            tmp = spmatrix(u1.V,[nq-1 for j in range(len(u1))],u1.I,(nq,nq))
            if not u0 == 0.0:
                tmp += spmatrix(u0,range(nq),range(nq),(nq,nq))
            B.append(tmp)
            ptr += Nq[i]

        # SDP constraints
        for i in range(len(Ns)):
            ns = Ns[i]
            u = v[ptr:ptr+ns**2]
            I,J = misc.ind2sub(ns,u.I)
            tmp = chompack.tril(spmatrix(u.V,I,J,(ns,ns)))
            B.append(tmp)
            ptr += ns**2

        Ai = spdiag(B)
        P._A[:,k] = Ai[:]

    P._agg_sparsity()
    sol = P.solve_esd(kktsolver=kktsolver)

    x = sol['x']
    s = sol['s']
    n = P.n

    # recover primal and dual variables
    N = 0
    if Nl:
        xl = x[:n*Nl+1:n+1]
        N += Nl
    else: xl = matrix([])

    xq = []
    if Nq:
        for i in Nq:
            xq.append( matrix([sum(x[(n+1)*N:(n+1)*(N+i):n+1]), 2*x[N+i-1,N:N+i-1].T]))
            N += i
    else:
        xq = matrix([])

    xs = []
    if Ns:
        for i in Ns:
            xs.append(x[N:N+i,N:N+i])
            N += i
    else:
        xs = [matrix([])]

    N = 0
    if Nl:
        sl = s[:n*Nl+1:n+1]
        N += Nl
    else: sl = matrix([])

    sq = []
    if Nq:
        for i in Nq:
            sq.append( matrix([s[N+i-1,N+i-1], s[N+i-1,N:N+i-1].T]))
            N += i
    else:
        sq = matrix([])

    ss = []
    if Ns:
        for i in Ns:
            ss.append(s[N:N+i,N:N+i])
            N += i
    else:
        ss = [matrix([])]

    sol['x'] = sol.pop('y')
    sol['z'] = sparse([xl,sparse(xq),sparse([z[:] for z in xs])])
    sol['s'] = sparse([sl,sparse(sq),sparse([z[:] for z in ss])])

    return sol


def lp(c,G,h,kktsolver='chol'):
    return conelp(c,G,h,dims={'l':G.size[0],'q':[],'s':[]},kktsolver=kktsolver)

def socp(c,Gl=None,hl=None,Gq=None,hq=None,kktsolver='chol'):
    from cvxopt import sparse
    dims = {'l':0,'q':[],'s':[]}
    G = []; h = []
    if Gl is not None and hl is not None:
        dims['l'].append(Gl.size[0])
        G.append(Gl)
        h.append(hl)
    if Gq is not None and hq is not None:
        for j in range(len(Gq)):
            dims['q'].append(Gq[j].size[0])
            G.append(Gq[j])
            h.append(hq[j])
    else:
        raise ValueError("'Gq' and 'hq' cannot be zero")

    sol = conelp(c,sparse(G),sparse(h),dims=dims,kktsolver=kktsolver)

    Nl = dims['l']
    N = 0
    if Nl:
        sol['zl'] = sol['z'][N:Nl]
        sol['sl'] = sol['s'][N:Nl]
        N += Nl
    else:
        sol['zl'] = None
        sol['sl'] = None
    zq = []
    sq = []

    for ns in dims['q']:
        zq += [sol['z'][N:N+ns]]
        sq += [sol['s'][N:N+ns]]
        N += ns

    sol['zq'] = zq
    sol['sq'] = sq
    sol.pop('s')
    sol.pop('z')

    return sol

def sdp(c,Gl=None,hl=None,Gs=None,hs=None,kktsolver='chol'):
    from cvxopt import sparse, spmatrix, sqrt
    from smcp import misc
    dims = {'l':0,'q':[],'s':[]}
    G = []; h = []
    if Gl is not None and hl is not None:
        dims['l'] = Gl.size[0]
        G.append(Gl)
        h.append(hl)
    if Gs is not None and hs is not None:
        for j in range(len(Gs)):
            dims['s'].append(int(sqrt(Gs[j].size[0])))
            G.append(Gs[j])
            h.append(hs[j][:])
    else:
        raise ValueError("'Gs' and 'hs' cannot be zero")

    sol = conelp(c,sparse(G),sparse(h),dims=dims,kktsolver=kktsolver)

    Nl = dims['l']
    N = 0
    if Nl:
        sol['zl'] = sol['z'][N:Nl]
        sol['sl'] = sol['s'][N:Nl]
        N += Nl
    else:
        sol['zl'] = None
        sol['sl'] = None
    zs = []
    ss = []

    for ns in dims['s']:
        u = sol['z'][N:N+ns**2]
        I,J = misc.ind2sub(ns,u.I)
        zs += [spmatrix(u.V,I,J,(ns,ns))]

        u = sol['s'][N:N+ns**2]
        I,J = misc.ind2sub(ns,u.I)
        ss += [spmatrix(u.V,I,J,(ns,ns))]
        N += ns**2

    sol['zs'] = zs
    sol['ss'] = ss
    sol.pop('s')
    sol.pop('z')

    return sol


__all__ = ['chordalsolver_feas','chordalsolver_esd','lp','socp','sdp','conelp','options']
