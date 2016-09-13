# Copyright 2010 M. S. Andersen & L. Vandenberghe
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

from smcp import SDP, misc
import chompack
from cvxopt import matrix, spmatrix, sparse

def embed_SDP(P,order="AMD",cholmod=False):
    if not isinstance(P,SDP): raise ValueError("not an SDP object")
    if order=='AMD':
        from cvxopt.amd import order
    elif order=='METIS':
        from cvxopt.metis import order
    else: raise ValueError("unknown ordering: %s " %(order))
    p = order(P.V)

    if cholmod:
        from cvxopt import cholmod
        V = +P.V + spmatrix([float(i+1) for i in range(P.n)],range(P.n),range(P.n))
        F = cholmod.symbolic(V,p=p)
        cholmod.numeric(V,F)
        f = cholmod.getfactor(F)
        fd = [(j,i) for i,j in enumerate(f[:P.n**2:P.n+1])]
        fd.sort()
        ip = matrix([j for _,j in fd])
        Ve = chompack.tril(chompack.perm(chompack.symmetrize(f),ip))
        Ie = misc.sub2ind((P.n,P.n),Ve.I,Ve.J)
    else:
        #Vc,n = chompack.embed(P.V,p)
        symb = chompack.symbolic(P.V,p)
        #Ve = chompack.sparse(Vc)
        Ve = symb.sparsity_pattern(reordered=False)
        Ie = misc.sub2ind((P.n,P.n),Ve.I,Ve.J)
    Pe = SDP()
    Pe._A = +P.A; Pe._b = +P.b
    Pe._A[:,0] += spmatrix(0.0,Ie,[0 for i in range(len(Ie))],(Pe._A.size[0],1))
    Pe._agg_sparsity()
    Pe._pname = P._pname + "_embed"
    Pe._ischordal = True; Pe._blockstruct = P._blockstruct
    return Pe      

try: 
    import pylab
    _PYLAB = True
except:
    _PYLAB = False

if _PYLAB:
    def spy(P,i=None,file=None,scale=None):
        """Generates sparsity plot using Pylab"""
        if type(P) is spmatrix:
            V = chompack.symmetrize(chompack.tril(P))
            n = V.size[0]
        else:
            if not P._A: raise AttributeError("SDP data missing")
            n = +P.n; 
            if i == None: V = chompack.symmetrize(P.V)
            elif i>=0 and i<=P.m and P._A:
                r = P._A.CCS[1][P._A.CCS[0][i]:P._A.CCS[0][i+1]]
                if type(r) is int: I = [r%n]; J = [r/n]
                else: I,J = misc.ind2sub(n,r)
                V = chompack.symmetrize(spmatrix(0.,I,J,(n,n)))
            else: raise ValueError("index out of range")

        from math import floor
        msize = max(1,int(floor(100./n)))

        if file==None: pylab.ion()
        else: pylab.ioff()
        f = pylab.figure(figsize=(6,6)); f.clf()
        if scale is None: scale = 1.0 
        I = V.I*scale+1; J = (n-V.J)*scale
        p = pylab.plot(I,J, 's', linewidth = 1, hold = 'False')
        pylab.setp(p, markersize = msize, markerfacecolor = 'k')
        g = pylab.gca()
        pylab.axis([0.5,n*scale+0.5,0.5,n*scale+0.5])
        g.set_aspect('equal')
        locs,labels = pylab.yticks()
        locs = locs[locs<=n*scale]; locs = locs[locs>=1]
        pylab.yticks(locs[::-1]-(locs[-1]-n*scale-1)-locs[0],
                     [str(int(loc)) for loc in locs])
        if file: pylab.savefig(file)

    def nzcols_hist(P,file=None):
        if file==None: pylab.ion()
        else: pylab.ioff()
        nzcols = +P.nzcols
        Nmax = max(nzcols)
        y = matrix(0,(Nmax,1))
        for n in nzcols:
            y[n-1] += 1
        y = sparse(y)
        f = pylab.figure(); f.clf()
        pylab.stem(y.I+1,y.V)#; pylab.xlim(-1,Nmax+2)
        if file: pylab.savefig(P._pname + "_nzcols_hist.png")

    def nnz_hist(P,file=None):
        if file==None: pylab.ion()
        else: pylab.ioff()
        nnz = +P.nnzs
        Nmax = max(nnz)
        y = matrix(0,(Nmax,1))
        for n in nnz:
            y[n-1] += 1
        y = sparse(y)
        f = pylab.figure(); f.clf()
        pylab.stem(y.I+1,y.V)#; pylab.xlim(-1,Nmax+2)
        if file: pylab.savefig(P._pname + "_nnz_hist.png")

    def clique_hist(P,file=None):
        if not P.ischordal: raise TypeError("Nonchordal")
        if file==None: pylab.ion()
        else: pylab.ioff()
        V = +P.V 
        p = chompack.maxcardsearch(V)
        #Vc,n = chompack.embed(V,p)
        symb = chompack.symbolic(V,p)
        #D = chompack.info(Vc); N = len(D)
        N = symb.Nsn
        #Ns = [len(v['S']) for v in D]; Nu = [len(v['U']) for v in D]
        #Nw = [len(v['U']) + len(v['S']) for v in D]
        Ns = [len(v) for v in symb.supernodes()]
        Nu = [len(v) for v in symb.separators()]
        Nw = [len(v) for v in symb.cliques()]
        
        f = pylab.figure(); f.clf()
        f.text(0.58,0.40,"Number of cliques: %i" % (len(Nw)))
        f.text(0.61,0.40-1*0.07,"$\sum | W_i | = %i$" % (sum(Nw)))
        f.text(0.61,0.40-2*0.07,"$\sum | V_i | = %i$" % (sum(Ns)))
        f.text(0.61,0.40-3*0.07,"$\sum | U_i | = %i$" % (sum(Nu)))
        f.text(0.61,0.40-4*0.07,"$\max_i\,| W_i | = %i$" % (max(Nw)))

        pylab.subplot(221)
        Nmax = max(Nw)
        y = matrix(0,(Nmax,1))
        for n in Nw :
            y[n-1] += 1
        y = sparse(y)
        pylab.stem(y.I+1,y.V,'k'); pylab.xlim(0, Nmax+1)
        pylab.title("Cliques")
        
        
        Nmax = max(Nu)
        y = matrix(0,(Nmax,1))
        if Nmax > 0: 
            pylab.subplot(222)
            for n in Nu :
                y[n-1] += 1
            y = sparse(y)
            pylab.stem(y.I+1,y.V,'k'); pylab.xlim(0, Nmax+1)
            pylab.title("Separators")

        pylab.subplot(223)
        Nmax = max(Ns)
        y = matrix(0,(Nmax,1))
        for n in Ns :
            y[n-1] += 1
        y = sparse(y)
        pylab.stem(y.I+1,y.V,'k'); pylab.xlim(0, Nmax+1)
        pylab.title("Residuals")

        if file: pylab.savefig(file)


