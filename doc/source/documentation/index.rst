#############
Documentation
#############

Overview
""""""""

Let :math:`\mathbf{S}^n` denote the set of symmetric matrices of order :math:`n`, and let 
and let :math:`C \bullet X` denote the standard inner product on :math:`\mathbf{S}^n`.

:math:`\mathbf{S}_V^n` is the set of symmetric matrices of order :math:`n` and with sparsity pattern :math:`V`, *i.e.*, :math:`X \in \mathbf{S}_V^n` if and only if :math:`X_{ij} = 0` for all :math:`(i,j) \neq V`. We will assume that all diagonal elements are included in :math:`V`. The projection :math:`Y` of :math:`X` on the subspace :math:`\mathbf{S}_V^n` is denoted :math:`Y=P_V(X)`, *i.e.*, :math:`Y_{ij} = X_{ij}` if :math:`(i,j) \in V` and :math:`Y_{ij} = 0` otherwise. The inequality signs :math:`\succeq` and :math:`\succ` denote matrix inequality. We define :math:`|V|` as the number of nonzeros in the lower triangular part of :math:`V`, and :math:`\mathrm{nnz}(A_i)` denotes the number of nonzeros in the matrix A_i.

:math:`\mathbf{S}_{V,+}^n` and :math:`\mathbf{S}_{V,++}^n` are the sets of positive semidefinite and positive definite matrices in :math:`\mathbf{S}_V^n`, and similarly, :math:`\mathbf{S}_{V,c+}^n = \{ P_V(X)\ |\ X \succeq 0 \}` and :math:`\mathbf{S}_{V,c++}^n = \{ P_V(X)\ |\ X \succ 0 \}` are the sets of matrices in :math:`\mathbf{S}_V^n` that have a positive semidefinite completion and a positive definite completion, respectively. We denote with :math:`\succeq_{\rm c}` and :math:`\succ_{\rm c}` matrix inequality with respect to the cone :math:`\mathbf{S}_{V,c+}^n`.


SMCP solves a pair of primal and dual linear cone programs:

.. math::

   \begin{array}{llllll}
    (P)  & \mbox{minimize}   & C \bullet X        & \qquad (D) & \mbox{maximize}   & b^Ty  \\
         & \mbox{subject to} & A_i \bullet X = b_i, i=1,\ldots,m &            & \mbox{subject to} & \sum_{i=1}^m y_iA_i + S = C \\
	 &                   & X \succeq_{\rm c} 0  &            &             & S \succeq 0.
   \end{array}

The variables are :math:`X \in \mathbf{S}_V^n`, :math:`S \in \mathbf{S}_V^n`, and :math:`y \in \mathbf{R}^m`, and the problem data are the matrix :math:`C \in \mathbf{S}_V^n`, the vector :math:`b \in \mathbf{R}^m`, and :math:`A_i \in \mathbf{S}_V^n, i=1,\ldots,m`.

Compositions of cones are handled implicitly by defining a block diagonal sparsity pattern :math:`V`.
Dense blocks and general sparse blocks correspond to standard positive semidefinite matrix constraints, diagonal blocks corresponds to linear inequality constraints, and second-order cone constraints can be embedded in an LMI with an "arrow pattern", *i.e*,

.. math::
   
   \lVert x \rVert_2 \leq t \quad \Leftrightarrow \quad
   \begin{bmatrix}
	t  & x^T \\
	x  & t I
   \end{bmatrix} \succeq 0.


The chordal SDP solvers
"""""""""""""""""""""""

.. function:: smcp.solvers.chordalsolver_feas(A,b[,primalstart[,dualstart[,scaling='primal'[,kktsolver='chol']]]])

   Solves the pair of cone programs (P) and (D) using a feasible start
   interior-point method. If no primal and/or dual feasible starting
   point is specified, the algorithm tries to find a feasible starting
   point based on some simple heuristics. An exception is raised if no
   starting point can be found. In this case a Phase I problem must be
   solved, or the (experimental) infeasible start interior-point
   method :func:`chordalsolver_esd <smcp.solvers.chordalsolver_esd>` can be
   used.

   The columns of the sparse matrix ``A`` are vectors of length
   :math:`n^2` and the :math:`m+1` columns of ``A`` are:

   .. math::

      \left[ \mathbf{vec}(C) \ \mathbf{vec}(A_1) \ \cdots \ \mathbf{vec}(A_m) \right].

   Only the rows of ``A`` corresponding to the lower triangular
   elements of the aggregate sparsity pattern :math:`V` are accessed.
   
   The optional argument `primalstart` is a dictionary with the key
   `x` which can be used to specify an initial value for the primal
   variable :math:`X`.  Similarly, the optional argument `dualstart`
   must be a dictionary with keys `s` and `y`.

   The optional argument `scaling` takes one of the values
   :const:`'primal'` (default) or :const:`'dual'`.

   The optional argument `kktsolver` is used to specify the KKT
   solver. Possible values include:

   :const:`'chol'` (default)
	solves the KKT system via a Cholesky factorization of the Schur complement  

   :const:`'qr'`
   	solves the KKT system via a QR factorization

   The solver returns a dictionary with the following keys:

   :const:`'primal objective'`, :const:`'dual objective'`
        primal objective value and dual objective value.

   :const:`'primal infeasibility'`, :const:`'dual infeasibility'`
   	residual norms of primal and dual infeasibility.
   			   
   :const:`'x'`, :const:`'s'`, and :const:`'y'`
   	primal and dual variables.

   :const:`'iterations'`
	number of iterations.

   :const:`'cputime'`, :const:`'time'`
   	total cputime and real time.

   :const:`'gap'`
	duality gap.

   :const:`'relative gap'`	
   	relative duality gap.

   :const:`'status'`
	* has the value :const:`'optimal'` if 
   
	  .. math::		
	   
		\frac{\| b-\mathcal{A}(X) \|_2}{\max\{1,\|b\|_2 \}} \leq \epsilon_{\rm feas}, \qquad 
		\frac{\|\mathcal{A}^{\rm adj}(y) + S -  C\|_F}{\max\{1,\|C\|_F \}} \leq \epsilon_{\rm feas}, \qquad
		X \succ_{\rm c} 0,\qquad
      	   	S \succ 0,
				
          and

    	  .. math::
   
	      X\bullet S \leq \epsilon_{\rm abs} \quad \text{or} \quad \left( \min\{C\bullet X, -b^Ty\} \leq 0 ,
	      \frac{X\bullet S}{-\min\{C\bullet X,-b^Ty \}} \leq \epsilon_{\rm rel} \right).
      
        * has the value :const:`'unknown'` otherwise.

   The following options can be set using the dictionary
   :const:`smcp.solvers.options`:

   :const:`'delta'` (default: 0.9) 
        a positive constant between 0 and
        1; an approximate tangent direction is computed when the
        Newton decrement is less than :const:`delta`. 
  
   :const:`'eta'` (default: :const:`None`) 
	:const:`None` or a positive float. If :const:`'eta'` is a
        positive number, a step in the approximate tangent
        direction is taken such that
	   
	   .. math::
	      
	      \Omega(X+\alpha \Delta X, S + \alpha \Delta S) \approx \eta
 	      		      
	where :math:`\Omega(X,S)` is the proximity function

	   .. math::
	   
	      \Omega(X,S) = \phi_{\rm c}(X) + \phi(S) + n\cdot \log\frac{X\bullet S}{n} + n.
	
	If :const:`'eta'` is :const:`None`, the step length
	:math:`\alpha` in the approximate tangent direction is
	computed as
	
	   .. math::
	   
	      \alpha_p &= \arg \max \{ \alpha \in (0,1] \,|\, X + \alpha \Delta X \succeq_{\rm c} 0 \} \\
	      \alpha_d &= \arg \max \{ \alpha \in (0,1] \,|\, S + \alpha \Delta S \succeq 0 \}	\\
	      \alpha &= \texttt{step}\cdot\min(\alpha_p,\alpha_d)
	     
	where :math:`\texttt{step}` is the value of the option :const:`'step'` (default: 0.98).
	      
   :const:`'prediction'` (default: :const:`True`)  
	:const:`True` or :const:`False`. This option is effective only
        when :const:`'eta'` is :const:`None`. If
        :const:`'prediction'` is :const:`True`, a step in the
        approximate tangent direction is never taken but only used to
        predict the duality gap.  If :const:`'prediction'` is :const:`False`, 
	a step in the approximate tangent direction is taken.
    
   :const:`'step'` (default: 0.98)
        positive float between 0 and 1.

   :const:`'lifting'` (default: :const:`True`)
        :const:`True` or :const:`False`; determines whether or not to 
	apply lifting before taking a step in the approximate tangent direction.

   :const:`'show_progress'` 
        :const:`True` or :const:`False`; turns the
        output to the screen on or off (default: :const:`True`).

   :const:`'maxiters'` 
        maximum number of iterations (default: :const:`100`).

   :const:`'abstol'` 
    	absolute accuracy (default: :const:`1e-6`).

   :const:`'reltol'` 
        relative accuracy (default: :const:`1e-6`).

   :const:`'feastol'`
        tolerance for feasibility conditions (default: :const:`1e-8`).

   :const:`'refinement'` 
   	number of iterative refinement steps when solving KKT equations 
   	(default: :const:`1`).

   :const:`'cholmod'`
	use Cholmod's AMD embedding (defaults: :const:`False`).

   :const:`'dimacs'`
	report DIMACS error measures (default: :const:`True`).
	
.. function:: smcp.solvers.chordalsolver_esd(A, b[, primalstart[, dualstart[, scaling='primal'[, kktsolver='chol']]]])

   Solves the pair of cone programs (P) and (D) using an extended self-dual embedding. This solver is currently experimental.
   
   The columns of the sparse matrix ``A`` are vectors of length
   :math:`n^2` and the :math:`m+1` columns of ``A`` are:

   .. math::

      \left[ \mathbf{vec}(C) \ \mathbf{vec}(A_1) \ \cdots \ \mathbf{vec}(A_m) \right].

   Only the rows of ``A`` corresponding to the lower triangular
   elements of the aggregate sparsity pattern :math:`V` are accessed.
   
   The optional argument `primalstart` is a dictionary with the key
   `x` which can be used to specify an initial value for the primal
   variable :math:`X`.  Similarly, the optional argument `dualstart`
   must be a dictionary with keys `s` and `y`.

   The optional argument `scaling` takes one of the values
   :const:`'primal'` (default) or :const:`'dual'`.

   The optional argument `kktsolver` is used to specify the KKT
   solver. Possible values include:

   :const:`'chol'` (default)
	solves the KKT system via a Cholesky factorization of the Schur complement  

   :const:`'qr'`
   	solves the KKT system via a QR factorization

   The solver returns a dictionary with the following keys:

   :const:`'primal objective'`, :const:`'dual objective'`
        primal objective value and dual objective value.

   :const:`'primal infeasibility'`, :const:`'dual infeasibility'`
   	residual norms of primal and dual infeasibility.
   			   
   :const:`'x'`, :const:`'s'`, and :const:`'y'`
   	primal and dual variables.

   :const:`'iterations'`
	number of iterations.

   :const:`'cputime'`, :const:`'time'`
   	total cputime and real time.

   :const:`'gap'`
	duality gap.

   :const:`'relative gap'`	
   	relative duality gap.

   :const:`'status'`
	* has the value :const:`'optimal'` if 
   
	  .. math::
	   
		\frac{(1/\tau) \| \tau b-\mathcal{A}(X) \|_2}{\max\{1,\|b\|_2 \}} \leq \epsilon_{\rm feas}, \qquad 
		\frac{(1/\tau) \|\mathcal{A}^{\rm adj}(y) + S - \tau C\|_F}{\max\{1,\|C\|_F \}} \leq \epsilon_{\rm feas}, \qquad
		X \succ_{\rm c} 0,\qquad
      	   	S \succ 0,
      
	  and

    	  .. math::
   
	      \frac{X\bullet S}{\tau^2} \leq \epsilon_{\rm abs} \quad \text{or} \quad \left( \min\{C\bullet X, -b^Ty\} \leq 0 , \frac{(1/\tau) X\bullet S}{-\min\{C\bullet X,-b^Ty \}} \leq \epsilon_{\rm rel} \right).


   	* has the value :const:`'primal infeasible'` if    
   
	  .. math::
      
	     b^Ty = 1, \qquad
      	     \frac{\| \mathcal{A}^{\rm adj}(y) + S \|_F}{\max\{1,\|C\|_F\}} \leq \epsilon_{\rm feas}, \qquad
	     S \succ 0.

   	* has the value :const:`'dual infeasible'` if 
   
    	  .. math::
       
	     C\bullet X = -1, \qquad
      	     \frac{\| \mathcal{A}(X) \|_2}{\max\{1,\|b\|_2\}} \leq \epsilon_{\rm feas}, \qquad
             X \succ_{\rm c} 0.

	* has the value :const:`'unknown'` if maximum number iterations is reached or if a numerical error is encountered.

   The following options can be set using the dictionary :const:`smcp.solvers.options`:
   
   :const:`'show_progress'`  
        :const:`True` or :const:`False`; turns the output to the screen on or 
        off (default: :const:`True`).

   :const:`'maxiters'` 
        maximum number of iterations (default: :const:`100`).

   :const:`'abstol'` 
    	absolute accuracy (default: :const:`1e-6`).

   :const:`'reltol'` 
        relative accuracy (default: :const:`1e-6`).

   :const:`'feastol'`
        tolerance for feasibility conditions (default: :const:`1e-8`).

   :const:`'refinement'` 
   	number of iterative refinement steps when solving KKT equations 
   	(default: :const:`1`).

   :const:`'cholmod'`
	use Cholmod's AMD embedding (defaults: :const:`False`).

   :const:`'dimacs'`
	report DIMACS error measures (default: :const:`True`).


Solver interfaces
"""""""""""""""""

The following functions implement CVXOPT-like interfaces to the experimental solver :func:`chordalsolver_esd <smcp.solvers.chordalsolver_esd>`. 

.. function:: smcp.solvers.conelp(c, G, h[, dims[, kktsolver='chol']])

   Interface to :func:`chordalsolver_esd <smcp.solvers.chordalsolver_esd>`.

.. function:: smcp.solvers.lp(c, G, h[, kktsolver='chol'])

   Interface to :func:`conelp <smcp.solvers.conelp>`; see `CVXOPT documentation <http://abel.ee.ucla.edu/cvxopt/documentation/>`_ for more information.

.. function:: smcp.solvers.socp(c[, Gl, hl[, Gq, hq[, kktsolver='chol']]])

   Interface to :func:`conelp <smcp.solvers.conelp>`; see `CVXOPT documentation <http://abel.ee.ucla.edu/cvxopt/documentation/>`_ for more information.

.. function:: smcp.solvers.sdp(c[, Gl, hl[, Gs, hs[, kktsolver='chol']]])
   
   Interface to :func:`conelp <smcp.solvers.conelp>`; see `CVXOPT documentation <http://abel.ee.ucla.edu/cvxopt/documentation/>`_ for more information.

The SDP object
""""""""""""""

.. class:: SDP(filename)

   Class for SDP problems. Simplifies reading and writing SDP data files and includes a wrapper for :func:`chordalsolver_esd <smcp.solvers.chordalsolver_esd>`.

   The constructor accepts sparse SDPA data files (extension 'dat-s') and data files created with the :meth:`save <SDP.save>` method (extension 'pkl'). Data files compressed with Bzip2 can also be read (extensions 'dat-s.bz2' and 'pkl.bz2'). 

   .. attribute:: m
      
      Number of constraints.

   .. attribute:: n
      
      Order of semidefinite variable.

   .. attribute:: A

      Problem data: sparse matrix of size :math:`n^2 \times (m+1)` with columns :math:`\mathbf{vec}(C),\mathbf{vec}(A_1),\ldots,\mathbf{vec}(A_m)`. Only the lower triangular elements of :math:`C,A_1,\ldots,A_m` are stored.
   
   .. attribute:: b
   
      Problems data: vector of length :math:`m`.

   .. attribute:: V

      Sparse matrix with aggregate sparsity pattern (lower triangle).

   .. attribute:: nnz
   
      Number of nonzero elements in lower triangle of aggregate sparsity pattern.

   .. attribute:: nnzs

      Vector with number of nonzero elements in lower triangle of :math:`A_0,\ldots,A_m`.

   .. attribute:: nzcols

      Vector with number of nonzero columns in :math:`A_1,\ldots,A_m`.

   .. attribute:: issparse

      True if the number of nonzeros is less than :math:`0.5 \cdot n(n+1)/2`, otherwise false.

   .. attribute:: ischordal	
      
      True if aggregate sparsity pattern is chordal, otherwise false.
	       
   .. method:: get_A(i)

      Returns the :math:`i`'th coeffiecient matrix :math:`A_i` (:math:`0\leq i \leq m`) as a sparse matrix. Only lower triangular elements are stored.      	       

   .. method:: write_sdpa([fname[, compress=False]])

      Writes SDP data to SDPA sparse data file. The extension 'dat-s' is automatically added to the filename. The method is an interface to :func:`sdpa_write <smcp.misc.sdpa_write>`.
      
      If ``compress`` is true, the data file is compressed with Bzip2 and 'bz2' is appended to the filename.

   .. method:: save([fname[, compress=False]])

      Writes SDP data to file using cPickle. The extension 'pkl' is automatically added to the filename.  
            
      If ``compress`` is true, the data file is compressed with Bzip2 and 'bz2' is appended to the filename.
   
   .. method:: solve_feas([scaling='primal'[, kktsolver='chol'[, primalstart, [ dualstart]]]])

      Interface to the feasible start solver :func:`chordalsolver_feas <smcp.solvers.chordalsolver_feas>`. Returns dictionary with solution.

   .. method:: solve_phase1([kktsolver='chol'[,M=1e5]])
   
      Solves a Phase I problem to find a feasible (primal) starting point:

      .. math::

      	 \begin{array}{ll}
	   \mbox{minimize} & s \\
	   \mbox{subject to} & A_i \bullet X = b_i, \quad i=i,\ldots,m \\
	                     & \mathbf{tr}(X) \leq M \\
	                     & X + (s-\epsilon)I \succeq_{\rm c} 0,\, s \geq 0
	 \end{array}

      The variables are :math:`X \in \mathbf{S}_V^n` and :math:`s \in \mathbf{R}`, and :math:`\epsilon \in \mathbf{R}_{++}` is a small constant. If :math:`s^\star < \epsilon`, the method returns :math:`X^\star` (which is a strictly feasible starting point in the original problem) and a dictionary (with information about the Phase I problem). If :math:`s >= \epsilon` the method returns (`None`, `None`).

   .. method:: solve_esd([scaling='primal'[, kktsolver='chol'[, primalstart, [ dualstart]]]])

      Interface to :func:`chordalsolver_esd <smcp.solvers.chordalsolver_esd>`. Returns dictionary with solution.


   .. method:: solve_cvxopt([primalstart, [ dualstart]])

      Interface to :func:`cvxopt.solvers.sdp`. Returns dictionary with solution. (Note that this simple interface does not yet specify block structure properly.)

The following example demostrates how to load and solve a problem from an SDPA sparse data file:

>>> from smcp import SDP
>>> P = SDP('qpG11.dat-s')
>>> print P
<SDP: n=1600, m=800, nnz=3200> qpG11
>>> sol = P.solve_feas(kktsolver='chol')
>>> print sol['primal objective']
-2448.6588977
>>> print sol['dual objective']
-2448.65913565
>>> print sol['gap']
0.00023794772363
>>> print sol['relative gap']
9.71747121876e-08

Auxiliary routines
""""""""""""""""""
.. function:: smcp.completion(X)
   
   Computes the maximum determinant positive definite completion of a sparse matrix X.

   Example:

   >>> from smcp import mtxnorm_SDP, completion
   >>> P = mtxnorm_SDP(p=10,q=2,r=10)
   >>> sol = P.solve_feas(kktsolver='chol')
   >>> X = completion(sol['x'])

.. function:: smcp.misc.ind2sub(siz, ind)

   Converts indices to subscripts.

   :param siz: matrix order
   :type siz: integer
   :param ind: vector with indices
   :type ind: matrix   
   :returns: matrix ``I`` with row subscripts and matrix ``J`` with column subscripts

.. function:: smcp.misc.sub2ind(siz, I, J)

   Converts subscripts to indices.

   :param siz: matrix size
   :type siz: integer tuple
   :param I: row subscripts
   :type I: matrix
   :param J: column subscripts
   :type J: matrix
   :returns: matrix with indices

.. function:: smcp.misc.sdpa_read(file_obj)

   Reads data from sparse SDPA data file (file extension: 'dat-s').  
   A description of the sparse SDPA file format can be found in the document `SDPLIB/FORMAT <http://infohost.nmt.edu/~sdplib/FORMAT>`_ and in the `SDPA User's Manual <http://sdpa.indsys.chuo-u.ac.jp/sdpa/download.html>`_.

   Example:

   >>> f = open('qpG11.dat-s')
   >>> A, b, blockstruct = smcp.misc.sdpa_read(f)
   >>> f.close()


.. function:: smcp.misc.sdpa_readhead(file_obj)

   Reads header from sparse SDPA data file and returns the order :math:`n`, the number of constraints :math:`m`, and a vector with block sizes.

   Example:

   >>> f = open('qpG11.dat-s')
   >>> n, m, blockstruct = smcp.misc.sdpa_readhead(f)
   >>> f.close()

.. function:: smcp.misc.sdpa_write(file_obj, A, b, blockstruct)

   Writes SDP data to sparse SDPA file.

   Example:
   
   >>> f = open('my_data_file.dat-s','w')
   >>> smcp.misc.sdpa_write(f,A,b,blockstruct)
   >>> f.close()

Analysis routines
"""""""""""""""""

.. function:: smcp.analysis.embed_SDP(P[, order[, cholmod]])

   Computes chordal embedding and returns SDP object with chordal sparsity pattern.

   :param P: SDP object with problem data
   :type P: :class:`SDP`
   :param order: 'AMD' (default) or 'METIS'
   :type order: string
   :param cholmod: use Cholmod to compute embedding (default is false)
   :type cholmod: boolean
   :returns: SDP object with chordal sparsity

   Note that CVXOPT must be compiled and linked to METIS in order to use the METIS ordering.

The following routines require `Matplotlib <http://matplotlib.sourceforge.net>`_:

.. function:: smcp.analysis.spy(P[, i[, file[, scale]]])

   Plots aggregate sparsity pattern of SDP object ``P`` or sparsity pattern of :math:`A_i`. 

   :param P: SDP object with problem data
   :type P: :class:`SDP`
   :param i: index between 0 and m
   :type i: integer
   :param file: saves plot to file
   :type file: string
   :param scale: downsamples plot 
   :type scale: float

.. function:: smcp.analysis.clique_hist(P)

   Plots clique histogram if ``P.ischordal`` is true, and otherwise an exception is raised.

   :param P: SDP object with problem data	     
   :type P: :class:`SDP`

.. function:: smcp.analysis.nnz_hist(P)

   Plots histogram of number of nonzeros in lower triangle of :math:`A_1,\ldots,A_m`.

   :param P: SDP object with problem data	     
   :type P: :class:`SDP`


Random problem generators
"""""""""""""""""""""""""

.. class:: mtxnorm_SDP(p, q, r[, density[, seed]])

   Inherits from :class:`SDP` class.
   
   Generates random data :math:`F_i,G \in \mathbf{R}^{p\times q}` for the matrix norm minimization problem

   .. math::

       \begin{array}{ll}
       \mbox{minimize} & \lVert F(z) + G \rVert_2
       \end{array}

   with the variable :math:`z\in \mathbf{R}^r` and where :math:`F(z) = z_1F_1 + \cdots z_r F_r`. The problem is cast as an equivalent SDP:

   .. math::

       \begin{array}{ll}
       \mbox{minimize} & t \\
       \mbox{subject to} & 
       \begin{bmatrix} 
          tI & (F(z)+G)^T \\
	  F(z)+G & tI
       \end{bmatrix} \succeq 0.
       \end{array}
   
   The sparsity of :math:`F_i` can optionally be chosen by specifying the parameter ``density`` which must be a float between 0 and 1 (default is 1 which corresponds to dense matrices). 

   Example:

   >>> from smcp import mtxnorm_SDP
   >>> P = mtxnorm_SDP(p=200,q=10,r=200)
   >>> print P
   <SDP: n=210, m=201, nnz=2210> mtxnorm_p200_q10_r200
   >>> sol = P.solve_feas(kktsolver='qr')

.. class:: base.band_SDP(n, m, bw[, seed])

   Generates random SDP with band sparsity and `m` constraints, of order `n`, and with bandwidth `bw` (`bw=0` corresponds to a diagonal, `bw=1` is tridiagonal etc.). Returns :class:`SDP` object. The optional parameter `seed` sets the random number generator seed.

   Example:

   >>> from smcp import band_SDP
   >>> P = band_SDP(n=100,m=100,bw=2,seed=10)
   >>> print P
   <SDP: n=100, m=100, nnz=297> band_n100_m100_bw2
   >>> X,p1sol = P.solve_phase1(kktsolver='qr')
   >>> P.solve_feas(kktsolver='qr',primalstart={'x':X})
   >>> print sol['primal objective'],sol['dual objective']
   31.2212701455 31.2212398351


.. class:: base.rand_SDP(V, m[, density[, seed]])

   Generates random SDP with sparsity pattern V and m constraints. Returns :class:`SDP` object.

   The sparsity of :math:`A_i` can optionally be chosen by specifying the parameter ``density`` which must be a float between 0 and 1 (default is 1 which corresponds to dense matrices). 
