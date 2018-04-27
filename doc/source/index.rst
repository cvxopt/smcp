###########################################################
SMCP --- Python extension for Sparse Matrix Cone Programs
###########################################################

.. title:: Home

SMCP is a software package for solving linear sparse matrix cone programs.
The code is experimental and it is released to accompany the following paper:

.. seealso::

      M. S. Andersen, J. Dahl, and L. Vandenberghe, `Implementation of
	 nonsymmetric interior-point methods for linear optimization
	 over sparse matrix cones
	 <http://doi.org/10.1007/s12532-010-0016-2>`_,
	 *Mathematical Programming Computation*, 2010.


The package provides an implementation of a nonsymmetric interior-point method which is based on chordal matrix techniques. Only one type of cone is used, but this cone includes the three canonical cones ---
the nonnegative orthant, the second-order cone, and the positive semidefinite cone --- as special cases.
The efficiency of the solver depends not only on the dimension of the cone, but also on its *structure*. Nonchordal sparsity patterns are handled using chordal embedding techniques.

In its current form, SMCP is implemented in Python and C, and it relies on the Python extensions `CHOMPACK <http://cvxopt.github.io/chompack>`_ and `CVXOPT <http://cvxopt.org>`_ for most computations.

Current release
--------------------

Version 0.4.5 (December 2017) includes:

* Nonsymmetric feasible start interior-point methods (primal and dual scaling methods)
* Two KKT system solvers: one solves the symmetric indefinite augmented system and the other solves the positive definite system of normal equations
* Read/write routines for SDPA sparse data files ('dat-s').
* Simple interface to CVXOPT SDP solver


Future releases
--------------------

We plan to turn SMCP into a C library with Python and Matlab interfaces. Future releases may include additional functionality as listed below:

* Explicitly handle free variables
* Iterative solver for the KKT system
* Automatic selection of KKT solver and chordal embedding technique


Availability
--------------------

The source package is available from the :doc:`download/index` section. The source package
includes source code, documentation, and installation guidelines.


Authors
--------------------

SMCP is developed by `Martin S. Andersen <http://compute.dtu.dk/~mskan>`_ and `Lieven Vandenberghe <http://www.ee.ucla.edu/~vandenbe>`_ .

Feedback and bug reports
-------------------------

We welcome feedback, and bug reports are much appreciated. Please
report bugs through our `Github repository <https://github.org/cvxopt/chompack>`_.

.. toctree::
      :hidden:

      copyright
      download/index
      documentation/index
      testproblems/index
      benchmarks/index
