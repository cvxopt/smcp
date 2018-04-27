#########################
Download and installation
#########################

Installation with pip
-------------------------

SMCP can be installed via pip using the following command:

.. code-block:: bash

    pip install smcp


Installation from source
-------------------------

The package requires Python version 2.7 or newer. To build the
package from source, the Python header files and libraries must be
installed, as well as the core binaries. SMCP also requires the Python
extension modules `CVXOPT 1.1.9 <http://cvxopt.org>`_ or
later and `CHOMPACK 2.3.2 <http://cvxopt.github.io/chompack>`_ or later.

The source package is available `here <https://github.com/cvxopt/smcp>`_. The
extension can be built and installed as follows:

.. code-block:: bash

    python setup.py install
    python example.py
