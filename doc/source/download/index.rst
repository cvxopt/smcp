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

The package requires Python version 3.8 or newer. To build the
package from source, the Python header files and libraries must be
installed, as well as the core binaries. SMCP also requires the Python
extension modules `CVXOPT 1.3.3 <http://cvxopt.org>`_ or
later and `CHOMPACK 2.3.4 <http://cvxopt.github.io/chompack>`_ or later.

The source package is available `here <https://github.com/cvxopt/smcp>`_. The
extension can be built and installed as follows:

.. code-block:: bash

    git clone https://github.com/cvxopt/smcp.git
    cd smcp
    python -m build --wheel
    pip install dist/smcp-*.whl
    pytest tests
