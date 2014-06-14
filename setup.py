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

from distutils.core import setup, Extension
import os, sys

if sys.version < '2.6':
    sys.exit('ERROR: Sorry, python 2.6 is required for this extension.')

LIBRARIES = ['gomp']
EXTRA_COMPILE_ARGS = ['-fopenmp']
# Compile without OpenMP under Mac OS 
if sys.platform.startswith('darwin'):
    LIBRARIES.remove('gomp')
    EXTRA_COMPILE_ARGS.remove('-fopenmp')

misc = Extension('misc',
#                 include_dirs = [ CVXOPT_SRC ],
                 libraries = LIBRARIES,
                 extra_compile_args = EXTRA_COMPILE_ARGS,                    
                 sources = ['src/C/misc.c'])

setup(name="smcp",
      version="0.4",
      description="Python extension for solving sparse matrix cone programs",
      author="Martin S. Andersen and Lieven Vandenberghe",
      author_email="martin.skovgaard.andersen@gmail.com, vandenbe@ee.ucla.edu",
      url="http://cvxopt.github.io/smcp",
      license = 'GNU GPL version 3',
      packages = ['smcp',],
      package_dir = {'smcp':'src/python'},
      ext_package = "smcp",
      ext_modules = [misc],
      requires = ["cvxopt (>=1.1.6)","chompack (>=2.0)"])
