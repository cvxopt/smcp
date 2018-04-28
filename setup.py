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

from setuptools import setup, Extension
import os, sys
import versioneer

if sys.version < '2.7':
    sys.exit('ERROR: Sorry, python 2.7 is required for this extension.')

LIBRARIES = os.environ.get("SMCP_LIBRARIES",[])
if type(LIBRARIES) is str: LIBRARIES = LIBRARIES.strip().split(';')

EXTRA_COMPILE_ARGS = os.environ.get("SMCP_EXTRA_COMPILE_ARGS",[])
if type(EXTRA_COMPILE_ARGS) is str: EXTRA_COMPILE_ARGS = EXTRA_COMPILE_ARGS.strip().split(';')

misc = Extension('misc',
#                 include_dirs = [ CVXOPT_SRC ],
                 libraries = LIBRARIES,
                 extra_compile_args = EXTRA_COMPILE_ARGS,
                 sources = ['src/C/misc.c'])

setup(name="smcp",
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description="Python extension for solving sparse matrix cone programs",
      author="Martin S. Andersen and Lieven Vandenberghe",
      author_email="martin.skovgaard.andersen@gmail.com, vandenbe@ee.ucla.edu",
      url="http://cvxopt.github.io/smcp",
      license = 'GNU GPL version 3',
      packages = ['smcp',],
      package_dir = {'smcp':'src/python'},
      ext_package = "smcp",
      ext_modules = [misc],
      install_requires = ["cvxopt (>=1.1.9)","chompack (>=2.3.2)"])
