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
"""
Python extension for chordal matrix cone programs.
"""

from . import solvers
from .base import SDP, mtxnorm_SDP, band_SDP, completion
from cvxopt import matrix, spmatrix, sparse

__all__ = ['solvers','SDP','mtxnorm_SDP','band_SDP','completion',\
               'matrix','spmatrix','sparse']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
