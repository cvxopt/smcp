# Copyright 2010-2017 M. S. Andersen & L. Vandenberghe
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

import smcp

# Generate random matrix norm minimization problem
q = 5; p = 50-q; m = 40
P = smcp.mtxnorm_SDP(p,q,m)

#
try:
    # try feasible solver
    sol = P.solve_feas(kktsolver='qr',scaling='primal')
except:
    # solve phase 1 problem to find feasible starting point
    X0,p1sol = P.solve_phase1(kktsolver='qr')
    # try feasible solver with feasible starting point
    sol = P.solve_feas(kktsolver='qr',primalstart={'x':X0})
