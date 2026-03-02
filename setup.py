# Copyright 2010-2026 M. S. Andersen & L. Vandenberghe
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
import os

LIBRARIES = os.environ.get("SMCP_LIBRARIES", [])
if isinstance(LIBRARIES, str):
    LIBRARIES = LIBRARIES.strip().split(";")

EXTRA_COMPILE_ARGS = os.environ.get("SMCP_EXTRA_COMPILE_ARGS", [])
if isinstance(EXTRA_COMPILE_ARGS, str):
    EXTRA_COMPILE_ARGS = EXTRA_COMPILE_ARGS.strip().split(";")

misc = Extension(
    "misc",
    libraries=LIBRARIES,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    sources=["src/C/misc.c"],
)

setup(ext_package="smcp", ext_modules=[misc])
