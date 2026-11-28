"""Spherical harmonic vector wind analysis for Skyborn.

This module is based on the ajdawson/windspharm project (https://github.com/ajdawson/windspharm),
originally authored by Andrew Dawson. The current version is maintained by Qianye Su and is licensed under the BSD-3-Clause.

Main Classes:
    VectorWind: Enhanced interface for wind analysis on regular grids
    ReducedVectorWind: Packed reduced-Gaussian wind analysis

Example:
    >>> from skyborn.windspharm import VectorWind
    >>> import numpy as np
    >>>
    >>> # Create sample wind data
    >>> nlat, nlon = 73, 144
    >>> u = np.random.randn(nlat, nlon)
    >>> v = np.random.randn(nlat, nlon)
    >>>
    >>> # Initialize VectorWind with type hints and modern interface
    >>> vw = VectorWind(u, v, gridtype='gaussian')
    >>>
    >>> # Calculate various fields with improved documentation
    >>> vorticity = vw.vorticity()
    >>> divergence = vw.divergence()
    >>> psi, chi = vw.sfvp()

Notes:
    ``VectorWind`` and ``ReducedVectorWind`` default to ``layout="auto"``, which
    automatically detects C-order ``(..., nlat, nlon)`` arrays (zero-copy) as
    well as legacy ``(nlat, nlon, ...)`` arrays without requiring manual
    transposition. Pass ``lat_order="S2N"`` or ``lat=...`` to process
    south-to-north data directly, or use the ``xarray`` interface for
    automatic coordinate and metadata handling.
"""

from __future__ import absolute_import

from . import reduced, standard, tools, xarray
from .reduced import ReducedVectorWind
from .standard import VectorWind

__all__ = [
    "VectorWind",
    "ReducedVectorWind",
    "standard",
    "reduced",
    "tools",
    "xarray",
]


__author__ = "Qianye Su"
__license__ = "BSD-3-Clause"
