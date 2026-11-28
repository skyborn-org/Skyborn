"""
Spherical harmonic vector wind computations.

This module provides the main VectorWind class for analyzing atmospheric wind fields
using spherical harmonic transforms. It enables calculation of vorticity, divergence,
streamfunction, velocity potential, and other dynamical quantities.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

from skyborn.spharm import Spharmt, gaussian_lats_wts
from skyborn.spharm import spherical_harmonics as spharm_mod

__all__ = ["VectorWind"]

_DTYPE_F32 = np.dtype(np.float32)
_DTYPE_F64 = np.dtype(np.float64)
_LAYOUT_ALIASES = {"auto": "auto", "c": "c", "legacy": "legacy"}
_GRIDTYPE_IS_GAUSSIAN = {"regular": False, "gaussian": True}


def _score_grid_pair(nlat: int, nlon: int, is_gaussian: bool) -> int:
    """Score how plausible a (nlat, nlon) pair is for a global spherical grid."""
    if nlat < 3 or nlon < 4:
        return -1
    if not is_gaussian:
        if nlon == 2 * (nlat - 1) or nlon == 2 * nlat:
            return 4
    elif (nlon & 1) == 0 and (
        nlon == 2 * nlat or (5 * nlon >= 9 * nlat and 4 * nlon <= 9 * nlat)
    ):
        return 4
    # 1.2 <= (nlon / nlat) <= 4.0 already implies nlon > nlat for nlat >= 3
    if 5 * nlon >= 6 * nlat and nlon <= 4 * nlat:
        return 3 if (nlon & 1) == 0 else 2
    return 1 if nlon >= nlat else 0


@lru_cache(maxsize=256)
def _cached_resolve_auto_layout(
    shape: Tuple[int, ...], is_gaussian: bool, f_only: bool
) -> str:
    """Resolve and cache 'auto' layout ('c' vs 'legacy') from array shape and order."""
    score_c = _score_grid_pair(shape[-2], shape[-1], is_gaussian)
    score_l = _score_grid_pair(shape[0], shape[1], is_gaussian)
    if score_c > score_l:
        return "c"
    if score_l > score_c:
        return "legacy"
    return "legacy" if f_only else "c"

# Type aliases for better readability
ArrayLike = Union[np.ndarray, np.ma.MaskedArray]
GridType = Literal["regular", "gaussian"]
LegFunc = Literal["stored", "computed"]
PrecisionMode = Literal["auto", "single", "float32", "double", "float64"]
LayoutMode = Literal["legacy", "c", "auto"]
LatOrderMode = Literal["north_to_south", "south_to_north", "N2S", "S2N", "auto"]


class VectorWind:
    """
    Vector wind analysis using spherical harmonic transforms.

    This class provides methods for analyzing atmospheric wind fields through
    spherical harmonic decomposition. It can compute vorticity, divergence,
    streamfunction, velocity potential, and perform Helmholtz decomposition.

    Parameters
    ----------
    u : array_like
        Zonal wind component. Shape may be ``(..., nlat, nlon)`` (C-order,
        zero-copy), ``(nlat, nlon)``, or legacy ``(nlat, nlon, *extra_dims)``.
        When ``layout="auto"`` (default), both conventions are recognized
        automatically without manual transposition.
    v : array_like
        Meridional wind component. Must have same shape as u.
    gridtype : {'regular', 'gaussian'}, default 'regular'
        Type of input grid:
        - 'regular': evenly-spaced lat/lon grid
        - 'gaussian': Gaussian latitude grid
    rsphere : float, default 6.3712e6
        Earth radius in meters for spherical harmonic computations.
    legfunc : {'stored', 'computed'}, default 'stored'
        Legendre function computation method:
        - 'stored': precompute and store (faster, more memory)
        - 'computed': compute on-the-fly (slower, less memory)
    precision : {'auto', 'single', 'float32', 'double', 'float64'}, default 'auto'
        Public output precision mode. 'auto' preserves the promoted input
        floating precision, 'single'/'float32' returns float32/complex64
        outputs, and 'double'/'float64' returns float64/complex128 outputs.
    layout : {'auto', 'c', 'legacy'}, default 'auto'
        Array axis convention. ``'auto'`` detects C-order ``(..., nlat, nlon)``
        and legacy ``(nlat, nlon, ...)`` automatically (preferring C-order).
    lat_order : {'auto', 'N2S', 'S2N', 'north_to_south', 'south_to_north'}, default 'auto'
        Latitude direction of the input grid (supports zero-copy south-to-north).
    lat : ndarray, optional
        Optional 1D latitude coordinate array for automatic ``S2N``/``N2S`` detection.

    Attributes
    ----------
    u : ndarray
        Zonal wind component
    v : ndarray
        Meridional wind component
    gridtype : str
        Grid type used
    rsphere : float
        Sphere radius in meters used by the underlying transform
    legfunc : str
        Legendre function handling mode used by the underlying transform
    precision : str
        Public output precision mode
    layout : str
        Resolved array layout mode (``'c'`` or ``'legacy'``)
    lat_order : str
        Normalized latitude order setting
    south_to_north : bool
        Whether the instance operates on south-to-north latitude rows
    s : Spharmt
        Spherical harmonic transform object

    Examples
    --------
    >>> import numpy as np
    >>> from skyborn.windspharm import VectorWind
    >>>
    >>> # Create sample wind field
    >>> nlat, nlon = 73, 144
    >>> u = np.random.randn(12, nlat, nlon)
    >>> v = np.random.randn(12, nlat, nlon)
    >>>
    >>> # Initialize VectorWind (C-order detected automatically)
    >>> vw = VectorWind(u, v, gridtype='gaussian')
    >>>
    >>> # Calculate dynamical quantities
    >>> vorticity = vw.vorticity()
    >>> divergence = vw.divergence()
    >>> psi, chi = vw.sfvp()

    Notes
    -----
    With the Pure-Rust ``skybornspharm`` backend, C-order ``(..., nlat, nlon)``
    arrays and south-to-north grids (via ``lat_order="S2N"`` or ``lat=...``)
    execute directly without extra transposition or latitude copying.
    """

    def __init__(
        self,
        u: ArrayLike,
        v: ArrayLike,
        gridtype: GridType = "regular",
        rsphere: float = 6.3712e6,
        legfunc: LegFunc = "stored",
        precision: PrecisionMode = "auto",
        layout: LayoutMode = "auto",
        lat_order: LatOrderMode = "auto",
        lat: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize VectorWind instance with comprehensive data validation.

        This method performs thorough validation of input wind components including
        checks for missing values, infinite values, shape compatibility, and
        dimensional requirements.

        Args:
            u: Zonal wind component with latitude/longitude axes.
            v: Meridional wind component with the same shape as ``u``.
            gridtype: Grid type - ``"regular"`` or ``"gaussian"``.
            rsphere: Sphere radius in meters.
            legfunc: Legendre function handling mode - ``"stored"`` or
                ``"computed"``.
            precision: Public output precision mode. ``"auto"`` preserves the
                promoted input floating precision, ``"single"`` returns
                float32 outputs, and ``"double"`` returns float64 outputs.
            layout: Array axis ordering - ``"auto"`` (default, auto-detects
                C-order ``(..., lat, lon)`` and legacy ``(lat, lon, ...)``),
                ``"c"`` for ``(..., lat, lon)``, or ``"legacy"`` for ``(lat, lon, ...)``.
            lat_order: Latitude ordering mode (``"north_to_south"``,
                ``"south_to_north"``, ``"N2S"``, ``"S2N"``, or ``"auto"``).
            lat: Optional 1D latitude array for south-to-north auto-detection.

        Raises:
            ValueError: If the input arrays fail shape, dimensional, or finite
                value validation.
        """
        # Step 1: Handle masked arrays and resolve layout from original array metadata
        raw_u = u if type(u) is np.ndarray else np.asarray(u)
        self._output_dtype = self._infer_output_dtype(raw_u, v)
        self._precision = Spharmt._validate_precision(precision)
        self._lat_order = lat_order
        self._lat = None if lat is None else np.asarray(lat, dtype=np.float64)
        norm_layout = self._normalize_layout(layout)
        self._layout = self._resolve_layout(
            raw_u.shape, raw_u.flags, norm_layout, gridtype, self._lat
        )
        self.u = self._process_input_array(u, "u")
        self.v = self._process_input_array(v, "v")

        # Step 2: Comprehensive data validation
        self._validate_input_data()

        # Step 3: Validate dimensions and shapes
        self._validate_dimensions()

        # Step 4: Initialize spherical harmonic transform
        nlat, nlon = self._horizontal_shape()
        spharm_layout = (
            "auto" if (norm_layout == "auto" and self.u.ndim == 2) else self._layout
        )
        self._initialize_spharmt(
            nlon,
            nlat,
            gridtype,
            rsphere,
            legfunc,
            precision,
            layout=spharm_layout,
            lat_order=self._lat_order,
            lat=self._lat,
        )
        self._setup_cached_arrays()

        # Step 5: Create method aliases for backward compatibility
        self._setup_method_aliases()

    @classmethod
    def _from_owned_input(
        cls,
        u: ArrayLike,
        v: ArrayLike,
        gridtype: GridType = "regular",
        rsphere: float = 6.3712e6,
        legfunc: LegFunc = "stored",
        precision: PrecisionMode = "auto",
        layout: LayoutMode = "auto",
        lat_order: LatOrderMode = "auto",
        lat: Optional[np.ndarray] = None,
    ) -> "VectorWind":
        """Create an instance from validated wrapper-owned arrays."""
        self = cls.__new__(cls)
        raw_u = u if type(u) is np.ndarray else np.asarray(u)
        self._output_dtype = self._infer_output_dtype(raw_u, v)
        self._precision = Spharmt._validate_precision(precision)
        self._lat_order = lat_order
        self._lat = None if lat is None else np.asarray(lat, dtype=np.float64)
        norm_layout = self._normalize_layout(layout)
        self._layout = self._resolve_layout(
            raw_u.shape, raw_u.flags, norm_layout, gridtype, self._lat
        )
        self.u = self._process_input_array(u, "u", copy=False)
        self.v = self._process_input_array(v, "v", copy=False)
        self._validate_input_data()
        self._validate_dimensions()
        nlat, nlon = self._horizontal_shape()
        spharm_layout = (
            "auto" if (norm_layout == "auto" and self.u.ndim == 2) else self._layout
        )
        self._initialize_spharmt(
            nlon,
            nlat,
            gridtype,
            rsphere,
            legfunc,
            precision,
            layout=spharm_layout,
            lat_order=self._lat_order,
            lat=self._lat,
        )
        self._setup_cached_arrays()
        self._setup_method_aliases()
        return self

    @staticmethod
    def _normalize_layout(layout: str) -> str:
        """Validate layout option."""
        norm = _LAYOUT_ALIASES.get(layout) if type(layout) is str else None
        if norm is None:
            norm = _LAYOUT_ALIASES.get(str(layout).lower())
        if norm is None:
            raise ValueError(
                f'layout must be "legacy", "c", or "auto", got "{layout}"'
            )
        return norm

    @staticmethod
    def _grid_pair_score(nlat: int, nlon: int, gridtype: str) -> int:
        """Score how plausible a (nlat, nlon) pair is for a global spherical grid."""
        is_gaussian = (
            _GRIDTYPE_IS_GAUSSIAN.get(gridtype) if type(gridtype) is str else None
        )
        if is_gaussian is None:
            is_gaussian = str(gridtype).lower() == "gaussian"
        return _score_grid_pair(nlat, nlon, is_gaussian)

    @classmethod
    def _resolve_layout(
        cls,
        shape: Tuple[int, ...],
        flags: Any,
        layout: str,
        gridtype: str,
        lat: Optional[np.ndarray] = None,
    ) -> str:
        """Resolve 'auto' layout to concrete 'c' or 'legacy' axis order."""
        norm = _LAYOUT_ALIASES.get(layout) if type(layout) is str else None
        if norm is None:
            norm = cls._normalize_layout(layout)
        if norm != "auto":
            return norm
        if len(shape) <= 2:
            return "c"

        if lat is not None and lat.ndim == 1:
            nlat_ref = lat.size
            c_ok = shape[-2] == nlat_ref and shape[-2] >= 3 and shape[-1] >= 4
            l_ok = shape[0] == nlat_ref and shape[0] >= 3 and shape[1] >= 4
            if c_ok and not l_ok:
                return "c"
            if l_ok and not c_ok:
                return "legacy"

        is_gaussian = (
            _GRIDTYPE_IS_GAUSSIAN.get(gridtype) if type(gridtype) is str else None
        )
        if is_gaussian is None:
            is_gaussian = str(gridtype).lower() == "gaussian"

        if flags is None:
            f_only = False
        else:
            try:
                f_only = bool(flags.f_contiguous and not flags.c_contiguous)
            except AttributeError:
                f_only = bool(
                    getattr(flags, "f_contiguous", False)
                    and not getattr(flags, "c_contiguous", True)
                )

        return _cached_resolve_auto_layout(shape, is_gaussian, f_only)

    def _horizontal_shape(self) -> Tuple[int, int]:
        """Return (nlat, nlon) according to configured layout."""
        if getattr(self, "_layout", "legacy") == "c":
            return int(self.u.shape[-2]), int(self.u.shape[-1])
        return int(self.u.shape[0]), int(self.u.shape[1])

    # ------------------------------------------------------------------
    # Input processing, dtype handling, and validation
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_output_dtype(*arrays: ArrayLike) -> np.dtype:
        """Return the public output dtype for floating inputs."""
        if len(arrays) == 2:
            a0, a1 = arrays
            if type(a0) is np.ndarray and type(a1) is np.ndarray:
                d0, d1 = a0.dtype, a1.dtype
                if d0 == _DTYPE_F32 and d1 == _DTYPE_F32:
                    return _DTYPE_F32
                if (d0 == _DTYPE_F64 or d1 == _DTYPE_F64) and (
                    d0 == _DTYPE_F32 or d0 == _DTYPE_F64
                ) and (d1 == _DTYPE_F32 or d1 == _DTYPE_F64):
                    return _DTYPE_F64
        elif len(arrays) == 1:
            a0 = arrays[0]
            if type(a0) is np.ndarray:
                d0 = a0.dtype
                if d0 == _DTYPE_F32:
                    return _DTYPE_F32
                if d0 == _DTYPE_F64:
                    return _DTYPE_F64

        output_dtype = None
        for arr in arrays:
            dtype = arr.dtype if type(arr) is np.ndarray else np.dtype(np.asarray(arr).dtype)
            if np.issubdtype(dtype, np.floating):
                public_dtype = dtype
            else:
                public_dtype = _DTYPE_F64
            output_dtype = (
                public_dtype
                if output_dtype is None
                else np.result_type(output_dtype, public_dtype)
            )
        return _DTYPE_F64 if output_dtype is None else output_dtype

    def _restore_output_dtype(
        self, data: np.ndarray, output_dtype: Optional[np.dtype] = None
    ) -> np.ndarray:
        """Cast grid-space public outputs back to the input precision."""
        array = data if type(data) is np.ndarray else np.asarray(data)
        precision = getattr(self, "_precision", "auto")
        if precision == "double":
            target_dtype = np.float64
        elif precision == "single":
            target_dtype = np.float32
        else:
            target_dtype = (
                self._output_dtype if output_dtype is None else np.dtype(output_dtype)
            )
        if array.dtype == target_dtype:
            return array
        return array.astype(target_dtype, copy=False)

    def _process_input_array(
        self, arr: ArrayLike, name: str, copy: bool = True
    ) -> np.ndarray:
        """
        Process input array handling masked arrays and creating copies.

        Parameters
        ----------
        arr : array_like
            Input array (u or v wind component)
        name : str
            Name of the array ('u' or 'v') for error messages

        Returns
        -------
        ndarray
            Processed numpy array
        """
        if type(arr) is np.ndarray and (arr.dtype == _DTYPE_F32 or arr.dtype == _DTYPE_F64):
            return arr.copy() if copy else arr

        try:
            # Handle masked arrays
            if hasattr(arr, "filled"):
                masked = np.ma.asarray(arr)
                dtype = masked.dtype
                if not np.issubdtype(dtype, np.floating):
                    dtype = _DTYPE_F64
                processed = np.ma.asarray(masked, dtype=dtype).filled(fill_value=np.nan)
                if copy:
                    processed = processed.copy()
            else:
                raw = np.asarray(arr)
                dtype = np.dtype(raw.dtype)
                if not np.issubdtype(dtype, np.floating):
                    dtype = _DTYPE_F64
                processed = np.asarray(raw, dtype=dtype)
                if copy:
                    processed = processed.copy()
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert {name} to numpy array: {e}")

        return processed

    def _validate_input_data(self) -> None:
        """
        Comprehensive validation of input wind data.

        Checks for:
        - Missing values (NaN)
        - Infinite values (±inf)
        - Shape compatibility
        - Data type compatibility
        """
        if not np.isfinite(self.u).all() or not np.isfinite(self.v).all():
            nan_locations, inf_locations = self._invalid_value_locations()

            if nan_locations:
                raise ValueError(
                    f"Input wind components contain missing values (NaN). "
                    f"Found in {', '.join(nan_locations)}. "
                    f"Please ensure all wind data is finite and valid."
                )

            raise ValueError(
                f"Input wind components contain infinite values. "
                f"Found in {', '.join(inf_locations)}. "
                f"Please ensure all wind data is finite."
            )

        # Check for extremely large values that might cause numerical issues
        # u_max = np.abs(self.u).max()
        # v_max = np.abs(self.v).max()
        # max_reasonable = 1e6  # 1000 km/s should be more than enough for any wind

        # if u_max > max_reasonable or v_max > max_reasonable:
        #     raise ValueError(
        #         f"Input wind components contain extremely large values "
        #         f"(max |u|={u_max:.2e}, max |v|={v_max:.2e}). "
        #         f"Please check units and data validity."
        #     )

    def _invalid_value_locations(self) -> Tuple[list[str], list[str]]:
        """Count NaN and infinite values after the fast finite check fails."""
        nan_locations: list[str] = []
        inf_locations: list[str] = []

        for name, values in (("u", self.u), ("v", self.v)):
            invalid = ~np.isfinite(values)
            invalid_count = int(invalid.sum())
            if not invalid_count:
                continue

            invalid_values = values[invalid]
            nan_count = int(np.isnan(invalid_values).sum())
            inf_count = invalid_count - nan_count
            if nan_count:
                nan_locations.append(f"{name}: {nan_count} NaN values")
            if inf_count:
                inf_locations.append(f"{name}: {inf_count} infinite values")

        return nan_locations, inf_locations

    def _validate_dimensions(self) -> None:
        """
        Validate array dimensions and shapes.
        """
        # Check shape compatibility
        if self.u.shape != self.v.shape:
            raise ValueError(
                f"Wind components must have identical shapes. "
                f"Got u: {self.u.shape}, v: {self.v.shape}"
            )

        # Check dimensionality
        if self.u.ndim < 2:
            raise ValueError(
                f"Wind components must be at least 2D arrays. "
                f"Got {self.u.ndim}D arrays with shape {self.u.shape}. "
                f"Expected shape: (nlat, nlon) or (nlat, nlon, *extra_dims)"
            )

        # Check minimum grid size
        nlat, nlon = self._horizontal_shape()
        if nlat < 3 or nlon < 4:
            raise ValueError(
                f"Grid too small for spherical harmonic analysis. "
                f"Got (nlat={nlat}, nlon={nlon}). "
                f"Minimum requirements: nlat ≥ 3, nlon ≥ 4"
            )

        # Check for reasonable grid sizes
        # if nlat > 10000 or nlon > 10000:
        #     import warnings

        #     warnings.warn(
        #         f"Very large grid detected (nlat={nlat}, nlon={nlon}). "
        #         f"This may require significant memory and computation time.",
        #         UserWarning,
        #     )

    # ------------------------------------------------------------------
    # Transform setup and small per-instance caches
    # ------------------------------------------------------------------

    def _initialize_spharmt(
        self,
        nlon: int,
        nlat: int,
        gridtype: str,
        rsphere: float,
        legfunc: str,
        precision: Literal["auto", "single", "double"] = "auto",
        layout: Literal["legacy", "c", "auto"] = "legacy",
        lat_order: str = "auto",
        lat: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize spherical harmonic transform object with validation.

        Parameters
        ----------
        nlon, nlat : int
            Grid dimensions
        gridtype : str
            Grid type
        rsphere : float
            Sphere radius
        legfunc : str
            Legendre function computation method
        """
        # Validate gridtype
        gridtype_lower = gridtype.lower()
        if gridtype_lower not in ("regular", "gaussian"):
            raise ValueError(
                f"Invalid grid type: '{gridtype}'. " f"Must be 'regular' or 'gaussian'"
            )

        # Validate rsphere
        if not isinstance(rsphere, (int, float)) or rsphere <= 0:
            raise ValueError(
                f"Invalid sphere radius: {rsphere}. "
                f"Must be a positive number (in meters)"
            )

        # Validate legfunc
        if legfunc not in ("stored", "computed"):
            raise ValueError(
                f"Invalid legfunc option: '{legfunc}'. "
                f"Must be 'stored' or 'computed'"
            )

        # Store configuration
        self.gridtype = gridtype_lower
        self.rsphere = rsphere
        self.legfunc = legfunc

        try:
            # Create spherical harmonic transform object
            spharm_kwargs = {
                "nlon": nlon,
                "nlat": nlat,
                "gridtype": self.gridtype,
                "rsphere": rsphere,
                "legfunc": legfunc,
                "precision": precision,
            }
            if hasattr(spharm_mod, "SpharmTransform"):
                spharm_kwargs["layout"] = layout
                spharm_kwargs["lat_order"] = lat_order
                spharm_kwargs["lat"] = lat
            self.s = Spharmt(**spharm_kwargs)
            self.precision = getattr(self.s, "precision", self._precision)
            self.layout = self._layout
            self.lat_order = getattr(self.s, "lat_order", self._lat_order)
            self.south_to_north = bool(getattr(self.s, "south_to_north", False))
        except Exception as e:
            # Provide more informative error messages
            if "invalid input dimensions" in str(e).lower():
                raise ValueError(
                    f"Invalid grid dimensions for {gridtype_lower} grid: "
                    f"(nlat={nlat}, nlon={nlon}). "
                    f"Please check grid size requirements for spherical harmonics."
                ) from e
            elif "spharm" in str(e).lower():
                raise ValueError(
                    f"Spherical harmonic initialization failed: {e}. "
                    f"Please ensure spharm module is properly compiled."
                ) from e
            else:
                raise ValueError(
                    f"Failed to initialize spherical harmonic transform: {e}"
                ) from e

    def _setup_method_aliases(self) -> None:
        """Set up method aliases for backward compatibility."""
        self.rotationalcomponent = self.nondivergentcomponent
        self.divergentcomponent = self.irrotationalcomponent

    def _setup_cached_arrays(self) -> None:
        """Initialize small per-instance caches for repeated wrapper work."""
        self._latitude_cache: dict[tuple[str, int], np.ndarray] = {}
        self._coriolis_cache: dict[tuple[str, int, float, str], np.ndarray] = {}
        self._planetary_vorticity_spec_cache: dict[
            tuple[str, int, int, float, int], np.ndarray
        ] = {}
        self._vector_analysis_layout_cache: Optional[
            Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]
        ] = None

    def _latitude_values(self) -> np.ndarray:
        """Return latitude coordinates for this wind grid."""
        explicit_lat = getattr(self, "_lat", None)
        if explicit_lat is not None:
            return explicit_lat

        s2n = bool(getattr(getattr(self, "s", None), "south_to_north", False))
        key = (self.gridtype, self.s.nlat, s2n) if s2n else (self.gridtype, self.s.nlat)
        cache = getattr(self, "_latitude_cache", None)
        if cache is None:
            cache = {}
            self._latitude_cache = cache

        lat = cache.get(key)
        if lat is not None:
            return lat

        nlat = self.s.nlat
        if self.gridtype == "gaussian":
            lat, _ = gaussian_lats_wts(nlat)
        elif nlat % 2:
            lat = np.linspace(90, -90, nlat)
        else:
            dlat = 180.0 / nlat
            lat = np.arange(90 - dlat / 2.0, -90, -dlat)

        if s2n:
            lat = lat[::-1].copy()

        cache[key] = np.asarray(lat)
        return cache[key]

    def _coriolis_values(
        self, omega: Optional[float], dtype: Optional[np.dtype] = None
    ) -> np.ndarray:
        """Return cached latitude-only Coriolis values."""
        if omega is None:
            omega = 7.292e-05

        try:
            omega_value = float(omega)
        except (TypeError, ValueError):
            raise ValueError(f"invalid value for omega: {omega!r}")

        output_dtype = self._output_dtype if dtype is None else np.dtype(dtype)
        s2n = bool(getattr(getattr(self, "s", None), "south_to_north", False))
        key = (self.gridtype, self.s.nlat, omega_value, output_dtype.str, s2n)
        cache = getattr(self, "_coriolis_cache", None)
        if cache is None:
            cache = {}
            self._coriolis_cache = cache

        coriolis = cache.get(key)
        if coriolis is not None:
            return coriolis

        coriolis = 2.0 * omega_value * np.sin(np.deg2rad(self._latitude_values()))
        coriolis = coriolis.astype(output_dtype, copy=False)
        cache[key] = coriolis
        return coriolis

    # ------------------------------------------------------------------
    # Internal numerical helpers. These avoid public dtype restoration
    # between intermediate steps in chained diagnostics.
    # ------------------------------------------------------------------

    def _planetary_vorticity(
        self,
        omega: Optional[float] = None,
        materialize: bool = True,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """Return planetary vorticity before public dtype restore."""
        coriolis = self._coriolis_values(omega, dtype=dtype)
        return self._broadcast_planetaryvorticity(coriolis, materialize=materialize)

    def _broadcast_planetaryvorticity(
        self, coriolis: np.ndarray, materialize: bool
    ) -> np.ndarray:
        """Broadcast latitude-only Coriolis values to the wind field shape."""
        if getattr(self, "_layout", "legacy") == "c":
            indices = (np.newaxis,) * (len(self.u.shape) - 2) + (
                slice(None),
                np.newaxis,
            )
        else:
            indices = (slice(None),) + (np.newaxis,) * (len(self.u.shape) - 1)
        broadcast = np.broadcast_to(coriolis[indices], self.u.shape)
        if materialize:
            return np.array(broadcast, copy=True)
        return broadcast

    def _planetary_vorticity_spec(
        self,
        truncation: Optional[int] = None,
        omega: Optional[float] = None,
        spectral_ndim: int = 0,
    ) -> np.ndarray:
        """Return cached planetary-vorticity spectrum, broadcast for reuse."""
        try:
            omega_value = 7.292e-05 if omega is None else float(omega)
        except (TypeError, ValueError):
            raise ValueError(f"invalid value for omega: {omega!r}")
        ntrunc = self.s._validate_ntrunc(truncation, self.s.nlat - 1)
        s2n = bool(getattr(self.s, "south_to_north", False))
        key = (self.gridtype, self.s.nlat, self.s.nlon, omega_value, ntrunc, s2n)
        cache = getattr(self, "_planetary_vorticity_spec_cache", None)
        if cache is None:
            cache = {}
            self._planetary_vorticity_spec_cache = cache

        fspec = cache.get(key)
        if fspec is None:
            coriolis = self._coriolis_values(omega_value, dtype=self._output_dtype)
            grid = np.broadcast_to(coriolis[:, np.newaxis], (self.s.nlat, self.s.nlon))
            fspec = self.s.grdtospec(grid, ntrunc=ntrunc)
            cache[key] = fspec

        if spectral_ndim <= 0:
            return fspec

        if getattr(self, "_layout", "legacy") == "c":
            return fspec[(np.newaxis,) * spectral_ndim + (slice(None),)]
        return fspec[(slice(None),) + (np.newaxis,) * spectral_ndim]

    def _gradient_components(
        self, chi: ArrayLike, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return vector gradient before public dtype restore."""
        try:
            chi = chi.filled(fill_value=np.nan) if hasattr(chi, "filled") else chi
        except AttributeError:
            pass

        if np.any(np.isnan(chi)):
            raise ValueError("chi cannot contain missing values")

        try:
            chi_spec = self.s.grdtospec(chi, ntrunc=truncation)
        except ValueError:
            raise ValueError("input field is not compatible")

        return self.s.getgrad(chi_spec)

    def _vector_analysis_layout_inputs(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
        """Return one cached C-contiguous (nt, nlat, nlon) wind-field pair."""
        cache = getattr(self, "_vector_analysis_layout_cache", None)
        if cache is not None:
            return cache

        if hasattr(self.s, "_core"):
            _, _, layout_u, layout_v, extra_shape = self.s._prepare_paired_grid_inputs(
                self.u, self.v, "vector wind analysis"
            )
            cache = (layout_u, layout_v, extra_shape)
            self._vector_analysis_layout_cache = cache
            return cache

        # Compatibility path for legacy Fortran-backed skybornspharm (<0.2.0)
        if self.u.ndim == 2:
            extra_shape: Tuple[int, ...] = ()
            layout_u = np.asfortranarray(np.expand_dims(self.u, 2), dtype=np.float32)
            layout_v = np.asfortranarray(np.expand_dims(self.v, 2), dtype=np.float32)
        else:
            extra_shape = self.u.shape[2:]
            nt = math.prod(extra_shape)
            layout_u = np.asfortranarray(
                self.u.reshape(self.u.shape[0], self.u.shape[1], nt),
                dtype=np.float32,
            )
            layout_v = np.asfortranarray(
                self.v.reshape(self.v.shape[0], self.v.shape[1], nt),
                dtype=np.float32,
            )

        cache = (layout_u, layout_v, extra_shape)
        self._vector_analysis_layout_cache = cache
        return cache

    def _vector_analysis_spectral(
        self, truncation: Optional[int] = None, component: Optional[str] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return cached-layout vector-analysis spectra for wind-derived diagnostics."""
        if component not in (None, "vrt", "div"):
            raise ValueError(f"unsupported component: {component!r}")

        if hasattr(self.s, "_core"):
            layout_u, layout_v, batch_shape = self._vector_analysis_layout_inputs()
            ntrunc = self.s._validate_ntrunc(truncation, self.s.nlat - 1)
            resolved_layout = (
                "c" if getattr(self, "_layout", "legacy") == "c" else "legacy"
            )
            s2n = bool(getattr(self.s, "south_to_north", False))
            if component is None:
                vrt_2d, div_2d = self.s._core.getvrtdivspec(
                    layout_u, layout_v, ntrunc=ntrunc, south_to_north=s2n
                )
                return (
                    self.s._restore_spectral_output(
                        vrt_2d, resolved_layout, batch_shape
                    ),
                    self.s._restore_spectral_output(
                        div_2d, resolved_layout, batch_shape
                    ),
                )
            if component == "vrt":
                vrt_2d = self.s._core.getvrtspec(
                    layout_u, layout_v, ntrunc=ntrunc, south_to_north=s2n
                )
                return self.s._restore_spectral_output(
                    vrt_2d, resolved_layout, batch_shape
                )
            div_2d = self.s._core.getdivspec(
                layout_u, layout_v, ntrunc=ntrunc, south_to_north=s2n
            )
            return self.s._restore_spectral_output(
                div_2d, resolved_layout, batch_shape
            )

        if component is None and not hasattr(
            self.s, "_analyze_vector_harmonics_flattened"
        ):
            return self.s.getvrtdivspec(self.u, self.v, ntrunc=truncation)
        if component == "vrt" and not hasattr(
            self.s, "_analyze_vector_harmonic_component_flattened"
        ):
            return self.s.getvrtspec(self.u, self.v, ntrunc=truncation)
        if component == "div" and not hasattr(
            self.s, "_analyze_vector_harmonic_component_flattened"
        ):
            return self.s.getdivspec(self.u, self.v, ntrunc=truncation)

        # Compatibility path for legacy Fortran-backed skybornspharm (<0.2.0)
        layout_u, layout_v, extra_shape = self._vector_analysis_layout_inputs()
        spherepack = getattr(spharm_mod, "_spherepack", None)
        if spherepack is None:
            if component is None:
                return self.s.getvrtdivspec(self.u, self.v, ntrunc=truncation)
            if component == "vrt":
                return self.s.getvrtspec(self.u, self.v, ntrunc=truncation)
            return self.s.getdivspec(self.u, self.v, ntrunc=truncation)

        if component is None:
            br, bi, cr, ci, _, ntrunc = self.s._analyze_vector_harmonics_flattened(
                layout_u,
                layout_v,
                extra_shape,
                truncation,
                "vector wind analysis",
            )
            vrtspec, divspec = spherepack.twodtooned_vrtdiv(
                br, bi, cr, ci, ntrunc, self.s.rsphere
            )
            return (
                self.s._restore_spectral_shape(vrtspec, extra_shape),
                self.s._restore_spectral_shape(divspec, extra_shape),
            )

        coeff_r, coeff_i, _, ntrunc = (
            self.s._analyze_vector_harmonic_component_flattened(
                layout_u,
                layout_v,
                extra_shape,
                truncation,
                "vector wind analysis",
                component,
            )
        )
        if component == "vrt":
            dataspec = spherepack.twodtooned_vrt(
                coeff_r, coeff_i, ntrunc, self.s.rsphere
            )
        else:
            dataspec = spherepack.twodtooned_div(
                coeff_r, coeff_i, ntrunc, self.s.rsphere
            )

        return self.s._restore_spectral_shape(dataspec, extra_shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def grid_info(self) -> dict:
        """
        Get information about the grid configuration.

        Returns
        -------
        dict
            Dictionary containing grid information
        """
        nlat, nlon = self._horizontal_shape()
        return {
            "gridtype": self.gridtype,
            "nlat": nlat,
            "nlon": nlon,
            "shape": self.u.shape,
            "rsphere": self.rsphere,
            "legfunc": self.legfunc,
            "total_points": nlat * nlon,
        }

    def magnitude(self) -> np.ndarray:
        """
        Calculate wind speed (magnitude of vector wind).

        Returns
        -------
        ndarray
            Wind speed field computed as sqrt(u² + v²)

        Examples
        --------
        >>> wind_speed = vw.magnitude()
        """
        return self._restore_output_dtype(np.hypot(self.u, self.v))

    def vrtdiv(self, truncation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate relative vorticity and horizontal divergence.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.
            If None, uses the default truncation based on grid resolution.

        Returns
        -------
        vorticity : ndarray
            Relative vorticity field
        divergence : ndarray
            Horizontal divergence field

        See Also
        --------
        vorticity : Calculate only vorticity
        divergence : Calculate only divergence

        Examples
        --------
        >>> vrt, div = vw.vrtdiv()
        >>> vrt_t13, div_t13 = vw.vrtdiv(truncation=13)
        """
        vrtspec, divspec = self._vector_analysis_spectral(truncation=truncation)
        vrt, div = self.s._spectogrd_pair(vrtspec, divspec)
        return (
            self._restore_output_dtype(vrt),
            self._restore_output_dtype(div),
        )

    def vorticity(self, truncation: Optional[int] = None) -> np.ndarray:
        """
        Calculate relative vorticity.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        ndarray
            Relative vorticity field (∇ × v)

        See Also
        --------
        vrtdiv : Calculate both vorticity and divergence
        absolutevorticity : Calculate absolute vorticity

        Examples
        --------
        >>> vrt = vw.vorticity()
        >>> vrt_t13 = vw.vorticity(truncation=13)
        """
        return self._restore_output_dtype(
            self.s.spectogrd(
                self._vector_analysis_spectral(truncation=truncation, component="vrt")
            )
        )

    def divergence(self, truncation: Optional[int] = None) -> np.ndarray:
        """
        Calculate horizontal divergence.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        ndarray
            Horizontal divergence field (∇ · v)

        See Also
        --------
        vrtdiv : Calculate both vorticity and divergence

        Examples
        --------
        >>> div = vw.divergence()
        >>> div_t13 = vw.divergence(truncation=13)
        """
        return self._restore_output_dtype(
            self.s.spectogrd(
                self._vector_analysis_spectral(truncation=truncation, component="div")
            )
        )

    def planetaryvorticity(self, omega: Optional[float] = None) -> np.ndarray:
        """
        Calculate planetary vorticity (Coriolis parameter).

        Parameters
        ----------
        omega : float, optional
            Earth's angular velocity in rad/s. Default is 7.292e-5 s⁻¹.

        Returns
        -------
        ndarray
            Planetary vorticity field (2Ω sin φ)

        See Also
        --------
        absolutevorticity : Calculate absolute vorticity

        Examples
        --------
        >>> f = vw.planetaryvorticity()
        >>> f_custom = vw.planetaryvorticity(omega=7.2921150e-5)
        """
        return self._planetary_vorticity(
            omega=omega, materialize=True, dtype=self._output_dtype
        )

    def absolutevorticity(
        self, omega: Optional[float] = None, truncation: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate absolute vorticity (relative + planetary vorticity).

        Parameters
        ----------
        omega : float, optional
            Earth's angular velocity in rad/s. Default is 7.292e-5 s⁻¹.
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        ndarray
            Absolute vorticity field (ζ + f)

        See Also
        --------
        vorticity : Calculate relative vorticity
        planetaryvorticity : Calculate planetary vorticity

        Examples
        --------
        >>> abs_vrt = vw.absolutevorticity()
        >>> abs_vrt_t13 = vw.absolutevorticity(omega=7.2921150e-5, truncation=13)
        """
        vrt = self.s.spectogrd(
            self._vector_analysis_spectral(truncation=truncation, component="vrt")
        )
        return self._restore_output_dtype(
            vrt + self._planetary_vorticity(omega=omega, materialize=False)
        )

    def sfvp(self, truncation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate streamfunction and velocity potential.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        streamfunction : ndarray
            Streamfunction field (ψ)
        velocity_potential : ndarray
            Velocity potential field (χ)

        See Also
        --------
        streamfunction : Calculate only streamfunction
        velocitypotential : Calculate only velocity potential

        Examples
        --------
        >>> psi, chi = vw.sfvp()
        >>> psi_t13, chi_t13 = vw.sfvp(truncation=13)
        """
        vrtspec, divspec = self._vector_analysis_spectral(truncation=truncation)
        psi, chi = self.s._spectogrd_pair(
            self.s._invlapspec(vrtspec, "sfvp"),
            self.s._invlapspec(divspec, "sfvp"),
        )
        return self._restore_output_dtype(psi), self._restore_output_dtype(chi)

    def streamfunction(self, truncation: Optional[int] = None) -> np.ndarray:
        """
        Calculate streamfunction.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        ndarray
            Streamfunction field (ψ)

        See Also
        --------
        sfvp : Calculate both streamfunction and velocity potential

        Examples
        --------
        >>> psi = vw.streamfunction()
        >>> psi_t13 = vw.streamfunction(truncation=13)
        """
        psi = self.s.spectogrd(
            self.s._invlapspec(
                self._vector_analysis_spectral(truncation=truncation, component="vrt"),
                "streamfunction",
            )
        )
        return self._restore_output_dtype(psi)

    def velocitypotential(self, truncation: Optional[int] = None) -> np.ndarray:
        """
        Calculate velocity potential.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        ndarray
            Velocity potential field (χ)

        See Also
        --------
        sfvp : Calculate both streamfunction and velocity potential

        Examples
        --------
        >>> chi = vw.velocitypotential()
        >>> chi_t13 = vw.velocitypotential(truncation=13)
        """
        chi = self.s.spectogrd(
            self.s._invlapspec(
                self._vector_analysis_spectral(truncation=truncation, component="div"),
                "velocitypotential",
            )
        )
        return self._restore_output_dtype(chi)

    def helmholtz(
        self, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Helmholtz decomposition of vector wind.

        Decomposes the wind field into irrotational (divergent) and
        non-divergent (rotational) components.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        u_chi : ndarray
            Zonal component of irrotational wind
        v_chi : ndarray
            Meridional component of irrotational wind
        u_psi : ndarray
            Zonal component of non-divergent wind
        v_psi : ndarray
            Meridional component of non-divergent wind

        See Also
        --------
        irrotationalcomponent : Get only irrotational component
        nondivergentcomponent : Get only non-divergent component

        Examples
        --------
        >>> u_chi, v_chi, u_psi, v_psi = vw.helmholtz()
        >>> u_chi_t13, v_chi_t13, u_psi_t13, v_psi_t13 = vw.helmholtz(truncation=13)
        """
        vrtspec, divspec = self._vector_analysis_spectral(truncation=truncation)
        psispec = self.s._invlapspec(vrtspec, "helmholtz")
        chispec = self.s._invlapspec(divspec, "helmholtz")
        v_psi, u_psi = self.s.getgrad(psispec)
        u_chi, v_chi = self.s.getgrad(chispec)
        return (
            self._restore_output_dtype(u_chi),
            self._restore_output_dtype(v_chi),
            self._restore_output_dtype(-u_psi),
            self._restore_output_dtype(v_psi),
        )

    def irrotationalcomponent(
        self, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate irrotational (divergent) component of vector wind.

        Note
        ----
        If both irrotational and non-divergent components are needed,
        use `helmholtz()` method for efficiency.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        u_chi : ndarray
            Zonal component of irrotational wind
        v_chi : ndarray
            Meridional component of irrotational wind

        See Also
        --------
        helmholtz : Complete Helmholtz decomposition
        nondivergentcomponent : Non-divergent component

        Examples
        --------
        >>> u_chi, v_chi = vw.irrotationalcomponent()
        >>> u_chi_t13, v_chi_t13 = vw.irrotationalcomponent(truncation=13)
        """
        u_chi, v_chi = self.s.getgrad(
            self.s._invlapspec(
                self._vector_analysis_spectral(truncation=truncation, component="div"),
                "irrotationalcomponent",
            )
        )
        return self._restore_output_dtype(u_chi), self._restore_output_dtype(v_chi)

    def nondivergentcomponent(
        self, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate non-divergent (rotational) component of vector wind.

        Note
        ----
        If both non-divergent and irrotational components are needed,
        use `helmholtz()` method for efficiency.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        u_psi : ndarray
            Zonal component of non-divergent wind
        v_psi : ndarray
            Meridional component of non-divergent wind

        See Also
        --------
        helmholtz : Complete Helmholtz decomposition
        irrotationalcomponent : Irrotational component

        Examples
        --------
        >>> u_psi, v_psi = vw.nondivergentcomponent()
        >>> u_psi_t13, v_psi_t13 = vw.nondivergentcomponent(truncation=13)
        """
        psispec = self.s._invlapspec(
            self._vector_analysis_spectral(truncation=truncation, component="vrt"),
            "nondivergentcomponent",
        )
        v_psi, u_psi = self.s.getgrad(psispec)
        return self._restore_output_dtype(-u_psi), self._restore_output_dtype(v_psi)

    def gradient(
        self, chi: ArrayLike, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate vector gradient of a scalar field on the sphere.

        Parameters
        ----------
        chi : array_like
            Scalar field with shape (nlat, nlon) or (nlat, nlon, nfields)
            matching the dimensions of the VectorWind instance.
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.

        Returns
        -------
        u_chi : ndarray
            Zonal component of vector gradient (∂χ/∂λ)
        v_chi : ndarray
            Meridional component of vector gradient (∂χ/∂φ)

        Examples
        --------
        >>> abs_vrt = vw.absolutevorticity()
        >>> avrt_u, avrt_v = vw.gradient(abs_vrt)
        >>> avrt_u_t13, avrt_v_t13 = vw.gradient(abs_vrt, truncation=13)
        """
        output_dtype = self._infer_output_dtype(chi)
        u_chi, v_chi = self._gradient_components(chi, truncation=truncation)
        return (
            self._restore_output_dtype(u_chi, output_dtype),
            self._restore_output_dtype(v_chi, output_dtype),
        )

    def truncate(
        self, field: ArrayLike, truncation: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply spectral truncation to a scalar field.

        This is useful to represent fields consistently with the output
        of other VectorWind methods.

        Parameters
        ----------
        field : array_like
            Scalar field with shape (nlat, nlon) or (nlat, nlon, nfields)
            matching the dimensions of the VectorWind instance.
        truncation : int, optional
            Triangular truncation limit. If None, defaults to nlat-1.

        Returns
        -------
        ndarray
            Field with spectral truncation applied

        Examples
        --------
        >>> field_trunc = vw.truncate(scalar_field)
        >>> field_t21 = vw.truncate(scalar_field, truncation=21)
        """
        try:
            field = (
                field.filled(fill_value=np.nan) if hasattr(field, "filled") else field
            )
        except AttributeError:
            pass

        if np.any(np.isnan(field)):
            raise ValueError("field cannot contain missing values")

        output_dtype = self._infer_output_dtype(field)

        try:
            field_spec = self.s.grdtospec(field, ntrunc=truncation)
        except ValueError:
            raise ValueError("field is not compatible")

        return self._restore_output_dtype(self.s.spectogrd(field_spec), output_dtype)

    def rossbywavesource(
        self, truncation: Optional[int] = None, omega: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate Rossby wave source.

        The Rossby wave source is defined as:
        S = -ζₐ∇·v - v_χ·∇ζₐ

        where:
        - ζₐ is absolute vorticity (relative + planetary)
        - ∇·v is horizontal divergence
        - v_χ is the irrotational (divergent) wind component
        - ∇ζₐ is the gradient of absolute vorticity

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.
            If None, uses the default truncation based on grid resolution.
        omega : float, optional
            Earth's angular velocity in rad/s. Default is 7.292e-5 s⁻¹.

        Returns
        -------
        ndarray
            Rossby wave source field (s⁻²)

        See Also
        --------
        absolutevorticity : Calculate absolute vorticity
        divergence : Calculate horizontal divergence
        irrotationalcomponent : Calculate irrotational wind component
        gradient : Calculate vector gradient

        Notes
        -----
        The Rossby wave source quantifies the generation of Rossby wave activity
        in the atmosphere. Positive values indicate wave generation, while
        negative values indicate wave absorption or dissipation.

        The calculation involves several steps:
        1. Compute absolute vorticity (relative + planetary)
        2. Compute horizontal divergence
        3. Compute irrotational wind components
        4. Compute gradients of absolute vorticity
        5. Combine terms according to the RWS formula

        Examples
        --------
        >>> rws = vw.rossbywavesource()
        >>> rws_t21 = vw.rossbywavesource(truncation=21)
        >>> rws_custom_omega = vw.rossbywavesource(omega=7.2921150e-5)

        References
        ----------
        Sardeshmukh, P. D., & Hoskins, B. J. (1988). The generation of global
        rotational flow by steady idealized tropical heating. Journal of the
        Atmospheric Sciences, 45(7), 1228-1251.
        """
        # Reuse the first vector analysis for both grid-space and spectral terms.
        vrtspec, divspec = self._vector_analysis_spectral(truncation=truncation)
        vrt, div = self.s._spectogrd_pair(vrtspec, divspec)
        eta = vrt + self._planetary_vorticity(omega=omega, materialize=False)

        # Reconstruct the divergent wind from the already-available divergence
        # spectrum instead of triggering a second vector analysis.
        uchi, vchi = self.s.getgrad(self.s._invlapspec(divspec, "rossbywavesource"))

        # Reuse relative-vorticity spectra plus a cached planetary-vorticity
        # spectrum so absolute-vorticity gradients do not trigger a full-field
        # scalar analysis of eta on every call.
        etaspec = vrtspec + self._planetary_vorticity_spec(
            truncation=truncation,
            omega=omega,
            spectral_ndim=max(vrtspec.ndim - 1, 0),
        )
        etax, etay = self.s.getgrad(etaspec)

        # Combine components to form Rossby wave source
        # S = -eta * div - (uchi * etax + vchi * etay)
        rws = -eta * div - (uchi * etax + vchi * etay)

        return self._restore_output_dtype(rws)
