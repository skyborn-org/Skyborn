"""
Common functionality shared across windspharm interfaces.

This module provides utility functions that are used by multiple windspharm
interfaces (standard, xarray, etc.) for grid type detection, dimension
ordering, and data reshaping operations.

Main Functions:
    get_apiorder: Calculate dimension reordering for API compatibility
    inspect_gridtype: Determine grid type from latitude coordinates
    to3d: Reshape array to 3D format for spherical harmonic analysis
"""

from __future__ import annotations

from typing import Any, List, Literal, Tuple, Union

import numpy as np

from skyborn.spharm import gaussian_lats_wts

__all__ = [
    "get_apiorder",
    "infer_output_dtype",
    "inspect_gridtype",
    "normalize_layout",
    "process_input_array",
    "to3d",
    "unwrap_masked_scalar",
]

# Type aliases for better readability
ArrayLike = Union[np.ndarray, np.ma.MaskedArray]
GridType = Literal["regular", "gaussian"]

_DTYPE_F32 = np.dtype(np.float32)
_DTYPE_F64 = np.dtype(np.float64)
_LAYOUT_ALIASES = {"auto": "auto", "c": "c", "legacy": "legacy"}


def normalize_layout(layout: str) -> str:
    """Validate and normalize axis layout mode ('auto', 'c', or 'legacy')."""
    norm = _LAYOUT_ALIASES.get(layout) if type(layout) is str else None
    if norm is None:
        norm = _LAYOUT_ALIASES.get(str(layout).lower())
    if norm is None:
        raise ValueError(f'layout must be "legacy", "c", or "auto", got "{layout}"')
    return norm


def infer_output_dtype(*arrays: ArrayLike) -> np.dtype:
    """Return the promoted public floating output dtype for array inputs."""
    if len(arrays) == 2:
        a0, a1 = arrays
        if type(a0) is np.ndarray and type(a1) is np.ndarray:
            d0, d1 = a0.dtype, a1.dtype
            if d0 == _DTYPE_F32 and d1 == _DTYPE_F32:
                return _DTYPE_F32
            if d0 in (_DTYPE_F32, _DTYPE_F64) and d1 in (_DTYPE_F32, _DTYPE_F64):
                return _DTYPE_F64
    elif len(arrays) == 1:
        a0 = arrays[0]
        if type(a0) is np.ndarray:
            d0 = a0.dtype
            if d0 == _DTYPE_F32 or d0 == _DTYPE_F64:
                return d0

    output_dtype = None
    for arr in arrays:
        dtype = (
            arr.dtype if type(arr) is np.ndarray else np.dtype(np.asarray(arr).dtype)
        )
        public_dtype = dtype if np.issubdtype(dtype, np.floating) else _DTYPE_F64
        output_dtype = (
            public_dtype
            if output_dtype is None
            else np.result_type(output_dtype, public_dtype)
        )
    return _DTYPE_F64 if output_dtype is None else output_dtype


def unwrap_masked_scalar(field: Any) -> np.ndarray:
    """Convert masked/array scalar input to ndarray, replacing masks with NaN."""
    try:
        unwrapped = (
            field.filled(fill_value=np.nan) if hasattr(field, "filled") else field
        )
    except AttributeError:
        unwrapped = field
    return unwrapped if type(unwrapped) is np.ndarray else np.asarray(unwrapped)


def process_input_array(arr: ArrayLike, name: str, copy: bool = True) -> np.ndarray:
    """Coerce an array or MaskedArray wind input to floating ndarray."""
    if type(arr) is np.ndarray and (arr.dtype == _DTYPE_F32 or arr.dtype == _DTYPE_F64):
        return arr.copy() if copy else arr

    try:
        if hasattr(arr, "filled"):
            masked = np.ma.asarray(arr)
            dtype = (
                masked.dtype if np.issubdtype(masked.dtype, np.floating) else _DTYPE_F64
            )
            processed = np.ma.asarray(masked, dtype=dtype).filled(fill_value=np.nan)
        else:
            raw = np.asarray(arr)
            dtype = np.dtype(raw.dtype)
            if not np.issubdtype(dtype, np.floating):
                dtype = _DTYPE_F64
            processed = np.asarray(raw, dtype=dtype)
        return processed.copy() if copy else processed
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {name} to numpy array: {exc}") from exc


def get_apiorder(
    ndim: int, latitude_dim: int, longitude_dim: int
) -> Tuple[List[int], List[int]]:
    """
    Calculate dimension ordering for API-compatible array transposition.

    This function determines the proper dimension ordering to move latitude
    and longitude dimensions to the first two positions, which is required
    by the spherical harmonic analysis routines.

    Parameters
    ----------
    ndim : int
        Total number of dimensions in the array
    latitude_dim : int
        Index of the latitude dimension
    longitude_dim : int
        Index of the longitude dimension

    Returns
    -------
    apiorder : list of int
        List of dimension indices in the order required for API compatibility.
        Latitude and longitude dimensions are moved to positions 0 and 1.
    reorder : list of int
        Inverse mapping to restore original dimension order after processing

    Examples
    --------
    >>> # For 4D array (time, level, lat, lon)
    >>> apiorder, reorder = get_apiorder(4, 2, 3)
    >>> print(apiorder)  # [2, 3, 0, 1] - lat, lon, time, level
    >>> print(reorder)   # [2, 3, 0, 1] - inverse mapping
    """
    apiorder = list(range(ndim))
    apiorder.remove(latitude_dim)
    apiorder.remove(longitude_dim)
    apiorder.insert(0, latitude_dim)
    apiorder.insert(1, longitude_dim)
    reorder = [apiorder.index(i) for i in range(ndim)]
    return apiorder, reorder


def inspect_gridtype(latitudes: ArrayLike) -> GridType:
    """
    Determine grid type by examining latitude coordinate values.

    Analyzes the spacing and distribution of latitude points to determine
    whether they correspond to a regular (equally-spaced) or Gaussian grid.
    Accepts monotonic global grids ordered north-to-south or south-to-north.

    Parameters
    ----------
    latitudes : array_like
        Array of monotonically increasing or decreasing latitude coordinate
        values in degrees.

    Returns
    -------
    gridtype : {'regular', 'gaussian'}
        Detected grid type:
        - 'regular': Equally-spaced latitude grid
        - 'gaussian': Gaussian latitude grid (optimal for spectral methods)

    Raises
    ------
    ValueError
        If the grid type cannot be determined or if latitudes don't match
        either regular or Gaussian patterns within tolerance

    Examples
    --------
    >>> # Regular grid with 73 points including poles
    >>> lats_regular = np.linspace(90, -90, 73)
    >>> gridtype = inspect_gridtype(lats_regular)
    >>> print(gridtype)  # 'regular'
    >>>
    >>> # Gaussian grid
    >>> from skyborn.spharm import gaussian_lats_wts
    >>> lats_gauss, _ = gaussian_lats_wts(64)
    >>> gridtype = inspect_gridtype(lats_gauss)
    >>> print(gridtype)  # 'gaussian'
    """
    # Convert to numpy array for consistent handling
    latitudes = np.asarray(latitudes, dtype=np.float64)

    # Define tolerance for floating-point comparisons
    # This must be much smaller than typical grid spacings
    tolerance = 5e-4

    # Get the number of latitude points
    nlat = len(latitudes)
    if nlat < 2:
        raise ValueError(
            f"Need at least 2 latitude points for grid type detection, got {nlat}"
        )

    # Normalize to north-to-south for global grid verification
    lat_n2s = latitudes[::-1] if latitudes[0] < latitudes[-1] else latitudes

    # Check if latitudes are equally spaced
    diffs = np.abs(np.diff(lat_n2s))
    equally_spaced = (np.abs(diffs - diffs[0]) < tolerance).all()

    if not equally_spaced:
        # Not equally-spaced - check if they match Gaussian latitudes
        try:
            gauss_reference, _ = gaussian_lats_wts(nlat)
            difference = np.abs(lat_n2s - gauss_reference)
            if np.any(difference > tolerance):
                raise ValueError(
                    f"Latitudes are neither equally-spaced nor Gaussian. "
                    f"Maximum difference from Gaussian: {difference.max():.6f} degrees, "
                    f"tolerance: {tolerance:.6f} degrees"
                )
            gridtype = "gaussian"
        except Exception as e:
            raise ValueError(
                f"Failed to generate Gaussian reference latitudes for {nlat} points: {e}"
            ) from e
    else:
        # Equally-spaced - verify they match global regular grid pattern
        if nlat % 2:
            # Odd number of latitudes includes both poles
            equal_reference = np.linspace(90, -90, nlat)
        else:
            # Even number of latitudes excludes poles
            delta_latitude = 180.0 / nlat
            equal_reference = np.linspace(
                90 - 0.5 * delta_latitude, -90 + 0.5 * delta_latitude, nlat
            )

        difference = np.abs(lat_n2s - equal_reference)
        if np.any(difference > tolerance):
            raise ValueError(
                f"Equally-spaced latitudes don't match global regular grid pattern. "
                f"Maximum difference: {difference.max():.6f} degrees, "
                f"tolerance: {tolerance:.6f} degrees. "
                f"Grid may be regional or use non-standard spacing."
            )
        gridtype = "regular"
    return gridtype


def to3d(array: ArrayLike) -> np.ndarray:
    """
    Reshape array to 3D format for spherical harmonic analysis.

    Converts input arrays to the (nlat, nlon, nfields) format required
    by spherical harmonic routines, where any additional dimensions beyond
    latitude and longitude are flattened into a single 'fields' dimension.

    Parameters
    ----------
    array : array_like
        Input array with shape (nlat, nlon, ...) where latitude and longitude
        are the first two dimensions

    Returns
    -------
    ndarray
        Reshaped array with shape (nlat, nlon, nfields) where nfields is the
        product of all dimensions beyond the first two

    Examples
    --------
    >>> # 4D array: (nlat=73, nlon=144, ntime=12, nlevel=17)
    >>> data_4d = np.random.randn(73, 144, 12, 17)
    >>> data_3d = to3d(data_4d)
    >>> print(data_3d.shape)  # (73, 144, 204)
    >>>
    >>> # Already 3D array remains unchanged
    >>> data_3d_input = np.random.randn(73, 144, 10)
    >>> result = to3d(data_3d_input)
    >>> print(result.shape)  # (73, 144, 10)
    """
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError(
            f"Array must have at least 2 dimensions (lat, lon), got {array.ndim}D"
        )

    # Calculate new shape: (nlat, nlon, product_of_remaining_dims)
    new_shape = array.shape[:2] + (np.prod(array.shape[2:], dtype=int),)
    return array.reshape(new_shape)
