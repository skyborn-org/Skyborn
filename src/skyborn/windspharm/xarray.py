"""
Spherical harmonic vector wind computations with xarray interface.

This module provides a VectorWind class that works with xarray DataArrays,
preserving coordinate information and metadata throughout the computation process.
It serves as a high-level interface to the standard VectorWind implementation.

Main Class:
    VectorWind: xarray-aware interface for wind field analysis

Example:
    >>> import xarray as xr
    >>> from skyborn.windspharm.xarray import VectorWind
    >>>
    >>> # Load wind data as xarray DataArrays
    >>> u = xr.open_dataarray('u_wind.nc')
    >>> v = xr.open_dataarray('v_wind.nc')
    >>>
    >>> # Create VectorWind instance
    >>> vw = VectorWind(u, v)
    >>>
    >>> # Compute with preserved metadata
    >>> vorticity = vw.vorticity()
    >>> streamfunction = vw.streamfunction()
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr

__all__ = ["VectorWind", "ReducedVectorWind"]

from . import reduced, standard
from ._common import get_apiorder, inspect_gridtype

# Type aliases for better readability
DataArray = xr.DataArray
LegFunc = str  # 'stored' or 'computed'


class VectorWind:
    """
    Vector wind analysis using xarray DataArrays.

    This class provides a high-level interface for spherical harmonic wind analysis
    that preserves xarray coordinate information and metadata. It wraps the standard
    VectorWind implementation while maintaining CF-compliant attributes.

    Parameters
    ----------
    u, v : xarray.DataArray
        Zonal and meridional wind components. Must have the same dimensions,
        coordinates, and contain no missing values. Should include latitude
        and longitude dimensions with appropriate coordinate information.
    rsphere : float, default 6.3712e6
        Earth radius in meters for spherical harmonic computations.
    legfunc : {'stored', 'computed'}, default 'stored'
        Legendre function computation method:
        - 'stored': precompute and store (faster, more memory)
        - 'computed': compute on-the-fly (slower, less memory)
    gridtype : {'regular', 'gaussian'}, optional
        Explicit grid type override. If omitted, the grid type is inferred from
        latitude coordinates.

    Attributes
    ----------
    _api : standard.VectorWind
        Underlying standard VectorWind instance
    _reorder : tuple
        Original dimension ordering for output reconstruction
    _ishape : tuple
        Original data shape
    _coords : list
        Original coordinate information

    Examples
    --------
    >>> import xarray as xr
    >>> from skyborn.windspharm.xarray import VectorWind
    >>>
    >>> # Load wind components
    >>> u = xr.open_dataarray('u850.nc')
    >>> v = xr.open_dataarray('v850.nc')
    >>>
    >>> # Create VectorWind instance
    >>> vw = VectorWind(u, v)
    >>>
    >>> # Compute vorticity with preserved metadata
    >>> vorticity = vw.vorticity()
    >>> print(vorticity.attrs)  # CF-compliant attributes
    >>>
    >>> # Helmholtz decomposition
    >>> u_chi, v_chi, u_psi, v_psi = vw.helmholtz()
    """

    def __init__(
        self,
        u: DataArray,
        v: DataArray,
        rsphere: float = 6.3712e6,
        legfunc: LegFunc = "stored",
        gridtype: Optional[str] = None,
        precision: Literal["auto", "single", "double"] = "auto",
    ) -> None:
        """Initialize VectorWind instance with comprehensive validation."""
        # Validate input types
        if not isinstance(u, xr.DataArray):
            raise TypeError(f"u must be xarray.DataArray, got {type(u).__name__}")
        if not isinstance(v, xr.DataArray):
            raise TypeError(f"v must be xarray.DataArray, got {type(v).__name__}")

        # Validate coordinate compatibility
        self._validate_coordinates(u, v)

        # Find and validate latitude/longitude coordinates
        lat, lat_dim = _find_latitude_coordinate(u)
        lon, lon_dim = _find_longitude_coordinate(u)

        lat_values = np.asarray(lat.values)
        is_south_to_north = bool(lat_values[0] < lat_values[-1])

        # Determine grid type from monotonic global latitudes (N2S or S2N)
        if gridtype is None:
            gridtype = inspect_gridtype(lat_values)
        else:
            gridtype = gridtype.lower()
            if gridtype not in ("regular", "gaussian"):
                raise ValueError(
                    f"Invalid grid type: '{gridtype}'. Must be 'regular' or 'gaussian'"
                )

        # Store original dimension order for output reconstruction
        self._reorder = u.dims

        use_c_layout = hasattr(standard.spharm_mod, "SpharmTransform")
        if use_c_layout:
            trailing_lat_lon = lat_dim == u.ndim - 2 and lon_dim == u.ndim - 1
            if trailing_lat_lon:
                self._needs_transpose = False
            else:
                self._needs_transpose = True
                c_order = [
                    dim
                    for idx, dim in enumerate(u.dims)
                    if idx not in (lat_dim, lon_dim)
                ] + [u.dims[lat_dim], u.dims[lon_dim]]
                u = u.transpose(*c_order)
                v = v.transpose(*c_order)

            self._ishape = u.shape
            self._coords = [u.coords[name] for name in u.dims]
            u_data = u.values
            v_data = v.values
            self._u_component_dtype = standard.VectorWind._infer_output_dtype(u_data)
            self._v_component_dtype = standard.VectorWind._infer_output_dtype(v_data)

            self._api = standard.VectorWind._from_owned_input(
                u_data,
                v_data,
                gridtype=gridtype,
                rsphere=rsphere,
                legfunc=legfunc,
                precision=precision,
                layout="c",
                lat=lat_values,
            )
        else:
            # Legacy fallback for older skybornspharm (<0.2.0)
            if is_south_to_north:
                u = _reverse(u, lat_dim)
                v = _reverse(v, lat_dim)
                lat, lat_dim = _find_latitude_coordinate(u)
            apiorder, _ = get_apiorder(u.ndim, lat_dim, lon_dim)
            apiorder = [u.dims[i] for i in apiorder]
            self._needs_transpose = tuple(apiorder) != u.dims
            u = u.transpose(*apiorder)
            v = v.transpose(*apiorder)
            self._ishape = u.shape
            self._coords = [u.coords[name] for name in u.dims]
            u_data = u.values
            v_data = v.values
            self._u_component_dtype = standard.VectorWind._infer_output_dtype(u_data)
            self._v_component_dtype = standard.VectorWind._infer_output_dtype(v_data)
            self._api = standard.VectorWind._from_owned_input(
                u_data,
                v_data,
                gridtype=gridtype,
                rsphere=rsphere,
                legfunc=legfunc,
                precision=precision,
            )

        self.gridtype = self._api.gridtype
        self.rsphere = self._api.rsphere
        self.legfunc = self._api.legfunc
        self.precision = getattr(self._api, "precision", precision)
        self.s = self._api.s
        self.rotationalcomponent = self.nondivergentcomponent
        self.divergentcomponent = self.irrotationalcomponent

    def _validate_coordinates(self, u: DataArray, v: DataArray) -> None:
        """
        Validate that u and v have compatible coordinates.

        Parameters
        ----------
        u, v : DataArray
            Wind components to validate

        Raises
        ------
        ValueError
            If dimensions or coordinate values don't match
        """
        # Check dimension names
        if u.dims != v.dims:
            raise ValueError(
                f"u and v must have identical dimensions. "
                f"Got u: {u.dims}, v: {v.dims}"
            )

        # Check coordinate values
        u_coords = [u.coords[name].values for name in u.dims]
        v_coords = [v.coords[name].values for name in v.dims]

        mismatched_coords = []
        for i, (uc, vc) in enumerate(zip(u_coords, v_coords)):
            try:
                if not (uc == vc).all():
                    mismatched_coords.append(u.dims[i])
            except (ValueError, TypeError):
                # Handle different shapes or types
                mismatched_coords.append(u.dims[i])

        if mismatched_coords:
            raise ValueError(
                f"u and v must have identical coordinate values. "
                f"Mismatched coordinates: {mismatched_coords}"
            )

    def _metadata(self, data: Any, name: str, **attributes: Any) -> DataArray:
        """
        Create DataArray with proper metadata and coordinate information.

        Parameters
        ----------
        data : array_like
            Data to wrap in DataArray
        name : str
            Variable name
        **attributes
            Additional attributes to set

        Returns
        -------
        DataArray
            Properly formatted DataArray with coordinates and metadata
        """
        # Reshape to original structure when needed
        data_arr = np.asarray(data)
        if data_arr.shape != self._ishape:
            data_arr = data_arr.reshape(self._ishape)

        # Create DataArray with coordinates
        result = xr.DataArray(data_arr, coords=self._coords, name=name)

        # Restore original dimension order only when input axes were transposed
        if getattr(self, "_needs_transpose", True):
            result = result.transpose(*self._reorder)

        # Set attributes
        for attr, value in attributes.items():
            result.attrs[attr] = value

        return result

    def _component_metadata(
        self, data: Any, dtype: np.dtype, name: str, **attributes: Any
    ) -> DataArray:
        """Wrap stored wind components using the public component dtype."""
        restored = self._api._restore_output_dtype(data, dtype)
        return self._metadata(restored, name, **attributes)

    def u(self) -> DataArray:
        """
        Get zonal component of vector wind.

        Returns
        -------
        DataArray
            Zonal (eastward) wind component with CF-compliant attributes

        Examples
        --------
        >>> u_wind = vw.u()
        >>> print(u_wind.attrs['standard_name'])  # 'eastward_wind'
        """
        return self._component_metadata(
            self._api.u,
            self._u_component_dtype,
            "u",
            units="m s**-1",
            standard_name="eastward_wind",
            long_name="eastward_component_of_wind",
        )

    def v(self) -> DataArray:
        """
        Get meridional component of vector wind.

        Returns
        -------
        DataArray
            Meridional (northward) wind component with CF-compliant attributes

        Examples
        --------
        >>> v_wind = vw.v()
        >>> print(v_wind.attrs['standard_name'])  # 'northward_wind'
        """
        return self._component_metadata(
            self._api.v,
            self._v_component_dtype,
            "v",
            units="m s**-1",
            standard_name="northward_wind",
            long_name="northward_component_of_wind",
        )

    def magnitude(self) -> DataArray:
        """
        Calculate wind speed (magnitude of vector wind).

        Returns
        -------
        DataArray
            Wind speed with CF-compliant attributes

        Examples
        --------
        >>> wind_speed = vw.magnitude()
        >>> print(wind_speed.attrs['standard_name'])  # 'wind_speed'
        """
        magnitude = self._api.magnitude()
        return self._metadata(
            magnitude,
            "speed",
            units="m s**-1",
            standard_name="wind_speed",
            long_name="wind_speed",
        )

    def vrtdiv(self, truncation: Optional[int] = None) -> Tuple[DataArray, DataArray]:
        """
        Calculate relative vorticity and horizontal divergence.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation

        Returns
        -------
        vorticity : DataArray
            Relative vorticity with CF-compliant attributes
        divergence : DataArray
            Horizontal divergence with CF-compliant attributes

        See Also
        --------
        vorticity : Calculate only vorticity
        divergence : Calculate only divergence

        Examples
        --------
        >>> vrt, div = vw.vrtdiv()
        >>> vrt_t13, div_t13 = vw.vrtdiv(truncation=13)
        """
        vrt, div = self._api.vrtdiv(truncation=truncation)

        vrt_da = self._metadata(
            vrt,
            "vorticity",
            units="s**-1",
            standard_name="atmosphere_relative_vorticity",
            long_name="relative_vorticity",
        )

        div_da = self._metadata(
            div,
            "divergence",
            units="s**-1",
            standard_name="divergence_of_wind",
            long_name="horizontal_divergence",
        )

        return vrt_da, div_da

    def vorticity(self, truncation: Optional[int] = None) -> DataArray:
        """
        Calculate relative vorticity.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation

        Returns
        -------
        DataArray
            Relative vorticity field with CF-compliant attributes

        See Also
        --------
        vrtdiv : Calculate both vorticity and divergence
        absolutevorticity : Calculate absolute vorticity

        Examples
        --------
        >>> vrt = vw.vorticity()
        >>> vrt_t13 = vw.vorticity(truncation=13)
        """
        vrt = self._api.vorticity(truncation=truncation)
        return self._metadata(
            vrt,
            "vorticity",
            units="s**-1",
            standard_name="atmosphere_relative_vorticity",
            long_name="relative_vorticity",
        )

    def divergence(self, truncation: Optional[int] = None) -> DataArray:
        """
        Calculate horizontal divergence.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation

        Returns
        -------
        DataArray
            Horizontal divergence field with CF-compliant attributes

        See Also
        --------
        vrtdiv : Calculate both vorticity and divergence

        Examples
        --------
        >>> div = vw.divergence()
        >>> div_t13 = vw.divergence(truncation=13)
        """
        div = self._api.divergence(truncation=truncation)
        return self._metadata(
            div,
            "divergence",
            units="s**-1",
            standard_name="divergence_of_wind",
            long_name="horizontal_divergence",
        )

    def planetaryvorticity(self, omega: Optional[float] = None) -> DataArray:
        """
        Calculate planetary vorticity (Coriolis parameter).

        Parameters
        ----------
        omega : float, optional
            Earth's angular velocity in rad/s. Default is 7.292e-5 s⁻¹

        Returns
        -------
        DataArray
            Planetary vorticity (Coriolis parameter) with CF-compliant attributes

        See Also
        --------
        absolutevorticity : Calculate absolute vorticity

        Examples
        --------
        >>> f = vw.planetaryvorticity()
        >>> f_custom = vw.planetaryvorticity(omega=7.2921150e-5)
        """
        f = self._api.planetaryvorticity(omega=omega)
        return self._metadata(
            f,
            "coriolis",
            units="s**-1",
            standard_name="coriolis_parameter",
            long_name="planetary_vorticity",
        )

    def absolutevorticity(
        self, omega: Optional[float] = None, truncation: Optional[int] = None
    ) -> DataArray:
        """Absolute vorticity (sum of relative and planetary vorticity).

        **Optional arguments:**

        *omega*
            Earth's angular velocity. The default value if not specified
            is 7.292x10**-5 s**-1.

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *avorticity*
            The absolute (relative + planetary) vorticity.

        **See also:**

        `~VectorWind.vorticity`, `~VectorWind.planetaryvorticity`.

        **Examples:**

        Compute absolute vorticity::

            avrt = vw.absolutevorticity()

        Compute absolute vorticity and apply spectral truncation at
        triangular T13, also override the default value for Earth's
        angular velocity::

            avrt = vw.absolutevorticity(omega=7.2921150e-5, truncation=13)

        """
        avrt = self._api.absolutevorticity(omega=omega, truncation=truncation)
        avrt = self._metadata(
            avrt,
            "absolute_vorticity",
            units="s**-1",
            standard_name="atmosphere_absolute_vorticity",
            long_name="absolute_vorticity",
        )
        return avrt

    def sfvp(self, truncation: Optional[int] = None) -> Tuple[DataArray, DataArray]:
        """
        Calculate streamfunction and velocity potential.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation

        Returns
        -------
        streamfunction : DataArray
            Streamfunction field with CF-compliant attributes
        velocity_potential : DataArray
            Velocity potential field with CF-compliant attributes

        See Also
        --------
        streamfunction : Calculate only streamfunction
        velocitypotential : Calculate only velocity potential

        Examples
        --------
        >>> psi, chi = vw.sfvp()
        >>> psi_t13, chi_t13 = vw.sfvp(truncation=13)
        """
        sf, vp = self._api.sfvp(truncation=truncation)

        sf_da = self._metadata(
            sf,
            "streamfunction",
            units="m**2 s**-1",
            standard_name="atmosphere_horizontal_streamfunction",
            long_name="streamfunction",
        )

        vp_da = self._metadata(
            vp,
            "velocity_potential",
            units="m**2 s**-1",
            standard_name="atmosphere_horizontal_velocity_potential",
            long_name="velocity potential",
        )

        return sf_da, vp_da

    def streamfunction(self, truncation: Optional[int] = None) -> DataArray:
        """Streamfunction.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *sf*
            The streamfunction.

        **See also:**

        `~VectorWind.sfvp`.

        **Examples:**

        Compute streamfunction::

            sf = vw.streamfunction()

        Compute streamfunction and apply spectral truncation at
        triangular T13::

            sfT13 = vw.streamfunction(truncation=13)

        """
        sf = self._api.streamfunction(truncation=truncation)
        sf = self._metadata(
            sf,
            "streamfunction",
            units="m**2 s**-1",
            standard_name="atmosphere_horizontal_streamfunction",
            long_name="streamfunction",
        )
        return sf

    def velocitypotential(self, truncation: Optional[int] = None) -> DataArray:
        """Velocity potential.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *vp*
            The velocity potential.

        **See also:**

        `~VectorWind.sfvp`.

        **Examples:**

        Compute velocity potential::

            vp = vw.velocitypotential()

        Compute velocity potential and apply spectral truncation at
        triangular T13::

            vpT13 = vw.velocitypotential(truncation=13)

        """
        vp = self._api.velocitypotential(truncation=truncation)
        vp = self._metadata(
            vp,
            "velocity_potential",
            units="m**2 s**-1",
            standard_name="atmosphere_horizontal_velocity_potential",
            long_name="velocity potential",
        )
        return vp

    def helmholtz(
        self, truncation: Optional[int] = None
    ) -> Tuple[DataArray, DataArray, DataArray, DataArray]:
        """Irrotational and non-divergent components of the vector wind.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *uchi*, *vchi*, *upsi*, *vpsi*
            Zonal and meridional components of irrotational and
            non-divergent wind components respectively.

        **See also:**

        `~VectorWind.irrotationalcomponent`,
        `~VectorWind.nondivergentcomponent`.

        **Examples:**

        Compute the irrotational and non-divergent components of the
        vector wind::

            uchi, vchi, upsi, vpsi = vw.helmholtz()

        Compute the irrotational and non-divergent components of the
        vector wind and apply spectral truncation at triangular T13::

            uchiT13, vchiT13, upsiT13, vpsiT13 = vw.helmholtz(truncation=13)

        """
        uchi, vchi, upsi, vpsi = self._api.helmholtz(truncation=truncation)
        uchi = self._metadata(
            uchi, "u_chi", units="m s**-1", long_name="irrotational_eastward_wind"
        )
        vchi = self._metadata(
            vchi, "v_chi", units="m s**-1", long_name="irrotational_northward_wind"
        )
        upsi = self._metadata(
            upsi, "u_psi", units="m s**-1", long_name="non_divergent_eastward_wind"
        )
        vpsi = self._metadata(
            vpsi, "v_psi", units="m s**-1", long_name="non_divergent_northward_wind"
        )
        return uchi, vchi, upsi, vpsi

    def irrotationalcomponent(
        self, truncation: Optional[int] = None
    ) -> Tuple[DataArray, DataArray]:
        """Irrotational (divergent) component of the vector wind.

        .. note::

           If both the irrotational and non-divergent components are
           required then `~VectorWind.helmholtz` should be used instead.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *uchi*, *vchi*
            The zonal and meridional components of the irrotational wind
            respectively.

        **See also:**

        `~VectorWind.helmholtz`.

        **Examples:**

        Compute the irrotational component of the vector wind::

            uchi, vchi = vw.irrotationalcomponent()

        Compute the irrotational component of the vector wind and apply
        spectral truncation at triangular T13::

            uchiT13, vchiT13 = vw.irrotationalcomponent(truncation=13)

        """
        uchi, vchi = self._api.irrotationalcomponent(truncation=truncation)
        uchi = self._metadata(
            uchi, "u_chi", units="m s**-1", long_name="irrotational_eastward_wind"
        )
        vchi = self._metadata(
            vchi, "v_chi", units="m s**-1", long_name="irrotational_northward_wind"
        )
        return uchi, vchi

    def nondivergentcomponent(
        self, truncation: Optional[int] = None
    ) -> Tuple[DataArray, DataArray]:
        """Non-divergent (rotational) component of the vector wind.

        .. note::

           If both the non-divergent and irrotational components are
           required then `~VectorWind.helmholtz` should be used instead.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *upsi*, *vpsi*
            The zonal and meridional components of the non-divergent
            wind respectively.

        **See also:**

        `~VectorWind.helmholtz`.

        **Examples:**

        Compute the non-divergent component of the vector wind::

            upsi, vpsi = vw.nondivergentcomponent()

        Compute the non-divergent component of the vector wind and apply
        spectral truncation at triangular T13::

            upsiT13, vpsiT13 = vw.nondivergentcomponent(truncation=13)

        """
        upsi, vpsi = self._api.nondivergentcomponent(truncation=truncation)
        upsi = self._metadata(
            upsi, "u_psi", units="m s**-1", long_name="non_divergent_eastward_wind"
        )
        vpsi = self._metadata(
            vpsi, "v_psi", units="m s**-1", long_name="non_divergent_northward_wind"
        )
        return upsi, vpsi

    def gradient(
        self, chi: DataArray, truncation: Optional[int] = None
    ) -> Tuple[DataArray, DataArray]:
        """
        Calculate vector gradient of a scalar field on the sphere.

        Parameters
        ----------
        chi : DataArray
            Scalar field with same latitude/longitude dimensions as wind components
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation

        Returns
        -------
        u_gradient : DataArray
            Zonal component of vector gradient
        v_gradient : DataArray
            Meridional component of vector gradient

        Examples
        --------
        >>> abs_vrt = vw.absolutevorticity()
        >>> avrt_u, avrt_v = vw.gradient(abs_vrt)
        >>> avrt_u_t13, avrt_v_t13 = vw.gradient(abs_vrt, truncation=13)
        """
        if not isinstance(chi, xr.DataArray):
            raise TypeError(
                f"Scalar field must be xarray.DataArray, got {type(chi).__name__}"
            )

        name = chi.name or "field"

        # Process coordinate ordering similar to initialization
        lat, lat_dim = _find_latitude_coordinate(chi)
        lon, lon_dim = _find_longitude_coordinate(chi)

        target_s2n = bool(getattr(self._api.s, "south_to_north", False))
        chi_s2n = bool(lat.values[0] < lat.values[-1])
        if chi_s2n != target_s2n:
            chi = _reverse(chi, lat_dim)
            lat, lat_dim = _find_latitude_coordinate(chi)

        reorder = chi.dims
        use_c_layout = getattr(self._api, "_layout", "legacy") == "c"
        if use_c_layout:
            needs_transpose = not (
                lat_dim == chi.ndim - 2 and lon_dim == chi.ndim - 1
            )
            if needs_transpose:
                c_order = [
                    dim
                    for idx, dim in enumerate(chi.dims)
                    if idx not in (lat_dim, lon_dim)
                ] + [chi.dims[lat_dim], chi.dims[lon_dim]]
                chi = chi.transpose(*c_order)
        else:
            apiorder, _ = get_apiorder(chi.ndim, lat_dim, lon_dim)
            apiorder = [chi.dims[i] for i in apiorder]
            needs_transpose = tuple(apiorder) != chi.dims
            if needs_transpose:
                chi = chi.transpose(*apiorder)

        ishape = chi.shape
        coords = [chi.coords[n] for n in chi.dims]

        # Compute gradient using standard API
        u_grad, v_grad = self._api.gradient(chi.values, truncation=truncation)

        # Reshape and create DataArrays
        u_grad_arr = np.asarray(u_grad)
        v_grad_arr = np.asarray(v_grad)
        if u_grad_arr.shape != ishape:
            u_grad_arr = u_grad_arr.reshape(ishape)
        if v_grad_arr.shape != ishape:
            v_grad_arr = v_grad_arr.reshape(ishape)

        u_name = f"zonal_gradient_of_{name}"
        v_name = f"meridional_gradient_of_{name}"

        u_da = xr.DataArray(
            u_grad_arr, coords=coords, name=u_name, attrs={"long_name": u_name}
        )
        v_da = xr.DataArray(
            v_grad_arr, coords=coords, name=v_name, attrs={"long_name": v_name}
        )

        # Restore original dimension order if transposed
        if needs_transpose:
            u_da = u_da.transpose(*reorder)
            v_da = v_da.transpose(*reorder)

        return u_da, v_da

    def rossbywavesource(
        self, truncation: Optional[int] = None, omega: Optional[float] = None
    ) -> DataArray:
        """
        Calculate Rossby wave source.

        The Rossby wave source quantifies the generation of Rossby wave activity
        in the atmosphere through the interaction of divergence with absolute
        vorticity and the advection of absolute vorticity by the irrotational wind.

        Parameters
        ----------
        truncation : int, optional
            Triangular truncation limit for spherical harmonic computation.
            If None, uses the default truncation based on grid resolution.
        omega : float, optional
            Earth's angular velocity in rad/s. Default is 7.292e-5 s⁻¹.

        Returns
        -------
        DataArray
            Rossby wave source field with CF-compliant attributes

        See Also
        --------
        absolutevorticity : Calculate absolute vorticity
        divergence : Calculate horizontal divergence
        irrotationalcomponent : Calculate irrotational wind component
        gradient : Calculate vector gradient

        Notes
        -----
        The Rossby wave source is defined as:
        S = -ζₐ∇·v - v_χ·∇ζₐ

        where:
        - ζₐ is absolute vorticity (relative + planetary)
        - ∇·v is horizontal divergence
        - v_χ is the irrotational (divergent) wind component
        - ∇ζₐ is the gradient of absolute vorticity

        Positive values indicate Rossby wave generation, while negative values
        indicate wave absorption or dissipation.

        Examples
        --------
        >>> rws = vw.rossbywavesource()
        >>> rws_t21 = vw.rossbywavesource(truncation=21)
        >>> rws_custom_omega = vw.rossbywavesource(omega=7.2921150e-5)

        # Create a plot of Rossby wave source
        >>> import matplotlib.pyplot as plt
        >>> import cartopy.crs as ccrs
        >>>
        >>> ax = plt.axes(projection=ccrs.PlateCarree())
        >>> rws.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
        ...                   levels=20, cmap='RdBu_r')
        >>> ax.coastlines()
        >>> ax.gridlines()
        >>> plt.title('Rossby Wave Source')
        >>> plt.show()

        References
        ----------
        Sardeshmukh, P. D., & Hoskins, B. J. (1988). The generation of global
        rotational flow by steady idealized tropical heating. Journal of the
        Atmospheric Sciences, 45(7), 1228-1251.
        """
        rws = self._api.rossbywavesource(truncation=truncation, omega=omega)
        return self._metadata(
            rws,
            "rossby_wave_source",
            units="s**-2",
            standard_name="rossby_wave_source",
            long_name="rossby_wave_source_term",
            description="Generation term for Rossby wave activity",
        )

    def _prepare_scalar_dataarray(
        self, field: DataArray
    ) -> Tuple[DataArray, Tuple[Any, ...], bool, Tuple[int, ...], list[Any]]:
        """Align a scalar DataArray to the underlying API's lat/lon order and layout."""
        if not isinstance(field, xr.DataArray):
            raise TypeError(
                f"Field must be xarray.DataArray, got {type(field).__name__}"
            )

        lat, lat_dim = _find_latitude_coordinate(field)
        lon, lon_dim = _find_longitude_coordinate(field)

        target_s2n = bool(getattr(self._api.s, "south_to_north", False))
        field_s2n = bool(lat.values[0] < lat.values[-1])
        if field_s2n != target_s2n:
            field = _reverse(field, lat_dim)
            lat, lat_dim = _find_latitude_coordinate(field)

        reorder = field.dims
        use_c_layout = getattr(self._api, "_layout", "legacy") == "c"
        if use_c_layout:
            needs_transpose = not (
                lat_dim == field.ndim - 2 and lon_dim == field.ndim - 1
            )
            if needs_transpose:
                c_order = [
                    dim
                    for idx, dim in enumerate(field.dims)
                    if idx not in (lat_dim, lon_dim)
                ] + [field.dims[lat_dim], field.dims[lon_dim]]
                field = field.transpose(*c_order)
        else:
            apiorder, _ = get_apiorder(field.ndim, lat_dim, lon_dim)
            apiorder = [field.dims[i] for i in apiorder]
            needs_transpose = tuple(apiorder) != field.dims
            if needs_transpose:
                field = field.transpose(*apiorder)

        ishape = field.shape
        coords = [field.coords[n] for n in field.dims]
        return field, reorder, needs_transpose, ishape, coords

    def truncate(self, field: DataArray, truncation: Optional[int] = None) -> DataArray:
        """
        Apply spectral truncation to a scalar field.

        Parameters
        ----------
        field : DataArray
            Scalar field with same latitude/longitude dimensions as wind components
        truncation : int, optional
            Triangular truncation limit. If None, defaults to nlat-1

        Returns
        -------
        DataArray
            Field with spectral truncation applied

        Examples
        --------
        >>> field_trunc = vw.truncate(scalar_field)
        >>> field_t21 = vw.truncate(scalar_field, truncation=21)
        """
        prepared, reorder, needs_transpose, ishape, coords = (
            self._prepare_scalar_dataarray(field)
        )

        # Apply truncation using standard API
        field_trunc = np.asarray(
            self._api.truncate(prepared.values, truncation=truncation)
        )
        if field_trunc.shape != ishape:
            field_trunc = field_trunc.reshape(ishape)

        # Restore dimension order without coercing the computed result into the
        # input array's dtype.
        result = xr.DataArray(
            field_trunc,
            coords=coords,
            name=prepared.name,
            attrs=prepared.attrs,
        )
        if needs_transpose:
            result = result.transpose(*reorder)

        return result

    def invlap(self, field: DataArray, truncation: Optional[int] = None) -> DataArray:
        """
        Solve the spherical Poisson equation ``nabla^2 psi = field`` on the grid.

        Parameters
        ----------
        field : DataArray
            Scalar field with same latitude/longitude dimensions as wind components
        truncation : int, optional
            Triangular truncation limit. If None, defaults to nlat-1

        Returns
        -------
        DataArray
            Inverse Laplacian named ``"inverse_laplacian"`` preserving dims/coords
        """
        prepared, reorder, needs_transpose, ishape, coords = (
            self._prepare_scalar_dataarray(field)
        )

        field_inv = np.asarray(
            self._api.invlap(prepared.values, truncation=truncation)
        )
        if field_inv.shape != ishape:
            field_inv = field_inv.reshape(ishape)

        result = xr.DataArray(
            field_inv,
            coords=coords,
            name="inverse_laplacian",
            attrs={"long_name": "inverse_laplacian"},
        )
        if needs_transpose:
            result = result.transpose(*reorder)

        return result


class ReducedVectorWind:
    """
    Xarray interface for packed reduced-Gaussian vector wind analysis.

    Supports packed reduced-Gaussian point dimensions of length ``sum(pl)`` at
    any axis position, preferring C-order ``(..., sum(pl))`` or legacy
    ``(sum(pl), ...)`` without intermediate transposition.
    """

    def __init__(
        self,
        u: DataArray,
        v: DataArray,
        pl: Any,
        rsphere: float = 6.3712e6,
        legfunc: LegFunc = "stored",
        precision: Literal["auto", "single", "double"] = "auto",
        lat_order: str = "auto",
        lat: Optional[Any] = None,
    ) -> None:
        if not isinstance(u, xr.DataArray):
            raise TypeError(f"u must be xarray.DataArray, got {type(u).__name__}")
        if not isinstance(v, xr.DataArray):
            raise TypeError(f"v must be xarray.DataArray, got {type(v).__name__}")

        self._validate_coordinates(u, v)
        self._reorder = u.dims
        pl_arr = reduced.ReducedVectorWind._validate_pl(pl)
        npoints = int(pl_arr.sum())
        if u.ndim <= 1:
            self._needs_transpose = False
            target_layout = "c"
        elif u.shape[-1] == npoints and u.shape[0] != npoints:
            self._needs_transpose = False
            target_layout = "c"
        elif u.shape[0] == npoints and u.shape[-1] != npoints:
            self._needs_transpose = False
            target_layout = "legacy"
        elif npoints in u.shape:
            if u.shape[-1] == npoints and u.shape[0] == npoints:
                self._needs_transpose = False
                u_flags = u.values.flags
                target_layout = (
                    "legacy"
                    if u_flags.f_contiguous and not u_flags.c_contiguous
                    else "c"
                )
            else:
                point_dim = u.shape.index(npoints)
                self._needs_transpose = True
                c_order = [
                    dim for idx, dim in enumerate(u.dims) if idx != point_dim
                ] + [u.dims[point_dim]]
                u = u.transpose(*c_order)
                v = v.transpose(*c_order)
                target_layout = "c"
        else:
            self._needs_transpose = False
            target_layout = "auto"

        self._ishape = u.shape
        self._coords = [u.coords[name] for name in u.dims]
        self._u_component_dtype = reduced.ReducedVectorWind._infer_output_dtype(
            u.values
        )
        self._v_component_dtype = reduced.ReducedVectorWind._infer_output_dtype(
            v.values
        )

        self._api = reduced.ReducedVectorWind(
            u.values,
            v.values,
            pl_arr,
            rsphere=rsphere,
            legfunc=legfunc,
            precision=precision,
            layout=target_layout,
            lat_order=lat_order,
            lat=lat,
        )
        self.pl = self._api.pl
        self.gridtype = self._api.gridtype
        self.rsphere = self._api.rsphere
        self.legfunc = self._api.legfunc
        self.precision = getattr(self._api, "precision", precision)
        self.s = self._api.s
        self.rotationalcomponent = self.nondivergentcomponent
        self.divergentcomponent = self.irrotationalcomponent

    def _validate_coordinates(self, u: DataArray, v: DataArray) -> None:
        if u.dims != v.dims:
            raise ValueError(
                f"u and v must have identical dimensions. "
                f"Got u: {u.dims}, v: {v.dims}"
            )

        mismatched_coords = []
        for dim in u.dims:
            try:
                if not (u.coords[dim].values == v.coords[dim].values).all():
                    mismatched_coords.append(dim)
            except (ValueError, TypeError):
                mismatched_coords.append(dim)

        if mismatched_coords:
            raise ValueError(
                f"u and v must have identical coordinate values. "
                f"Mismatched coordinates: {mismatched_coords}"
            )

    def _metadata(self, data: Any, name: str, **attributes: Any) -> DataArray:
        result = xr.DataArray(
            np.asarray(data).reshape(self._ishape), coords=self._coords, name=name
        )
        if getattr(self, "_needs_transpose", True):
            result = result.transpose(*self._reorder)
        for attr, value in attributes.items():
            result.attrs[attr] = value
        return result

    def _component_metadata(
        self, data: Any, dtype: np.dtype, name: str, **attributes: Any
    ) -> DataArray:
        restored = self._api._restore_output_dtype(data, dtype)
        return self._metadata(restored, name, **attributes)

    def u(self) -> DataArray:
        """Return the packed zonal wind component."""
        return self._component_metadata(
            self._api.u,
            self._u_component_dtype,
            "u",
            units="m s**-1",
            standard_name="eastward_wind",
            long_name="eastward_component_of_wind",
        )

    def v(self) -> DataArray:
        """Return the packed meridional wind component."""
        return self._component_metadata(
            self._api.v,
            self._v_component_dtype,
            "v",
            units="m s**-1",
            standard_name="northward_wind",
            long_name="northward_component_of_wind",
        )

    def magnitude(self) -> DataArray:
        """Return wind speed on the packed reduced grid."""
        return self._metadata(
            self._api.magnitude(),
            "speed",
            units="m s**-1",
            standard_name="wind_speed",
            long_name="wind_speed",
        )

    def vrtdiv(self, truncation: Optional[int] = None) -> Tuple[DataArray, DataArray]:
        """Return relative vorticity and horizontal divergence."""
        vrt, div = self._api.vrtdiv(truncation=truncation)
        return (
            self._metadata(
                vrt,
                "vorticity",
                units="s**-1",
                standard_name="atmosphere_relative_vorticity",
                long_name="relative_vorticity",
            ),
            self._metadata(
                div,
                "divergence",
                units="s**-1",
                standard_name="divergence_of_wind",
                long_name="horizontal_divergence",
            ),
        )

    def vorticity(self, truncation: Optional[int] = None) -> DataArray:
        """Return relative vorticity."""
        return self._metadata(
            self._api.vorticity(truncation=truncation),
            "vorticity",
            units="s**-1",
            standard_name="atmosphere_relative_vorticity",
            long_name="relative_vorticity",
        )

    def divergence(self, truncation: Optional[int] = None) -> DataArray:
        """Return horizontal divergence."""
        return self._metadata(
            self._api.divergence(truncation=truncation),
            "divergence",
            units="s**-1",
            standard_name="divergence_of_wind",
            long_name="horizontal_divergence",
        )

    def planetaryvorticity(self, omega: Optional[float] = None) -> DataArray:
        """Return planetary vorticity."""
        return self._metadata(
            self._api.planetaryvorticity(omega=omega),
            "coriolis",
            units="s**-1",
            standard_name="coriolis_parameter",
            long_name="planetary_vorticity",
        )

    def absolutevorticity(
        self, omega: Optional[float] = None, truncation: Optional[int] = None
    ) -> DataArray:
        """Return absolute vorticity."""
        return self._metadata(
            self._api.absolutevorticity(omega=omega, truncation=truncation),
            "absolute_vorticity",
            units="s**-1",
            standard_name="atmosphere_absolute_vorticity",
            long_name="absolute_vorticity",
        )

    def sfvp(self, truncation: Optional[int] = None) -> Tuple[DataArray, DataArray]:
        """Return streamfunction and velocity potential."""
        sf, vp = self._api.sfvp(truncation=truncation)
        return (
            self._metadata(
                sf,
                "streamfunction",
                units="m**2 s**-1",
                standard_name="atmosphere_horizontal_streamfunction",
                long_name="streamfunction",
            ),
            self._metadata(
                vp,
                "velocity_potential",
                units="m**2 s**-1",
                standard_name="atmosphere_horizontal_velocity_potential",
                long_name="velocity potential",
            ),
        )

    def streamfunction(self, truncation: Optional[int] = None) -> DataArray:
        """Return streamfunction."""
        return self._metadata(
            self._api.streamfunction(truncation=truncation),
            "streamfunction",
            units="m**2 s**-1",
            standard_name="atmosphere_horizontal_streamfunction",
            long_name="streamfunction",
        )

    def velocitypotential(self, truncation: Optional[int] = None) -> DataArray:
        """Return velocity potential."""
        return self._metadata(
            self._api.velocitypotential(truncation=truncation),
            "velocity_potential",
            units="m**2 s**-1",
            standard_name="atmosphere_horizontal_velocity_potential",
            long_name="velocity potential",
        )

    def helmholtz(
        self, truncation: Optional[int] = None
    ) -> Tuple[DataArray, DataArray, DataArray, DataArray]:
        """Return irrotational and non-divergent wind components."""
        uchi, vchi, upsi, vpsi = self._api.helmholtz(truncation=truncation)
        return (
            self._metadata(
                uchi,
                "u_chi",
                units="m s**-1",
                long_name="irrotational_eastward_wind",
            ),
            self._metadata(
                vchi,
                "v_chi",
                units="m s**-1",
                long_name="irrotational_northward_wind",
            ),
            self._metadata(
                upsi,
                "u_psi",
                units="m s**-1",
                long_name="non_divergent_eastward_wind",
            ),
            self._metadata(
                vpsi,
                "v_psi",
                units="m s**-1",
                long_name="non_divergent_northward_wind",
            ),
        )

    def irrotationalcomponent(
        self, truncation: Optional[int] = None
    ) -> Tuple[DataArray, DataArray]:
        """Return irrotational wind component."""
        uchi, vchi = self._api.irrotationalcomponent(truncation=truncation)
        return (
            self._metadata(
                uchi,
                "u_chi",
                units="m s**-1",
                long_name="irrotational_eastward_wind",
            ),
            self._metadata(
                vchi,
                "v_chi",
                units="m s**-1",
                long_name="irrotational_northward_wind",
            ),
        )

    def nondivergentcomponent(
        self, truncation: Optional[int] = None
    ) -> Tuple[DataArray, DataArray]:
        """Return non-divergent wind component."""
        upsi, vpsi = self._api.nondivergentcomponent(truncation=truncation)
        return (
            self._metadata(
                upsi,
                "u_psi",
                units="m s**-1",
                long_name="non_divergent_eastward_wind",
            ),
            self._metadata(
                vpsi,
                "v_psi",
                units="m s**-1",
                long_name="non_divergent_northward_wind",
            ),
        )

    def gradient(
        self, chi: DataArray, truncation: Optional[int] = None
    ) -> Tuple[DataArray, DataArray]:
        """Return vector gradient of a packed scalar field."""
        if not isinstance(chi, xr.DataArray):
            raise TypeError(
                f"Scalar field must be xarray.DataArray, got {type(chi).__name__}"
            )
        name = chi.name or "field"
        api_dims = tuple(c.name for c in self._coords)
        if chi.dims != api_dims and set(chi.dims) == set(api_dims):
            chi = chi.transpose(*api_dims)
        u_grad, v_grad = self._api.gradient(chi.values, truncation=truncation)
        return (
            self._metadata(
                u_grad,
                f"zonal_gradient_of_{name}",
                long_name=f"zonal_gradient_of_{name}",
            ),
            self._metadata(
                v_grad,
                f"meridional_gradient_of_{name}",
                long_name=f"meridional_gradient_of_{name}",
            ),
        )

    def _prepare_scalar_dataarray(self, field: DataArray) -> DataArray:
        if not isinstance(field, xr.DataArray):
            raise TypeError(
                f"Field must be xarray.DataArray, got {type(field).__name__}"
            )
        api_dims = tuple(c.name for c in self._coords)
        if field.dims != api_dims and set(field.dims) == set(api_dims):
            field = field.transpose(*api_dims)
        return field

    def truncate(self, field: DataArray, truncation: Optional[int] = None) -> DataArray:
        """Apply spectral truncation to a packed scalar field."""
        field = self._prepare_scalar_dataarray(field)
        truncated = self._api.truncate(field.values, truncation=truncation)
        result = xr.DataArray(
            truncated.reshape(self._ishape),
            coords=self._coords,
            name=field.name,
            attrs=field.attrs,
        )
        if getattr(self, "_needs_transpose", True):
            result = result.transpose(*self._reorder)
        return result

    def invlap(self, field: DataArray, truncation: Optional[int] = None) -> DataArray:
        """Solve the spherical Poisson equation ``nabla^2 psi = field`` on the packed grid."""
        prepared = self._prepare_scalar_dataarray(field)
        inv = self._api.invlap(prepared.values, truncation=truncation)
        return self._metadata(
            inv,
            "inverse_laplacian",
            long_name="inverse_laplacian",
        )

    def rossbywavesource(
        self, truncation: Optional[int] = None, omega: Optional[float] = None
    ) -> DataArray:
        """Return Rossby wave source on the packed reduced grid."""
        return self._metadata(
            self._api.rossbywavesource(truncation=truncation, omega=omega),
            "rossby_wave_source",
            units="s**-2",
            standard_name="rossby_wave_source",
            long_name="rossby_wave_source_term",
            description="Generation term for Rossby wave activity",
        )


def _reverse(array: DataArray, dim: int) -> DataArray:
    """
    Reverse an xarray DataArray along a given dimension.

    Parameters
    ----------
    array : DataArray
        Input DataArray to reverse
    dim : int
        Dimension index to reverse

    Returns
    -------
    DataArray
        Array with specified dimension reversed
    """
    slicers = [slice(None)] * array.ndim
    slicers[dim] = slice(None, None, -1)
    return array[tuple(slicers)]


def _find_coord_and_dim(
    array: DataArray, predicate: Callable[[Any], bool], name: str
) -> Tuple[Any, int]:
    """
    Find a dimension coordinate in DataArray that satisfies a predicate.

    Parameters
    ----------
    array : DataArray
        Input DataArray to search
    predicate : callable
        Function that returns True for the desired coordinate
    name : str
        Name of coordinate type for error messages

    Returns
    -------
    coord : coordinate
        Found coordinate that satisfies predicate
    dim : int
        Dimension index of the coordinate

    Raises
    ------
    ValueError
        If no coordinate or multiple coordinates found
    """
    candidates = [
        coord for coord in [array.coords[n] for n in array.dims] if predicate(coord)
    ]

    if not candidates:
        raise ValueError(f"Cannot find a {name} coordinate")

    if len(candidates) > 1:
        raise ValueError(f"Multiple {name} coordinates are not allowed")

    coord = candidates[0]
    dim = array.dims.index(coord.name)
    return coord, dim


def _find_latitude_coordinate(array: DataArray) -> Tuple[Any, int]:
    """
    Find latitude dimension coordinate in an xarray DataArray.

    Parameters
    ----------
    array : DataArray
        Input DataArray to search for latitude coordinate

    Returns
    -------
    lat_coord : coordinate
        Latitude coordinate
    lat_dim : int
        Latitude dimension index

    Raises
    ------
    ValueError
        If latitude coordinate cannot be found or multiple found
    """

    def is_latitude(coord: Any) -> bool:
        """Check if coordinate represents latitude."""
        return (
            coord.name
            in ("latitude", "lat", "LAT", "LATITUDE", "Y", "y", "LATS", "YLAT")
            or coord.attrs.get("units") == "degrees_north"
            or coord.attrs.get("axis") == "Y"
        )

    return _find_coord_and_dim(array, is_latitude, "latitude")


def _find_longitude_coordinate(array: DataArray) -> Tuple[Any, int]:
    """
    Find longitude dimension coordinate in an xarray DataArray.

    Parameters
    ----------
    array : DataArray
        Input DataArray to search for longitude coordinate

    Returns
    -------
    lon_coord : coordinate
        Longitude coordinate
    lon_dim : int
        Longitude dimension index

    Raises
    ------
    ValueError
        If longitude coordinate cannot be found or multiple found
    """

    def is_longitude(coord: Any) -> bool:
        """Check if coordinate represents longitude."""
        return (
            coord.name
            in ("longitude", "lon", "LON", "LONGITUDE", "X", "x", "LONS", "XLONG")
            or coord.attrs.get("units") == "degrees_east"
            or coord.attrs.get("axis") == "X"
        )

    return _find_coord_and_dim(array, is_longitude, "longitude")
