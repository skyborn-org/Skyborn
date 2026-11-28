"""Reduced-Gaussian vector wind analysis for packed fields."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np

from skyborn.spharm import ReducedGaussianSpharmt, gaussian_lats_wts

ArrayLike = Union[np.ndarray, np.ma.MaskedArray]
LegFunc = Literal["stored", "computed"]
PrecisionMode = Literal["auto", "single", "float32", "double", "float64"]
LayoutMode = Literal["legacy", "c", "auto"]
LatOrderMode = Literal["north_to_south", "south_to_north", "N2S", "S2N", "auto"]


class ReducedVectorWind:
    """
    Vector wind analysis on packed reduced Gaussian grids.

    Parameters
    ----------
    u, v:
        Packed zonal and meridional wind components with shape
        ``(..., sum(pl))`` (C-order, zero-copy) or ``(sum(pl), ...)`` (legacy).
    pl:
        Reduced Gaussian row lengths, ordered north-to-south.
    rsphere:
        Sphere radius in meters.
    legfunc:
        Legendre function mode passed to grouped Gaussian ``Spharmt`` backends.
    precision:
        Public output precision mode. ``"auto"`` preserves the promoted input
        floating precision, ``"single"``/``"float32"`` returns float32 outputs,
        and ``"double"``/``"float64"`` returns float64 outputs.
    layout:
        Array axis ordering: ``"auto"`` (default, auto-detects C-order
        ``(..., sum(pl))`` and legacy ``(sum(pl), ...)``), ``"c"``, or ``"legacy"``.
    lat_order:
        Latitude direction mode (``"auto"``, ``"N2S"``, ``"S2N"``,
        ``"north_to_south"``, or ``"south_to_north"``).
    lat:
        Optional 1D latitude array for south-to-north auto-detection.

    Attributes
    ----------
    u, v:
        Packed zonal and meridional wind components.
    pl:
        Reduced Gaussian row lengths ordered north-to-south.
    gridtype:
        Fixed grid type label ``"reduced_gaussian"``.
    rsphere:
        Sphere radius in meters.
    legfunc:
        Legendre function handling mode used by the underlying transform.
    precision:
        Public output precision mode.
    layout:
        Resolved array layout mode (``"c"`` or ``"legacy"``).
    lat_order:
        Normalized latitude ordering mode.
    south_to_north:
        Whether input/output rings are ordered south-to-north.
    s:
        Underlying :class:`skyborn.spharm.ReducedGaussianSpharmt` instance.
    """

    def __init__(
        self,
        u: ArrayLike,
        v: ArrayLike,
        pl: ArrayLike,
        rsphere: float = 6.3712e6,
        legfunc: LegFunc = "stored",
        precision: PrecisionMode = "auto",
        layout: LayoutMode = "auto",
        lat_order: LatOrderMode = "auto",
        lat: Optional[np.ndarray] = None,
    ) -> None:
        """
        Create a reduced-Gaussian vector wind analysis wrapper.

        Args:
            u: Packed zonal wind component.
            v: Packed meridional wind component with the same shape as ``u``.
            pl: Reduced Gaussian row lengths ordered north-to-south.
            rsphere: Sphere radius in meters.
            legfunc: Legendre function handling mode - ``"stored"`` or
                ``"computed"``.
            precision: Public output precision mode. ``"auto"`` preserves the
                promoted input floating precision, ``"single"`` returns
                float32 outputs, and ``"double"`` returns float64 outputs.
            layout: Array axis convention (``"auto"``, ``"c"``, or ``"legacy"``).
            lat_order: Latitude ordering mode (``"auto"``, ``"N2S"``, or ``"S2N"``).
            lat: Optional 1D latitude coordinate array for S2N auto-detection.

        Raises:
            ValueError: If ``u``, ``v``, or ``pl`` fails validation.
        """
        raw_u = np.asarray(u)
        self.pl = self._validate_pl(pl)
        self._output_dtype = self._infer_output_dtype(u, v)
        self._precision = ReducedGaussianSpharmt._validate_precision(precision)
        self._lat = None if lat is None else np.asarray(lat, dtype=np.float64)
        self.u = self._process_input_array(u, "u")
        self.v = self._process_input_array(v, "v")
        self._layout, self._extra_shape = self._validate_input_data(
            layout, raw_u.flags
        )
        self._nt = (
            int(np.prod(self._extra_shape, dtype=int)) if self._extra_shape else 1
        )
        spharm_layout = (
            "auto"
            if (str(layout).lower() == "auto" and self.u.ndim == 1)
            else self._layout
        )
        self.s = ReducedGaussianSpharmt(
            self.pl,
            rsphere=rsphere,
            legfunc=legfunc,
            precision=precision,
            layout=spharm_layout,
            lat_order=lat_order,
            lat=self._lat,
        )
        self._u_backend = self._normalize_grid_backend(self.u)
        self._v_backend = self._normalize_grid_backend(self.v)

        self.gridtype = self.s.gridtype
        self.rsphere = self.s.rsphere
        self.legfunc = self.s.legfunc
        self.precision = self.s.precision
        self.layout = self._layout
        self.lat_order = self.s.lat_order
        self.south_to_north = self.s.south_to_north
        self.rotationalcomponent = self.nondivergentcomponent
        self.divergentcomponent = self.irrotationalcomponent
        self._latitude_cache: Optional[np.ndarray] = None
        self._planetary_vorticity_cache: Dict[
            Tuple[float, str, Tuple[int, ...]], np.ndarray
        ] = {}
        self._planetary_vorticity_spec_cache: Dict[Tuple[float, int], np.ndarray] = {}
        self._spectral_cache: Dict[
            Tuple[str, int], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        ] = {}

    @staticmethod
    def _validate_pl(pl: ArrayLike) -> np.ndarray:
        arr = np.asarray(pl, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"pl must be rank 1, got shape {arr.shape}")
        if arr.size < 3:
            raise ValueError(f"pl must contain at least 3 rows, got {arr.size}")
        if np.any(arr < 4):
            raise ValueError("each reduced Gaussian row must contain >= 4 points")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _infer_output_dtype(*arrays: ArrayLike) -> np.dtype:
        output_dtype = None
        for arr in arrays:
            dtype = np.dtype(np.asarray(arr).dtype)
            if np.issubdtype(dtype, np.floating):
                public_dtype = dtype
            else:
                public_dtype = np.dtype(np.float64)
            output_dtype = (
                public_dtype
                if output_dtype is None
                else np.result_type(output_dtype, public_dtype)
            )
        return np.dtype(np.float64 if output_dtype is None else output_dtype)

    def _resolved_output_dtype(
        self, output_dtype: Optional[np.dtype] = None
    ) -> np.dtype:
        if self._precision == "double":
            return np.dtype(np.float64)
        if self._precision == "single":
            return np.dtype(np.float32)
        return self._output_dtype if output_dtype is None else np.dtype(output_dtype)

    def _restore_output_dtype(
        self, data: np.ndarray, output_dtype: Optional[np.dtype] = None
    ) -> np.ndarray:
        array = np.asarray(data)
        target_dtype = self._resolved_output_dtype(output_dtype)
        if array.dtype == target_dtype:
            return array
        return array.astype(target_dtype, copy=False)

    def _normalize_grid_backend(
        self, data: np.ndarray, compute_dtype: Optional[np.dtype] = None
    ) -> np.ndarray:
        if compute_dtype is None:
            target_dtype = (
                np.dtype(np.float64)
                if self._resolved_output_dtype() == np.dtype(np.float64)
                else np.dtype(np.float32)
            )
        else:
            target_dtype = np.dtype(compute_dtype)

        npoints = int(self.pl.sum())
        arr = np.asarray(data)
        if arr.ndim == 1:
            return np.ascontiguousarray(arr, dtype=target_dtype).reshape(1, npoints).T
        if getattr(self, "_layout", "legacy") == "c":
            return np.ascontiguousarray(
                arr.reshape(self._nt, npoints), dtype=target_dtype
            ).T
        return np.asfortranarray(arr.reshape(npoints, self._nt), dtype=target_dtype)

    def _restore_grid(
        self, datagrid: np.ndarray, output_dtype: Optional[np.dtype] = None
    ) -> np.ndarray:
        arr = np.asarray(datagrid)
        if not self._extra_shape:
            out = arr[:, 0]
        elif getattr(self, "_layout", "legacy") == "c":
            out = arr.T.reshape(self._extra_shape + (int(self.pl.sum()),))
        else:
            out = arr.reshape((int(self.pl.sum()),) + self._extra_shape)
        return out.astype(self._resolved_output_dtype(output_dtype), copy=False)

    @staticmethod
    def _process_input_array(arr: ArrayLike, name: str) -> np.ndarray:
        try:
            if hasattr(arr, "filled"):
                masked = np.ma.asarray(arr)
                dtype = masked.dtype
                if not np.issubdtype(dtype, np.floating):
                    dtype = np.dtype(np.float64)
                return np.ma.asarray(masked, dtype=dtype).filled(np.nan).copy()

            raw = np.asarray(arr)
            dtype = np.dtype(raw.dtype)
            if not np.issubdtype(dtype, np.floating):
                dtype = np.dtype(np.float64)
            return np.asarray(raw, dtype=dtype).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot convert {name} to numpy array: {exc}") from exc

    def _validate_input_data(
        self, layout: str = "auto", raw_flags: object = None
    ) -> Tuple[str, Tuple[int, ...]]:
        if self.u.shape != self.v.shape:
            raise ValueError(
                f"Wind components must have identical shapes. "
                f"Got u: {self.u.shape}, v: {self.v.shape}"
            )
        if self.u.ndim < 1:
            raise ValueError(
                f"Packed wind components must be at least 1D, got {self.u.ndim}D"
            )
        expected = int(self.pl.sum())
        norm_layout = str(layout).lower()
        if norm_layout not in ("legacy", "c", "auto"):
            raise ValueError(
                f'layout must be "legacy", "c", or "auto", got "{layout}"'
            )

        if self.u.ndim == 1:
            if self.u.shape[0] != expected:
                raise ValueError(
                    f"Packed first dimension must be sum(pl)={expected}, "
                    f"got {self.u.shape[0]}"
                )
            resolved_layout, extra_shape = "c", ()
        elif norm_layout == "legacy":
            if self.u.shape[0] != expected:
                raise ValueError(
                    f"Packed first dimension must be sum(pl)={expected}, "
                    f"got {self.u.shape[0]}"
                )
            resolved_layout, extra_shape = "legacy", tuple(self.u.shape[1:])
        elif norm_layout == "c":
            if self.u.shape[-1] != expected:
                raise ValueError(
                    f"Packed trailing dimension must be sum(pl)={expected}, "
                    f"got {self.u.shape[-1]}"
                )
            resolved_layout, extra_shape = "c", tuple(self.u.shape[:-1])
        else:
            is_c = self.u.shape[-1] == expected
            is_legacy = self.u.shape[0] == expected
            if is_c and not is_legacy:
                resolved_layout, extra_shape = "c", tuple(self.u.shape[:-1])
            elif is_legacy and not is_c:
                resolved_layout, extra_shape = "legacy", tuple(self.u.shape[1:])
            elif is_c and is_legacy:
                if getattr(raw_flags, "f_contiguous", False) and not getattr(
                    raw_flags, "c_contiguous", True
                ):
                    resolved_layout, extra_shape = "legacy", tuple(self.u.shape[1:])
                else:
                    resolved_layout, extra_shape = "c", tuple(self.u.shape[:-1])
            else:
                raise ValueError(
                    f"Packed wind dimension must be sum(pl)={expected}, "
                    f"got shape {self.u.shape}"
                )

        if not np.isfinite(self.u).all() or not np.isfinite(self.v).all():
            raise ValueError("Input wind components must be finite")
        return resolved_layout, extra_shape

    def _validate_scalar_field(self, field: ArrayLike, name: str) -> np.ndarray:
        try:
            field = field.filled(np.nan) if hasattr(field, "filled") else field
        except AttributeError:
            pass
        array = np.asarray(field)
        if array.shape != self.u.shape:
            raise ValueError(f"{name} is not compatible")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} cannot contain missing or infinite values")
        return array

    @property
    def grid_info(self) -> dict:
        return {
            "gridtype": self.gridtype,
            "nlat": int(self.pl.size),
            "npoints": int(self.pl.sum()),
            "shape": self.u.shape,
            "rsphere": self.rsphere,
            "legfunc": self.legfunc,
            "pl": self.pl.copy(),
        }

    def _latitude_values(self) -> np.ndarray:
        if self._latitude_cache is None:
            s2n = bool(getattr(self.s, "south_to_north", False))
            pl_ordered = self.pl[::-1] if s2n else self.pl
            explicit_lat = getattr(self, "_lat", None)
            if explicit_lat is not None:
                lat_ordered = explicit_lat
            else:
                lat, _ = gaussian_lats_wts(int(self.pl.size))
                lat_ordered = lat[::-1] if s2n else lat
            self._latitude_cache = np.repeat(lat_ordered, pl_ordered)
        return self._latitude_cache

    def _planetary_vorticity_values(
        self,
        omega: Optional[float] = None,
        dtype: Optional[np.dtype] = None,
        shape: Optional[Tuple[int, ...]] = None,
        point_axis: Optional[int] = None,
    ) -> np.ndarray:
        omega_value = 7.292e-05 if omega is None else float(omega)
        target_dtype = self._u_backend.dtype if dtype is None else np.dtype(dtype)
        target_shape = self.u.shape if shape is None else tuple(shape)
        resolved_axis = (
            (-1 if getattr(self, "_layout", "legacy") == "c" else 0)
            if point_axis is None
            else point_axis
        )
        key = (omega_value, target_dtype.str, target_shape, resolved_axis)
        cached = self._planetary_vorticity_cache.get(key)
        if cached is not None:
            return cached

        npoints = int(self.pl.sum())
        values = np.asarray(
            2.0 * omega_value * np.sin(np.deg2rad(self._latitude_values())),
            dtype=target_dtype,
        )
        if len(target_shape) == 1:
            broadcast = np.broadcast_to(values, target_shape)
            result = np.array(broadcast, copy=True)
        elif resolved_axis == 0:
            row = values.reshape(npoints, *([1] * (len(target_shape) - 1)))
            result = np.broadcast_to(row, target_shape)
        else:
            row = values.reshape(*([1] * (len(target_shape) - 1)), npoints)
            result = np.broadcast_to(row, target_shape)
        self._planetary_vorticity_cache[key] = result
        return result

    def _planetary_vorticity(
        self, omega: Optional[float] = None, materialize: bool = True
    ) -> np.ndarray:
        data = self._planetary_vorticity_values(
            omega=omega,
            dtype=self._output_dtype if materialize else self._u_backend.dtype,
            shape=self.u.shape,
        )
        if materialize:
            return np.array(data, copy=True)
        return data

    def _planetary_vorticity_grid(self, omega: Optional[float] = None) -> np.ndarray:
        return self._planetary_vorticity_values(
            omega=omega,
            dtype=self._u_backend.dtype,
            shape=self._u_backend.shape,
            point_axis=0,
        )

    def _planetary_vorticity_spec(
        self, truncation: Optional[int] = None, omega: Optional[float] = None
    ) -> np.ndarray:
        ntrunc = self._ntrunc(truncation)
        omega_value = 7.292e-05 if omega is None else float(omega)
        key = (omega_value, ntrunc)
        cached = self._planetary_vorticity_spec_cache.get(key)
        if cached is not None:
            return cached

        dataspec = self.s._analyze_scalar(self._planetary_vorticity_grid(omega), ntrunc)
        self._planetary_vorticity_spec_cache[key] = dataspec
        return dataspec

    def _coriolis(self, omega: Optional[float] = None) -> np.ndarray:
        return self._planetary_vorticity_values(
            omega=omega, shape=(int(self.pl.sum()),)
        )

    def _ntrunc(self, truncation: Optional[int]) -> int:
        return self.s._validate_ntrunc(truncation)

    def _spectogrd_pair(
        self,
        spec_a: np.ndarray,
        spec_b: np.ndarray,
        truncation: Optional[int] = None,
        output_dtype: Optional[np.dtype] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ntrunc = self._ntrunc(truncation)
        grid_a, grid_b = self.s._synthesize_scalar_pair(spec_a, spec_b, ntrunc)
        return self._restore_grid(grid_a, output_dtype), self._restore_grid(
            grid_b, output_dtype
        )

    def _spectogrd(
        self,
        spec: np.ndarray,
        truncation: Optional[int] = None,
        output_dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        ntrunc = self._ntrunc(truncation)
        return self._restore_grid(self.s._synthesize_scalar(spec, ntrunc), output_dtype)

    def _vector_analysis_spectral(
        self, truncation: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        ntrunc = self._ntrunc(truncation)
        key = ("vrtdiv", ntrunc)
        cached = self._spectral_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        vrtspec, divspec = self.s._analyze_wind(
            self._u_backend, self._v_backend, ntrunc
        )
        self._spectral_cache[key] = (vrtspec, divspec)
        self._spectral_cache[("vrt", ntrunc)] = vrtspec
        self._spectral_cache[("div", ntrunc)] = divspec
        return vrtspec, divspec

    def _vorticity_spectral(self, truncation: Optional[int]) -> np.ndarray:
        ntrunc = self._ntrunc(truncation)
        key = ("vrt", ntrunc)
        cached = self._spectral_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        full = self._spectral_cache.get(("vrtdiv", ntrunc))
        if full is not None:
            vrtspec = full[0]  # type: ignore[index]
        else:
            vrtspec = self.s._analyze_vorticity(
                self._u_backend, self._v_backend, ntrunc
            )
        self._spectral_cache[key] = vrtspec
        return vrtspec

    def _divergence_spectral(self, truncation: Optional[int]) -> np.ndarray:
        ntrunc = self._ntrunc(truncation)
        key = ("div", ntrunc)
        cached = self._spectral_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        full = self._spectral_cache.get(("vrtdiv", ntrunc))
        if full is not None:
            divspec = full[1]  # type: ignore[index]
        else:
            divspec = self.s._analyze_divergence(
                self._u_backend, self._v_backend, ntrunc
            )
        self._spectral_cache[key] = divspec
        return divspec

    def magnitude(self) -> np.ndarray:
        """Return wind speed."""
        return self._restore_output_dtype(np.hypot(self.u, self.v))

    def vrtdiv(self, truncation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return relative vorticity and horizontal divergence."""
        vrtspec, divspec = self._vector_analysis_spectral(truncation)
        return self._spectogrd_pair(vrtspec, divspec, truncation)

    def vorticity(self, truncation: Optional[int] = None) -> np.ndarray:
        """Return relative vorticity."""
        vrtspec = self._vorticity_spectral(truncation)
        return self._spectogrd(vrtspec, truncation)

    def divergence(self, truncation: Optional[int] = None) -> np.ndarray:
        """Return horizontal divergence."""
        divspec = self._divergence_spectral(truncation)
        return self._spectogrd(divspec, truncation)

    def planetaryvorticity(self, omega: Optional[float] = None) -> np.ndarray:
        """Return planetary vorticity on the packed reduced grid."""
        return self._restore_output_dtype(self._planetary_vorticity(omega=omega))

    def absolutevorticity(
        self, omega: Optional[float] = None, truncation: Optional[int] = None
    ) -> np.ndarray:
        """Return absolute vorticity."""
        vrt = self.vorticity(truncation=truncation)
        return self._restore_output_dtype(
            vrt + self._planetary_vorticity(omega=omega, materialize=False)
        )

    def sfvp(self, truncation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return streamfunction and velocity potential."""
        vrtspec, divspec = self._vector_analysis_spectral(truncation)
        return self._spectogrd_pair(
            self.s._invlap(vrtspec),
            self.s._invlap(divspec),
            truncation,
        )

    def streamfunction(self, truncation: Optional[int] = None) -> np.ndarray:
        """Return streamfunction."""
        vrtspec = self._vorticity_spectral(truncation)
        return self._spectogrd(self.s._invlap(vrtspec), truncation)

    def velocitypotential(self, truncation: Optional[int] = None) -> np.ndarray:
        """Return velocity potential."""
        divspec = self._divergence_spectral(truncation)
        return self._spectogrd(self.s._invlap(divspec), truncation)

    def helmholtz(
        self, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return irrotational and non-divergent wind components."""
        vrtspec, divspec = self._vector_analysis_spectral(truncation)
        psispec = self.s._invlap(vrtspec)
        chispec = self.s._invlap(divspec)
        u_chi, v_chi, v_psi, u_psi = self.s._synthesize_gradient_pair(
            chispec, psispec, truncation
        )
        return (
            self._restore_grid(u_chi),
            self._restore_grid(v_chi),
            self._restore_grid(-u_psi),
            self._restore_grid(v_psi),
        )

    def irrotationalcomponent(
        self, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return irrotational wind component."""
        divspec = self._divergence_spectral(truncation)
        u_chi, v_chi = self.s._synthesize_gradient(self.s._invlap(divspec), truncation)
        return self._restore_grid(u_chi), self._restore_grid(v_chi)

    def nondivergentcomponent(
        self, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return non-divergent wind component."""
        vrtspec = self._vorticity_spectral(truncation)
        psispec = self.s._invlap(vrtspec)
        v_psi, u_psi = self.s._synthesize_gradient(psispec, truncation)
        return self._restore_grid(-u_psi), self._restore_grid(v_psi)

    def gradient(
        self, chi: ArrayLike, truncation: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return vector gradient of a packed scalar field."""
        output_dtype = self._infer_output_dtype(chi)
        compute_dtype = (
            np.dtype(np.float64)
            if self._resolved_output_dtype(output_dtype) == np.dtype(np.float64)
            else np.dtype(np.float32)
        )
        field = self._validate_scalar_field(chi, "chi")
        chi_spec = self.s._analyze_scalar(
            self._normalize_grid_backend(field, compute_dtype=compute_dtype),
            truncation,
        )
        u_chi, v_chi = self.s._synthesize_gradient(chi_spec, truncation)
        return (
            self._restore_grid(u_chi, output_dtype),
            self._restore_grid(v_chi, output_dtype),
        )

    def truncate(
        self, field: ArrayLike, truncation: Optional[int] = None
    ) -> np.ndarray:
        """Apply spectral truncation to a packed scalar field."""
        output_dtype = self._infer_output_dtype(field)
        compute_dtype = (
            np.dtype(np.float64)
            if self._resolved_output_dtype(output_dtype) == np.dtype(np.float64)
            else np.dtype(np.float32)
        )
        scalar = self._validate_scalar_field(field, "field")
        field_spec = self.s._analyze_scalar(
            self._normalize_grid_backend(scalar, compute_dtype=compute_dtype),
            truncation,
        )
        return self._spectogrd(field_spec, truncation, output_dtype)

    def rossbywavesource(
        self, truncation: Optional[int] = None, omega: Optional[float] = None
    ) -> np.ndarray:
        """Return Rossby wave source on the packed reduced grid."""
        vrtspec, divspec = self._vector_analysis_spectral(truncation)
        vrt, div = self.s._synthesize_scalar_pair(vrtspec, divspec, truncation)
        eta = vrt + self._planetary_vorticity_grid(omega=omega)
        uchi, vchi = self.s._synthesize_gradient(self.s._invlap(divspec), truncation)
        etaspec = vrtspec + self._planetary_vorticity_spec(truncation, omega=omega)
        etax, etay = self.s._synthesize_gradient(etaspec, truncation)
        rws = -eta * div - (uchi * etax + vchi * etay)
        return self._restore_grid(rws)
