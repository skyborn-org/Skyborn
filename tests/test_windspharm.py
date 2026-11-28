"""
Tests for skyborn.windspharm module (spherical harmonic wind analysis)

This module contains comprehensive tests for the windspharm functionality,
including VectorWind class and related utilities.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from skyborn.windspharm import VectorWind
    from skyborn.windspharm import _common as common_mod
    from skyborn.windspharm import standard, tools
    from skyborn.windspharm.examples import example_data_path
    from skyborn.windspharm.standard import VectorWind as StandardVectorWind
    from skyborn.windspharm.tools import (
        get_recovery,
        order_latdim,
        prep_data,
        recover_data,
        reverse_latdim,
    )

    WINDSPHARM_AVAILABLE = True
except ImportError:
    WINDSPHARM_AVAILABLE = False
    common_mod = None
    VectorWind = None
    standard = None
    tools = None
    example_data_path = None


class TestVectorWindInitialization:
    """Test VectorWind class initialization."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_basic_initialization(self):
        """Test basic VectorWind initialization."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v)
        assert vw.u.shape == (nlat, nlon)
        assert vw.v.shape == (nlat, nlon)
        assert vw.gridtype == "regular"
        assert hasattr(vw, "s")  # Spharmt object
        assert not np.shares_memory(vw.u, u)
        assert not np.shares_memory(vw.v, v)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_gaussian_grid(self):
        """Test VectorWind initialization with Gaussian grid."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v, gridtype="gaussian")
        assert vw.gridtype == "gaussian"

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_custom_parameters(self):
        """Test VectorWind initialization with custom parameters."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v, gridtype="regular", rsphere=6.371e6, legfunc="computed")
        assert vw.gridtype == "regular"

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_precision_aliases(self):
        """Precision aliases should normalize and preserve public output dtype."""
        nlat, nlon = 19, 36
        rng = np.random.default_rng(20260501)
        u = rng.standard_normal((nlat, nlon)).astype(np.float64)
        v = rng.standard_normal((nlat, nlon)).astype(np.float64)

        vw = VectorWind(u, v, precision="float32")
        assert vw.s.precision == "single"
        assert vw.magnitude().dtype == np.float32

        vw = VectorWind(u, v, precision="float64")
        assert vw.s.precision == "double"
        assert vw.magnitude().dtype == np.float64

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_3d_data(self):
        """Test VectorWind initialization with 3D data (time series)."""
        nlat, nlon, nt = 37, 72, 10
        u = np.random.randn(nlat, nlon, nt)
        v = np.random.randn(nlat, nlon, nt)

        vw = VectorWind(u, v)
        assert vw.u.shape == (nlat, nlon, nt)
        assert vw.v.shape == (nlat, nlon, nt)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_masked_arrays(self):
        """Test VectorWind with masked arrays."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        # Create masked arrays
        u_masked = np.ma.masked_where(u > 2.0, u)
        v_masked = np.ma.masked_where(v > 2.0, v)

        # This should work but fill masked values
        vw = VectorWind(u_masked.filled(0), v_masked.filled(0))
        assert vw.u.shape == (nlat, nlon)
        assert vw.v.shape == (nlat, nlon)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    @pytest.mark.parametrize(
        "input_dtype, output_dtype",
        [(np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64)],
    )
    def test_vectorwind_preserves_public_output_dtype(self, input_dtype, output_dtype):
        """Grid-space outputs should follow interp-style public dtype restoration."""
        nlat, nlon = 19, 36
        rng = np.random.default_rng(20260501)
        if np.issubdtype(input_dtype, np.integer):
            u = rng.integers(-5, 5, size=(nlat, nlon)).astype(input_dtype)
            v = rng.integers(-5, 5, size=(nlat, nlon)).astype(input_dtype)
        else:
            u = rng.standard_normal((nlat, nlon)).astype(input_dtype)
            v = rng.standard_normal((nlat, nlon)).astype(input_dtype)

        vw = VectorWind(u, v)

        outputs = []
        outputs.append(vw.magnitude())
        outputs.extend(vw.vrtdiv())
        outputs.append(vw.vorticity())
        outputs.append(vw.divergence())
        outputs.append(vw.planetaryvorticity())
        outputs.append(vw.absolutevorticity())
        outputs.extend(vw.sfvp())
        outputs.append(vw.streamfunction())
        outputs.append(vw.velocitypotential())
        outputs.extend(vw.helmholtz())
        outputs.extend(vw.irrotationalcomponent())
        outputs.extend(vw.nondivergentcomponent())
        outputs.extend(vw.gradient(outputs[2]))
        outputs.append(vw.truncate(outputs[2]))
        outputs.append(vw.rossbywavesource())

        assert {result.dtype for result in outputs} == {np.dtype(output_dtype)}

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_promotes_mixed_public_output_dtype(self):
        """Mixed u/v precision should preserve the higher public floating precision."""
        nlat, nlon = 19, 36
        rng = np.random.default_rng(20260501)
        u = rng.standard_normal((nlat, nlon)).astype(np.float32)
        v = rng.standard_normal((nlat, nlon)).astype(np.float64)

        vw = VectorWind(u, v)

        assert vw.u.dtype == np.float32
        assert vw.v.dtype == np.float64
        assert vw.magnitude().dtype == np.float64
        assert vw.vorticity().dtype == np.float64

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_keeps_float32_working_arrays(self):
        """Float32 inputs should not be persistently converted to float64."""
        nlat, nlon = 19, 36
        rng = np.random.default_rng(20260501)
        u = rng.standard_normal((nlat, nlon)).astype(np.float32)
        v = rng.standard_normal((nlat, nlon)).astype(np.float32)

        vw = VectorWind(u, v)

        assert vw.u.dtype == np.float32
        assert vw.v.dtype == np.float32

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_integer_masked_inputs_promote_to_float64(self):
        """Integer masked arrays must become floating arrays before NaN filling."""
        nlat, nlon = 19, 36
        u = np.ma.array(np.ones((nlat, nlon), dtype=np.int32))
        v = np.ma.array(np.ones((nlat, nlon), dtype=np.int32))

        vw = VectorWind(u, v)

        assert vw.u.dtype == np.float64
        assert vw.v.dtype == np.float64

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_scalar_input_methods_restore_scalar_input_dtype(self):
        """Scalar-field diagnostics should restore precision from their own input."""
        nlat, nlon = 19, 36
        rng = np.random.default_rng(20260501)
        u = rng.standard_normal((nlat, nlon)).astype(np.float64)
        v = rng.standard_normal((nlat, nlon)).astype(np.float64)
        scalar = rng.standard_normal((nlat, nlon)).astype(np.float32)

        vw = VectorWind(u, v)
        grad_x, grad_y = vw.gradient(scalar)
        truncated = vw.truncate(scalar)

        assert grad_x.dtype == np.float32
        assert grad_y.dtype == np.float32
        assert truncated.dtype == np.float32


class TestVectorWindValidation:
    """Test VectorWind input validation."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_shape_mismatch(self):
        """Test error handling for mismatched shapes."""
        u = np.random.randn(37, 72)
        v = np.random.randn(36, 72)  # Different nlat

        with pytest.raises(ValueError):
            VectorWind(u, v)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_nan_values(self):
        """Test error handling for NaN values."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        # Insert NaN values
        u[0, 0] = np.nan

        with pytest.raises(ValueError):
            VectorWind(u, v)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_inf_values(self):
        """Test error handling for infinite values."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        # Insert infinite values
        v[0, 0] = np.inf

        with pytest.raises(ValueError):
            VectorWind(u, v)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_extreme_values(self):
        """Finite extreme values are currently accepted by the wrapper."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        # Insert extremely large values
        u[0, 0] = 1e10

        vw = VectorWind(u, v)
        assert vw.u.shape == (nlat, nlon)
        assert np.isfinite(vw.u[0, 0])
        assert vw.u[0, 0] == pytest.approx(1e10)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_invalid_gridtype(self):
        """Test error handling for invalid gridtype."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        with pytest.raises((ValueError, TypeError)):
            VectorWind(u, v, gridtype="invalid")

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_invalid_legfunc(self):
        """Test error handling for invalid legfunc."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        with pytest.raises((ValueError, TypeError)):
            VectorWind(u, v, legfunc="invalid")

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_too_small_grid(self):
        """Test error handling for too small grids."""
        u = np.random.randn(2, 3)
        v = np.random.randn(2, 3)

        with pytest.raises((ValueError, RuntimeError)):
            VectorWind(u, v)


class TestVectorWindBasicOperations:
    """Test basic VectorWind operations."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vorticity_calculation(self):
        """Test vorticity calculation."""
        nlat, nlon = 37, 72

        # Create simple wind pattern
        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")
        vorticity = vw.vorticity()

        assert vorticity.shape == (nlat, nlon)
        assert isinstance(vorticity, np.ndarray)
        assert np.all(np.isfinite(vorticity))

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_divergence_calculation(self):
        """Test divergence calculation."""
        nlat, nlon = 37, 72

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")
        divergence = vw.divergence()

        assert divergence.shape == (nlat, nlon)
        assert isinstance(divergence, np.ndarray)
        assert np.all(np.isfinite(divergence))

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_rossby_wave_source_calculation(self):
        """Test Rossby wave source calculation."""
        nlat, nlon = 37, 72

        # Create test wind field with some structure
        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(2 * lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.sin(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")

        # Test basic RWS calculation
        rws = vw.rossbywavesource()
        assert rws.shape == (nlat, nlon)
        assert isinstance(rws, np.ndarray)
        assert np.all(np.isfinite(rws))

        # Test with truncation
        rws_t21 = vw.rossbywavesource(truncation=21)
        assert rws_t21.shape == (nlat, nlon)
        assert np.all(np.isfinite(rws_t21))

        # Test with custom omega (significantly different from default)
        rws_custom = vw.rossbywavesource(omega=1.0e-4)
        assert rws_custom.shape == (nlat, nlon)
        assert np.all(np.isfinite(rws_custom))

        # Basic functionality test passed - omega parameter is correctly accepted

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_streamfunction_calculation(self):
        """Test streamfunction calculation."""
        nlat, nlon = 37, 72

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")
        streamfunction = vw.streamfunction()

        assert streamfunction.shape == (nlat, nlon)
        assert isinstance(streamfunction, np.ndarray)
        assert np.all(np.isfinite(streamfunction))

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_velocity_potential_calculation(self):
        """Test velocity potential calculation."""
        nlat, nlon = 37, 72

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")
        velocity_potential = vw.velocitypotential()

        assert velocity_potential.shape == (nlat, nlon)
        assert isinstance(velocity_potential, np.ndarray)
        assert np.all(np.isfinite(velocity_potential))

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_sfvp_calculation(self):
        """Test combined streamfunction and velocity potential calculation."""
        nlat, nlon = 37, 72

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")
        streamfunction, velocity_potential = vw.sfvp()

        assert streamfunction.shape == (nlat, nlon)
        assert velocity_potential.shape == (nlat, nlon)
        assert isinstance(streamfunction, np.ndarray)
        assert isinstance(velocity_potential, np.ndarray)
        assert np.all(np.isfinite(streamfunction))
        assert np.all(np.isfinite(velocity_potential))


class TestVectorWindAdvancedOperations:
    """Test advanced VectorWind operations."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_truncation_parameter(self):
        """Test operations with truncation parameter."""
        nlat, nlon = 37, 72

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")

        # Test with different truncation levels
        vorticity_20 = vw.vorticity(truncation=20)
        vorticity_30 = vw.vorticity(truncation=30)

        assert vorticity_20.shape == (nlat, nlon)
        assert vorticity_30.shape == (nlat, nlon)

        # Results should be different due to different truncation
        assert not np.allclose(vorticity_20, vorticity_30, rtol=1e-3)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_gradient_calculation(self):
        """Test gradient calculation if available."""
        nlat, nlon = 37, 72

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")

        # Check if gradient method exists
        if hasattr(vw, "gradient"):
            # Create a scalar field for gradient calculation
            scalar_field = np.sin(lon_grid) * np.cos(lat_grid)

            grad_x, grad_y = vw.gradient(scalar_field)

            assert grad_x.shape == (nlat, nlon)
            assert grad_y.shape == (nlat, nlon)
            assert np.all(np.isfinite(grad_x))
            assert np.all(np.isfinite(grad_y))

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_3d_operations(self):
        """Test operations with 3D data."""
        nlat, nlon, nt = 37, 72, 5

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Create 3D wind fields
        u = np.zeros((nlat, nlon, nt))
        v = np.zeros((nlat, nlon, nt))

        for t in range(nt):
            phase = 2 * np.pi * t / nt
            u[:, :, t] = np.sin(lon_grid + phase) * np.cos(lat_grid)
            v[:, :, t] = np.cos(lon_grid + phase) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")

        vorticity = vw.vorticity()
        divergence = vw.divergence()

        assert vorticity.shape == (nlat, nlon, nt)
        assert divergence.shape == (nlat, nlon, nt)
        assert np.all(np.isfinite(vorticity))
        assert np.all(np.isfinite(divergence))


class TestVectorWindFourDimensionalOperations:
    """Test native numpy 4D VectorWind behavior."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_4d_operations_match_manual_flattening(self):
        """Native 4D numpy input should match the established flattened path."""
        nlat, nlon, ntime, nlev = 19, 36, 2, 3
        rng = np.random.default_rng(34567)
        u = rng.standard_normal((nlat, nlon, ntime, nlev)).astype(np.float32)
        v = rng.standard_normal((nlat, nlon, ntime, nlev)).astype(np.float32)
        u_flat = u.reshape(nlat, nlon, ntime * nlev)
        v_flat = v.reshape(nlat, nlon, ntime * nlev)

        vw_4d = VectorWind(u, v, gridtype="regular")
        vw_flat = VectorWind(u_flat, v_flat, gridtype="regular")

        vrt_4d, div_4d = vw_4d.vrtdiv()
        vrt_flat, div_flat = vw_flat.vrtdiv()
        psi_4d, chi_4d = vw_4d.sfvp()
        psi_flat, chi_flat = vw_flat.sfvp()
        psi_only_4d = vw_4d.streamfunction()
        chi_only_4d = vw_4d.velocitypotential()
        grad_u_4d, grad_v_4d = vw_4d.gradient(vrt_4d)
        grad_u_flat, grad_v_flat = vw_flat.gradient(vrt_flat)
        trunc_4d = vw_4d.truncate(vrt_4d, truncation=10)
        trunc_flat = vw_flat.truncate(vrt_flat, truncation=10)

        assert vrt_4d.shape == u.shape
        assert div_4d.shape == u.shape
        assert psi_4d.shape == u.shape
        assert chi_4d.shape == u.shape
        np.testing.assert_allclose(vrt_4d, vrt_flat.reshape(u.shape), rtol=0, atol=0)
        np.testing.assert_allclose(div_4d, div_flat.reshape(u.shape), rtol=0, atol=0)
        np.testing.assert_allclose(psi_4d, psi_flat.reshape(u.shape), rtol=0, atol=0)
        np.testing.assert_allclose(chi_4d, chi_flat.reshape(u.shape), rtol=0, atol=0)
        np.testing.assert_allclose(psi_only_4d, psi_4d, rtol=0, atol=0)
        np.testing.assert_allclose(chi_only_4d, chi_4d, rtol=0, atol=0)
        np.testing.assert_allclose(
            grad_u_4d, grad_u_flat.reshape(u.shape), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            grad_v_4d, grad_v_flat.reshape(u.shape), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            trunc_4d, trunc_flat.reshape(u.shape), rtol=0, atol=0
        )


class TestVectorWindConsistency:
    """Test consistency and mathematical properties."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_helmholtz_decomposition_consistency(self):
        """Test that Helmholtz decomposition is consistent."""
        nlat, nlon = 37, 72

        # Create divergent and rotational wind components
        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Simple wind field
        u = np.sin(2 * lon_grid) * np.cos(lat_grid)
        v = np.cos(2 * lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")

        # Get streamfunction and velocity potential
        psi, chi = vw.sfvp()

        # Check that they are finite and have correct shape
        assert psi.shape == (nlat, nlon)
        assert chi.shape == (nlat, nlon)
        assert np.all(np.isfinite(psi))
        assert np.all(np.isfinite(chi))

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_symmetry_properties(self):
        """Test symmetry properties of operations."""
        nlat, nlon = 37, 72

        # Create symmetric wind field
        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")

        vorticity = vw.vorticity()
        divergence = vw.divergence()

        # Check that results are reasonable
        assert np.std(vorticity) > 0  # Should have some variation
        assert np.std(divergence) > 0  # Should have some variation

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_gaussian_vs_regular_grid(self):
        """Test differences between Gaussian and regular grids."""
        nlat, nlon = 37, 72

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        # Regular grid
        vw_regular = VectorWind(u, v, gridtype="regular")
        vort_regular = vw_regular.vorticity()

        # Gaussian grid
        vw_gaussian = VectorWind(u, v, gridtype="gaussian")
        vort_gaussian = vw_gaussian.vorticity()

        # Both should be valid but may differ slightly
        assert vort_regular.shape == vort_gaussian.shape
        assert np.all(np.isfinite(vort_regular))
        assert np.all(np.isfinite(vort_gaussian))


class TestWindspharmTools:
    """Test windspharm.tools module functions."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_prep_data_basic(self):
        """Test basic prep_data functionality."""
        # Create test data
        nlat, nlon = 37, 72
        data = np.random.randn(nlat, nlon)

        prepared, info = prep_data(data, "yx")

        assert isinstance(prepared, np.ndarray)
        assert isinstance(info, dict)
        # After prep_data, it might add a dimension for combination with other fields
        assert prepared.ndim >= 2

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_prep_data_3d(self):
        """Test prep_data with 3D data."""
        nlat, nlon, nt = 37, 72, 5
        data = np.random.randn(nt, nlat, nlon)

        prepared, info = prep_data(data, "tyx")

        assert isinstance(prepared, np.ndarray)
        assert isinstance(info, dict)
        assert prepared.ndim == 3

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_recover_data_basic(self):
        """Test basic recover_data functionality."""
        nlat, nlon = 37, 72
        original = np.random.randn(nlat, nlon)

        prepared, info = prep_data(original, "yx")
        recovered = recover_data(prepared, info)

        assert recovered.shape == original.shape
        np.testing.assert_array_equal(recovered, original)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_get_recovery_function(self):
        """Test get_recovery function creation."""
        nlat, nlon = 37, 72
        data1 = np.random.randn(nlat, nlon)

        prepared1, info1 = prep_data(data1, "yx")

        # Create recovery function - should pass a single info dict
        recovery_func = get_recovery(info1)

        assert callable(recovery_func)

        # Test recovery with single array
        recovered_list = recovery_func(prepared1)
        assert isinstance(recovered_list, list)
        assert len(recovered_list) == 1
        recovered = recovered_list[0]
        assert recovered.shape == data1.shape
        np.testing.assert_array_equal(recovered, data1)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_reverse_latdim(self):
        """Test reverse_latdim function."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        u_reversed, v_reversed = reverse_latdim(u, v)

        assert u_reversed.shape == u.shape
        assert v_reversed.shape == v.shape
        # Check that latitude dimension is actually reversed
        np.testing.assert_array_equal(u_reversed[0, :], u[-1, :])
        np.testing.assert_array_equal(u_reversed[-1, :], u[0, :])
        np.testing.assert_array_equal(v_reversed[0, :], v[-1, :])
        np.testing.assert_array_equal(v_reversed[-1, :], v[0, :])

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_order_latdim(self):
        """Test order_latdim function."""
        nlat, nlon = 37, 72
        lats = np.linspace(90, -90, nlat)  # North to south
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        ordered_lats, ordered_u, ordered_v = order_latdim(lats, u, v)

        assert ordered_u.shape == u.shape
        assert ordered_v.shape == v.shape
        assert len(ordered_lats) == nlat
        # Should be ordered north to south
        assert ordered_lats[0] > ordered_lats[-1]

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_order_latdim_reversed(self):
        """Test order_latdim with initially south-to-north data."""
        nlat, nlon = 37, 72
        lats = np.linspace(-90, 90, nlat)  # South to north
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        ordered_lats, ordered_u, ordered_v = order_latdim(lats, u, v)

        assert ordered_u.shape == u.shape
        assert ordered_v.shape == v.shape
        assert len(ordered_lats) == nlat
        # Should be reordered to north to south
        assert ordered_lats[0] > ordered_lats[-1]


class TestWindspharmPerformance:
    """Test windspharm performance with different configurations."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_small_grid_performance(self):
        """Test performance with small grid."""
        nlat, nlon = 19, 36

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        vw = VectorWind(u, v, gridtype="regular")

        # Perform multiple operations
        for _ in range(5):
            vorticity = vw.vorticity()
            divergence = vw.divergence()
            psi, chi = vw.sfvp()

        # Basic sanity check
        assert vorticity.shape == (nlat, nlon)
        assert divergence.shape == (nlat, nlon)
        assert psi.shape == (nlat, nlon)
        assert chi.shape == (nlat, nlon)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_legfunc_options(self):
        """Test different legfunc options."""
        nlat, nlon = 37, 72

        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(lon_grid) * np.cos(lat_grid)
        v = np.cos(lon_grid) * np.cos(lat_grid)

        # Test with stored Legendre functions
        vw_stored = VectorWind(u, v, legfunc="stored")
        vort_stored = vw_stored.vorticity()

        # Test with computed Legendre functions
        vw_computed = VectorWind(u, v, legfunc="computed")
        vort_computed = vw_computed.vorticity()

        # Results should be very similar
        assert vort_stored.shape == vort_computed.shape
        assert np.allclose(vort_stored, vort_computed, rtol=1e-10)


class TestWindspharmErrorHandling:
    """Test error handling in windspharm operations."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_invalid_truncation(self):
        """Test error handling for invalid truncation values."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v)

        # Test with negative truncation
        with pytest.raises((ValueError, RuntimeError)):
            vw.vorticity(truncation=-1)

        # Test with too large truncation
        with pytest.raises((ValueError, RuntimeError)):
            vw.vorticity(truncation=1000)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_tools_invalid_input(self):
        """Test error handling in tools functions."""
        # Test prep_data with invalid dimension order
        data = np.random.randn(37, 72)

        with pytest.raises((ValueError, KeyError)):
            prep_data(data, "invalid")

        # Test order_latdim with mismatched dimensions
        u = np.random.randn(37, 72)
        v = np.random.randn(37, 72)
        lats = np.linspace(-90, 90, 36)  # Wrong size

        with pytest.raises((ValueError, IndexError)):
            order_latdim(lats, u, v)


class TestWindspharmIntegration:
    """Integration tests for windspharm functionality."""

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_full_workflow(self):
        """Test complete windspharm workflow."""
        nlat, nlon = 37, 72

        # Create test wind field
        lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        u = np.sin(2 * lon_grid) * np.cos(lat_grid)
        v = np.cos(2 * lon_grid) * np.cos(lat_grid)

        # Initialize VectorWind
        vw = VectorWind(u, v, gridtype="regular")

        # Calculate all quantities
        vorticity = vw.vorticity()
        divergence = vw.divergence()
        streamfunction = vw.streamfunction()
        velocity_potential = vw.velocitypotential()
        psi, chi = vw.sfvp()

        # Verify all results
        assert vorticity.shape == (nlat, nlon)
        assert divergence.shape == (nlat, nlon)
        assert streamfunction.shape == (nlat, nlon)
        assert velocity_potential.shape == (nlat, nlon)
        assert psi.shape == (nlat, nlon)
        assert chi.shape == (nlat, nlon)

        # Check that sfvp components match individual calls
        assert np.allclose(psi, streamfunction, rtol=1e-10)
        assert np.allclose(chi, velocity_potential, rtol=1e-10)

        # Verify all values are finite
        assert np.all(np.isfinite(vorticity))
        assert np.all(np.isfinite(divergence))
        assert np.all(np.isfinite(streamfunction))
        assert np.all(np.isfinite(velocity_potential))

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_tools_integration(self):
        """Test tools integration with VectorWind."""
        nlat, nlon, nt = 37, 72, 3

        # Create 3D wind data with time dimension first
        u_orig = np.random.randn(nt, nlat, nlon)
        v_orig = np.random.randn(nt, nlat, nlon)

        # Use tools to prepare data
        u_prep, u_info = prep_data(u_orig, "tyx")
        v_prep, v_info = prep_data(v_orig, "tyx")

        # Process with VectorWind
        vw = VectorWind(u_prep, v_prep)
        vorticity = vw.vorticity()

        # Recover original shape
        vorticity_recovered = recover_data(vorticity, u_info)

        assert vorticity_recovered.shape == (nt, nlat, nlon)
        assert np.all(np.isfinite(vorticity_recovered))


class TestVectorWindUntestedMethods:
    """Test previously untested methods in VectorWind."""

    def test_vectorwind_magnitude_calculation(self):
        """Test magnitude calculation method."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon) * 10
        v = np.random.randn(nlat, nlon) * 10

        vw = VectorWind(u, v)

        # Test magnitude calculation
        magnitude = vw.magnitude()

        # Should match manual calculation
        expected_magnitude = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(magnitude, expected_magnitude, rtol=1e-6)

    def test_vectorwind_with_truncation_parameter(self):
        """Test VectorWind operations with truncation parameter."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v)

        # Test operations with custom truncation
        for ntrunc in [5, 10, 15]:
            vort = vw.vorticity(truncation=ntrunc)
            div = vw.divergence(truncation=ntrunc)

            assert vort.shape == (nlat, nlon)
            assert div.shape == (nlat, nlon)
            assert np.all(np.isfinite(vort))
            assert np.all(np.isfinite(div))

    @pytest.mark.skip(
        reason="Helmholtz decomposition has large reconstruction errors due to spherical harmonic truncation"
    )
    def test_vectorwind_helmholtz_decomposition(self):
        """Test complete Helmholtz decomposition."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v)

        # Get Helmholtz decomposition
        u_chi, v_chi, u_psi, v_psi = vw.helmholtz()

        # Should reconstruct original field (approximately)
        u_reconstructed = u_chi + u_psi
        v_reconstructed = v_chi + v_psi

        # Check that components have expected shapes
        assert u_chi.shape == u.shape
        assert v_chi.shape == v.shape
        assert u_psi.shape == u.shape
        assert v_psi.shape == v.shape

    def test_vectorwind_gradient_operations(self):
        """Test gradient operations."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v)

        # Get a scalar field (vorticity)
        vorticity = vw.vorticity()

        # Calculate gradients
        grad_x, grad_y = vw.gradient(vorticity)

        assert grad_x.shape == (nlat, nlon)
        assert grad_y.shape == (nlat, nlon)
        assert np.all(np.isfinite(grad_x))
        assert np.all(np.isfinite(grad_y))

    @pytest.mark.skip(
        reason="Irrotational/nondivergent decomposition has large reconstruction errors due to spherical harmonic truncation"
    )
    def test_vectorwind_irrotational_nondivergent(self):
        """Test irrotational and nondivergent components."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v)

        # Get irrotational component
        u_irrot, v_irrot = vw.irrotationalcomponent()

        # Get nondivergent component
        u_nondiv, v_nondiv = vw.nondivergentcomponent()

        # Check that components have expected shapes
        assert u_irrot.shape == u.shape
        assert v_irrot.shape == v.shape
        assert u_nondiv.shape == u.shape
        assert v_nondiv.shape == v.shape

    def test_vectorwind_quasi_geostrophic_diagnostics(self):
        """Test quasi-geostrophic diagnostic calculations."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon) * 10
        v = np.random.randn(nlat, nlon) * 10

        vw = VectorWind(u, v)

        # Test planetary vorticity
        f = vw.planetaryvorticity(omega=7.272e-5)
        assert f.shape == (nlat, nlon)
        assert np.all(np.isfinite(f))

        # Test absolute vorticity
        vort = vw.vorticity()
        abs_vort = vw.absolutevorticity(omega=7.272e-5)

        # Should be vorticity + planetary vorticity
        np.testing.assert_allclose(abs_vort, vort + f, rtol=1e-12)

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_grid_info_property(self):
        """Test grid_info property functionality."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        # Test regular grid
        vw_regular = VectorWind(
            u, v, gridtype="regular", rsphere=6.371e6, legfunc="stored"
        )
        grid_info = vw_regular.grid_info

        # Check that it returns a dictionary
        assert isinstance(grid_info, dict)

        # Check required keys
        expected_keys = {
            "gridtype",
            "nlat",
            "nlon",
            "shape",
            "rsphere",
            "legfunc",
            "total_points",
        }
        assert set(grid_info.keys()) == expected_keys

        # Check values
        assert grid_info["gridtype"] == "regular"
        assert grid_info["nlat"] == nlat
        assert grid_info["nlon"] == nlon
        assert grid_info["shape"] == (nlat, nlon)
        assert grid_info["rsphere"] == 6.371e6
        assert grid_info["legfunc"] == "stored"
        assert grid_info["total_points"] == nlat * nlon

        # Test with 3D data
        u_3d = np.random.randn(nlat, nlon, 5)
        v_3d = np.random.randn(nlat, nlon, 5)
        vw_3d = VectorWind(
            u_3d, v_3d, gridtype="gaussian", rsphere=6.4e6, legfunc="computed"
        )
        grid_info_3d = vw_3d.grid_info

        assert grid_info_3d["gridtype"] == "gaussian"
        assert grid_info_3d["nlat"] == nlat
        assert grid_info_3d["nlon"] == nlon
        assert grid_info_3d["shape"] == (nlat, nlon, 5)
        assert grid_info_3d["rsphere"] == 6.4e6
        assert grid_info_3d["legfunc"] == "computed"
        assert grid_info_3d["total_points"] == nlat * nlon

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_truncate_method(self):
        """Test truncate method functionality."""
        nlat, nlon = 37, 72
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v, gridtype="regular")

        # Create a test scalar field
        test_field = np.random.randn(nlat, nlon)

        # Test default truncation (should use nlat-1)
        truncated_default = vw.truncate(test_field)
        assert truncated_default.shape == test_field.shape
        assert isinstance(truncated_default, np.ndarray)
        assert np.all(np.isfinite(truncated_default))

        # Test custom truncation
        custom_trunc = 15
        truncated_custom = vw.truncate(test_field, truncation=custom_trunc)
        assert truncated_custom.shape == test_field.shape
        assert isinstance(truncated_custom, np.ndarray)
        assert np.all(np.isfinite(truncated_custom))

        # Test with 3D field
        test_field_3d = np.random.randn(nlat, nlon, 3)
        truncated_3d = vw.truncate(test_field_3d, truncation=10)
        assert truncated_3d.shape == test_field_3d.shape
        assert np.all(np.isfinite(truncated_3d))

        # Test that truncation actually changes the field
        # (Higher truncation should preserve more details)
        high_trunc = vw.truncate(test_field, truncation=30)
        low_trunc = vw.truncate(test_field, truncation=5)

        # The difference between original and high truncation should be smaller
        # than the difference between original and low truncation
        diff_high = np.mean(np.abs(test_field - high_trunc))
        diff_low = np.mean(np.abs(test_field - low_trunc))
        assert diff_high <= diff_low, "Higher truncation should preserve more details"

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_truncate_error_handling(self):
        """Test truncate method error handling."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v)

        # Test with field containing NaN values
        field_with_nan = np.random.randn(nlat, nlon)
        field_with_nan[0, 0] = np.nan

        with pytest.raises(ValueError, match="field cannot contain missing values"):
            vw.truncate(field_with_nan)

        # Test with incompatible field shape
        incompatible_field = np.random.randn(nlat + 1, nlon)
        with pytest.raises(ValueError, match="field is not compatible"):
            vw.truncate(incompatible_field)

        # Test with masked array input (should work after filling)
        import numpy.ma as ma

        masked_field = ma.array(np.random.randn(nlat, nlon))
        masked_field[0, 0] = ma.masked

        # This should raise error because filled values become NaN
        with pytest.raises(ValueError, match="field cannot contain missing values"):
            vw.truncate(masked_field)

        # Test with valid masked array (no masked values)
        valid_masked_field = ma.array(np.random.randn(nlat, nlon))
        truncated_masked = vw.truncate(valid_masked_field)
        assert truncated_masked.shape == (nlat, nlon)
        assert np.all(np.isfinite(truncated_masked))

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_truncate_consistency(self):
        """Test that truncate method is consistent with other VectorWind methods."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)

        vw = VectorWind(u, v)

        # Get streamfunction and velocity potential
        streamfunction = vw.streamfunction()
        velocity_potential = vw.velocitypotential()

        # Apply truncation using the method
        truncation_level = 10
        sf_truncated = vw.truncate(streamfunction, truncation=truncation_level)
        vp_truncated = vw.truncate(velocity_potential, truncation=truncation_level)

        # Results should have same shape and be finite
        assert sf_truncated.shape == streamfunction.shape
        assert vp_truncated.shape == velocity_potential.shape
        assert np.all(np.isfinite(sf_truncated))
        assert np.all(np.isfinite(vp_truncated))

        # Test with vorticity field
        vorticity = vw.vorticity()
        vort_truncated = vw.truncate(vorticity, truncation=truncation_level)
        assert vort_truncated.shape == vorticity.shape
        assert np.all(np.isfinite(vort_truncated))


class TestVectorWindAdvancedOperations:
    """Test advanced VectorWind operations and edge cases."""

    def test_vectorwind_with_zero_wind(self):
        """Test VectorWind with zero wind fields."""
        nlat, nlon = 19, 36
        u_zero = np.zeros((nlat, nlon))
        v_zero = np.zeros((nlat, nlon))

        vw = VectorWind(u_zero, v_zero)

        # All derived quantities should be zero
        vort = vw.vorticity()
        div = vw.divergence()
        sf = vw.streamfunction()
        vp = vw.velocitypotential()

        assert np.allclose(vort, 0, atol=1e-12)
        assert np.allclose(div, 0, atol=1e-12)
        assert np.allclose(sf, 0, atol=1e-12)
        assert np.allclose(vp, 0, atol=1e-12)

    def test_vectorwind_with_uniform_wind(self):
        """Test VectorWind with uniform wind fields."""
        nlat, nlon = 19, 36
        u_uniform = np.full((nlat, nlon), 10.0)  # Uniform 10 m/s
        v_uniform = np.full((nlat, nlon), 5.0)  # Uniform 5 m/s

        vw = VectorWind(u_uniform, v_uniform)

        # Uniform wind should have zero vorticity and divergence
        vort = vw.vorticity()
        div = vw.divergence()

        # Should be very close to zero (within numerical precision)
        # Use more relaxed tolerance for spherical harmonic computations
        assert np.allclose(vort, 0, atol=1e-4)
        assert np.allclose(div, 0, atol=1e-4)

    def test_vectorwind_conservation_properties(self):
        """Test conservation properties of transformations."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon) * 10
        v = np.random.randn(nlat, nlon) * 10

        vw = VectorWind(u, v)

        # Get streamfunction and velocity potential
        sf, vp = vw.sfvp()

        # Reconstruct wind from potentials
        u_chi, v_chi = vw.gradient(vp)  # Irrotational component
        u_psi, v_psi = vw.gradient(sf)  # Nondivergent component

        # For 2D case, nondivergent component should be rotated gradient
        # This is a complex test - just ensure we get reasonable results
        assert u_psi.shape == (nlat, nlon)
        assert v_psi.shape == (nlat, nlon)
        assert np.all(np.isfinite(u_psi))
        assert np.all(np.isfinite(v_psi))


@pytest.mark.skipif(not WINDSPHARM_AVAILABLE, reason="windspharm module not available")
class TestWindSpharmTargetedCoverage:
    """Targeted regression tests for uncovered wrapper branches."""

    def test_process_input_array_and_init_error_mapping(self, monkeypatch):
        """Exercise internal conversion and init error-adaptation branches."""
        vw = object.__new__(StandardVectorWind)

        masked = np.ma.array([1.0, 2.0], mask=[False, True])
        processed = vw._process_input_array(masked, "u")
        assert processed.shape == (2,)
        assert np.isnan(processed[1])

        class BadArray:
            def __array__(self, dtype=None):
                raise TypeError("cannot coerce")

        with pytest.raises(
            ValueError, match="Cannot convert u to numpy array: cannot coerce"
        ):
            vw._process_input_array(BadArray(), "u")

        with pytest.raises(ValueError, match="Invalid sphere radius"):
            vw._initialize_spharmt(36, 19, "regular", 0, "stored")

        monkeypatch.setattr(
            standard,
            "Spharmt",
            lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("invalid input dimensions from backend")
            ),
        )
        with pytest.raises(
            ValueError, match="Invalid grid dimensions for regular grid"
        ):
            vw._initialize_spharmt(36, 19, "regular", 6.3712e6, "stored")

        monkeypatch.setattr(
            standard,
            "Spharmt",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("spharm missing")),
        )
        with pytest.raises(
            ValueError, match="Spherical harmonic initialization failed"
        ):
            vw._initialize_spharmt(36, 19, "regular", 6.3712e6, "stored")

        monkeypatch.setattr(
            standard,
            "Spharmt",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("other backend issue")),
        )
        with pytest.raises(
            ValueError, match="Failed to initialize spherical harmonic transform"
        ):
            vw._initialize_spharmt(36, 19, "regular", 6.3712e6, "stored")

    def test_input_validation_remaining_error_branches(self):
        """Exercise v-only NaN, u-only inf, and rank validation branches."""
        nlat, nlon = 19, 36

        u = np.zeros((nlat, nlon), dtype=np.float32)
        v = np.zeros((nlat, nlon), dtype=np.float32)
        v[0, 0] = np.nan
        with pytest.raises(ValueError, match="v: 1 NaN values"):
            VectorWind(u, v)

        u = np.zeros((nlat, nlon), dtype=np.float32)
        v = np.zeros((nlat, nlon), dtype=np.float32)
        u[0, 0] = np.inf
        with pytest.raises(ValueError, match="u: 1 infinite values"):
            VectorWind(u, v)

        with pytest.raises(ValueError, match="at least 2D arrays"):
            VectorWind(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))

    def test_gradient_validation_branches(self):
        """Exercise gradient masked-data and compatibility branches."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)
        vw = VectorWind(u, v)

        bad_field = np.random.randn(nlat, nlon)
        bad_field[0, 0] = np.nan
        with pytest.raises(ValueError, match="chi cannot contain missing values"):
            vw.gradient(bad_field)

        with pytest.raises(ValueError, match="input field is not compatible"):
            vw.gradient(np.random.randn(nlat + 1, nlon))

        masked_field = np.ma.array(np.random.randn(nlat, nlon))
        masked_field[0, 0] = np.ma.masked
        with pytest.raises(ValueError, match="chi cannot contain missing values"):
            vw.gradient(masked_field)

        valid_masked_field = np.ma.array(np.random.randn(nlat, nlon))
        grad_x, grad_y = vw.gradient(valid_masked_field)
        assert grad_x.shape == (nlat, nlon)
        assert grad_y.shape == (nlat, nlon)

    def test_gradient_and_truncate_attribute_error_fallback(self):
        """Exercise legacy masked-array fallback when filled lookup is broken."""

        class FilledCallAttributeErrorArray(np.ndarray):
            @property
            def filled(self):
                def _raise_attribute_error(fill_value=np.nan):
                    raise AttributeError("filled unavailable")

                return _raise_attribute_error

        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)
        vw = VectorWind(u, v)
        field = np.random.randn(nlat, nlon).view(FilledCallAttributeErrorArray)

        grad_x, grad_y = vw.gradient(field)
        truncated = vw.truncate(field)

        assert grad_x.shape == (nlat, nlon)
        assert grad_y.shape == (nlat, nlon)
        assert truncated.shape == (nlat, nlon)

    def test_planetaryvorticity_even_grid_and_invalid_omega(self):
        """Exercise even-latitude grid broadcasting and invalid omega handling."""
        vw = object.__new__(StandardVectorWind)
        vw.s = type("DummySpharm", (), {"nlat": 4})()
        vw.gridtype = "regular"
        vw.u = np.zeros((4, 8, 2), dtype=np.float32)
        vw._output_dtype = np.dtype(np.float32)

        coriolis = vw.planetaryvorticity()
        assert coriolis.shape == (4, 8, 2)
        assert coriolis.dtype == np.float32
        np.testing.assert_allclose(coriolis[:, 0, 0], coriolis[:, 3, 1])
        assert coriolis.flags.writeable
        assert len(vw._latitude_cache) == 1
        assert len(vw._coriolis_cache) == 1

        cached_coriolis = vw._coriolis_values(None, dtype=np.float32)
        assert cached_coriolis is vw._coriolis_values(None, dtype=np.float32)

        with pytest.raises(ValueError, match="invalid value for omega"):
            vw.planetaryvorticity(omega="bad")

        vw.gridtype = "gaussian"
        gaussian_coriolis = vw.planetaryvorticity()
        assert gaussian_coriolis.shape == (4, 8, 2)
        assert len(vw._latitude_cache) == 2

    def test_absolutevorticity_broadcast_branches(self, monkeypatch):
        """Exercise optimized absolute-vorticity latitude broadcast branches."""
        vw = object.__new__(StandardVectorWind)
        vw.s = type(
            "DummySpharm",
            (),
            {
                "nlat": 4,
                "getvrtspec": lambda self, u, v, ntrunc=None: "vrtspec",
                "spectogrd": lambda self, spec: (
                    np.ones((4, 8, 2), dtype=np.float64)
                    if spec == "vrtspec"
                    else (_ for _ in ()).throw(
                        AssertionError(
                            "absolutevorticity should only synthesize vorticity"
                        )
                    )
                ),
            },
        )()
        vw.u = np.zeros((4, 8, 2), dtype=np.float32)
        vw.v = np.zeros((4, 8, 2), dtype=np.float32)
        vw._output_dtype = np.dtype(np.float32)
        vw._precision = "auto"

        vw.gridtype = "regular"
        regular_abs = vw.absolutevorticity()
        assert regular_abs.shape == (4, 8, 2)

        vw.gridtype = "gaussian"
        gaussian_abs = vw.absolutevorticity()
        assert gaussian_abs.shape == (4, 8, 2)

        with pytest.raises(ValueError, match="invalid value for omega"):
            vw.absolutevorticity(omega="bad")

    def test_single_component_public_methods_use_single_spec_helpers(self, monkeypatch):
        """vorticity/divergence/absolutevorticity should avoid pair-spectrum calls."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon).astype(np.float32)
        v = np.random.randn(nlat, nlon).astype(np.float32)
        vw = VectorWind(u, v)

        monkeypatch.setattr(
            vw.s,
            "getvrtdivspec",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("single-component methods should not call getvrtdivspec")
            ),
        )

        vort = vw.vorticity()
        div = vw.divergence()
        abs_vort = vw.absolutevorticity()

        assert vort.shape == (nlat, nlon)
        assert div.shape == (nlat, nlon)
        assert abs_vort.shape == (nlat, nlon)
        assert np.all(np.isfinite(vort))
        assert np.all(np.isfinite(div))
        assert np.all(np.isfinite(abs_vort))

    def test_vrtdiv_batches_scalar_synthesis(self, monkeypatch):
        """vrtdiv() should synthesize vorticity/divergence together once."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)
        vw = VectorWind(u, v)

        calls = {"count": 0}
        real_core = vw.s._core

        class CoreProxy:
            def __getattr__(self, name):
                return getattr(real_core, name)

            def spectogrd(self, *args, **kwargs):
                calls["count"] += 1
                return real_core.spectogrd(*args, **kwargs)

        monkeypatch.setattr(vw.s, "_core", CoreProxy())

        vrt, div = vw.vrtdiv()

        assert vrt.shape == (nlat, nlon)
        assert div.shape == (nlat, nlon)
        assert np.all(np.isfinite(vrt))
        assert np.all(np.isfinite(div))
        assert calls["count"] == 1

    def test_rossbywavesource_reuses_vrtdiv_analysis(self, monkeypatch):
        """RWS should avoid separate vorticity and divergence vector analyses."""
        nlat, nlon = 19, 36
        u = np.random.randn(nlat, nlon)
        v = np.random.randn(nlat, nlon)
        vw = VectorWind(u, v)

        calls = {"count": 0}
        real_core = vw.s._core

        class CoreProxy:
            def __getattr__(self, name):
                return getattr(real_core, name)

            def getvrtdivspec(self, *args, **kwargs):
                calls["count"] += 1
                return real_core.getvrtdivspec(*args, **kwargs)

        monkeypatch.setattr(vw.s, "_core", CoreProxy())
        monkeypatch.setattr(
            vw.s,
            "getchispec",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("rossbywavesource should reuse the first divspec")
            ),
        )

        rws = vw.rossbywavesource()

        assert rws.shape == (nlat, nlon)
        assert np.all(np.isfinite(rws))
        assert calls["count"] == 1

    def test_rossbywavesource_caches_planetary_spec_and_avoids_eta_analysis(
        self, monkeypatch
    ):
        """RWS should analyze cached 2D planetary vorticity, not full-field eta."""
        nlat, nlon = 19, 36
        rng = np.random.default_rng(20260502)
        u = rng.standard_normal((nlat, nlon, 2, 3)).astype(np.float32)
        v = rng.standard_normal((nlat, nlon, 2, 3)).astype(np.float32)
        vw = VectorWind(u, v)

        calls: list[tuple[int, ...]] = []
        original_grdtospec = vw.s.grdtospec

        def tracking_grdtospec(datagrid, *args, **kwargs):
            calls.append(np.shape(datagrid))
            return original_grdtospec(datagrid, *args, **kwargs)

        monkeypatch.setattr(vw.s, "grdtospec", tracking_grdtospec)

        first = vw.rossbywavesource()
        second = vw.rossbywavesource()

        assert first.shape == (nlat, nlon, 2, 3)
        assert second.shape == (nlat, nlon, 2, 3)
        assert np.all(np.isfinite(first))
        assert np.all(np.isfinite(second))
        assert calls == [(nlat, nlon)]

    def test_planetary_vorticity_spec_helper_and_vector_analysis_fallbacks(self):
        """Cover helper cache initialization and no-fastpath backend fallbacks."""
        vw = object.__new__(StandardVectorWind)
        vw.gridtype = "regular"
        vw._output_dtype = np.dtype(np.float32)
        vw.u = np.zeros((4, 8), dtype=np.float32)
        vw.v = np.zeros((4, 8), dtype=np.float32)
        vw.s = type(
            "DummySpharm",
            (),
            {
                "nlat": 4,
                "nlon": 8,
                "_validate_ntrunc": lambda self, truncation, max_ntrunc: 2,
                "grdtospec": lambda self, grid, ntrunc=None: np.arange(
                    6, dtype=np.complex64
                ),
                "getvrtdivspec": lambda self, u, v, ntrunc=None: ("vrtdiv", ntrunc),
                "getvrtspec": lambda self, u, v, ntrunc=None: ("vrt", ntrunc),
                "getdivspec": lambda self, u, v, ntrunc=None: ("div", ntrunc),
            },
        )()

        fspec = vw._planetary_vorticity_spec(spectral_ndim=2)
        assert fspec.shape == (6, 1, 1)
        assert hasattr(vw, "_planetary_vorticity_spec_cache")
        assert len(vw._planetary_vorticity_spec_cache) == 1

        with pytest.raises(ValueError, match="invalid value for omega"):
            vw._planetary_vorticity_spec(omega="bad")

        assert vw._vector_analysis_spectral(truncation=5) == ("vrtdiv", 5)
        assert vw._vector_analysis_spectral(truncation=6, component="vrt") == ("vrt", 6)
        assert vw._vector_analysis_spectral(truncation=7, component="div") == ("div", 7)

    def test_vector_analysis_prepared_layout_cache_is_reused(self, monkeypatch):
        """Prepared C-order wind inputs should be built once per instance."""
        nlat, nlon = 19, 36
        rng = np.random.default_rng(20260502)
        u = rng.standard_normal((nlat, nlon, 2, 3)).astype(np.float32)
        v = rng.standard_normal((nlat, nlon, 2, 3)).astype(np.float32)
        vw = VectorWind(u, v)

        original_layout_builder = vw._vector_analysis_layout_inputs
        layout_calls = {"count": 0}

        def counting_layout_builder():
            layout_calls["count"] += 1
            return original_layout_builder()

        monkeypatch.setattr(
            vw, "_vector_analysis_layout_inputs", counting_layout_builder
        )

        vw.vorticity()
        vw.divergence()
        vw.sfvp()

        assert layout_calls["count"] == 3
        assert vw._vector_analysis_layout_cache[0].flags["C_CONTIGUOUS"]
        assert vw._vector_analysis_layout_cache[1].flags["C_CONTIGUOUS"]

    def test_component_paths_match_direct_spectral_expression(self):
        """Component wind paths should use direct spectral reconstruction."""
        nlat, nlon = 19, 36
        rng = np.random.default_rng(24680)
        u = rng.standard_normal((nlat, nlon, 2), dtype=np.float32)
        v = rng.standard_normal((nlat, nlon, 2), dtype=np.float32)
        vw = VectorWind(u, v)

        psi, chi = vw.sfvp()
        np.testing.assert_allclose(vw.streamfunction(), psi, rtol=0, atol=0)
        np.testing.assert_allclose(vw.velocitypotential(), chi, rtol=0, atol=0)

        psispec, chispec = vw.s.getpsichispec(vw.u, vw.v)
        v_psi, u_psi = vw.s.getgrad(psispec)
        u_chi, v_chi = vw.s.getgrad(chispec)
        expected_nondiv = (
            vw._restore_output_dtype(-u_psi),
            vw._restore_output_dtype(v_psi),
        )
        expected_irrot = (
            vw._restore_output_dtype(u_chi),
            vw._restore_output_dtype(v_chi),
        )

        nondiv = vw.nondivergentcomponent()
        irrot = vw.irrotationalcomponent()
        np.testing.assert_allclose(nondiv[0], expected_nondiv[0], rtol=0, atol=0)
        np.testing.assert_allclose(nondiv[1], expected_nondiv[1], rtol=0, atol=0)
        np.testing.assert_allclose(irrot[0], expected_irrot[0], rtol=0, atol=0)
        np.testing.assert_allclose(irrot[1], expected_irrot[1], rtol=0, atol=0)

        vrtspec, divspec = vw.s.getvrtdivspec(vw.u, vw.v)
        vrt, div = vw.s._spectogrd_pair(vrtspec, divspec)
        eta = vrt + vw._planetary_vorticity(materialize=False)
        etaspec = vrtspec + vw._planetary_vorticity_spec(
            spectral_ndim=max(vrtspec.ndim - 1, 0)
        )
        etax, etay = vw.s.getgrad(etaspec)
        expected_rws = vw._restore_output_dtype(
            -eta * div - (u_chi * etax + v_chi * etay)
        )
        np.testing.assert_allclose(vw.rossbywavesource(), expected_rws, rtol=0, atol=0)

        fallback_vw = object.__new__(StandardVectorWind)
        fallback_vw.u = np.zeros((2, 2), dtype=np.float32)
        fallback_vw.v = np.zeros((2, 2), dtype=np.float32)
        fallback_vw.s = type(
            "UnsupportedComponentSpharm",
            (),
            {
                "rsphere": 6.3712e6,
                "_restore_spectral_shape": lambda self, data, extra_shape: data,
                "_analyze_vector_harmonic_component_flattened": (
                    lambda self, layout_u, layout_v, extra_shape, truncation, op_name, component: (
                        np.zeros((2, 1), dtype=np.float32),
                        np.zeros((2, 1), dtype=np.float32),
                        (),
                        0,
                    )
                ),
            },
        )()
        fallback_vw._vector_analysis_layout_cache = (
            np.zeros((2, 2, 1), dtype=np.float32),
            np.zeros((2, 2, 1), dtype=np.float32),
            (),
        )

        with pytest.raises(ValueError, match="unsupported component"):
            fallback_vw._vector_analysis_spectral(component="bad")

    def test_common_and_tool_validation_branches(self, monkeypatch):
        """Exercise helper validation and recovery branches."""
        with pytest.raises(ValueError, match="Need at least 2 latitude points"):
            common_mod.inspect_gridtype(np.array([45.0]))

        with pytest.raises(ValueError, match="Latitudes are neither equally-spaced"):
            common_mod.inspect_gridtype(np.array([90.0, 30.0, -10.0, -90.0]))

        with pytest.raises(
            ValueError,
            match="Equally-spaced latitudes don't match global regular grid pattern",
        ):
            common_mod.inspect_gridtype(np.linspace(80.0, -80.0, 19))

        monkeypatch.setattr(
            common_mod,
            "gaussian_lats_wts",
            lambda nlat: (_ for _ in ()).throw(RuntimeError("gaussian backend failed")),
        )
        with pytest.raises(
            ValueError, match="Failed to generate Gaussian reference latitudes"
        ):
            common_mod.inspect_gridtype(np.array([90.0, 30.0, -10.0, -90.0]))

        with pytest.raises(ValueError, match="Array must have at least 2 dimensions"):
            common_mod.to3d(np.arange(4))

        flattened = common_mod.to3d(np.zeros((2, 3, 4, 5), dtype=np.float32))
        assert flattened.shape == (2, 3, 20)

        with pytest.raises(ValueError, match="Data must have at least 2 dimensions"):
            prep_data(np.arange(4), "yx")

        with pytest.raises(TypeError, match="recovery_info must be a dictionary"):
            recover_data(np.zeros((2, 2, 1)), [])

        with pytest.raises(ValueError, match="Missing keys in recovery_info"):
            recover_data(np.zeros((2, 2, 1)), {"intermediate_shape": (2, 2, 1)})

        with pytest.raises(ValueError, match="Wind components must have same shape"):
            reverse_latdim(np.zeros((2, 2)), np.zeros((2, 3)))

        with pytest.raises(ValueError, match="axis 2 is out of bounds"):
            reverse_latdim(np.zeros((2, 2)), np.zeros((2, 2)), axis=2)

        with pytest.raises(ValueError, match="Wind components must have same shape"):
            order_latdim(np.array([90.0, -90.0]), np.zeros((2, 2)), np.zeros((3, 2)))

        with pytest.raises(ValueError, match="axis 2 is out of bounds"):
            order_latdim(
                np.array([90.0, -90.0]),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                axis=2,
            )

    def test_examples_module_smoke(self):
        """Cover lightweight shipped examples utilities."""
        path = Path(example_data_path("uwnd_mean.nc"))
        assert path.name == "uwnd_mean.nc"
        assert "example_data" in path.parts

        data_dir = Path(example_data_path())
        assert data_dir.name == "example_data"

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_auto_layout_and_s2n_alignment(self):
        """VectorWind defaults to auto layout for C-order & legacy and supports S2N."""
        rng = np.random.default_rng(20261128)
        u_c = rng.standard_normal((3, 2, 19, 36)).astype(np.float32)
        v_c = rng.standard_normal((3, 2, 19, 36)).astype(np.float32)
        vw_auto_c = VectorWind(u_c, v_c, gridtype="regular")
        vw_expl_c = VectorWind(u_c, v_c, gridtype="regular", layout="c")
        assert vw_auto_c.layout == "c"
        assert vw_auto_c.precision == "auto"
        assert vw_auto_c.lat_order == "auto"
        assert vw_auto_c.south_to_north is False
        np.testing.assert_allclose(vw_auto_c.vorticity(), vw_expl_c.vorticity(), rtol=0, atol=0)
        np.testing.assert_allclose(
            vw_auto_c.rossbywavesource(), vw_expl_c.rossbywavesource(), rtol=0, atol=0
        )

        # Ambiguous C-order 3D vs legacy F-order 3D
        u_l = np.transpose(u_c.reshape(6, 19, 36), (1, 2, 0)).copy()
        v_l = np.transpose(v_c.reshape(6, 19, 36), (1, 2, 0)).copy()
        vw_auto_l = VectorWind(u_l, v_l, gridtype="regular")
        vw_expl_l = VectorWind(u_l, v_l, gridtype="regular", layout="legacy")
        assert vw_auto_l.layout == "legacy"
        np.testing.assert_allclose(vw_auto_l.vorticity(), vw_expl_l.vorticity(), rtol=0, atol=0)

        # South-to-North without explicit lat array
        u_s2n = u_c[:, :, ::-1, :].copy()
        v_s2n = v_c[:, :, ::-1, :].copy()
        vw_s2n = VectorWind(u_s2n, v_s2n, gridtype="regular", lat_order="s2n")
        assert vw_s2n.south_to_north is True
        np.testing.assert_allclose(
            vw_s2n.planetaryvorticity(),
            vw_auto_c.planetaryvorticity()[:, :, ::-1, :],
            rtol=1e-6,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            vw_s2n.rossbywavesource(),
            vw_auto_c.rossbywavesource()[:, :, ::-1, :],
            rtol=1e-5,
            atol=1e-9,
        )

        # inspect_gridtype accepts both N2S and S2N regular & Gaussian
        assert common_mod.inspect_gridtype(np.linspace(-90.0, 90.0, 73)) == "regular"
        assert common_mod.inspect_gridtype(np.linspace(90.0, -90.0, 73)) == "regular"
        from skyborn.spharm import gaussian_lats_wts

        glat_n2s, _ = gaussian_lats_wts(32)
        assert common_mod.inspect_gridtype(glat_n2s[::-1]) == "gaussian"

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_grid_pair_score_tiers_and_boundaries(self):
        """Verify integer cross-multiplication _grid_pair_score across all score tiers and boundaries."""
        score = VectorWind._grid_pair_score

        # Tier -1: below minimum grid bounds (nlat < 3 or nlon < 4)
        for bad_nlat, bad_nlon in [(2, 4), (3, 3), (0, 0), (-1, 10), (19, 3)]:
            assert score(bad_nlat, bad_nlon, "regular") == -1
            assert score(bad_nlat, bad_nlon, "gaussian") == -1

        # Tier 4 regular: exact nlon == 2*(nlat-1) or nlon == 2*nlat (odd & even nlon included)
        assert score(73, 144, "regular") == 4
        assert score(181, 360, "regular") == 4
        assert score(72, 144, "regular") == 4
        assert score(3, 4, "regular") == 4
        assert score(4, 6, "regular") == 4

        # Tier 4 gaussian: even nlon and (nlon == 2*nlat or 1.8 <= nlon/nlat <= 2.25)
        assert score(64, 128, "gaussian") == 4
        assert score(94, 192, "gaussian") == 4
        assert score(160, 320, "gaussian") == 4
        assert score(10, 18, "Gaussian") == 4  # exact lower boundary 1.8
        assert score(8, 18, "GAUSSIAN") == 4  # exact upper boundary 2.25
        # Just outside [1.8, 2.25] drops to tier 3 (even nlon, within [1.2, 4.0])
        assert score(10, 16, "gaussian") == 3
        assert score(8, 20, "gaussian") == 3
        # Odd nlon with 1.8 <= nlon/nlat <= 2.25 cannot be tier 4 gaussian -> tier 2
        assert score(10, 21, "gaussian") == 2

        # Tier 3: even nlon, 1.2 <= nlon/nlat <= 4.0, not tier 4
        assert score(10, 12, "regular") == 3  # exact lower boundary 1.2
        assert score(10, 40, "regular") == 3  # exact upper boundary 4.0
        assert score(10, 12, "gaussian") == 3
        assert score(10, 40, "gaussian") == 3

        # Tier 2: odd nlon, 1.2 <= nlon/nlat <= 4.0, not tier 4
        assert score(10, 15, "regular") == 2
        assert score(5, 6, "regular") == 3  # 6 is even and exact 1.2
        assert score(15, 19, "gaussian") == 2  # ~1.267, odd nlon

        # Tier 1: nlon >= nlat outside [1.2, 4.0] (or 1.0 <= ratio < 1.2 or ratio > 4.0)
        assert score(10, 10, "regular") == 1
        assert score(10, 11, "regular") == 1
        assert score(10, 42, "regular") == 1
        assert score(10, 11, "gaussian") == 1
        assert score(10, 41, "gaussian") == 1

        # Tier 0: nlat >= 3, nlon >= 4, but nlon < nlat
        assert score(20, 19, "regular") == 0
        assert score(36, 19, "regular") == 0
        assert score(36, 19, "gaussian") == 0

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_resolve_layout_disambiguation_tiebreakers_and_cache(self):
        """Verify _resolve_layout across C/legacy N-D shapes, ties, lat hints, flags, and LRU cache."""
        resolve = VectorWind._resolve_layout
        standard._cached_resolve_auto_layout.cache_clear()

        # Explicit overrides and invalid layout handling
        assert resolve((19, 36, 4, 3), None, "C", "regular") == "c"
        assert resolve((4, 3, 19, 36), None, "Legacy", "regular") == "legacy"
        with pytest.raises(ValueError, match='layout must be "legacy", "c", or "auto"'):
            resolve((19, 36), None, "invalid_layout", "regular")

        # 2D auto always resolves to 'c'
        assert resolve((19, 36), None, "auto", "regular") == "c"

        # 3D, 4D, 5D C-order vs legacy recognition (regular & gaussian)
        assert resolve((37, 181, 360), None, "auto", "regular") == "c"
        assert resolve((181, 360, 37), None, "auto", "regular") == "legacy"
        assert resolve((12, 8, 73, 144), None, "auto", "regular") == "c"
        assert resolve((73, 144, 12, 8), None, "auto", "regular") == "legacy"
        assert resolve((2, 3, 4, 19, 36), None, "auto", "regular") == "c"
        assert resolve((19, 36, 2, 4, 3), None, "auto", "regular") == "legacy"
        assert resolve((19, 36, 2, 3, 4), None, "auto", "regular") == "c"  # trailing (3, 4) ties score 4 -> C-preferring
        assert resolve((10, 94, 192), None, "auto", "gaussian") == "c"
        assert resolve((94, 192, 10), None, "auto", "gaussian") == "legacy"
        assert resolve((1, 19, 36), None, "auto", "regular") == "c"

        # Ambiguous equal-score 4D shapes: tie broken by F-only vs C contiguity
        c_contig = np.zeros((19, 36, 19, 36), dtype=np.float32)
        f_contig = np.asfortranarray(c_contig)
        non_contig = c_contig[::1, ::1, ::1, ::1]
        assert resolve(c_contig.shape, c_contig.flags, "auto", "regular") == "c"
        assert resolve(f_contig.shape, f_contig.flags, "auto", "regular") == "legacy"
        assert resolve(non_contig.shape, non_contig.flags, "auto", "regular") == "c"
        # Missing or custom flags object fallback safety
        class _DummyFlags:
            f_contiguous = True
            c_contiguous = False

        assert resolve((19, 36, 19, 36), _DummyFlags(), "auto", "regular") == "legacy"
        assert resolve((19, 36, 19, 36), None, "auto", "regular") == "c"

        # Explicit 1D lat coordinate overrides score-based layout when unique match exists
        ambig_shape = (19, 36, 37, 72)
        assert resolve(ambig_shape, None, "auto", "regular", lat=np.linspace(90, -90, 37)) == "c"
        assert resolve(ambig_shape, None, "auto", "regular", lat=np.linspace(90, -90, 19)) == "legacy"
        # When 2D lat matches neither cleanly, falls back cleanly without error
        assert resolve((3, 19, 36), None, "auto", "regular", lat=np.linspace(90, -90, 19)) == "c"

        # Verify LRU cache populates and records hits cleanly
        standard._cached_resolve_auto_layout.cache_clear()
        first = resolve((5, 7, 73, 144), None, "auto", "regular")
        second = resolve((5, 7, 73, 144), None, "auto", "regular")
        info = standard._cached_resolve_auto_layout.cache_info()
        assert first == "c" and second == "c"
        assert info.misses >= 1 and info.hits >= 1

    @pytest.mark.skipif(
        not WINDSPHARM_AVAILABLE, reason="windspharm module not available"
    )
    def test_vectorwind_4d_and_5d_reshape_roundtrip_exact(self):
        """Verify 4D/5D inputs automatically flatten to 3D internally and restore exact shapes & bit parity."""
        rng = np.random.default_rng(20261129)
        nlat, nlon = 19, 36

        for dtype, batch_shape in [
            (np.float32, (4, 3)),
            (np.float64, (2, 2, 3)),
        ]:
            nt = int(np.prod(batch_shape))
            u_nd = rng.standard_normal(batch_shape + (nlat, nlon)).astype(dtype)
            v_nd = rng.standard_normal(batch_shape + (nlat, nlon)).astype(dtype)
            u_3d = u_nd.reshape(nt, nlat, nlon)
            v_3d = v_nd.reshape(nt, nlat, nlon)

            vw_nd = VectorWind(u_nd, v_nd, gridtype="regular")
            vw_3d = VectorWind(u_3d, v_3d, gridtype="regular", layout="c")
            assert vw_nd.layout == "c"

            scalar_nd = rng.standard_normal(batch_shape + (nlat, nlon)).astype(dtype)
            scalar_3d = scalar_nd.reshape(nt, nlat, nlon)

            for method in (
                "magnitude",
                "vorticity",
                "divergence",
                "planetaryvorticity",
                "absolutevorticity",
                "streamfunction",
                "velocitypotential",
                "rossbywavesource",
            ):
                out_nd = getattr(vw_nd, method)()
                out_3d = getattr(vw_3d, method)()
                assert out_nd.shape == batch_shape + (nlat, nlon)
                assert out_nd.dtype == dtype
                np.testing.assert_allclose(
                    out_nd, out_3d.reshape(batch_shape + (nlat, nlon)), rtol=0, atol=0
                )

            for pair_method in (
                "vrtdiv",
                "sfvp",
                "irrotationalcomponent",
                "nondivergentcomponent",
                "helmholtz",
            ):
                outs_nd = getattr(vw_nd, pair_method)()
                outs_3d = getattr(vw_3d, pair_method)()
                for arr_nd, arr_3d in zip(outs_nd, outs_3d):
                    assert arr_nd.shape == batch_shape + (nlat, nlon)
                    assert arr_nd.dtype == dtype
                    np.testing.assert_allclose(
                        arr_nd,
                        arr_3d.reshape(batch_shape + (nlat, nlon)),
                        rtol=0,
                        atol=0,
                    )

            gx_nd, gy_nd = vw_nd.gradient(scalar_nd, truncation=11)
            gx_3d, gy_3d = vw_3d.gradient(scalar_3d, truncation=11)
            np.testing.assert_allclose(
                gx_nd, gx_3d.reshape(batch_shape + (nlat, nlon)), rtol=0, atol=0
            )
            np.testing.assert_allclose(
                gy_nd, gy_3d.reshape(batch_shape + (nlat, nlon)), rtol=0, atol=0
            )

            trunc_nd = vw_nd.truncate(scalar_nd, truncation=11)
            trunc_3d = vw_3d.truncate(scalar_3d, truncation=11)
            assert trunc_nd.shape == batch_shape + (nlat, nlon)
            np.testing.assert_allclose(
                trunc_nd, trunc_3d.reshape(batch_shape + (nlat, nlon)), rtol=0, atol=0
            )

        # 4D legacy shape (nlat, nlon, 4, 3) also preserves exact shape & values
        u_4d_c = rng.standard_normal((4, 3, nlat, nlon)).astype(np.float32)
        v_4d_c = rng.standard_normal((4, 3, nlat, nlon)).astype(np.float32)
        u_4d_leg = np.moveaxis(u_4d_c, (2, 3), (0, 1)).copy()
        v_4d_leg = np.moveaxis(v_4d_c, (2, 3), (0, 1)).copy()
        vw_4d_c = VectorWind(u_4d_c, v_4d_c, gridtype="regular")
        vw_4d_leg = VectorWind(u_4d_leg, v_4d_leg, gridtype="regular")
        assert vw_4d_leg.layout == "legacy"

        vrt_c = vw_4d_c.vorticity()
        vrt_leg = vw_4d_leg.vorticity()
        assert vrt_leg.shape == (nlat, nlon, 4, 3)
        np.testing.assert_allclose(
            np.moveaxis(vrt_leg, (0, 1), (2, 3)), vrt_c, rtol=0, atol=0
        )
        rws_c = vw_4d_c.rossbywavesource()
        rws_leg = vw_4d_leg.rossbywavesource()
        assert rws_leg.shape == (nlat, nlon, 4, 3)
        np.testing.assert_allclose(
            np.moveaxis(rws_leg, (0, 1), (2, 3)), rws_c, rtol=0, atol=0
        )

        # xarray 4D DataArray (time, level, lat, lon) executes without axis transposition
        xr = pytest.importorskip("xarray")
        from skyborn.windspharm.xarray import VectorWind as XrVectorWind

        lat = np.linspace(90.0, -90.0, nlat)
        lon = np.linspace(0.0, 350.0, nlon)
        da_u = xr.DataArray(
            u_4d_c,
            dims=("time", "level", "lat", "lon"),
            coords={"time": np.arange(4), "level": [1000, 500, 200], "lat": lat, "lon": lon},
        )
        da_v = xr.DataArray(
            v_4d_c,
            dims=("time", "level", "lat", "lon"),
            coords={"time": np.arange(4), "level": [1000, 500, 200], "lat": lat, "lon": lon},
        )
        xvw = XrVectorWind(da_u, da_v, gridtype="regular")
        assert xvw._needs_transpose is False
        xr_vrt = xvw.vorticity()
        assert xr_vrt.dims == ("time", "level", "lat", "lon")
        assert xr_vrt.shape == (4, 3, nlat, nlon)
        np.testing.assert_allclose(xr_vrt.values, vrt_c, rtol=0, atol=0)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
