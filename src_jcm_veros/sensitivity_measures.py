# Self-contained "measure" definitions for jax.jvp sensitivity experiments.
#
# A `Measure` bundles together:
#   - `compute(coupled_carry) -> jnp.ndarray`: a JAX-traceable diagnostic,
#     used inside `forecast`/`jax.jvp` in `main_sensitivity_*.py`.
#   - `dims`/`get_coords`: the named dimensions and coordinate arrays of
#     that diagnostic, for writing it to netCDF.
#   - `save`/`plot`: how to write the jax.jvp + ensemble sensitivity results
#     to netCDF and render the comparison figure.
#
# `main_sensitivity_*.py` picks one `Measure` from `MEASURES` and runs
# `forecast`/`jax.jvp`/the ensemble loop generically against
# `measure.compute`, then calls `measure.save(...)`/`measure.plot(...)`.
# Adding a new diagnostic only requires adding a new `Measure` here --
# the simulation/save/plot driver code does not change.

from dataclasses import dataclass, field
from typing import Callable, Sequence, Dict, Any

import numpy as np
import jax.numpy as jnp
import xarray as xr
import matplotlib.pyplot as plt

from veros_helper import _OCEAN_GHOST_CELL


def symmetric_levels(*arrays, n=11, n_std=2.0):
    """Build contour levels that are symmetric about zero, spanning
    +/- `n_std` standard deviations of the combined data."""
    values = jnp.concatenate([jnp.asarray(a).ravel() for a in arrays])
    vmax = n_std * float(jnp.std(values))
    if vmax == 0:
        vmax = 1.0
    return jnp.linspace(-vmax, vmax, n)


def masked_zonal_mean(field, mask, axis=0):
    """Mask-weighted mean of `field` along `axis`, skipping points where
    `mask == 0` (e.g. land points in Veros' `maskT`)."""
    mask = jnp.asarray(mask)
    weighted = jnp.sum(field * mask, axis=axis)
    count = jnp.sum(mask, axis=axis)
    return weighted / jnp.maximum(count, 1.0)



@dataclass
class Measure:
    """A single diagnostic extracted from `coupled_carry`, together with
    everything needed to save and plot its sensitivity to an SST
    perturbation."""

    variable_specs: Dict[str, Dict[str, Any]] # varname => variable information
    compute: Callable  # coupled_carry -> jnp.ndarray

    def save(
        self,
        data_file,
        *,
        tangent_temp_initial,
        tangent_measure,
        ensemble,
    ):
        """Save the jax.jvp tangent and ensemble-mean sensitivity estimates
        for this measure to a netCDF file, so that plotting can be done
        independently from the (expensive) simulation."""

        data_vars = {
            "tangent_temp_initial": (("lon", "lat", "depth"), np.asarray(tangent_temp_initial))
        }

        for i, (varname, varspec) in enumerate(self.variable_specs.items()):
            
            data_vars[f"tangent_{varname}"] = (tuple(varspec["dims"]), np.asarray(tangent_measure[i]))
            
            number_of_ensemble_members = len(ensemble)
            if number_of_ensemble_members > 0:
                
                # The i-th variable is stored in the i-th element of each ensemble member
                _ensemble_data = np.stack([np.asarray(ensemble[n][i]) for n in range(len(ensemble))])
                data_vars[f"ensemble_tangent_{varname}"] = (
                    ("ensemble",) + tuple(varspec["dims"]), _ensemble_data,
                )



        ds = xr.Dataset(data_vars=data_vars)

        print(f"Saving simulation output into: {data_file}")
        ds.to_netcdf(data_file)

def compose_measures(*measures):
    """Combine several `Measure`s into one, so a single jax.jvp/ensemble
    run can produce all of their diagnostics from one shared trajectory
    instead of rerunning the (expensive) simulation once per measure.

    Each sub-measure's `compute` is called on the same
    `(coupled_carry, predictions)`; the resulting tuples are concatenated
    in `measures` order, and `variable_specs` are merged the same way, so
    `Measure.save` writes every variable out unmodified.
    """
    variable_specs = {}
    for m in measures:
        variable_specs.update(m.variable_specs)

    def compute(coupled_carry, predictions):
        results = ()
        for m in measures:
            results += m.compute(coupled_carry, predictions)
        return results

    return Measure(variable_specs=variable_specs, compute=compute)


# ---------------------------------------------------------------------------
# Concrete measures
# ---------------------------------------------------------------------------

def _lonlat_coords(model, ocn_model):
    atm_model = model.components["atm"].raw_component
    lon = np.asarray(atm_model.coords.horizontal.longitudes) * 180 / np.pi
    lat = np.asarray(atm_model.coords.horizontal.latitudes) * 180 / np.pi
    return dict(lon=lon, lat=lat)


def _latdepth_coords(model, ocn_model):
    atm_model = model.components["atm"].raw_component
    lat = np.asarray(atm_model.coords.horizontal.latitudes) * 180 / np.pi
    depth = np.asarray(ocn_model.state.variables.zt)
    return dict(lat=lat, depth=depth)


def _ocean_temperature_zonal_mean(coupled_carry, predictions):
    g = _OCEAN_GHOST_CELL
    vs = coupled_carry["ocn"]["state"].variables
    temp = vs.temp[g:-g, g:-g, :, vs.tau]
    mask = vs.maskT[g:-g, g:-g, :]
    # Returned as a 1-tuple so the output lines up positionally with
    # `variable_specs` (and with `ensemble[n][i]` in `save`), even though
    # this measure currently has only one variable.
    return (masked_zonal_mean(temp, mask, axis=0),)

def _sea_surface_temperature(coupled_carry, predictions):
    predictionsg = _OCEAN_GHOST_CELL
    vs = coupled_carry["ocn"]["state"].variables
    temp = vs.temp[g:-g, g:-g, :, vs.tau]
    mask = vs.maskT[g:-g, g:-g, :]
    # Returned as a 1-tuple so the output lines up positionally with
    # `variable_specs` (and with `ensemble[n][i]` in `save`), even though
    # this measure currently has only one variable.
    return (masked_zonal_mean(temp, mask, axis=0),)

def _ocean_temperature(coupled_carry, predictions):
    g = _OCEAN_GHOST_CELL
    vs = coupled_carry["ocn"]["state"].variables
    temp = vs.temp[g:-g, g:-g, :, vs.tau]
    return (temp,)


OCEAN_TEMPERATURE_ZONAL_MEAN = Measure(
    variable_specs=dict(ocean_temperature_zonal_mean = dict(dims=["latitude", "depth"])),
    compute=_ocean_temperature_zonal_mean,
)

OCEAN_TEMPERATURE = Measure(
    variable_specs=dict(ocean_temperature = dict(dims=["longitude", "latitude", "depth"])),
    compute=_ocean_temperature,
)


def _ocean_northward_heat_transport(coupled_carry, predictions):
    """Northward ocean heat transport [W] as a function of latitude,
    integrated zonally and over depth.

    Mirrors the meridional-transport pattern used by Veros' own
    `overturning` diagnostic (`dxt * cosu * v * dzt`, masked by `maskV`;
    see `veros.diagnostics.overturning.diagnose_kernel`), but multiplies by
    temperature -- averaged from consecutive T-points onto the intervening
    V-point the same way that diagnostic interpolates density onto V-faces
    -- and by `rho_0 * cp_0` to get a heat flux instead of a volume
    transport.
    """
    g = _OCEAN_GHOST_CELL
    ocn_state = coupled_carry["ocn"]["state"]
    vs = ocn_state.variables
    rho_0 = ocn_state.settings.rho_0
    cp_0 = 3991.86795711963  # J / (kg K); Veros hardcodes this per-setup, there is no settings.cp_0

    # Average temperature at consecutive T-points (j, j+1) onto the
    # intervening V-point, matching the latitude range of `v`/`maskV` ([g:-g]).
    temp_face = 0.5 * (
        vs.temp[g:-g, g:-g, :, vs.tau] + vs.temp[g:-g, g + 1:-g + 1, :, vs.tau]
    )
    v = vs.v[g:-g, g:-g, :, vs.tau]
    mask = vs.maskV[g:-g, g:-g, :]

    fac = vs.dxt[g:-g, None, None] * vs.cosu[None, g:-g, None] * vs.dzt[None, None, :]
    heat_flux = rho_0 * cp_0 * v * temp_face * mask * fac

    # Sum over longitude (axis 0) and depth (axis 2), leaving a function
    # of latitude (the ocean's V-grid) only. Returned as a 1-tuple so the
    # output lines up positionally with `variable_specs`.
    return (jnp.sum(heat_flux, axis=(0, 2)),)


OCEAN_NORTHWARD_HEAT_TRANSPORT = Measure(
    variable_specs=dict(ocean_northward_heat_transport=dict(dims=["latitude"])),
    compute=_ocean_northward_heat_transport,
)


MEASURES = {
    "ocean_temperature": OCEAN_TEMPERATURE,
    "ocean_temperature_zonal_mean": OCEAN_TEMPERATURE_ZONAL_MEAN,
    "ocean_northward_heat_transport": OCEAN_NORTHWARD_HEAT_TRANSPORT,
}
