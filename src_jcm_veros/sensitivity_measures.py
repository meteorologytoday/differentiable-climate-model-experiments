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

from model_setup import _OCEAN_GHOST_CELL


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

OCEAN_TEMPERATURE_ZONAL_MEAN = Measure(
    variable_specs=dict(ocean_temperature_zonal_mean = dict(dims=["latitude", "depth"])),
    compute=_ocean_temperature_zonal_mean,
)


MEASURES = {
    "ocean_temperature_zonal_mean": OCEAN_TEMPERATURE_ZONAL_MEAN,
}
