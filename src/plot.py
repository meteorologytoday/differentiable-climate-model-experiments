from pathlib import Path

import xarray as xr
import numpy as np

truth_year = 20
exp_root = Path("/p/projects/poem/tienyiao/projects/differentiable_model/differentiable_experiments/experiment_set")
exp_name = "output_T31_02-04_aquaplanet_equilibrium_with_1year_spinup_sst_30days_avg"
algo_name = "RMSPropMomentum"
ds = xr.open_mfdataset(str(exp_root / exp_name / f"training_{algo_name:s}/training_result-*.nc"), engine="netcdf4", combine="nested", concat_dim="iteration").isel(iteration=slice(None, -10))
ds_truth = xr.open_mfdataset(str(exp_root / exp_name / "spinup/ocn-*.nc"), engine="netcdf4").isel(time=slice(360*truth_year-30, 360*truth_year)).mean(dim=["longitude", "time"])

nlat = ds.dims["lat"]
niteration = ds.dims["iteration"]
lat = np.linspace(-90, 90, nlat+1)
lat = (lat[1:] + lat[:-1])/2


print(ds)

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import cmocean as cmo



# Figure: Loss function
fig, ax = plt.subplots(1, 1, squeeze=False)

ax_flatten = ax.flatten()

ax_idx = 0

_ax = ax_flatten[ax_idx]; ax_idx += 1
_ax.plot(np.arange(niteration), ds["loss"].to_numpy(), color="black")

trans = blended_transform_factory(_ax.transAxes, _ax.transData)
_ax.plot([0, 1], [0, 0], transform=trans, color="gray", linestyle="dashed")
_ax.set_title("Loss function")
#_ax.set_ylim([None, 20000])
_ax.set_ylabel("[$\\left( \\mathrm{W}/m^2 \\right)^2$]")

for _ax in ax_flatten:
    _ax.set_xlabel("[# of iterations]")
    _ax.grid()

fig.savefig(f"Spinup-loss-function_{algo_name:s}.png", dpi=200)
fig.savefig(f"Spinup-loss-function_{algo_name:s}.svg")


# Figure: SST function

num_lines = 5
# Create a colormap (e.g., 'viridis', 'plasma', 'coolwarm')
cmap = plt.colormaps['plasma_r']
# Generate colors evenly spaced along the colormap
colors = [cmap(i) for i in np.linspace(0, 1, num_lines+1)][1:]
sampling_indices = np.floor( np.linspace(0, 1, num_lines) * (ds.dims["iteration"]-1) ).astype(int)


fig, ax = plt.subplots(1, 1, squeeze=False)

ax_flatten = ax.flatten()

ax_idx = 0

_ax = ax_flatten[ax_idx]; ax_idx += 1
for i, sampling_index in enumerate(sampling_indices):
    _ax.plot(lat, ds["sst"].to_numpy()[sampling_index, :], color=colors[i], zorder=i, label=f"iter-{sampling_index}")

_ax.plot(lat, ds_truth["sea_surface_temperature"].to_numpy(), color="red", linewidth=3, linestyle="dashed", label=f"{truth_year:d}-year spinup")
_ax.legend()
_ax.set_title("SST evolution during training (orange = target)")
#_ax.set_ylim([None, 20000])
_ax.set_ylabel("[K]")
_ax.grid()

for _ax in ax_flatten:
    _ax.set_xlabel("latitudinal direction")

fig.suptitle(f"Algo: {algo_name:s}")

fig.savefig(f"SST-evolution_{algo_name:s}.png", dpi=200)
fig.savefig(f"SST-evolution_{algo_name:s}.svg")

plt.show()

