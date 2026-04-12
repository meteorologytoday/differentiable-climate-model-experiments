from pathlib import Path

import xarray as xr
import numpy as np

ds = xr.open_mfdataset("/p/projects/poem/tienyiao/projects/differentiable_model/differentiable_experiments/experiment_set/output_T31_02-04_aquaplanet_equilibrium_with_fully_spinup_sst/spinup/ocn-*.nc", engine="netcdf4")

print(ds)

loss = (ds["total_heat_flux"].mean(dim="longitude").rolling(time=30,center=True).mean()**2.0).mean(dim="latitude").to_numpy()
sst = ds["sea_surface_temperature"].mean(dim="longitude")
lat = ds.coords["latitude2D"].isel(longitude=0).to_numpy()

print(f"Length of data: {len(loss)}")

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import cmocean as cmo

# Figure: Spin-up loss function
fig, ax = plt.subplots(1, 1, squeeze=False)

ax_flatten = ax.flatten()

ax_idx = 0

_ax = ax_flatten[ax_idx]; ax_idx += 1
_ax.plot(loss, color="black")

trans = blended_transform_factory(_ax.transAxes, _ax.transData)
_ax.plot([0, 1], [0, 0], transform=trans, color="gray", linestyle="dashed")
_ax.set_title("Spinup loss function")
#_ax.set_ylim([None, ])
_ax.set_ylabel("[$\\left( \\mathrm{W}/m^2 \\right)^2$]")

for _ax in ax_flatten:
    _ax.grid()
    _ax.set_xlabel("[day]")

fig.savefig("spin-up-loss.png", dpi=200)

# Figure: Spin-up sst

fig, ax = plt.subplots(1, 1, squeeze=False)

ax_flatten = ax.flatten()
ax_idx = 0

_ax = ax_flatten[ax_idx]; ax_idx += 1

sampling_slices = [
    ( f"{y:d} year",  slice( y*360-30,  y*360) )
    for y in [1, 5, 10, 15, 20]
]

num_lines = len(sampling_slices)
cmap = plt.colormaps["plasma_r"]
# Generate colors evenly spaced along the colormap
colors = [cmap(i) for i in np.linspace(0, 1, num_lines+1)[1:]]


for i, (legend_text, sampling_slice) in enumerate(sampling_slices):
    _ax.plot(lat, sst.isel(time=sampling_slice).mean(dim="time").to_numpy(), color=colors[i], label=legend_text)

_ax.legend()
_ax.grid()
_ax.set_title("Spinup SST")
_ax.set_ylabel("[K]")

for _ax in ax_flatten:
    _ax.set_xlabel("[degree N]")

fig.savefig("spin-up-sst.png", dpi=200)
plt.show()

