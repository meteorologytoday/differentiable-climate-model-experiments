from pathlib import Path

import xarray as xr
import numpy as np

truth_year = 20
exp_root = Path("/p/projects/poem/tienyiao/projects/differentiable_model/differentiable_experiments/experiment_set")
exp_name = "output_T31_02-04_aquaplanet_equilibrium_with_1year_spinup_sst_30days_avg"

algo_names = ["RMSPropMomentum", "RMSProp"]

ds_truth = xr.open_mfdataset(str(exp_root / exp_name / "spinup/ocn-*.nc"), engine="netcdf4").isel(time=slice(360*truth_year-30, 360*truth_year)).mean(dim=["longitude", "time"])

data = {
    algo_name : xr.open_mfdataset(str(exp_root / exp_name / f"training_{algo_name:s}/training_result-*.nc"), engine="netcdf4", combine="nested", concat_dim="iteration").isel(iteration=slice(None, -10))
    for algo_name in algo_names
}

ref_ds = data[list(data.keys())[0]]
nlat = ref_ds.dims["lat"]
lat = np.linspace(-90, 90, nlat+1)
lat = (lat[1:] + lat[:-1])/2

niteration = ref_ds.dims["iteration"]

import matplotlib as mplt
mplt.use("Agg")

import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
from matplotlib.transforms import blended_transform_factory
import cmocean as cmo


# Figure: Loss function
fig, ax = plt.subplots(1, 1, squeeze=False)

ax_flatten = ax.flatten()

ax_idx = 0

_ax = ax_flatten[ax_idx]; ax_idx += 1
for i, (algo_name, ds) in enumerate(data.items()):
    niteration = ds.dims["iteration"]
    #color = [
    #    "dodgerblue",
    #    "",
    #][i]
    _ax.plot(np.arange(niteration), ds["loss"].to_numpy(), label=algo_name)

trans = blended_transform_factory(_ax.transAxes, _ax.transData)
_ax.plot([0, 1], [0, 0], transform=trans, color="gray", linestyle="dashed")
_ax.set_title("Loss function")
_ax.set_ylabel("[$\\left( \\mathrm{W}/m^2 \\right)^2$]")
_ax.legend()

for _ax in ax_flatten:
    _ax.set_xlabel("[# of iterations]")
    _ax.grid()

fig.savefig(f"poster-loss-function_multiple_algos.svg")
fig.savefig(f"poster-loss-function_multiple_algos.png", dpi=200)


