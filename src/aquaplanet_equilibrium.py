# # Use gradient descend to find equilibrium SST on an aquaplanet

import argparse
import runpy
import sys
from pathlib import Path
import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp
import xarray as xr
from tqdm import tqdm

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
print(f"Note: jcm.__file__ = {str(jcm.__file__)}")
import jax_datetime as jdt

from jem.components import JCM, SlabOceanModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools
from jem.utils.bulk_op import stack_objects
from optimizers import HamitonianMethod, RMSProp, RMSPropMomentum, LBFGS
jax.config.update("jax_compilation_cache_dir", "./.jax_cache")
print("Devices: ", jax.devices())

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def _load_config(config_path: str):
    return runpy.run_path(config_path)["cfg"]

_parser = argparse.ArgumentParser()
_parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
cfg = _load_config(_parser.parse_known_args()[0].config)

spinup_trajectory_interval = jdt.to_timedelta(cfg.spinup_interval_days, "day")
training_trajectory_interval = jdt.to_timedelta(cfg.atmosphere_memory_days + cfg.average_days, "day")
long_direct_simulation_time = jdt.to_timedelta(360 * cfg.spinup_total_years, "day")
long_direct_simulation_iterations = int(long_direct_simulation_time / spinup_trajectory_interval)
initial_condition_time = jdt.to_timedelta(360 * cfg.initial_condition_year, "day")

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
one_second = jdt.to_timedelta(1, "second")

for d in [cfg.output_dir_spinup, cfg.output_dir_training]:
    d.mkdir(exist_ok=True, parents=True)

def get_spinup_checkpoint_file(iteration):
    return (
        cfg.output_dir_spinup /
        f"adjusted_carray_spinup-iterations-{iteration:03d}"
        f"_eachinterval-{cfg.spinup_interval_days:d}"
        f"-days.chkpt"
    ).resolve()

def get_training_checkpoint_file(iteration):
    return (
        cfg.output_dir_training /
        f"iterations-{iteration:03d}.chkpt"
    ).resolve()

target_spinup_checkpoint_file = get_spinup_checkpoint_file(
    int(initial_condition_time / spinup_trajectory_interval) - 1
)

mapper = BasicMapper()
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("ocn", "forcing.total_heat_flux"),
    regridder = lambda x: x,  # identity is default
)
mapper.add_mapping(
    source = ("ocn", "state.sea_surface_temperature"),
    target = ("atm", "forcing.sea_surface_temperature"),
)

# Construct Model
atm_model = jcm.model.Model(
    coords=get_speedy_coords(spectral_truncation=cfg.spectral_truncation),
    start_date=start_datetime,
    time_step=30.0,
)

atm_model = JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)

model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=SlabOceanModel(
            grid_specification=f"JCM::T{cfg.spectral_truncation:d}",
            start_datetime=start_datetime,
            timestep=coupling_timestep / one_second,
        ),
    ),
    mappers=dict(mapper=mapper),
)

tree_tools.print_tree(model.get_info(), root="Model")
initial_carry = model.initialize()

shared_setting = dict(
    workflow=["mapper", "atm", "ocn"],
    tqdm_kwargs = dict(
        position=1
    ),
)


training_trajectory_function = model.generate_trajectory_function(
    iterations = int(training_trajectory_interval / coupling_timestep),
    **shared_setting,
)


# Spin-up model
# I want atmosphere to be turbulent to start with. So I need to spin-up the model for about 30 days.

print(f"Check the target spin-up checkpoint file {str(target_spinup_checkpoint_file)}")
if target_spinup_checkpoint_file.exists():
    print(f"Target spin-up checkpoint file exists.")
else:
    print(f"Target spin-up checkpoint file does not exist. Need to spin-up now.")
    adjusted_carry = initial_carry
    spinup_trajectory_function = model.generate_trajectory_function(
        iterations = int(spinup_trajectory_interval / coupling_timestep),
        **shared_setting,
    )

    for current_iteration in range(long_direct_simulation_iterations):
        print(f"Long direct simulation iteration {current_iteration:d}/{long_direct_simulation_iterations:d}...")
        adjusted_carry, spinup_predictions = spinup_trajectory_function(adjusted_carry)
        spinup_checkpoint_file = get_spinup_checkpoint_file(current_iteration)
        print(f"Save checkpoint file: {str(spinup_checkpoint_file):s}")
        ocp.save_pytree(spinup_checkpoint_file, adjusted_carry, overwrite=True)
        output_dict = model.predictions_to_xarray(spinup_predictions)
        for component_name, ds in output_dict.items():
            output_file = cfg.output_dir_spinup / f"{component_name:s}-{current_iteration:03d}.nc"
            print("Output file: ", str(output_file))
            ds.to_netcdf(output_file, engine="netcdf4")
            ds.close()
    del adjusted_carry


# Define the Loss Function
    
print(f"Load spin-up file {str(target_spinup_checkpoint_file)}")
adjusted_carry = ocp.load_pytree(target_spinup_checkpoint_file, initial_carry)
initial_sst = jnp.mean(adjusted_carry["ocn"]["state"].sea_surface_temperature, axis=0)
nlon = adjusted_carry["ocn"]["state"].sea_surface_temperature.shape[0]

def loss_function(sst):
    adjusted_carry["ocn"]["state"].sea_surface_temperature = jnp.repeat(sst[None, :], nlon, axis=0)
    _, predictions = training_trajectory_function(adjusted_carry)
    return jnp.mean(jnp.mean(predictions["ocn"]["forcing"]["total_heat_flux"][-cfg.average_days:, :, :], axis=0)**2)


def generic_output_callback(history, i, method: str, loop_idx: int, stage_idx: int):
    output_file = cfg.output_dir_training / f"training_result-loop_{loop_idx:02d}-stage_{stage_idx:02d}_{method}-{i:05d}.nc"

    data_vars = dict(
        loss = (("iteration",), history["loss"]),
        sst = (("iteration", "lat"), history["x"]),
        dloss_dsst = (("iteration", "lat"), history["dloss_dx"]),
    )

    if method == "HamitonianMethod":
        data_vars.update(dict(
            sst_momentum = (("iteration", "lat"), history["p"]),
            sst_kinetic_energy = (("iteration",), history["K"]),
        ))
    elif method == "RMSProp":
        data_vars.update(dict(
            square_dloss_dsst = (("iteration", "lat"), history["square_dloss_dx"]),
        ))
    elif method == "RMSPropMomentum":
        data_vars.update(dict(
            square_dloss_dsst = (("iteration", "lat"), history["square_dloss_dx"]),
            sst_kinetic_energy = (("iteration",), history["K"]),
        ))

    ds_result = xr.Dataset(
        data_vars = data_vars,
        coords = dict(
        )
    )

    print(f"Save training results to : {str(output_file)}")
    ds_result.to_netcdf(output_file, unlimited_dims="iteration")

    return jnp.all(jnp.isfinite(history["x"]))

_optimizer_classes = {
    "HamitonianMethod": HamitonianMethod,
    "RMSProp": RMSProp,
    "RMSPropMomentum": RMSPropMomentum,
    "LBFGS": LBFGS,
}

for stage in cfg.stages:
    if stage.method not in _optimizer_classes:
        raise ValueError(f"Unknown optimization method: '{stage.method}'")

# Build optimizer instances once so JIT-compiled step functions are reused across loops
_optimizer_instances = [
    _optimizer_classes[stage.method](loss_function, **stage.optimizer_kwargs)
    for stage in cfg.stages
]

print("Running Optimization...")
start_time = time.perf_counter()
current_x = initial_sst
for loop_idx in range(cfg.stage_loops):
    for stage_idx, (stage, optimizer) in enumerate(zip(cfg.stages, _optimizer_instances)):
        print(f"Loop {loop_idx}, stage {stage_idx}: {stage.method} for {stage.iterations} iterations")
        stage_callback = functools.partial(
            generic_output_callback,
            method=stage.method,
            loop_idx=loop_idx,
            stage_idx=stage_idx,
        )
        _final_carry, _ = optimizer(
            initial_x=current_x,
            iterations=stage.iterations,
            callback=stage_callback,
            callback_interval=stage.callback_interval,
        )
        current_x = _final_carry["x"]

end_time = time.perf_counter()
print(f"Cost of time: {end_time - start_time} seconds.")
