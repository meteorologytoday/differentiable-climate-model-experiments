# # Use gradient descend to find equilibrium SST on a fully coupled planet

import argparse
import runpy
import sys
from pathlib import Path
import time
import functools

import jax
jax.config.update("jax_enable_x64", True)

print("jax.config of jax_enable_x64: ", jax.config.read("jax_enable_x64"))
import jax.numpy as jnp
import numpy as np
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
from jem.utils.checkpoints import (
    save_coupled_carry, load_coupled_carry,
    save_veros_carry, load_veros_carry,
)

from training_tools.optimizers import HamitonianMethod, RMSProp, RMSPropMomentum, LBFGS, pack
from training_tools.model_context import ModelContext

from coupled_jcm_veros_model_setup import build_model
from jcm_helper import freeze_solar_at

jax.config.update("jax_compilation_cache_dir", "./.jax_cache")
print("Devices: ", jax.devices())

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def _load_config(config_path: str):
    return runpy.run_path(config_path)["config"]

if __name__ == "__main__":
    
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    _parser.add_argument("--truncation-number", type=int, help="Truncation number", default=31)
    _parser.add_argument("--jcm-timestep-min", type=int, help="JCM timestep in minutes", default=30)
    _parser.add_argument("--veros-timestep-min", type=int, help="Veros timestep in minutes", default=60)
    _parser.add_argument("--terrain-planet-type", type=str, help="Simulation name for output", required=True)

    args = _parser.parse_known_args()[0]
    config = _load_config(args.config)

    spinup_trajectory_interval = jdt.to_timedelta(config.spinup_interval_days, "day")
    training_trajectory_interval = jdt.to_timedelta(config.training_trajectory_days, "day")
    long_direct_simulation_time = jdt.to_timedelta(365 * config.spinup_total_years, "day")
    long_direct_simulation_iterations = int(long_direct_simulation_time / spinup_trajectory_interval)
    initial_condition_time = jdt.to_timedelta(365 * config.initial_condition_year, "day")

    calendar = "365_day"
    truncation_number = args.truncation_number
    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    one_second = jdt.to_timedelta(1, "second")

    for d in [config.output_dir_spinup, config.output_dir_training]:
        d.mkdir(exist_ok=True, parents=True)

    print("config.output_dir_spinup = ", config.output_dir_spinup)

    def get_spinup_checkpoint_file(iteration):
        return (
            config.output_dir_spinup /
            f"adjusted_carray_spinup-iterations-{iteration:03d}"
            f"_eachinterval-{config.spinup_interval_days:d}"
            f"-days.chkpt"
        ).resolve()

    def get_training_checkpoint_file(iteration):
        return (
            config.output_dir_training /
            f"iterations-{iteration:03d}.chkpt"
        ).resolve()

    target_spinup_checkpoint_file = get_spinup_checkpoint_file(
        int(initial_condition_time / spinup_trajectory_interval) - 1
    )

    # Fix solar geometry at the vernal equinox so NH and SH receive equal insolation.
    freeze_solar_at("2000-03-20", calendar=calendar)

    # Build the coupled JCM + Veros + SlabOceanModel system. Packaged as a
    # function in `model_setup.py` so that other scripts (e.g. a jax.grad
    # sensitivity experiment) can build exactly the same model.
    model, model_config = build_model(
        truncation_number=truncation_number,
        start_datetime=start_datetime,
        coupling_timestep=coupling_timestep,
        calendar=calendar,
        terrain_planet_type=args.terrain_planet_type,
        jcm_dt = args.jcm_timestep_min * 60.0,
        veros_dt_mom=args.veros_timestep_min * 60,
        veros_dt_tracer=args.veros_timestep_min * 60,
    )
    ocn_model = model.components["ocn"].raw_component

    tree_tools.print_tree(model.get_info(), root="Model")
    initial_carry = model.initialize()

    shared_setting = dict(
        workflow=model_config["workflow"],
        tqdm_kwargs = dict(
            position=1
        ),
    )

    training_trajectory_function = model.generate_trajectory_function(
        iterations = int(training_trajectory_interval / coupling_timestep),
        checkpoint=True,
        **shared_setting,
    )

    # Spin-up model
    
    print(f"Check the target spin-up checkpoint file {str(target_spinup_checkpoint_file)}")
    if target_spinup_checkpoint_file.exists():
        print(f"Target spin-up checkpoint file exists.")
    else:
        print(f"Target spin-up checkpoint file does not exist. Need to spin-up now.")
        carry = initial_carry
        spinup_trajectory_function = model.generate_trajectory_function(
            iterations = int(spinup_trajectory_interval / coupling_timestep),
            **shared_setting,
        )

        for current_iteration in range(long_direct_simulation_iterations):
            print(f"Long direct simulation iteration {current_iteration:d}/{long_direct_simulation_iterations:d}...")
            carry, spinup_predictions = spinup_trajectory_function(carry)
            spinup_checkpoint_file = get_spinup_checkpoint_file(current_iteration)
            print(f"Save checkpoint file: {str(spinup_checkpoint_file):s}")
            save_coupled_carry(
                carry, spinup_checkpoint_file,
                component_savers={"ocn": save_veros_carry},
            )
            
            output_dict = model.predictions_to_xarray(spinup_predictions)
            for component_name, ds in output_dict.items():
                output_file = config.output_dir_spinup / f"{component_name:s}-{current_iteration:03d}.nc"
                print("Output file: ", str(output_file))
                ds.to_netcdf(output_file, engine="netcdf4")
                ds.close()

        del carry


    # Build model context and wire up experiment-specific factories

    print(f"Load spin-up file {str(target_spinup_checkpoint_file)}")
    initial_carry = load_coupled_carry(
        target_spinup_checkpoint_file, ["atm", "ocn", "fakelnd"],
        component_loaders={"ocn": lambda path: load_veros_carry(path, ocn_model)},
    )

    context = ModelContext(
        carry=initial_carry,
        training_trajectory_function=training_trajectory_function,
        config=config,
    )

    loss_function = config.loss_fn_factory(context)
    initial_x = config.initial_x_factory(context)
    flat_initial_x, unpack = pack(initial_x)
    flat_loss_function = lambda flat: loss_function(unpack(flat))
    output_callback = config.output_callback_factory(context, unpack_function=unpack)

    _optimizer_classes = {
        "HamitonianMethod": HamitonianMethod,
        "RMSProp": RMSProp,
        "RMSPropMomentum": RMSPropMomentum,
        "LBFGS": LBFGS,
    }

    for stage in config.stages:
        if stage.method not in _optimizer_classes:
            raise ValueError(f"Unknown optimization method: '{stage.method}'")

    # Build optimizer instances once so JIT-compiled step functions are reused across loops
    _optimizer_instances = [
        _optimizer_classes[stage.method](flat_loss_function, **stage.optimizer_kwargs)
        for stage in config.stages
    ]

    print("Running Optimization...")
    start_time = time.perf_counter()
    current_x = flat_initial_x
    for loop_idx in range(config.stage_loops):
        for stage_idx, (stage, optimizer) in enumerate(zip(config.stages, _optimizer_instances)):
            print(f"Loop {loop_idx}, stage {stage_idx}: {stage.method} for {stage.iterations} iterations")
            stage_callback = functools.partial(
                output_callback,
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
