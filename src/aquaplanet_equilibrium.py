# %% [markdown]
# # Use gradient descend to find equilibrium SST on an aquaplanet

# %%
from pathlib import Path
import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp
import xarray as xr
from tqdm import tqdm
# %%
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
print(f"Note: jcm.__file__ = {str(jcm.__file__)}")
import jax_datetime as jdt

# %%
from jem.components import JCM, SlabOceanModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools
from jem.utils.bulk_op import stack_objects
from optimizers import HamitonianMethod, RMSProp, RMSPropMomentum, LBFGS
# %%
jax.config.update("jax_compilation_cache_dir", "./.jax_cache")
print("Devices: ", jax.devices())

# %%

average_days = 30
atmosphere_memory_days = 10

optimization_method = "LBFGS"
optimization_iterations = 1000
optimization_callback_interval = 10

simulation_name = f"02-04_aquaplanet_equilibrium_with_1year_spinup_sst_{average_days:d}days_avg"
spectral_truncation = 31
result_file = Path("training_result.nc")
output_dir = Path(f"experiment_set/output_T{spectral_truncation:d}_{simulation_name}").resolve()
output_dir_spinup = output_dir / "spinup"
output_dir_training = output_dir / f"training_{optimization_method:s}"
for d in [output_dir_spinup, output_dir_training]:
    d.mkdir(exist_ok=True, parents=True)

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
one_second = jdt.to_timedelta(1, "second")
spinup_trajectory_interval = jdt.to_timedelta(40, "day")
training_trajectory_interval = jdt.to_timedelta(atmosphere_memory_days + average_days, "day")

long_direct_simulation_time = jdt.to_timedelta(360*20, "day")  #
long_direct_simulation_iterations = int(long_direct_simulation_time / spinup_trajectory_interval)
initial_condition_time = jdt.to_timedelta(360, "day")

def get_spinup_checkpoint_file(iteration):
    return (
        output_dir_spinup / 
        f"adjusted_carray_spinup-iterations-{iteration:03d}"
        f"_eachinterval-{int(spinup_trajectory_interval/jdt.to_timedelta(1, 'day')):d}"
        f"-days.chkpt"
    ).resolve()

def get_training_checkpoint_file(iteration):
    return (
        output_dir_training / 
        f"iterations-{iteration:03d}.chkpt"
    ).resolve()

target_spinup_checkpoint_file = get_spinup_checkpoint_file(int(initial_condition_time / spinup_trajectory_interval) - 1)

# %% [markdown]
# ## Add Mapper

# %%
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

# %% [markdown]
# ## Construct Model

# %%
atm_model = jcm.model.Model(
    coords=get_speedy_coords(spectral_truncation=spectral_truncation),
    start_date=start_datetime,
    time_step=30.0,
)

# %%
atm_model = JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)

# %%
model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=SlabOceanModel(
            grid_specification=f"JCM::T{spectral_truncation:d}",
            start_datetime=start_datetime,
            timestep=coupling_timestep / one_second,
        ),
    ),
    mappers=dict(mapper=mapper),
)

# %%
print("Model info: ") 
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


# %% [markdown]
# ## Spin-up model
# I want atmosphere to be turbulent to start with. So I need to spin-up the model for about 30 days.

# %%
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
        if True: #current_iteration+1 in [1, spinup_iterations, long_direct_simulation_iterations]:
            output_dict = model.predictions_to_xarray(spinup_predictions)
            for component_name, ds in output_dict.items():
                output_file = output_dir_spinup / f"{component_name:s}-{current_iteration:03d}.nc"
                print("Output file: ", str(output_file))
                ds.to_netcdf(output_file, engine="netcdf4")
                ds.close()
    del adjusted_carry


#adjusted_carry = ocp.load_pytree(target_spinup_checkpoint_file, initial_carry)

# %% [markdown]
# ## Define the Loss Function

# %%

def loss_function(sst):
    print(f"Load spin-up file {str(target_spinup_checkpoint_file)}")
    adjusted_carry = ocp.load_pytree(target_spinup_checkpoint_file, initial_carry)
    nlon = adjusted_carry["ocn"]["state"].sea_surface_temperature.shape[0]
    adjusted_carry["ocn"]["state"].sea_surface_temperature = jnp.repeat(sst[None, :], nlon, axis=0)
    _, predictions = training_trajectory_function(adjusted_carry)
    return jnp.mean(jnp.mean(predictions["ocn"]["forcing"]["total_heat_flux"][-average_days:, :, :], axis=0)**2) 


initial_carry = ocp.load_pytree(target_spinup_checkpoint_file, initial_carry)
initial_sst = jnp.mean(initial_carry["ocn"]["state"].sea_surface_temperature, axis=0)

# %%

def generic_output_callback(history, i, method: str):
    output_file = output_dir_training / f"training_result-{i:05d}.nc"

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
    elif method == "LBFGS":
        pass


    ds_result = xr.Dataset(
        data_vars = data_vars,
        coords = dict(
        )
    )

    print(f"Save training results to : {str(output_file)}")
    ds_result.to_netcdf(output_file, unlimited_dims="iteration")

    return jnp.all(jnp.isfinite(history["x"]))

specialized_output_callback = functools.partial(
    generic_output_callback,
    method=optimization_method,
)

# %%

print("Running Optimization...")
start_time = time.perf_counter()
if optimization_method == "HamitonianMethod":
    training_history = HamitonianMethod(
        initial_x = initial_sst,
        loss_function = loss_function,
        iterations = optimization_iterations,
        timestep = 0.01,
        callback = specialized_output_callback,
        callback_interval=optimization_callback_interval,
        momentum_cap = 200,
    )
elif optimization_method == "RMSProp":
    training_history = RMSProp(
        initial_x = initial_sst,
        loss_function = loss_function,
        iterations = optimization_iterations,
        callback = specialized_output_callback,
        callback_interval=optimization_callback_interval,
        memory_factor = 0.9,
        learning_rate = 5e-2,
        divide_by_zero_tolerance = 1e-8,
    )
elif optimization_method == "RMSPropMomentum":
    training_history = RMSPropMomentum(
        initial_x = initial_sst,
        loss_function = loss_function,
        iterations = optimization_iterations,
        callback = specialized_output_callback,
        callback_interval=optimization_callback_interval,
        memory_factor_square_dloss_dx = 0.9,
        memory_factor_momentum = 0.9,
        learning_rate = 5e-2,
        divide_by_zero_tolerance = 1e-8,
    )
elif optimization_method == "LBFGS":
    training_history = LBFGS(
        initial_x = initial_sst,
        loss_function = loss_function,
        iterations = optimization_iterations,
        callback = specialized_output_callback,
        callback_interval=optimization_callback_interval,
        learning_rate = 1e-1,
    )


else:
    raise ValueError(f"Error: Implementation of the optimization_method '{optimization_method:s}' does not exist.")
end_time = time.perf_counter()
cost_of_time = end_time - start_time
print(f"Cost of time: {cost_of_time} seconds.")



