# %% [markdown]
# # Use gradient descend to find equilibrium SST on an aquaplanet

# %%
from pathlib import Path
import time

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
from jem.utils.optimizers import HamitonianMethod
# %%
jax.config.update("jax_compilation_cache_dir", "./.jax_cache")
print("Devices: ", jax.devices())

# %%
simulation_name = "02-04_aquaplanet_equilibrium_with_fully_spinup_sst"
spectral_truncation = 31
result_file = Path("training_result.nc")
output_dir = Path(f"output_T{spectral_truncation:d}_{simulation_name}").resolve()
output_dir_spinup = output_dir / "spinup"
output_dir_training = output_dir / "training"
for d in [output_dir_spinup, output_dir_training]:
    d.mkdir(exist_ok=True, parents=True)

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
one_second = jdt.to_timedelta(1, "second")
trajectory_interval = jdt.to_timedelta(40, "day")

long_direct_simulation_time = jdt.to_timedelta(360*5, "day")  #
long_direct_simulation_iterations = int(long_direct_simulation_time / trajectory_interval)
spinup_time = jdt.to_timedelta(360*5, "day")
spinup_iterations = int(spinup_time / trajectory_interval)

def get_spinup_checkpoint_file(iteration):
    return (
        output_dir_spinup / 
        f"adjusted_carray_spinup-iterations-{iteration:03d}"
        f"_eachinterval-{int(trajectory_interval/jdt.to_timedelta(1, 'day')):d}"
        f"-days.chkpt"
    ).resolve()

def get_training_checkpoint_file(iteration):
    return (
        output_dir_training / 
        f"iterations-{iteration:03d}.chkpt"
    ).resolve()

target_spinup_checkpoint_file = get_spinup_checkpoint_file(spinup_iterations-1)

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
    time_step=10.0,
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

trajectory_function = model.generate_trajectory_function(
        workflow=["mapper", "atm", "ocn"],
        iterations = int(trajectory_interval / coupling_timestep),
        tqdm_kwargs = dict(
            position=1
        ),
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
    for current_iteration in range(long_direct_simulation_iterations):
        print(f"Long direct simulation iteration {current_iteration:d}/{long_direct_simulation_iterations:d}...")
        adjusted_carry, spinup_predictions = trajectory_function(adjusted_carry)
        spinup_checkpoint_file = get_spinup_checkpoint_file(current_iteration)
        print(f"Save checkpoint file: {str(spinup_checkpoint_file):s}")
        ocp.save_pytree(spinup_checkpoint_file, adjusted_carry, overwrite=True)
        if current_iteration+1 in [1, spinup_iterations, long_direct_simulation_iterations]:
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
    _, predictions = trajectory_function(adjusted_carry)
    return jnp.mean(jnp.mean(predictions["ocn"]["forcing"]["total_heat_flux"][-30:, :, :], axis=0)**2) 


initial_carry = ocp.load_pytree(target_spinup_checkpoint_file, initial_carry)
initial_sst = jnp.mean(initial_carry["ocn"]["state"].sea_surface_temperature, axis=0)

# %%

def output(history, i):
    output_file = output_dir_training / f"training_result-{i:05d}.nc"

    ds_result = xr.Dataset(
        data_vars = dict(
            loss = (("iteration",), history["loss"]),
            sst = (("iteration", "lat"), history["x"]),
            sst_momentum = (("iteration", "lat"), history["p"]),
            dloss_dsst = (("iteration", "lat"), history["dloss_dx"]),
            sst_kinetic_energy = (("iteration",), history["K"]),
        ),
        coords = dict(
        )
    )

    print(f"Save training results to : {str(output_file)}")
    ds_result.to_netcdf(output_file, unlimited_dims="iteration")

    return jnp.all(jnp.isfinite(history["x"]))

# %%

print("Running HamitonianMethod...")
start_time = time.perf_counter()
training_history = HamitonianMethod(
    initial_x = initial_sst,
    loss_function = loss_function,
    iterations = 300,
    timestep = 0.01,
    callback = output,
    callback_interval=5,
)
end_time = time.perf_counter()
cost_of_time = end_time - start_time
print(f"Cost of time: {cost_of_time} seconds.")



