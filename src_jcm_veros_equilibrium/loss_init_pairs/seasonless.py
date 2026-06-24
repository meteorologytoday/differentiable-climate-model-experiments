from functools import partial
import jax.numpy as jnp
from veros_helper import set_ocean_temperature, get_ocean_temperature, set_ocean_salinity, get_ocean_salinity
from callbacks import standard_output_callback

def seasonless_loss(context, *, average_length, start_index, compare_index):
    """
    Loss factory for the zonally symmetric, seasonless equilibrium search.

    compute the mean of time from [start_index:start_index+average_length]
    and [compare_index:compare_index+average_length].


    x shape: (lat,). The zonal symmetry assumption lives here — x is broadcast
    to (nlon, nlat) before being written into the carry.

    Loss = mean squared net heat flux over the last average_days of the
    trajectory, i.e. we want the ocean to be in radiative equilibrium.
    """
    carry = context.carry
    trajectory_fn = context.training_trajectory_function

    def loss(x):
        ocean_temp = x[0]
        ocean_salt = x[1]

        ocn_state = carry["ocn"]["state"]
        set_ocean_temperature(ocn_state, ocean_temp)       
        set_ocean_salinity(ocn_state, ocean_salt)
        _, predictions = trajectory_fn(carry)
        

        diff = lambda x: (
              jnp.mean(x[compare_index:compare_index+average_length, :, :, :], axis=0)
            - jnp.mean(x[start_index:start_index+average_length, :, :, :], axis=0)
        )

        temp_difference = diff(predictions["ocn"]["temp"])
        salt_difference = diff(predictions["ocn"]["salt"])

        return jnp.mean(temp_difference ** 2)
        #return jnp.mean(jnp.mean(temp_difference, axis=0) ** 2)

    return loss


def seasonless_initial_x(context):
    """Initial x: (ocean_temp (nlon, nlat, nz), ocean_salt (nlon, nlat, nz)) from spinup state."""
    return (
        get_ocean_temperature(context.carry["ocn"]["state"]),
        get_ocean_salinity(context.carry["ocn"]["state"]),
    )


seasonless_output_callback = partial(
    standard_output_callback,
    x_dims=(("lon", "lat", "z"), ("lon", "lat", "z")),
    x_var_names=("ocean_temp", "ocean_salt"),
)
