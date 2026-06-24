import jax.numpy as jnp
from veros_helper import set_ocean_temperature, get_ocean_temperature, set_ocean_salinity, get_ocean_salinity
import jax

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
    trajectory_fn = jax.checkpoint(context.training_trajectory_function)
    begin_ocn_temp = get_ocean_temperature(carry["ocn"]["state"])

    def loss(x):
        ocean_temp = x[0]
        ocean_salt = x[1]

        ocn_state = carry["ocn"]["state"]
        set_ocean_temperature(ocn_state, ocean_temp)       
        set_ocean_salinity(ocn_state, ocean_salt)
        _, predictions = trajectory_fn(carry)
        

        diff = lambda x: (
              jnp.mean(x[compare_index:compare_index+average_days, :, :], axis=0)
            - jnp.mean(x[start_index:start_index+average_days, :, :], axis=0)
        )

        temp_difference = diff(predictions["ocn"]["temp"])
        salt_difference = diff(predictions["ocn"]["salt"])

        return jnp.mean(diff ** 2)

    return loss


def seasonless_initial_x(context):
    """Initial x: zonal mean of spinup SST, shape (lat,)."""
    return (
        get_ocean_temperature(context.carry["ocn"]["state"]),
        get_ocean_salinity(context.carry["ocn"]["state"]),
    )
