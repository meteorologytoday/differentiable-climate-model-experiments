import jax.numpy as jnp
import jax
from veros_helper import get_ocean_temperature

def cyclic_loss(context):
    """
    Loss factory for the zonally symmetric cyclic equilibrium search.

    x shape: (lat,). The optimization variable is the initial SST, broadcast
    to (nlon, nlat) before being written into the carry.

    Loss = mean of squared zonal-mean Veros ocean temperature difference
    between the end and the beginning of the trajectory. Using the full 3-D
    prognostic temperature field (nlon, nlat, nz) rather than just SST gives a
    stronger cyclic constraint. Minimizing this loss drives the initial SST
    toward a state where the full ocean temperature profile is periodic over the
    trajectory length; set training_trajectory_days = 365 in the config to
    target an annual cycle.
    """
    carry = context.carry
    trajectory_fn = context.training_trajectory_function

    def loss(sst):
        nlon = carry["ocn"]["state"].sea_surface_temperature.shape[0]
        carry["ocn"]["state"].sea_surface_temperature = (
            jnp.repeat(sst[None, :], nlon, axis=0)
        )
        begin_ocn_temp = get_ocean_temperature(carry["ocn"]["state"])
        final_carry, _ = trajectory_fn(carry)
        final_ocn_temp = get_ocean_temperature(final_carry["ocn"]["state"])
        begin_ocn_temp_zonal_mean = jnp.mean(begin_ocn_temp, axis=0)
        final_ocn_temp_zonal_mean = jnp.mean(final_ocn_temp, axis=0)
        jax.debug.print("DEBUG:begin_ocn_temp_mean = {v}", v=jnp.mean(begin_ocn_temp))
        jax.debug.print("DEBUG:final_ocn_temp_mean = {v}", v=jnp.mean(final_ocn_temp))
        return jnp.mean((final_ocn_temp_zonal_mean - begin_ocn_temp_zonal_mean) ** 2)

    return loss


def cyclic_initial_x(context):
    """Initial x: zonal mean of spinup SST, shape (lat,)."""
    jax.debug.print("DEBUG:cyclic_initial_x is called")
    return jnp.mean(context.carry["ocn"]["state"].sea_surface_temperature, axis=0)
