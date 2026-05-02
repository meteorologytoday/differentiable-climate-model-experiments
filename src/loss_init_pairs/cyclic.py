import jax.numpy as jnp


def cyclic_loss(context):
    """
    Loss factory for the zonally symmetric cyclic equilibrium search.

    x shape: (lat,). The optimization variable is the initial SST, broadcast
    to (nlon, nlat) before being written into the carry.

    Loss = sum of squared zonal-mean SST difference between the end and the
    beginning of the trajectory. Minimizing this loss drives the initial SST
    toward a state that is periodic over the trajectory length; set
    training_trajectory_days = 360 in the config to target an annual cycle.
    """
    carry = context.carry
    trajectory_fn = context.training_trajectory_function

    def loss(sst):
        nlon = carry["ocn"]["state"].sea_surface_temperature.shape[0]
        carry["ocn"]["state"].sea_surface_temperature = (
            jnp.repeat(sst[None, :], nlon, axis=0)
        )
        final_carry, _ = trajectory_fn(carry)
        final_sst = final_carry["ocn"]["state"].sea_surface_temperature
        final_sst_zonal_mean = jnp.mean(final_sst, axis=0)
        return jnp.sum((final_sst_zonal_mean - sst) ** 2)

    return loss


def cyclic_initial_x(context):
    """Initial x: zonal mean of spinup SST, shape (lat,)."""
    return jnp.mean(context.carry["ocn"]["state"].sea_surface_temperature, axis=0)
