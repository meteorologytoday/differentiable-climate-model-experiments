import jax.numpy as jnp


def seasonless_loss(context, *, average_days):
    """
    Loss factory for the zonally symmetric, seasonless equilibrium search.

    x shape: (lat,). The zonal symmetry assumption lives here — x is broadcast
    to (nlon, nlat) before being written into the carry.

    Loss = mean squared net heat flux over the last average_days of the
    trajectory, i.e. we want the ocean to be in radiative equilibrium.
    """
    carry = context.carry
    trajectory_fn = context.training_trajectory_function

    def loss(sst):
        nlon = carry["ocn"]["state"].sea_surface_temperature.shape[0]
        carry["ocn"]["state"].sea_surface_temperature = (
            jnp.repeat(sst[None, :], nlon, axis=0)
        )
        _, predictions = trajectory_fn(carry)
        flux = predictions["ocn"]["forcing"]["total_heat_flux"][-average_days:, :, :]
        return jnp.mean(jnp.mean(flux, axis=0) ** 2)

    return loss


def seasonless_initial_x(context):
    """Initial x: zonal mean of spinup SST, shape (lat,)."""
    return jnp.mean(context.carry["ocn"]["state"].sea_surface_temperature, axis=0)
