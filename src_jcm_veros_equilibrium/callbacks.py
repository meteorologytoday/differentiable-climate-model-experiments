import jax.numpy as jnp
import xarray as xr


def standard_output_callback(context, x_dims=("lat",)):
    """
    Generic output callback factory.

    x_dims: dimension names for the optimization variable x.
            Default ("lat",) for zonally symmetric SST.
            Use ("lon", "lat") for a full 2D SST field.
    """
    output_dir = context.config.output_dir_training

    def callback(history, i, method, loop_idx, stage_idx):
        output_file = (
            output_dir /
            f"training_result-loop_{loop_idx:03d}-stage_{stage_idx:02d}"
            f"-iter_{i:03d}-{method}.nc"
        )

        iter_and_x = ("iteration",) + x_dims
        data_vars = dict(
            loss=(("iteration",), history["loss"]),
            sst=(iter_and_x, history["x"]),
            dloss_dsst=(iter_and_x, history["dloss_dx"]),
        )

        if method == "HamitonianMethod":
            data_vars.update(dict(
                sst_momentum=(iter_and_x, history["p"]),
                sst_kinetic_energy=(("iteration",), history["K"]),
            ))
        elif method == "RMSProp":
            data_vars.update(dict(
                square_dloss_dsst=(iter_and_x, history["square_dloss_dx"]),
            ))
        elif method == "RMSPropMomentum":
            data_vars.update(dict(
                square_dloss_dsst=(iter_and_x, history["square_dloss_dx"]),
                sst_kinetic_energy=(("iteration",), history["K"]),
            ))

        ds_result = xr.Dataset(data_vars=data_vars, coords=dict())
        print(f"Save training results to : {str(output_file)}")
        ds_result.to_netcdf(output_file, unlimited_dims="iteration")

        return jnp.all(jnp.isfinite(history["x"]))

    return callback
