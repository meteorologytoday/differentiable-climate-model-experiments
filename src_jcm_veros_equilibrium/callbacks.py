import jax
import jax.numpy as jnp
import xarray as xr
from typing import Callable, Optional, Sequence


def standard_output_callback(
    context,
    unpack_function: Optional[Callable] = None,
    x_dims: Sequence[tuple] = (("lat",),),
    x_var_names: Sequence[str] = ("sst",),
):
    """
    Generic output callback factory.

    unpack_function: if provided, applied via jax.vmap to all x-shaped history
                     arrays before saving. Should be the unpack function returned
                     by pack().
    x_dims: one dimension-name tuple per optimization variable.
            Default (("lat",),) for zonally symmetric SST.
            Use (("lon", "lat", "z"), ("lon", "lat", "z")) for 3D temp + salt.
    x_var_names: one name per leaf of the unpacked x pytree.
                 For a plain array, use a single name, e.g. ("sst",).
                 For a tuple (ocean_temp, ocean_salt), use ("ocean_temp", "ocean_salt").
    """
    output_dir = context.config.output_dir_training

    def _unpack_iter(flat_arr):
        if unpack_function is None:
            return [flat_arr]
        unpacked = jax.vmap(unpack_function)(flat_arr)
        return list(unpacked)

    def callback(history, i, method, loop_idx, stage_idx):
        output_file = (
            output_dir /
            f"training_result-loop_{loop_idx:03d}-stage_{stage_idx:02d}"
            f"-iter_{i:03d}-{method}.nc"
        )

        x_leaves = _unpack_iter(history["x"])
        grad_leaves = _unpack_iter(history["dloss_dx"])

        data_vars = dict(loss=(("iteration",), history["loss"]))
        for name, dims, x_arr, grad_arr in zip(x_var_names, x_dims, x_leaves, grad_leaves):
            iter_dims = ("iteration",) + dims
            data_vars[name] = (iter_dims, x_arr)
            data_vars[f"dloss_d{name}"] = (iter_dims, grad_arr)

        if method == "HamitonianMethod":
            p_leaves = _unpack_iter(history["p"])
            for name, dims, p_arr in zip(x_var_names, x_dims, p_leaves):
                data_vars[f"{name}_momentum"] = (("iteration",) + dims, p_arr)
            data_vars["kinetic_energy"] = (("iteration",), history["K"])
        elif method == "RMSProp":
            sq_leaves = _unpack_iter(history["square_dloss_dx"])
            for name, dims, sq_arr in zip(x_var_names, x_dims, sq_leaves):
                data_vars[f"square_dloss_d{name}"] = (("iteration",) + dims, sq_arr)
        elif method == "RMSPropMomentum":
            sq_leaves = _unpack_iter(history["square_dloss_dx"])
            for name, dims, sq_arr in zip(x_var_names, x_dims, sq_leaves):
                data_vars[f"square_dloss_d{name}"] = (("iteration",) + dims, sq_arr)
            data_vars["kinetic_energy"] = (("iteration",), history["K"])

        ds_result = xr.Dataset(data_vars=data_vars, coords=dict())
        print(f"Save training results to : {str(output_file)}")
        ds_result.to_netcdf(output_file, unlimited_dims="iteration")

        return jnp.all(jnp.isfinite(history["x"]))

    return callback
