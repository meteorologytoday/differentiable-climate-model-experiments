from typing import List
import time

import jax
import jax.numpy as jnp
import optax


def stack_objects(objs: List):
    """Stack a list of pytrees with the same structure into a single pytree."""
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return stacked


def scan_with_callback(f, init, xs, callback=None, callback_interval: int = 1):
    if len(xs) % callback_interval != 0:
        raise ValueError("Length of xs must be multiple of callback_interval.")

    carry = init
    batched_history = []
    batches = int(len(xs) / callback_interval)
    for batch in range(batches):
        print(f"[batch {batch+1}/{batches:d}] Running")
        _start_time = time.time()
        carry, _history = jax.lax.scan(
            f,
            carry,
            xs[batch*callback_interval:(batch+1)*callback_interval],
        )
        batched_history.append(_history)
        _end_time = time.time()
        print(f"[batch {batch+1}/{batches:d}] Cost of time: {_end_time - _start_time:.2f} seconds.")
        if callback is not None:
            continue_scan = callback(_history, batch * callback_interval)
            if not continue_scan:
                print("Callback requires to stop.")
                break

    return carry, batched_history


class RMSPropMomentum:
    def __init__(
        self,
        loss_function,
        memory_factor_square_dloss_dx: float = 0.9,
        memory_factor_momentum: float = 0.9,
        learning_rate: float = 0.01,
        divide_by_zero_tolerance: float = 1e-8,
    ):
        value_and_grad = jax.value_and_grad(loss_function)

        @jax.jit
        def step_function(carry, step):
            loss, dloss_dx = value_and_grad(carry['x'])
            x = carry["x"]
            p = carry["p"]
            square_dloss_dx = carry["square_dloss_dx"]
            square_dloss_dx = memory_factor_square_dloss_dx * square_dloss_dx + (1.0 - memory_factor_square_dloss_dx) * (dloss_dx**2)
            p = memory_factor_momentum * p - learning_rate / (divide_by_zero_tolerance + square_dloss_dx**0.5) * dloss_dx
            x = x + p
            new_carry = {'x': x, 'p': p, 'square_dloss_dx': square_dloss_dx}
            predictions = {
                'step': step, 'loss': loss, 'dloss_dx': dloss_dx,
                'x': x, 'p': p, 'square_dloss_dx': square_dloss_dx,
                'K': jnp.mean(p**2) / 2,
            }
            return new_carry, predictions

        self._step_function = step_function

    def __call__(self, initial_x, iterations, callback=None, callback_interval=1,
                 initial_p=None, initial_square_dloss_dx=None):
        if initial_p is None:
            print("initial_p is None. Set to zero.")
            initial_p = jnp.zeros_like(initial_x)
        if initial_square_dloss_dx is None:
            print("initial_square_dloss_dx is None. Set to zero.")
            initial_square_dloss_dx = jnp.zeros_like(initial_x)
        initial_carry = {'x': initial_x, 'p': initial_p, 'square_dloss_dx': initial_square_dloss_dx}
        return scan_with_callback(
            self._step_function, initial_carry, jnp.arange(iterations),
            callback=callback, callback_interval=callback_interval,
        )


class RMSProp:
    def __init__(
        self,
        loss_function,
        memory_factor: float = 0.9,
        learning_rate: float = 0.01,
        divide_by_zero_tolerance: float = 1e-8,
    ):
        value_and_grad = jax.value_and_grad(loss_function)

        @jax.jit
        def step_function(carry, step):
            loss, dloss_dx = value_and_grad(carry['x'])
            x = carry["x"]
            square_dloss_dx = carry["square_dloss_dx"]
            square_dloss_dx = memory_factor * square_dloss_dx + (1.0 - memory_factor) * (dloss_dx**2)
            x = x - learning_rate / (divide_by_zero_tolerance + square_dloss_dx**0.5) * dloss_dx
            new_carry = {'x': x, 'square_dloss_dx': square_dloss_dx}
            predictions = {
                'step': step, 'loss': loss, 'dloss_dx': dloss_dx,
                'x': x, 'square_dloss_dx': square_dloss_dx,
            }
            return new_carry, predictions

        self._step_function = step_function

    def __call__(self, initial_x, iterations, callback=None, callback_interval=1,
                 initial_square_dloss_dx=None):
        if initial_square_dloss_dx is None:
            print("initial_square_dloss_dx is None. Set to zero.")
            initial_square_dloss_dx = jnp.zeros_like(initial_x)
        initial_carry = {'x': initial_x, 'square_dloss_dx': initial_square_dloss_dx}
        return scan_with_callback(
            self._step_function, initial_carry, jnp.arange(iterations),
            callback=callback, callback_interval=callback_interval,
        )


class HamitonianMethod:
    def __init__(
        self,
        loss_function,
        gravity: float = 1.0,
        timestep: float = 0.01,
        friction_timescale: float = 10.0,
    ):
        value_and_grad = jax.value_and_grad(loss_function)

        @jax.jit
        def step_function(carry, step):
            loss, dloss_dx = value_and_grad(carry['x'])
            x = carry["x"]
            p = carry["p"]
            p += timestep * (-gravity * dloss_dx - p / friction_timescale)
            x += timestep * p
            new_carry = {'x': x, 'p': p}
            predictions = {
                'step': step, 'loss': loss, 'dloss_dx': dloss_dx,
                'x': x, 'p': p, 'K': jnp.mean(p**2) / 2,
            }
            return new_carry, predictions

        self._step_function = step_function

    def __call__(self, initial_x, iterations, callback=None, callback_interval=1, initial_p=None):
        if initial_p is None:
            print("initial_p is None. Set to zero.")
            initial_p = jnp.zeros_like(initial_x)
        initial_carry = {'x': initial_x, 'p': initial_p}
        return scan_with_callback(
            self._step_function, initial_carry, jnp.arange(iterations),
            callback=callback, callback_interval=callback_interval,
        )


class LBFGS:
    def __init__(self, loss_function, learning_rate: float = 0.1):
        value_and_grad = jax.value_and_grad(loss_function)
        opt = optax.scale_by_lbfgs()

        @jax.jit
        def step_function(carry, step):
            loss, dloss_dx = value_and_grad(carry['x'])
            update_vector, new_optimizer_state = opt.update(dloss_dx, carry['optimizer_state'], carry['x'])
            new_x = carry['x'] - learning_rate * update_vector
            new_carry = {'x': new_x, 'optimizer_state': new_optimizer_state}
            predictions = {'step': step, 'loss': loss, 'dloss_dx': dloss_dx, 'x': new_x}
            return new_carry, predictions

        self._step_function = step_function
        self._opt = opt

    def __call__(self, initial_x, iterations, callback=None, callback_interval=1):
        initial_carry = {'x': initial_x, 'optimizer_state': self._opt.init(initial_x)}
        return scan_with_callback(
            self._step_function, initial_carry, jnp.arange(iterations),
            callback=callback, callback_interval=callback_interval,
        )


if __name__ == "__main__":

    import numpy as np
    from pathlib import Path
    import xarray as xr
    import functools

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    def loss_function(x):
        return jnp.sum(x**2) + jnp.sum(x)

    def generic_output_callback(history, i, method: str):
        if method == "":
            raise ValueError("Method cannot be empty.")

        output_file = output_dir / method / f"training_result-{i:05d}.nc"
        output_file.parent.mkdir(exist_ok=True, parents=True)

        data_vars = dict(
            step = (("iteration",), history["step"]),
            loss = (("iteration",), history["loss"]),
            x = (("iteration", "lat"), history["x"]),
            dloss_dx = (("iteration", "lat"), history["dloss_dx"]),
        )

        if method == "HamitonianMethod":
            data_vars.update(dict(p=(("iteration", "lat"), history["p"]), K=(("iteration",), history["K"])))
        elif method in ("RMSProp", "RMSPropMomentum"):
            data_vars.update(dict(square_dloss_dx=(("iteration", "lat"), history["square_dloss_dx"])))

        ds_result = xr.Dataset(data_vars=data_vars, coords=dict())
        print(f"Save training results to : {str(output_file)}")
        ds_result.to_netcdf(output_file, unlimited_dims="iteration")

        return jnp.all(jnp.isfinite(history["x"]))

    xx, yy = jnp.meshgrid(jnp.linspace(-5, 5, 100), jnp.linspace(-5, 5, 100), indexing="ij")
    grid_points = jnp.stack([xx, yy], axis=-1)
    loss_map = jax.vmap(loss_function)(grid_points.reshape(-1, 2)).reshape(xx.shape)

    initial_x = jnp.array([1.0, 5.0])
    iterations = 2000
    callback_interval = 5

    HamitonianMethod(loss_function, timestep=0.01)(
        initial_x=initial_x,
        initial_p=jnp.array([5.0, -1.0]),
        iterations=iterations,
        callback=functools.partial(generic_output_callback, method="HamitonianMethod"),
        callback_interval=callback_interval,
    )

    RMSProp(loss_function)(
        initial_x=initial_x,
        iterations=iterations,
        callback=functools.partial(generic_output_callback, method="RMSProp"),
        callback_interval=callback_interval,
    )

    RMSPropMomentum(loss_function)(
        initial_x=initial_x,
        iterations=iterations,
        callback=functools.partial(generic_output_callback, method="RMSPropMomentum"),
        callback_interval=callback_interval,
    )

    LBFGS(loss_function, learning_rate=0.1)(
        initial_x=initial_x,
        iterations=iterations,
        callback=functools.partial(generic_output_callback, method="LBFGS"),
        callback_interval=callback_interval,
    )

    predictions = {
        optimization_method: xr.open_mfdataset(
            str(output_dir / optimization_method / "training_result-*.nc"),
            combine="nested", concat_dim="iteration",
        )
        for optimization_method in ["HamitonianMethod", "RMSProp", "RMSPropMomentum"]
    }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    mappable = ax[1].contourf(xx[:, 0], yy[0, :], loss_map, cmap="plasma")

    for _optimization_method, _predictions in predictions.items():
        t = _predictions["step"]
        x = _predictions["x"][:, 0]
        y = _predictions["x"][:, 1]
        ax[0].plot(t, _predictions["loss"], label=_optimization_method)
        ax[1].plot(x, y, label=_optimization_method)

    ax[0].set_title("(a) L(t)")
    ax[1].set_title("(b) Trajectory")
    plt.colorbar(mappable, ax=ax[1])
    for _ax in ax.flatten():
        _ax.legend()

    fig.savefig("Optimizer_demo.png", dpi=200)
    fig.savefig("Optimizer_demo.svg")
    plt.show()
