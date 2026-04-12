from typing import List
import time

import jax
import jax.numpy as jnp
import optax

def stack_objects(
    objs: List,
):
    """
    A tool function that stack dataclasses together.

    Args:

        objs : A list of objects that need to be stacked

    Returns:

        stacked : Stacked object.

    """
    # objs is a list of pytrees with same structure
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return stacked

def scan(f, init, xs, callback=None, callback_interval:int=1):
    if len(xs) % callback_interval != 0:
        raise ValueError("Length of xs mut be multiple of callback_interval.")

    carry = init
    batches = int( len(xs) / callback_interval )
    for batch in range(batches):
        print(f"[batch {batch+1}/{batches:d}] Running")
        _start_time = time.time()
        carry, history = jax.lax.scan(
            f,
            carry,
            xs[batch*callback_interval:(batch+1)*callback_interval],
        )
        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        print(f"[batch {batch+1}/{batches:d}] Cost of time: {_elapsed_time:.2f} seconds.")
        if callback is not None:
            continue_scan = callback(history, batch * callback_interval)
            if not continue_scan:
                print("Callback requires to stop.")
                break

def RMSPropMomentum(
    initial_x,
    loss_function,
    iterations: int,
    callback = None,
    callback_interval:int = 1,
    initial_p = None,
    initial_square_dloss_dx = None,  # moving average of squared gradient
    memory_factor_square_dloss_dx: float = 0.9,  
    memory_factor_momentum: float = 0.9,  
    learning_rate: float = 0.01, 
    divide_by_zero_tolerance: float = 1e-8,
):

    if initial_p is None:
        print("initial_p is None. Set to zero.")
        initial_p = jnp.zeros_like(initial_x)

    if initial_square_dloss_dx is None:
        print("initial_square_dloss_dx is None. Set to zero.")
        initial_square_dloss_dx = jnp.zeros_like(initial_x)
    
    initial_carry = {
        'x' : initial_x,
        'p' : initial_p,
        'square_dloss_dx' : initial_square_dloss_dx,
    }
    
    value_and_grad_loss_function = jax.value_and_grad(loss_function)

    @jax.jit
    def step_function(carry, step):
        loss, dloss_dx = value_and_grad_loss_function(carry['x'])
        
        x = carry["x"]
        p = carry["p"]
        square_dloss_dx = carry["square_dloss_dx"]
        
        square_dloss_dx = memory_factor_square_dloss_dx * square_dloss_dx + (1.0 - memory_factor_square_dloss_dx) * (dloss_dx**2)
        p = memory_factor_momentum * p - learning_rate / ( divide_by_zero_tolerance + square_dloss_dx**0.5 ) * dloss_dx
        x = x + p # equivalent to stepping forward with timestep = 1.0
        
        new_carry = {
            'x': x,
            'p': p,
            'square_dloss_dx': square_dloss_dx,
        }
        
        predictions = {
            'step' : step,
            'loss' : loss,
            'dloss_dx' : dloss_dx,
            'x' : x,
            'p' : p,
            'square_dloss_dx' : square_dloss_dx,
            'K' : jnp.mean(p**2)/2,
        }
        
        return new_carry, predictions
    
    scan(
        step_function, 
        initial_carry,
        jnp.arange(iterations),
        callback=callback,
        callback_interval=callback_interval,
    )


def RMSProp(
    initial_x,
    loss_function,
    iterations: int,
    callback = None,
    callback_interval:int = 1,
    initial_square_dloss_dx = None,  # moving average of squared gradient
    memory_factor: float = 0.9,  
    learning_rate: float = 0.01, 
    divide_by_zero_tolerance: float = 1e-8,
):
    if initial_square_dloss_dx is None:
        print("initial_square_dloss_dx is None. Set to zero.")
        initial_square_dloss_dx = jnp.zeros_like(initial_x)
    
    initial_carry = {
        'x' : initial_x,
        'square_dloss_dx' : initial_square_dloss_dx,
    }
    
    value_and_grad_loss_function = jax.value_and_grad(loss_function)

    @jax.jit
    def step_function(carry, step):
        loss, dloss_dx = value_and_grad_loss_function(carry['x'])
        
        x = carry["x"]
        square_dloss_dx = carry["square_dloss_dx"]
        
        square_dloss_dx = memory_factor * square_dloss_dx + (1.0 - memory_factor) * (dloss_dx**2)
        x = x - learning_rate / ( divide_by_zero_tolerance + square_dloss_dx**0.5 ) * dloss_dx
        
        new_carry = {
            'x': x,
            'square_dloss_dx': square_dloss_dx,
        }
        
        predictions = {
            'step' : step,
            'loss' : loss,
            'dloss_dx' : dloss_dx,
            'x' : x,
            'square_dloss_dx' : square_dloss_dx,
        }
        
        return new_carry, predictions
    
    scan(
        step_function, 
        initial_carry,
        jnp.arange(iterations),
        callback=callback,
        callback_interval=callback_interval,
    )

def HamitonianMethod(
    initial_x,
    loss_function,
    iterations: int,
    callback = None,
    callback_interval:int = 1,
    initial_p = None,
    gravity: float = 1.0,
    timestep: float = 0.01,
    friction_timescale: float = 10.0,
):
    if initial_p is None:
        print("initial_p is None. Set to zero.")
        initial_p = jnp.zeros_like(initial_x)
    
    initial_carry = {
        'x' : initial_x,
        'p' : initial_p,
    }
    
    value_and_grad_loss_function = jax.value_and_grad(loss_function)

    @jax.jit
    def step_function(carry, step):
        loss, dloss_dx = value_and_grad_loss_function(carry['x'])
        
        x = carry["x"]
        p = carry["p"]
        
        p += timestep * ( - gravity * dloss_dx - p / friction_timescale )
        x += timestep * p
        
        new_carry = {
            'x': x,
            'p': p,
        }
        
        predictions = {
            'step' : step,
            'loss' : loss,
            'dloss_dx' : dloss_dx,
            'x' : x,
            'p' : p,
            'K' : jnp.mean(p**2)/2,
        }
        
        return new_carry, predictions
    
    scan(
        step_function, 
        initial_carry,
        jnp.arange(iterations),
        callback=callback,
        callback_interval=callback_interval,
    )


def LBFGS(
    initial_x,
    loss_function,
    iterations: int,
    learning_rate: float = 1e-1,
    callback = None,
    callback_interval:int = 1,
):
    
    opt = optax.scale_by_lbfgs()
    initial_carry = {
        'x' : initial_x,
        'optimizer_state' : opt.init(initial_x),
    }

    value_and_grad_loss_function = jax.value_and_grad(loss_function)

    @jax.jit
    def step_function(carry, step):
        loss, dloss_dx = value_and_grad_loss_function(carry['x'])
        update_vector, new_optimizer_state = opt.update(dloss_dx, carry['optimizer_state'], carry['x'])
        new_x = carry['x'] - learning_rate * update_vector

        new_carry = {
            'x' : new_x,
            'optimizer_state': new_optimizer_state,
        } 
        
        predictions = {
            'step' : step,
            'loss' : loss,
            'dloss_dx' : dloss_dx,
            'x' : new_x,
        }
        
        return new_carry, predictions
    
    scan(
        step_function, 
        initial_carry,
        jnp.arange(iterations),
        callback=callback,
        callback_interval=callback_interval,
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
            data_vars.update(dict(
                p = (("iteration", "lat"), history["p"]),
                K = (("iteration",), history["K"]),
            ))
        elif method == "RMSProp":
            data_vars.update(dict(
                square_dloss_dx = (("iteration", "lat"), history["square_dloss_dx"]),
            ))
        elif method == "RMSPropMomentum":
            data_vars.update(dict(
                square_dloss_dx = (("iteration", "lat"), history["square_dloss_dx"]),
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
   
    xx, yy = jnp.meshgrid(jnp.linspace(-5, 5, 100), jnp.linspace(-5, 5, 100), indexing="ij")
    grid_points = jnp.stack([xx, yy], axis=-1)
    loss_map = jax.vmap(loss_function)(grid_points.reshape(-1, 2)).reshape(xx.shape)
 
    initial_x = jnp.array([1.0, 5.0])
    
    iterations = 2000
    callback_interval = 5

    HamitonianMethod(
        initial_x = initial_x,
        initial_p = jnp.array([5.0, -1.0]),
        loss_function = loss_function,
        iterations = iterations,
        timestep = 0.01,
        callback = functools.partial(generic_output_callback, method="HamitonianMethod"),
        callback_interval=callback_interval,
    )
    
    RMSProp(
        initial_x = initial_x,
        loss_function = loss_function,
        iterations = iterations,
        callback = functools.partial(generic_output_callback, method="RMSProp"),
        callback_interval=callback_interval,
    )
 
    RMSPropMomentum(
        initial_x = initial_x,
        loss_function = loss_function,
        iterations = iterations,
        callback = functools.partial(generic_output_callback, method="RMSPropMomentum"),
        callback_interval=callback_interval,
    )
 
    LBFGS(
        initial_x = initial_x,
        loss_function = loss_function,
        iterations = iterations,
        learning_rate = 0.1,
        callback = functools.partial(generic_output_callback, method="LBFGS"),
        callback_interval=callback_interval,
    )
    
    predictions = {
        optimization_method : xr.open_mfdataset(str(output_dir / optimization_method / f"training_result-*.nc"), combine="nested", concat_dim="iteration")
        for optimization_method in ["HamitonianMethod", "RMSProp", "RMSPropMomentum"]
    }

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    mappable = ax[1].contourf(xx[:, 0], yy[0, :], loss_map, cmap="plasma")

    for _optimization_method , _predictions in predictions.items():
        t = _predictions["step"]
        x = _predictions["x"][:, 0]
        y = _predictions["x"][:, 1]

        ax[0].plot(t, _predictions["loss"], label=_optimization_method)
        ax[1].plot(x, y, label=_optimization_method)

    ax[0].set_title("(a) L(t)")
    ax[1].set_title("(b) Trajectory")

    cb = plt.colorbar(mappable, ax=ax[1])
    
    for _ax in ax.flatten():
        _ax.legend()
    
    fig.savefig("Optimizer_demo.png", dpi=200)    
    fig.savefig("Optimizer_demo.svg")    
    plt.show()
