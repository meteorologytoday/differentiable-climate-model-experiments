from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrd

import optax

# Define objective
dim = 1

w_optimal = jnp.zeros(dim)

def loss_function(w):
    return 0.5 * jnp.sum( (w - w_optimal)**2 )

# Define optimizer
lr = 5e-1
opt = optax.scale_by_lbfgs()

# Initialize optimization
w = jrd.normal(jrd.PRNGKey(10), (dim,))
state = opt.init(w)

w_record = [w, ]
# Run optimization
for i in range(16):
    v, g = jax.value_and_grad(loss_function)(w)
    print(f'Iteration: {i}, Value:{v:.2e}')
    u, state = opt.update(g, state, w)
    w = w - lr * u
    w_record.append(w)

print(f'Final value: {loss_function(w):.2e}')


import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1)

for i, _w, in enumerate(w_record):
    ax.scatter(_w[0], loss_function(_w), s=20, c="red")



plt.show()
