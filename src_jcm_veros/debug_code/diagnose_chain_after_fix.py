"""
Diagnostic: after the tke-clamp fix confirmed step 2 is clean, find which
step in the 24-step coupling chain is the NEXT NaN source.

Uses Python-level for-loops (not jax.lax.fori_loop) so each step is a
separate jax.jit-traced function -- this makes it possible to stop at any
individual step and inspect, rather than tracing a loop body that hides the
per-step behaviour.

Reports any_NaN_in_tangent after each step from step 1 up to 24.
"""
import jax
import jax.numpy as jnp
import jax_datetime as jdt

from model_setup import build_model, get_ocean_surface_temperature, set_ocean_surface_temperature

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")

model, config = build_model(
    truncation_number=31,
    start_datetime=start_datetime,
    coupling_timestep=coupling_timestep,
    calendar="365_day",
)
ocn_model = model.components["ocn"].raw_component

initial_coupled_carry = model.initialize()
ocn_carry0 = initial_coupled_carry["ocn"]

sst_initial = get_ocean_surface_temperature(ocn_carry0["state"])
shape2D = sst_initial.shape
tangent_sst_initial = jnp.zeros_like(sst_initial).at[shape2D[0] // 2, shape2D[1] // 2].set(1.0)

READOUT_FIELDS = [
    "sqrttke", "mxl", "kappaM", "kappaH", "Prandtlnumber", "K_diss_v",
    "tke", "tke_surf_corr", "tke_diss",
    "sqrteke", "K_gm", "K_iso", "eke",
    "du", "dv", "u", "v", "w",
    "temp", "salt",
    "u_wgrid", "v_wgrid", "w_wgrid",
]


def readout(state):
    vs = state.variables
    return tuple(getattr(vs, name) for name in READOUT_FIELDS)


def any_nan_in_tangent(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = jnp.asarray(leaf)
        if jnp.issubdtype(arr.dtype, jnp.floating) and bool(jnp.any(jnp.isnan(arr))):
            return True
    return False


def make_n_steps(n):
    """Chain exactly n raw model.step() calls via Python for-loop."""
    @jax.jit
    def fn(sst):
        state = jax.tree_util.tree_map(lambda x: x, ocn_carry0["state"])
        state = set_ocean_surface_temperature(state, sst)
        for _ in range(n):
            ocn_model.step(state)
        return readout(state)
    return fn


# Test n = 1, 2, 3, 4, 6, 8, 12, 16, 20, 24 to find the crossover
# (steps are compiled fresh each time due to different n, JIT-cache differs).
steps_to_test = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]

print("Scanning chain length for first NaN reappearance (post-tke-clamp-fix)...")
for n in steps_to_test:
    fn = make_n_steps(n)
    _, t = jax.jvp(fn, (sst_initial,), (tangent_sst_initial,))
    nan = any_nan_in_tangent(t)
    print(f"  n={n:2d} steps: any NaN = {nan}")
    if nan:
        print(f"  -> NaN first appears at or before step {n}")
        # Now bisect between the last-clean step and n
        if n > 1:
            prev_clean = steps_to_test[steps_to_test.index(n) - 1]
            print(f"  Bisecting between step {prev_clean} (clean) and step {n} (NaN)...")
            for m in range(prev_clean + 1, n):
                fm = make_n_steps(m)
                _, tm = jax.jvp(fm, (sst_initial,), (tangent_sst_initial,))
                nanm = any_nan_in_tangent(tm)
                print(f"    n={m:2d} steps: any NaN = {nanm}")
                if nanm:
                    print(f"  -> NaN first appears at step {m}")
                    break
        break

print("Done.")
