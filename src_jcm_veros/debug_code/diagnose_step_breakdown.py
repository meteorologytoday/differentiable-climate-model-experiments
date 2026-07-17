# Diagnostic: bisect a single Veros `model.step()` call to localize exactly
# which physics routine first turns the SST-perturbation tangent into NaN.
#
# `diagnose_tangent_propagation.py` showed that the tangent is intact through
# perturb+read-back (stage0) but is already all-NaN after "one ocean
# coupling-step" (stage1). However a single JEM coupling-step actually chains
# `steps_per_coupling_timestep = coupling_timestep / dt_tracer` raw
# `model.step()` calls (here 86400s / 3600s = 24).
#
# This script first checks whether a SINGLE raw `model.step()` call already
# produces a NaN tangent:
#   - If yes, it bisects that one step into its constituent routines
#     (mirroring the call sequence in `veros/veros.py: Veros.step`) and
#     reports after which routine the NaN first appears.
#   - If no, it bisects across the chain of `model.step()` calls instead,
#     to find after how many steps the NaN first appears.
import jax
import jax.numpy as jnp
import jax_datetime as jdt

# `model_setup` (via `jem.components`) sets the Veros runtime backend to "jax"
# before any `veros.core` submodule is imported -- importing those submodules
# first locks the runtime settings and crashes with
# "Runtime settings cannot be modified after import of core modules".
from model_setup import build_model, get_ocean_surface_temperature, set_ocean_surface_temperature
from veros.core import eke, tke, momentum, thermodynamics, advection

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
truncation_number = 31
calendar = "365_day"

model, config = build_model(
    truncation_number=truncation_number,
    start_datetime=start_datetime,
    coupling_timestep=coupling_timestep,
    calendar=calendar,
)
ocn_model = model.components["ocn"].raw_component

initial_coupled_carry = model.initialize()
ocn_carry0 = initial_coupled_carry["ocn"]

sst_initial = get_ocean_surface_temperature(ocn_carry0["state"])
shape2D = sst_initial.shape
tangent_sst_initial = jnp.zeros_like(sst_initial).at[shape2D[0] // 2, shape2D[1] // 2].set(1.0)


def perturb(sst):
    ocn_state = jax.tree_util.tree_map(lambda x: x, ocn_carry0["state"])
    return set_ocean_surface_temperature(ocn_state, sst)


def any_nan_in_tangent(tree):
    found = False
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = jnp.asarray(leaf)
        if jnp.issubdtype(arr.dtype, jnp.floating) and bool(jnp.any(jnp.isnan(arr))):
            found = True
    return found


def report(name, tangent):
    print(f"[{name}] any NaN in tangent = {any_nan_in_tangent(tangent)}")


# `jax.jvp` requires its function's output to be a pytree of plain arrays --
# returning the full `state` (a `Lockable`/`StrictContainer` carrying string
# and metadata fields alongside its array variables) makes JAX choke while
# reconstructing the traced pytree. Instead we read out a fixed, broad set of
# float state variables -- the union of every routine's `KernelOutput` fields
# in the bisected call chain -- so a NaN is visible regardless of which
# routine first introduces it, even if it hasn't yet propagated all the way
# to the prognostic fields (temp/salt/u/v) by the time we stop.
READOUT_FIELDS = [
    # eke.set_eke_diffusivities / eke.integrate_eke
    "L_rossby", "L_rhines", "eke_len", "sqrteke", "K_gm", "K_iso",
    "eke", "eke_diss_iw", "eke_diss_tke",
    # tke.set_tke_diffusivities / tke.integrate_tke
    "sqrttke", "mxl", "kappaM", "kappaH", "Prandtlnumber", "K_diss_v",
    "tke", "tke_surf_corr", "tke_diss",
    # momentum.momentum
    "du", "dv", "du_cor", "dv_cor", "du_adv", "dv_adv", "u", "v", "w",
    # thermodynamics.thermodynamics
    "temp", "salt",
    # advection.calculate_velocity_on_wgrid
    "u_wgrid", "v_wgrid", "w_wgrid",
]


def readout(state):
    vs = state.variables
    return tuple(getattr(vs, name) for name in READOUT_FIELDS)


# ---- Level 1: does a single raw `model.step()` call already go NaN? ----
@jax.jit
def one_step(sst):
    state = perturb(sst)
    ocn_model.step(state)
    return readout(state)


print("Computing: a single raw model.step() call...")
_, t_one_step = jax.jvp(one_step, (sst_initial,), (tangent_sst_initial,))
report("single model.step()", t_one_step)
single_step_is_nan = any_nan_in_tangent(t_one_step)


SUBSTEP_NAMES = [
    "eke.set_eke_diffusivities",
    "tke.set_tke_diffusivities",
    "momentum.momentum",
    "thermodynamics.thermodynamics",
    "advection.calculate_velocity_on_wgrid",
    "eke.integrate_eke",
    "tke.integrate_tke",
]


def make_partial_step(num_substeps):
    @jax.jit
    def partial_step(sst):
        state = perturb(sst)
        settings = state.settings

        substep_fns = [
            lambda: eke.set_eke_diffusivities(state),
            lambda: tke.set_tke_diffusivities(state),
            lambda: momentum.momentum(state),
            lambda: thermodynamics.thermodynamics(state),
            (lambda: advection.calculate_velocity_on_wgrid(state))
            if (settings.enable_eke or settings.enable_tke or settings.enable_idemix)
            else (lambda: None),
            (lambda: eke.integrate_eke(state)) if settings.enable_eke else (lambda: None),
            (lambda: tke.integrate_tke(state)) if settings.enable_tke else (lambda: None),
        ]

        for fn in substep_fns[:num_substeps]:
            fn()

        return readout(state)

    return partial_step


def make_n_steps(n):
    @jax.jit
    def n_steps(sst):
        state = perturb(sst)

        def body(_, state):
            ocn_model.step(state)
            return state

        state = jax.lax.fori_loop(0, n, body, state)
        return readout(state)

    return n_steps


if single_step_is_nan:
    print("A single model.step() already produces a NaN tangent.")
    print("Bisecting that one step into its constituent routines (in call order)...")
    for i, name in enumerate(SUBSTEP_NAMES, start=1):
        stage_fn = make_partial_step(i)
        _, t = jax.jvp(stage_fn, (sst_initial,), (tangent_sst_initial,))
        report(f"after substep {i} ({name})", t)
else:
    print("A single model.step() is clean -- the NaN must emerge from chaining several steps.")
    print("Bisecting across the chain of model.step() calls (a coupling-step runs 24)...")
    for n in [2, 4, 8, 12, 16, 20, 24]:
        stage_fn = make_n_steps(n)
        _, t = jax.jvp(stage_fn, (sst_initial,), (tangent_sst_initial,))
        report(f"after {n} chained model.step() calls", t)

print("Done.")
