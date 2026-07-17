# Diagnostic: localize the NaN that appears between the 1st and 2nd chained
# `model.step()` calls.
#
# `diagnose_step_breakdown.py` established that:
#   - a single raw `model.step()` call keeps the SST-perturbation tangent
#     finite (`any NaN in tangent = False`)
#   - chaining just 2 calls already produces NaN, and it stays NaN for
#     4/8/12/16/20/24 chained calls
#
# So the singularity isn't in the initial state -- it only appears in the
# state *evolved* by one full step (e.g. a quantity that starts away from a
# singular point but lands exactly on it after one step's evolution, or an
# Adam-Bashforth `taum1` slot that is identically zero on step 1 but becomes
# data-dependent from step 2 onward).
#
# This script runs step 1 to completion (clean), then bisects step 2 into its
# constituent routines (same call order as `veros/veros.py: Veros.step`),
# reporting after which routine of the *second* step the tangent first turns
# to NaN.
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


# Same readout union as `diagnose_step_breakdown.py` -- a fixed, broad set of
# float state variables (the union of every routine's `KernelOutput` fields in
# the bisected call chain), so a NaN is visible regardless of which routine
# first introduces it, even mid-step.
READOUT_FIELDS = [
    "L_rossby", "L_rhines", "eke_len", "sqrteke", "K_gm", "K_iso",
    "eke", "eke_diss_iw", "eke_diss_tke",
    "sqrttke", "mxl", "kappaM", "kappaH", "Prandtlnumber", "K_diss_v",
    "tke", "tke_surf_corr", "tke_diss",
    "du", "dv", "du_cor", "dv_cor", "du_adv", "dv_adv", "u", "v", "w",
    "temp", "salt",
    "u_wgrid", "v_wgrid", "w_wgrid",
]


def readout(state):
    vs = state.variables
    return tuple(getattr(vs, name) for name in READOUT_FIELDS)


SUBSTEP_NAMES = [
    "eke.set_eke_diffusivities",
    "tke.set_tke_diffusivities",
    "momentum.momentum",
    "thermodynamics.thermodynamics",
    "advection.calculate_velocity_on_wgrid",
    "eke.integrate_eke",
    "tke.integrate_tke",
]


def make_second_step_partial(num_substeps):
    @jax.jit
    def fn(sst):
        state = perturb(sst)

        # Run the FIRST step to completion (already shown to be clean on its own).
        ocn_model.step(state)

        # Bisect the SECOND step into its constituent routines.
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

        for substep_fn in substep_fns[:num_substeps]:
            substep_fn()

        return readout(state)

    return fn


# Baseline: confirm step 1 alone (read out right after it) is clean here too.
@jax.jit
def after_first_step(sst):
    state = perturb(sst)
    ocn_model.step(state)
    return readout(state)


print("Computing: readout right after the 1st model.step() (sanity check, expect clean)...")
_, t0 = jax.jvp(after_first_step, (sst_initial,), (tangent_sst_initial,))
report("after 1st model.step()", t0)

print("Bisecting the 2nd model.step() into its constituent routines...")
for i, name in enumerate(SUBSTEP_NAMES, start=1):
    stage_fn = make_second_step_partial(i)
    _, t = jax.jvp(stage_fn, (sst_initial,), (tangent_sst_initial,))
    report(f"after 1st step + substep {i} of 2nd step ({name})", t)

print("Done.")
