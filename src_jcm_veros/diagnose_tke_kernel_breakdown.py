# Diagnostic: bisect the *internals* of `set_tke_diffusivities_kernel` on the
# SECOND chained `model.step()` call to find exactly which intermediate
# quantity first turns the SST-perturbation tangent into NaN.
#
# `diagnose_second_step_breakdown.py` narrowed the NaN onset down to:
#   - clean right after the 1st full model.step()
#   - clean after eke.set_eke_diffusivities of the 2nd step
#   - NaN right after tke.set_tke_diffusivities of the 2nd step
#
# Notably, the `sqrt_singularity_removed` (Huber-smoothed sqrt) fix already
# applied to `vs.tke`/`vs.Nsqr` inside this very kernel did NOT prevent this
# -- so the remaining singularity must be in some *other* operation in the
# kernel whose problematic input values only arise once the kernel has been
# fed state that has been evolved by a full step (e.g. `vs.K_diss_v`, which
# starts at its initial value on the 1st call but is populated by
# `momentum`/`thermodynamics` from then on).
#
# This script reimplements `set_tke_diffusivities_kernel`'s computation
# (mirroring `veros/core/tke.py`, including the `sqrt_singularity_removed`
# fix) as a sequence of pure-JAX expressions, run on the state evolved by one
# full `model.step()`, and reports tangent-NaN status after each intermediate.
import jax
import jax.numpy as jnp
import jax_datetime as jdt

# `model_setup` (via `jem.components`) sets the Veros runtime backend to "jax"
# before any `veros.core` submodule is imported -- importing those submodules
# first locks the runtime settings and crashes with
# "Runtime settings cannot be modified after import of core modules".
from model_setup import build_model, get_ocean_surface_temperature, set_ocean_surface_temperature
from veros.core import utilities

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


STAGE_NAMES = [
    "1: sqrttke = sqrt_singularity_removed(tke[tau])",
    "2: mxl = sqrt(2)*sqrttke / sqrt_singularity_removed(Nsqr[tau]) * maskW",
    "3: mxl after backwards vertical-limiting for_loop + bottom clamp",
    "4: mxl after forwards vertical-limiting for_loop + maskW-floor clamp",
    "5: kappaM = minimum(kappaM_max, c_k * mxl * sqrttke)",
    "6: Rinumber = Nsqr[tau] / maximum(K_diss_v / maximum(1e-12, kappaM), 1e-12)",
    "7: Prandtlnumber = maximum(1.0, minimum(10, 6.6*Rinumber))",
    "8: kappaH = maximum(kappaH_min, kappaM / Prandtlnumber)",
    "9: kappaH after enable_kappaH_profile maximum-with-arctan-profile",
    "10: kappaM final = maximum(kappaM_min, kappaM)",
]


def make_kernel_partial(stage_idx):
    @jax.jit
    def fn(sst):
        state = perturb(sst)
        ocn_model.step(state)  # run the 1st step to completion (already shown clean)

        vs = state.variables
        settings = state.settings
        nz = state.dimensions["zt"]

        outputs = []

        sqrttke = utilities.sqrt_singularity_removed(vs.tke[:, :, :, vs.tau])
        outputs.append(sqrttke)
        if stage_idx == 1:
            return tuple(outputs)

        mxl = jnp.sqrt(2) * sqrttke / utilities.sqrt_singularity_removed(vs.Nsqr[:, :, :, vs.tau]) * vs.maskW
        outputs.append(mxl)
        if stage_idx == 2:
            return tuple(outputs)

        # tke_mxl_choice == 2 branch (mitgcm/OPA-style bounding), as in tke.py
        def backwards_pass(kinv, mxl):
            k = nz - kinv - 1
            return mxl.at[:, :, k].set(jnp.minimum(mxl[:, :, k], mxl[:, :, k + 1] + vs.dzt[k + 1]))

        mxl = jax.lax.fori_loop(1, nz, backwards_pass, mxl)
        mxl = mxl.at[:, :, -1].set(jnp.minimum(mxl[:, :, -1], settings.mxl_min + vs.dzt[-1]))
        outputs.append(mxl)
        if stage_idx == 3:
            return tuple(outputs)

        def forwards_pass(k, mxl):
            return mxl.at[:, :, k].set(jnp.minimum(mxl[:, :, k], mxl[:, :, k - 1] + vs.dzt[k]))

        mxl = jax.lax.fori_loop(1, nz, forwards_pass, mxl)
        mxl = jnp.maximum(mxl, settings.mxl_min)
        outputs.append(mxl)
        if stage_idx == 4:
            return tuple(outputs)

        kappaM = jnp.minimum(settings.kappaM_max, settings.c_k * mxl * sqrttke)
        outputs.append(kappaM)
        if stage_idx == 5:
            return tuple(outputs)

        K_diss_v = utilities.enforce_boundaries(vs.K_diss_v, settings.enable_cyclic_x)
        Rinumber = vs.Nsqr[:, :, :, vs.tau] / jnp.maximum(K_diss_v / jnp.maximum(1e-12, kappaM), 1e-12)
        outputs.append(Rinumber)
        if stage_idx == 6:
            return tuple(outputs)

        if settings.enable_Prandtl_tke:
            Prandtlnumber = jnp.maximum(1.0, jnp.minimum(10, 6.6 * Rinumber))
        else:
            Prandtlnumber = jnp.full_like(Rinumber, settings.Prandtl_tke0)
        outputs.append(Prandtlnumber)
        if stage_idx == 7:
            return tuple(outputs)

        kappaH = jnp.maximum(settings.kappaH_min, kappaM / Prandtlnumber)
        outputs.append(kappaH)
        if stage_idx == 8:
            return tuple(outputs)

        if settings.enable_kappaH_profile:
            kappaH = jnp.maximum(
                kappaH,
                (0.8 + 1.05 / settings.pi * jnp.arctan((-vs.zw[None, None, :] - 2500.0) / 222.2)) * 1e-4,
            )
        outputs.append(kappaH)
        if stage_idx == 9:
            return tuple(outputs)

        kappaM_final = jnp.maximum(settings.kappaM_min, kappaM)
        outputs.append(kappaM_final)
        return tuple(outputs)

    return fn


print("Bisecting set_tke_diffusivities_kernel's internals on the 2nd model.step() call...")
for i, name in enumerate(STAGE_NAMES, start=1):
    stage_fn = make_kernel_partial(i)
    _, t = jax.jvp(stage_fn, (sst_initial,), (tangent_sst_initial,))
    report(f"stage {name}", t)

print("Done.")
