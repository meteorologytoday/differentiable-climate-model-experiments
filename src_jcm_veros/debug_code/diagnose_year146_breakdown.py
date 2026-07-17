# Step day-by-day through year 146 of the long coupled run (job 849210),
# starting from the last healthy checkpoint (end of year 145, batch_00145),
# to find the exact day on which `ocn.state.{temp,salt,u,v,Nsqr}` first goes
# non-finite, and to inspect the grid point (full-grid idx=(0,6,0) for
# Nsqr/v/salt/temp, idx=(0,2,0) for u) where the original crash first
# reported NaNs.
#
# Built with `debug_mode=True`, matching job 849210's configuration, so the
# `report_first_nonfinite` diagnostics fire if/when the state goes
# non-finite. `model_setup.py`'s `jax.debug.breakpoint()` call has been
# commented out (fatal without a TTY), so a non-finite state no longer kills
# the job -- the scan just keeps propagating NaNs and the script's own
# host-side `np.isfinite` checks (below) catch and report the first day on
# which that happens.
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax_datetime as jdt

import jem
from jem.utils.checkpoints import load_coupled_carry, load_veros_carry
import jem.utils.tree_tools as tree_tools

from model_setup import build_model

print(f"jem library is located at: {jem.__file__}")
print(f"Available devices: {jax.devices()}")


def first_nonfinite_index(arr):
    arr = np.asarray(arr)
    bad = np.argwhere(~np.isfinite(arr))
    if bad.size == 0:
        return None
    return tuple(int(i) for i in bad[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="output_T31/long_run/checkpoint/batch_00145",
                         help="Restart checkpoint to step forward from (last healthy state).")
    parser.add_argument("--truncation-number", type=int, default=31)
    parser.add_argument("--max-days", type=int, default=365, help="Max number of days (coupling steps) to step through.")
    args = parser.parse_args()

    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(24, "hour")
    calendar = "365_day"

    model, config = build_model(
        truncation_number=args.truncation_number,
        start_datetime=start_datetime,
        coupling_timestep=coupling_timestep,
        calendar=calendar,
        debug_mode=True,
    )
    ocn_model = model.components["ocn"].raw_component

    print("Model info: ")
    tree_tools.print_tree(model.get_info(), root="Model")

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    print(f"Loading checkpoint from: {checkpoint_dir}")
    carry = model.initialize()
    carry = load_coupled_carry(
        checkpoint_dir, ["atm", "ocn", "fakelnd"],
        component_loaders={"ocn": lambda path: load_veros_carry(path, ocn_model)},
    )

    # Same workflow ordering as `main_forward.py` (the script that produced
    # the long run): ["mapper", "ocn", "atm", "fakelnd"].
    trajectory_function = model.generate_trajectory_function(
        workflow=["mapper", "ocn", "atm", "fakelnd"],
        iterations=1,
        jitted=True,
        show_progress=False,
    )

    # Static grid context, printed once.
    vs0 = carry["ocn"]["state"].variables
    settings0 = carry["ocn"]["state"].settings
    print(f"enable_cyclic_x = {settings0.enable_cyclic_x}")
    print(f"temp.shape (nx, ny, nz) = {vs0.temp.shape}")
    print(f"zt = {np.asarray(vs0.zt)}")
    print(f"dzt = {np.asarray(vs0.dzt)}")
    print(f"kbot[0, :] = {np.asarray(vs0.kbot[0, :])}")
    print(f"kbot[1, :] = {np.asarray(vs0.kbot[1, :])}")
    for (x, y, z), label in (((0, 6, 0), "Nsqr/v/salt/temp"), ((0, 2, 0), "u")):
        print(
            f"At (x={x},y={y},z={z}) [{label}]: "
            f"maskT={float(vs0.maskT[x, y, z]):.0f}, "
            f"maskU={float(vs0.maskU[x, y, z]):.0f}, "
            f"maskV={float(vs0.maskV[x, y, z]):.0f}, "
            f"maskW={float(vs0.maskW[x, y, z]):.0f}, "
            f"kbot={int(vs0.kbot[x, y])}"
        )

    fields_to_check = ["temp", "salt", "u", "v", "Nsqr"]
    watch_points = {"Nsqr": (0, 6, 0), "v": (0, 6, 0), "salt": (0, 6, 0), "temp": (0, 6, 0), "u": (0, 2, 0)}

    for day in range(args.max_days):

        final_carry, predictions = trajectory_function(carry)

        vs = final_carry["ocn"]["state"].variables
        tau = int(vs.tau)

        values = {f: np.asarray(getattr(vs, f)[..., tau]) for f in fields_to_check}

        max_u = float(np.max(np.abs(values["u"])))
        max_v = float(np.max(np.abs(values["v"])))
        min_nsqr = float(np.min(values["Nsqr"]))
        watch_str = ", ".join(
            f"{f}{watch_points[f]}={values[f][watch_points[f]]:.6e}" for f in fields_to_check
        )
        print(f"[day={day:3d}] max|u|={max_u:.6e}, max|v|={max_v:.6e}, min(Nsqr)={min_nsqr:.6e}, {watch_str}")

        bad_fields = {f: first_nonfinite_index(values[f]) for f in fields_to_check}
        bad_fields = {f: idx for f, idx in bad_fields.items() if idx is not None}

        if bad_fields:
            print(f"\n*** First non-finite values detected on day {day} (within year 146) ***")
            for f, idx in bad_fields.items():
                print(f"  {f}: first non-finite at idx={idx}, value={values[f][idx]!r}")
            break

        carry = final_carry

    else:
        print(f"\nNo non-finite values detected within {args.max_days} days.")

    print("Done.")
