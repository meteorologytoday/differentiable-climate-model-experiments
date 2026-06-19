# Response of the coupled model to an SST bump using jax.jvp, compared
# against a direct (finite-difference ensemble) estimate.
#
# The quantity whose sensitivity is studied is a "measure" -- see
# `sensitivity_measures.py`. Swapping `--measure` changes what is computed,
# saved, and plotted without touching the driver code below.
import numpy as np
import jax
import jax.numpy as jnp
import argparse
from pathlib import Path

import jcm
import jax_datetime as jdt

import jem
from jem.utils.checkpoints import load_coupled_carry, load_veros_carry
import jem.utils.tree_tools as tree_tools

from model_setup import build_model
from veros_helper import get_ocean_temperature, set_ocean_temperature
from sensitivity_measures import MEASURES


print(f"jcm library is located at: {jcm.__file__}")
print(f"jem library is located at: {jem.__file__}")

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")


def report_tangent(name, tangent):
    tangent = jnp.asarray(tangent)
    print(
        f"[{name}] max|tangent| = {float(jnp.max(jnp.abs(tangent))):.6e}, "
        f"sum|tangent| = {float(jnp.sum(jnp.abs(tangent))):.6e}, "
        f"any nonzero = {bool(jnp.any(tangent != 0.0))}, "
        f"any nan = {bool(jnp.any(jnp.isnan(tangent)))}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-dir", type=str, help="Restart dir", default=None)
    parser.add_argument("--total-simulation-days", type=int, help="Total time of simulation in days", default=5)
    parser.add_argument("--truncation-number", type=int, help="Truncation number", default=31)
    parser.add_argument("--test-ensemble-members", type=int, nargs="+", help="The ensemble members to be used", default=[1])
    parser.add_argument("--output-filename", type=str, help="The result in netcdf file.", default="sensitivity_data.nc")
    parser.add_argument("--measure", type=str, choices=sorted(MEASURES.keys()), default="ocean_temperature_zonal_mean",
                         help="Which diagnostic to compute the sensitivity of.")
    args = parser.parse_args()

    measure = MEASURES[args.measure]

    # Configurations
    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    simulation_name = "sensitivity"
    output_dir = (Path("output") / simulation_name).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    one_second = jdt.to_timedelta(1, "second")
    truncation_number = args.truncation_number
    calendar = "365_day"
    test_ensemble_members = np.array(args.test_ensemble_members)
    ensemble_members = np.amax(test_ensemble_members)
    simulation_interval = jdt.to_timedelta(args.total_simulation_days, "day")

    output_file = output_dir / args.output_filename
    if not output_file.exists():

        # Build the coupled JCM + Veros + SlabOceanModel system. Packaged as a
        # function in `model_setup.py` so that other scripts (e.g. a jax.grad
        # sensitivity experiment) can build exactly the same model.
        model, config = build_model(
            truncation_number=truncation_number,
            start_datetime=start_datetime,
            coupling_timestep=coupling_timestep,
            calendar=calendar,
        )
        ocn_model = model.components["ocn"].raw_component

        print("Model info: ")
        tree_tools.print_tree(model.get_info(), root="Model")

        # Currently, we still call initialize() even if there is a restart directory assigned.
        # This is because there are some constants or model configuration that may actually
        # happen during initialization. Since how these numbers are pre-determined is not enforced
        # in our framework, the simple solution is to still call initialize, follow the workflow
        # of a completely new restart, then replace this initial_coupled_carry if restart files
        # are loaded.
        initial_coupled_carry = model.initialize()

        # Checkpoint
        if (args.restart_dir is not None) and (args.restart_dir != ""):
            restart_dir = Path(args.restart_dir)
            if not restart_dir.exists():
                raise FileNotFoundError(f"The specified restart directory {str(restart_dir):s} does not exist.")

            print(f"Use restart files in {str(restart_dir):s}")
            initial_coupled_carry = load_coupled_carry(
                restart_dir, ["atm", "ocn", "fakelnd"],
                component_loaders={"ocn": lambda path: load_veros_carry(path, ocn_model)},
            )

        trajectory_function = model.generate_trajectory_function(
            workflow=config["workflow"],
            iterations = int(simulation_interval / coupling_timestep),
        )

        @jax.jit
        def forecast(temp, salt, u, v):

            # Work on a structural copy of the ocean state so that perturbing it
            # here cannot mutate (or leak tracers into) `initial_coupled_carry`,
            # which is captured by closure and must stay reusable across calls.
            ocn_state = jax.tree_util.tree_map(lambda x: x, initial_coupled_carry["ocn"]["state"])
            ocn_state = set_ocean_temperature(ocn_state, temp)

            modified_carry = dict(
                initial_coupled_carry,
                ocn=dict(
                    initial_coupled_carry["ocn"],
                    state=ocn_state,               # overwrite ocn with perturbed one
                ),
            )
            final_carry, predictions = trajectory_function(modified_carry)
            return measure.compute(final_carry, predictions)
        
        temp_initial = get_ocean_temp(initial_coupled_carry["ocn"]["state"])
        
        atm_model = model.components["atm"].raw_component
        lat = atm_model.coords.horizontal.latitudes * 180/jnp.pi
        lon = atm_model.coords.horizontal.longitudes * 180/jnp.pi
        llon, llat = jnp.meshgrid(lon, lat, indexing="ij")
        
        def gaussian(x, y, xc, yc, sigma_x, sigma_y):
            return jnp.exp( -  (x-xc)**2 / (2*sigma_x**2) - (y-yc)**2 / (2*sigma_y**2) )

        tangent_sst_initial = gaussian(llon, llat, 180.0, 0.0, 5, 8) * 1
        tangent_sst_initial /= jnp.sum(tangent_sst_initial**2)**0.5

        # Use jax.jvp to obtain the sensitivity of `measure` to the SST perturbation
        print("Compute sensitivity using jax.jvp...")
        measure_final, tangent_measure = jax.jvp(forecast, (sst_initial,), (tangent_sst_initial,))

        report_tangent(f"tangent_{args.measure} (jvp)", tangent_measure)

        print("Compute sensitivity using direct method")
        epsilon = 0.01
        sst_noise_magnitude = 0.001
        
        ensemble = []
        for i in range(ensemble_members):
            print(f"Running ensemble member ({i:d}/{ensemble_members:d})")
            _sst_perturbation = sst_noise_magnitude * jax.random.normal(shape=sst_initial.shape, key=jax.random.PRNGKey(i))
            _measure_final_perturbed = forecast(sst_initial + tangent_sst_initial * epsilon + _sst_perturbation)
            _sensitivity_measure = jax.tree.map(lambda a, b: (a - b) / epsilon, _measure_final_perturbed, measure_final)

            ensemble.append(_sensitivity_measure)

        print("Saving simulation output...")
        measure.save(
            output_file,
            tangent_sst_initial=tangent_sst_initial,
            tangent_measure=tangent_measure,
            ensemble=ensemble,
        )


    if output_file.exists():
        pass
        """
        output_figure = output_dir / "sensitivity_comparison.png"
        measure.plot(
            output_file,
            output_figure,
            f"Response time: {simulation_interval / jdt.to_timedelta(1, 'day'):.1f} days",
        )
        """
        print(f"Already done. File {str(output_file)} exists.")
    else:
        raise Exception("Something went wrong. No output file detected.")
