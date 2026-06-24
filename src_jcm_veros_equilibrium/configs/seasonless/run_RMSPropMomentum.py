from functools import partial
from configs.base import Config, Stage
from loss_init_pairs.seasonless import seasonless_loss, seasonless_initial_x, seasonless_output_callback

average_length = 30
start_index = 30
compare_index = 60

config = Config(
    loss_fn_factory=partial(seasonless_loss, average_length=average_length, start_index=start_index, compare_index=compare_index),
    initial_x_factory=seasonless_initial_x,
    output_callback_factory=seasonless_output_callback,
    simulation_label="fully_coupled_equilibrium",
    training_trajectory_days=compare_index + average_length,
    training_label="RMSPropMomentum",
    stages=[
        Stage(
            method="RMSPropMomentum",
            iterations=5,
            callback_interval=5,
            optimizer_kwargs=dict(
                memory_factor_square_dloss_dx=0.9,
                memory_factor_momentum=0.9,
                learning_rate=5e-2,
                divide_by_zero_tolerance=1e-8,
            ),
        ),
    ],
    stage_loops=50,
)
