from functools import partial
from configs.base import Config, Stage
from loss_init_pairs.seasonless import seasonless_loss, seasonless_initial_x
from callbacks import standard_output_callback

average_days = 30
atmosphere_memory_days = 10

config = Config(
    loss_fn_factory=partial(seasonless_loss, average_days=average_days),
    initial_x_factory=seasonless_initial_x,
    output_callback_factory=standard_output_callback,
    simulation_label="aquaplanet_equilibrium_with_1year_spinup_sst",
    training_trajectory_days=average_days + atmosphere_memory_days,
    training_label="LBFGS",
    stages=[
        Stage("LBFGS", 1000, callback_interval=10, optimizer_kwargs={"learning_rate": 1e-1}),
    ],
    stage_loops=1,
)
