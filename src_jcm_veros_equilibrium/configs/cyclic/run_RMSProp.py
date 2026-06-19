from configs.base import Config, Stage
from loss_init_pairs.cyclic import cyclic_loss, cyclic_initial_x
from callbacks import standard_output_callback

config = Config(
    loss_fn_factory=cyclic_loss,
    initial_x_factory=cyclic_initial_x,
    output_callback_factory=standard_output_callback,
    simulation_label="fullly-coupled_cyclic_equilibrium_with_2year_spinup",
    training_trajectory_days=365,
    spinup_total_years = 2,
    training_label="RMSProp",
    stages=[
        Stage(
            method="RMSProp",
            iterations=20,
            callback_interval=5,
            optimizer_kwargs=dict(
                memory_factor=0.9,
                learning_rate=5e-2,
                divide_by_zero_tolerance=1e-8,
            ),
        ),
    ],
    stage_loops=50,
)
