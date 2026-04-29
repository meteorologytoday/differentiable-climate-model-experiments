from configs.base import Config, Stage

cfg = Config(
    training_label="Mixed_RMSProp_RMSMomentum",
    stages=[
        Stage(
            method="RMSPropMomentum",
            iterations=20,
            callback_interval=10,
            optimizer_kwargs=dict(
                memory_factor_square_dloss_dx=0.9,
                memory_factor_momentum=0.9,
                learning_rate=5e-2,
                divide_by_zero_tolerance=1e-8,
            ),
        ),
        Stage(
            method="RMSProp",
            iterations=50,
            callback_interval=10,
            optimizer_kwargs=dict(
                memory_factor=0.9,
                learning_rate=5e-2,
                divide_by_zero_tolerance=1e-8,
            ),
        ),
    ],
    stage_loops=10,
)
