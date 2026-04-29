from configs.base import Config

cfg = Config(
    optimization_method="RMSPropMomentum",
    optimization_iterations=100,
    optimization_callback_interval=10,
    optimizer_kwargs=dict(
        memory_factor_square_dloss_dx = 0.9,
        memory_factor_momentum = 0.9,
        learning_rate = 5e-2,
        divide_by_zero_tolerance = 1e-8,
    ),
)
