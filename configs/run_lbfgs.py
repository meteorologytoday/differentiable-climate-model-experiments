from configs.base import Config

cfg = Config(
    optimization_method="LBFGS",
    optimization_iterations=1000,
    optimization_callback_interval=10,
    optimizer_kwargs={"learning_rate": 1e-1},
)
