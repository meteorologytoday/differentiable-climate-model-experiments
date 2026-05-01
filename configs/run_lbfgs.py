from configs.base import Config, Stage

config = Config(
    training_label="LBFGS",
    stages=[
        Stage("LBFGS", 1000, callback_interval=10, optimizer_kwargs={"learning_rate": 1e-1}),
    ],
    stage_loops=1,
)
