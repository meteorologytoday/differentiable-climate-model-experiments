from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List


@dataclass
class Stage:
    method: str
    iterations: int
    callback_interval: int = 10
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    # Required fields (must precede fields with defaults).
    loss_fn_factory: Callable
    initial_x_factory: Callable
    output_callback_factory: Callable
    simulation_label: str

    # Model
    spectral_truncation: int = 31

    # Temporal
    training_trajectory_days: int = 40
    spinup_interval_days: int = 365
    spinup_total_years: int = 20
    initial_condition_year: int = 1

    # Optimization
    stages: List[Stage] = field(default_factory=lambda: [
        Stage("LBFGS", 1000, optimizer_kwargs={"learning_rate": 1e-1}),
    ])
    stage_loops: int = 1

    # Experiment identity — used to construct output directory names
    training_label: str = "training"

    # Output
    output_root: Path = Path("experiment_set")

    def __post_init__(self):
        self.output_root = Path(self.output_root)

    @property
    def simulation_name(self) -> str:
        return f"{self.simulation_label}_{self.average_days:d}days_avg"

    @property
    def output_dir(self) -> Path:
        return (self.output_root / f"output_T{self.spectral_truncation:d}_{self.simulation_name}").resolve()

    @property
    def output_dir_spinup(self) -> Path:
        return self.output_dir / "spinup"

    @property
    def output_dir_training(self) -> Path:
        return self.output_dir / f"training_{self.training_label}"
