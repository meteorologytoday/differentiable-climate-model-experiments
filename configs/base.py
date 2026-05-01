from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Stage:
    method: str
    iterations: int
    callback_interval: int = 10
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    # Model
    spectral_truncation: int = 31

    # Temporal
    average_days: int = 30
    atmosphere_memory_days: int = 10
    spinup_interval_days: int = 40
    spinup_total_years: int = 20
    initial_condition_year: int = 1

    # Optimization
    stages: List[Stage] = field(default_factory=lambda: [
        Stage("LBFGS", 1000, optimizer_kwargs={"learning_rate": 1e-1}),
    ])
    stage_loops: int = 1

    # Factories — each is a callable (ModelContext) -> experiment-specific object.
    # Defaults to equilibrium experiment factories; override in run_<name>.py.
    # None triggers lazy import of the equilibrium defaults in __post_init__.
    loss_fn_factory: Optional[Callable] = None
    initial_x_factory: Optional[Callable] = None
    output_callback_factory: Optional[Callable] = None

    # Experiment identity — used to construct output directory names
    simulation_label: str = "02-04_aquaplanet_equilibrium_with_1year_spinup_sst"
    training_label: str = "training"

    # Output
    output_root: Path = Path("experiment_set")

    def __post_init__(self):
        self.output_root = Path(self.output_root)
        # Lazy import avoids a hard dependency on src/ at configs/base.py import time.
        if self.loss_fn_factory is None:
            from loss_init_pairs.seasonless import seasonless_loss
            self.loss_fn_factory = seasonless_loss
        if self.initial_x_factory is None:
            from loss_init_pairs.seasonless import seasonless_initial_x
            self.initial_x_factory = seasonless_initial_x
        if self.output_callback_factory is None:
            from callbacks import standard_output_callback
            self.output_callback_factory = standard_output_callback

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
