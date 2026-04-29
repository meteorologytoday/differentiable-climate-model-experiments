from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


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
    optimization_method: str = "LBFGS"
    optimization_iterations: int = 1000
    optimization_callback_interval: int = 10
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {"learning_rate": 1e-1})

    # Experiment identity — used to construct output directory names
    simulation_label: str = "02-04_aquaplanet_equilibrium_with_1year_spinup_sst"

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
        return self.output_dir / f"training_{self.optimization_method}"
