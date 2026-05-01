from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelContext:
    carry: dict
    training_trajectory_function: Callable
    config: object  # Config; untyped to avoid circular import
