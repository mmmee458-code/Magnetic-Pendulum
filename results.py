from dataclasses import dataclass
import numpy as np

from config import SimulationParameters

@dataclass
class SimulationResult:
    """
    Stores the output of a magnetic pendulum simulation and its
    derived measurements.

    The SimulationResult object acts as the central data container
    passed between the simulation, measurement, and visualization
    stages of the program.
    """

    # Simulation configuration
    parameters: SimulationParameters
    resolution: int
    backend: str = "cuda"


    # Basic simulation results
    runtime: float | None = None
    states: np.ndarray | None = None
    magnet_positions: np.ndarray | None = None

    # Basin classification
    basins: np.ndarray | None = None
    settled: np.ndarray | None = None
    settled_percentage: float | None = None

    # Basin entropy measurements
    basin_entropy: float | None = None
    boundary_basin_entropy: float | None = None

    # Uncertainty exponent measurements
    uncertainty_exponent: float | None = None
    intercept: float | None = None
    r_squared: float | None = None

    # Raw uncertainty-exponent data
    delta_values: np.ndarray | None = None
    uncertainty_fractions: np.ndarray | None = None

    # log-transformed data used for regression
    log_delta: np.ndarray | None = None
    log_fractions: np.ndarray | None = None

    # Percentage of sample for which both trajectories
    # successfully reached a settled state.
    uncertainty_settled_percentage: np.ndarray | None = None

@dataclass
class SweepResult:
    parameter: str
    values: np.ndarray

    basin_entropy: np.ndarray
    boundary_basin_entropy: np.ndarray
    uncertainty_exponent: np.ndarray | None

    runtime: np.ndarray
    settled_percentage: np.ndarray
