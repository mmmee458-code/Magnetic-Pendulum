from config import SimulationParameters
from basin_visualizations import plot_basin_map, plot_uncertainty_exponent
from numpy import float32
from results import SimulationResult
from experiments import sweep_parameter, plot_parameter_sweep, save_sweep_results
import numpy as np
from backends_comparison import compare_backends, plot_compare_backends

parameters = SimulationParameters(
    drag=float32(0.1),
    magnet_strength=float32(0.8),

    height=float32(0.25),
    gravity=float32(0.1),

    radius=float32(1.5),
    magnet_radius_factor=float32(0.8),
    magnet_count=3,
    epsilon=float32(0.0),
    magnet_seed=42,

    dt=float32(0.01),
    max_time=float32(350),
    acceleration_tol=float32(0.01),
    velocity_tol=float32(0.005),

    resolution=500,
    colors=["red", "yellow", "black", "purple", "turquoise"],
)


epsilon_values = np.linspace(0.0, 0.5, 20)

if __name__ == "__main__":
    results1 = sweep_parameter(
        parameters=parameters,
        parameter_name="epsilon",
        parameter_values=epsilon_values,
        backend="cuda",
    )

    save_sweep_results(
        results1,
        "results/epsilon_sweep1.csv"
    )

    plot_parameter_sweep(results1)
