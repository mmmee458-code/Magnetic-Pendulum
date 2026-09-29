import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import SimulationParameters
from magnet_locations import magnet_locations
from simulation import run_simulation, run_uncertainty_exponent
from results import SimulationResult


def plot_basin_map(
        result: SimulationResult
) -> None:
   """
   Generate and display a magnetic pendulum basin map visualization.

   This function creates a basin map by integrating trajectories from a square grid
   of initial positions in the pendulum plane. Each initial condition is assigned
   a final attractor based on the magnet it converges to after simulation. The
   resulting basin classification is displayed as a color map, where each color
   corresponds to a different attracting magnet.

   The function also calculates the basin entropy (Sb) and boundary basin entropy
   (Sbb) from the resulting basin map. It displays whether Sbb exceeds ln(2),
   which is a sufficient condition for fractal basin boundaries according to
   Daza et al. (2016).

   Magnet positions are generated on a circular arrangement with optional angular
   perturbations controlled by the simulation parameter epsilon.

   :param resolution: Number of grid points along each spatial axis.
       The total number of simulated trajectories is resolution^2.
   :type resolution: int

   :param magnet_count: Number of magnets in the magnetic pendulum system.
   :type magnet_count: int

   :param parameters: Simulation parameters containing physical constants,
       numerical integration settings, convergence tolerances, and visualization
       settings.
   :type parameters: SimulationParameters

   :return: None

   :raises ValueError:
       If resolution or magnet_count are not positive, or there are too few magnet colors in
       parameters.colors to display the basin.
   """
   resolution = result.resolution
   parameters = result.parameters
   magnet_count = parameters.magnet_count

   if resolution <= 0:
      raise ValueError("resolution must be positive")

   if magnet_count <= 0:
      raise ValueError("magnet_count must be positive")

   if len(parameters.colors) < magnet_count:
      raise ValueError(
         "Number of colors must be at least equal to magnet_count"
      )

   if result.magnet_positions is None:
      magnet_positions = magnet_locations(parameters)
      result.magnet_positions = magnet_positions


   if result.boundary_basin_entropy is None:
       run_simulation(result)

   basins = result.basins
   percent_settled = result.settled_percentage

   fig, ax = plt.subplots(figsize=(14, 16))

   fig.suptitle(f"Magnetic Pendulum Simulation - {len(result.magnet_positions)} Magnets - {"CUDA" if result.backend else "CPU"}",
                fontsize=14,
                fontweight='bold',
                color='black')

   # Basin Map

   basin_map = basins.reshape(resolution, resolution)
   radius = parameters.radius
   ax.set_aspect("equal")
   cmap = mcolors.ListedColormap(parameters.colors[:result.parameters.magnet_count])

   ax.imshow(basin_map,
              origin="lower",
              cmap=cmap,
              extent=(-radius, radius, -radius, radius),
              interpolation="nearest")

   ax.set_title(f"\nDrag={parameters.drag:.4f}, Magnet Strength={parameters.magnet_strength:.2f}, Resolution={resolution} x {resolution}"
                f"\nSettled Trajectories: {percent_settled:.2f}%")

   ax.set_xlabel("x")
   ax.set_ylabel("y")

   # Plotting Magnet labels
   for i, (x, y) in enumerate(result.magnet_positions):
      ax.scatter(x,
                  y,
                  s=400,
                  alpha=0.65,
                  color="black")

      ax.text(x,
               y,
               f"{i + 1}",
               color="white",
               ha="center",
               va="center",
               fontsize=10)

   # Calculates and displays the basin entropy
   Sbb = result.boundary_basin_entropy
   Sb = result.basin_entropy

   if Sbb > np.log(2):
       fig.text(0.5,
                .02,
                f"Boundary basin entropy is {Sbb:.4f}"
                f"\nSince this is larger than {np.log(2):.4f}, the boundaries are fractal.",
                ha="center",
                va="bottom",
                fontsize=10,
                style="italic")

   else:
       fig.text(0.5,
                .02,
                f"Boundary basin entropy is {Sbb:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                style="italic")

   plt.subplots_adjust(bottom=0.15, hspace=0.3)
   plt.show()

def plot_uncertainty_exponent(result):

    if result.uncertainty_exponent is None:
        run_uncertainty_exponent(result)


    plt.figure(figsize=(8,6))


    # fitted line
    fit_log = (
            result.log_delta * result.uncertainty_exponent
            + result.intercept)

    fit = np.exp(fit_log)
    delta = np.exp(result.log_delta)
    fractions = np.exp(result.log_fractions)

    plt.loglog(
        delta,
        fractions,
        "o",
        label=f"Simulation Data"
    )

    plt.loglog(
        delta,
        fit,
        "-",
        label=f"Fit: exponent = {result.uncertainty_exponent:.4f}"
    )

    plt.xlabel(r"$\delta$")
    plt.ylabel(r"$f(\delta)$")

    plt.title(
        f"Uncertainty Exponent\n"
        f"R²={result.r_squared:.4f}"
    )

    plt.legend()
    plt.grid(True)
    plt.show()
