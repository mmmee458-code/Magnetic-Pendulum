import numpy as np
from scipy.stats import linregress
from config import SimulationParameters
from utilities import settled_percentage, progress_bar
from backends import get_simulator

def uncertainty_exponent(
        magnet_positions: np.ndarray,
        parameters: SimulationParameters,
        num_samples: int = 10000,
        max_delta: float = 0.3,
        delta_count: int = 30,
        seed: int = 42,
        backend: str="cuda"
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
   """
   Calculate the uncertainty exponent for a magnetic pendulum basin map.

   The uncertainty exponent quantifies how sensitive final attractor outcomes
   are to small perturbations in initial conditions. It is commonly used to
   characterize fractal basin boundaries in nonlinear dynamical systems.

   For a perturbation magnitude δ, the fraction of uncertain trajectories is
   expected to follow a power-law relationship:

       f(δ) ~ δ^α

   where α is the uncertainty exponent. Smaller values of α indicate more
   complex and fractal basin boundaries, while α = 1 corresponds to smooth
   basin boundaries.

   The method samples random initial conditions, creates perturbed copies by
   applying random displacements of varying magnitudes, integrates both sets
   of trajectories, and measures the fraction of trajectories that converge to
   different attractors among trajectories where both the original and perturbed
   simulations reach equilibrium. A linear regression on the logarithm of the
   perturbation size and uncertainty fraction is then used to estimate α.

   The uncertainty exponent is widely used in the study of fractal basin boundaries
   and was developed within the framework of chaotic scattering and nonlinear
   dynamical systems, including the work of Grebogi, Ott, and Yorke on fractal
   basin boundaries.

   :param magnet_positions: Coordinates of all magnets in the system.
       Each row contains the x and y position of one magnet.
       Shape: (M, 2), where M is the number of magnets.
   :type magnet_positions: numpy.ndarray

   :param parameters: Simulation parameters containing physical constants,
       numerical integration settings, and convergence tolerances.
   :type parameters: SimulationParameters

   :param num_samples: Number of initial conditions sampled.
   :type num_samples: int

   :param max_delta: Maximum perturbation magnitude.
   :type max_delta: float

   :param delta_count: Number of perturbation magnitudes sampled.
   :type delta_count: int

   :param seed: Random seed used to generate reproducible initial conditions
       and perturbation directions.
   :type seed: int

   :param use_cuda:
    If True, use the CUDA/GPU simulator. If False, use the CPU simulator.
   :type use_cuda: bool

   :return: Tuple containing:
       uncertainty_exponent_ (float): Estimated uncertainty exponent α, obtained from the
           relationship f(δ) ~ δ^α.
       intercept (float): Intercept of the log-log regression.
       r_squared (float): Coefficient of determination for the regression fit.
       log_delta (numpy.ndarray): Logarithm of perturbation magnitudes used
           in the regression.
       log_fractions (numpy.ndarray): Logarithm of uncertainty fractions used
           in the regression.
       settled_percentage_values (numpy.ndarray): Fraction of trajectories
           that successfully reached a settled state for each perturbation size.
   :rtype: tuple[float, float, float, numpy.ndarray, numpy.ndarray, numpy.ndarray]

   :raises ValueError:
       If `num_samples` is less than or equal to zero, `max_delta` is less than
       or equal to zero, `delta_count` is less than two, or the simulation radius
       is not positive.

   :raises RuntimeError:
       If fewer than two valid perturbation values remain after filtering,
       preventing calculation of the log-log regression.
   """

   if num_samples <= 0:
       raise ValueError("num_samples must be positive")

   if max_delta <= 0:
       raise ValueError("max_delta must be positive")

   if delta_count < 2:
       raise ValueError("delta_count must be at least 2")

   if parameters.radius <= 0:
       raise ValueError("parameters.radius must be positive")

   radius = parameters.radius

   delta_values = np.logspace(-6, np.log10(max_delta / parameters.radius), delta_count) * radius
   uncertainty_fractions = []

   step_ = 0

   rng = np.random.default_rng(seed)

   samples = np.empty((num_samples, 4), dtype=np.float32)
   samples[:, 0] = rng.uniform(-radius, radius, num_samples)  # X-values
   samples[:, 1] = rng.uniform(-radius, radius, num_samples)  # Y-values
   samples[:, 2] = 0  # X-Velocities = 0
   samples[:, 3] = 0  # Y-Velocities = 0

   settled_percentage_values = []


   run = get_simulator(backend)

   # Integrate original initial conditions once.

   states_original, _ = run(
       magnet_positions,
       parameters,
       initial_states=samples.copy())

   _, basin_original, settled_original = settled_percentage(
       states_original,
       magnet_positions,
       parameters
   )


   for perturbation_size in delta_values:

       # Create empty arrays
       perturbed_samples = np.empty((num_samples, 4), dtype=np.float32)

       # Create perturbed samples
       angles = rng.uniform(0, 2*np.pi, num_samples)
       perturbed_samples[:, 0] = samples[:, 0] + perturbation_size * np.cos(angles)
       perturbed_samples[:, 1] = samples[:, 1] + perturbation_size * np.sin(angles)
       perturbed_samples[:, 2], perturbed_samples[:, 3] = 0, 0

       # Integrating samples and perturbed_samples

       states_perturbed, _ = run(
           magnet_positions,
           parameters,
           initial_states=perturbed_samples.copy())

       _, basin_perturbed, settled_perturbed = settled_percentage(states_perturbed,
                                                                    magnet_positions,
                                                                    parameters)

       valid = (settled_original & settled_perturbed)
       settled_percentage_values.append(float(np.mean(valid)))

       if not np.any(valid):
           uncertainty_fractions.append(np.nan)

       else:
           uncertain = (basin_original[valid] != basin_perturbed[valid])
           uncertainty_fraction = np.mean(uncertain)

           uncertainty_fractions.append(uncertainty_fraction)

       step_ += 1

       progress_bar(step_,
                    delta_count,
                    quantity="uncertainty exponent",
                    beep=True)

   settled_percentage_values = np.array(settled_percentage_values)



   # Remove uncertainty values =0, since they will result in log = -infinity
   valid_mask = (
       (np.array(uncertainty_fractions) > 0) &
       np.isfinite(uncertainty_fractions)
   )

   delta_values_filtered = delta_values[valid_mask]
   uncertainty_fractions_filtered = np.array(uncertainty_fractions)[valid_mask]

   log_delta = np.log(delta_values_filtered)
   log_fractions = np.log(uncertainty_fractions_filtered)

   if len(log_delta) < 2:
       raise RuntimeError(
           "Insufficient valid perturbation values to calculate uncertainty exponent."
       )

   uncertainty_exponent_, intercept, r_value, p_value, std_err = linregress(log_delta, log_fractions)
   r_squared = r_value**2


   return (uncertainty_exponent_,
           intercept,
           r_squared,
           log_delta,
           log_fractions,
           settled_percentage_values
   )


def basin_entropy(
        basin: np.ndarray,
        resolution: int,
        box_length: int = 5
) -> tuple[float, float]:
   """
   Calculate basin entropy and boundary basin entropy for a magnetic pendulum
   basin map using the method introduced by Daza et al. (2016).

   Basin entropy measures the unpredictability of final attractor outcomes by
   calculating the Gibbs entropy within boxes of the basin map. The boundary
   basin entropy (Sbb) considers only boxes containing multiple attractors and
   quantifies the complexity of basin boundaries.

   For each box, the Gibbs entropy is calculated as:

       S_i = -Σ p_j * log(p_j)

   where p_j is the fraction of pixels in the box belonging to attractor j.

   A sufficient condition for fractal basin boundaries is:

       Sbb > ln(2)

   as established in:
   "Basin Entropy: A New Tool to Characterize Unpredictability in Dynamical
   Systems" by Daza et al. (2016).

   The basin map is partitioned into square boxes of side length `box_length`.
   The average entropy over all boxes gives the basin entropy (Sb), while the
   average entropy over only boxes containing multiple attractors gives the
   boundary basin entropy (Sbb).

   :param basin: One-dimensional array containing final attractor indices for
       each initial condition.
       Shape: (resolution^2,).
   :type basin: numpy.ndarray

   :param resolution: Resolution of the square basin map. The basin array is
       reshaped into a resolution × resolution grid.
   :type resolution: int

   :param box_length: Side length of each square entropy box in pixels.
       Optional; default = 5
   :type box_length: int

   :return:
       Tuple containing:
       Sbb (float): Boundary basin entropy, averaged over boxes containing
           multiple attractors.
       Sb (float): Basin entropy, averaged over all boxes.
   :rtype: tuple[float, float]

    :raises ValueError:
       If `resolution` or `box_length` are not positive, if the basin array
       length does not equal resolution squared, or if `resolution` is not
       divisible by `box_length`

   :note:
       The logarithm uses natural base e. Therefore, entropy values are measured
       in nats.
   """

   if resolution <= 0:
       raise ValueError("resolution must be positive")

   if box_length <= 0:
       raise ValueError("box_length must be positive")

   if basin.size != resolution**2:
       raise ValueError("basin size must equal resolution squared")

   if resolution % box_length != 0:
       raise ValueError("resolution must be divisible by box_length")

   basin_2d = basin.reshape(resolution, resolution)
   entropy_sum = 0.0
   boundary_box_count = 0
   box_count = 0

   total_boxes = (resolution // box_length) ** 2
   completed_boxes = 0

   for i in range(0, resolution - box_length + 1, box_length):

       for j in range(0, resolution - box_length + 1, box_length):

           box = basin_2d[i:i+box_length, j:j+box_length]
           _, counts = np.unique(box, return_counts=True)

           if len(counts) > 1:
               boundary_box_count += 1

           total = box.size
           probabilities = counts / total
           entropy_sum -= np.sum(probabilities * np.log(probabilities))

           box_count += 1
           completed_boxes += 1

           progress_bar(completed_boxes, total_boxes, quantity="Basin entropy")

   Sb = entropy_sum/box_count

   if boundary_box_count == 0:
       Sbb = 0.0
   else:
       Sbb = entropy_sum / boundary_box_count


   return Sbb, Sb
