import numpy as np
from config import SimulationParameters

def magnet_locations(
        parameters: SimulationParameters
) -> np.ndarray:
    """
    Generate magnet positions arranged approximately uniformly on a circle.

    Magnets are initially arranged at equally spaced angular positions
    on a circle. Rotational symmetry is therefore exact when epsilon = 0.

    Symmetry breaking is introduced by independently displacing each
    magnet by a distance epsilon * radius in a random direction.

    The perturbation parameter is therefore dimensionless:
        epsilon = perturbation_distance / radius

    :return:
        Array of shape ``(number_of_magnets, 2)`` where each row contains the
        Cartesian coordinates ``[x, y]`` of one magnet.
    :rtype: numpy.ndarray

    :raises ValueError:
        If ``number_of_magnets`` is less than or equal to zero.

    Example:
>>> magnet_locations(1, 5)
array([[0.8, 0.],
      [0.2472136, 0.76084521],
      [-0.6472136, 0.4702282],
      [-0.6472136, -0.4702282],
      [0.2472136, -0.76084521]])
   """
    magnet_count = parameters.magnet_count
    seed = parameters.magnet_seed
    epsilon = parameters.epsilon
    scaling_factor = parameters.magnet_radius_factor
    rad = parameters.radius

    if magnet_count <= 0:
        raise ValueError("number_of_magnets must be positive")

    if rad <= 0:
        raise ValueError("radius must be positive")

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    if scaling_factor <= 0:
        raise ValueError("magnet_radius_factor must be positive")

    rng = np.random.default_rng(seed)
    magnet_radius = scaling_factor * rad

    # Equally spaced angular positions
    angles = np.arange(magnet_count) * (
        2.0 * np.pi / magnet_count
    )

    # Perfectly symmetric magnet_positions
    locations = np.empty(
        (magnet_count, 2),
        dtype=np.float32
    )

    locations[:, 0] = (magnet_radius * np.cos(angles))
    locations[:, 1] = (magnet_radius * np.sin(angles))

    # Symmetry-breaking perturbation

    perturbation_distance = epsilon * rad

    if perturbation_distance > 0:
        displacement_angles = rng.uniform(
            0.0,
            2.0 * np.pi,
            magnet_count
        )

        locations[:, 0] += (
            perturbation_distance
            * np.cos(displacement_angles)
        )

        locations[:, 1] += (
            perturbation_distance
            * np.sin(displacement_angles)
        )

    return locations
