from dataclasses import dataclass


@dataclass
class SimulationParameters:
    """
    Stores all physical and numerical parameters required for a magnetic
    pendulum simulation.

    This dataclass acts as a centralized configuration object that is passed
    explicitly to simulation functions. Replacing global variables with a
    configuration object makes simulations reproducible, configurable, easier
    to test, and allows multiple simulations with different parameter sets to
    coexist.

    :param drag:
        Damping coefficient controlling the velocity-dependent drag force.
        Larger values dissipate kinetic energy more rapidly.
    :type drag: float

    :param magnet_strength:
        Strength parameter controlling the magnitude of the magnetic force
        exerted by each magnet.
    :type magnet_strength: float

    :param epsilon:
        Dimensionless perturbation magnitude applied to the magnet positions
        relative to their perfectly symmetric circular arrangement. A value of
        zero preserves rotational symmetry.
    :type epsilon: float

    :param height:
        Vertical separation between the pendulum bob and the magnet plane.
        This prevents the magnetic force from becoming singular as the bob
        approaches a magnet.
    :type height: float

    :param gravity:
        Strength of the restoring force toward the origin. This models the
        pendulum's gravitational tendency to return to its equilibrium
        position.
    :type gravity: float

    :param radius:
        Radius of the square region over which initial pendulum positions are
        generated.
    :type radius: float

    :param dt:
        Time step used by the velocity Verlet integration scheme.
    :type dt: float

    :param max_time:
        Maximum simulated physical time before the simulation is considered
        to have failed to converge.
    :type max_time: float

    :param colors:
        List of colors used to visualize attractor basins corresponding to
        different magnets.
    :type colors: list[str]

    :param acceleration_tol:
        Maximum acceleration magnitude for a trajectory to be considered
        settled.
    :type acceleration_tol: float

    :param velocity_tol:
        Maximum velocity magnitude for a trajectory to be considered settled.
    :type velocity_tol: float
    """

    # Physical Parameters

    drag: float
    magnet_strength: float
    height: float
    gravity: float

    # Geometry
    radius: float
    magnet_radius_factor: float
    magnet_count: int
    epsilon: float
    magnet_seed: int

    # Numerical Parameters
    dt: float
    max_time: float
    acceleration_tol: float
    velocity_tol: float


    # Basin-map Parameters
    resolution: int
    colors: list[str]
