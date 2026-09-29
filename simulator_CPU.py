import numpy as np
import time
from numba import njit, prange
from config import SimulationParameters
from utilities import progress_bar, settled_percentage
from results import SimulationResult
from forces import net_force

@njit(fastmath=True)
def simulator_step(
        states: np.ndarray,
        magnet_positions: np.ndarray,
        drag: float,
        magnet_strength: float,
        height: float,
        gravity: float,
        dt: float
) -> np.ndarray:
   """
    Advance the simulation by one time step using the velocity Verlet
    integration scheme.

    Computes the current accelerations from the net force, updates the
    pendulum positions, recomputes the accelerations at the new positions,
    and then updates the velocities using the average of the old and new
    accelerations.

    The update equations are
        x(t + Δt) = x(t) + v(t)Δt + ½a(t)Δt²

        v(t + Δt) = v(t) + ½[a(t) + a(t + Δt)]Δt

    Assumes a unit pendulum mass so that acceleration equals the net force.

    :param states:
        State vectors for all pendulum trajectories. The array has shape
        (4, N), where N is the number of trajectories. Rows correspond to
        [x, y, vx, vy].
    :type states: numpy.ndarray

    :param magnet_positions:
        Array of shape (M, 2) containing the Cartesian coordinates of the
        magnets.
    :type magnet_positions: numpy.ndarray

    :param drag:
        Drag coefficient controlling the damping force.
    :type drag: float

    :param magnet_strength:
        Strength parameter controlling the magnetic force.
    :type magnet_strength: float

    :param height:
        Vertical distance between the pendulum bob and the magnet plane.
    :type height: float

    :param gravity:
        Restoring-force coefficient toward the origin.
    :type gravity: float

    :param dt:
        Numerical integration time step.
    :type dt: float

    :return: None

    :note:
        Velocity Verlet is a second-order accurate integration method with
        significantly better energy behavior than the forward Euler method.
   """

   accelerations = net_force(states,
                             magnet_positions,
                             drag,
                             magnet_strength,
                             height,
                             gravity)

   states[:, 0] += states[:, 2] * dt + 0.5 * accelerations[:, 0] * dt**2
   states[:, 1] += states[:, 3] * dt + 0.5 * accelerations[:, 1] * dt**2

   new_accelerations = net_force(states,
                             magnet_positions,
                             drag,
                             magnet_strength,
                             height,
                             gravity)

   states[:, 2] += 0.5 * (accelerations[:, 0]+new_accelerations[:, 0]) * dt
   states[:, 3] += 0.5 * (accelerations[:, 1]+new_accelerations[:, 1]) * dt

   return new_accelerations

def simulator(
        magnet_positions: np.ndarray,
        parameters: SimulationParameters,
        resolution: int | None = None,
        settling_threshold: float = 99,
        initial_states: np.ndarray | None = None,
        bar: bool | None = True
) -> tuple[np.ndarray, float]:
  """
    Integrate the magnetic pendulum equations of motion until the system
    reaches equilibrium.

    This function initializes a collection of pendulum states and repeatedly
    advances them using the velocity Verlet integrator. Integration continues
    until a specified percentage of trajectories satisfy the settling criteria
    or the maximum simulation time is exceeded.

    The initial conditions may be generated automatically from a uniform
    Cartesian grid or supplied explicitly.

    :param magnet_positions:
        Array of shape (M, 2) containing the Cartesian coordinates of the
        magnets.
    :type magnet_positions: numpy.ndarray

    :param parameters:
        Simulation parameters containing the physical constants, numerical
        integration settings, and convergence tolerances.
    :type parameters: SimulationParameters

    :param resolution:
        Number of grid points along each axis used to generate initial
        conditions. A value of ``resolution`` produces
        ``resolution²`` pendulum trajectories. This parameter must be
        provided if ``initial_states`` is omitted.
    :type resolution: int, optional

    :param settling_threshold:
        Percentage of trajectories that must satisfy the settling criteria
        before integration terminates.
    :type settling_threshold: float, optional

    :param initial_states:
        Optional array of shape (4, N) containing user-supplied initial
        conditions. If provided, ``resolution`` must be omitted.
    :type initial_states: numpy.ndarray, optional

    :return:
        Final state vectors for all trajectories. The returned array has
        shape (4, N), where the rows correspond to
        [x, y, vx, vy].
    :rtype: numpy.ndarray

    :raises ValueError:
        If neither ``resolution`` nor ``initial_states`` is supplied, or if
        both are supplied simultaneously.

    :raises RuntimeError:
        If the required settling percentage is not reached before the maximum
        simulation time is exceeded.

    :note:
        The simulation checks for convergence every fixed number of
        integration steps rather than after every step to reduce the
        computational overhead of evaluating the settling criteria.
  """
  start_time = time.perf_counter()

  if resolution is None and initial_states is None:
      raise ValueError(
          "Either resolution or initial states must be provided"
      )
  if resolution is not None and initial_states is not None:
      raise ValueError(
          "Provide either resolution or initial_states, not both"
      )

  if resolution is not None and resolution <= 0:
      raise ValueError("resolution must be positive")

  if initial_states is None:
      radius = parameters.radius
      xs = np.linspace(-radius,
                       radius,
                       resolution)

      ys = np.linspace(-radius,
                       radius,
                       resolution)

      X, Y = np.meshgrid(xs, ys)

      x_pos, y_pos = (X.ravel().astype(np.float32),
                      Y.ravel().astype(np.float32))

      x_vel, y_vel = (np.zeros_like(x_pos,
                                   dtype=np.float32),
                      np.zeros_like(y_pos,
                                    dtype=np.float32))

      states = (np.column_stack([x_pos,
                                y_pos,
                                x_vel,
                                y_vel])
                .astype(np.float32))

  else:
      states = initial_states.copy()

  CHECK_INTERVAL = 500
  settled = 0
  iterations = 0
  accelerations = None

  max_iterations = int(
      parameters.max_time / (parameters.dt * CHECK_INTERVAL)
  )

  while settled < settling_threshold and iterations < max_iterations:

    for _ in range(CHECK_INTERVAL):
      accelerations = simulator_step(states,
                         magnet_positions,
                         parameters.drag,
                         parameters.magnet_strength,
                         parameters.height, parameters.gravity,
                         parameters.dt)

    iterations +=1

    settled, _, _ = settled_percentage(
        states,
        magnet_positions,
        parameters,
        forces=accelerations
    )

    if bar:
        print(f"\rSettled: {settled:.2f}%", end="", flush=True)

        progress_bar(np.floor(settled),
                     100,
                     quantity="Settled Trajectories")

  if settled < settling_threshold:
    raise RuntimeError("Simulation failed to converge")

  runtime = time.perf_counter() - start_time

  return states, runtime
