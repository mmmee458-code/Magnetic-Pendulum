from numba import njit, prange, cuda
import numpy as np
from numpy import float32
import math


@njit(parallel=True, fastmath=True)
def net_force(
        states: np.ndarray,
        magnet_positions: np.ndarray,
        drag: float,
        magnet_strength: float,
        height: float,
        gravity: float
) -> np.ndarray:
   """
    Calculate the net force acting on each pendulum.

    Computes the total force on every pendulum state by summing the magnetic
    attraction from all magnets, the velocity-dependent drag force, and the
    restoring gravitational force toward the origin. The computation is fully
    vectorized, allowing forces to be evaluated simultaneously for all
    trajectories.

    :param states:
        State vectors for all pendulum trajectories. The array has shape
        (N, 4), where N is the number of trajectories. Each row contains
        [x, y, vx, vy].
    :type states: numpy.ndarray
    :param magnet_positions:
        Array of shape (M, 2) containing the Cartesian coordinates of the
        magnets.
    :type magnet_positions: numpy.ndarray
    :param drag:
        Drag coefficient controlling the damping force proportional to
        velocity.
    :type drag: float
    :param magnet_strength:
        Strength parameter controlling the magnitude of the magnetic force.
    :type magnet_strength: float
    :param height:
        Vertical distance between the pendulum bob and the magnet plane.
    :type height: float
    :param gravity:
        Restoring-force coefficient acting toward the origin.
    :type gravity: float

    :return:
        Array of shape (2, N) containing the net force components
        [Fx, Fy] for every pendulum trajectory.
    :rtype: numpy.ndarray

    :note:
    The magnetic force exerted by each magnet scales as
    r/(r² + h²)^(3/2), where h is the vertical separation between the
    pendulum and the magnets. A small constant is added to the planar
    distance computation to avoid division-by-zero when a pendulum lies
    directly above a magnet.
   """

   x_pos, y_pos = states[:, 0], states[:, 1]
   x_vel, y_vel = states[:, 2], states[:, 3]

   M = magnet_positions.shape[0] # Number of magnets
   N = x_pos.shape[0] # Number of pendulum points

   net_forces = np.empty((N, 2), dtype=np.float32)

   h2 = height * height

   # Loop over each magnet, calculating its contribution
   for j in prange(N):
       x = x_pos[j]
       y = y_pos[j]

       fx = np.float32(0.0)
       fy = np.float32(0.0)

       for i in range(M):

           dx = magnet_positions[i, 0] - x
           dy = magnet_positions[i, 1] - y
           r2 = (
               dx * dx
               + dy * dy
               + h2
               + np.float32(1e-8)
           )


           inv_r = np.float32(1.0) / np.sqrt(r2)
           inv_r2 = inv_r * inv_r
           inv_r3 = inv_r2 * inv_r

           force = magnet_strength * inv_r3

           fx += force * dx
           fy += force * dy

       # Drag force (damping)
       fx -= x_vel[j] * drag
       fy -= y_vel[j] * drag

       # Spring-like gravitational force
       fx -= x * gravity
       fy -= y * gravity

       net_forces[j, 0] = fx
       net_forces[j, 1] = fy

   return net_forces


@cuda.jit(cache=True, fastmath=True)
def net_force_cuda(
        states,
        magnet_positions,
        drag,
        magnet_strength,
        height,
        gravity,
        net_forces
):
    j = cuda.grid(1)

    N = states.shape[0]
    M = magnet_positions.shape[0]

    if j < N:
        x, y = states[j, 0], states[j, 1]
        vx, vy = states[j, 2], states[j, 3]

        fx = float32(0.0)
        fy = float32(0.0)

        h2 = height * height

        for i in range(M):
            dx = magnet_positions[i, 0] - x
            dy = magnet_positions[i, 1] - y

            r2 = dx * dx + dy * dy + h2 + float32(1e-8)

            inv_r = float32(1.0) / math.sqrt(r2)
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r2 * inv_r

            force = magnet_strength * inv_r3

            fx += force * dx
            fy += force * dy

        # drag
        fx -= vx * drag
        fy -= vy * drag

        # gravity
        fx -= x * gravity
        fy -= y * gravity

        net_forces[j, 0] = fx
        net_forces[j, 1] = fy
