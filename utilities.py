import sys
import numpy as np

from config import SimulationParameters
from forces import net_force


def progress_bar(
        current_step: int,
        total: int,
        bar_length: int = 20,
        quantity: str = "simulation",
        beep: bool = True,
        run: bool = True
) -> None:
   """
    Display an updating text-based progress bar in the terminal.

    The progress bar is written to standard output and updated in-place,
    allowing long-running computations to report their completion status
    without printing a new line for each update.

    This utility function is primarily used to monitor the progress of
    computationally intensive tasks such as basin map generation,
    basin entropy calculations, and uncertainty exponent analysis.

    :param current_step: Number of completed iterations or completed units of work.
    :type current_step: int

    :param total: Total number of iterations or units of work.
    :type total: int

    :param bar_length: Length of the progress bar in characters.
        Default is 20.
    :type bar_length: int

    :param quantity: Short description of the task being performed
        (e.g., "basin entropy" or "uncertainty exponent").
        Default is "simulation".
    :type quantity: str

    :param beep: If True, plays a completion sound once the task reaches
        100% completion. On Windows this uses ``winsound.Beep()``;
        on other platforms the terminal bell character is printed.
        Default is True.
    :type beep: bool

   :return: None. Updates the progress display in the terminal.
   :rtype: None
   """

   if run:

       fraction = current_step / total
       filled = int(np.floor(fraction * bar_length))
       bar = "#" * filled + "-" * (bar_length - filled)
       sys.stdout.write(f"\rCalculating {quantity}: [{bar}] {fraction*100:.2f}%")
       sys.stdout.flush()

       if current_step == total:
           if sys.platform=="win32" and beep:
               import winsound
               winsound.Beep(500, 500) # Frequency in Hz, duration in milliseconds
           else:
               print("\a")

   else:
       pass

def settled_percentage(
        states: np.ndarray,
        magnet_positions: np.ndarray,
        parameters: SimulationParameters,
        forces: np.ndarray | None = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Determine which pendulum trajectories have reached equilibrium.

    A trajectory is classified as settled when both its speed and net
    acceleration fall below user-defined tolerances. The function also
    determines the nearest magnet to every trajectory, allowing each initial
    condition to be assigned to a basin of attraction.

    :param states:
        State vectors for all pendulum trajectories. The array has shape
        (4, N), where N is the number of trajectories. Rows correspond to
        [x, y, vx, vy].
    :type states: numpy.ndarray

    :param magnet_positions:
        Array of shape (M, 2) containing the Cartesian coordinates of the
        magnets.
    :type magnet_positions: numpy.ndarray

    :param parameters:
        Simulation parameters containing the physical constants and settling
        tolerances.
    :type parameters: SimulationParameters

    :return:
        A tuple containing:

        - percent_settled: Percentage of trajectories satisfying the settling
          criteria.
        - nearest_indices: Index of the nearest magnet for every trajectory.
        - settled: Boolean array indicating whether each trajectory has
          converged.
    :rtype: tuple(float, numpy.ndarray, numpy.ndarray)

    :note:
        Speed is computed as the Euclidean norm of the velocity vector, while
        acceleration is computed as the Euclidean norm of the net force
        (assuming unit mass).
    """

    x, y, vx, vy = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    speed = np.sqrt(vx**2 + vy**2)

    if forces is None:
        forces = net_force(
            states,
            magnet_positions,
            parameters.drag,
            parameters.magnet_strength,
            parameters.height,
            parameters.gravity
        )


    acceleration = np.sqrt(forces[:, 0]**2 + forces[:, 1]**2)

    dx = magnet_positions[:, 0][:, None] - x[None, :] # (M, N)
    dy = magnet_positions[:, 1][:, None] - y[None, :] # (M, N)
    dist2 = dx**2 + dy**2
    nearest_indices = np.argmin(dist2, axis=0)

    settled = (
        (speed < parameters.velocity_tol) &
        (acceleration < parameters.acceleration_tol)
    )

    percent_settled = float(100 * np.mean(settled))


    return percent_settled, nearest_indices, settled
