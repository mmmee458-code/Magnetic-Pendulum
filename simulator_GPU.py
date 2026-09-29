from config import SimulationParameters
import time
import numpy as np
from numba import cuda
from utilities import progress_bar, settled_percentage
from forces import net_force_cuda


@cuda.jit
def update_position(states,
                    forces,
                    dt):
    j = cuda.grid(1)

    if j < states.shape[0]:

        states[j, 0] += states[j, 2]*dt + 0.5*forces[j, 0]*dt*dt
        states[j, 1] += states[j, 3]*dt + 0.5*forces[j, 1]*dt*dt


@cuda.jit
def update_velocity(states,
                    old_forces,
                    new_forces,
                    dt):

    j = cuda.grid(1)

    if j < states.shape[0]:
        states[j, 2] += 0.5*(old_forces[j, 0]+new_forces[j, 0])*dt
        states[j, 3] += 0.5*(old_forces[j, 1]+new_forces[j, 1])*dt


def simulator_step_cuda(
        d_states,
        d_magnet_positions,
        drag,
        magnet_strength,
        height,
        gravity,
        dt,
        d_old_forces,
        d_new_forces,
        blocks,
        threads
):

    # 1. Calculate forces at current position
    net_force_cuda[blocks, threads](
        d_states,
        d_magnet_positions,
        drag,
        magnet_strength,
        height,
        gravity,
        d_old_forces
    )

    # 2. Update position using old forces
    update_position[blocks, threads](
        d_states,
        d_old_forces,
        dt
    )

    # 3. Calculate forces at new position
    net_force_cuda[blocks, threads](
        d_states,
        d_magnet_positions,
        drag,
        magnet_strength,
        height,
        gravity,
        d_new_forces
    )

    # 4. Update velocity using average of old and new forces
    update_velocity[blocks, threads](
        d_states,
        d_old_forces,
        d_new_forces,
        dt
    )

    cuda.synchronize()

def simulator_CUDA(
        magnet_positions: np.ndarray,
        parameters: SimulationParameters,
        resolution: int | None = None,
        settling_threshold: float = 99,
        initial_states: np.ndarray | None = None,
        bar: bool | None = True
) -> np.ndarray:

    if resolution is None and initial_states is None:
        raise ValueError(
            "Either resolution or initial states must be provided"
        )

    if resolution is not None and initial_states is not None:
        raise ValueError(
            "Provide either resolution or initial states, not both"
        )

    start_time = time.perf_counter()

    # -------------------------
    # Initialize states
    # -------------------------

    if initial_states is None:
        radius = parameters.radius

        xs = np.linspace(
            -radius,
            radius,
            resolution
        )

        ys = np.linspace(
            -radius,
            radius,
            resolution
        )

        X, Y = np.meshgrid(xs, ys)

        x_pos = X.ravel().astype(np.float32)
        y_pos = Y.ravel().astype(np.float32)

        x_vel = np.zeros_like(
            x_pos,
            dtype=np.float32
        )

        y_vel = np.zeros_like(
            y_pos,
            dtype=np.float32
        )


        states = np.column_stack([
            x_pos,
            y_pos,
            x_vel,
            y_vel
        ]).astype(np.float32)


    else:
        states = initial_states.copy().astype(
            np.float32
        )

    N = states.shape[0]

    # -------------------------
    # GPU memory allocation
    # -------------------------

    d_states = cuda.to_device(states)
    d_magnets = cuda.to_device(
        np.ascontiguousarray(
            magnet_positions,
            dtype=np.float32
        )
    )

    d_old_forces = cuda.device_array(
        (N,2),
        dtype=np.float32
    )

    d_new_forces = cuda.device_array(
        (N,2),
        dtype=np.float32
    )

    threads = 256
    blocks = (
        N + threads - 1
    ) // threads

    # -------------------------
    # Integration loop
    # -------------------------

    CHECK_INTERVAL = 500

    max_iterations = int(
        parameters.max_time / (parameters.dt * CHECK_INTERVAL)
    )

    settled = 0
    iterations = 0


    while (
        settled < settling_threshold
        and iterations < max_iterations
    ):


        for _ in range(CHECK_INTERVAL):

            simulator_step_cuda(
                d_states,
                d_magnets,
                parameters.drag,
                parameters.magnet_strength,
                parameters.height,
                parameters.gravity,
                parameters.dt,
                d_old_forces,
                d_new_forces,
                blocks,
                threads
            )

        cuda.synchronize()


        # Copy only for convergence test

        states_host = d_states.copy_to_host()

        settled, _, _ = settled_percentage(
            states_host,
            magnet_positions,
            parameters,
        )

        if bar:
            progress_bar(
                np.floor(settled),
                100,
                quantity="Settled Trajectories"
            )


        iterations += 1


    if settled < settling_threshold:
        raise RuntimeError(
            "Simulation failed to converge"
        )

    # Final GPU -> CPU copy
    runtime = time.perf_counter() - start_time
    states = d_states.copy_to_host()

    return states, runtime
