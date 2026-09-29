from simulator_CPU import simulator
from simulator_GPU import simulator_CUDA


def get_simulator(backend):
    if backend == "cpu":
        return simulator
    elif backend == "cuda":
        return simulator_CUDA
    else:
        raise ValueError(f"Unknown backend: {backend}")
