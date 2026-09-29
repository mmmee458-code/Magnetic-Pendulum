import matplotlib.pyplot as plt
import numpy as np
from config import SimulationParameters
from experiments import sweep_parameter


def compare_backends(
        parameters: SimulationParameters,
        parameter_name: str,
        parameter_values,
        calculate_uncertainty: bool = False
):
    """
    Run the same parameter sweep on CPU and CUDA
    and return both sets of results.
    """

    print("CALCULATING CPU IMPLEMENTATION DATA:")
    cpu_results = sweep_parameter(
        parameters=parameters,
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        backend="cpu",
        calculate_uncertainty=calculate_uncertainty,
    )

    print("CALCULATING GPU IMPLEMENTATION DATA")
    gpu_results = sweep_parameter(
        parameters=parameters,
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        backend="cuda",
        calculate_uncertainty=calculate_uncertainty,
    )

    return cpu_results, gpu_results


def plot_compare_backends(results):
    """
    Plot CPU and CUDA results from compare_backends().

    Parameters
    ----------
    results : tuple
        (cpu_results, gpu_results)
    """
    cpu_results, gpu_results = results

    # Calculating speedup
    cpu_runtime = np.asarray(cpu_results["runtime"])
    gpu_runtime = np.asarray(gpu_results["runtime"])
    speedups = cpu_runtime / gpu_runtime
    average_speedup = np.mean(speedups)

    x = cpu_results["values"]
    parameter_name = cpu_results["parameter"]

    uncertainty_calculated = (
            "alpha" in cpu_results
            and len(cpu_results["alpha"]) > 0
    )

    plot_count = 4 if uncertainty_calculated in cpu_results else 3

    fig, axes = plt.subplots(
        plot_count,
        1,
        figsize=(9,12),
        sharex=True
    )

    # Basin entropy
    axes[0].plot(
        x,
        cpu_results["Sb"],
        "o-",
        label="CPU"
    )

    axes[0].plot(
        x,
        gpu_results["Sb"],
        "o-",
        label="CUDA"
    )

    axes[0].set_ylabel(r"$S_b$")
    axes[0].set_title(
        f"Basin entropy vs. {parameter_name}"
    )
    axes[0].legend()
    axes[0].grid(True)

    # Boundary basin entropy

    axes[1].plot(
        x,
        cpu_results["Sbb"],
        "o-",
        label="CPU"
    )

    axes[1].plot(
        x,
        gpu_results["Sbb"],
        "o-",
        label="CUDA"
    )

    axes[1].set_ylabel(r"$S_{bb}$")
    axes[1].set_title(
        f"Boundary basin entropy vs. {parameter_name}"
    )
    axes[1].legend()
    axes[1].grid(True)

    # Uncertainty exponent (if calculated)

    if uncertainty_calculated:
        axes[2].plot(
            x,
            cpu_results["alpha"],
            "o-",
            label="CPU"
        )

        axes[2].plot(
            x,
            gpu_results["alpha"],
            "o-",
            label="CUDA"
        )

        axes[2].set_ylabel(r"$\alpha$")
        axes[2].set_title(
            f"Uncertainty exponent vs. {parameter_name}"
        )
        axes[2].legend()
        axes[2].grid(True)


    # Runtime plots

    runtime_index = 3 if uncertainty_calculated else 2
    axes[runtime_index].plot(
        x,
        cpu_results["runtime"],
        "o-",
        label="CPU"
    )
    axes[runtime_index].plot(
        x,
        gpu_results["runtime"],
        "o-",
        label="CUDA"
    )
    axes[runtime_index].set_ylabel("Runtime (s)")
    axes[runtime_index].set_title(
        f"Runtime vs. {parameter_name}"
        f"(average GPU speedup: {average_speedup:.2f}x)"
    )
    axes[runtime_index].legend()
    axes[runtime_index].grid(True)

    axes[-1].set_xlabel(parameter_name)



    plt.tight_layout()
    plt.show()
