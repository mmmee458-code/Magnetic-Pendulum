import numpy as np
import matplotlib.pyplot as plt
import csv


from dataclasses import replace
from config import SimulationParameters
from results import SimulationResult
from simulation import run_simulation, run_uncertainty_exponent
from pathlib import Path


def sweep_parameter(
        parameters: SimulationParameters,
        parameter_name: str,
        parameter_values,
        backend: str="cuda",
        calculate_uncertainty: bool = False,
):
    """
    Sweep one SimulationParameters value and calculate
    Sb, Sbb, and the uncertainty exponent for each value.
    """

    if not hasattr(parameters, parameter_name):
        raise ValueError(
            f"'{parameter_name}' is not a valid SimulationParameters field"
        )

    parameter_values = np.asarray(parameter_values)

    if parameter_values.size == 0:
        raise ValueError("parameter_values cannot be empty")

    results = {
        "parameter": parameter_name,
        "values": parameter_values,
        "Sb": [],
        "Sbb": [],
        "alpha": [],
        "r_squared": [],
        "settled_percentage": [],
        "runtime": [],
    }

    for value in parameter_values:

        print(
            f"\nRunning {parameter_name} = {value}"
        )

        # Create a copy with only the selected parameter changed
        sweep_parameters = replace(
            parameters,
            **{parameter_name: value}
        )

        result = SimulationResult(
            parameters=sweep_parameters,
            resolution=sweep_parameters.resolution,
            backend=backend,
        )

        # Basin simulation
        run_simulation(result)

        results["Sb"].append(result.basin_entropy)
        results["Sbb"].append(result.boundary_basin_entropy)
        results["settled_percentage"].append(
            result.settled_percentage
        )
        results["runtime"].append(result.runtime)

        # Uncertainty exponent
        if calculate_uncertainty:
            run_uncertainty_exponent(result)

            results["alpha"].append(
                result.uncertainty_exponent
            )

            results["r_squared"].append(
                result.r_squared
            )

    # Convert lists to numpy arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.asarray(results[key])

    return results


def plot_parameter_sweep(results):

    x = results["values"]
    parameter_name = results["parameter"]

    uncertainty_calculated = (
        "alpha" in results
        and len(results["alpha"]) > 0
    )

    plot_count = 3 if uncertainty_calculated else 2

    fig, axes = plt.subplots(
        plot_count,
        1,
        figsize=(9, 14),
        sharex=True
    )

    axes[0].plot(x, results["Sb"], "o-")
    axes[0].set_ylabel(r"$S_b$")
    axes[0].set_title(
        f"Basin entropy vs. {parameter_name}"
    )
    axes[0].grid(True)

    axes[1].plot(x, results["Sbb"], "o-")
    axes[1].set_ylabel(r"$S_{bb}$")
    axes[1].set_title(
        f"Boundary basin entropy vs. {parameter_name}"
    )
    axes[1].grid(True)

    if uncertainty_calculated:
        axes[2].plot(x, results["alpha"], "o-")
        axes[2].set_ylabel(r"$\alpha$")
        axes[2].set_title(
            f"Uncertainty exponent vs. {parameter_name}"
        )
        axes[2].grid(True)

    axes[-1].set_xlabel(parameter_name)

    plt.tight_layout()
    plt.show()


def save_sweep_results(results, filename):
    """
    Save parameter sweep results to a CSV file

    :param results: dict
        Results returned by sweep_parameter()

    :param filename: str or Path
        Output CSV file path.
    """

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "parameter",
        "value",
        "Sb",
        "Sbb",
        "alpha",
        "r_squared",
        "settled_percentage",
        "runtime"
    ]

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(columns)

        parameter_name = results["parameter"]
        values = results["values"]

        for i, value in enumerate(values):

            alpha = (
                results["alpha"][i]
                if len(results["alpha"])>0
                else ""
            )

            r_squared = (
                results["r_squared"][i]
                if len(results["r_squared"]) > 0
                else ""
            )

            writer.writerow([
                parameter_name,
                value,
                results["Sb"][i],
                results["Sbb"][i],
                alpha,
                r_squared,
                results["settled_percentage"][i],
                results["runtime"][i]

            ])
