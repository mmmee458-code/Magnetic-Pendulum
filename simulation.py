from measurements import basin_entropy, uncertainty_exponent
from utilities import settled_percentage
from magnet_locations import magnet_locations
from backends import get_simulator


def run_simulation(result):
    parameters = result.parameters
    result.magnet_positions = magnet_locations(parameters)
    resolution = parameters.resolution


    simulator = get_simulator(result.backend)

    states, runtime = simulator(
            result.magnet_positions,
            parameters,
            resolution=resolution
    )

    result.states = states
    result.runtime = runtime


    percent_settled, basins, settled = settled_percentage(
        result.states,
        result.magnet_positions,
        result.parameters
    )

    result.settled_percentage = percent_settled
    result.basins = basins
    result.settled = settled

    # Basin entropy

    Sbb, Sb = basin_entropy(
        result.basins,
        parameters.resolution
    )

    result.basin_entropy = Sb
    result.boundary_basin_entropy = Sbb


def run_uncertainty_exponent(result):
    (uncertainty_exponent_,
     intercept,
     r_squared,
     log_delta,
     log_fractions,
     settled_percentage_values
     ) = uncertainty_exponent(
        result.magnet_positions,
        result.parameters,
        backend=result.backend
    )

    result.uncertainty_exponent = uncertainty_exponent_
    result.intercept = intercept
    result.r_squared = r_squared
    result.log_delta = log_delta
    result.log_fractions = log_fractions
    result.uncertainty_settled_percentage = settled_percentage_values
