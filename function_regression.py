import numpy as np
from scipy.optimize import curve_fit


def linear_model(x, a, b):
    return a + b * x

def quadratic_model(x, a, b, c):
    return a + b * x + c * x**2

def cubic_model(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

def logarithmic_model(x, a, b):
    return a + b * np.log(x)

def power_model(x, a, b):
    return a * x**b

def saturating_model(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c



def calculate_aic_bic(y, y_pred, num_parameters):
    """
    Calculate AIC and BIC for a fitted regression model.

    Parameters
    ----------
    y : array-like
        Observed values.
    y_pred : array-like
        Values predicted by the fitted model.

    num_parameters : int
        Number of fitted parameters in the model.

    Returns
    -------
    aic : float
        Akaike Information Criterion.

    bic : float
        Bayesian Information Criterion.

    rss : float
        Residual sum of squares.
    """

    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y.shape != y_pred.shape:
        raise ValueError(
            "y and y_pred must have the same shape"
        )

    if y.ndim != 1:
        raise ValueError(
            "y and y_pred must be one-dimensional"
        )

    if len(y) == 0:
        raise ValueError(
            "y and y_pred cannot be empty"
        )

    if num_parameters < 1:
        raise ValueError(
            "num_parameters must be at least 1"
        )

    residuals = y - y_pred
    rss = np.sum(residuals ** 2)

    n = len(y)

    if rss <= 0:
        raise ValueError(
            "RSS must be greater than zero"
        )

    aic = n * np.log(rss / n) + 2 * num_parameters
    bic = n * np.log(rss / n) + num_parameters * np.log(n)

    return aic, bic, rss


def compare_models(x, y):
    """
    Fit several candidate models and compare them using
    R², RSS, AIC, and BIC.

    Parameters
    ----------
    x : array-like
        Independent variable.

    y : array-like
        Observed dependent variable.

    Returns
    -------
    results : dict
        Regression results for each candidate model.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            "x and y must have the same shape"
        )

    if x.ndim != 1:
        raise ValueError(
            "x and y must be one-dimensional"
        )

    if len(x) < 4:
        raise ValueError(
            "At least 4 data points are required"
        )

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(
            "x and y must contain only finite values"
        )

    models = {
        "linear": linear_model,
        "quadratic": quadratic_model,
        "cubic": cubic_model,
        "exponential": exponential_model,
        "logarithmic": logarithmic_model,
        "power": power_model,
        "saturating": saturating_model,
    }

    results = {}

    for name, model in models.items():

        # Logarithmic and power models require x > 0
        if name in ("logarithmic", "power"):
            if np.any(x <= 0):
                results[name] = {
                    "error": "Model requires x > 0"
                }
                continue

        try:
            parameters, _ = curve_fit(
                model,
                x,
                y,
                maxfev=10000
            )

            y_pred = model(x, *parameters)

            residuals = y - y_pred
            rss = np.sum(residuals ** 2)

            tss = np.sum(
                (y - np.mean(y)) ** 2
            )

            if tss == 0:
                raise ValueError(
                    "y must contain more than one unique value"
                )

            r_squared = 1 - rss / tss

            num_parameters = len(parameters)

            aic, bic, _ = calculate_aic_bic(
                y,
                y_pred,
                num_parameters
            )

            results[name] = {
                "parameters": parameters,
                "predicted": y_pred,
                "rss": rss,
                "r_squared": r_squared,
                "aic": aic,
                "bic": bic,
            }

        except (RuntimeError, ValueError, OverflowError) as error:

            results[name] = {
                "error": str(error)
            }

    return results
