from collections.abc import Callable
from functools import wraps

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from QuasarCode import Settings

__all__ = (
    "plot_hexbin",
    "create_hexbin_colour_function",
    "create_hexbin_count",
    "create_hexbin_log10_count",
    "create_hexbin_sum",
    "create_hexbin_log10_sum",
    "create_hexbin_sum_log10",
    "create_hexbin_mean",
    "create_hexbin_weighted_mean",
    "create_hexbin_median",
    "create_hexbin_percentile"
)



def create_hexbin_colour_function(statistic: Callable[[np.ndarray], float]):
    """
    Create a function for calculating the value for colour mapping of hexbin cells.

    The return value of this function can be passed to `plot_hexbin` as an argument for the parameter `colour_function`.
    """

    def ready_hexbin_colour_function(min_value: float|None = None, max_value: float|None = None, default_bin_value: float = -np.inf):

        @wraps(statistic)
        def calculate_bin_colour(indices: np.ndarray, /) -> float:
            if len(indices) == 0:
                return default_bin_value
            try:
                result = statistic(indices)
            except Exception as e:
                if Settings.debug and Settings.verbose:
                    raise e
                return default_bin_value
            if min_value is not None and result < min_value:
                return min_value
            elif max_value is not None and result > max_value:
                return max_value
            else:
                return result

        return calculate_bin_colour
    
    return ready_hexbin_colour_function



def create_hexbin_count():
    """
    Assign hexbin cell colour to the number of elements that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_mean(indices: np.ndarray, /) -> float:
        return len(indices)
    return calculate_statistic_mean



def create_hexbin_log10_count():
    """
    Assign hexbin cell colour to the log_10 of the number of elements that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_mean(indices: np.ndarray, /) -> float:
        return np.log10(len(indices))
    return calculate_statistic_mean



def create_hexbin_fraction(mask: np.ndarray[tuple[int], np.dtype[np.bool_]]):
    """
    Assign hexbin cell colour to the fraction of of elements that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_fraction(indices: np.ndarray, /) -> float:
        return mask[indices].sum() / len(indices)
    return calculate_statistic_fraction



def create_hexbin_quantity_fraction(data: np.ndarray[tuple[int], np.dtype[np.floating]], mask: np.ndarray[tuple[int], np.dtype[np.bool_]]):
    """
    Assign hexbin cell colour to the fraction of of elements that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_quantity_fraction(indices: np.ndarray, /) -> float:
        subset = data[indices]
        return subset[mask[indices]].sum() / subset.sum()
    return calculate_statistic_quantity_fraction



def create_hexbin_sum(data: np.ndarray):
    """
    Assign hexbin cell colour to the sum of the elements from this dataset that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_mean(indices: np.ndarray, /) -> float:
        return data[indices].sum()
    return calculate_statistic_mean



def create_hexbin_log10_sum(data: np.ndarray):
    """
    Assign hexbin cell colour to the log_10 of the sum of the elements from this dataset that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_mean(indices: np.ndarray, /) -> float:
        return np.log10(data[indices].sum())
    return calculate_statistic_mean



def create_hexbin_sum_log10(data: np.ndarray):
    """
    Assign hexbin cell colour to the sum of log_10 of the the elements from this dataset that fall within the bin.

    Note, it is advised you use `create_hexbin_sum` and pass the `np.log10(data)` to it instead of using this function, unless retaining a second copy of the data array is problematic.
    """
    @create_hexbin_colour_function
    def calculate_statistic_mean(indices: np.ndarray, /) -> float:
        return np.log10(data[indices]).sum()
    return calculate_statistic_mean



def create_hexbin_mean(data: np.ndarray, offset: float = 0.0):
    """
    Assign hexbin cell colour to the mean of elements from this dataset that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_mean(indices: np.ndarray, /) -> float:
        return np.mean(data[indices]) + offset
    return calculate_statistic_mean



def create_hexbin_log10_mean(data: np.ndarray, offset: float = 0.0):
    """
    Assign hexbin cell colour to the mean of elements from this dataset that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_mean(indices: np.ndarray, /) -> float:
        return np.log10(np.mean(data[indices])) + offset
    return calculate_statistic_mean



def create_hexbin_weighted_mean(data: np.ndarray, weights: np.ndarray, offset: float = 0.0):
    """
    Assign hexbin cell colour to the mean of elements from this dataset that fall within the bin.

    Use the weights from an array of equal length.
    """
    @create_hexbin_colour_function
    def calculate_statistic_weighted_mean(indices: np.ndarray, /) -> float:
        return np.average(data[indices], weights = weights[indices]) + offset
    return calculate_statistic_weighted_mean



def create_hexbin_log10_weighted_mean(data: np.ndarray, weights: np.ndarray, offset: float = 0.0):
    """
    Assign hexbin cell colour to the mean of elements from this dataset that fall within the bin.

    Use the weights from an array of equal length.
    """
    @create_hexbin_colour_function
    def calculate_statistic_weighted_mean(indices: np.ndarray, /) -> float:
        return np.log10(np.average(data[indices], weights = weights[indices])) + offset
    return calculate_statistic_weighted_mean



def test_function__create_hexbin_log10_weighted_mean(data: np.ndarray, weights: np.ndarray, offset: float = 0.0):
    """
    Assign hexbin cell colour to the mean of elements from this dataset that fall within the bin.

    Use the weights from an array of equal length.
    """
    @create_hexbin_colour_function
    def calculate_statistic_weighted_mean(indices: np.ndarray, /) -> float:
        return np.log10(np.sum(data[indices] * weights[indices]) / np.sum(weights[indices]) / offset)
    return calculate_statistic_weighted_mean



def create_hexbin_median(data: np.ndarray):
    """
    Assign hexbin cell colour to the median of elements from this dataset that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_median(indices: np.ndarray, /) -> float:
        return np.median(data[indices])
    return calculate_statistic_median



def create_hexbin_percentile(data: np.ndarray, percentile: float):
    """
    Assign hexbin cell colour to the specified percentile of elements from this dataset that fall within the bin.
    """
    @create_hexbin_colour_function
    def calculate_statistic_percentile(indices: np.ndarray, /) -> float:
        return np.percentile(data[indices], percentile)
    return calculate_statistic_percentile



def plot_hexbin(
    x: np.ndarray,
    y: np.ndarray,
    colour_function: Callable[[float|None, float|None, float|None], Callable[[np.ndarray], float]]|None = None,
    vmin: float|None = None,
    vmax: float|None = None,
    vdefault: float = -np.inf,
    cmap = "viridis",
    alpha_values: np.ndarray[tuple[int], np.dtype[np.float64]]|None = None,
    gridsize: int | tuple[int, int] = 500,
    axis = None,
    edgecolor = None,
    **kwargs
) -> PolyCollection:
    """
    Plot a hexbin using a pre-defined colour scheme.

    kwargs will be passed to the `plt.hexbin` function call (or that of the provided axis).

    `colour_function` will default to `create_hexbin_count` if set as `None`.
    """

    hexes = (axis if axis is not None else plt).hexbin(
        x = x,
        y = y,
        C = np.arange(len(x)),
        reduce_C_function = (colour_function if colour_function is not None else create_hexbin_count())(vmin, vmax, vdefault),
        gridsize = gridsize,
        cmap = cmap,
        vmin = vmin,
        vmax = vmax,
        edgecolor = edgecolor,
        **kwargs
    )

    if alpha_values is not None:
        hexes.set_alpha(alpha_values)

    return hexes
