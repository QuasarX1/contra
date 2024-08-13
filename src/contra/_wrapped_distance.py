# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
import numpy as np

def calculate_wrapped_displacement(from_positions: np.ndarray, to_positions: np.ndarray, box_width: float) -> np.ndarray:
    """
    Calculates the true displacement vector between two points in a periodic box.
    This assumes that the true displacement is always the shortest distance between two points.
    If the two points are part of a timeseries dataset, the value of dt must be sufficiently small such that the particles cannot travel more than half the length of the box.
    """
    position_deltas = to_positions - from_positions
    deltas_needing_wrapping = np.abs(position_deltas) > box_width / 2
    position_deltas[deltas_needing_wrapping] = position_deltas[deltas_needing_wrapping] - (np.sign(position_deltas[deltas_needing_wrapping]) * box_width)
    del deltas_needing_wrapping
    return position_deltas

def calculate_wrapped_distance(from_position: np.ndarray, to_positions: np.ndarray, box_width: float, do_squared_distance = False) -> np.ndarray:
    """
    Calculates the length of the true displacement vector between two points in a periodic box.
    This assumes that the true displacement is always the shortest distance between two points.
    If the two points are part of a timeseries dataset, the value of dt must be sufficiently small such that the particles cannot travel more than half the length of the box.
    """
    displacement = calculate_wrapped_displacement(from_position, to_positions, box_width)
    squared_distance = (displacement**2).sum(axis = 1 if ((len(from_position.shape) > 1) or (len(to_positions.shape) > 1)) else 0)
    del displacement
    if do_squared_distance:
        return squared_distance
    else:
        return np.sqrt(squared_distance)
