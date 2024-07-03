# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from ._config import ContraSettings

from typing import Dict, Callable

import numpy as np
from scipy.interpolate import interp1d
from unyt import unyt_quantity

def get_L_star_mass_of_z() -> Callable[[float], float]:
    #TODO: fix singleton
    #line_points: Dict[float, float] = ContraSettings().L_star_mass_of_z
    line_points: Dict[float, float] = { # in Msun
            0.1  : 10**12,
            0.5  : 10**12.1,
            1.0  : -1,
            2.0  : -1,
            3.0  : -1,
            4.0  : -1,
            5.0  : -1,
            6.0  : -1,
            7.0  : -1,
            8.0  : -1,
            9.0  : -1,
            10.0 : -1
        }
    interpolation_partial = interp1d(np.array(line_points.keys()), np.array(line_points.values()))
    def L_star_mass_of_z(z: float) -> float:
        return interpolation_partial(np.array([z]))[0]
    return L_star_mass_of_z

def get_L_star_halo_mass_of_z() -> Callable[[float], unyt_quantity]:
    #TODO: fix singleton
    #line_points: Dict[float, float] = ContraSettings().L_star_halo_mass_of_z
    line_points: Dict[float, float] = { # in Msun
            0.0  : 10**12,
            0.1  : 10**12,
            1.0  : 10**12.05,
            2.0  : 10**12.15,
            3.0  : 10**12.2,
            4.0  : 10**12.16,
            5.0  : 10**12.1,
            6.0  : 10**12,
            7.0  : 10**11.8,
            8.0  : 10**11.2,
            9.0  : 10**11.1,
            10.0 : 10**10.8,
            100.0 : 10**10.8,
        }#TODO: THESE ARE APPROXIMATE VALUES - DO NOT USE!!!
    interpolation_partial = interp1d(np.array(list(line_points.keys())), np.array(list(line_points.values())))
    def L_star_halo_mass_of_z(z: float) -> unyt_quantity:
        return unyt_quantity(interpolation_partial(np.array([z]))[0], units = "Msun")
    return L_star_halo_mass_of_z
