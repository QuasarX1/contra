# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .._ParticleType import ParticleType

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from unyt import unyt_array

class SnapshotBase(object):
    """
    Base class type for snapshot data reader types.
    """

    def __init__(
                    self,
                    filepath: str,
                    redshift: float,
                    hubble_param: float,
                    expansion_factor: float
                ) -> None:
        self.__filepath: str = filepath
        self.__redshift: float = redshift
        self.__hubble_param: float = hubble_param
        self.__expansion_factor: float = expansion_factor

    @property
    def filepath(self) -> float:
        return self.__filepath

    @property
    def redshift(self) -> float:
        return self.__redshift

    @property
    def hubble_param(self) -> float:
        return self.__hubble_param
    @property
    def h(self) -> float:
        return self.__hubble_param

    @property
    def expansion_factor(self) -> float:
        return self.__expansion_factor
    @property
    def a(self) -> float:
        return self.__expansion_factor

    def remove_h_factor(self, data: np.ndarray) -> np.ndarray:
        return data / self.h

    def make_h_less(self, data: np.ndarray) -> np.ndarray:
        return data * self.h

    def to_physical(self, data: Union[np.ndarray, unyt_array]) -> Union[np.ndarray, unyt_array]:
        return data * self.a

    def to_comoving(self, data: Union[np.ndarray, unyt_array]) -> Union[np.ndarray, unyt_array]:
        return data / self.a

    @abstractmethod
    def get_smoothing_lengths(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_masses(self, particle_type: ParticleType) -> unyt_array:
        if particle_type == ParticleType.black_hole:
            raise ValueError("get_masses is not supported for black hole particle as they lack a simple mass field.")
        return self._get_masses(particle_type)
    @abstractmethod
    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_black_hole_subgrid_masses(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_positions(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_velocities(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")
