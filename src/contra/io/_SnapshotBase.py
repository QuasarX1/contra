# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .._ParticleType import ParticleType

from abc import ABC, abstractmethod
from typing import Awaitable, Union, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
from unyt import unyt_array

class SnapshotBase(ABC):
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

        self.__n_parts: Dict[ParticleType, int] = self._get_number_of_particles()

    @property
    def filepath(self) -> float:
        return self.__filepath

    @property
    def redshift(self) -> float:
        return self.__redshift
    @property
    def z(self) -> float:
        return self.redshift

    @property
    def hubble_param(self) -> float:
        return self.__hubble_param
    @property
    def h(self) -> float:
        return self.hubble_param

    @property
    def expansion_factor(self) -> float:
        return self.__expansion_factor
    @property
    def a(self) -> float:
        return self.expansion_factor

    def remove_h_factor(self, data: np.ndarray) -> np.ndarray:
        return data / self.h

    def make_h_less(self, data: np.ndarray) -> np.ndarray:
        return data * self.h

    def to_physical(self, data: Union[np.ndarray, unyt_array]) -> Union[np.ndarray, unyt_array]:
        return data * self.a

    def to_comoving(self, data: Union[np.ndarray, unyt_array]) -> Union[np.ndarray, unyt_array]:
        return data / self.a

    @abstractmethod
    def _get_number_of_particles(self) -> Dict[ParticleType, int]:
        """
        Called by constructor.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
    def number_of_particles(self, particle_type: ParticleType) -> int:
        return self.__n_parts[particle_type]

    @abstractmethod
    def get_IDs(self, particle_type: ParticleType) -> np.ndarray:
        raise NotImplementedError("Attempted to call an abstract method.")

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
    
    def get_volumes(self, particle_type: ParticleType) -> unyt_array:
        return self.get_smoothing_lengths(particle_type)**3 * (np.pi * (4/3))

    @abstractmethod
    def get_black_hole_subgrid_masses(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_black_hole_dynamical_masses(self) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_positions(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    @abstractmethod
    def get_velocities(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    def get_sfr(self, particle_type: ParticleType) -> unyt_array:
        if particle_type != ParticleType.gas:
            raise ValueError("get_sfr is not supported for particle type other than gas.")
        return self._get_sfr(particle_type)
    @abstractmethod
    def _get_sfr(self, particle_type: ParticleType) -> unyt_array:
        raise NotImplementedError("Attempted to call an abstract method.")

    # async versions

    async def get_IDs_async(self, particle_type: ParticleType) -> Awaitable[np.ndarray]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_IDs, particle_type)
#        return self.get_IDs(particle_type)

    async def get_smoothing_lengths_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_smoothing_lengths, particle_type)
#        return self.get_smoothing_lengths(particle_type)

    async def get_masses_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_masses, particle_type)
#        return self.get_masses(particle_type)

    async def get_black_hole_subgrid_masses_async(self) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_black_hole_subgrid_masses)
#        return self.get_black_hole_subgrid_masses()

    async def get_black_hole_dynamical_masses_async(self) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_black_hole_dynamical_masses)
#        return self.get_black_hole_dynamical_masses()

    async def get_positions_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_positions, particle_type)
#        return self.get_positions(particle_type)

    async def get_velocities_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_velocities, particle_type)
#        return self.get_velocities(particle_type)

    async def get_sfr_async(self, particle_type: ParticleType) -> Awaitable[unyt_array]:
        with ThreadPoolExecutor() as pool:
            return await asyncio.get_running_loop().run_in_executor(pool, self.get_sfr, particle_type)
#        return self.get_sfr(particle_type)
