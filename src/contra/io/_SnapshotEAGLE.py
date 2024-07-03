# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .._ParticleType import ParticleType
from ._SnapshotBase import SnapshotBase

from typing import Dict

import numpy as np
from unyt import unyt_array
from h5py import File as HDF5_File
from pyread_eagle import EagleSnapshot
from QuasarCode import Settings

class SnapshotEAGLE(SnapshotBase):
    """
    EAGLE snapshot data.
    """

    @staticmethod
    def make_reader_object(filepath: str) -> EagleSnapshot:
        """
        Create an EagleSnapshot instance.
        """
        return EagleSnapshot(filepath, Settings.verbose)

    def __init__(self, filepath: str) -> None:
        hdf5_reader = HDF5_File(filepath)

        with HDF5_File(filepath, "r") as hdf5_reader:
            redshift = hdf5_reader["Header"].attrs["Redshift"]#TODO: check path
            hubble_param = hdf5_reader["Header"].attrs["HubbleParam"]
            expansion_factor = hdf5_reader["Header"].attrs["ExpansionFactor"]

        self.__file_object = SnapshotEAGLE.make_reader_object(filepath)

        super().__init__(
            filepath = filepath,
            redshift = redshift,
            hubble_param = hubble_param,
            expansion_factor = expansion_factor
        )

    def _get_number_of_particles(self) -> Dict[ParticleType, int]:
        pass#TODO:

    def get_IDs(self, particle_type: ParticleType) -> np.ndarray:
        return self.__file_object.read_dataset(particle_type.value, "ParticleID")

    def get_smoothing_lengths(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.remove_h_factor(self.__file_object.read_dataset(particle_type.value, "SmoothingLength")), units = "Mpc")#TODO: find correct path name and are smoothing lengths h-less?

    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.__file_object.read_dataset(particle_type.value, "Mass"), units = "Msun*(10**10)")#TODO: DM particles???

    def get_black_hole_subgrid_masses(self) -> unyt_array:
        return unyt_array(self.__file_object.read_dataset(ParticleType.black_hole.value, "SubgridMass"), units = "Msun*(10**10)")#TODO: find correct path name

    def get_positions(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.remove_h_factor(self.__file_object.read_dataset(particle_type.value, "Positions")), units = "Mpc")#TODO: find correct path name

    def get_velocities(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.remove_h_factor(self.__file_object.read_dataset(particle_type.value, "Velocities")), units = "km/s")#TODO: find correct path name and unit and check that hubble param conversion works for non-Mpc units!

    def _get_sfr(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.__file_object.read_dataset(particle_type.value, "StarFormationRate"), units = "Msun*(10**10)/Gyr")#TODO: check path and units
