# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .._ParticleType import ParticleType
from ._SnapshotBase import SnapshotBase

from unyt import unyt_array
import swiftsimio as sw

class SnapshotSWIFT(SnapshotBase):
    """
    SWIFT snapshot data.
    """

    @staticmethod
    def make_reader_object(filepath: str) -> object:#TODO: find type
        """
        Create an swiftsimio instance.
        """
        return sw.load(filepath)

    def __init__(self, filepath: str):
        self.__file_object = SnapshotSWIFT.make_reader_object(filepath)

        super().__init__(#TODO: check retreval of info
            filepath = filepath,
            redshift = self.__file_object.redshift,
            hubble_param = self.__file_object.h,
            expansion_factor = self.__file_object.a
        )

    def get_smoothing_lengths(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).smoothing_lengths

    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).masses#TODO: DM particles???

    def get_black_hole_subgrid_masses(self) -> unyt_array:
        return self.__file_object.black_hole.subgrid_masses#TODO: find correct path name

    def get_positions(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).coordinates

    def get_velocities(self, particle_type: ParticleType) -> unyt_array:
        return particle_type.get_SWIFT_dataset(self.__file_object).velocities
