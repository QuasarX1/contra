# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .._ParticleType import ParticleType
from ._CatalogueBase import CatalogueBase
from ._SnapshotEAGLE import SnapshotEAGLE

import numpy as np
from unyt import unyt_array

class CatalogueSUBFIND(CatalogueBase):
    """
    SUBFIND catalogue data (EAGLE).
    """

    def __init__(
                    self,
                    filepath: str,
                    snapshot: SnapshotEAGLE,
                ) -> None:
        super().__init__(
            filepath = filepath,
            snapshot = snapshot
        )
    
    def _get_number_of_haloes(self) -> int:
        pass#TODO:

    def get_particle_IDs(self, particle_type: ParticleType) -> np.ndarray:
        pass#TODO:

    def get_halo_IDs(self, particle_type: ParticleType) -> np.ndarray:
        pass#TODO:

    def get_halo_masses(self, particle_type: ParticleType) -> unyt_array:
        pass#TODO:
