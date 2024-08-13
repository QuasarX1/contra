# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None

from typing import Union, List, Tuple, Dict
import re
import os

import numpy as np
from unyt import unyt_array, unyt_quantity
from h5py import File as HDF5_File
from pyread_eagle import EagleSnapshot
from QuasarCode import Settings, Console

from .._ParticleType import ParticleType
from ._SnapshotBase import SnapshotBase

class SnapshotEAGLE(SnapshotBase):
    """
    EAGLE snapshot data.
    """

    @staticmethod
    def make_reader_object(filepath: str) -> EagleSnapshot:
        """
        Create an EagleSnapshot instance.
        """
        Console.print_debug("Creating pyread_eagle object:")
        return EagleSnapshot(filepath, Settings.debug)

    def __init__(self, filepath: str) -> None:
        Console.print_debug(f"Loading EAGLE snapshot from: {filepath}")

        pattern = re.compile(r'.*sn[ai]pshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]sn[ai]p_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
        match = pattern.match(filepath)
        if not match:
            raise ValueError(f"Snapshot filepath \"{filepath}\" does not conform to the naming scheme of an EAGLE snapshot. EAGLE snapshot files must have a clear snapshot number component in both the folder and file names.")
        snap_num = match.group("number")

        hdf5_reader = HDF5_File(filepath)

        with HDF5_File(filepath, "r") as hdf5_reader:
            redshift = hdf5_reader["Header"].attrs["Redshift"]
            hubble_param = hdf5_reader["Header"].attrs["HubbleParam"]
            expansion_factor = hdf5_reader["Header"].attrs["ExpansionFactor"]
            omega_baryon = hdf5_reader["Header"].attrs["OmegaBaryon"]
            self.__number_of_particles = hdf5_reader["Header"].attrs["NumPart_Total"]
            self.__dm_mass_internal_units = hdf5_reader["Header"].attrs["MassTable"][1]
            self.__box_size_internal_units = hdf5_reader["Header"].attrs["BoxSize"]
            self.__length_h_exp: float = hdf5_reader["PartType1/Coordinates"].attrs["h-scale-exponent"]
            self.__length_a_exp: float = hdf5_reader["PartType1/Coordinates"].attrs["aexp-scale-exponent"]
            self.__length_cgs_conversion_factor: float = hdf5_reader["PartType1/Coordinates"].attrs["CGSConversionFactor"]
            try:
                self.__mass_h_exp: float = hdf5_reader["PartType0/Mass"].attrs["h-scale-exponent"]
                #self.__mass_a_exp: float = hdf5_reader["PartType0/Mass"].attrs["aexp-scale-exponent"]
                self.__mass_cgs_conversion_factor: float = hdf5_reader["PartType0/Mass"].attrs["CGSConversionFactor"]
            except:
                # Just in case there aren't any gas particles, use the expected values for EAGLE
                self.__mass_h_exp: float = -1.0
                #self.__mass_a_exp: float = 0.0
                self.__mass_cgs_conversion_factor: 1.989E43
            self.__velocity_h_exp: float = hdf5_reader["PartType1/Velocity"].attrs["h-scale-exponent"]
            self.__velocity_a_exp: float = hdf5_reader["PartType1/Velocity"].attrs["aexp-scale-exponent"]
            self.__velocity_cgs_conversion_factor: float = hdf5_reader["PartType1/Velocity"].attrs["CGSConversionFactor"]

            self.__cgs_unit_conversion_factor_density: float = hdf5_reader["Units"].attrs["UnitDensity_in_cgs"]
            self.__cgs_unit_conversion_factor_energy: float = hdf5_reader["Units"].attrs["UnitEnergy_in_cgs"]
            self.__cgs_unit_conversion_factor_length: float = hdf5_reader["Units"].attrs["UnitLength_in_cm"]
            self.__cgs_unit_conversion_factor_mass: float = hdf5_reader["Units"].attrs["UnitMass_in_g"]
            self.__cgs_unit_conversion_factor_pressure: float = hdf5_reader["Units"].attrs["UnitPressure_in_cgs"]
            self.__cgs_unit_conversion_factor_time: float = hdf5_reader["Units"].attrs["UnitTime_in_s"]
            self.__cgs_unit_conversion_factor_velocity: float = hdf5_reader["Units"].attrs["UnitVelocity_in_cm_per_s"]

            assert self.__length_cgs_conversion_factor == self.__cgs_unit_conversion_factor_length
            assert self.__mass_cgs_conversion_factor == self.__cgs_unit_conversion_factor_mass
            assert self.__velocity_cgs_conversion_factor == self.__cgs_unit_conversion_factor_velocity

        self.__file_object = SnapshotEAGLE.make_reader_object(filepath)
        Console.print_debug("Calling pyread_eagle object's select_region method:")
        self.__file_object.select_region(0.0, self.__box_size_internal_units, 0.0, self.__box_size_internal_units, 0.0, self.__box_size_internal_units)

        super().__init__(
            filepath = filepath,
            number = snap_num,
            redshift = redshift,
            hubble_param = hubble_param,
            omega_baryon = omega_baryon,
            expansion_factor = expansion_factor,
            box_size = unyt_array(np.array([self.__box_size_internal_units, self.__box_size_internal_units, self.__box_size_internal_units], dtype = float) * (hubble_param ** self.__length_h_exp) * self.__length_cgs_conversion_factor, units = "cm").to("Mpc")
        )

    @property
    def _file_object(self) -> EagleSnapshot:
        return self.__file_object
    
    def make_cgs_data(self, cgs_units: str, data: np.ndarray, h_exp: float, cgs_conversion_factor: float, a_exp: float = 0.0) -> unyt_array:
        """
        Convert raw data to a unyt_array object with the correct units.
        To retain data in co-moving space, omit the value for "a_exp".
        """
        return unyt_array(data * (self.h ** h_exp) * (self.a ** a_exp) * cgs_conversion_factor, units = cgs_units)
    
    def _convert_to_cgs_length(self, data: np.ndarray, propper = False) -> unyt_array:
        return self.make_cgs_data(
            "cm",
            data,
            h_exp = self.__length_h_exp,
            cgs_conversion_factor = self.__length_cgs_conversion_factor,
            a_exp = self.__length_a_exp if propper else 0.0
        )
    
    def _convert_to_cgs_mass(self, data: np.ndarray) -> unyt_array:
        return self.make_cgs_data(
            "g",
            data,
            h_exp = self.__mass_h_exp,
            cgs_conversion_factor = self.__mass_cgs_conversion_factor
        )
    
    def _convert_to_cgs_velocity(self, data: np.ndarray, propper = False) -> unyt_array:
        return self.make_cgs_data(
            "cm/s",
            data,
            h_exp = self.__velocity_h_exp,
            cgs_conversion_factor = self.__velocity_cgs_conversion_factor,
            a_exp = self.__velocity_a_exp if propper else 0.0
        )

    def _get_number_of_particles(self) -> Dict[ParticleType, int]:
        return { p : int(self.__number_of_particles[p.value]) for p in ParticleType.get_all() }

    def get_IDs(self, particle_type: ParticleType) -> np.ndarray:
        return self.__file_object.read_dataset(particle_type.value, "ParticleIDs")

    def get_smoothing_lengths(self, particle_type: ParticleType) -> unyt_array:
        return self._convert_to_cgs_length(self.__file_object.read_dataset(particle_type.value, "SmoothingLength")).to("Mpc")

    def _get_masses(self, particle_type: ParticleType) -> unyt_array:
        if particle_type == ParticleType.dark_matter:
            return self._convert_to_cgs_mass(np.full(self.number_of_particles(ParticleType.dark_matter), self.__dm_mass_internal_units)).to("Msun")
        return self._convert_to_cgs_mass(self.__file_object.read_dataset(particle_type.value, "Mass")).to("Msun")

    def get_black_hole_subgrid_masses(self) -> unyt_array:
        return self._convert_to_cgs_mass(self.__file_object.read_dataset(ParticleType.black_hole.value, "BH_Mass")).to("Msun")

    def get_black_hole_dynamical_masses(self) -> unyt_array:
        return self._convert_to_cgs_mass(self.__file_object.read_dataset(ParticleType.black_hole.value, "Mass")).to("Msun")

    def get_positions(self, particle_type: ParticleType) -> unyt_array:
        return self._convert_to_cgs_length(self.__file_object.read_dataset(particle_type.value, "Coordinates")).to("Mpc")

    def get_velocities(self, particle_type: ParticleType) -> unyt_array:
        return self._convert_to_cgs_velocity(self.__file_object.read_dataset(particle_type.value, "Velocity")).to("km/s")

    #TODO: check units and come up with a better way of doing the unit assignment and conversion
    def _get_sfr(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.__file_object.read_dataset(particle_type.value, "StarFormationRate"), units = "Msun/yr")

    def _get_metalicities(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.__file_object.read_dataset(particle_type.value, "Metallicity"), units = None)

    def _get_densities(self, particle_type: ParticleType) -> unyt_array:
        return self.make_cgs_data(
            "g/cm**3",
            self.__file_object.read_dataset(particle_type.value, "Density"),
            h_exp = 2.0,
            cgs_conversion_factor = self.__cgs_unit_conversion_factor_density
        ).to("Msun/Mpc**3")

    def _get_temperatures(self, particle_type: ParticleType) -> unyt_array:
        return unyt_array(self.__file_object.read_dataset(particle_type.value, "Temperature"), units = "K")

    @staticmethod
    def generate_filepaths(
       *snapshot_number_strings: str,
        directory: str,
        basename: str,
        file_extension: str = "hdf5",
        parallel_ranks: Union[List[int], None] = None
    ) -> Dict[
            str,
            Union[str, Dict[int, str]]
         ]:
        raise NotImplementedError("Not implemented for EAGLE. Update to generalise file path creation.")#TODO:

    @staticmethod
    def scrape_filepaths(#TODO: create file info objects that do this with a common interface for generating these objects to allow each subclass to keep the filepath formatting clear and obscured from the user
        catalogue_directory: str
    ) -> Tuple[
            Tuple[
                str,
                Tuple[str, ...],
                Tuple[int, ...],
                str
            ],
            ...
         ]:

        pattern = re.compile(r'.*snapshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snap_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')

        snapshots: Dict[str, List[str, List[str], List[int], str]] = {}

        for root, _, files in os.walk(catalogue_directory):
            for filename in files:
                match = pattern.match(os.path.join(root, filename))
                if match:
                    number = match.group("number")
                    redshift_int = match.group("redshift_int")
                    redshift_dec = match.group("redshift_dec")
                    parallel_index = int(match.group("parallel_index"))
                    extension = match.group("extension")

                    tag = f"{number}_z{redshift_int}p{redshift_dec}"
                    basename = os.path.join(f"snapshot_{tag}", f"snap_{tag}")

                    if tag not in snapshots:
                        snapshots[tag] = [basename, [number], [parallel_index], extension]
                    else:
                        assert basename == snapshots[tag][0]
                        assert extension == snapshots[tag][3]
                        snapshots[tag][2].append(parallel_index)

        for tag in snapshots:
            snapshots[tag][2].sort()

        return tuple([
            tuple([
                snapshots[tag][0],
                tuple(snapshots[tag][1]),
                tuple(snapshots[tag][2]),
                snapshots[tag][3]
            ])
            for tag
            in snapshots
        ])

    @staticmethod
    def generate_filepaths_from_partial_info(
        directory: str,
        basename: Union[str, None] = None,
        snapshot_number_strings: Union[List[str], None] = None,
        file_extension: Union[str, None] = None,
        parallel_ranks: Union[List[int], None] = None
    ) -> Dict[
            str,
            Union[str, Dict[int, str]]
         ]:
        if basename is not None or file_extension is not None or parallel_ranks is not None:
            raise NotImplementedError("TODO: some fields not supported for EAGLE. Change API to use objects with file info specific to sim types.")#TODO:

        snap_file_info = { snap[1][0] : snap for snap in SnapshotEAGLE.scrape_filepaths(directory) }
        selected_files = {}
        for num in (snapshot_number_strings if snapshot_number_strings is not None else snap_file_info.keys()):
            if num not in snap_file_info:
                raise FileNotFoundError("Snapshot numbers provided not all present in directory.")
            selected_files[num] = { i : os.path.join("/mnt/aridata1/users/aricrowe/replacement_EAGLE_snap/RefL0100N1504" if num == "006" else directory, f"{snap_file_info[num][0]}.{i}.{snap_file_info[num][3]}") for i in snap_file_info[num][2] }

        return selected_files

    @staticmethod
    def get_snapshot_order(snapshot_file_info: List[str], reverse = False) -> List[str]:
        snapshot_file_info = list(snapshot_file_info)
        snapshot_file_info.sort(key = int, reverse = reverse)
        return snapshot_file_info



#class SnipshotEAGLE(SnapshotEAGLE):
#    def get_smoothing_lengths(self, particle_type: ParticleType) -> unyt_array:
#        raise NotImplementedError("Smoothing length not avalible from EAGLE snipshots.")
#    def _get_sfr(self, particle_type: ParticleType) -> unyt_array:
#        raise NotImplementedError("SFR not avalible from EAGLE snipshots.")
#
#    @staticmethod#TODO:
#    def scrape_filepaths(
#        catalogue_directory: str
#    ) -> Tuple[
#            Tuple[
#                str,
#                Tuple[str, ...],
#                Union[Tuple[int, ...], None],
#                str
#            ],
#            ...
#         ]:
#
#        pattern = re.compile(r'^snipshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)/snip_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
