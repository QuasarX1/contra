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
    
    def restrict_data_propper_loading_region(self, min_x: float|unyt_quantity|None = None, max_x: float|unyt_quantity|None = None, min_y: float|unyt_quantity|None = None, max_y: float|unyt_quantity|None = None, min_z: float|unyt_quantity|None = None, max_z: float|unyt_quantity|None = None, clear_existing_region = True) -> None:
        self.restrict_data_comoving_loading_region(
            min_x / self.a if min_x is not None else None,
            max_x / self.a if max_x is not None else None,
            min_y / self.a if min_y is not None else None,
            max_y / self.a if max_y is not None else None,
            min_z / self.a if min_z is not None else None,
            max_z / self.a if max_z is not None else None,
            clear_existing_region
        )

    def restrict_data_comoving_loading_region(self, min_x: float|unyt_quantity|None = None, max_x: float|unyt_quantity|None = None, min_y: float|unyt_quantity|None = None, max_y: float|unyt_quantity|None = None, min_z: float|unyt_quantity|None = None, max_z: float|unyt_quantity|None = None, clear_existing_region = True) -> None:
        # Get the conversion factor to go from comoving to h-less comoving
        # (a.k.a., multiply inputs by this value!)
        conversion_factor = 1 / ((self.hubble_param ** self.__length_h_exp) * self.__length_cgs_conversion_factor)

        # Convert values so each field is filled with a floating point value
        if min_x is not None:
            if not isinstance(min_x, unyt_quantity):
                min_x = unyt_quantity(min_x, units = "Mpc", dtype = float)
            min_x = min_x.to("cm").value * conversion_factor
        else:
            min_x = 0.0
        if max_x is not None:
            if not isinstance(max_x, unyt_quantity):
                max_x = unyt_quantity(max_x, units = "Mpc", dtype = float)
            max_x = max_x.to("cm").value * conversion_factor
        else:
            max_x = self.__box_size_internal_units
        if min_y is not None:
            if not isinstance(min_y, unyt_quantity):
                min_y = unyt_quantity(min_y, units = "Mpc", dtype = float)
            min_y = min_y.to("cm").value * conversion_factor
        else:
            min_y = 0.0
        if max_y is not None:
            if not isinstance(max_y, unyt_quantity):
                max_y = unyt_quantity(max_y, units = "Mpc", dtype = float)
            max_y = max_y.to("cm").value * conversion_factor
        else:
            max_y = self.__box_size_internal_units
        if min_z is not None:
            if not isinstance(min_z, unyt_quantity):
                min_z = unyt_quantity(min_z, units = "Mpc", dtype = float)
            min_z = min_z.to("cm").value * conversion_factor
        else:
            min_z = 0.0
        if max_z is not None:
            if not isinstance(max_z, unyt_quantity):
                max_z = unyt_quantity(max_z, units = "Mpc", dtype = float)
            max_z = max_z.to("cm").value * conversion_factor
        else:
            max_z = self.__box_size_internal_units

        # Check for regions where the min and max values are the wrong way around - i.e. wrap around the box
        # Correct these to allow negitive values (to be handled later)
        if min_x > max_x:
            min_x = np.mod(min_x, self.__box_size_internal_units) # Move both endpoints inside the box
            max_x = np.mod(max_x, self.__box_size_internal_units) # Move both endpoints inside the box
            if min_x > max_x:                                     # If the endpoints are still the wrong way around:
                min_y = min_y - self.__box_size_internal_units    #     Move the min point into negitive space so that it gets wrapped later
        if min_y > max_y:
            min_y = np.mod(min_y, self.__box_size_internal_units)
            max_y = np.mod(max_y, self.__box_size_internal_units)
            if min_y > max_y:
                min_y = min_y - self.__box_size_internal_units
        if min_z > max_z:
            min_z = np.mod(min_z, self.__box_size_internal_units)
            max_z = np.mod(max_z, self.__box_size_internal_units)
            if min_z > max_z:
                min_z = min_z - self.__box_size_internal_units

        # Check for regions larger than the box
        #     Truncate these to be the sixe of the box in that dimension
        # Also check for regions where the maximum is outside the box
        #     Shift these to the other side of the box (reduces wrapping conditions that need checking later)
        if self.__box_size_internal_units < max_x:
            if min_x < 0.0 or min_x + self.__box_size_internal_units < max_x:
                min_x = 0.0
                max_x = self.__box_size_internal_units
            else:
                min_x = min_x - self.__box_size_internal_units
                max_x = max_x - self.__box_size_internal_units
        if self.__box_size_internal_units < max_y:
            if min_y < 0.0 or min_y + self.__box_size_internal_units < max_y:
                min_y = 0.0
                max_y = self.__box_size_internal_units
            else:
                min_y = min_y - self.__box_size_internal_units
                max_y = max_y - self.__box_size_internal_units
        if self.__box_size_internal_units < max_z:
            if min_z < 0.0 or min_z + self.__box_size_internal_units < max_z:
                min_z = 0.0
                max_z = self.__box_size_internal_units
            else:
                min_z = min_z - self.__box_size_internal_units
                max_z = max_z - self.__box_size_internal_units

        # Store the chunk as a tuple in a list
        # This will be used to break up the chunk if it crosses a boundary
        wrapped_region_chunks: List[Tuple[float, float, float, float, float, float]] = [
            (min_x, max_x, min_y, max_y, min_z, max_z)
        ]
        copy_region_chunks: List[Tuple[float, float, float, float, float, float]]

        # Check for boundarys being crossed by the initial region and mutate the existing region(s) while creating new regions for the offending space
        # Only need to check 0 boundary as above changes should mandate the max value be within the box
        if min_x < 0.0:
            copy_region_chunks = wrapped_region_chunks.copy()
            wrapped_region_chunks = []
            for region in copy_region_chunks:
                wrapped_region_chunks.append((0.0, *region[1:]))
                wrapped_region_chunks.append((self.__box_size_internal_units + region[0], self.__box_size_internal_units, *region[2:]))
        if min_y < 0.0:
            copy_region_chunks = wrapped_region_chunks.copy()
            wrapped_region_chunks = []
            for region in copy_region_chunks:
                wrapped_region_chunks.append((*region[:2], 0.0, *region[3:]))
                wrapped_region_chunks.append((*region[:2], self.__box_size_internal_units + region[2], self.__box_size_internal_units, *region[4:]))
        if min_z < 0.0:
            copy_region_chunks = wrapped_region_chunks.copy()
            wrapped_region_chunks = []
            for region in copy_region_chunks:
                wrapped_region_chunks.append((*region[:4], 0.0, region[5]))
                wrapped_region_chunks.append((*region[:4], self.__box_size_internal_units + region[4], self.__box_size_internal_units))

        if clear_existing_region:
            self.__file_object.clear_selection()
        for region in wrapped_region_chunks:
            if region[0] == region[1] or region[2] == region[3] or region[4] == region[5]:
                continue # If any of the axes are of 0 length, don't bother
                         # Should only occour when a region is specified adjacent to but entierly outside of the box
            self.__file_object.select_region(*region)
    
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
        catalogue_directory: str,
        snipshots: bool = False
    ) -> Tuple[
            Tuple[
                str,
                Tuple[str, ...],
                Tuple[int, ...],
                str
            ],
            ...
         ]:

#        pattern = re.compile(r'.*snapshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snap_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
        if not snipshots:
            pattern = re.compile(r'.*snapshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snap_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')
        else:
            pattern = re.compile(r'.*snipshot_(?P<number>\d{3})_z(?P<redshift_int>\d+)p(?P<redshift_dec>\d+)[\\/]snip_(?P=number)_z(?P=redshift_int)p(?P=redshift_dec)\.(?P<parallel_index>\d+)\.(?P<extension>\w+)$')

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
                    basename = os.path.join(f"snapshot_{tag}", f"snap_{tag}") if not snipshots else os.path.join(f"snipshot_{tag}", f"snip_{tag}")

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
        parallel_ranks: Union[List[int], None] = None,
        snipshots: bool = False
    ) -> Dict[
            str,
            Union[str, Dict[int, str]]
         ]:
        if basename is not None or file_extension is not None or parallel_ranks is not None:
            raise NotImplementedError("TODO: some fields not supported for EAGLE. Change API to use objects with file info specific to sim types.")#TODO:

        snap_file_info = { snap[1][0] : snap for snap in SnapshotEAGLE.scrape_filepaths(directory, snipshots = snipshots) }
        selected_files = {}
        for num in (snapshot_number_strings if snapshot_number_strings is not None else snap_file_info.keys()):
            if num not in snap_file_info:
                raise FileNotFoundError("Snapshot numbers provided not all present in directory.")
#            selected_files[num] = { i : os.path.join("/mnt/aridata1/users/aricrowe/replacement_EAGLE_snap/RefL0100N1504" if num == "006" else directory, f"{snap_file_info[num][0]}.{i}.{snap_file_info[num][3]}") for i in snap_file_info[num][2] }
            selected_files[num] = { i : os.path.join(directory, f"{snap_file_info[num][0]}.{i}.{snap_file_info[num][3]}") for i in snap_file_info[num][2] }

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
