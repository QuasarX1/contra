# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
import datetime
from typing import Union, Collection, List, Dict

import numpy as np
import h5py as h5
from unyt import unyt_quantity
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import Struct, TypedAutoProperty, NullableTypedAutoProperty, TypeShield, NestedTypeShield

from .._ParticleType import ParticleType
from ._SnapshotBase import SnapshotBase
from ._SnapshotSWIFT import SnapshotSWIFT
from ._SnapshotEAGLE import SnapshotEAGLE

class HeaderDataset(Struct):
    """
    version

    date

    target_file

    simulation_type

    redshift

    N_searched_snapshots

    output_file

    has_gas

    has_stars

    has_black_holes

    has_dark_matter

    has_statistics
    """

    version = TypedAutoProperty[VersionInfomation](TypeShield(VersionInfomation),
                doc = "Contra version number.")
    date = TypedAutoProperty[datetime.date](TypeShield(datetime.date),
                doc = "Date of execution (start of program).")
    target_file = TypedAutoProperty[str](TypeShield(str),
                doc = "Snapshot file defining positions of target particles.")
    simulation_type = TypedAutoProperty[str](TypeShield(str),
                doc = "Type of source simulation data.")
    redshift = TypedAutoProperty[float](TypeShield(float),
                doc = "Redshift of the target particle distribution.")
    N_searched_snapshots = TypedAutoProperty[int](TypeShield(int),
                doc = "Number of snapshots searched.")
#    searched_files = TypedAutoProperty[List[str]](TypeShield(List[str]),
#                doc = "List of catalogue files searched.")
    output_file = TypedAutoProperty[str](TypeShield(str),
                doc = "This output file location as generated by Contra.")
    has_gas = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains a dataset for gas particles.")
    has_stars = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains a dataset for star particles.")
    has_black_holes = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains a dataset for black hole particles.")
    has_dark_matter = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains a dataset for dark matter particles.")
    has_statistics = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains statistics for each snapshot searched.")



class ParticleTypeDataset(Struct):
    """
    particle_type

    redshifts

    halo_ids

    halo_masses

    halo_masses_scaled

    locations_pre_ejection
    """

    particle_type = TypedAutoProperty[ParticleType](TypeShield(ParticleType),
                doc = "Particle type.")
    redshifts = TypedAutoProperty[Collection[float]](NestedTypeShield(np.ndarray, np.float64),
                doc = "Redshift of last halo membership.")
    halo_ids = TypedAutoProperty[Collection[int]](NestedTypeShield(np.ndarray, np.int64),
                doc = "ID of last halo.")
    halo_masses = TypedAutoProperty[Collection[float]](NestedTypeShield(np.ndarray, np.float64),#TODO: make unyt
                doc = "Mass of last halo.")
    halo_masses_scaled = TypedAutoProperty[Collection[float]](NestedTypeShield(np.ndarray, np.float64),
                doc = "Mass of last halo as a fraction of the L_* mass at that redshift.")
    positions_pre_ejection = TypedAutoProperty[Collection[Collection[float]]](NestedTypeShield(np.ndarray, np.ndarray, np.float64),#TODO: make unyt
                doc = "Particle coordinates prior to final ejection.")



class SnapshotStatsDataset(Struct):
    """
    snapshot_filepath

    catalogue_filepath

    redshift

    N_particles

    N_particles_(gas | star | black_hole | dark_matter)

    N_haloes

    N_haloes_top_level

    N_halo_children

    N_halo_decendants

    N_halo_particles

    N_halo_particles_(gas | star | black_hole | dark_matter)

    N_particles_matched

    N_particles_matched_(gas | star | black_hole | dark_matter)

    particle_total_volume

    halo_particle_total_volume

    particles_matched_total_volume

    particles_matched_total_volume_(gas | star | black_hole | dark_matter)
    """

    snapshot_filepath = TypedAutoProperty[str](TypeShield(str),
                doc = "Filepath of snapshot file.")
    catalogue_filepath = TypedAutoProperty[str](TypeShield(str),
                doc = "Filepath of catalogue file.")
    redshift = TypedAutoProperty[float](TypeShield(float),
                doc = "Snapshot redshift.")
    N_particles = TypedAutoProperty[int](TypeShield(int),
                doc = "Number of particles in the snapshot.")
    N_particles_gas = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of gas particles in the snapshot.")
    N_particles_star = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of star particles in the snapshot.")
    N_particles_black_hole = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of black hole particles in the snapshot.")
    N_particles_dark_matter = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of dark matter particles in the snapshot.")
    N_haloes = TypedAutoProperty[int](TypeShield(int),
                doc = "Number of haloes.")
    N_haloes_top_level = TypedAutoProperty[int](TypeShield(int),
                doc = "Number of haloes with no parent.")
    N_halo_children = TypedAutoProperty[Collection[int]](NestedTypeShield(np.ndarray, np.int64),
                doc = "Number of direct children of each halo.")
    N_halo_decendants = TypedAutoProperty[Collection[int]](NestedTypeShield(np.ndarray, np.int64),
                doc = "Total number of decendants of each halo.")
    N_halo_particles = TypedAutoProperty[int](TypeShield(int),
                doc = "Number of particles in haloes.")
    N_halo_particles_gas = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of gas particles.")
    N_halo_particles_star = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of star particles.")
    N_halo_particles_black_hole = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of black hole particles.")
    N_halo_particles_dark_matter = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of dark matter particles.")
    N_particles_matched = TypedAutoProperty[int](TypeShield(int),
                doc = "Number of newly matched particles in this snapshot.")
    N_particles_matched_gas = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of newly matched gas particles in this snapshot.")
    N_particles_matched_star = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of newly matched star particles in this snapshot.")
    N_particles_matched_black_hole = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of newly matched black_hole particles in this snapshot.")
    N_particles_matched_dark_matter = NullableTypedAutoProperty[int](TypeShield(int),
                doc = "Number of newly matched dark_matter particles in this snapshot.")
    particle_total_volume = TypedAutoProperty[unyt_quantity](TypeShield(unyt_quantity),
                doc = "Total volume of all particles in the snapshot.")
    halo_particle_total_volume = TypedAutoProperty[unyt_quantity](TypeShield(unyt_quantity),
                doc = "Total volume particles in haloes in the snapshot.")
    particles_matched_total_volume = TypedAutoProperty[unyt_quantity](TypeShield(unyt_quantity),
                doc = "Total volume of newly matched particles in the snapshot.")
    particles_matched_total_volume_gas = NullableTypedAutoProperty[unyt_quantity](TypeShield(unyt_quantity),
                doc = "Total volume of newly matched gas particles in the snapshot.")
    particles_matched_total_volume_star = NullableTypedAutoProperty[unyt_quantity](TypeShield(unyt_quantity),
                doc = "Total volume of newly matched star particles in the snapshot.")
    particles_matched_total_volume_black_hole = NullableTypedAutoProperty[unyt_quantity](TypeShield(unyt_quantity),
                doc = "Total volume of newly matched black hole particles in the snapshot.")
    particles_matched_total_volume_dark_matter = NullableTypedAutoProperty[unyt_quantity](TypeShield(unyt_quantity),
                doc = "Total volume of newly matched dark matter particles in the snapshot.")
    
    def __str__(self):
        return f"""\
redshift:                                {self.redshift}
number of particles:                     {self.N_particles}
number of particles in haloes:           {self.N_halo_particles}
number of particles newley matched:      {self.N_particles_matched}
total particle volume:                   {self.particle_total_volume}
number of haloes:                        {self.N_haloes}
number of top level haloes:              {self.N_haloes_top_level}
total volume of particles in haloes:     {self.halo_particle_total_volume}
total volume of particles newly matched: {self.halo_particle_total_volume}\
""" + (f"""\
gas:
    number of particles:                     {self.N_particles_gas}
    number of particles in haloes:           {self.N_halo_particles_gas}
    number of particles newly matched:       {self.N_particles_matched_gas}
    total volume of particles newly matched: {self.N_particles_matched_gas}\
""" if self.N_particles_gas is not None        else "gas: -----------") + (f"""\
stars:
    number of particles:                     {self.N_particles_star}
    number of particles in haloes:           {self.N_halo_particles_star}
    number of particles newly matched:       {self.N_particles_matched_star}
    total volume of particles newly matched: {self.N_particles_matched_star}\
""" if self.N_particles_star is not None       else "stars: ---------") + (f"""\
black holes:
    number of particles:                     {self.N_particles_black_hole}
    number of particles in haloes:           {self.N_halo_particles_black_hole}
    number of particles newly matched:       {self.N_particles_matched_black_hole}
    total volume of particles newly matched: {self.N_particles_matched_black_hole}\
""" if self.N_particles_black_hole is not None else "black holes: ---") + (f"""\
dark mater:
    number of particles:                     {self.N_particles_dark_matter}
    number of particles in haloes:           {self.N_halo_particles_dark_matter}
    number of particles newly matched:       {self.N_particles_matched_dark_matter}
    total volume of particles newly matched: {self.N_particles_matched_dark_matter}\
""" if self.N_particles_dark_matter is not None else "dark mater: ---")
    


class ContraData(Struct):
    """
    Data output by Contra.

    To read a file, call load() and pass the filepath.
    """

    header = TypedAutoProperty[HeaderDataset](TypeShield(HeaderDataset),
                doc = "Header. Data about the Contra run and output file.")
    data = TypedAutoProperty[Dict[ParticleType, ParticleTypeDataset]](TypeShield(dict),#TODO: dict nested shield? (not cast!!!)
                doc = "Datasets by particle type.")
    snapshot_search_stats = NullableTypedAutoProperty[List[SnapshotStatsDataset]](NestedTypeShield(list, SnapshotStatsDataset),
                doc = "Search stats for each searched snapshot.")

    @property
    def gas(self) -> Union[ParticleTypeDataset, None]:
        return self.data[ParticleType.gas] if ParticleType.gas in self.data else None
    @property
    def stars(self) -> Union[ParticleTypeDataset, None]:
        return self.data[ParticleType.star] if ParticleType.star in self.data else None
    @property
    def black_holes(self) -> Union[ParticleTypeDataset, None]:
        return self.data[ParticleType.black_hole] if ParticleType.black_hole in self.data else None
    @property
    def dark_matter(self) -> Union[ParticleTypeDataset, None]:
        return self.data[ParticleType.dark_matter] if ParticleType.dark_matter in self.data else None
    
    def _get_snapshot(self, filepath: str) -> SnapshotBase:
        if self.header.simulation_type == "SWIFT":
            return SnapshotSWIFT(filepath)
        elif self.header.simulation_type == "EAGLE":
            return SnapshotEAGLE(filepath)
        else:
            raise NotImplementedError(f"\"ContraData.get_snapshot\" not implemented for source simulation type \"{self.header.simulation_type}\".")
    
    def get_target_snapshot(self) -> SnapshotBase:
        return self._get_snapshot(self.header.target_file)
    
    def get_snapshot(self, stats: SnapshotStatsDataset) -> SnapshotBase:
        return self._get_snapshot(stats.snapshot_filepath)
    
    @staticmethod
    def load(filepath: str) -> "ContraData":
        result = None
        reader = OutputReader(filepath)
        with reader:
            result = reader.read()
        return result



class OutputWriter(object):
    """
    Writes datasets to an HDF5 file.
    """

    def __init__(self, filepath: str) -> None:
        self.__filepath = filepath
        self.__file = None

        # Create the file
        f = h5.File(self.__filepath, "w")
        f.create_group("SnapshotStats")
        f.close()

    @property
    def is_open(self):
        return self.__file is not None
    
    def open(self):
        self.__file = h5.File(self.__filepath, "a")

    def close(self):
        self.__file.close()
        self.__file = None

    def __enter__(self) -> "OutputWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def write_header(self, header: HeaderDataset) -> None:
        if not self.is_open:
            raise IOError("File not open. Call open() first or use with statement.")
        d = self.__file.create_group("Header")
        d.attrs["Version"] = str(header.version)
        d.attrs["Date"] = datetime.datetime(year = header.date.year, month = header.date.month, day = header.date.day).timestamp()
        d.attrs["Snapshot"] = header.target_file
        d.attrs["SimulationType"] = header.simulation_type
        d.attrs["Redshift"] = header.redshift
        d.attrs["NumberSearchedSnapshots"] = header.N_searched_snapshots
#        d.attrs["SearchedFiles"] = header.searched_files
        d.attrs["OutputFile"] = header.output_file
        d.attrs["HasGas"] = header.has_gas
        d.attrs["HasStars"] = header.has_stars
        d.attrs["HasBlackHoles"] = header.has_black_holes
        d.attrs["HasDarkMatter"] = header.has_dark_matter
        d.attrs["HasStatistics"] = header.has_statistics

    def write_particle_type_dataset(self, dataset: ParticleTypeDataset) -> None:
        if not self.is_open:
            raise IOError("File not open. Call open() first or use with statement.")
        d = self.__file.create_group(dataset.particle_type.common_hdf5_name)
        d.create_dataset(name = "HaloRedshift", data = dataset.redshifts)
        d.create_dataset(name = "HaloID", data = dataset.halo_ids)
        d.create_dataset(name = "HaloMass", data = dataset.halo_masses)
        d.create_dataset(name = "RelitiveHaloMass", data = dataset.halo_masses_scaled)
        d.create_dataset(name = "PositionPreEjection", data = dataset.positions_pre_ejection)

    def write_snapshot_stats_dataset(self, index: int, stats: SnapshotStatsDataset) -> None:
        if not self.is_open:
            raise IOError("File not open. Call open() first or use with statement.")
        d = self.__file["SnapshotStats"].create_group(str(index))
        d.attrs["SnapshotFilepath"] = stats.snapshot_filepath
        d.attrs["CatalogueFilepath"] = stats.catalogue_filepath
        d.attrs["Redshift"] = stats.redshift
        d.attrs["NumberOfParticles"] = stats.N_particles
        d.attrs["NumberOfHaloes"] = stats.N_haloes
        d.attrs["NumberOfTopLevelHaloes"] = stats.N_haloes_top_level
        d.attrs["NumberOfHaloParticles"] = stats.N_halo_particles
        d.attrs["NumberOfMatchedParticles"] = stats.N_particles_matched
        d.attrs["VolumeOfParticles"] = stats.particle_total_volume.to("Mpc**3").value
        d.attrs["VolumeOfHaloParticles"] = stats.halo_particle_total_volume.to("Mpc**3").value
        d.attrs["VolumeOfMatchedParticles"] = stats.particles_matched_total_volume.to("Mpc**3").value

        d.create_dataset(name = "NumberOfTypedParticles", data = np.array([
            (stats.N_particles_gas         if stats.N_particles_gas         is not None else -1),
            (stats.N_particles_star        if stats.N_particles_star        is not None else -1),
            (stats.N_particles_black_hole  if stats.N_particles_black_hole  is not None else -1),
            (stats.N_particles_dark_matter if stats.N_particles_dark_matter is not None else -1)
        ], dtype = int))
        d.create_dataset(name = "NumberOfHaloTypedParticles", data = np.array([
            (stats.N_halo_particles_gas         if stats.N_halo_particles_gas         is not None else -1),
            (stats.N_halo_particles_star        if stats.N_halo_particles_star        is not None else -1),
            (stats.N_halo_particles_black_hole  if stats.N_halo_particles_black_hole  is not None else -1),
            (stats.N_halo_particles_dark_matter if stats.N_halo_particles_dark_matter is not None else -1)
        ], dtype = int))
        d.create_dataset(name = "NumberOfMatchedTypedParticles", data = np.array([
            (stats.N_particles_matched_gas         if stats.N_particles_matched_gas         is not None else -1),
            (stats.N_particles_matched_star        if stats.N_particles_matched_star        is not None else -1),
            (stats.N_particles_matched_black_hole  if stats.N_particles_matched_black_hole  is not None else -1),
            (stats.N_particles_matched_dark_matter if stats.N_particles_matched_dark_matter is not None else -1)
        ], dtype = int))
        d.create_dataset(name = "VolumeOfMatchedTypedParticles", data = np.array([
            (stats.particles_matched_total_volume_gas.to("Mpc**3").value         if stats.particles_matched_total_volume_gas         is not None else -1),
            (stats.particles_matched_total_volume_star.to("Mpc**3").value        if stats.particles_matched_total_volume_star        is not None else -1),
            (stats.particles_matched_total_volume_black_hole.to("Mpc**3").value  if stats.particles_matched_total_volume_black_hole  is not None else -1),
            (stats.particles_matched_total_volume_dark_matter.to("Mpc**3").value if stats.particles_matched_total_volume_dark_matter is not None else -1)
        ], dtype = float))
        d.create_dataset(name = "HaloesNumberOfChildren", data = stats.N_halo_children)
        d.create_dataset(name = "HaloesNumberOfDecendants", data = stats.N_halo_decendants)

    def write(self, data: ContraData) -> None:
        force_open = not self.is_open
        if force_open:
            self.open()
        self.write_header(data.header)
        for key in data.data:
            self.write_particle_type_dataset(data.data[key])
        if data.header.has_statistics:
            for i, stats in enumerate(data.snapshot_search_stats):
                self.write_snapshot_stats_dataset(i, stats)
        if force_open:
            self.close()



class OutputReader(object):
    """
    Reads saved Contra data.
    """

    def __init__(self, filepath: str) -> None:
        self.__filepath = filepath
        self.__file = None

    @property
    def is_open(self):
        return self.__file is not None
    
    def open(self):
        self.__file = h5.File(self.__filepath, "r")

    def close(self):
        self.__file.close()
        self.__file = None

    def __enter__(self) -> "OutputReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def read_header(self) -> HeaderDataset:
        if not self.is_open:
            raise IOError("File not open. Call open() first or use with statement.")
        s = HeaderDataset()
        s.version = VersionInfomation.from_string(self.__file["Header"].attrs["Version"])
        s.date = datetime.datetime.fromtimestamp((self.__file["Header"].attrs["Date"])).date()
        s.target_file = str(self.__file["Header"].attrs["Snapshot"])
        s.simulation_type = str(self.__file["Header"].attrs["SimulationType"])
        s.redshift = float(self.__file["Header"].attrs["Redshift"])
        s.N_searched_snapshots = int(self.__file["Header"].attrs["NumberSearchedSnapshots"])
#        s.searched_files = self.__file["Header"].attrs["SearchedFiles"]
        s.output_file = str(self.__file["Header"].attrs["OutputFile"])
        s.has_gas = bool(self.__file["Header"].attrs["HasGas"])
        s.has_stars = bool(self.__file["Header"].attrs["HasStars"])
        s.has_black_holes = bool(self.__file["Header"].attrs["HasBlackHoles"])
        s.has_dark_matter = bool(self.__file["Header"].attrs["HasDarkMatter"])
        s.has_statistics = bool(self.__file["Header"].attrs["HasStatistics"])
        return s

    def read_particle_type_dataset(self, part_type: ParticleType) -> ParticleTypeDataset:
        if not self.is_open:
            raise IOError("File not open. Call open() first or use with statement.")
        if part_type.common_hdf5_name not in self.__file:
            raise KeyError(f"Contra output file has no dataset \"{part_type.common_hdf5_name}\".")
        d = self.__file[part_type.common_hdf5_name]
        s = ParticleTypeDataset()
        s.redshifts = d["HaloRedshift"][:]
        s.halo_ids = d["HaloID"][:]
        s.halo_masses = d["HaloMass"][:]
        s.halo_masses_scaled = d["RelitiveHaloMass"][:]
        s.positions_pre_ejection = d["PositionPreEjection"][:]
        return s

    def read_snapshot_stats_dataset(self, index: int) -> SnapshotStatsDataset:
        if not self.is_open:
            raise IOError("File not open. Call open() first or use with statement.")
        n_snapshots = int(self.__file["Header"].attrs["NumberSearchedSnapshots"])
        true_index = index if index >= 0 else (n_snapshots - index)
        if true_index >= n_snapshots:
            raise IndexError(f"Index {index} out of bounds for number of snapshots avalible ({n_snapshots}).")
        d = self.__file[f"SnapshotStats/{index}"]
        s = SnapshotStatsDataset()

        s.snapshot_filepath = str(d.attrs["SnapshotFilepath"])
        s.catalogue_filepath = str(d.attrs["CatalogueFilepath"])
        s.redshift = float(d.attrs["Redshift"])
        s.N_particles = int(d.attrs["NumberOfParticles"])
        s.N_haloes = int(d.attrs["NumberOfHaloes"])
        s.N_haloes_top_level = int(d.attrs["NumberOfTopLevelHaloes"])
        s.N_halo_particles = int(d.attrs["NumberOfHaloParticles"])
        s.N_particles_matched = int(d.attrs["NumberOfMatchedParticles"])
        s.particle_total_volume = unyt_quantity(d.attrs["VolumeOfParticles"], units = "Mpc**3")
        s.halo_particle_total_volume = unyt_quantity(d.attrs["VolumeOfHaloParticles"], units = "Mpc**3")
        s.particles_matched_total_volume = unyt_quantity(d.attrs["VolumeOfMatchedParticles"], units = "Mpc**3")

        N_particles__by_type = d["NumberOfTypedParticles"][:]
        if N_particles__by_type[0] > -1:
            s.N_particles_gas = int(N_particles__by_type[0])
        if N_particles__by_type[1] > -1:
            s.N_particles_star = int(N_particles__by_type[1])
        if N_particles__by_type[2] > -1:
            s.N_particles_black_hole = int(N_particles__by_type[2])
        if N_particles__by_type[3] > -1:
            s.N_particles_dark_matter = int(N_particles__by_type[3])

        N_halo_particles__by_type = d["NumberOfHaloTypedParticles"][:]
        if N_halo_particles__by_type[0] > -1:
            s.N_halo_particles_gas = int(N_halo_particles__by_type[0])
        if N_halo_particles__by_type[1] > -1:
            s.N_halo_particles_star = int(N_halo_particles__by_type[1])
        if N_halo_particles__by_type[2] > -1:
            s.N_halo_particles_black_hole = int(N_halo_particles__by_type[2])
        if N_halo_particles__by_type[3] > -1:
            s.N_halo_particles_dark_matter = int(N_halo_particles__by_type[3])

        N_particles_matched__by_type = d["NumberOfMatchedTypedParticles"][:]
        if N_particles_matched__by_type[0] > -1:
            s.N_particles_matched_gas = int(N_particles_matched__by_type[0])
        if N_particles_matched__by_type[1] > -1:
            s.N_particles_matched_star = int(N_particles_matched__by_type[1])
        if N_particles_matched__by_type[2] > -1:
            s.N_particles_matched_black_hole = int(N_particles_matched__by_type[2])
        if N_particles_matched__by_type[3] > -1:
            s.N_particles_matched_dark_matter = int(N_particles_matched__by_type[3])

        particles_matched_total_volume__by_type = d["VolumeOfMatchedTypedParticles"][:]
        if particles_matched_total_volume__by_type[0] > -1:
            s.particles_matched_total_volume_gas = unyt_quantity(particles_matched_total_volume__by_type[0], units = "Mpc**3")
        if particles_matched_total_volume__by_type[1] > -1:
            s.particles_matched_total_volume_star = unyt_quantity(particles_matched_total_volume__by_type[1], units = "Mpc**3")
        if particles_matched_total_volume__by_type[2] > -1:
            s.particles_matched_total_volume_black_hole = unyt_quantity(particles_matched_total_volume__by_type[2], units = "Mpc**3")
        if particles_matched_total_volume__by_type[3] > -1:
            s.particles_matched_total_volume_dark_matter = unyt_quantity(particles_matched_total_volume__by_type[3], units = "Mpc**3")

        s.N_halo_children = d["HaloesNumberOfChildren"][:]
        s.N_halo_decendants = d["HaloesNumberOfDecendants"][:]

        return s
    
    def read(self) -> ContraData:
        force_open = not self.is_open
        if force_open:
            self.open()
        s = ContraData()
        s.header = self.read_header()
        loaded_data = {}
        for p in ParticleType.get_all():
            try:
                loaded_data[p] = self.read_particle_type_dataset(p)
            except:
                loaded_data.pop(p, None)
        s.data = loaded_data
        if s.header.has_statistics:
            loaded_snap_stats = []
            for i in range(s.header.N_searched_snapshots):
                loaded_snap_stats.append(self.read_snapshot_stats_dataset(i))
            s.snapshot_search_stats = loaded_snap_stats
        else:
            s.snapshot_search_stats = None
        if force_open:
            self.close()
        return s
