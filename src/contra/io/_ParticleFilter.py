# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
import datetime
from typing import Any, Union, Collection, List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import singledispatchmethod

import numpy as np
import h5py as h5
from unyt import unyt_quantity
from QuasarCode import Console
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import Struct, TypedAutoProperty, NullableTypedAutoProperty, TypeShield, NestedTypeShield

from .._ParticleType import ParticleType
from ._Output_Objects import ContraData

class ParticleFilter(Struct):
    """
    particle_type

    redshift

    snapshot_relitive_filepath

    allowed_ids

    snapshot_mask

    los_masks (nullable)
    """

    particle_type: ParticleType = TypedAutoProperty[ParticleType](TypeShield(ParticleType),
                doc = "Particle type.")
    redshift: float = TypedAutoProperty[float](TypeShield(float),
                doc = "Redshift of the snapshot to which this filter applies.")
    snapshot_relitive_filepath: str = TypedAutoProperty[str](TypeShield(str),
                doc = "The snapshot file to which this filter applies. The first file if snapshot is split into parallel components.")
    allowed_ids: np.ndarray = TypedAutoProperty[np.ndarray](NestedTypeShield(np.ndarray, np.int64),
                doc = "IDs of selected particles.")
    snapshot_mask: np.ndarray = TypedAutoProperty[np.ndarray](NestedTypeShield(np.ndarray, np.bool_),
                doc = "Numpy boolean mask for selected particles.")
    los_masks: Dict[str, List["LOSFilter"]] | None = NullableTypedAutoProperty[Dict[str, List["LOSFilter"]]](TypeShield(dict),
                doc = "Filter data for line-of-sight files associated with this snapshot.")

class LOSFilter(Struct):
    """
    particle_type

    redshift

    file_number

    los_index

    los_relitive_filepath

    snapshot_filter_object

    mask

    particle_ids
    """

    particle_type: ParticleType = TypedAutoProperty[ParticleType](TypeShield(ParticleType),
                doc = "Particle type.")
    redshift: float = TypedAutoProperty[float](TypeShield(float),
                doc = "Redshift of the line-of-sight to which this filter applies.")
    file_number: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Number of the file. May not be unique - use redshift and number for uniqueness.")
    los_index: int = TypedAutoProperty[float](TypeShield(int),
                doc = "Index of the line-of-sight within its file.")
    los_relitive_filepath: str = TypedAutoProperty[str](TypeShield(str),
                doc = "The line-of-sight file to which this filter applies.")
    snapshot_filter_object: "ParticleFilter" = TypedAutoProperty["ParticleFilter"](TypeShield(ParticleFilter),
                doc = "Snapshot filter object that this line of sight belongs to.")
    mask: np.ndarray = TypedAutoProperty[np.ndarray](NestedTypeShield(np.ndarray, np.bool_),
                doc = "Numpy boolean mask for selected particles.")
    particle_ids: np.ndarray = TypedAutoProperty[np.ndarray](NestedTypeShield(np.ndarray, np.int64),
                doc = "IDs of particles in the line-of-sight. Not usually avalible in the file.")

class ParticleFilterFile(Struct):
    """
    version

    date

    output_file

    contra_file

    snapshots_directory

    los_directory (nullable)

    has_gas

    has_stars

    has_black_holes

    has_dark_matter

    data
    """

    def __init__(self, filepath: str = None, *args, **kwargs) -> None:
        self.__filepath: Union[str, None] = filepath
        super().__init__(*args, **kwargs)
        if self.__filepath is not None:
            self.__read()

    version: VersionInfomation = TypedAutoProperty[VersionInfomation](TypeShield(VersionInfomation),
                doc = "Contra version number.")
    date: datetime.date = TypedAutoProperty[datetime.date](TypeShield(datetime.date),
                doc = "Date of execution (start of program).")
    output_file: str = TypedAutoProperty[str](TypeShield(str),
                doc = "This output file location as generated.")
    contra_file: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Contra output file used to generate this file.")
    snapshots_directory: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Directory containing associated snapshots.")
    los_directory: str | None = NullableTypedAutoProperty[str](TypeShield(str),
                doc = "Directory containing associated line-of-sight files.")
    simulation_type: str = TypedAutoProperty[str](TypeShield(str),
                doc = "Type of source simulation data.")
    has_gas: bool = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains a dataset for gas particles.")
    has_stars: bool = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains a dataset for star particles.")
    has_black_holes: bool = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains a dataset for black hole particles.")
    has_dark_matter: bool = TypedAutoProperty[bool](TypeShield(bool),
                doc = "This output file contains a dataset for dark matter particles.")

    data: Dict[ParticleType, Dict[str, ParticleFilter]] = TypedAutoProperty[Dict[ParticleType, Dict[str, ParticleFilter]]](TypeShield(dict),#TODO: dict nested shield? (not cast!!!)
                doc = "Datasets by particle type.")

    @property
    def gas(self) -> Union[Dict[str, ParticleFilter], None]:
        return self.data[ParticleType.gas] if ParticleType.gas in self.data else None
    @property
    def stars(self) -> Union[Dict[str, ParticleFilter], None]:
        return self.data[ParticleType.star] if ParticleType.star in self.data else None
    @property
    def black_holes(self) -> Union[Dict[str, ParticleFilter], None]:
        return self.data[ParticleType.black_hole] if ParticleType.black_hole in self.data else None
    @property
    def dark_matter(self) -> Union[Dict[str, ParticleFilter], None]:
        return self.data[ParticleType.dark_matter] if ParticleType.dark_matter in self.data else None

    class FilterKeyError(KeyError):
        """
        Requested key not found.
        """

    def __require_part_type(self, part_type: ParticleType) -> None:
        if part_type not in self.data:
            raise ParticleFilterFile.FilterKeyError(f"No filters for {part_type.name} particles.")
    
    def get_ids(self, part_type: ParticleType, snapshot_number: str):
        self. __require_part_type(part_type)
        return self.data[part_type][snapshot_number].allowed_ids

    def get_snapshot_mask(self, part_type: ParticleType, snapshot_number: str) -> np.ndarray:
        self. __require_part_type(part_type)
        return self.data[part_type][snapshot_number].snapshot_mask

    def get_los_mask(self, part_type: ParticleType, snapshot_number: str, los_redshift: int, file_number: str, los_index: int) -> np.ndarray:
        self. __require_part_type(part_type)
        return self.data[part_type][snapshot_number].los_masks[f"z{los_redshift}.{file_number}"][los_index]
    
    def get_contra_data(self) -> ContraData:
        return ContraData.load(self.contra_file)

    def __read(self) -> None:
        """
        Set field values from a file.
        WARNING: this WILL overwrite any existing values!
        """
        assert self.__filepath is not None
        with h5.File(self.__filepath, "r") as file:
            self.version = VersionInfomation.from_string(file["Header"].attrs["Version"])
            self.date = datetime.datetime.fromtimestamp((file["Header"].attrs["Date"])).date()
            self.output_file = str(file["Header"].attrs["OutputFile"])
            self.contra_file = str(file["Header"].attrs["SourceFile"])
            self.snapshots_directory = str(file["Header"].attrs["SnapshotDirectory"])
            if "LineOfSightDirectory" in file:
                self.los_directory = str(file["Header"].attrs["LineOfSightDirectory"])
            self.simulation_type = str(file["Header"].attrs["SimulationType"])
            self.has_gas = bool(file["Header"].attrs["HasGas"])
            self.has_stars = bool(file["Header"].attrs["HasStars"])
            self.has_black_holes = bool(file["Header"].attrs["HasBlackHoles"])
            self.has_dark_matter = bool(file["Header"].attrs["HasDarkMatter"])

            self.data = {}
            for part_type in ParticleType.get_all():
                if part_type.common_hdf5_name in file:
                    d = file[part_type.common_hdf5_name]
                    self.data[part_type] = ParticleFilter()
                    self.data[part_type].particle_type = part_type
                    self.data[part_type].redshift = d.attrs["Redshift"]
                    self.data[part_type].snapshot_relitive_filepath = d.attrs["Snapshot"]
                    self.data[part_type].allowed_ids = d["AllowedIDs"][:]
                    self.data[part_type].snapshot_mask = d["SnapshotMask"][:]
                    if "LinesOfSight" in d:
                        l = d["LinesOfSight"]
                        self.data[part_type].los_masks = {}
                        for file_key in list(l.keys()):
                            self.data[part_type].los_masks[file_key] = []
                            for los_key in list(l[file_key].keys()):
                                los = l[file_key][los_key]
                                f = LOSFilter()
                                f.particle_type = part_type
                                f.redshift = los.attrs["Redshift"]
                                f.file_number = los.attrs["FileNumber"]
                                f.los_index = los.attrs["LineOfSightIndex"]
                                f.los_relitive_filepath = los.attrs["LinesOfSightFile"]
                                f.snapshot_filter_object = self.data[part_type]
                                f.mask = los["Mask"]
                                f.particle_ids = los["ParticleIDs"]
                                self.data[part_type].los_masks[file_key].append(f)

    def save(self, filepath: str | None = None):
        if filepath is None:
            filepath = self.__filepath
        if filepath is None:
            raise ValueError("No filepath provided and no default filepath set.")
        with h5.File(self.__filepath, "w") as file:

            h = file.create_group("Header")
            h.attrs["Version"] =  str(self.version)
            h.attrs["Date"] = datetime.datetime(year = self.date.year, month = self.date.month, day = self.date.day).timestamp()
            h.attrs["OutputFile"] = self.output_file
            h.attrs["SourceFile"] = self.contra_file
            h.attrs["SnapshotDirectory"] = self.snapshots_directory
            h.attrs["LineOfSightDirectory"] = self.los_directory
            h.attrs["SimulationType"] = self.simulation_type
            h.attrs["HasGas"] = self.has_gas
            h.attrs["HasStars"] = self.has_stars
            h.attrs["HasBlackHoles"] = self.has_black_holes
            h.attrs["HasDarkMatter"] = self.has_dark_matter

            for part_type in self.data:
                d = file.create_group(part_type.common_hdf5_name)
                d.attrs["Redshift"] = self.data[part_type].redshift
                d.attrs["Snapshot"] = self.data[part_type].snapshot_relitive_filepath
                d.create_dataset("AllowedIDs", self.data[part_type].allowed_ids)
                d.create_dataset("SnapshotMask", self.data[part_type].snapshot_mask)
                if self.data[part_type].los_masks is not None:
                    l = d.create_group("LinesOfSight")
                    for los_file_key in self.data[part_type].los_masks:
                        los_filters = l.create_group(los_file_key)
                        for i in range(len(self.data[part_type].los_masks[los_file_key])):
                            f = self.data[part_type].los_masks[i]
                            los = los_filters.create_group(str(i))
                            los.attrs["Redshift"] = f.redshift
                            los.attrs["FileNumber"] = f.file_number
                            los.attrs["LineOfSightIndex"] = f.los_index
                            los.attrs["LinesOfSightFile"] = f.los_relitive_filepath
                            los.create_dataset("Mask", f.mask)
                            los.create_dataset("ParticleIDs", f.particle_ids)
