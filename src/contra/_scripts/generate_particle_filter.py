# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .. import VERSION, ParticleType, ArrayReorder, ArrayMapping, calculate_wrapped_distance
from .._L_star import get_L_star_halo_mass_of_z
from ..io import SnapshotBase, SnapshotEAGLE, SnapshotSWIFT, \
                ContraData, OutputReader, HeaderDataset, ParticleTypeDataset, \
                SnapshotStatsDataset, \
                ParticleFilterFile, SnapshotParticleFilter, LineOfSightParticleFilter, \
                LineOfSightFileEAGLE, LineOfSightFileSWIFT

import datetime
import os
from typing import Any, Union, List, Tuple, Dict
import asyncio

import numpy as np
from unyt import unyt_quantity, unyt_array
import h5py as h5
from QuasarCode import Settings, Console
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import ScriptWrapper

#contra-generate-particle-filter ../contra/EAGLE/L12N188/z3/contra-output.hdf5 --snapshots /users/aricrowe/EAGLE/L0012N0188/REFERENCE/data/ --lines-of-sight /users/aricrowe/EAGLE/L0012N0188/REFERENCE/data/los/ --los-only --EAGLE --gas --max-halo-mass 1000000000 --EAGLE-los-id-file ./output-test.hdf5 -v

def main():
    ScriptWrapper(
        command = "contra-generate-particle-filter",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 7, 7),
        description = "",
        parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.PositionalParam[str](
                name = "data",
                short_name = "i",
                description = "Input contra data file."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "snapshots",
                sets_param = "snapshot_directory",
                description = "Where to search for snapshots.\nDefaults to the snapshot location specified in the Contra output file."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "lines-of-sight",
                sets_param = "los_directory",
                description = "Where to search for line-of-sight files."
            ),
            ScriptWrapper.Flag(
                name = "los-only",
                description = "Do not compute filters for snapshot particles.",
                requirements = ["lines-of-sight"]
            ),
            ScriptWrapper.Flag(
                name = "EAGLE",
                sets_param = "is_EAGLE",
                description = "Running on EAGLE data.",
                conflicts = ["is_SWIFT"]
            ),
            ScriptWrapper.Flag(
                name = "SWIFT",
                sets_param = "is_SWIFT",
                description = "Running on data generated using SWIFT.",
                conflicts = ["is_EAGLE"]
            ),
            ScriptWrapper.Flag(
                name = "gas",
                short_name = "g",
                sets_param = "do_gas",
                description = "Run search on gas particles."
            ),
            ScriptWrapper.Flag(
                name = "stars",
                short_name = "s",
                sets_param = "do_stars",
                description = "Run search on star particles."
            ),
            ScriptWrapper.Flag(
                name = "black-holes",
                short_name = "bh",
                sets_param = "do_black_holes",
                description = "Run search on black hole particles."
            ),
            ScriptWrapper.Flag(
                name = "dark-matter",
                short_name = "dm",
                sets_param = "do_dark_matter",
                description = "Run search on dark matter particles."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "output-file",
                short_name = "o",
                sets_param = "output_filepath",
                default_value = "allowed-particles.hdf5",
                description = "File to save results to."
            ),
            ScriptWrapper.Flag(
                name = "allow-overwrite",
                description = "Allow an existing output file to be overwritten."
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-halo-mass",
                description = "Minimum absolute halo mass.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-halo-mass",
                description = "Maximum absolute halo mass.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-scaled-halo-mass",
                description = "Minimum halo mass as a fraction of M*(z).",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-scaled-halo-mass",
                description = "Maximum halo mass as a fraction of M*(z).",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-ejection-distance",
                description = "Minimum distance traveled by particle since last found in a halo.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-ejection-distance",
                description = "Maximum distance traveled by particle since last found in a halo.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[str](
                name = "EAGLE-los-id-file",
                sets_param = "recovered_eagle_los_ids",
                description = "File containing IDs recovered for particles in EAGLE Line of Sight files.",
                requirements = ["EAGLE", "lines-of-sight"]
            )
        )
    ).run_with_async(__main)

async def __main(
            data: str,
            snapshot_directory: str|None,
            los_directory: str|None,
            los_only: bool,
            is_EAGLE: bool,
            is_SWIFT: bool,
            do_gas: bool,
            do_stars: bool,
            do_black_holes: bool,
            do_dark_matter: bool,
            output_filepath: str,
            allow_overwrite: bool,
            min_halo_mass: float|None,
            max_halo_mass: float|None,
            min_scaled_halo_mass: float|None,
            max_scaled_halo_mass: float|None,
            min_ejection_distance: float|None,
            max_ejection_distance: float|None,
            recovered_eagle_los_ids: str|None
          ) -> None:

    # Store the current date at the start in case the file writing occours after midnight.
    start_date = datetime.date.today()

    if not (min_halo_mass or max_halo_mass or min_scaled_halo_mass or max_scaled_halo_mass or min_ejection_distance or max_ejection_distance):
        Console.print_error("Must specify at leas one filter argument.")
        Console.print_info("Terminating...")
        return

    snapshot_type_string: str
    if not (is_EAGLE or is_SWIFT):
        Console.print_error("Must specify either EAGLE or SWIFT simulation type.")
        Console.print_info("Terminating...")
        return
    else:
        if is_EAGLE:
            snapshot_type_string = "EAGLE"
            Console.print_verbose_info("Snapshot type: EAGLE")
        elif is_SWIFT:
            snapshot_type_string = "SWIFT"
            Console.print_verbose_info("Snapshot type: SWIFT")

    # Identify snapshot and catalogue types
    new_snapshot_type = SnapshotEAGLE if is_EAGLE else SnapshotSWIFT
    new_los_file_type = LineOfSightFileEAGLE if is_EAGLE else LineOfSightFileSWIFT

    Console.print_verbose_info("Particle Types:")

    if not (do_gas or do_stars or do_black_holes or do_dark_matter):
        Console.print_verbose_warning("    No particle type(s) specified. Enabling all particle types.")
        do_gas = do_stars = do_black_holes = do_dark_matter = True
    particle_types = []
    if do_gas:
        particle_types.append(ParticleType.gas)
        Console.print_verbose_info("    Filtering gas particles.")
    if do_stars:
        particle_types.append(ParticleType.star)
        Console.print_verbose_info("    Filtering star particles.")
    if do_black_holes:
        particle_types.append(ParticleType.black_hole)
        Console.print_verbose_info("    Filtering black hole particles.")
    if do_dark_matter:
        particle_types.append(ParticleType.dark_matter)
        Console.print_verbose_info("    Filtering dark matter particles.")

    if is_EAGLE and los_directory is not None and recovered_eagle_los_ids is None:
        Console.print_error("--EAGLE and --lines-of-sight options set without --EAGLE-los-id-file.\nFiltering EAGLE line-of-sight files requires the recovery of the particle IDs.")
        Console.print_info("Terminating...")
        return

    # Ensure that the path is an absolute path
    output_filepath = os.path.abspath(output_filepath)

    if os.path.exists(output_filepath):
        if not allow_overwrite:
            Console.print_error("Output file already exists. Either remove it first or explicitley enable overwrite.")
            Console.print_info("Terminating...")
            return
        else:
            Console.print_warning("Pre-existing output file will be overwritten.")
    elif False:#TODO: check for valid file location (to prevent an error at the last minute!)
        pass

    Console.print_info("Identifying snapshot and catalogue files.")

    # Read Contra header data
    contra_reader = OutputReader(data)
    with contra_reader:
        contra_header = contra_reader.read_header()
#        contra_header.N_searched_snapshots
#        contra_header.has_statistics

    # Find the correct path for the target snapshot
    snap_dir, *file = contra_header.target_snapshot.rsplit(os.path.sep, maxsplit = 1 if is_SWIFT else 2)
    if snapshot_directory is None:
        snapshot_directory = snap_dir
    target_snap = os.path.join(snapshot_directory, *file)

    # Ensure that the path is an absolute path
    snapshot_directory = os.path.abspath(snapshot_directory)

    contra_data = ContraData.load(data, include_stats = False)
#    snapshot = contra_data.get_target_snapshot()#TODO: new_snapshot_type(target_snap)
    snapshot = new_snapshot_type(target_snap)

    if not los_only:
        snapshot_masks: Dict[ParticleType, np.ndarray[Tuple[int], np.dtype[np.bool_]]|None] = { p : None for p in particle_types }
    selected_ids: Dict[ParticleType, np.ndarray[Tuple[int], np.dtype[int]]|None] = { p : None for p in particle_types }
    for part_type in particle_types:
        snap_mask = np.full(snapshot.number_of_particles(part_type), True, dtype = np.bool_)
        contra_particle_data = contra_data.data[part_type]
        if min_halo_mass is not None:
            snap_mask &= contra_particle_data.halo_masses >= min_halo_mass
        if max_halo_mass is not None:
            snap_mask &= contra_particle_data.halo_masses <= max_halo_mass
        if min_scaled_halo_mass is not None:
            snap_mask &= contra_particle_data.halo_masses_scaled >= min_scaled_halo_mass
        if max_scaled_halo_mass is not None:
            snap_mask &= contra_particle_data.halo_masses_scaled <= max_scaled_halo_mass
        if min_ejection_distance is not None or max_ejection_distance is not None:
            ejection_distances = calculate_wrapped_distance(np.array(contra_particle_data.positions_pre_ejection, dtype = float), snapshot.get_positions(part_type).to("Mpc").value, snapshot.box_size[0].to("Mpc").value)
            if min_ejection_distance is not None:
                snap_mask &= ejection_distances >= min_ejection_distance
            if max_ejection_distance is not None:
                snap_mask &= ejection_distances <= max_ejection_distance
        if not los_only:
            snapshot_masks[part_type] = snap_mask
        selected_ids[part_type] = snapshot.get_IDs(part_type)[snap_mask]

    filters = ParticleFilterFile(
        date = start_date,
        filepath = output_filepath,
        contra_file = data,
        simulation_type = snapshot_type_string
    )

    #TODO: add some form of description of what limits were applied!
    filters.description = f"""\
min_halo_mass={min_halo_mass}
max_halo_mass={max_halo_mass}
min_scaled_halo_mass={min_scaled_halo_mass}
max_scaled_halo_mass={max_scaled_halo_mass}
min_ejection_distance={min_ejection_distance}
max_ejection_distance={max_ejection_distance}\
"""

    if not los_only:
        #TODO: get snap object (EAGLE | SWIFT)
        #TODO: get IDs
        #TODO: create mask using np.isin

        filters.snapshots_directory = snapshot_directory
#        filters.snapshot_filters[snapshot.redshift] = {}
        filters.snapshot_filters[snapshot.file_name] = {}
        for part_type in particle_types:
#            filters.snapshot_filters[snapshot.redshift][part_type] = SnapshotParticleFilter(
            filters.snapshot_filters[snapshot.file_name][part_type] = SnapshotParticleFilter(
                particle_type = part_type,
                redshift = snapshot.redshift,
                snapshot_number = snapshot.number,
                filepath = target_snap,
                allowed_ids = selected_ids[part_type],
                mask = snapshot_masks[part_type]
            )

    if los_directory is not None:
        filters.line_of_sight_directory = los_directory
        if is_EAGLE:
            with h5.File(recovered_eagle_los_ids, "r") as id_recovery_file:
                avalible_id_recovered_EAGLE_los_files = list(id_recovery_file.keys())
        for los_filepath in new_los_file_type.get_files(los_directory):
            los_file = new_los_file_type(los_filepath)
            if is_EAGLE and los_file.file_name not in avalible_id_recovered_EAGLE_los_files:
                Console.print_verbose_warning(f"No recovered particle ID data for line-of-sight file \"{los_file.file_name}\". Skipping...")
                continue
            Console.print_info(f"Generating filters for \"{los_file.file_name}\"")
#            filters.line_of_sight_filters[los_file.redshift] = {}
            filters.line_of_sight_filters[los_file.file_name] = {}
            if is_EAGLE:
                with h5.File(recovered_eagle_los_ids, "r") as id_recovery_file:
                    avalible_id_recovered_EAGLE_los = list(id_recovery_file[los_file.file_name].keys())
            for i in range(los_file.number_of_sightlines):
                if is_EAGLE and f"LOS{i}" not in avalible_id_recovered_EAGLE_los:
                    Console.print_error(f"No recovered particle ID data for LOS{i} from file \"{los_file.file_name}\".\nThis likley indicates ID recovery was incomplete.")
                    Console.print_warning("Skipping...")
                    break
#                filters.line_of_sight_filters[los_file.redshift][i] = {}
                filters.line_of_sight_filters[los_file.file_name][i] = {}
                for part_type in particle_types:
                    if part_type != ParticleType.gas:
                        continue # Implemented in case gas is not specified and allows for future extension
                    los_ids: np.ndarray
                    if is_SWIFT:
                        los = los_file.get_sightline(i, cache_data = False)
                        los_ids = los.IDs#TODO: add field
                    elif is_EAGLE:
                        with h5.File(recovered_eagle_los_ids, "r") as id_recovery_file:
                            los_ids = id_recovery_file[los_file.file_name][f"LOS{i}"][:]
                    else:
                        raise RuntimeError("Should not be possible! Please report.")
                    allowed_los_particles = np.isin(los_ids, selected_ids[part_type])
#                    filters.line_of_sight_filters[los_file.redshift][i][part_type] = LineOfSightParticleFilter(
                    filters.line_of_sight_filters[los_file.file_name][i][part_type] = LineOfSightParticleFilter(
                        particle_type = part_type,
                        redshift = los_file.redshift,
                        file_name = los_file.file_name,
                        line_of_sight_index = i,
                        filepath = los_filepath,
                        allowed_ids = los_ids[allowed_los_particles],
                        mask = allowed_los_particles
                    )

    filters.save()
