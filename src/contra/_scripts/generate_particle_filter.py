# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .. import VERSION, ParticleType, ArrayReorder, ArrayMapping, calculate_wrapped_distance
from .._L_star import get_L_star_halo_mass_of_z
#from ..io import ContraData, OutputReader, HeaderDataset, ParticleTypeDataset, SnapshotStatsDataset
from ..io import SnapshotBase, SnapshotEAGLE, SnapshotSWIFT, \
                ParticleFilterFile, SnapshotParticleFilter, LineOfSightParticleFilter, \
                LineOfSightFileEAGLE, LineOfSightFileSWIFT
from ..io._Output_Objects__forwards import ContraData, OutputReader, HeaderDataset, ParticleTypeDataset, SnapshotStatsDataset

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
from QuasarCode.MPI import MPI_Config, synchronyse

#contra-generate-particle-filter ../contra/EAGLE/L12N188/z3/contra-output.hdf5 --snapshots /users/aricrowe/EAGLE/L0012N0188/REFERENCE/data/ --lines-of-sight /users/aricrowe/EAGLE/L0012N0188/REFERENCE/data/los/ --los-only --EAGLE --gas --max-halo-mass 1000000000 --EAGLE-los-id-file ./output-test.hdf5 -v

Console.mpi_output_root_rank_only()

def main():
    ScriptWrapper(
        command = "contra-generate-particle-filter",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.2.0"),
        edit_date = datetime.date(2024, 9, 19),
        description = "Use halo search data to produce filters for both snapshots and line-of-sight files (for use with SpecWizard) to indicate which particles should be considered.",
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
            ScriptWrapper.Flag(
                name = "exclude-untracked-particles",
                description = "Particles that have no identified halo are excluded from the filter.\nIf this is unset, these particles will be forceably included (regardless of the actual placeholder value in the contra output)."
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
            ),
            ScriptWrapper.OptionalParam[int](
                name = "file-index",
                description = "Compute the filter(s) for a specific file.\nCan only be used when doing only one type of file.",
                conversion_function = int
            ),
            ScriptWrapper.OptionalParam[float](
                name = "los-file-selection-offset-fraction",
                description = "What chronological fraction of the expansion factor range between two snapshot files should be attributed to the halo data in the earlier file (for line-of-sight files).\nDefaults to 0.5 - i.e. half the range allocated to each file.",
                conversion_function = float,
                default_value = 0.5
            ),
            ScriptWrapper.OptionalParam[float](
                name = "bits-per-mask-int",
                description = "Use an integer datatype for the mask field with the precision specified.\nSupports values of 8, 16, 32 & 64.",
                conversion_function = int
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
            exclude_untracked_particles: bool,
            min_halo_mass: float|None,
            max_halo_mass: float|None,
            min_scaled_halo_mass: float|None,
            max_scaled_halo_mass: float|None,
            min_ejection_distance: float|None,
            max_ejection_distance: float|None,
            recovered_eagle_los_ids: str|None,
            file_index: int|None,
            los_file_selection_offset_fraction: float,
            bits_per_mask_int: int|None
          ) -> None:

    # Store the current date at the start in case the file writing occours after midnight.
    start_date = datetime.date.today()

    use_MPI: bool = MPI_Config.comm_size > 1
    if use_MPI:
        Console.print_info(f"Running with MPI on {MPI_Config.comm_size} ranks.")
        Console.print_debug(f"Root MPI rank is {MPI_Config.root}.")

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

    if file_index is not None:
        if not los_only and los_directory is not None:
            Console.print_error("Specifying a specific file index is only allowed when running on either snapshots or line-of-sight files exclusivley!")
            Console.print_info("Terminating...")
            return

    if bits_per_mask_int is not None and bits_per_mask_int not in (8, 16, 32, 64):
        Console.print_error(f"Invalid number of bits specified. Got {bits_per_mask_int} but can only be one of 8, 16, 32 & 64")
        Console.print_info("Terminating...")
        return

    # Ensure that the path is an absolute path
    output_filepath = os.path.abspath(output_filepath)

    able_to_procede: bool
    if MPI_Config.is_root:
        able_to_procede = True
        if os.path.exists(output_filepath):
            if not allow_overwrite:
                if file_index is not None:
                    Console.print_verbose_info("Writing data to existing file as a specific file index has been specified.")
                else:
                    Console.print_error("Output file already exists. Either remove it first or explicitley enable overwrite.")
                    Console.print_info("Terminating...")
                    synchronyse("able_to_procede")
                    return
            else:
                Console.print_warning("Pre-existing output file will be overwritten.")
        elif False:#TODO: check for valid file location (to prevent an error at the last minute!)
            pass
        synchronyse("able_to_procede")
    else:
        synchronyse("able_to_procede")
        if not able_to_procede:
            return

    Console.print_info("Identifying snapshot and catalogue files and loading halo search data.")

    if snapshot_directory is not None:
        # Ensure that the path is an absolute path
        snapshot_directory = os.path.abspath(snapshot_directory)

    # Load halo search data and sim data information
    contra_data = ContraData.load(data, include_stats = False, alternate_simulation_data_directory = snapshot_directory)

    Console.print_info("Preparing data storage object.")

    # Object to store results
    filters = ParticleFilterFile(
        date = start_date,
        filepath = output_filepath,
        allow_parallel_write = file_index is not None,
        contra_file = data,
        simulation_type = contra_data.header.simulation_type
    )

    filters.description = f"""\
exclude_untracked_particles={exclude_untracked_particles}
min_halo_mass={min_halo_mass}
max_halo_mass={max_halo_mass}
min_scaled_halo_mass={min_scaled_halo_mass}
max_scaled_halo_mass={max_scaled_halo_mass}
min_ejection_distance={min_ejection_distance}
max_ejection_distance={max_ejection_distance}\
"""
    
    Console.print_info(f"Filter settings:\n{filters.description}")

    # Define how to create a particle mask for a snapshot
    def create_mask(snapshot: SnapshotBase, halo_search_data: ParticleTypeDataset):

        snap_mask = np.full(len(halo_search_data.halo_ids), True, dtype = np.bool_)

        if min_halo_mass is not None:
            snap_mask &= halo_search_data.halo_masses >= min_halo_mass
        if max_halo_mass is not None:
            snap_mask &= halo_search_data.halo_masses <= max_halo_mass

        if min_scaled_halo_mass is not None:
            snap_mask &= halo_search_data.halo_masses_scaled >= min_scaled_halo_mass
        if max_scaled_halo_mass is not None:
            snap_mask &= halo_search_data.halo_masses_scaled <= max_scaled_halo_mass
    
        if min_ejection_distance is not None or max_ejection_distance is not None:
            ejection_distances = calculate_wrapped_distance(np.array(halo_search_data.positions_pre_ejection, dtype = float), snapshot.get_positions(part_type).to("Mpc").value, snapshot.box_size[0].to("Mpc").value)
            if min_ejection_distance is not None:
                snap_mask &= ejection_distances >= min_ejection_distance
            if max_ejection_distance is not None:
                snap_mask &= ejection_distances <= max_ejection_distance

        if exclude_untracked_particles:
            # Only allow particles that are tracked
            snap_mask &= halo_search_data.halo_ids != -1
        else:
            # Force untracked particles to appear in the mask
            snap_mask |= halo_search_data.halo_ids == -1

        return snap_mask

    # Avalible snapshots by paricle type
    snapshot_nums_by_part_type: dict[ParticleType, tuple[str, ...]] = { p : tuple(contra_data.data[p].keys()) for p in contra_data.data.keys() if p in particle_types }

    if not los_only:
        # Do snapshots/snipshots

        Console.print_info("Doing snapshot filters.")

        filters.snapshots_directory = snapshot_directory

        for part_type, avalible_snapshot_numbers in snapshot_nums_by_part_type.items():
            for snapshot_number in avalible_snapshot_numbers if file_index is None else (avalible_snapshot_numbers[file_index] if len(avalible_snapshot_numbers) > file_index else tuple()):

                Console.print_info(f"    Snapshot {snapshot_number}.")

                filters.snapshot_filters[snapshot.file_name] = {}

                search_data: ParticleTypeDataset = contra_data.data[part_type][snapshot_number]
                snapshot: SnapshotBase = contra_data.get_snapshot(snapshot_number)

                mask = create_mask(snapshot, search_data)

                filters.snapshot_filters[snapshot.file_name][part_type] = SnapshotParticleFilter(
                    particle_type = part_type,
                    redshift = snapshot.redshift,
                    snapshot_number = snapshot.number,
                    filepath = snapshot.filepath,
                    allowed_ids = np.copy(snapshot.get_IDs(part_type)[mask]), # Copy the subset so the whole array isn't retained in memory!
                    mask = mask
                )

    if los_directory is not None:
        # Do line-of-sight files
        
        Console.print_info("Doing line-of-sight filters.")

        filters.line_of_sight_directory = los_directory

        if is_EAGLE:
            # Get information about which line-of-sight files have recovered IDs
            with h5.File(recovered_eagle_los_ids, "r") as id_recovery_file:
                avalible_id_recovered_EAGLE_los_files = list(id_recovery_file.keys())

        los_filepaths = new_los_file_type.get_files(los_directory)
        if file_index is not None:
            if file_index >= len(los_filepaths):
                los_filepaths = tuple()
            else:
                los_filepaths = (los_filepaths[file_index], )

        for los_filepath in los_filepaths:

            los_file = new_los_file_type(los_filepath)

            if is_EAGLE and los_file.file_name not in avalible_id_recovered_EAGLE_los_files:
                Console.print_verbose_warning(f"No recovered particle ID data for line-of-sight file \"{los_file.file_name}\". Skipping...")
                continue

            # Find closest snapshot number
            snapshots = contra_data.simulation_files.snipshots if is_EAGLE else contra_data.simulation_files.snapshots
            snapshot_redshifts = np.array([(s.tag_redshift if is_EAGLE else s.load().redshift) for s in snapshots], dtype = float)
            snapshot_expansion_factors = 1 / (1 + snapshot_redshifts)
            neighbouring_high_redshift_snapshot_index = np.where(snapshot_redshifts > los_file.redshift)[0][-1]
            neighbouring_low_redshift_snapshot_index = neighbouring_high_redshift_snapshot_index + 1
            #high_redshift_distance = np.abs(np.log(1 + snapshot_redshifts[neighbouring_high_redshift_snapshot_index]) - np.log(1 + los_file.redshift))
            #low_redshift_distance = np.abs(np.log(1 + snapshot_redshifts[neighbouring_low_redshift_snapshot_index]) - np.log(1 + los_file.redshift))
            high_redshift_distance = np.abs(snapshot_expansion_factors[neighbouring_high_redshift_snapshot_index] - (1 / (1 + los_file.redshift)))
            low_redshift_distance = np.abs(snapshot_expansion_factors[neighbouring_low_redshift_snapshot_index] - (1 / (1 + los_file.redshift)))
            closest_snapshot_number: str = snapshots.get_numbers()[neighbouring_high_redshift_snapshot_index if high_redshift_distance / (high_redshift_distance + low_redshift_distance) <= los_file_selection_offset_fraction else neighbouring_low_redshift_snapshot_index]
            if los_only:
                snapshot = contra_data.get_snapshot(closest_snapshot_number)

            if not any([closest_snapshot_number in contra_data.data[p] for p in snapshot_nums_by_part_type]):
                Console.print_verbose_warning(f"No contra data for snapshot ({closest_snapshot_number}) matching line-of-sight file \"{los_file.file_name}\". Skipping...")
                continue

            Console.print_info(f"Generating filters for \"{los_file.file_name}\" using halo search data for", ("snapshot" if not is_EAGLE else "snipshot"), f"{closest_snapshot_number}.")

            filters.line_of_sight_filters[los_file.file_name] = {}

            if is_EAGLE:
                # Get information about which line-of-sight datasets have IDs avalible
                with h5.File(recovered_eagle_los_ids, "r") as id_recovery_file:
                    avalible_id_recovered_EAGLE_los = list(id_recovery_file[los_file.file_name].keys())

            for i in range(los_file.number_of_sightlines):

                if is_EAGLE and f"LOS{i}" not in avalible_id_recovered_EAGLE_los:
                    Console.print_error(f"No recovered particle ID data for LOS{i} from file \"{los_file.file_name}\".\nThis likley indicates ID recovery was incomplete.")
                    Console.print_warning("Skipping...")
                    break

                filters.line_of_sight_filters[los_file.file_name][i] = {}

                for part_type in snapshot_nums_by_part_type:
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

                    allowed_particle_ids: np.ndarray
                    if not los_only:
                        # Read from data already in memory to save re-computing a mask
                        allowed_particle_ids = filters.snapshot_filters[closest_snapshot_number][part_type].allowed_ids
                    else:
                        allowed_particle_ids = snapshot.get_IDs(part_type)[create_mask(
                            snapshot,
                            contra_data.data[part_type][closest_snapshot_number]
                        )]

                    allowed_los_particles = np.isin(los_ids, allowed_particle_ids)
                    filters.line_of_sight_filters[los_file.file_name][i][part_type] = LineOfSightParticleFilter(
                        particle_type = part_type,
                        redshift = los_file.redshift,
                        file_name = los_file.file_name,
                        line_of_sight_index = i,
                        filepath = los_filepath,
                        # The elements of this array may not be unique for EAGLE data due to the ID recovery procedure producing duplicate particlre IDs
                        allowed_ids = np.copy(los_ids[allowed_los_particles]),
                        mask = allowed_los_particles
                    )

    Console.print_info("Saving data...", end = "")

    filters.save(use_mask_datatype = None if bits_per_mask_int is None else np.int8 if bits_per_mask_int == 8 else np.int16 if bits_per_mask_int == 16 else np.int32 if bits_per_mask_int == 32 else np.int64)

    print("done")


















'''
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

    contra_data = ContraData.load(data, include_stats = False, alternate_simulation_data_directory = snapshot_directory)

    contra_data.get_snapshot(snap_num)
    contra_data.gas[snap_num]



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
'''
