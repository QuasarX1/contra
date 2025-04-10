# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None

import datetime
import os
from typing import cast, Union, Generic, TypeVar
from collections.abc import Callable, Iterable
import asyncio

import numpy as np
from unyt import unyt_quantity, unyt_array
from QuasarCode import Settings, Console, Stopwatch
from QuasarCode.MPI import MPI_Config, mpi_barrier, synchronyse, mpi_get_slice, mpi_gather_array, mpi_scatter_array
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import ScriptWrapper

from astro_sph_tools import ParticleType
from astro_sph_tools.io.data_structures import SnapshotBase, CatalogueBase
from astro_sph_tools.tools import ArrayReorder_MPI_2
from astro_sph_tools.io.EAGLE import FileTreeScraper_EAGLE, SnapshotEAGLE, CatalogueSUBFIND

from .. import VERSION, SharedArray, SharedArray_TransmissionData, SharedArray_Shepherd, SharedArray_ParallelJob
from .._L_star import get_L_star_halo_mass_of_z
from ..io import SnapshotSWIFT, CatalogueSOAP#, OutputWriter, OutputReader, HeaderDataset, ParticleTypeDataset, SnapshotStatsDataset, CheckpointData
from ..io._Output_Objects import OutputWriter, OutputReader, DistributedOutputReader, HeaderDataset, ParticleTypeDataset, SnapshotStatsDataset

import gc

Console.mpi_output_root_rank_only()
Console.show_times()

USE_DEBUGGING_CHECKS = False

#SnapshotEAGLE.set_pyread_eagle_to_verbose()
CatalogueSUBFIND.LimitedMode = True # This is to stop unnecessary IO when using the large datasets from slowing the whole process down!



def main():
    ScriptWrapper(
        command = "contra-run-complete",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2025, 2, 10),
        description = "Itterate forwards in time through the snapshots and record the properties of the last halo a particle encountered for each snapshot.",
        parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.OptionalParam[str](
                name = "snapshots",
                sets_param = "snapshot_directory",
                default_value = ".",
                description = "Where to search for snapshots.\nDefaults to \".\" (i.e. the current directory)."
            ),
            ScriptWrapper.OptionalParam[str | None](
                name = "snapshot-basename",
                description = "Prefix to apply to all snapshot file numbers.\nMay be omitted if the snapshot directory contains only one set of snapshots.\nNot supported when using EAGLE data",
                conflicts = ["EAGLE"]
            ),
            ScriptWrapper.OptionalParam[str | None](
                name = "catalogue",
                sets_param = "catalogue_directory",
                description = "Where to search for catalogue files.\nDefaults to the same directory as the snapshots."
            ),
            ScriptWrapper.OptionalParam[str | None](
                name = "catalogue-membership-basename",
                description = "Prefix to apply to all catalogue membership file names.\nOnly required if the method of auto-resolving the filenames fails."
            ),
            ScriptWrapper.OptionalParam[str | None](
                name = "catalogue-properties-basename",
                description = "Prefix to apply to all catalogue properties file names.\nOnly required if the method of auto-resolving the filenames fails."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "skip-files",
                sets_param = "skip_file_numbers",
                conversion_function = ScriptWrapper.make_list_converter(","),
                default_value = [],
                description = "Numbers of file numbers to skip (of the snapshot or snipshot type being targeted)."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "skip-snapshot-files",
                sets_param = "skip_snapshot_numbers",
                conversion_function = ScriptWrapper.make_list_converter(","),
                default_value = [],
                description = "Snapshot file numbers to exclude when finding data.\nUse this to mask out any corrupted or missing snapshots."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "skip-snipshot-files",
                sets_param = "skip_snipshot_numbers",
                conversion_function = ScriptWrapper.make_list_converter(","),
                default_value = [],
                description = "Snipshot file numbers to exclude when finding data.\nUse this to mask out any corrupted or missing snipshots."
            ),
            ScriptWrapper.Flag(
                name = "use-snipshots",
                description = "Set this flag to use snipshot files instead of snapshots."
            ),
            ScriptWrapper.OptionalParam[str|None](
                name = "start-file",
                description = "Number of file to start from if not the first file.\nIncompatible with --start-file.",
                conflicts = ["start-redshift"]
            ),
            ScriptWrapper.OptionalParam[float|None](
                name = "start-redshift",
                description = "Redshift to start from if not the first file.\nThis will use the last file with a redshift at or exceeding this value.\nIncompatible with --start-file.",
                conversion_function = float,
                conflicts = ["start-file"]
            ),
            ScriptWrapper.Flag(
                name = "EAGLE",
                sets_param = "is_EAGLE",
                description = "Running on EAGLE data.",
                conflicts = ["is_SWIFT", "snapshot-basename"]
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
                default_value = "contra-output.hdf5",
                description = "File to save results to."
            ),
            ScriptWrapper.Flag(
                name = "allow-overwrite",
                description = "Allow an existing output file to be overwritten."
            ),
            ScriptWrapper.Flag(
                name = "restart",
                description = "Use checkpoints to restart the search from the latest avalible checkpoint.\nThis requires the output file to be explicitly specified with overwriting and checkpointing enabled.",
#                requirements = ["enable-checkpointing", "output-file", "allow-overwrite"]#TODO: fix
            )
        )
    ).run_with_async(__main)



async def __main(
#            target_snapshot: str,
            snapshot_directory: str,
            snapshot_basename: Union[str, None],
            catalogue_directory: Union[str, None],
            catalogue_membership_basename: Union[str, None],
            catalogue_properties_basename: Union[str, None],
            skip_file_numbers: list[str],
            skip_snapshot_numbers: list[str],
            skip_snipshot_numbers: list[str],
            use_snipshots: bool,
            start_file: str|None,
            start_redshift: float|None,
            is_EAGLE: bool,
            is_SWIFT: bool,
            do_gas: bool,
            do_stars: bool,
            do_black_holes: bool,
            do_dark_matter: bool,
            output_filepath: str,
            allow_overwrite: bool,
            restart: bool
          ) -> None:

    # Store the current date at the start in case the file writing occours after midnight.
    start_date = datetime.date.today()

    # Identify if MPI is avalible and if more than one rank is avalible
    if not Settings.mpi_avalible:
        raise ModuleNotFoundError("Unable to find support for MPI.")
    Console.print_info(f"Running with MPI on {MPI_Config.comm_size} ranks.")
    Console.print_debug(f"Root MPI rank is {MPI_Config.root}.")

    if skip_file_numbers is not None:
        if not use_snipshots:
            if skip_snapshot_numbers is not None:
                skip_snapshot_numbers.extend(skip_file_numbers)
                skip_file_numbers = skip_snapshot_numbers
            else:
                skip_snapshot_numbers = skip_file_numbers
        else:
            if skip_snipshot_numbers is not None:
                skip_snipshot_numbers.extend(skip_file_numbers)
                skip_file_numbers = skip_snipshot_numbers
            else:
                skip_snipshot_numbers = skip_file_numbers
    elif not use_snipshots:
        skip_file_numbers = skip_snapshot_numbers
    else:
        skip_file_numbers = skip_snipshot_numbers

    # Validate sim type
    if not (is_EAGLE or is_SWIFT):
        Console.print_error("Must specify either EAGLE or SWIFT simulation type.")
        return
    else:
        if is_EAGLE:
            Console.print_verbose_info("Snapshot type: EAGLE")
        elif is_SWIFT:
            Console.print_verbose_info("Snapshot type: SWIFT")

    # Validate particle types
    #     If none are specified, use all types
    Console.print_verbose_info("Particle Types:")
    if not (do_gas or do_stars or do_black_holes or do_dark_matter):
        Console.print_verbose_warning("    No particle type(s) specified. Enabling all particle types.")
        do_gas = do_stars = do_black_holes = do_dark_matter = True
    particle_types: list[ParticleType] = []
    if do_gas:
        particle_types.append(ParticleType.gas)
        Console.print_verbose_info("    Tracking gas particles")
    if do_stars:
        particle_types.append(ParticleType.star)
        Console.print_verbose_info("    Tracking star particles")
    if do_black_holes:
        particle_types.append(ParticleType.black_hole)
        Console.print_verbose_info("    Tracking black hole particles")
    if do_dark_matter:
        particle_types.append(ParticleType.dark_matter)
        Console.print_verbose_info("    Tracking dark matter particles")

    Console.print_info("Identifying snapshot and catalogue files")

    # Ensure that the path is an absolute path
    snapshot_directory = os.path.abspath(snapshot_directory)

    # Get snapshot file information
    simulation_file_scraper = FileTreeScraper_EAGLE(
        snapshot_directory,
        skip_snapshot_numbers = skip_snapshot_numbers,
        skip_snipshot_numbers = skip_snipshot_numbers
    )# if is_EAGLE else FileTreeScraper_SWIFT(snapshot_directory, skip_snapshot_numbers = skip_file_numbers)
    N_snapshots = len(simulation_file_scraper.snipshots if use_snipshots else simulation_file_scraper.snapshots)

    # Check for start file
    if start_file is not None:
        if start_file not in (simulation_file_scraper.snipshots if use_snipshots else simulation_file_scraper.snapshots).get_numbers():
            Console.print_debug((simulation_file_scraper.snipshots if use_snipshots else simulation_file_scraper.snapshots).get_numbers())
            Console.print_error(f"Unable to find a sn{'i' if use_snipshots else 'a'}pshot with a number matching \"{start_file}\".")
            return
    elif start_redshift is not None:
        start_file = (simulation_file_scraper.snipshots if use_snipshots else simulation_file_scraper.snapshots).find_file_number_from_redshift(start_redshift)
        Console.print_info(f"Found file number {start_file} for start redshift {start_redshift}.")

    # Ensure that the path is an absolute path
    output_filepath = os.path.abspath(output_filepath)

    # Validate the output file and exit all ranks if in an unacceptable state
    able_to_procede: bool
    if MPI_Config.is_root:
        able_to_procede = True
        if os.path.exists(output_filepath + f".{MPI_Config.rank}"):
            if not (allow_overwrite or restart):
                able_to_procede = False
                synchronyse("able_to_procede")
                Console.print_error("Output file already exists. Either remove it first or explicitley enable overwrite.")
                return
            elif allow_overwrite:
                Console.print_warning("Pre-existing output file will be overwritten")
        elif False:#TODO: check for valid file location (to prevent an error at the last minute!)
            pass
        synchronyse("able_to_procede")
    else:
        synchronyse("able_to_procede")
        if not able_to_procede:
            return

    # Create output file
#    if not use_MPI or MPI_Config.is_root:
    output_file = OutputWriter(output_filepath + f".{MPI_Config.rank}", overwrite = not restart)

    # Define variables used to pass data between searches
    snapshot: SnapshotBase
    catalogue: CatalogueBase
    shapshot_particle_ids:                  np.ndarray|None = None
    snapshot_last_halo_ids:                 np.ndarray|None = None
    snapshot_last_halo_masses:              np.ndarray|None = None
#    snapshot_last_halo_masses_scaled:       np.ndarray|None = None
    snapshot_last_halo_redshifts:           np.ndarray|None = None
    snapshot_last_halo_particle_positions:  np.ndarray|None = None

    write_header: bool = not restart
    if restart:
        # It is possible to have blank files created but not populated with a header due to an error.
        # Check that a header can be read and if not, write the header.
        wrong_number_of_files: bool
        if MPI_Config.is_root:
            wrong_number_of_files = False
            try:
                with DistributedOutputReader(output_filepath, map_to_mpi = True) as output_file_reader:
                    if output_file_reader.number_of_files != MPI_Config.comm_size:
                        wrong_number_of_files = True
                    if not wrong_number_of_files:
                        output_file_reader.read_header() # This may generate an error if the header is missing
            except:
                Console.print_warning("Unable to read header. Header group will be created.")
                write_header = True
        synchronyse("write_header")
        synchronyse("wrong_number_of_files")
        if wrong_number_of_files:
            if MPI_Config.is_root:
                Console.print_error(f"Restarting from {output_file_reader.number_of_files} files but using {MPI_Config.comm_size} ranks.\nThe number of ranks MUST match the number of files when restarting!")
            return

    # Create the output file and header (if not restarting)
#    if not use_MPI or MPI_Config.is_root:
    if write_header:
        if not restart:
            Console.print_info("Creating file and writing header...", end = "")
        else:
            Console.print_info("Writing missing header...", end = "")

        with output_file:
            output_file.write_header(HeaderDataset(
                version = VersionInfomation.from_string(VERSION),
                date = start_date,
                simulation_type = "SWIFT" if is_SWIFT else "EAGLE",
                simulation_directory = snapshot_directory,
                N_searched_snapshots = N_snapshots,
                uses_snipshots = use_snipshots,
                output_file = output_filepath,
                has_gas = do_gas,
                has_stars = do_stars,
                has_black_holes = do_black_holes,
                has_dark_matter = do_dark_matter,
                has_statistics = False,
                skipped_file_numbers = tuple(skip_file_numbers)
            ))

        Console.print_raw("done", flush = True)

    else:
        # It is possible to have blank files created but not populated with a header due to an error.
        # Check that a header can be read and if not, raise an error and terminate.
        pass

    Console.print_debug("Waiting at MPI barrier.")
    mpi_barrier()
    Console.print_debug("Continuing...")

    # Initialise the search dispatcher object
    # This allows the search to be parallelised on the current rank over haloes with one halo per subprocess
    searcher = SnapshotSearcher(64, SnapshotEAGLE, CatalogueSUBFIND) if is_EAGLE else SnapshotSearcher(64, SnapshotSWIFT, CatalogueSOAP)

    # Itterate over selected particle types
    for particle_type in particle_types:

        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")

        Console.print_info(f"Running search for {particle_type.name} particles.")

        if restart:
            # Load data for current state

            last_completed_numerical_file_number: int

            Console.print_info("Reading checkpoint data:")

            # Read checkpoint data
            with DistributedOutputReader(output_filepath, map_to_mpi = True) as output_file_reader:

                Console.print_info("    Loading data...", end = "")

                checkpoint = output_file_reader.read_checkpoint(particle_type)

            if checkpoint is not None:

                snapshot_last_halo_ids = checkpoint.halo_ids
                snapshot_last_halo_masses = checkpoint.halo_masses
#                snapshot_last_halo_masses_scaled = checkpoint.halo_masses_scaled
                snapshot_last_halo_redshifts = checkpoint.redshifts
                snapshot_last_halo_particle_positions = checkpoint.positions_pre_ejection

                last_completed_numerical_file_number = int(checkpoint.file_number)

                shapshot_particle_ids = (simulation_file_scraper.snipshots if use_snipshots else simulation_file_scraper.snapshots).get_by_number(checkpoint.file_number).load().get_IDs(particle_type)

                Console.print_raw("done")

                Console.print_info(f"    Restarting from {'snapshot' if not use_snipshots else 'snipshot'} {last_completed_numerical_file_number + 1}/{N_snapshots}")

            else:
                last_completed_numerical_file_number = -1
                Console.print_raw("failed - no data to restart from.")

            Console.print_debug("Waiting at MPI barrier.")
            mpi_barrier()
            Console.print_debug("Continuing...")

#        with Stopwatch("Search") as stopwatch:
#            barrier_time = stopwatch.get_elapsed_time_lap()
#            mpi_barrier()
#            all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
#            if MPI_Config.is_root:
#                Console.print_debug("Search barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

        start_index: int
        if start_file is not None:
            start_index = (simulation_file_scraper.snipshot_catalogues if use_snipshots else simulation_file_scraper.catalogues).get_numbers().index(start_file)
        else:
            start_index = 0

        aggregate_times = { "IO" : 0.0, "COMPUTE" : 0.0, "CONSTRUCTION" : 0.0, "DEAD" : 0.0, None : 0.0 }

        for catalogue_info in (simulation_file_scraper.snipshot_catalogues if use_snipshots else simulation_file_scraper.catalogues)[start_index:]:

            # Timing recording
            stopwatch = Stopwatch.start_new("Search")
            time_labels: list[str] = []
            time_categories: list[str] = [] # "IO" | "COMPUTE" | "CONSTRUCTION" | "DEAD" | None
            start_times: list[float] = []
            end_times: list[float] = []
            def t_start(label, io = False, compute = False, construction = False, dead = False):
                time_labels.append(label)
                time_categories.append("IO" if io else "COMPUTE" if compute else "CONSTRUCTION" if construction else "DEAD" if dead else None)
                start_times.append(stopwatch.get_elapsed_time())
            def t_end():
                end_times.append(stopwatch.get_elapsed_time())

#                barrier_time = stopwatch.get_elapsed_time_lap()
#                mpi_barrier()
#                all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
#                if MPI_Config.is_root:
#                    Console.print_debug("Search barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

            if restart and catalogue_info.number_numerical <= last_completed_numerical_file_number:
                Console.print_info(f"{catalogue_info.number} already complete. Skipping." if not is_EAGLE else f"{catalogue_info.number} (redshift {catalogue_info.tag_redshift}) already complete. Skipping.")
                del stopwatch
                continue
            Console.print_info(f"Doing {catalogue_info.number}" if not is_EAGLE else f"Doing {catalogue_info.number} (redshift {catalogue_info.tag_redshift})")

            t_start("Dead time before loading IO objects", dead = True)
            Console.print_debug("Waiting at MPI barrier.")
            mpi_barrier()
            Console.print_debug("Continuing...")
            t_end()

            Console.print_verbose_info("    Loading Snapshot and catalogue objects.")

            t_start("Loading custom IO helper objects.", construction = True)
            catalogue = catalogue_info.load()
            snapshot = catalogue.snapshot
            t_end()

            t_start("Dead time after loading IO objects", dead = True)
            Console.print_debug("Waiting at MPI barrier.")
            mpi_barrier()
            Console.print_debug("Continuing...")
            t_end()
            t_start("")

            if not is_EAGLE:
                Console.print_info(f"    Redshift {catalogue.redshift}")
            if len(catalogue) == 0:
                Console.print_info("    No halos. Skipping.")
                del stopwatch
                continue
            Console.print_info(f"    Catalogue has {len(catalogue)} haloes.")

#                barrier_time = stopwatch.get_elapsed_time_lap()
#                mpi_barrier()
#                all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
#                if MPI_Config.is_root:
#                    Console.print_debug("Search barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

            Console.print_info("    Searching for halo membership.")
            t_end()
            t_start("Calling searcher")

            (
                shapshot_particle_ids,
                snapshot_last_halo_ids,
                snapshot_last_halo_masses,
#                snapshot_last_halo_masses_scaled,
                snapshot_last_halo_redshifts,
                snapshot_last_halo_particle_positions
            ) = searcher(
                particle_type,
                catalogue,
                shapshot_particle_ids,
                snapshot_last_halo_ids,
                snapshot_last_halo_masses,
#                snapshot_last_halo_masses_scaled,
                snapshot_last_halo_redshifts,
                snapshot_last_halo_particle_positions,
                t_start, t_end
            )

            t_end()
            t_start("")

#                barrier_time = stopwatch.get_elapsed_time_lap()
#                mpi_barrier()
#                all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
#                if MPI_Config.is_root:
#                    Console.print_debug("Search barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

            Console.print_verbose_info("    Creating output struct.")
            t_end()

            t_start("Creating writer struct", construction = True)
            results = ParticleTypeDataset(
                particle_type = particle_type,
                target_redshift = catalogue.redshift,
                file_number = catalogue_info.number,
                length = snapshot.number_of_particles(ParticleType.gas),
                length_this_rank = snapshot.number_of_particles_this_rank(ParticleType.gas),
                redshifts = snapshot_last_halo_redshifts,
                halo_ids = snapshot_last_halo_ids,
                halo_masses = snapshot_last_halo_masses,
#                halo_masses_scaled = snapshot_last_halo_masses_scaled,
                positions_pre_ejection = snapshot_last_halo_particle_positions
            )
            t_end()

            t_start("Opening output files", io = True)
            with output_file:
                t_end()
                t_start("Writing data", io = True)

                Console.print_verbose_info("    Writing data.")

                output_file.write_particle_type_dataset(results)
                t_end()
                t_start("Executing transaction?")
            t_end()

            t_start("Waiting for barrier", dead = True)
            Console.print_debug("Waiting at MPI barrier.")
            mpi_barrier()
            Console.print_debug("Continuing...")
            t_end()

            # Timing output
            if len(end_times) != len(start_times) or len(end_times) != len(time_labels) or len(end_times) != len(time_categories):
                Console.print_verbose_error("Timing lists have lengths that don't match. Dumping raw data as a precaution:")
                Console.print_debug("Labels:", time_labels)
                Console.print_debug("Categories:", time_categories)
                Console.print_debug("Start Times:", start_times)
                Console.print_debug("End Times:", end_times)
            else:
                for i in range(len(time_labels)):
                    Console.print_verbose_info(f"({(time_categories[i] if time_categories[i] is not None else ""):12s}) {end_times[i] - start_times[i]:9.2f} {time_labels[i]}")
                total_io = sum([end_times[i] - start_times[i] for i in range(len(time_labels)) if time_categories[i] == "IO"])
                total_compute = sum([end_times[i] - start_times[i] for i in range(len(time_labels)) if time_categories[i] == "COMPUTE"])
                total_construction = sum([end_times[i] - start_times[i] for i in range(len(time_labels)) if time_categories[i] == "CONSTRUCTION"])
                total_dead = sum([end_times[i] - start_times[i] for i in range(len(time_labels)) if time_categories[i] == "DEAD"])
                total_misc = sum([end_times[i] - start_times[i] for i in range(len(time_labels)) if time_categories[i] == None])
                aggregate_times["IO"] += total_io
                aggregate_times["COMPUTE"] += total_compute
                aggregate_times["CONSTRUCTION"] += total_construction
                aggregate_times["DEAD"] += total_dead
                aggregate_times[None] += total_misc
                #Console.mpi_output_all_ranks()
                Console.print_verbose_info(f"Timing results:\nIO:           {total_io:9.2f} s ({aggregate_times["IO"]:9.2f} s)\nCOMPUTE:      {total_compute:9.2f} s ({aggregate_times["COMPUTE"]:9.2f} s)\nCONSTRUCTION: {total_construction:9.2f} s ({aggregate_times["CONSTRUCTION"]:9.2f} s)\nDEAD:         {total_dead:9.2f} s ({aggregate_times["DEAD"]:9.2f} s)\nMISC.:        {total_misc:9.2f} s ({aggregate_times[None]:9.2f} s)")
                #Console.mpi_output_root_rank_only()

            stopwatch.stop()
            del stopwatch

#            with DistributedOutputReader(output_filepath, map_to_mpi = True) as output_file_reader:
#                file_data = output_file_reader.read_particle_type_dataset(ParticleType.gas, catalogue_info.number)
#                if (file_data.halo_ids != snapshot_last_halo_ids).sum() > 0:
#                    print(f"Fail on rank {MPI_Config.rank}.", (file_data.halo_ids != snapshot_last_halo_ids).sum(), snapshot_last_halo_ids.shape[0])
##                print(MPI_Config.rank, (d["HaloID"][rank_slice] != data.halo_ids).sum(), flush = True)
#                collected_inputs = mpi_gather_array(snapshot_last_halo_ids)
#                collected_file_results = mpi_gather_array(file_data.halo_ids)
#                if MPI_Config.is_root:
##                    collected_inputs = np.concatenate(collected_inputs)
##                    collected_file_results = np.concatenate(collected_file_results)
#                    Console.print_raw("Written data != input data:", (collected_file_results != collected_inputs).sum())

#            stopwatch.lap()

#            break
    
#            stopwatch.print_lap()

#            barrier_time = stopwatch.get_elapsed_time_lap()
#            mpi_barrier()
#            all_barrier_times = MPI_Config.comm.gather(barrier_time, root = MPI_Config.root)
#            if MPI_Config.is_root:
#                Console.print_debug("Search barrier delay:", ", ".join([f"{v - min(all_barrier_times):.1f}" for v in all_barrier_times]))

    Console.print_info("DONE")

import hashlib

T_snapshot = TypeVar("T_snapshot", bound = SnapshotBase)
T_catalogue = TypeVar("T_catalogue", bound = CatalogueBase)
class SnapshotSearcher(Generic[T_snapshot, T_catalogue]):

    def __init__(self, number_of_workers: int, snapshot_type: type[T_snapshot], catalogue_type: type[T_catalogue]):#, mass_of_L_star_at_z: Callable[[float], float]):

        self.__number_of_workers = number_of_workers
        self.__snapshot_type = snapshot_type
        self.__catalogue_type = catalogue_type

#        self.__get_mass_of_L_star = mass_of_L_star_at_z

    def __call__(
            self,
            particle_type: ParticleType,
            catalogue: T_catalogue,
            prior_particle_ids: np.ndarray|None,
            prior_particle_last_halo_ids: np.ndarray|None,
            prior_particle_last_halo_masses: np.ndarray|None,
#            prior_particle_last_halo_masses_scaled: np.ndarray|None,
            prior_particle_last_halo_redshifts: np.ndarray|None,
            prior_particle_last_halo_positions: np.ndarray|None,
            t_start: Callable[[str], None], t_end: Callable[[], None]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        t_end()
        t_start("")
        
        if not isinstance(catalogue, self.__catalogue_type):
            raise TypeError(f"Unexpected catalogue type {type(catalogue).__name__}. Expected {self.__catalogue_type.__name__}.")
        if not isinstance(catalogue.snapshot, self.__snapshot_type):
            raise TypeError(f"Unexpected snapshot type {type(catalogue.snapshot).__name__}. Expected {self.__snapshot_type.__name__}.")

        Console.print_verbose_info("    Reading snapshot data.")
        
        snapshot = catalogue.snapshot

        t_end()

        t_start("Dead time before IO", dead = True)
        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")
        t_end()

        t_start("snapshot.number_of_particles_this_rank", io = True)
        n_particles = snapshot.number_of_particles_this_rank(particle_type)
        t_end()

        t_start("Dead time between IO", dead = True)
        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")
        t_end()

        t_start("snapshot.get_IDs", io = True)
        snapshot_particle_ids = snapshot.get_IDs(particle_type)
        t_end()

        t_start("Dead time after IO", dead = True)
        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")
        t_end()
#        if USE_DEBUGGING_CHECKS:
#            _gathered_data = mpi_gather_array(snapshot_particle_ids)
#            if MPI_Config.is_root:
#    #            _gathered_data = np.concatenate(_gathered_data)
#                Console.print_raw(f"ParticleIDs Array Hash:\n{hashlib.sha256(_gathered_data.tobytes()).hexdigest()}") # 9a6ec (L100@000)

        t_start("Creating initial data arrays", construction = True)
        Console.print_verbose_info("    Allocating result array memory.")

        Console.print_verbose_info("        (last halo ids)", verbosity_level = 1)
        snapshot_particle_last_halo_ids = np.empty(n_particles, dtype = np.int64)
        Console.print_verbose_info("        (last halo masses)", verbosity_level = 1)
        snapshot_particle_last_halo_masses = np.empty(n_particles, dtype = np.float64)
#        Console.print_verbose_info("        (last halo scaled masses)", verbosity_level = 1)
#        snapshot_particle_last_halo_masses_scaled = np.empty(n_particles, dtype = np.float64)
        Console.print_verbose_info("        (last halo redshifts)", verbosity_level = 1)
        snapshot_particle_last_halo_redshifts = np.empty(n_particles, dtype = np.float64)
        Console.print_verbose_info("        (last halo particle positions)", verbosity_level = 1)
        snapshot_particle_last_halo_positions = np.empty((n_particles, 3), dtype = np.float64)
        Console.print_verbose_info("        (done)", verbosity_level = 1)

        t_end()

        if prior_particle_ids is None:
            # This is the first snapshot

            t_start("Filling initial data - no reorder", construction = True)

            Console.print_verbose_info("    No previous results. Filling result arrays with placeholder data.")

            snapshot_particle_last_halo_ids.fill(-1)
            snapshot_particle_last_halo_masses.fill(np.nan)
#            snapshot_particle_last_halo_masses_scaled.fill(np.nan)
            snapshot_particle_last_halo_redshifts.fill(np.nan)
            snapshot_particle_last_halo_positions.fill(np.nan)

            t_end()

        else:
            # Reorganise existing data

            t_start("Pre-reorder dead time", dead = True)
            Console.print_debug("Waiting at MPI barrier.")
            mpi_barrier()
            Console.print_debug("Continuing...")
            t_end()

            t_start("Creating reorder", construction = True)

            Console.print_verbose_info("    Calculating reorder for previous result data.")

            #transition_to_new_order = ArrayReorder_MPI.create(prior_particle_ids, snapshot_particle_ids)
            transition_to_new_order = ArrayReorder_MPI_2.create(prior_particle_ids, snapshot_particle_ids)

            t_end()

#            if USE_DEBUGGING_CHECKS:
#                test_mpi_reorder = transition_to_new_order(prior_particle_ids)
#                test_mpi_reorder2 = transition_to_new_order(prior_particle_last_halo_ids)
#
#                all_prior_particle_ids = mpi_gather_array(prior_particle_ids)
#                all_prior_dataset = mpi_gather_array(prior_particle_last_halo_ids)
#                all_snapshot_particle_ids = mpi_gather_array(snapshot_particle_ids)
#                all_test_mpi_reorder = mpi_gather_array(test_mpi_reorder)
#                all_test_mpi_reorder2 = mpi_gather_array(test_mpi_reorder2)
#                if MPI_Config.is_root:
##                    all_prior_particle_ids = np.concatenate(all_prior_particle_ids)
##                    all_prior_dataset = np.concatenate(all_prior_dataset)
##                    all_snapshot_particle_ids = np.concatenate(all_snapshot_particle_ids)
##                    all_test_mpi_reorder = np.concatenate(all_test_mpi_reorder)
##                    all_test_mpi_reorder2 = np.concatenate(all_test_mpi_reorder2)
#                    r = ArrayReorder_2.create(all_prior_particle_ids, all_snapshot_particle_ids)
#                    test_reorder = r(all_prior_particle_ids)
#                    test_reorder2 = r(all_prior_dataset)
#                    Console.print_raw("Non-mpi reorder != target result:", (test_reorder != all_snapshot_particle_ids).sum(), f"({hashlib.sha256(test_reorder.tobytes()).hexdigest()[-5:]} & {hashlib.sha256(all_snapshot_particle_ids.tobytes()).hexdigest()[-5:]})")
#                    Console.print_raw("Non-mpi reorder != MPI result:   ", (test_reorder != all_test_mpi_reorder).sum(), f"({hashlib.sha256(test_reorder.tobytes()).hexdigest()[-5:]} & {hashlib.sha256(all_test_mpi_reorder.tobytes()).hexdigest()[-5:]})")
#                    Console.print_raw("Non-mpi reorder data != MPI result data:   ", (test_reorder2 != all_test_mpi_reorder2).sum(), f"({hashlib.sha256(test_reorder2.tobytes()).hexdigest()[-5:]} & {hashlib.sha256(all_test_mpi_reorder2.tobytes()).hexdigest()[-5:]})")

            Console.print_verbose_info("    Reordering...", end = "")

#            if USE_DEBUGGING_CHECKS:
#                _gathered_data = mpi_gather_array(prior_particle_last_halo_ids)
#                if MPI_Config.is_root:
##                    _gathered_data = np.concatenate(_gathered_data)
#                    Console.print_raw(f"prior_particle_last_halo_ids (before reorder) Array Hash:\n{hashlib.sha256(_gathered_data.tobytes()).hexdigest()}")
            #transition_to_new_order(prior_particle_last_halo_ids, output_array = snapshot_particle_last_halo_ids)#TODO: using output_array causes an error!!!
            t_start("Reordering prior_particle_last_halo_ids", compute = True)
            snapshot_particle_last_halo_ids = transition_to_new_order(prior_particle_last_halo_ids)
            t_end()

#            if USE_DEBUGGING_CHECKS:
#                _gathered_data = mpi_gather_array(snapshot_particle_last_halo_ids)
#                if MPI_Config.is_root:
##                    _gathered_data = np.concatenate(_gathered_data)
#                    Console.print_raw(f"snapshot_particle_last_halo_ids (before update) Array Hash:\n{hashlib.sha256(_gathered_data.tobytes()).hexdigest()}")

#            transition_to_new_order(prior_particle_last_halo_masses, output_array = snapshot_particle_last_halo_masses)
##            transition_to_new_order(prior_particle_last_halo_masses_scaled, output_array = snapshot_particle_last_halo_masses_scaled)
#            transition_to_new_order(prior_particle_last_halo_redshifts, output_array = snapshot_particle_last_halo_redshifts)
#            transition_to_new_order(prior_particle_last_halo_positions, output_array = snapshot_particle_last_halo_positions)
            t_start("Reordering prior_particle_last_halo_masses", compute = True)
            snapshot_particle_last_halo_masses = transition_to_new_order(prior_particle_last_halo_masses)
            t_end()
#            snapshot_particle_last_halo_masses_scaled = transition_to_new_order(prior_particle_last_halo_masses_scaled)
            t_start("Reordering prior_particle_last_halo_redshifts", compute = True)
            snapshot_particle_last_halo_redshifts = transition_to_new_order(prior_particle_last_halo_redshifts)
            t_end()
            t_start("Reordering prior_particle_last_halo_positions", compute = True)
            snapshot_particle_last_halo_positions = transition_to_new_order(prior_particle_last_halo_positions)
            t_end()

            t_start("Post-reorder dead time - this should be insignificant", dead = True)
            Console.print_debug("Waiting at MPI barrier.")
            mpi_barrier()
            Console.print_debug("Continuing...")
            t_end()

            t_start("Freeing memory")
            del transition_to_new_order
            del prior_particle_last_halo_ids
            del prior_particle_last_halo_masses
            del prior_particle_last_halo_redshifts
            del prior_particle_last_halo_positions
            gc.collect()
            t_end()

            Console.print_raw_verbose("done")

        t_start("Dead time before IO (x3)", dead = True)
        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")
        t_end()

        t_start("Reading halo indexes and masses", io = True)
        Console.print_verbose_info("    Reading catalogue data.")

        Console.print_verbose_info("        Halo IDs by Particle.", verbosity_level = 1)
        #if MPI_Config.is_root:
        #    #particle_halo_ids = catalogue.get_halo_IDs_by_snapshot_particle(particle_type, snapshot_particle_ids = snapshot_particle_ids)
        #    particle_halo_ids = mpi_scatter_array(
        #        catalogue.get_halo_indexes_by_snapshot_particle(particle_type, snapshot_particle_ids = mpi_gather_array(snapshot_particle_ids)),
        #        elements_this_rank = n_particles
        #    )
        #else:
        #    mpi_gather_array(snapshot_particle_ids)
        #    particle_halo_ids = mpi_scatter_array(None, elements_this_rank = n_particles)
        particle_halo_ids = catalogue.get_halo_indexes_by_snapshot_particle(particle_type)
        Console.print_verbose_info("        Halo IDs.", verbosity_level = 1)
        #halo_ids = catalogue.get_halo_indexes(particle_type)
        halo_ids = catalogue.get_halo_indexes()
        Console.print_verbose_info("        Halo Masses.", verbosity_level = 1)
        halo_masses = catalogue.get_halo_masses(catalogue.BasicHaloDefinitions.SO_200_CRIT.value).to("Msun").value
        Console.print_verbose_info("        (done)", verbosity_level = 1)

        t_end()

        t_start("Dead time after IO (x3)", dead = True)
        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")
        t_end()

#        if prior_particle_ids is None:
#            Console.print_debug(n_particles, None, snapshot_particle_ids.shape, snapshot_particle_last_halo_ids.shape, particle_halo_ids.shape)
#        else:
#            Console.print_debug(n_particles, prior_particle_ids.shape, snapshot_particle_ids.shape, snapshot_particle_last_halo_ids.shape, particle_halo_ids.shape)

        t_start("Creating update mask")
        snapshot_particle_update_mask = np.full(n_particles, False, dtype = np.bool_)
        t_end()

        t_start("Halo search", compute = True)
        Console.print_verbose_info("    Searching haloes.")

        #for x in range(halo_ids.shape[0]):
        for x in np.where(np.isin(halo_ids, np.unique(particle_halo_ids)))[0]:
            # FOF groups with no SUBFIND subhaloes have a group mass but an M200 of 0
            # We don't want to consider these groups as they are likley transient in nature
            if halo_masses[x] > 0.0:
                SnapshotSearcher.update_halo_particles(
                    x,
                    snapshot_particle_update_mask,
                    snapshot_particle_last_halo_ids,
                    snapshot_particle_last_halo_masses,
#                    snapshot_particle_last_halo_masses_scaled,
#                    self.__get_mass_of_L_star(snapshot.redshift),
                    halo_ids,
                    halo_masses,
                    particle_halo_ids
                )
        t_end()

        t_start("Freeing memory")
        Console.print_verbose_info("    Freeing catalogue memory.")
        del halo_ids
        del halo_masses
        del particle_halo_ids
        gc.collect()
        t_end()

        t_start("Dead time before IO", dead = True)
        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")
        t_end()

        t_start("Reading particle positions", io = True)
        Console.print_verbose_info("    Reading particle positions.")

        snapshot_positions = snapshot.get_positions(particle_type)
        t_end()

        t_start("Dead time after IO", dead = True)
        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")
        t_end()

        t_start("Updating mappable data", compute = True)
        Console.print_verbose_info("    Mapping snapshot and global values.")

        snapshot_particle_last_halo_redshifts[snapshot_particle_update_mask] = snapshot.redshift
        snapshot_particle_last_halo_positions[snapshot_particle_update_mask, :] = snapshot_positions[snapshot_particle_update_mask, :]
        t_end()

        t_start("Freeing memory")
        del snapshot_positions
        gc.collect()
        t_end()


#        if USE_DEBUGGING_CHECKS:
#            _gathered_data = mpi_gather_array(snapshot_particle_last_halo_ids)
#            if MPI_Config.is_root:
##                _gathered_data = np.concatenate(_gathered_data)
#                Console.print_raw(f"snapshot_particle_last_halo_ids Array Hash:\n{hashlib.sha256(_gathered_data.tobytes()).hexdigest()}") # cc706 (L100@000)

        t_start("Dead time before calculation of number of matches", dead = True)
        Console.print_debug("Waiting at MPI barrier.")
        mpi_barrier()
        Console.print_debug("Continuing...")
        t_end()

        t_start("Gathering number of matched particles")
        matches_per_rank: list[int]|None = MPI_Config.comm.gather(snapshot_particle_update_mask.sum())
        Console.print_info(f"    Identified {sum(matches_per_rank if matches_per_rank is not None else [-1])} particles in haloes.")
        t_end()

        Console.print_verbose_info("    Halo search step complete.")

        t_start("Exiting searcher")

        return (
            snapshot_particle_ids,
            snapshot_particle_last_halo_ids,
            snapshot_particle_last_halo_masses,
#            snapshot_particle_last_halo_masses_scaled,
            snapshot_particle_last_halo_redshifts,
            snapshot_particle_last_halo_positions
        )

    @staticmethod
    def update_halo_particles(
        halo_index: int,
        updated_indexes_mask: np.ndarray,
        snapshot_particle_last_halo_ids: np.ndarray,
        snapshot_particle_last_halo_masses: np.ndarray,
#        snapshot_particle_last_halo_masses_scaled: np.ndarray,
#        halo_mass_at_L_star: float,
        halo_ids: np.ndarray,
        halo_masses: np.ndarray,
        snapshot_particle_halo_ids: np.ndarray
    ) -> None:
        
        #Console.print_debug(f"Halo {halo_index}")

        halo_id = halo_ids[halo_index]
        particle_data_mask = snapshot_particle_halo_ids == halo_id
        if np.any(particle_data_mask):
        #if particle_data_mask.sum() > 0:
            updated_indexes_mask[particle_data_mask] = True
            snapshot_particle_last_halo_ids[particle_data_mask] = halo_id
            snapshot_particle_last_halo_masses[particle_data_mask] = halo_masses[halo_index]
#            snapshot_particle_last_halo_masses_scaled[particle_data_mask] = halo_masses[halo_index] / halo_mass_at_L_star
