# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .. import VERSION, Stopwatch, ParticleType, ArrayReorder, ArrayReorder_2, ArrayReorder_MPI, ArrayMapping, SharedArray, SharedArray_TransmissionData, SharedArray_Shepherd, SharedArray_ParallelJob
from .._L_star import get_L_star_halo_mass_of_z
from ..io import SnapshotBase, SnapshotEAGLE, FileTreeScraper_EAGLE, SnapshotSWIFT, CatalogueBase, CatalogueSUBFIND, CatalogueSOAP#, OutputWriter, OutputReader, HeaderDataset, ParticleTypeDataset, SnapshotStatsDataset, CheckpointData
#from ..io._Output_Objects__forwards import OutputWriter, OutputReader, HeaderDataset, ParticleTypeDataset, SnapshotStatsDataset
from ..io._Output_Objects import OutputWriter, OutputReader, HeaderDataset, ParticleTypeDataset, SnapshotStatsDataset

import datetime
import os
from typing import cast, Union, List, Tuple, Dict, Generic, TypeVar
from collections.abc import Callable, Iterable
import asyncio

import numpy as np
from unyt import unyt_quantity, unyt_array
from QuasarCode import Settings, Console
from QuasarCode.MPI import MPI_Config, mpi_barrier, synchronyse, mpi_get_slice
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import ScriptWrapper

Console.mpi_output_root_rank_only()



def main():
    ScriptWrapper(
        command = "contra-run-complete",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 8, 31),
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
                description = "Prefix to apply to all catalogue membership file names.\nOnly required if the methof of auto-resolving the filenames fails."
            ),
            ScriptWrapper.OptionalParam[str | None](
                name = "catalogue-properties-basename",
                description = "Prefix to apply to all catalogue properties file names.\nOnly required if the methof of auto-resolving the filenames fails."
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
    use_MPI: bool = Settings.mpi_avalible and MPI_Config.comm_size > 1
    if not use_MPI or MPI_Config.is_root:
        Console.print_info(f"Running with MPI on {MPI_Config.comm_size} ranks.")
        Console.print_debug(f"Root MPI rank is {MPI_Config.root}.")

    # Validate sim type
    if not (is_EAGLE or is_SWIFT):
        if not use_MPI or MPI_Config.is_root:
            Console.print_error("Must specify either EAGLE or SWIFT simulation type.")
        return
    else:
        if not use_MPI or MPI_Config.is_root:
            if is_EAGLE:
                Console.print_verbose_info("Snapshot type: EAGLE")
            elif is_SWIFT:
                Console.print_verbose_info("Snapshot type: SWIFT")

    # Validate particle types
    #     If none are specified, use all types
    if not use_MPI or MPI_Config.is_root:
        Console.print_verbose_info("Particle Types:")
    if not (do_gas or do_stars or do_black_holes or do_dark_matter):
        if not use_MPI or MPI_Config.is_root:
            Console.print_verbose_warning("    No particle type(s) specified. Enabling all particle types.")
        do_gas = do_stars = do_black_holes = do_dark_matter = True
    particle_types: list[ParticleType] = []
    if do_gas:
        particle_types.append(ParticleType.gas)
        if not use_MPI or MPI_Config.is_root:
            Console.print_verbose_info("    Tracking gas particles")
    if do_stars:
        particle_types.append(ParticleType.star)
        if not use_MPI or MPI_Config.is_root:
            Console.print_verbose_info("    Tracking star particles")
    if do_black_holes:
        particle_types.append(ParticleType.black_hole)
        if not use_MPI or MPI_Config.is_root:
            Console.print_verbose_info("    Tracking black hole particles")
    if do_dark_matter:
        particle_types.append(ParticleType.dark_matter)
        if not use_MPI or MPI_Config.is_root:
            Console.print_verbose_info("    Tracking dark matter particles")

    if not use_MPI or MPI_Config.is_root:
        Console.print_info("Identifying snapshot and catalogue files")

    # Ensure that the path is an absolute path
    snapshot_directory = os.path.abspath(snapshot_directory)

    # Get snapshot file information
    simulation_file_scraper = FileTreeScraper_EAGLE(snapshot_directory)# if is_EAGLE else FileTreeScraper_SWIFT(snapshot_directory)
    N_snapshots = len(simulation_file_scraper.snipshots if is_EAGLE else simulation_file_scraper.snapshots)

    # Ensure that the path is an absolute path
    output_filepath = os.path.abspath(output_filepath)

    # Validate the output file and exit all ranks if in an unacceptable state
    able_to_procede: bool
    if not use_MPI or MPI_Config.is_root:
        able_to_procede = True
        if os.path.exists(output_filepath):
            if not (allow_overwrite or restart):
                able_to_procede = False
                synchronyse("able_to_procede")
                Console.print_error("Output file already exists. Either remove it first or explicitley enable overwrite.")
                return
            elif allow_overwrite:
                if not use_MPI or MPI_Config.is_root:
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
    output_file = OutputWriter(output_filepath, overwrite = not restart)

    # Define variables used to pass data between searches
    snapshot: SnapshotBase
    catalogue: CatalogueBase
    result_data_shepherd: SharedArray_Shepherd|None = None
    shapshot_particle_ids:                  SharedArray|None = None
    snapshot_last_halo_ids:                 SharedArray|None = None
    snapshot_last_halo_masses:              SharedArray|None = None
    snapshot_last_halo_masses_scaled:       SharedArray|None = None
    snapshot_last_halo_redshifts:           SharedArray|None = None
    snapshot_last_halo_particle_positions:  SharedArray|None = None

    # Create the output file and header (if not restarting)
#    if not use_MPI or MPI_Config.is_root:
    if not restart:
        if not use_MPI or MPI_Config.is_root:
            Console.print_info("Creating file...", end = "")

        with output_file:
            output_file.write_header(HeaderDataset(
                version = VersionInfomation.from_string(VERSION),
                date = start_date,
                simulation_type = "SWIFT" if is_SWIFT else "EAGLE",
                simulation_directory = snapshot_directory,
                N_searched_snapshots = N_snapshots,
                uses_snipshots = is_EAGLE,
                output_file = output_filepath,
                has_gas = do_gas,
                has_stars = do_stars,
                has_black_holes = do_black_holes,
                has_dark_matter = do_dark_matter,
                has_statistics = False
            ))

        if not use_MPI or MPI_Config.is_root:
            Console.print_raw("done", flush = True)

    mpi_barrier()

    # Initialise the search dispatcher object
    # This allows the search to be parallelised on the current rank over haloes with one halo per subprocess
    searcher = SnapshotSearcher(64, SnapshotEAGLE, CatalogueSUBFIND, lambda _: 1.0) if is_EAGLE else SnapshotSearcher(64, SnapshotSWIFT, CatalogueSOAP, lambda _: 1.0)

    # Itterate over selected particle types
    for particle_type in particle_types:

        if not use_MPI or MPI_Config.is_root:
            Console.print_info(f"Running search for {particle_type.name} particles.")

        if restart:
            # Load data for current state

            last_completed_numerical_file_number: int

            Console.print_info("Reading checkpoint data:")

            # Read checkpoint data
            with OutputReader(output_filepath) as output_file_reader:

                Console.print_info("    Loading data...", end = "")

                checkpoint = output_file_reader.read_checkpoint(particle_type)

            if checkpoint is not None:

                result_data_shepherd = SharedArray_Shepherd()
                snapshot_last_halo_ids = SharedArray.as_shared(checkpoint.halo_ids, name = "snap-last-halo-ids", shepherd = result_data_shepherd)
                snapshot_last_halo_masses = SharedArray.as_shared(checkpoint.halo_masses, name = "snap-last-halo-masses", shepherd = result_data_shepherd)
                snapshot_last_halo_masses_scaled = SharedArray.as_shared(checkpoint.halo_masses_scaled, name = "snap-last-halo-masses-scaled", shepherd = result_data_shepherd)
                snapshot_last_halo_redshifts = SharedArray.as_shared(checkpoint.redshifts, name = "snap-last-halo-redshifts", shepherd = result_data_shepherd)
                snapshot_last_halo_particle_positions = SharedArray.as_shared(checkpoint.positions_pre_ejection, name = "snap-last-halo-particle-positions", shepherd = result_data_shepherd)

                last_completed_numerical_file_number = int(checkpoint.file_number)

                shapshot_particle_ids = SharedArray.as_shared((simulation_file_scraper.snipshots if is_EAGLE else simulation_file_scraper.snapshots).get_by_number(checkpoint.file_number).load().get_IDs(particle_type), name = "snap-particle-ids", shepherd = result_data_shepherd)

                Console.print_raw("done")

                Console.print_info(f"    Restarting from {'snapshot' if not is_EAGLE else 'snipshot'} {last_completed_numerical_file_number + 1}/{N_snapshots}")

            else:
                last_completed_numerical_file_number = -1
                Console.print_raw("failed - no data to restart from.")

#        stopwatch = Stopwatch("Search", show_time_since_lap = True)

        for catalogue_info in (simulation_file_scraper.snipshot_catalogues if is_EAGLE else simulation_file_scraper.catalogues):

            if restart and catalogue_info.number_numerical <= last_completed_numerical_file_number:
                Console.print_info(f"{catalogue_info.number} already complete. Skipping." if not is_EAGLE else f"{catalogue_info.number} (redshift {catalogue_info.tag_redshift}) already complete. Skipping.")
                continue
            Console.print_info(f"Doing {catalogue_info.number}" if not is_EAGLE else f"Doing {catalogue_info.number} (redshift {catalogue_info.tag_redshift})")

            Console.print_verbose_info("    Loading Snapshot and catalogue objects.")

            catalogue = catalogue_info.load()
            snapshot = catalogue.snapshot

            if not is_EAGLE:
                Console.print_info(f"    Redshift {catalogue.redshift}")
            if len(catalogue) == 0:
                Console.print_info("    No halos. Skipping.")
                continue
            Console.print_info(f"    Catalogue has {len(catalogue)} haloes.")

            Console.print_verbose_info("    Seaching for halo membership.")

            (
                result_data_shepherd,
                shapshot_particle_ids,
                snapshot_last_halo_ids,
                snapshot_last_halo_masses,
                snapshot_last_halo_masses_scaled,
                snapshot_last_halo_redshifts,
                snapshot_last_halo_particle_positions
            ) = searcher(
                particle_type,
                catalogue,
                result_data_shepherd,
                shapshot_particle_ids,
                snapshot_last_halo_ids,
                snapshot_last_halo_masses,
                snapshot_last_halo_masses_scaled,
                snapshot_last_halo_redshifts,
                snapshot_last_halo_particle_positions
            )

            Console.print_verbose_info("    Creating output struct.")

            results = ParticleTypeDataset(
                particle_type = particle_type,
                target_redshift = catalogue.redshift,
                file_number = catalogue_info.number,
                length = snapshot.number_of_particles(ParticleType.gas),
                length_this_rank = snapshot.number_of_particles_this_rank(ParticleType.gas),
                redshifts = snapshot_last_halo_redshifts.data,
                halo_ids = snapshot_last_halo_ids.data,
                halo_masses = snapshot_last_halo_masses.data,
                halo_masses_scaled = snapshot_last_halo_masses_scaled.data,
                positions_pre_ejection = snapshot_last_halo_particle_positions.data
            )

            with output_file:

                Console.print_verbose_info("    Writing data.")

#                try:
                output_file.write_particle_type_dataset(results)
#                except KeyError as e:
#                    if restart:
#                        output_file.write_particle_type_dataset(results, overwrite = True)
#                        output_file.increase_number_of_snapshots(N_snapshots)
#                    else:
#                        raise e

#            stopwatch.lap()

        if result_data_shepherd is not None: # Check just in case no snapshots were itterated over!
            result_data_shepherd.free()



T_snapshot = TypeVar("T_snapshot", bound = SnapshotBase)
T_catalogue = TypeVar("T_catalogue", bound = CatalogueBase)
class SnapshotSearcher(Generic[T_snapshot, T_catalogue]):

    def __init__(self, number_of_workers: int, snapshot_type: type[T_snapshot], catalogue_type: type[T_catalogue], mass_of_L_star_at_z: Callable[[float], float]):

        self.__number_of_workers = number_of_workers
        self.__snapshot_type = snapshot_type
        self.__catalogue_type = catalogue_type

        self.__get_mass_of_L_star = mass_of_L_star_at_z

    def __call__(
            self,
            particle_type: ParticleType,
            catalogue: T_catalogue,
            prior_particle_data_shepherd: SharedArray_Shepherd|None,
            prior_particle_ids: SharedArray|None,
            prior_particle_last_halo_ids: SharedArray|None,
            prior_particle_last_halo_masses: SharedArray|None,
            prior_particle_last_halo_masses_scaled: SharedArray|None,
            prior_particle_last_halo_redshifts: SharedArray|None,
            prior_particle_last_halo_positions: SharedArray|None
    ) -> tuple[SharedArray_Shepherd, SharedArray, SharedArray, SharedArray, SharedArray, SharedArray, SharedArray]:
        
        if not isinstance(catalogue, self.__catalogue_type):
            raise TypeError(f"Unexpected catalogue type {type(catalogue).__name__}. Expected {self.__catalogue_type.__name__}.")
        if not isinstance(catalogue.snapshot, self.__snapshot_type):
            raise TypeError(f"Unexpected snapshot type {type(catalogue.snapshot).__name__}. Expected {self.__snapshot_type.__name__}.")

        Console.print_verbose_info("    Reading snapshot data.")
        
        snapshot = catalogue.snapshot

        n_particles = snapshot.number_of_particles_this_rank(particle_type)

        results_shepherd = SharedArray_Shepherd()

        snapshot_particle_ids = SharedArray.as_shared(snapshot.get_IDs(particle_type), name = "snap-particle-ids", shepherd = results_shepherd)

        Console.print_verbose_info("    Allocating result array memory.")

        snapshot_particle_last_halo_ids = SharedArray.create(n_particles, np.int64, name = "snap-last-halo-ids", shepherd = results_shepherd)
        snapshot_particle_last_halo_masses = SharedArray.create(n_particles, np.float64, name = "snap-last-halo-masses", shepherd = results_shepherd)
        snapshot_particle_last_halo_masses_scaled = SharedArray.create(n_particles, np.float64, name = "snap-last-halo-masses-scaled", shepherd = results_shepherd)
        snapshot_particle_last_halo_redshifts = SharedArray.create(n_particles, np.float64, name = "snap-last-halo-redshifts", shepherd = results_shepherd)
        snapshot_particle_last_halo_positions = SharedArray.create((n_particles, 3), np.float64, name = "snap-last-halo-particle-positions", shepherd = results_shepherd)

        if prior_particle_data_shepherd is None:
            # This is the first snapshot

            Console.print_verbose_info("    No previous results. Filling result arrays with placeholder data.")

            snapshot_particle_last_halo_ids.fill(-1)
            snapshot_particle_last_halo_masses.fill(np.nan)
            snapshot_particle_last_halo_masses_scaled.fill(np.nan)
            snapshot_particle_last_halo_redshifts.fill(np.nan)
            snapshot_particle_last_halo_positions.fill(np.nan)

        else:
            # Reorganise existing data

            Console.print_verbose_info("    Calculating reorder for previous result data.")

#            transition_to_new_order = (ArrayReorder_MPI if MPI_Config.comm_size > 1 else ArrayReorder_2).create(prior_particle_ids.data, snapshot_particle_ids.data)
            transition_to_new_order = ArrayReorder_MPI.create(prior_particle_ids.data, snapshot_particle_ids.data)

            test_all_input_ids = np.concatenate(MPI_Config.comm.gather(prior_particle_ids.data))
            transition_to_new_order__test = ArrayReorder_2.create(test_all_input_ids, np.concatenate(MPI_Config.comm.gather(snapshot_particle_ids.data)))
            test_result = transition_to_new_order__test(test_all_input_ids)
            mpi_test_result = np.concatenate(MPI_Config.comm.gather(transition_to_new_order(prior_particle_ids.data)))

            if MPI_Config.is_root:
                Console.print("MPI reorder output shape:", mpi_test_result.shape)
                Console.print("v2 reorder output shape: ", test_result.shape)
                Console.print("N matches:               ", (mpi_test_result == test_result).sum())
                Console.print("N differences:           ", (mpi_test_result != test_result).sum())

            Console.print_verbose_info("    Reordering...", end = "")

            transition_to_new_order(prior_particle_last_halo_ids.data, output_array = snapshot_particle_last_halo_ids.data)
            transition_to_new_order(prior_particle_last_halo_masses.data, output_array = snapshot_particle_last_halo_masses.data)
            transition_to_new_order(prior_particle_last_halo_masses_scaled.data, output_array = snapshot_particle_last_halo_masses_scaled.data)
            transition_to_new_order(prior_particle_last_halo_redshifts.data, output_array = snapshot_particle_last_halo_redshifts.data)
            transition_to_new_order(prior_particle_last_halo_positions.data, output_array = snapshot_particle_last_halo_positions.data)

            Console.print_raw_verbose("done")

            Console.print_verbose_info("    Releasing memory from previous results.")

            prior_particle_data_shepherd.free()

        with SharedArray_Shepherd() as shared_memory: # Used to free all memory used for data not returned

            Console.print_verbose_info("    Reading catalogue data.")

            halo_ids = SharedArray.as_shared(catalogue.get_halo_IDs(particle_type), name = "halo-ids", shepherd = shared_memory)
            halo_masses = SharedArray.as_shared(catalogue.get_halo_masses(particle_type), name = "halo-masses", shepherd = shared_memory)
            particle_halo_ids = SharedArray.as_shared(catalogue.get_halo_IDs_by_snapshot_particle(particle_type, snapshot_particle_ids = snapshot_particle_ids.data), name = "particle-halo-ids", shepherd = shared_memory)

            if prior_particle_data_shepherd is None:
                Console.print_debug(n_particles, None, snapshot_particle_ids.data.shape, snapshot_particle_last_halo_ids.data.shape, particle_halo_ids.data.shape)
            else:
                Console.print_debug(n_particles, prior_particle_ids.data.shape, snapshot_particle_ids.data.shape, snapshot_particle_last_halo_ids.data.shape, particle_halo_ids.data.shape)

            snapshot_positions = SharedArray.as_shared(snapshot.get_positions(particle_type), name = "particle-positions", shepherd = shared_memory)

            snapshot_particle_update_mask = SharedArray.create(n_particles, np.bool_, name = "snap-particle-update-mask", shepherd = shared_memory).fill(False)

            Console.print_verbose_info("    Distributing search tasks to workers.")

            #SharedArray_ParallelJob(SnapshotSearcher.update_halo_particles, pool_size = self.__number_of_workers, number_of_chunks = self.__number_of_workers, ignore_return = True).execute(
            #    range(halo_ids.data.shape[0]),
            #    snapshot_particle_update_mask,
            #    snapshot_particle_last_halo_ids,
            #    snapshot_particle_last_halo_masses,
            #    snapshot_particle_last_halo_masses_scaled,
            #    self.__get_mass_of_L_star(snapshot.redshift),
            #    halo_ids,
            #    halo_masses,
            #    particle_halo_ids
            #)
            for x in range(halo_ids.data.shape[0]):
                SnapshotSearcher.update_halo_particles(
                    x,
                    snapshot_particle_update_mask,
                    snapshot_particle_last_halo_ids,
                    snapshot_particle_last_halo_masses,
                    snapshot_particle_last_halo_masses_scaled,
                    self.__get_mass_of_L_star(snapshot.redshift),
                    halo_ids,
                    halo_masses,
                    particle_halo_ids
                )

            Console.print_verbose_info("    Mapping snapshot result data.")

            snapshot_particle_last_halo_redshifts.data[snapshot_particle_update_mask.data] = snapshot.redshift
            snapshot_particle_last_halo_positions.data[snapshot_particle_update_mask.data, :] = snapshot_positions.data[snapshot_particle_update_mask.data, :]

            Console.print_info(f"    Identified {snapshot_particle_update_mask.data.sum()} particles in haloes.")

        Console.print_verbose_info("    Halo search step complete.")

        return results_shepherd, snapshot_particle_ids, snapshot_particle_last_halo_ids, snapshot_particle_last_halo_masses, snapshot_particle_last_halo_masses_scaled, snapshot_particle_last_halo_redshifts, snapshot_particle_last_halo_positions

#    @staticmethod
#    def find_haloes(
#        i: int,
#        updated_indexes_mask: SharedArray,
#        snapshot_particle_last_halo_ids: SharedArray,
#        snapshot_particle_last_halo_masses: SharedArray,
#        halo_ids: SharedArray,
#        halo_masses: SharedArray,
#        snapshot_particle_halo_ids: SharedArray
#    ) -> None:
#
#        halo_id = snapshot_particle_halo_ids.data[i]
#        if halo_id != -1:
#            halo_index = np.where(halo_ids == snapshot_particle_halo_ids.data[i])[0][0]
#            updated_indexes_mask.data[i] = True
#            snapshot_particle_last_halo_ids.data[i] = halo_id
#            snapshot_particle_last_halo_masses.data[i] = halo_masses.data[halo_index]

    @staticmethod
    def update_halo_particles(
        halo_index: int,
        updated_indexes_mask: SharedArray,
        snapshot_particle_last_halo_ids: SharedArray,
        snapshot_particle_last_halo_masses: SharedArray,
        snapshot_particle_last_halo_masses_scaled: SharedArray,
        halo_mass_at_L_star: float,
        halo_ids: SharedArray,
        halo_masses: SharedArray,
        snapshot_particle_halo_ids: SharedArray
    ) -> None:
        
        Console.print_debug(f"Halo {halo_index}")

        halo_id = halo_ids.data[halo_index]
        particle_data_mask = snapshot_particle_halo_ids.data == halo_id
        if particle_data_mask.sum() > 0:
            updated_indexes_mask.data[particle_data_mask] = True
            snapshot_particle_last_halo_ids.data[particle_data_mask] = halo_id
            snapshot_particle_last_halo_masses.data[particle_data_mask] = halo_masses.data[halo_index]
            snapshot_particle_last_halo_masses_scaled.data[particle_data_mask] = halo_masses.data[halo_index] / halo_mass_at_L_star
