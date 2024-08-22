# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
#BAD_TESTING_SNIPSHOTS = ("snip_354_z000p225.0.hdf5", "snip_288_z000p770.0.hdf5", "snip_224_z001p605.0.hdf5", "snip_159_z002p794.0.hdf5")
from contra import ParticleType, ArrayReorder, ArrayReorder_2, SharedArray, SharedArray_TransmissionData, calculate_wrapped_displacement, calculate_wrapped_distance, Stopwatch
from contra.io import SnapshotEAGLE, LineOfSightFileEAGLE
import h5py as h5
import numpy as np
from typing import cast, Any, Union, List, Dict, Tuple, Iterable, Iterator
import os
from QuasarCode import Console, Settings
from QuasarCode.Tools import ScriptWrapper, ArrayVisuliser
import datetime
import multiprocessing
from multiprocessing import shared_memory
import ctypes
import time
from matplotlib import pyplot as plt
from unyt import unyt_quantity



class PartialZip(Iterator):
    def __init__(self, iterable_indexes: Iterable[int], /, *items: Union[Iterable, Any], strict: bool = False):
        self.__length = len(items)
        self.__iterables_mask = np.isin(np.arange(self.__length), np.array(iterable_indexes, dtype = int))
        self.__statics_mask = ~self.__iterables_mask
        self.__statics = [items[i] for i in range(self.__length) if self.__statics_mask[i]]
        self.__zip: zip = zip(*[items[i] for i in range(self.__length) if self.__iterables_mask[i]], strict = strict)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Tuple[Any, ...]:
        iterable_items = self.__zip.__next__()
        next_iterable = 0
        next_static = 0
        result: List[Any] = [None] * self.__length
        for i in range(self.__length):
            if self.__statics_mask[i]:
                result[i] = self.__statics[next_static]
                next_static += 1
            else:
                result[i] = iterable_items[next_iterable]
                next_iterable += 1
        return tuple(result)



def main():
    ScriptWrapper(
        command = "contra-generate-eagle-los-ids",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 8, 1),
        description = "Attempts to identify the particle IDs for gas particle data in the line-of-sight outputs from EAGLE simulation runs.\nThis is done by interpolating the position, mass, metallicity, temperature, and density of particles\nfound in both the snipshots with redshifts either side of the target line-of-sight file.",
        parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.PositionalParam[str](
                name = "output-file",
                description = "File to write recovered IDs into."
            ),
            ScriptWrapper.PositionalParam[str](
                name = "data-directory",
                description = "EAGLE simulation output directory containing snipshot folders."
            ),
            ScriptWrapper.PositionalParam[str](
                name = "los-directory",
                description = "Directory containing line-of-sight files."
            ),
            ScriptWrapper.OptionalParam[int](
                name = "number-of-processes",
                short_name = "n",
                sets_param = "n_processes",
                description = "Number of processes to use. Defaults to 1 (i.e. serial execution)",
                conversion_function = int,
                default_value = 1
            ),
            ScriptWrapper.OptionalParam[float](
                name = "fractional-los-quantity-error",
                short_name = "e",
                description = "Fractional change away from the limit to the Line of Sight quantity being searched\nin order to account for any floating point precision error.\nDefault value is 0.001",
                conversion_function = float,
                default_value = 0.001
            ),
            ScriptWrapper.OptionalParam[float](
                name = "column-width-multiplier",
                short_name = "w",
                description = "Multiplier for the width of the column of particles around the line-of-sight from which to search for matches.\nWhen using the default data-loading scheme, the width is axis-dependant and is applied to the estimated positions of particles. When using the memory-efficient data-loading scheme, the width uses twice the greatest distance in any axis and is applied to the raw particle data and NOT the estimated positions!\nThis value ought to be greater than 1.0 to account for particle movement.\nThe initial width is determined by the distribution of particle positions in the line-of-sight data.\nDefault value is 2.0",
                conversion_function = float,
                default_value = 2.0
            ),
            ScriptWrapper.OptionalParam[str](
                name = "do-file",
                short_name = "f",
                sets_param = "selected_file",
                description = "Do only a specific Line of Sight file.\nSpecify the full name of the file (including extension) but not the path."
            ),
            ScriptWrapper.OptionalParam[int](
                name = "do-index",
                short_name = "i",
                sets_param = "selected_los_index",
                description = "Do only a specific Line of Sight from the specified file file.",
                conversion_function = int,
                requirements = ["do-file"]
            ),
            ScriptWrapper.OptionalParam[int](
                name = "do-particle-index",
                short_name = "p",
                sets_param = "selected_los_particle_index",
                description = "Do only a single particle from a specific Line of Sight from the specified file file.\nThis will not write any data, but instead output statistics about the selection.\nAdditionally, multiprocessing will be disabled if more than one process has been specified.",
                conversion_function = int,
                requirements = ["do-index"]
            ),
            ScriptWrapper.Flag(
                name = "force-selection",
                description = "If the targeted file/sightline already exists, overwrite it.",
                requirements = ["do-file"]
            ),
            ScriptWrapper.Flag(
                name = "plot-selection-stats",
                description = "Plot statistics for selected particles in the specified sightline.\nOnly valid when running on a single sightline.",
                requirements = ["do-index"],
                conflicts = ["do-particle-index"]
            ),
            ScriptWrapper.Flag(
                name = "efficient-memory-usage",
                short_name = "m",
                sets_param = "use_memory_efficient_data_loading",
                description = "Load data using a method that prioritises memory efficiency.\nThis may be required when targeting large simulation volumes to avoid needing to load the entire volume.\nNote: this may be SLOWER if running on systems that are capable of loading the whole volume if there are a large number of lines-of-sight!",
            ),
            ScriptWrapper.OptionalParam[str](
                name = "start-file",
                sets_param = "parallel_start_file",
                description = "Start from a specific file.\nSpecify the full name of the file (including extension) but not the path.\nThe number of files to do must also be specified!",
                requirements = ["number-of-files"],
                conflicts = ["do-file", "start-file-index"]
            ),
            ScriptWrapper.OptionalParam[int](
                name = "start-file-index",
                sets_param = "parallel_start_file_index",
                description = "Start from a specific file.\nSpecify the index of the file in assending redshift order.\nThe number of files to do must also be specified!",
                conversion_function = int,
                requirements = ["number-of-files"],
                conflicts = ["do-file", "start-file"]
            ),
            ScriptWrapper.OptionalParam[int](
                name = "number-of-files",
                sets_param = "number_of_files_to_do",
                description = "Number of files to be run when a start file is specified. This can be greater than the number of remaining files.",
                conversion_function = int
            )
        )
    ).run(__main)



def __main(
            output_file: str,
            data_directory: str,
            los_directory: str,
            n_processes: int,
            fractional_los_quantity_error: float,
            column_width_multiplier: float,
            selected_file: str|None,
            selected_los_index: int|None,
            selected_los_particle_index: int|None,
            force_selection: bool,
            plot_selection_stats: bool,
            use_memory_efficient_data_loading: bool,
            parallel_start_file: str|None,
            parallel_start_file_index: int|None,
            number_of_files_to_do: int|None
          ) -> None:

    STOPWATCH = Stopwatch("Command", print_on_start = False, show_time_since_lap = False)
    CHUNK_STOPWATCH = Stopwatch("Section", print_on_start = False, show_time_since_lap = True, synchronise = STOPWATCH)
    
    if not os.path.exists(los_directory):
        Console.print_error(f"Line-of-sight file directory \"{los_directory}\" does not exist.")
        Console.print_info("Terminating...")
        return

    # Get the line of sight filepaths in redshift order (assending z)

    los_files = LineOfSightFileEAGLE.get_files(los_directory)
#    los_files = [os.path.join(los_directory, file) for file in list(*os.walk(los_directory))[2] if file[:8] == "part_los"]
#    los_files.sort(key = lambda v: float(v.rsplit("z", maxsplit = 1)[1].rsplit(".", maxsplit = 1)[0]))

    # Check that a specified los file (if any) is avalible

    if selected_file is not None:
        selected_file = os.path.join(los_directory, selected_file)
        if selected_file not in los_files:
            Console.print_error(f"A specific Line of Sight file was specified ({selected_file}) but does not exist at location \"{los_directory}\".")
            Console.print_info("Terminating...")
            return
    elif parallel_start_file is not None or parallel_start_file_index is not None:
        if parallel_start_file is not None:
            parallel_start_file = cast(str, parallel_start_file)
            parallel_start_file = os.path.join(los_directory, parallel_start_file)
            if parallel_start_file not in los_files:
                Console.print_error(f"A specific Line of Sight file was specified ({parallel_start_file}) but does not exist at location \"{los_directory}\".")
                Console.print_info("Terminating...")
                return
        else:
            parallel_start_file_index = cast(int, parallel_start_file_index)
            if (parallel_start_file_index >= 0 and len(los_files) <= parallel_start_file_index) or (parallel_start_file_index < 0 and len(los_files) < -parallel_start_file_index):
                Console.print_error(f"Too few line-of-sight files ({len(los_files)}) at location \"{los_directory}\" for start file index {parallel_start_file_index}.")
                Console.print_info("Terminating...")
                return
            parallel_start_file = los_files[parallel_start_file_index]
        if cast(int, number_of_files_to_do) < 1:
            Console.print_error(f"Number of line-of-sight files requested for recovery ({number_of_files_to_do}) is less than 1. Must be at least 1.")
            Console.print_info("Terminating...")
            return
        start_file_index = los_files.index(parallel_start_file)
        los_files = los_files[start_file_index : (start_file_index + cast(int, number_of_files_to_do))]

    # Get the snipshot filepaths in redshift order (assending z)

    snipshot_files = [os.path.join(folder, files[0].rsplit(".", maxsplit = 2)[0] + ".0.hdf5") for (folder, _, files) in os.walk(data_directory) if folder.rsplit(os.path.sep, maxsplit = 1)[1][:8] == "snipshot"]
    snipshot_redshifts = [float(v.rsplit("z", maxsplit = 1)[1].rsplit(".", maxsplit = 2)[0].replace("p", ".")) for v in snipshot_files]
    snipshot_files.sort(key = lambda v: float(v.rsplit("z", maxsplit = 1)[1].rsplit(".", maxsplit = 2)[0].replace("p", ".")))
    snipshot_redshifts.sort()

    # Create lookups for the snipshot information by snipshot number

    snapshot_redshifts: Dict[int, float] = { int(file.rsplit(os.path.sep, maxsplit = 1)[1].split("_")[1]) : snipshot_redshifts[i] for i, file in enumerate(snipshot_files) }
    snapshots: Dict[int, SnapshotEAGLE] = {}
    snap_expansion_factors: Dict[int, float] = {}
    snapshot_filepaths_by_number: Dict[int, str] = { int(file.rsplit(os.path.sep, maxsplit = 1)[1].split("_")[1]) : file for file in snipshot_files }

    # Get the snipshot numbers in assending order (decending z)

    snap_nums = list(snapshot_redshifts.keys())
    snap_nums.sort()

    # Define a way to find the two snipshots that lie either side of a particular redshift value

    def find_neighbouring_snapshots(z: float) -> Tuple[int, int]:
        if z > snapshot_redshifts[snap_nums[0]] or z < snapshot_redshifts[snap_nums[-1]]:
            raise ValueError(f"Redshift {z} outside of redshift range of avalible data.")
        lower_redshift_snap_num = snap_nums[0]
        while lower_redshift_snap_num not in snapshot_redshifts or snapshot_redshifts[lower_redshift_snap_num] > z:
            lower_redshift_snap_num += 1
        return (lower_redshift_snap_num - 1, lower_redshift_snap_num)

    # Create output file

    if not os.path.exists(output_file):
        h5.File(output_file, "w").close()
        complete_files = []
    else:
        with h5.File(output_file, "r") as file:
            complete_files = list(file)

    # Itterate over one line-of-sight file at a time
    # If a single file is selected, just use that file only

    for f in los_files if selected_file is None else (selected_file, ):

        # Get an object of reading los data

        sightline_file = LineOfSightFileEAGLE(f)

        # Identify the file name (used for saving data)

        output_file_group_name = f.rsplit(os.path.sep, maxsplit = 1)[-1]

        # Skip any completed files (allows for restarting) and identify the number of complete lines if not all are done
        # This will retain any sightlines already completed

        completed_sightlines = 0
        if output_file_group_name in complete_files:
            with h5.File(output_file, "r") as file:
                completed_sightlines = len(list(file[output_file_group_name]))
            if completed_sightlines == len(sightline_file) and not force_selection:
                continue
        else:
            with h5.File(output_file, "a") as file:
                file.create_group(output_file_group_name)

        # State which file is being targeted

        Console.print_info(output_file_group_name)

        if Settings.verbose or Settings.debug:
            STOPWATCH.start()
            CHUNK_STOPWATCH.start(print = False)

        # Identify which lines of sight need to be computed for this file

        do_sightlines: Iterable
        if selected_los_index is None:
            # Do only incomplete lines
            do_sightlines = range(completed_sightlines, len(sightline_file))
        else:
            # Do only the specified los, but first check if we are allowed to overwrite it should it already exist (but only if any data will be written)!
            min_valid_sightline = 0 if force_selection else completed_sightlines
            if selected_los_particle_index is None and (selected_los_index < min_valid_sightline or selected_los_index >= len(sightline_file)):
                Console.print_error(f"Selected sightline ({selected_los_index}) outside of valid range (inclusive) {min_valid_sightline} -> {len(sightline_file) - 1}.\nTo re-compute a sightline for which there is already data, use the --force-selection flag.")
                return
            do_sightlines = (selected_los_index, )

        # Are we running a test search on a specific particle?

        if selected_los_particle_index is not None:
            # Check the particle index is valid
            n_avalible_particles = sightline_file.get_sightline_length(do_sightlines[0])
            if (selected_los_particle_index >= 0 and selected_los_particle_index >= n_avalible_particles) or (selected_los_particle_index < 0 and -selected_los_particle_index > n_avalible_particles):
                Console.print_error(f"Selected sightline particle index ({selected_los_particle_index}) out of range for sightline ({selected_los_index}) with {n_avalible_particles} particles.")
                return

        # Get the numbers of the snipshots that border the line of sight file's redshift

        snap_num_initial, snap_num_final = find_neighbouring_snapshots(sightline_file.z)

        if Settings.verbose or Settings.debug:
            CHUNK_STOPWATCH.lap("Preliminary state set.")

        # Create EAGLE snapshot reader objects
        # These are cached to avoid unnessessary initilisation if the same snipshot is used multiple times

        if snap_num_initial not in snapshots:
            snapshots[snap_num_initial] = SnapshotEAGLE(snapshot_filepaths_by_number[snap_num_initial])
            snap_expansion_factors[snap_num_initial] = snapshots[snap_num_initial].a
        if snap_num_final not in snapshots:
            snapshots[snap_num_final] = SnapshotEAGLE(snapshot_filepaths_by_number[snap_num_final])
            snap_expansion_factors[snap_num_final] = snapshots[snap_num_final].a
        box_size = snapshots[snap_num_final].box_size[0].value
        half_box_size = box_size / 2

        if Settings.verbose or Settings.debug:
            CHUNK_STOPWATCH.lap("Snapshot objects created.")

        # Are the extra interpolated fields needed?

        get_extra_data = plot_selection_stats or selected_los_particle_index is not None

        if not use_memory_efficient_data_loading: # Load all relevant fields into memory and interpolate
            # This method avoids the need to read snapshot data and perform interpolation for each los
            # This comes at the cost of needing to load the entirety of all relivent snipshot fields into memory

            # The number of gas particles will decrease over time,
            # so find the numberof particles that will appear in both snipshots

#            if Settings.verbose or Settings.debug:
#                timer_start = time.time()

            n_remaining_snip_particles = snapshots[snap_num_final].number_of_particles(ParticleType.gas)

            low_z_snipshot_ids                  = SharedArray.create(n_remaining_snip_particles,      np.int64  , name = "low_z_snipshot_ids")
            high_z_snipshot_particle_data       = SharedArray.create((n_remaining_snip_particles, 5), np.float64, name = "high_z_snipshot_particle_data")
            interpolated_snipshot_particle_data = SharedArray.create((n_remaining_snip_particles, 3 if not get_extra_data else 5), np.float64, name = "interpolated_snipshot_particle_data")
            low_z_snipshot_particle_data        = SharedArray.create((n_remaining_snip_particles, 5), np.float64, name = "low_z_snipshot_particle_data")

            # Determine snipshot data reorder
            STOPWATCH.print_verbose_info("Reading initial IDs.")
            initial_ids = snapshots[snap_num_initial].get_IDs(ParticleType.gas)
            STOPWATCH.print_verbose_info("Reading final IDs.")
            final_ids = snapshots[snap_num_final].get_IDs(ParticleType.gas)
            STOPWATCH.print_verbose_info("Computing reorder.")
            order_by = ArrayReorder_2.create(initial_ids, final_ids)

            # Read snipshot particle data into shared memory
            STOPWATCH.print_verbose_info("Reading initial positions.")
            high_z_snipshot_all_particle_positions = snapshots[snap_num_initial].get_positions(ParticleType.gas).to("Mpc").value
            high_z_snipshot_particle_data.data[:, 0] = order_by(high_z_snipshot_all_particle_positions[:, 0])
            high_z_snipshot_particle_data.data[:, 1] = order_by(high_z_snipshot_all_particle_positions[:, 1])
            high_z_snipshot_particle_data.data[:, 2] = order_by(high_z_snipshot_all_particle_positions[:, 2])
            STOPWATCH.print_verbose_info(f"Reading initial masses.")
            high_z_snipshot_particle_data.data[:, 3] = order_by(snapshots[snap_num_initial].get_masses(ParticleType.gas).to("Msun").value)
            STOPWATCH.print_verbose_info("Reading initial metallicities.")
            high_z_snipshot_particle_data.data[:, 4] = order_by(snapshots[snap_num_initial].get_metalicities(ParticleType.gas).value)
            low_z_snipshot_ids.data[:] = final_ids
            STOPWATCH.print_verbose_info("Reading final positions.")
            low_z_snipshot_particle_data.data[:, :3] = snapshots[snap_num_final].get_positions(ParticleType.gas).to("Mpc").value
            STOPWATCH.print_verbose_info("Reading final masses.")
            low_z_snipshot_particle_data.data[:, 3]  = snapshots[snap_num_final].get_masses(ParticleType.gas).to("Msun").value
            STOPWATCH.print_verbose_info("Reading final metallicities.")
            low_z_snipshot_particle_data.data[:, 4]  = snapshots[snap_num_final].get_metalicities(ParticleType.gas).value

#            if Settings.verbose or Settings.debug:
#                timer_data_read = time.time()
#                Console.print_info(f"({time.time()}) Reading data took {timer_data_read - timer_start}s.")

            if Settings.verbose or Settings.debug:
                CHUNK_STOPWATCH.lap("Volume data loaded.")

    #        Console.print_debug(((final_ids - order_by(initial_ids)) != 0).sum())
    #        Console.print_debug(((low_z_snipshot_particle_data.data[:, 3] - high_z_snipshot_particle_data.data[:, 3]) < 0).sum())
    #        Console.print_debug(((low_z_snipshot_particle_data.data[:, 4] - high_z_snipshot_particle_data.data[:, 4]) < 0).sum())
    #        invalid_particles_mask = (low_z_snipshot_particle_data.data[:, 4] - high_z_snipshot_particle_data.data[:, 4]) < 0
    #        invalid_indexes = np.where(invalid_particles_mask)[0]
    #        print(flush = True)
    #        for i in invalid_indexes[:20]:
    #            Console.print_debug("Index:", i)
    #            Console.print_debug("ID:", order_by(initial_ids)[i], "->", final_ids[i])
    #            Console.print_debug("Mass:", high_z_snipshot_particle_data.data[i, 3], "->", low_z_snipshot_particle_data.data[i, 3])
    #            Console.print_debug("Metallicity:", high_z_snipshot_particle_data.data[i, 4], "->", low_z_snipshot_particle_data.data[i, 4])
    #            Console.print_debug("Metallicity change:", low_z_snipshot_particle_data.data[i, 4] - high_z_snipshot_particle_data.data[i, 4])
    #            Console.print_debug("Metallicity change:", 100 * (low_z_snipshot_particle_data.data[i, 4] - high_z_snipshot_particle_data.data[i, 4]) / high_z_snipshot_particle_data.data[i, 4], "%")
    #            print(flush = True)
    #        percentage_changes = 100 * np.abs(low_z_snipshot_particle_data.data[invalid_indexes, 4] - high_z_snipshot_particle_data.data[invalid_indexes, 4]) / high_z_snipshot_particle_data.data[invalid_indexes, 4]
    #        Console.print_debug("Min absolute metallicity change:", percentage_changes.min(), "%")
    #        Console.print_debug("Max absolute metallicity change:", percentage_changes.max(), "%")
    #        Console.print_debug("Mean absolute metallicity change:", percentage_changes.mean(), "%")
    #        Console.print_debug("Median absolute metallicity change:", np.median(percentage_changes), "%")
    #        return

            # Interpolate snipshot data to match line-of-sight redshift
            interp_fraction = (sightline_file.a - snap_expansion_factors[snap_num_initial]) / (snap_expansion_factors[snap_num_final] - snap_expansion_factors[snap_num_initial])
            if not get_extra_data:
                interpolated_snipshot_particle_data.data[:, :3] = high_z_snipshot_particle_data.data[:, :3] * (1 - interp_fraction) + (low_z_snipshot_particle_data.data[:, :3] * interp_fraction)
            else:
                interpolated_snipshot_particle_data.data[:, :] = high_z_snipshot_particle_data.data[:, :] * (1 - interp_fraction) + (low_z_snipshot_particle_data.data[:, :] * interp_fraction)

#            Console.print_verbose_info(f"({time.time()}) Done interpolating data. Re-computing positions that need wrapping.")

            if Settings.verbose or Settings.debug:
                STOPWATCH.print("Unwrapped interpolation completed.")

            position_delta = low_z_snipshot_particle_data.data[:, :3] - high_z_snipshot_particle_data.data[:, :3]
            coords_needing_wrapping = np.abs(position_delta) > half_box_size
            for i in range(3):
                this_coord_needs_wrapping = coords_needing_wrapping[:, i]
                UNWRAPPED_FINAL_POSITION = low_z_snipshot_particle_data.data[this_coord_needs_wrapping, i] - (np.sign(position_delta[this_coord_needs_wrapping, i]) * box_size)
                UNWRAPPED_INTERPOLATED_POSITION = high_z_snipshot_particle_data.data[this_coord_needs_wrapping, i] + (interp_fraction * (UNWRAPPED_FINAL_POSITION - high_z_snipshot_particle_data.data[this_coord_needs_wrapping, i]))

                needs_re_wrapping = (UNWRAPPED_INTERPOLATED_POSITION < 0) | (UNWRAPPED_INTERPOLATED_POSITION > box_size)

                WRAPPING_CORRECTED_SUBSET = UNWRAPPED_INTERPOLATED_POSITION[needs_re_wrapping] - (np.sign(UNWRAPPED_INTERPOLATED_POSITION[needs_re_wrapping]) * box_size)
                
                UNWRAPPED_INTERPOLATED_POSITION[needs_re_wrapping] = WRAPPING_CORRECTED_SUBSET

                interpolated_snipshot_particle_data.data[this_coord_needs_wrapping, i] = UNWRAPPED_INTERPOLATED_POSITION

#            if Settings.verbose or Settings.debug:
#                timer_stop = time.time()
#                Console.print_info(f"({time.time()}) Interpolating data took {timer_stop - timer_data_read}s.")

            if Settings.verbose or Settings.debug:
                CHUNK_STOPWATCH.lap("Volume data interpolated.")

        for los_index in do_sightlines:
            Console.print_info(f"LOS{los_index}", end = " " if not (Settings.verbose or Settings.debug) else "\n")

            if Settings.verbose or Settings.debug:
#                timer_start = time.time()
                LOS_STOPWATCH = Stopwatch("Line-of-sight", print_on_start = False)

            LOS_STOPWATCH.print_verbose_info("Loading line-of-sight data.")

            los = sightline_file.get_sightline(los_index, cache_data = False)
            los_n_particles = len(los)

            los_data = SharedArray.create((los_n_particles, 5), np.float64, name = "los_data")
            los_data.data[:, :3] = los.positions_comoving.to("Mpc").value
            los_data.data[:, 3] = los.masses.to("Msun").value
            los_data.data[:, 4] = los.metallicities.value

            LOS_STOPWATCH.print_verbose_info("Creating shared memory for result data.")

            selected_particle_indexes = SharedArray.create(los_n_particles, np.int64, name = "selected_particle_indexes")
            #selected_particle_indexes = SharedArray.create(1, np.int64, name = "selected_particle_indexes")
            selected_particle_ids = SharedArray.create(los_n_particles, np.int64, name = "selected_particle_ids")

            # Get the displacements from the line-of-sight (delta in los axis must be ignored)
            los_displacements = calculate_wrapped_displacement(los.start_position.to("Mpc").value, los_data.data[:, :3], box_size)

            if use_memory_efficient_data_loading:

                LOS_STOPWATCH.print_verbose_info("Loading data from volume chunks.")

                column_width: float = 0.0
                for i in range(3):
                    if i == np.where(los.direction > 0)[0][0]:
                        continue
#                    vmin_delta = np.abs(los.start_position.to("Mpc").value[i] - los_data.data[:, i].min())
#                    vmax_delta = np.abs(los.start_position.to("Mpc").value[i] - los_data.data[:, i].max())
                    vmin_delta = np.abs(los_displacements[:, i].min())
                    vmax_delta = np.abs(los_displacements[:, i].max())
                    delta = max(vmin_delta, vmax_delta) * 2
                    column_width = max(column_width, delta)

                (
                    masked__low_z_snipshot_ids,
                    masked__high_z_snipshot_particle_data,
                    masked__interpolated_snipshot_particle_data,
                    masked__low_z_snipshot_particle_data
                ) = read_data_column_and_interpolate(
                    snapshots[snap_num_initial],
                    snapshots[snap_num_final],
                    sightline_file.a,
                    np.where(los.direction > 0)[0][0],
                    cast(Tuple[int, int], tuple(los.start_position.to("Mpc")[np.where(~(los.direction > 0))[0]])),
                    column_width = unyt_quantity(column_width * column_width_multiplier, "Mpc"),
                    read_unnessessary_data = get_extra_data
                )

            else:

                LOS_STOPWATCH.print_verbose_info("Restricting data size.")

                los_region_particle_mask = np.full_like(low_z_snipshot_ids.data, True, dtype = np.bool_)
                for i in range(3):
                    if i == np.where(los.direction > 0)[0][0]:
                        continue

                    LOS_STOPWATCH.print_debug(f"Restricting in axis {i}")

                    los_coordinate = los.start_position.to("Mpc").value[i]

                    vmin_delta = np.abs(los_displacements[:, i].min())
                    vmax_delta = np.abs(los_displacements[:, i].max())
                    half_column_width = max(vmin_delta, vmax_delta) * column_width_multiplier

                    wrapping_required = los_coordinate - half_column_width < 0 or los_coordinate + half_column_width > box_size
                    if not wrapping_required:
                        los_region_particle_mask &= (interpolated_snipshot_particle_data.data[:, i] >= los_coordinate - half_column_width)
                        los_region_particle_mask &= (interpolated_snipshot_particle_data.data[:, i] <= los_coordinate + half_column_width)

                    else:
                        # Selection region overlaps the box boundary
                        # Apply wrapping to coordinates before masking

                        LOS_STOPWATCH.print_verbose_info(f"Shifting and wrapping particle positions as line-of-sight too close to edge (axis index {i}).")

                        # Shift coords
                        shift_displacement = half_box_size - los_coordinate
                        interpolated_snipshot_particle_data.data[:, i] += shift_displacement

                        # Record which are wrapped
                        wrapped_particles_mask = (interpolated_snipshot_particle_data.data[:, i] < 0) | (interpolated_snipshot_particle_data.data[:, i] > box_size)

                        # Wrap coords
                        np.mod(interpolated_snipshot_particle_data.data[:, i], box_size, out = interpolated_snipshot_particle_data.data[:, i], where = wrapped_particles_mask)

                        # Compute maxk changes
                        los_region_particle_mask &= (interpolated_snipshot_particle_data.data[:, i] >= half_box_size - half_column_width)
                        los_region_particle_mask &= (interpolated_snipshot_particle_data.data[:, i] <= half_box_size + half_column_width)

                        # Reverse the coordinate changes made above

                        # Undo shift
                        interpolated_snipshot_particle_data.data[:, i] -= shift_displacement
                        
                        # Wrap coords
                        np.mod(interpolated_snipshot_particle_data.data[:, i], box_size, out = interpolated_snipshot_particle_data.data[:, i], where = wrapped_particles_mask)

                n_masked_particles = los_region_particle_mask.sum()

                masked__low_z_snipshot_ids = SharedArray.create(n_masked_particles, np.int64, name = "masked__low_z_snipshot_ids")
                masked__high_z_snipshot_particle_data = SharedArray.create((n_masked_particles, high_z_snipshot_particle_data.data.shape[1]), np.float64, name = "masked__high_z_snipshot_particle_data")
                masked__interpolated_snipshot_particle_data = SharedArray.create((n_masked_particles, interpolated_snipshot_particle_data.data.shape[1]), np.float64, name = "masked__interpolated_snipshot_particle_data")
                masked__low_z_snipshot_particle_data = SharedArray.create((n_masked_particles, low_z_snipshot_particle_data.data.shape[1]), np.float64, name = "masked__low_z_snipshot_particle_data")

                masked__low_z_snipshot_ids.data[:] = low_z_snipshot_ids.data[los_region_particle_mask]
                masked__high_z_snipshot_particle_data.data[:, :] = high_z_snipshot_particle_data.data[los_region_particle_mask, :]
                masked__interpolated_snipshot_particle_data.data[:, :] = interpolated_snipshot_particle_data.data[los_region_particle_mask, :]
                masked__low_z_snipshot_particle_data.data[:, :] = low_z_snipshot_particle_data.data[los_region_particle_mask, :]

            if Settings.verbose or Settings.debug:
                LOS_STOPWATCH.lap("Line-of-sight data loaded.")
#                time_after_data_load = time.time()

            if n_processes > 1 and selected_los_particle_index is None:
                with multiprocessing.Pool(processes = n_processes) as pool:
                    pool.starmap(
                        find_match_parallel_wrapper,
                        PartialZip(
                            [0],
                            range(los_n_particles),
                            selected_particle_indexes.info,
                            selected_particle_ids.info,
                            np.where(los.direction > 0)[0][0],
                            los.start_position.to("Mpc").value,
                            los_data.info,
                            fractional_los_quantity_error,
                            box_size,
                            masked__low_z_snipshot_ids.info,
                            masked__high_z_snipshot_particle_data.info,
                            masked__interpolated_snipshot_particle_data.info,
                            masked__low_z_snipshot_particle_data.info
                        )
                    )
            else:
                for i in range(los_n_particles) if selected_los_particle_index is None else (selected_los_particle_index, ):
                    find_match(
                        i,
                        selected_particle_indexes.data,
                        selected_particle_ids.data,
                        np.where(los.direction > 0)[0][0],
                        los.start_position.to("Mpc").value,
                        los_data.data,
                        fractional_los_quantity_error,
                        box_size,
                        masked__low_z_snipshot_ids.data,
                        masked__high_z_snipshot_particle_data.data,
                        masked__interpolated_snipshot_particle_data.data,
                        masked__low_z_snipshot_particle_data.data
                    )

            if Settings.verbose or Settings.debug:
                LOS_STOPWATCH.lap("Particle IDs matched.")
#                time_after_recovery = time.time()

            if selected_los_particle_index is None:
                (print if not (Settings.verbose or Settings.debug) else Console.print_info)("duplicates:", len(los) - np.unique(selected_particle_ids.data).shape[0], end = (" " if not (Settings.verbose or Settings.debug) else "\n"), flush = True)
                with h5.File(output_file, "a") as file:
                    if f"LOS{los_index}" in file[output_file_group_name]:
                        if force_selection:
                            del file[output_file_group_name][f"LOS{los_index}"]
                        else:
                            raise KeyError("Attempted to write data to a LoS group that already exists when not using the --force-selection option.\nThis should not be possible - please report this.")
                    file[output_file_group_name].create_dataset(f"LOS{los_index}", data = selected_particle_ids.data)
                #print("(written to file)", end = " ", flush = True)
                if Settings.verbose or Settings.debug:
                    LOS_STOPWATCH.stop("Line-of-sight data saved.")
                    LOS_STOPWATCH.print_info(f"Line-of-sight complete ({' + '.join(LOS_STOPWATCH.get_lap_delta_times(string = True))})")
#                    timer_stop = time.time()
#                    print(f"    took {timer_stop - timer_start}s ({time_after_data_load - timer_start} + {time_after_recovery - time_after_data_load} + {timer_stop - time_after_recovery})", flush = True)
                if plot_selection_stats:
                    Console.print_verbose_info("Plotting statistics.")
#                    delta_distances = np.sqrt(((los_data.data[:, :3] - masked__interpolated_snipshot_particle_data.data[selected_particle_indexes.data, :3])**2).sum(axis = 1))
                    delta_distances = calculate_wrapped_distance(masked__interpolated_snipshot_particle_data.data[selected_particle_indexes.data, :3], los_data.data[:, :3], box_size) * 1000 # Conversion to kpc
                    delta_masses = los_data.data[:, 3] - masked__interpolated_snipshot_particle_data.data[selected_particle_indexes.data, 3]
                    delta_metallicities = los_data.data[:, 4] - masked__interpolated_snipshot_particle_data.data[selected_particle_indexes.data, 4]
                    def get_bin_edges(n: int, data: np.ndarray):
                        vmin = data.min()
                        vmax = data.max()
                        if vmin >= 0:
                            return np.linspace(0, vmax, n)
                        elif vmax <= 0:
                            return np.linspace(vmin, 0, n)
                        elif vmin != vmax:
                            max_to_width_ratio = vmax / (vmax - vmin)
                            return list(np.linspace(vmin, 0, int(np.ceil(n * (1 - max_to_width_ratio))))[:-1]) + list(np.linspace(0, vmax, int(np.ceil(n * max_to_width_ratio))))
                        else:
                            delta = np.abs(vmax) / 10
                            return [vmax - delta, vmax, vmax + delta]
                    fig, axes = plt.subplots(1, 3, figsize = (18, 6), layout = "tight")
                    axes[0].set_title("Distance")
                    axes[0].hist(delta_distances, bins = get_bin_edges(60, delta_distances), log = True)
                    axes[0].set_xlabel("|$r_{\\rm interp. -> los}$| kpc")
                    axes[1].set_title("$\\Delta$ Mass")
                    axes[1].hist(delta_masses, bins = get_bin_edges(60, delta_masses), log = True)
                    axes[1].set_xlabel("$M_{\\rm los}$ - $M$ $\\rm M_{\\rm \\odot}$")
                    axes[2].set_title("$\\Delta$ Metallicity")
                    axes[2].hist(delta_metallicities, bins = get_bin_edges(60, delta_metallicities), log = True)
                    axes[2].set_xlabel("$Z_{\\rm los}$ - $Z$")
                    plt.savefig(os.path.join(os.path.split(output_file)[0], "stats.png"))
            else:
                if Settings.verbose or Settings.debug:
                    LOS_STOPWATCH.stop()
                    LOS_STOPWATCH.print_info(f"Line-of-sight complete ({' + '.join(LOS_STOPWATCH.get_lap_delta_times(string = True))})")
#                    timer_stop = time.time()
#                    print(f"    took {timer_stop - timer_start}s ({time_after_data_load - timer_start} + {time_after_recovery - time_after_data_load} + {timer_stop - time_after_recovery})", flush = True)
                selected_particle_index = selected_particle_indexes.data[selected_los_particle_index]#np.where(low_z_snipshot_ids.data == selected_particle_ids.data[selected_los_particle_index])[0][0]
                delta_position = masked__interpolated_snipshot_particle_data.data[selected_particle_index, :3] - los_data.data[selected_los_particle_index, :3]
                delta_distance = np.sqrt((delta_position**2).sum())
                delta_mass = masked__interpolated_snipshot_particle_data.data[selected_particle_index, 3] - los_data.data[selected_los_particle_index, 3]
                delta_metallicity = masked__interpolated_snipshot_particle_data.data[selected_particle_index, 4] - los_data.data[selected_los_particle_index, 4]
                print(f"""\
Target Line of Sight Particle:
       particle index = {selected_los_particle_index}
            los index = {los_index}
             los file = {f}
    los file redshift = {sightline_file.z}

             position = {los_data.data[selected_los_particle_index, :3]} Mpc
                 mass = {los_data.data[selected_los_particle_index, 3]} Msun
          metallicity = {los_data.data[selected_los_particle_index, 4]}

Matched particle:
                   ID = {selected_particle_ids.data[selected_los_particle_index]}
                index = {selected_particle_index} (data subset, NOT snapshot)

Matched particle (previous snapshot/snipshot):
                 file = {snapshot_filepaths_by_number[snap_num_initial]}
             redshift = {snapshots[snap_num_initial].z}
             position = {masked__high_z_snipshot_particle_data.data[selected_particle_index, :3]} Mpc
                 mass = {masked__high_z_snipshot_particle_data.data[selected_particle_index, 3]} Msun
          metallicity = {masked__high_z_snipshot_particle_data.data[selected_particle_index, 4]}

Matched particle (interpolated):
             fraction = {interp_fraction}
             redshift = {sightline_file.z}
             position = {masked__interpolated_snipshot_particle_data.data[selected_particle_index, :3]} Mpc
                 mass = {masked__interpolated_snipshot_particle_data.data[selected_particle_index, 3]} Msun
          metallicity = {masked__interpolated_snipshot_particle_data.data[selected_particle_index, 4]}
           \u0394 position =  {delta_distance} Mpc ({'+' if delta_position[0] >= 0 else ''}{delta_position[0]}, {'+' if delta_position[1] >= 0 else ''}{delta_position[1]}, {'+' if delta_position[2] >= 0 else ''}{delta_position[2]})
               \u0394 mass = {'+' if delta_mass >= 0 else ''}{delta_mass} Msun
        \u0394 metallicity = {'+' if delta_metallicity >= 0 else ''}{delta_metallicity}

Matched particle (subsequent snapshot/snipshot):
                 file = {snapshot_filepaths_by_number[snap_num_final]}
             redshift = {snapshots[snap_num_final].z}
             position = {masked__low_z_snipshot_particle_data.data[selected_particle_index, :3]} Mpc
                 mass = {masked__low_z_snipshot_particle_data.data[selected_particle_index, 3]} Msun
          metallicity = {masked__low_z_snipshot_particle_data.data[selected_particle_index, 4]}
""", flush = True)
            los_data.free(force = True)
            selected_particle_indexes.free(force = True)
            selected_particle_ids.free(force = True)
            masked__low_z_snipshot_ids.free(force = True)
            masked__high_z_snipshot_particle_data.free(force = True)
            masked__interpolated_snipshot_particle_data.free(force = True)
            masked__low_z_snipshot_particle_data.free(force = True)
        if not use_memory_efficient_data_loading:
            low_z_snipshot_ids.free(force = True)
            high_z_snipshot_particle_data.free(force = True)
            interpolated_snipshot_particle_data.free(force = True)
            low_z_snipshot_particle_data.free(force = True)
        print(flush = True)



def read_data_column_and_interpolate(
        initial_snap: SnapshotEAGLE,
        final_snap: SnapshotEAGLE,
        line_of_sight_expansion_factor: float,
        projection_axis_index: int,
        face_centre: Tuple[unyt_quantity, unyt_quantity],
        column_width: unyt_quantity = unyt_quantity(2.0, "Mpc"),
        read_unnessessary_data: bool = False
) -> Tuple[SharedArray, SharedArray, SharedArray, SharedArray]:

    Console.print_verbose_info(f"Reading particle data with column width of {column_width.value} {column_width.units}.")

    # Compute the region limits

    half_column_width = column_width / 2
    restriction_region_args: List[unyt_quantity|None] = [None, None, None, None, None, None]
    face_index = 0
    for i in range(3):
        if i == projection_axis_index:
            continue # Non need to explicitly set values for the axis of projection as this will be automatically set to the length of the box
        restriction_region_args[i * 2] = face_centre[face_index] - half_column_width
        restriction_region_args[i * 2 + 1] = face_centre[face_index] + half_column_width
        face_index += 1

    Console.print_debug(f"LoS centre coords are {face_centre}.")
    Console.print_debug(f"Column width is {column_width}.")
    Console.print_verbose_info(f"Column limits are {', '.join([((str(v.value) + ' ' + str(v.units)) if v is not None else '---') for v in restriction_region_args])}.")

    # Restrict the region to read data from

    initial_snap.restrict_data_comoving_loading_region(*restriction_region_args)
    final_snap.restrict_data_comoving_loading_region(*restriction_region_args)

    # Find the particles that appear in both snapshots

    Console.print_verbose_info("Loading particle ID data.")

    initial_ids = initial_snap.get_IDs(ParticleType.gas)
    final_ids = final_snap.get_IDs(ParticleType.gas)

    common_ids = np.intersect1d(initial_ids, final_ids, assume_unique = True)
    n_returned_particles = common_ids.shape[0]

    Console.print_verbose_info(f"Got {n_returned_particles} particles.")

    # Create callable objects for reorganising the read data

    initial_sorter = ArrayReorder_2.create(initial_ids, common_ids)
    final_sorter = ArrayReorder_2.create(final_ids, common_ids)

    # Delete unnessessary ID data

    del initial_ids, final_ids

    # Create shared memory for loaded and interpolated data

    ordered_data_ids = SharedArray.create(n_returned_particles, np.int64, name = "ordered_data_ids")
    initial_snapshot_data = SharedArray.create((n_returned_particles, 5), np.float64, name = "high_z_snipshot_particle_data")
    final_snapshot_data = SharedArray.create((n_returned_particles, 5), np.float64, name = "low_z_snipshot_particle_data")
    interpolated_snapshot_data = SharedArray.create((n_returned_particles, 3 if not read_unnessessary_data else 5), np.float64, name = "interpolated_snipshot_particle_data") # Only need interpolated positions

    # Move the IDs into shared memory
    ordered_data_ids.data[:] = common_ids[:]

    # Load the position data

    Console.print_verbose_info("Loading particle position data.")

    initial_snapshot_data.data[:, :3] = initial_sorter(initial_snap.get_positions(ParticleType.gas).to("Mpc").value)
    final_snapshot_data.data[:, :3] = final_sorter(final_snap.get_positions(ParticleType.gas).to("Mpc").value)

    # Interpolate positions to match the line-of-sight redshift (before remaining data is loaded into memory)

    Console.print_verbose_info("Interpolating particle positions.")

    interp_fraction = (line_of_sight_expansion_factor - initial_snap.a) / (final_snap.a - initial_snap.a)
    interpolated_snapshot_data.data[:, :3] = initial_snapshot_data.data[:, :3] * (1 - interp_fraction) + (final_snapshot_data.data[:, :3] * interp_fraction)

    box_size = final_snap.box_size[0].to("Mpc").value
    position_delta = final_snapshot_data.data[:, :3] - initial_snapshot_data.data[:, :3]
    coords_needing_wrapping = np.abs(position_delta) > box_size / 2
    for i in range(3):
        this_coord_needs_wrapping = coords_needing_wrapping[:, i]
        UNWRAPPED_FINAL_POSITION = final_snapshot_data.data[this_coord_needs_wrapping, i] - (np.sign(position_delta[this_coord_needs_wrapping, i]) * box_size)
        UNWRAPPED_INTERPOLATED_POSITION = initial_snapshot_data.data[this_coord_needs_wrapping, i] + (interp_fraction * (UNWRAPPED_FINAL_POSITION - initial_snapshot_data.data[this_coord_needs_wrapping, i]))

        needs_re_wrapping = (UNWRAPPED_INTERPOLATED_POSITION < 0) | (UNWRAPPED_INTERPOLATED_POSITION > box_size)

        WRAPPING_CORRECTED_SUBSET = UNWRAPPED_INTERPOLATED_POSITION[needs_re_wrapping] - (np.sign(UNWRAPPED_INTERPOLATED_POSITION[needs_re_wrapping]) * box_size)
        
        UNWRAPPED_INTERPOLATED_POSITION[needs_re_wrapping] = WRAPPING_CORRECTED_SUBSET

        interpolated_snapshot_data.data[this_coord_needs_wrapping, i] = UNWRAPPED_INTERPOLATED_POSITION

    # Load remaining data

    Console.print_verbose_info("Loading particle mass and metallicity data.")

    initial_snapshot_data.data[:, 3] = initial_sorter(initial_snap.get_masses(ParticleType.gas).to("Msun").value)
    final_snapshot_data.data[:, 3] = final_sorter(final_snap.get_masses(ParticleType.gas).to("Msun").value)

    initial_snapshot_data.data[:, 4] = initial_sorter(initial_snap.get_metalicities(ParticleType.gas).value)
    final_snapshot_data.data[:, 4] = final_sorter(final_snap.get_metalicities(ParticleType.gas).value)

    if read_unnessessary_data:
        Console.print_verbose_info("Interpolating particle mass and metallicity data.")
        interpolated_snapshot_data.data[:, 3:] = initial_snapshot_data.data[:, 3:] * (1 - interp_fraction) + (final_snapshot_data.data[:, 3:] * interp_fraction)

    # Explicitly clean up the sorting opbjects which will contain large arrays

    del initial_sorter, final_sorter

    # Return the shared memory arrays

    return ordered_data_ids, initial_snapshot_data, interpolated_snapshot_data, final_snapshot_data



def find_match(
        los_particle_index: int,
        recovered_los_particle_snapshot_indexes: np.ndarray,
        recovered_los_particle_ids: np.ndarray,
        los_projection_axis_index: int,
        los_start_position: np.ndarray,
        los_particle_data: np.ndarray,
        fractional_los_quantity_error: float,
        box_side_length: float,
        snap_particle_ids: np.ndarray,
        high_z_particle_data: np.ndarray,
        interpolated_particle_data: np.ndarray,
        low_z_particle_data: np.ndarray
    ) -> None:
    """
    Identify the most likley matching particle ID for a particle in a line-of-sight.
    """

    los_target_particle_data = los_particle_data[los_particle_index]

    # Particle mass is greater than or equal to the initial mass of each particle in the higher redshift snipshot
    initial_mass_filter = high_z_particle_data[:, 3] <= los_target_particle_data[3] * (1.0 + fractional_los_quantity_error)

    # Particle mass is less than or equal to the initial mass of each particle in the higher redshift snipshot
    final_mass_filter = low_z_particle_data[:, 3] >= los_target_particle_data[3] * (1.0 - fractional_los_quantity_error)

    # Particle metallicity is greater than or equal to the initial mass of each particle in the higher redshift snipshot
    initial_metallicity_filter = high_z_particle_data[:, 4] <= los_target_particle_data[4] * (1.0 + fractional_los_quantity_error)

    # Particle metallicity is less than or equal to the initial mass of each particle in the higher redshift snipshot
    final_metallicity_filter = low_z_particle_data[:, 4] >= los_target_particle_data[4] * (1.0 - fractional_los_quantity_error)

    # Particle mass is within the mass-range transitioned through by each particle in the snipshot data
    transition_mass_filter = initial_mass_filter & final_mass_filter

    # Particle metallicity is within the metallicity-range transitioned through by each particle in the snipshot data
    transition_metallicity_filter = initial_metallicity_filter & final_metallicity_filter

    # Combin both mass and metallicity filters
    valid_particles_mass_and_metallicity = transition_mass_filter & transition_metallicity_filter

    if valid_particles_mass_and_metallicity.sum() == 0:
        print(flush = True)
        Console.print_error(f"Unable to find valid nearby particle patch for line-of-sight particle index {los_particle_index}.\nNo nearby particles had acceptable mass and mtallicity restrictions.")
        Console.print_info("Continuing using the nearest particle.")#TODO: why aren't these gettng printed???

        # Unable to restrict search
        valid_particles_mass_and_metallicity[:] = True
        valid_interpolated_particles = interpolated_particle_data

    else:
        # Mask the particle data based on which particles are valid
        valid_interpolated_particles = interpolated_particle_data[valid_particles_mass_and_metallicity]

#    position_deltas = valid_interpolated_particles[:, :3] - los_particle_data[los_particle_index, :3]
#    deltas_needing_wrapping = np.abs(position_deltas) > box_side_length / 2
#    position_deltas[deltas_needing_wrapping] = position_deltas[deltas_needing_wrapping] - (np.sign(position_deltas[deltas_needing_wrapping]) * box_side_length)
#    distances_squared = (position_deltas**2).sum(axis = 1)
#    del position_deltas

    distances_squared = calculate_wrapped_distance(los_particle_data[los_particle_index, :3], valid_interpolated_particles[:, :3], box_side_length, do_squared_distance = True)

    selected_part_subset_index = distances_squared.argmin()
    del distances_squared

    #TODO: is this really nessesary? Performance test?
    if recovered_los_particle_snapshot_indexes.shape[0] > 1:
        # Indexes are required
        recovered_los_particle_snapshot_indexes[los_particle_index] = np.where(valid_particles_mass_and_metallicity)[0][selected_part_subset_index]
        recovered_los_particle_ids[los_particle_index] = snap_particle_ids[recovered_los_particle_snapshot_indexes[los_particle_index]]
    else:
        # No indexes are required, so use the most efficient option for getting the ID
        recovered_los_particle_ids[los_particle_index] = snap_particle_ids[valid_particles_mass_and_metallicity][selected_part_subset_index]



def find_match_parallel_wrapper(
        los_particle_index: int,
        recovered_los_particle_snapshot_indexes: SharedArray_TransmissionData,
        recovered_los_particle_ids: SharedArray_TransmissionData,
        los_projection_axis_index: int,
        los_start_position: np.ndarray,
        los_particle_data: SharedArray_TransmissionData,
        fractional_los_quantity_error: float,
        box_side_length: float,
        snap_particle_ids: SharedArray_TransmissionData,
        high_z_particle_data: SharedArray_TransmissionData,
        interpolated_particle_data: SharedArray_TransmissionData,
        low_z_particle_data: SharedArray_TransmissionData
    ) -> None:
    loaded__recovered_los_particle_snapshot_indexes = recovered_los_particle_snapshot_indexes.load()
    loaded__recovered_los_particle_ids = recovered_los_particle_ids.load()
    loaded__los_particle_data = los_particle_data.load()
    loaded__snap_particle_ids = snap_particle_ids.load()
    loaded__high_z_particle_data = high_z_particle_data.load()
    loaded__interpolated_particle_data = interpolated_particle_data.load()
    loaded__low_z_particle_data = low_z_particle_data.load()
    find_match(
        los_particle_index,
        loaded__recovered_los_particle_snapshot_indexes.data,
        loaded__recovered_los_particle_ids.data,
        los_projection_axis_index,
        los_start_position,
        loaded__los_particle_data.data,
        fractional_los_quantity_error,
        box_side_length,
        loaded__snap_particle_ids.data,
        loaded__high_z_particle_data.data,
        loaded__interpolated_particle_data.data,
        loaded__low_z_particle_data.data
    )
    loaded__recovered_los_particle_snapshot_indexes.free()
    loaded__recovered_los_particle_ids.free()
    loaded__los_particle_data.free()
    loaded__snap_particle_ids.free()
    loaded__high_z_particle_data.free()
    loaded__interpolated_particle_data.free()
    loaded__low_z_particle_data.free()
