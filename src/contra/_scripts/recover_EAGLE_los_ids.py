# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
#BAD_TESTING_SNIPSHOTS = ("snip_354_z000p225.0.hdf5", "snip_288_z000p770.0.hdf5", "snip_224_z001p605.0.hdf5", "snip_159_z002p794.0.hdf5")
from contra import ParticleType, ArrayReorder, SharedArray, SharedArray_TransmissionData, calculate_wrapped_displacement, calculate_wrapped_distance
from contra.io import SnapshotEAGLE, LineOfSightFileEAGLE
import h5py as h5
import numpy as np
from typing import Any, Union, List, Dict, Tuple, Iterable, Iterator
import os
from QuasarCode import Console, Settings
from QuasarCode.Tools import ScriptWrapper, ArrayVisuliser
import datetime
import multiprocessing
from multiprocessing import shared_memory
import ctypes
import time
from matplotlib import pyplot as plt

ENABLE_WRAPPED_INTERPOLATION = True

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
            )
        )
    ).run_with_async(__main)



async def __main(
            output_file: str,
            data_directory: str,
            los_directory: str,
            n_processes: int,
            fractional_los_quantity_error: float,
            selected_file: str | None,
            selected_los_index: int | None,
            selected_los_particle_index: int | None,
            force_selection: bool,
            plot_selection_stats: bool
          ) -> None:

    # Get the line of sight filepaths in redshift order (assending z)
    los_files = [os.path.join(los_directory, file) for file in list(*os.walk(los_directory))[2] if file[:8] == "part_los"]
#    Console.print_debug(los_files)
#    return
    los_files.sort(key = lambda v: float(v.rsplit("z", maxsplit = 1)[1].rsplit(".", maxsplit = 1)[0]))

    if selected_file is not None:
        selected_file = os.path.join(los_directory, selected_file)
        if selected_file not in los_files:
            Console.print_error("A specific Line of Sight file was specified ({selected_file}) but does not exist at location \"{los_directory}\".")
            return

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
    for f in los_files if selected_file is None else (selected_file, ):
        sightline_file = LineOfSightFileEAGLE(f)

        output_file_group_name = f.rsplit(os.path.sep, maxsplit = 1)[-1]

        # Skip any completed files (allows for restarting)
        # This will confirm that all sightlines have been completed and will retain any sightlines already completed
        completed_sightlines = 0
        if output_file_group_name in complete_files:
            with h5.File(output_file, "r") as file:
                completed_sightlines = len(list(file[output_file_group_name]))
            if completed_sightlines == len(sightline_file):
                continue
        else:
            with h5.File(output_file, "a") as file:
                file.create_group(output_file_group_name)

        Console.print_info(output_file_group_name)

        if selected_los_index is None:
            do_sightlines = range(completed_sightlines, len(sightline_file))
        else:
            min_valid_sightline = 0 if force_selection else completed_sightlines
            if selected_los_particle_index is None and (selected_los_index < min_valid_sightline or selected_los_index >= len(sightline_file)):
                Console.print_error(f"Selected sightline ({selected_los_index}) outside of valid range (inclusive) {min_valid_sightline} -> {len(sightline_file) - 1}.\nTo re-compute a sightline for which there is already data, use the --force-selection flag.")
                return
            do_sightlines = (selected_los_index, )

        if selected_los_particle_index is not None:
            n_avalible_particles = sightline_file.get_sightline_length(do_sightlines[0])
            if (selected_los_particle_index >= 0 and selected_los_particle_index >= n_avalible_particles) or (selected_los_particle_index < 0 and -selected_los_particle_index > n_avalible_particles):
                Console.print_error(f"Selected sightline particle index ({selected_los_particle_index}) out of range for sightline ({selected_los_index}) with {n_avalible_particles} particles.")
                return

        # Get the numbers of the snipshots that border the line of sight file's redshift
        snap_num_initial, snap_num_final = find_neighbouring_snapshots(sightline_file.z)

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

        n_remaining_snip_particles = snapshots[snap_num_final].number_of_particles(ParticleType.gas)

        low_z_snipshot_ids                  = SharedArray.create(n_remaining_snip_particles,      np.int64  , name = "low_z_snipshot_ids")
        high_z_snipshot_particle_data       = SharedArray.create((n_remaining_snip_particles, 5), np.float64, name = "high_z_snipshot_particle_data")
        interpolated_snipshot_particle_data = SharedArray.create((n_remaining_snip_particles, 5), np.float64, name = "interpolated_snipshot_particle_data")
        low_z_snipshot_particle_data        = SharedArray.create((n_remaining_snip_particles, 5), np.float64, name = "low_z_snipshot_particle_data")

        # Determine snipshot data reorder
        initial_ids = snapshots[snap_num_initial].get_IDs(ParticleType.gas)
        final_ids = snapshots[snap_num_final].get_IDs(ParticleType.gas)
        order_by = ArrayReorder.create(initial_ids, final_ids)

        # Read snipshot particle data into shared memory
        high_z_snipshot_all_particle_positions = snapshots[snap_num_initial].get_positions(ParticleType.gas).to("Mpc").value
        high_z_snipshot_particle_data.data[:, 0] = order_by(high_z_snipshot_all_particle_positions[:, 0])
        high_z_snipshot_particle_data.data[:, 1] = order_by(high_z_snipshot_all_particle_positions[:, 1])
        high_z_snipshot_particle_data.data[:, 2] = order_by(high_z_snipshot_all_particle_positions[:, 2])
        high_z_snipshot_particle_data.data[:, 3] = order_by(snapshots[snap_num_initial].get_masses(ParticleType.gas).to("Msun").value)
        high_z_snipshot_particle_data.data[:, 4] = order_by(snapshots[snap_num_initial].get_metalicities(ParticleType.gas).value)
        low_z_snipshot_ids.data[:] = final_ids
        low_z_snipshot_particle_data.data[:, :3] = snapshots[snap_num_final].get_positions(ParticleType.gas).to("Mpc").value
        low_z_snipshot_particle_data.data[:, 3]  = snapshots[snap_num_final].get_masses(ParticleType.gas).to("Msun").value
        low_z_snipshot_particle_data.data[:, 4]  = snapshots[snap_num_final].get_metalicities(ParticleType.gas).value

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
        interpolated_snipshot_particle_data.data[:, :] = high_z_snipshot_particle_data.data * (1 - interp_fraction) + (low_z_snipshot_particle_data.data * interp_fraction)

#        position_delta = low_z_snipshot_particle_data.data[:, :3] - high_z_snipshot_particle_data.data[:, :3]
#        coords_needing_wrapping = np.abs(position_delta) > half_box_size
##        Console.print_debug(np.where(coords_needing_wrapping.sum(axis = 1) > 0)[0])
#        Console.print_debug(interp_fraction)
#        Console.print_debug(position_delta[67, :])
#        print()
#        Console.print_debug(high_z_snipshot_particle_data.data[67, :3])
#        Console.print_debug(interpolated_snipshot_particle_data.data[67, :3])
#        Console.print_debug(low_z_snipshot_particle_data.data[67, :3])
##        return

        if ENABLE_WRAPPED_INTERPOLATION:
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

#        print()
#        Console.print_debug(high_z_snipshot_particle_data.data[67, :3])
#        Console.print_debug(interpolated_snipshot_particle_data.data[67, :3])
#        Console.print_debug(low_z_snipshot_particle_data.data[67, :3])
#
#        return

#        if ENABLE_WRAPPED_INTERPOLATION:#NOT WORKING!!!!!
#            position_delta = low_z_snipshot_particle_data.data[:, :3] - high_z_snipshot_particle_data.data[:, :3]
#            coords_needing_wrapping = np.abs(position_delta) > half_box_size
#            for i in range(3):
#                interpolated_snipshot_particle_data.data[coords_needing_wrapping[:, i], i] = (half_box_size - position_delta[coords_needing_wrapping[:, i], i]) * interp_fraction + high_z_snipshot_particle_data.data[coords_needing_wrapping[:, i], i]
#            del position_delta
#            del coords_needing_wrapping
#            for i in range(3):
#                new_coords_needing_wrapping = (interpolated_snipshot_particle_data.data[:, i] > box_size) | (interpolated_snipshot_particle_data.data[:, i] < 0)
#                interpolated_snipshot_particle_data.data[new_coords_needing_wrapping, i] = interpolated_snipshot_particle_data.data[new_coords_needing_wrapping, i] - (box_size * np.sign(interpolated_snipshot_particle_data.data[new_coords_needing_wrapping, i]))

        for los_index in do_sightlines:
            Console.print_verbose_info(f"LOS{los_index}", end = " ", flush = True)

            if Settings.verbose:
                timer_start = time.time()

            los = sightline_file.get_sightline(los_index, cache_data = False)
            los_n_particles = len(los)

            los_data = SharedArray.create((los_n_particles, 5), np.float64, name = "los_data")
            los_data.data[:, :3] = los.positions_comoving.to("Mpc").value
            los_data.data[:, 3] = los.masses.to("Msun").value
            los_data.data[:, 4] = los.metallicities.value

            selected_particle_indexes = SharedArray.create(los_n_particles, np.int64, name = "selected_particle_indexes")
            #selected_particle_indexes = SharedArray.create(1, np.int64, name = "selected_particle_indexes")
            selected_particle_ids = SharedArray.create(los_n_particles, np.int64, name = "selected_particle_ids")

            los_region_particle_mask = np.full_like(low_z_snipshot_ids.data, True, dtype = np.bool_)
            for i in range(3):
                if i == np.where(los.direction > 0)[0][0]:
                    continue
                vmin = los_data.data[:, i].min()
                vmax = los_data.data[:, i].max()
                half_vdelta = (vmax - vmin) / 2
                los_region_particle_mask &= (interpolated_snipshot_particle_data.data[:, i] >= vmin - half_vdelta)
                los_region_particle_mask &= (interpolated_snipshot_particle_data.data[:, i] <= vmax + half_vdelta)
            n_masked_particles = los_region_particle_mask.sum()

            masked__low_z_snipshot_ids = SharedArray.create(n_masked_particles, np.int64, name = "masked__low_z_snipshot_ids")
            masked__high_z_snipshot_particle_data = SharedArray.create((n_masked_particles, high_z_snipshot_particle_data.data.shape[1]), np.float64, name = "masked__high_z_snipshot_particle_data")
            masked__interpolated_snipshot_particle_data = SharedArray.create((n_masked_particles, interpolated_snipshot_particle_data.data.shape[1]), np.float64, name = "masked__interpolated_snipshot_particle_data")
            masked__low_z_snipshot_particle_data = SharedArray.create((n_masked_particles, low_z_snipshot_particle_data.data.shape[1]), np.float64, name = "masked__low_z_snipshot_particle_data")

            masked__low_z_snipshot_ids.data[:] = low_z_snipshot_ids.data[los_region_particle_mask]
            masked__high_z_snipshot_particle_data.data[:, :] = high_z_snipshot_particle_data.data[los_region_particle_mask, :]
            masked__interpolated_snipshot_particle_data.data[:, :] = interpolated_snipshot_particle_data.data[los_region_particle_mask, :]
            masked__low_z_snipshot_particle_data.data[:, :] = low_z_snipshot_particle_data.data[los_region_particle_mask, :]

            if Settings.verbose:
                time_after_data_load = time.time()

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

            if Settings.verbose:
                time_after_recovery = time.time()

            if selected_los_particle_index is None:
                if Settings.verbose:
                    print("duplicates:", len(los) - np.unique(selected_particle_ids.data).shape[0], end = " ", flush = True)
                with h5.File(output_file, "a") as file:
                    if f"LOS{los_index}" in file[output_file_group_name]:
                        if force_selection:
                            del file[output_file_group_name][f"LOS{los_index}"]
                        else:
                            raise KeyError("Attempted to write data to a LoS group that already exists when not using the --force-selection option.\nThis should not be possible - please report this.")
                    file[output_file_group_name].create_dataset(f"LOS{los_index}", data = selected_particle_ids.data)
                if Settings.verbose:
                    print("(written to file)", end = " ", flush = True)
                    timer_stop = time.time()
                    print(f"    took {timer_stop - timer_start}s ({time_after_data_load - timer_start} + {time_after_recovery - time_after_data_load} + {timer_stop - time_after_recovery})", flush = True)
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
                if Settings.verbose:
                    timer_stop = time.time()
                    print(f"    took {timer_stop - timer_start}s ({time_after_data_load - timer_start} + {time_after_recovery - time_after_data_load} + {timer_stop - time_after_recovery})", flush = True)
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
                index = {selected_particle_index}

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
        low_z_snipshot_ids.free(force = True)
        high_z_snipshot_particle_data.free(force = True)
        interpolated_snipshot_particle_data.free(force = True)
        low_z_snipshot_particle_data.free(force = True)
        if Settings.verbose:
            print(flush = True)



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



'''
async def old__main(
            output_file: str,
            data_directory: str,
            los_directory: str,
            n_processes: int
          ) -> None:

    los_files = [os.path.join(los_directory, file) for file in list(*os.walk(los_directory))[2]]
    los_files.sort(key = lambda v: float(v.rsplit("z", maxsplit = 1)[1].rsplit(".", maxsplit = 1)[0]))

    snipshot_files = [os.path.join(folder, files[0].rsplit(".", maxsplit = 2)[0] + ".0.hdf5") for (folder, _, files) in os.walk(data_directory) if folder.rsplit(os.path.sep, maxsplit = 1)[1][:8] == "snipshot"]
    snipshot_redshifts = [float(v.rsplit("z", maxsplit = 1)[1].rsplit(".", maxsplit = 2)[0].replace("p", ".")) for v in snipshot_files]
    snipshot_files.sort(key = lambda v: float(v.rsplit("z", maxsplit = 1)[1].rsplit(".", maxsplit = 2)[0].replace("p", ".")))
    snipshot_redshifts.sort()
    snapshot_redshifts: Dict[int, float] = { int(file.rsplit(os.path.sep, maxsplit = 1)[1].split("_")[1]) : snipshot_redshifts[i] for i, file in enumerate(snipshot_files) if file.rsplit(os.path.sep, maxsplit = 1)[1] not in BAD_TESTING_SNIPSHOTS }
    snapshots: Dict[int, SnapshotEAGLE] = {}#{ int(file.rsplit(os.path.sep, maxsplit = 1)[1].split("_")[1]) : SnapshotEAGLE(file) for file in snipshot_files if file.rsplit(os.path.sep, maxsplit = 1)[1] not in BAD_TESTING_SNIPSHOTS }
    snapshot_filepaths_by_number: Dict[int, str] = { int(file.rsplit(os.path.sep, maxsplit = 1)[1].split("_")[1]) : file for file in snipshot_files if file.rsplit(os.path.sep, maxsplit = 1)[1] not in BAD_TESTING_SNIPSHOTS }

#    snap_nums = list(snapshots.keys())
    snap_nums = list(snapshot_redshifts.keys())
    snap_nums.sort()

#    snap_redshifts = { n : snapshots[n].z for n in snap_nums}
#    snap_expansion_factors = { n : snapshots[n].a for n in snap_nums }
    snap_expansion_factors: Dict[int, float] = {}

#    def find_neighbouring_snapshots(z: float) -> Tuple[int, int]:
#        if z > snapshots[snap_nums[0]].z or z < snapshots[snap_nums[-1]].z:
#            raise ValueError(f"Redshift {z} outside of redshift range of avalible data.")
#        lower_redshift_snap_num = snap_nums[0]
#        while snapshots[lower_redshift_snap_num].z > z:
#            lower_redshift_snap_num += 1
#        return (lower_redshift_snap_num - 1, lower_redshift_snap_num)]

    def find_neighbouring_snapshots(z: float) -> Tuple[int, int]:
        if z > snapshot_redshifts[snap_nums[0]] or z < snapshot_redshifts[snap_nums[-1]]:
            raise ValueError(f"Redshift {z} outside of redshift range of avalible data.")
        lower_redshift_snap_num = snap_nums[0]
        while lower_redshift_snap_num not in snapshot_redshifts or snapshot_redshifts[lower_redshift_snap_num] > z:
            lower_redshift_snap_num += 1
        return (lower_redshift_snap_num - 1, lower_redshift_snap_num)

    element_weightings = np.array([
        1.0, # X-position
        1.0, # Y-position
        1.0, # Z-position
        1.0, # Mass
        1.0, # Metalicity
        1.0, # Temperature
        1.0, # Density
    ])

    # Create output file
    if not os.path.exists(output_file):
        h5.File(output_file, "w").close()
        complete_files = []
    else:
        with h5.File(output_file, "r") as file:
            complete_files = list(file)

    for f in los_files:
        sightline_file = LineOfSightFileEAGLE(f)

        output_file_group_name = f.rsplit(os.path.sep, maxsplit = 1)[-1]

        completed_sightlines = 0
        if output_file_group_name in complete_files:
            with h5.File(output_file, "r") as file:
                completed_sightlines = len(list(file[output_file_group_name]))
            if completed_sightlines == len(sightline_file):
                continue
        else:
            with h5.File(output_file, "a") as file:
                file.create_group(output_file_group_name)

        Console.print_info(output_file_group_name)

        snap_num_initial, snap_num_final = find_neighbouring_snapshots(sightline_file.z)

        if snap_num_initial not in snapshots:
            snapshots[snap_num_initial] = SnapshotEAGLE(snapshot_filepaths_by_number[snap_num_initial])
            snap_expansion_factors[snap_num_initial] = snapshots[snap_num_initial].a
        if snap_num_final not in snapshots:
            snapshots[snap_num_final] = SnapshotEAGLE(snapshot_filepaths_by_number[snap_num_final])
            snap_expansion_factors[snap_num_final] = snapshots[snap_num_final].a
        selected_snap_nums = [snap_num_initial, snap_num_final]

        # Snipshot data
        raw_ids = { n : snapshots[n].get_IDs(ParticleType.gas) for n in selected_snap_nums}
        matching_order = { selected_snap_nums[i] : ArrayReorder.create(raw_ids[selected_snap_nums[i]], raw_ids[selected_snap_nums[i + 1]]) for i in range(len(selected_snap_nums) - 1) }

        raw_positions    = { n : snapshots[n].get_positions(ParticleType.gas).to("Mpc").value       for n in selected_snap_nums }
        raw_positions_x  = { n : raw_positions[n][:, 0]                                             for n in selected_snap_nums }
        raw_positions_y  = { n : raw_positions[n][:, 1]                                             for n in selected_snap_nums }
        raw_positions_z  = { n : raw_positions[n][:, 2]                                             for n in selected_snap_nums }
        raw_masses       = { n : snapshots[n].get_masses(ParticleType.gas).to("Msun").value         for n in selected_snap_nums }
        raw_metalicities = { n : snapshots[n].get_metalicities(ParticleType.gas).value              for n in selected_snap_nums }
        raw_temperatures = { n : snapshots[n].get_temperatures(ParticleType.gas).to("K").value      for n in selected_snap_nums }
        raw_densities    = { n : snapshots[n].get_densities(ParticleType.gas).to("g/(cm**3)").value for n in selected_snap_nums }

        matched_positions_x  = { n : matching_order[n](raw_positions_x [n]) for n in selected_snap_nums[:-1] }
        matched_positions_y  = { n : matching_order[n](raw_positions_y [n]) for n in selected_snap_nums[:-1] }
        matched_positions_z  = { n : matching_order[n](raw_positions_z [n]) for n in selected_snap_nums[:-1] }
        matched_masses       = { n : matching_order[n](raw_masses      [n]) for n in selected_snap_nums[:-1] }
        matched_metalicities = { n : matching_order[n](raw_metalicities[n]) for n in selected_snap_nums[:-1] }
        matched_temperatures = { n : matching_order[n](raw_temperatures[n]) for n in selected_snap_nums[:-1] }
        matched_densities    = { n : matching_order[n](raw_densities   [n]) for n in selected_snap_nums[:-1] }
        # End Snipshot data

        interp_fraction = (sightline_file.a - snap_expansion_factors[snap_num_initial]) / (snap_expansion_factors[snap_num_final] - snap_expansion_factors[snap_num_initial])
        #TODO: apply box wrapping?
        interpolated_positions_x  = matched_positions_x [snap_num_initial] * (1 - interp_fraction) + (raw_positions_x [snap_num_final] * interp_fraction)
        interpolated_positions_y  = matched_positions_y [snap_num_initial] * (1 - interp_fraction) + (raw_positions_y [snap_num_final] * interp_fraction)
        interpolated_positions_z  = matched_positions_z [snap_num_initial] * (1 - interp_fraction) + (raw_positions_z [snap_num_final] * interp_fraction)
        interpolated_masses       = matched_masses      [snap_num_initial] * (1 - interp_fraction) + (raw_masses      [snap_num_final] * interp_fraction)
        interpolated_metalicities = matched_metalicities[snap_num_initial] * (1 - interp_fraction) + (raw_metalicities[snap_num_final] * interp_fraction)
        interpolated_temperatures = matched_temperatures[snap_num_initial] * (1 - interp_fraction) + (raw_temperatures[snap_num_final] * interp_fraction)
        interpolated_densities    = matched_densities   [snap_num_initial] * (1 - interp_fraction) + (raw_densities   [snap_num_final] * interp_fraction)

        interpolated_vectors = np.array([interpolated_positions_x,
                                        interpolated_positions_y,
                                        interpolated_positions_z,
                                        interpolated_masses,
                                        interpolated_metalicities,
                                        interpolated_temperatures,
                                        interpolated_densities]).T

        # Create array types for sharing the interpolated snapshot vector data and IDs
        n_snap_particles = len(raw_ids[snap_num_final])
        snap_data_double_array_type = multiprocessing.Array(ctypes.c_double, n_snap_particles * 7)
        snap_data_long_array_type = multiprocessing.Array(ctypes.c_long, n_snap_particles)#TODO: check what data type is used for IDs!

#        los_data_double_array_type = multiprocessing.Array(ctypes.c_double, * 7)
#        los_data_long_array_type = multiprocessing.Array(ctypes.c_long, )#TODO: check what data type is used for IDs!

        shared_interpolated_vectors = np.ctypeslib.as_array(snap_data_double_array_type.get_obj())
        shared_interpolated_vectors = shared_interpolated_vectors.reshape(n_snap_particles, 7)
        shared_interpolated_vectors[:, :] = interpolated_vectors[:, :]

        shared_snapshot_ids = np.ctypeslib.as_array(snap_data_long_array_type.get_obj())
        shared_snapshot_ids = shared_snapshot_ids.reshape(n_snap_particles, 7)

        for los_index in range(completed_sightlines, len(sightline_file)):
            print(f"LOS{los_index}", end = " ", flush = True)

            los = sightline_file.get_sightline(los_index)
                
            los_quantity_vectors = np.array([los.positions_comoving.to("Mpc")[:, 0].value,
                                            los.positions_comoving.to("Mpc")[:, 1].value,
                                            los.positions_comoving.to("Mpc")[:, 2].value,
                                            los.masses.to("Msun").value,
                                            los.metallicities.value,
                                            los.temperatures.to("K").value,
                                            los.densities_comoving.to("g/(cm**3)").value]).T

            if n_processes > 1:
                pool = multiprocessing.Pool(processes = n_processes)
                pool.starmap(
                    find_match,
                    zip(
                        range(completed_sightlines, len(sightline_file)),
                        x,
                        y,
                        los_quantity_vectors,
                        ,
                        ,
                        ,
                        ,
                        ,
                        interpolated_vectors,
                        raw_ids[snap_num_final]
                    )
                )

            # Find matches
            selected_ids = np.full(len(los), -1, dtype = int)
            for los_part_index in range(len(los)):
                vector_offsets = (np.abs(interpolated_vectors - los_quantity_vectors[los_part_index]) * element_weightings).sum(axis = 1)
                matched_index = vector_offsets.argmin()
                selected_ids[los_part_index] = raw_ids[snap_num_final][matched_index]
            print("duplicates:", len(los) - np.unique(selected_ids).shape[0], "unset:", (selected_ids == -1).sum(), end = " ", flush = True)
            with h5.File(output_file, "a") as file:
                file[output_file_group_name].create_dataset(f"LOS{los_index}", data = selected_ids)
            print("(written to file)", flush = True)
        print(flush = True)
'''
