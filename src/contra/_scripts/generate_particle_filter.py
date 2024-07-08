# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .. import VERSION, ParticleType, ArrayReorder, ArrayMapping
from .._L_star import get_L_star_halo_mass_of_z
from ..io import SnapshotBase, SnapshotEAGLE, SnapshotSWIFT, \
                OutputReader, HeaderDataset, ParticleTypeDataset, \
                SnapshotStatsDataset, \
                ParticleFilterFile, ParticleFilter, LOSFilter

import datetime
import os
from typing import Union, List, Tuple, Dict
import asyncio

import numpy as np
from unyt import unyt_quantity, unyt_array
from QuasarCode import Settings, Console
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import ScriptWrapper

#TODO: los files for EAGLE have no particle IDs
# try identifying the IDs using intermediate_ids = np.array([int.from_bytes(coords[i, :].tobytes(), sys.byteorder) for i in range(coords.shape[0])])

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
            ScriptWrapper.OptionalParam(
                name = "min-halo-mass",
                description = "Minimum absolute halo mass."
            ),
            ScriptWrapper.OptionalParam(
                name = "max-halo-mass",
                description = "Maximum absolute halo mass."
            ),
            ScriptWrapper.OptionalParam(
                name = "min-scaled-halo-mass",
                description = "Minimum halo mass as a fraction of M*(z)."
            ),
            ScriptWrapper.OptionalParam(
                name = "max-scaled-halo-mass",
                description = "Maximum halo mass as a fraction of M*(z)."
            ),
            ScriptWrapper.OptionalParam(
                name = "min-ejection-distance",
                description = "Minimum distance traveled by particle since last found in a halo."
            ),
            ScriptWrapper.OptionalParam(
                name = "max-ejection-distance",
                description = "Maximum distance traveled by particle since last found in a halo."
            )
        )
    ).run_with_async(__main)

async def __main(
            data: str,
            snapshot_directory: str,
            los_directory: str | None,
            is_EAGLE: bool,
            is_SWIFT: bool,
            do_gas: bool,
            do_stars: bool,
            do_black_holes: bool,
            do_dark_matter: bool,
            output_filepath: str,
            allow_overwrite: bool,
            search_snapshots: Union[List[str], None],
            require_include_target_snapshot: bool,
            do_stats: bool,
            min_halo_mass,
            max_halo_mass,
            min_scaled_halo_mass,
            max_scaled_halo_mass,
            min_ejection_distance,
            max_ejection_distance
          ) -> None:

    # Store the current date at the start in case the file writing occours after midnight.
    start_date = datetime.date.today()

    if not (min_halo_mass or max_halo_mass or min_scaled_halo_mass or max_scaled_halo_mass or min_ejection_distance or max_ejection_distance):
        Console.print_error("Must specify at leas one filter argument.")
        Console.print_info("Terminating...")
        return

    if not (is_EAGLE or is_SWIFT):
        Console.print_error("Must specify either EAGLE or SWIFT simulation type.")
        Console.print_info("Terminating...")
        return
    else:
        if is_EAGLE:
            Console.print_verbose_info("Snapshot type: EAGLE")
        elif is_SWIFT:
            Console.print_verbose_info("Snapshot type: SWIFT")

    # Identify snapshot and catalogue types
    new_snapshot_type = SnapshotEAGLE if is_EAGLE else SnapshotSWIFT

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
    contra_header = contra_reader.read_header()
#    contra_header.N_searched_snapshots
    contra_header.has_statistics

    # Ensure that the path is an absolute path
    snapshot_directory = os.path.abspath(snapshot_directory)

    # Find the correct path for the target snapshot
    snap_dir, file = contra_header.target_snapshot.split(os.path.sep, maxsplit = 1 if is_SWIFT else 2)
    if snapshot_directory is None:
        snapshot_directory = snap_dir
    target_snap = os.path.join(snapshot_directory, file)

    filters = ParticleFilterFile(
        version = ,
        date = ,
        output_file = ,
        contra_file = ,
        snapshots_directory = ,
        has_gas = ,
        has_stars = ,
        has_black_holes = ,
        has_dark_matter = ,
        data = 
    )
    if los_directory is not None:
        filters.los_directory = los_directory

    #TODO: add some form of description of what limits were applied!

    filters.save(output_filepath)






















#    snapshot_files: Union[Dict[str, Union[str, Tuple[str, ...]]], None] = None
#    if is_SWIFT:
#        snapshot_files = SnapshotSWIFT.generate_filepaths_from_partial_info(snapshot_directory, snapshot_basename)
#    else: # is_EAGLE
#        raise NotImplementedError("TODO: add EAGLE support!")#TODO:
#        snapshot_files = SnapshotEAGLE.generate_filepaths_from_partial_info(snapshot_directory, snapshot_basename)
#    snapshot_files: Dict[str, Union[str, Tuple[str, ...]]]
    snapshot_files = new_snapshot_type.generate_filepaths_from_partial_info(snapshot_directory, snapshot_basename)

    # Ensure requested snapshots are valid
    if target_snapshot not in snapshot_files:
        Console.print_error(f"Target snapshot identifier \"{target_snapshot}\" not found. Valid values are:\n    " + "\n    ".join(snapshot_files.keys()))
    if search_snapshots is not None:
        for test_snap_key in search_snapshots:
            if test_snap_key not in snapshot_files:
                Console.print_error(f"Search snapshot identifier \"{test_snap_key}\" not found. Valid values are:\n    " + "\n    ".join(snapshot_files.keys()))

    # Identify search target snapshots
    search_snapshot_file_order: List[str]
    if search_snapshots is None:
#        if is_SWIFT:
#            search_snapshot_file_order = SnapshotSWIFT.get_snapshot_order(snapshot_files.keys(), reverse = True)
#        else: # is_EAGLE
#            raise NotImplementedError("TODO: add EAGLE support!")#TODO:
        search_snapshot_file_order = new_snapshot_type.get_snapshot_order(snapshot_files.keys(), reverse = True)
        search_snapshot_file_order = search_snapshot_file_order[search_snapshot_file_order.index(target_snapshot) :]
    else:
        search_snapshot_file_order = []
        if target_snapshot not in search_snapshots and require_include_target_snapshot:
            search_snapshot_file_order.append(target_snapshot)
        search_snapshot_file_order.extend(search_snapshots)

    if isinstance(snapshot_files[search_snapshot_file_order[0]], str):
        Console.print_verbose_info("Got the following snapshot files:\n    " + "\n    ".join([snapshot_files[key] for key in search_snapshot_file_order]))
    else:
        Console.print_verbose_info("Got the following snapshot files:\n    " + "\n    ".join([(snapshot_files[key][min(snapshot_files[key].keys())] + f" ({min(snapshot_files[key].keys())} -> {max(snapshot_files[key].keys())})") for key in search_snapshot_file_order]))

    # Get filepath info for catalogue files

    # Ensure that the path is an absolute path
    catalogue_directory = os.path.abspath(catalogue_directory) if catalogue_directory is not None else snapshot_directory

#    catalogue_files: Union[Dict[str, Tuple[Union[str, Tuple[str, ...]], Union[str, Tuple[str, ...]]]], None] = None
#    if is_SWIFT:
#        catalogue_files = CatalogueSOAP.generate_filepaths_from_partial_info(catalogue_directory, catalogue_membership_basename, catalogue_properties_basename, snapshot_number_strings = search_snapshot_file_order)
#    else: # is_EAGLE
#        raise NotImplementedError("TODO: add EAGLE support!")#TODO:
#        catalogue_files = CatalogueSUBFIND.generate_filepaths_from_partial_info(catalogue_directory, catalogue_membership_basename, catalogue_properties_basename, snapshot_number_strings = search_snapshot_file_order)
    catalogue_files = new_catalogue_type.generate_filepaths_from_partial_info(catalogue_directory, catalogue_membership_basename, catalogue_properties_basename, snapshot_number_strings = search_snapshot_file_order)

    if isinstance(catalogue_files[search_snapshot_file_order[0]][0], str):
        Console.print_verbose_info("Got the following catalogue membership files:\n    " + "\n    ".join([catalogue_files[key][0] for key in search_snapshot_file_order]))
    else:
        Console.print_verbose_info("Got the following catalogue membership files:\n    " + "\n    ".join([(catalogue_files[key][0][0] + f" ({0} -> {len(catalogue_files[key][0]) - 1})") for key in search_snapshot_file_order]))

    if isinstance(catalogue_files[search_snapshot_file_order[0]][1], str):
        Console.print_verbose_info("Got the following catalogue properties files:\n    " + "\n    ".join([catalogue_files[key][1] for key in search_snapshot_file_order]))
    else:
        Console.print_verbose_info("Got the following catalogue properties files:\n    " + "\n    ".join([(catalogue_files[key][1][0] + f" ({0} -> {len(catalogue_files[key][1]) - 1})") for key in search_snapshot_file_order]))


    # Ensure that the path is an absolute path
    output_filepath = os.path.abspath(output_filepath)

    if os.path.exists(output_filepath):
        if not allow_overwrite:
            Console.print_error(f"Output file already exists. Either remove it first or explicitley enable overwrite.")
            return
        else:
            Console.print_warning("Pre-existing output file will be overwritten.")
    elif False:#TODO: check for valid file location (to prevent an error at the last minute!)
        pass



    Console.print_info("Loading snapshot and catalogue files...", end = "\n" if (is_EAGLE and Settings.debug) else "")
    timer_start = datetime.datetime.now()

    # Load target snapshot and catalogue
    target_snap = new_snapshot_type(snapshot_files[target_snapshot] if isinstance(snapshot_files[target_snapshot], str) else snapshot_files[target_snapshot][min(snapshot_files[target_snapshot].keys())])
    target_cat = new_catalogue_type(*catalogue_files[target_snapshot], target_snap)

    snapshots: List[SnapshotBase] = []
    catalogues: List[CatalogueBase] = []
    for snap_key in search_snapshot_file_order:
        snapshots.append(new_snapshot_type((snapshot_files[snap_key]) if isinstance(snapshot_files[snap_key], str) else snapshot_files[snap_key][min(snapshot_files[snap_key].keys())]) if snap_key != target_snapshot else target_snap)
        catalogues.append(new_catalogue_type(*catalogue_files[snap_key], snapshots[-1]) if snap_key != target_snapshot else target_cat)
#        if is_SWIFT:
#            catalogues.append(CatalogueSOAP(*catalogue_files[snap_key], snapshots[-1]))
#        else: # is_EAGLE
#            pass
    N_snapshots = len(snapshots)

    print(f"done (took {int((datetime.datetime.now() - timer_start).total_seconds())} s)")

#    # Ensure search values are in increasing redshift (i.e. backwards in time)
#    snapshots.sort(key = lambda v: v.z)
#    catalogues.sort(key = lambda v: v.z)

    # Get L_* values for each snapshot's redshift
    L_star_mass_func = get_L_star_halo_mass_of_z()
    L_star_mass_by_snapshot: List[unyt_quantity] = [L_star_mass_func(cat.z) for cat in catalogues]



    Console.print_info("Creating file.")

    # Create output file
    output_file = OutputWriter(output_filepath)

    with output_file:
        output_file.write_header(HeaderDataset(
            version = VersionInfomation.from_string(VERSION),
            date = start_date,
            target_snapshot = target_snap.filepath,
            target_catalogue_membership_file = target_cat.membership_filepath,
            target_catalogue_properties_file = target_cat.properties_filepath,
            simulation_type = "SWIFT" if is_SWIFT else "EAGLE",
            redshift = target_snap.z,
            N_searched_snapshots = N_snapshots,
            output_file = output_filepath,
            has_gas = do_gas,
            has_stars = do_stars,
            has_black_holes = do_black_holes,
            has_dark_matter = do_dark_matter,
            has_statistics = do_stats
        ))




    Console.print_info("Begining reverse search.")

    # Perform backwards search
    #TODO: do star and BH checks need to check gas part table for particle ansestor? how does this work in SWIFT and EAGLE???

    target_ids                 = { part_type: target_snap.get_IDs(part_type)                                                                  for part_type in particle_types }
    missing_particles          = { part_type:            np.full_like(target_ids[part_type],          True,   dtype = bool )                  for part_type in particle_types }
    last_redshift              = { part_type:            np.full_like(target_ids[part_type],          np.nan, dtype = float)                  for part_type in particle_types }
    last_halo_id               = { part_type:            np.full_like(target_ids[part_type],          -1,     dtype = int  )                  for part_type in particle_types }
    last_halo_mass             = { part_type: unyt_array(np.full_like(target_ids[part_type],          np.nan, dtype = float), units = "Msun") for part_type in particle_types }
    last_halo_mass_L_star_frac = { part_type: unyt_array(np.full_like(target_ids[part_type],          np.nan, dtype = float), units = None  ) for part_type in particle_types }
    pre_ejection_coords        = { part_type: unyt_array(np.full((target_ids[part_type].shape[0], 3), np.nan, dtype = float), units = "Mpc" ) for part_type in particle_types }

    for snap_index, (snap, cat, L_star_mass) in enumerate(zip(snapshots, catalogues, L_star_mass_by_snapshot)):
        Console.print_info(f"Doing snapshot {snap_index + 1}/{N_snapshots} (z={snap.z})")

        if do_stats:

            Console.print_verbose_info("    Calculating initial statistics")

            # Compute initial statistics for the snapshot
            snap_stats = SnapshotStatsDataset.initialise_partial(snap, cat)

            # These are calculated during the search
            snap_stats.N_particles_matched = 0
            snap_stats.particles_matched_total_volume = unyt_quantity(0.0, units = "Mpc**3")

        # Load halo properties here as some parent haloes might not have target part type members!
        redshift = cat.z
        all_halo_ids = cat.get_halo_IDs()
        all_halo_masses = cat.get_halo_masses()

        # Perform search for each particle type
        for part_type in particle_types:
            Console.print_info(f"    Doing {part_type.name} particles...", end = "")

#            redshift = cat.z
            selected_halo_ids = cat.get_halo_IDs(part_type)
            selected_top_level_halo_ids = cat.get_halo_top_level_parent_IDs(part_type)
            selected_catalogue_particle_ids = cat.get_particle_IDs(part_type)

            # Use only the top most halo mass
            mapping_halo_to_selected_top_level = ArrayMapping(all_halo_ids, selected_top_level_halo_ids)
            selected_top_level_halo_masses = mapping_halo_to_selected_top_level(all_halo_masses)

            # Generate allignment function for catalogue to new matches in snapshot
            order_cat_particles_to_snap_targets = ArrayReorder.create(selected_catalogue_particle_ids, target_ids[part_type], target_order_filter = missing_particles[part_type])
            order_snap_to_targets = ArrayReorder.create(snap.get_IDs(part_type), target_ids[part_type], target_order_filter = missing_particles[part_type])
            mapping_halo_to_snap_targets = ArrayMapping(
                selected_halo_ids,
                order_snap_to_targets(
                    cat.get_halo_IDs_by_snapshot_particle(part_type),
                    default_value = -1
                )
            )

            # Store values for new matches
            last_redshift[part_type][order_snap_to_targets.target_filter] = redshift
            mapping_halo_to_snap_targets(selected_halo_ids,                            output_array = last_halo_id[part_type]              )
            mapping_halo_to_snap_targets(selected_top_level_halo_masses,               output_array = last_halo_mass[part_type]            )
            mapping_halo_to_snap_targets(selected_top_level_halo_masses / L_star_mass, output_array = last_halo_mass_L_star_frac[part_type])
            order_snap_to_targets(       snap.get_positions(part_type),                output_array = pre_ejection_coords[part_type]       )

            if do_stats:
                # Update stats
                setattr(snap_stats, f"N_halo_particles_{part_type.name.replace(' ', '_')}", len(selected_catalogue_particle_ids))
                n_matched = int(order_cat_particles_to_snap_targets.target_filter.sum())
                setattr(snap_stats, f"N_particles_matched_{part_type.name.replace(' ', '_')}", n_matched)
                snap_stats.N_particles_matched = snap_stats.N_particles_matched + n_matched
                matched_volume = ((snap.get_smoothing_lengths(part_type)[np.isin(snap.get_IDs(part_type), cat.get_particle_IDs(part_type))])**3).sum() * (np.pi * (4/3))
                setattr(snap_stats, f"particles_matched_total_volume_{part_type.name.replace(' ', '_')}", unyt_quantity(matched_volume.to("Mpc**3").value, units = "Mpc**3"))
                snap_stats.particles_matched_total_volume = snap_stats.particles_matched_total_volume + matched_volume

            # Update the tracking filter
            missing_particles[part_type][order_cat_particles_to_snap_targets.target_filter] = False

            print("done")

        if do_stats:

            Console.print_verbose_info("    Writing statistics to file")

            with output_file:
                output_file.write_snapshot_stats_dataset(snap_index, snap_stats)

            if Settings.verbose or Settings.debug:
                (Console.print_info if Settings.verbose else Console.print_debug)(f"    Stats for snapshot {snap_index + 1}:\n        " + str(snap_stats).replace("\n", "\n        "))

    Console.print_info("Reverse search complete.")
    Console.print_info("Writing final output.")

    
    with output_file:
        for part_type in particle_types:
            output_file.write_particle_type_dataset(ParticleTypeDataset(
                particle_type = part_type,
                redshifts = last_redshift[part_type],
                halo_ids = last_halo_id[part_type],
                halo_masses = last_halo_mass[part_type].value,
                halo_masses_scaled = last_halo_mass_L_star_frac[part_type].value,
                positions_pre_ejection = pre_ejection_coords[part_type].value
            ))

    Console.print_info("Finished. Stopping...")
