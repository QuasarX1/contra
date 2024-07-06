# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .. import VERSION, ParticleType, ArrayReorder, ArrayMapping
from .._L_star import get_L_star_halo_mass_of_z
from ..io import SnapshotBase, SnapshotEAGLE, SnapshotSWIFT, CatalogueBase, CatalogueSUBFIND, CatalogueSOAP, OutputWriter, HeaderDataset, ParticleTypeDataset, SnapshotStatsDataset

import datetime
import os
from typing import Union, List, Tuple, Dict
import asyncio

import numpy as np
from unyt import unyt_quantity, unyt_array
from QuasarCode import Settings, Console
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import ScriptWrapper

'''
def OLD_main():
    ScriptWrapper(
        command = "contra-run",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("3.0.0"),#TODO: double check old code!
        edit_date = datetime.date(2024, 6, 10),
        description = "",
        parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.PositionalParam[str](
                name = "target_snapshot",
                short_name = "t",
                description = "Snapshot file that defines the initial particle distribution."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "snapshots",
                sets_param = "snapshot_directory",
                default_value = ".",
                description = "Where to search for snapshots if an absolute path is not specified\nas part of the target. Targets with relitive paths will be relitive to this location.\nDefaults to \".\" (i.e. the current directory)."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "snapshot-basename",
                default_value = "snap_",
                description = "Prefix to apply to all snapshot file names if the target path contains no path component."
            ),
            ScriptWrapper.Flag(
                name = "snapshot-parallel",
                sets_param = "snapshots_parallel_format",
                description = "Are snapshots written as parrallel outputs?\nI.e., multiple files for a single snapshot."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "snapshot-extension",
                default_value = ".hdf5",
                description = "File extension for snapshot files. Excludes any parallel naming component."
            ),
            ScriptWrapper.OptionalParam[Union[str, None]](
                name = "target-catalogue",
                short_name = "c",
                sets_param = "target_catalogue",
                description = "Catalogue file matching the target file.\nOnly valid when not specifying catalogue information explicitly if the target snapshot is specified by number only."
            ),
            ScriptWrapper.OptionalParam[Union[str, None]](
                name = "catalogue",
                sets_param = "catalogue_directory",
                description = "Where to search for catalogue files."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "catalogue-membership-basename",
                default_value = "SOAP_halo_membership_",
                description = "Prefix to apply to all catalogue membership file names.\nIgnored if the catalogue target path contains no path component."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "catalogue-properties-basename",
                default_value = "SOAP_halo_properties_",
                description = "Prefix to apply to all catalogue properties file names.\nIgnored if the catalogue target path contains no path component."
            ),
            ScriptWrapper.Flag(
                name = "catalogue-parallel",
                sets_param = "catalogue_parallel_format",
                description = "Is catalogue written as parrallel outputs?\nI.e., multiple files for a single snapshot's catalogue."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "catalogue-extension",
                default_value = ".hdf5",
                description = "File extension for catalogue files. Excludes any parallel naming component."
            ),
            ScriptWrapper.Flag(
                name = "EAGLE",
                sets_param = "is_EAGLE",
                description = "Running on EAGLE data."
            ),
            ScriptWrapper.Flag(
                name = "SWIFT",
                sets_param = "is_SWIFT",
                description = "Running on data generated using SWIFT."
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
            ScriptWrapper.OptionalParam[Union[List[str], None]](
                name = "search-snapshots",
                conversion_function = ScriptWrapper.make_list_converter(";"),
                description = "Snapshots that should be searched."
            ),
            ScriptWrapper.Flag(
                name = "ignore-target-snapshot",
                inverted = True,
                sets_param = "require_include_target_snapshot",
                description = "Do not add the target snapshot to the search list if manually ommitted."
            )
        )
    ).run(__main)
'''

def main():
    ScriptWrapper(
        command = "contra-run",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("3.0.0"),#TODO: double check old code!
        edit_date = datetime.date(2024, 6, 10),
        description = "",
        parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.PositionalParam[str](
                name = "target_snapshot",
                short_name = "t",
                description = "Identifier of the snapshot file that defines the initial particle distribution.\nFor SWIFT data, this is usually a four digit number: e.g. \"0015\".\nFor EAGLE data, this is a three digit number: e.g. \"015\"."
            ),
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
                conflicts = ["snapshot-basename"]
            ),
            ScriptWrapper.Flag(
                name = "SWIFT",
                sets_param = "is_SWIFT",
                description = "Running on data generated using SWIFT."
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
            ScriptWrapper.OptionalParam[Union[List[str], None]](
                name = "search-snapshots",
                conversion_function = ScriptWrapper.make_list_converter(";"),
                description = "Identifiers of snapshots that should be searched (in search order)."
            ),
            ScriptWrapper.Flag(
                name = "ignore-target-snapshot",
                inverted = True,
                sets_param = "require_include_target_snapshot",
                description = "Do not add the target snapshot to the search list if manually ommitted."
            ),
            ScriptWrapper.Flag(
                name = "skip-statistics",
                inverted = True,
                sets_param = "do_stats",
                description = "Do not calculate snapshot statistics.\nSignificant performance increas but minimal data output."
            )
        )
    ).run_with_async(__main)

async def __main(
            target_snapshot: str,
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
            search_snapshots: Union[List[str], None],
            require_include_target_snapshot: bool,
            do_stats: bool
          ) -> None:

    # Store the current date at the start in case the file writing occours after midnight.
    start_date = datetime.date.today()

    if not (is_EAGLE or is_SWIFT):
        Console.print_error("Must specify either EAGLE or SWIFT simulation type.")
        return
    else:
        if is_EAGLE:
            Console.print_verbose_info("Snapshot type: EAGLE")
        elif is_SWIFT:
            Console.print_verbose_info("Snapshot type: SWIFT")

    # Identify snapshot and catalogue types
    new_snapshot_type = SnapshotEAGLE if is_EAGLE else SnapshotSWIFT
    new_catalogue_type = CatalogueSUBFIND if is_EAGLE else CatalogueSOAP

    Console.print_verbose_info("Particle Types:")

    if not (do_gas or do_stars or do_black_holes or do_dark_matter):
        Console.print_verbose_warning("    No particle type(s) specified. Enabling all particle types.")
        do_gas = do_stars = do_black_holes = do_dark_matter = True
    particle_types = []
    if do_gas:
        particle_types.append(ParticleType.gas)
        Console.print_verbose_info("    Tracking gas particles.")
    if do_stars:
        particle_types.append(ParticleType.star)
        Console.print_verbose_info("    Tracking star particles.")
    if do_black_holes:
        particle_types.append(ParticleType.black_hole)
        Console.print_verbose_info("    Tracking black hole particles.")
    if do_dark_matter:
        particle_types.append(ParticleType.dark_matter)
        Console.print_verbose_info("    Tracking dark matter particles.")

    Console.print_info("Identifying snapshot and catalogue files.")

    # Ensure that the path is an absolute path
    snapshot_directory = os.path.abspath(snapshot_directory)

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
#            snap_stats = SnapshotStatsDataset()
#
#            # Simple calls
#            snap_stats.snapshot_filepath = snap.filepath
#            snap_stats.catalogue_membership_filepath = cat.membership_filepath
#            snap_stats.catalogue_properties_filepath = cat.properties_filepath
#            snap_stats.redshift = cat.z
#            snap_stats.N_particles = sum([snap.number_of_particles(p) for p in ParticleType.get_all()])
#            for part_type in ParticleType.get_all():
#                setattr(snap_stats, f"N_particles_{part_type.name.replace(' ', '_')}", snap.number_of_particles(part_type))
#            snap_stats.N_haloes = len(cat)
#            snap_stats.N_halo_children = cat.number_of_children
#            snap_stats.N_halo_decendants = cat.number_of_decendants
#
#            # Async operation functions
#
#            async def calc__particle_total_volume():
#                snap_stats.particle_total_volume = sum(
#                    [
#                        (smoothing_lengths.to("Mpc")**3).sum()
#                        for smoothing_lengths
#                        in await asyncio.gather(*[snap.get_smoothing_lengths_async(p) for p in ParticleType.get_all()])
#                    ],
#                    start = unyt_quantity(0.0, units = "Mpc**3")
#                ) * (np.pi * 4/3)
#
#            async def calc__N_haloes_top_level():
#                snap_stats.N_haloes_top_level = int((await cat.get_halo_parent_IDs_async() == -1).sum())
#
#            async def calc__N_halo_particles_of_type(part_type: ParticleType):
#                return len(await cat.get_particle_IDs_async(part_type))
#            async def calc__N_halo_particles():
#                (
#                    snap_stats.N_halo_particles_gas,
#                    snap_stats.N_halo_particles_star,
#                    snap_stats.N_halo_particles_black_hole,
#                    snap_stats.N_halo_particles_dark_matter,
#                ) = await asyncio.gather(
#                    calc__N_halo_particles_of_type(ParticleType.gas),
#                    calc__N_halo_particles_of_type(ParticleType.star),
#                    calc__N_halo_particles_of_type(ParticleType.black_hole),
#                    calc__N_halo_particles_of_type(ParticleType.dark_matter)
#                )
#
#            # Run all async functions
#            await asyncio.gather(
#                calc__particle_total_volume(),
#                calc__N_haloes_top_level(),
#                calc__N_halo_particles()
#            )
#
#    #        #TODO: NOT PER HALO!?!?
#    #        snap_stats.N_halo_particles = np.zeros(snap_stats.N_haloes, dtype = int)
#    #        snap_stats.halo_particle_total_volume = np.zeros(snap_stats.N_haloes, dtype = float)
#    #        for p in ParticleType.get_all():
#    #            halo_indexes_in_snap_order = cat.get_halo_IDs_by_snapshot_particle(p) - cat.get_halo_IDs()[0]
#    #            halo_particle_mask = halo_indexes_in_snap_order != -1 - cat.get_halo_IDs()[0]#TODO: get a better way of doing this offset
#    #            snap_stats.N_halo_particles = snap_stats.N_halo_particles + np.bincount(halo_indexes_in_snap_order[halo_particle_mask])
#    #            snap_stats.halo_particle_total_volume = snap_stats.halo_particle_total_volume + np.bincount(halo_indexes_in_snap_order[halo_particle_mask], weights = snap.get_smoothing_lengths(p)[halo_particle_mask]**3) * (np.pi * 4/3)
#    #        Console.print_debug(4)#TODO:REMOVE
#
#            snap_stats.N_halo_particles = sum([len(cat.get_particle_IDs(p)) for p in ParticleType.get_all()])
#            snap_stats.halo_particle_total_volume = sum([(ArrayReorder.create(snap.get_IDs(p), cat.get_particle_IDs(p))(snap.get_smoothing_lengths(p).to("Mpc"))**3).sum() for p in ParticleType.get_all()], start = unyt_quantity(0.0, units = "Mpc**3")) * (np.pi * 4/3)

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









'''
def OLD__main(
            target_snapshot_filepath: str,
            is_EAGLE: bool, is_SWIFT: bool,
            do_gas: bool, do_stars: bool, do_black_holes: bool, do_dark_matter: bool,
            output_filepath: str, allow_overwrite: bool,
            search_snapshot_filepaths: Union[List[str], None], require_include_target_snapshot: bool
          ) -> None:
    
    # Store the current date at the start in case the file writing occours after midnight.
    start_date = datetime.date.today()

    # Ensure that the path is an absolute path
    target_snapshot_filepath = os.path.abspath(target_snapshot_filepath)

    if not os.path.exists(target_snapshot_filepath):
        Console.print_error(f"Unable to locate target snapshot: {target_snapshot_filepath}")
        return
    else:
        Console.print_verbose_info(f"Target snapshot: {target_snapshot_filepath}")

    if not (is_EAGLE and is_SWIFT):
        Console.print_error("Must specify either EAGLE or SWIFT simulation type.")
        return
    else:
        if is_EAGLE:
            Console.print_verbose_info("Snapshot type: EAGLE")
        elif is_SWIFT:
            Console.print_verbose_info("Snapshot type: SWIFT")
    
    if not (do_gas or do_stars or do_black_holes or do_dark_matter):
        Console.print_verbose_warning("No particle type(s) specified. Enabling all particle types.")
        do_gas = do_stars = do_black_holes = do_dark_matter = True
    particle_types = []
    if do_gas:
        particle_types.append(ParticleType.gas)
        Console.print_verbose_info("Tracking gas particles")
    if do_stars:
        particle_types.append(ParticleType.star)
        Console.print_verbose_info("Tracking star particles")
    if do_black_holes:
        particle_types.append(ParticleType.black_hole)
        Console.print_verbose_info("Tracking black hole particles")
    if do_dark_matter:
        particle_types.append(ParticleType.dark_matter)
        Console.print_verbose_info("Tracking dark matter particles")

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

    if search_snapshot_filepaths is None:
        pass#TODO: search for files at highter redshift

    # Ensure that the paths are absolute paths
    search_snapshot_filepaths = [os.path.abspath(f) for f in search_snapshot_filepaths]

    # Ensure that the target snap is included if required
    if require_include_target_snapshot and target_snapshot_filepath not in search_snapshot_filepaths:
        search_snapshot_filepaths.insert(0, target_snapshot_filepath)
    


    # Identify snapshot and catalogue types
    new_snapshot = SnapshotEAGLE if is_EAGLE else SnapshotSWIFT
    new_catalogue = CatalogueSUBFIND if is_EAGLE else CatalogueSOAP

    # Load target snapshot
    target_snap = (SnapshotEAGLE if is_EAGLE else SnapshotSWIFT)(target_snapshot_filepath)

    snapshots: List[SnapshotBase] = []
    catalogues: List[CatalogueBase] = []
    for snapshot_filepath, catalogue_filepath in zip(search_snapshot_filepaths, search_catalogue_filepaths):
        snapshots.append(new_snapshot(snapshot_filepath) if snapshot_filepath != target_snapshot_filepath else target_snap)
        catalogues.append(new_catalogue(catalogue_filepath, snapshots[-1]))
    N_snapshots = len(snapshots)

    # Ensure search values are in increasing redshift (i.e. backwards in time)
    snapshots.sort(key = lambda v: v.z)
    catalogues.sort(key = lambda v: v.z)

    # Get L_* values for each snapshot's redshift
    L_star_mass_func = get_L_star_mass_of_z()
    L_star_mass_by_snapshot = [L_star_mass_func(cat.z) for cat in catalogues]



    # Create output file
    output_file = OutputWriter(output_filepath)

    with output_file:
        output_file.write_header(HeaderDataset(
            version = VersionInfomation.from_string(VERSION),
            date = start_date,
            target_file = target_snap.filepath,
            redshift = target_snap.z,
            searched_files = N_snapshots,
            output_file = output_filepath,
            has_gas = do_gas,
            has_stars = do_stars,
            has_black_holes = do_black_holes,
            has_dark_matter = do_dark_matter
        ))



    # Perform backwards search
    #TODO: do star and BH checks need to check gas part table for particle ansestor? how does this work in SWIFT and EAGLE???

    target_ids                 = { part_type: target_snap.get_IDs(part_type)                             for part_type in particle_types }
    missing_particles          = { part_type: np.full_like(target_ids[part_type], True)                  for part_type in particle_types }
    last_redshift              = { part_type: np.full_like(target_ids[part_type], np.nan, dtype = float) for part_type in particle_types }
    last_halo_id               = { part_type: np.full_like(target_ids[part_type], -1,     dtype = int)   for part_type in particle_types }
    last_halo_mass             = { part_type: np.full_like(target_ids[part_type], np.nan, dtype = float) for part_type in particle_types }
    last_halo_mass_L_star_frac = { part_type: np.full_like(target_ids[part_type], np.nan, dtype = float) for part_type in particle_types }
    pre_ejection_coords        = { part_type: np.full((target_ids[part_type].shape[0], 3), np.nan, dtype = float) for part_type in particle_types }

    for snap_index, (snap, cat, L_star_mass) in enumerate(zip(snapshots, catalogues, L_star_mass_by_snapshot)):
        Console.print_info(f"Doing snapshot {snap_index + 1}/{N_snapshots} (z={snap.z})")

        snap_stats = SnapshotStatsDataset()
        snap_stats.snapshot_filepath = snap.filepath
        snap_stats.catalogue_filepath = cat.filepath
        snap_stats.redshift = cat.z
#        snap_stats.N_particles = sum([snap.get_masses(p).shape[0] for p in ParticleType.get_all()])
        snap_stats.N_particles = sum([snap.number_of_particles(p) for p in ParticleType.get_all()])
        for part_type in ParticleType.get_all():
            setattr(snap_stats, f"N_particles_{part_type.name.replace(' ', '_')}", snap.number_of_particles(part_type))
        snap_stats.N_haloes = len(cat)
        snap_stats.particle_total_volume = (4 / 3) * np.pi * sum([(snap.get_smoothing_lengths(p).to("Mpc")**3).sum() for p in ParticleType.get_all()], start = unyt_quantity(0.0, units = "Mpc"))

        snap_stats.N_haloes_top_level = None#TODO:
        snap_stats.N_halo_particles = None#TODO:
        snap_stats.halo_particle_total_volume = None#TODO:
        snap_stats.N_halo_children = None#TODO:
        snap_stats.N_halo_top_level_children = None#TODO:

        snap_stats.N_particles_matched = 0
        snap_stats.particles_matched_total_volume = unyt_quantity(0.0, units = "Mpc")

        for part_type in particle_types:
            Console.print_info(f"    Doing {part_type.name} particles...", end = "")

            redshift = cat.z
            halo_masses = cat.get_halo_masses(part_type)
            catalogue_particle_ids = cat.get_particle_IDs(part_type)

            #TODO: parent haloes only???

            # Generate allignment function for catalogue to new matches in snapshot
            order = ArrayReorder.create(catalogue_particle_ids, target_ids[part_type], target_order_filter = missing_particles[part_type])

            # Store values for new matches
            last_redshift[part_type][order.target_filter] = redshift
            order(source_data = cat.get_halo_IDs(part_type),                                         output_array = last_halo_id[part_type])
            order(source_data = halo_masses,                                                         output_array = last_halo_mass[part_type])
            order(source_data = halo_masses / L_star_mass(redshift),                                 output_array = last_halo_mass_L_star_frac[part_type])
            order(source_data = cat.snapshot_orderby_halo(part_type, snap.get_positions(part_type)), output_array = pre_ejection_coords[part_type])

            # Update stats
            setattr(snap_stats, f"N_halo_particles_{part_type.name.replace(' ', '_')}", len(catalogue_particle_ids))
            n_matched = order.target_filter.sum()
            setattr(snap_stats, f"N_particles_matched_{part_type.name.replace(' ', '_')}", n_matched)
            snap_stats.N_particles_matched = snap_stats.N_particles_matched + n_matched
            matched_volume = ((4 / 3) * np.pi * ((snap.get_smoothing_lengths(part_type)[np.isin(snap.get_IDs(part_type), cat.get_particle_IDs(part_type))])**3).sum())
            setattr(snap_stats, f"particles_matched_total_volume_{part_type.name.replace(' ', '_')}", matched_volume)
            snap_stats.particles_matched_total_volume = snap_stats.particles_matched_total_volume + matched_volume

            # Update the tracking filter
            missing_particles[part_type][order.target_filter] = False

            print("done")

        with output_file:
            output_file.write_snapshot_stats_dataset(snap_stats)

        if Settings.verbose or Settings.debug:
            (Console.print_info if Settings.verbose else Console.print_debug)(f"Stats for snapshot {snap_index + 1}:\n{snap_stats}")

    
    with output_file:
        for part_type in particle_types:
            output_file.write_particle_type_dataset(ParticleTypeDataset(
                particle_type = part_type,
                redshifts = last_redshift[part_type],
                halo_ids = last_halo_id[part_type],
                halo_masses = last_halo_mass[part_type],
                halo_masses_scaled = last_halo_mass_L_star_frac[part_type],
                positions_pre_ejection = pre_ejection_coords[part_type]
            ))
'''
