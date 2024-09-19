# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
import multiprocessing.managers
from .. import VERSION, ParticleType, SharedArray, SharedArray_TransmissionData
from ..io import SnapshotEAGLE, CatalogueSUBFIND, FileTreeScraper_EAGLE, SimulationSnapshotFiles_EAGLE, SimulationSnipshotFiles_EAGLE, SimulationSnapshotCatalogueFiles_EAGLE, SimulationSnipshotCatalogueFiles_EAGLE

import datetime
import os
from collections.abc import Collection, Sequence, Mapping, Iterable
#from typing import Union, List, Tuple, Dict
import asyncio
import multiprocessing

import numpy as np
from unyt import unyt_quantity, unyt_array
import h5py as h5
from QuasarCode import Settings, Console
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import ScriptWrapper



#TODO: check pluralisation! (and spelling)
SNAPSHOT_FIELDS_TO_CHECK = ("ParticleIDs", "Coordinates", "Mass", "SmoothingLength")
SNAPSHOT_FIELDS_TO_CHECK__DM = ("ParticleIDs", "Coordinates")
SNIPSHOT_FIELDS_TO_CHECK = ("ParticleIDs", "Coordinates", "Mass")

CATALOGUE_PROPERTIES_GROUPS_TO_CHECK = ("FOF", "IDs", "Subhalo")
CATALOGUE_PROPERTIES_FIELDS_TO_CHECK = {
    "FOF"     : ("FirstSubhaloID", "GroupCentreOfPotential", "GroupMass", "GroupOffset", "Group_M_Crit200", "Group_M_Crit500", "NumOfSubhalos"),
#    "IDs"     : ("ParticleID", "Particle_Binding_Energy"),
    "Subhalo" : ("CentreOfMass", "CentreOfPotential", "GroupNumber", "IDMostBound", "Mass", "MassType", "SubGroupNumber", "SubLength", "SubLengthType", "SubOffset", "Velocity")
}
SNAPSHOT_CATALOGUE_MEMBERSHIP_FIELDS_TO_CHECK = ("ParticleIDs", "Coordinates", "Mass", "SmoothingLength")
SNIPSHOT_CATALOGUE_MEMBERSHIP_FIELDS_TO_CHECK = ("ParticleIDs", "Coordinates", "Mass")
CATALOGUE_MEMBERSHIP_FIELDS_TO_CHECK__DM = ("ParticleIDs", "Coordinates")



def main():
    ScriptWrapper(
        command = "contra-check-eagle",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 8, 28),
        description = "Check the integrity of EAGLE data.\nChecks for missing files, files that exist but have no data and files missing key fields.",
        parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.OptionalParam[str](
                name = "data",
                short_name = "i",
                sets_param = "data_directory",
                default_value = ".",
                description = "Root directoy of an EAGLE dataset.\nDefaults to the cirrent working directory."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "output-file",
                short_name = "o",
                sets_param = "output_filepath",
                default_value = "eagle-integrity-errors.txt",
                description = "File to write any issues into."
            ),
            ScriptWrapper.OptionalParam[int](
                name = "processes",
                short_name = "n",
                sets_param = "number_of_processes",
                conversion_function = int,
                default_value = 1,
                description = "Number of processes to use."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "specific-snapshots",
                sets_param = "selected_snapshot_numbers",
                conversion_function = ScriptWrapper.make_list_converter(";"),
                default_value = None,
                description = "Check only the snapshots with these numbers.\nSpecify the snapshot numbers only as a semicolon (;) seperated list.\nSpecify as a single - to indicate all the remaining snapshots. This can only be at the end of the list."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "specific-snipshots",
                sets_param = "selected_snipshot_numbers",
                conversion_function = ScriptWrapper.make_list_converter(";"),
                default_value = None,
                description = "Check only the snipshots with these numbers.\nSpecify the snipshot numbers only as a semicolon (;) seperated list.\nSpecify as a single - to indicate all the remaining snipshots. This can only be at the end of the list."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "specific-catalogues",
                sets_param = "selected_snapshot_catalogue_numbers",
                conversion_function = ScriptWrapper.make_list_converter(";"),
                default_value = None,
                description = "Check only the snapshot catalogues with these numbers.\nSpecify the snapshot numbers only as a semicolon (;) seperated list.\nSpecify as a single - to indicate all the remaining snapshot catalogues. This can only be at the end of the list."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "specific-snipshot-catalogues",
                sets_param = "selected_snipshot_catalogue_numbers",
                conversion_function = ScriptWrapper.make_list_converter(";"),
                default_value = None,
                description = "Check only the snipshots catalogues with these numbers.\nSpecify the snapshot numbers only as a semicolon (;) seperated list.\nSpecify as a single - to indicate all the remaining snipshot catalogues. This can only be at the end of the list."
            )
        )
    ).run(__main)



def __main(
            data_directory: str,
            output_filepath: str,
            number_of_processes: int,
            selected_snapshot_numbers: list[str]|None,
            selected_snipshot_numbers: list[str]|None,
            selected_snapshot_catalogue_numbers: list[str]|None,
            selected_snipshot_catalogue_numbers: list[str]|None
          ) -> None:

    # If any of the restrictions are set, initialise the unset ones to blank lists to prevent their computation
    if selected_snapshot_numbers is not None or selected_snipshot_numbers is not None or selected_snapshot_catalogue_numbers is not None or selected_snipshot_catalogue_numbers is not None:
        if selected_snapshot_numbers is None:
            selected_snapshot_numbers = []
        if selected_snipshot_numbers is None:
            selected_snipshot_numbers = []
        if selected_snapshot_catalogue_numbers is None:
            selected_snapshot_catalogue_numbers = []
        if selected_snipshot_catalogue_numbers is None:
            selected_snipshot_catalogue_numbers = []

    # Find the true path of the target directory
    data_directory = os.path.realpath(data_directory)

    Console.print_info(f"Scraping EAGLE file data from \"{data_directory}\".")

    files = FileTreeScraper_EAGLE(data_directory)

#    snap_info = SnapshotEAGLE.generate_filepaths_from_partial_info(data_directory)
#    snip_info = SnapshotEAGLE.generate_filepaths_from_partial_info(data_directory, snipshots = True)
#    cat_info = CatalogueSUBFIND.generate_filepaths_from_partial_info(data_directory)
#    snip_cat_info = CatalogueSUBFIND.generate_filepaths_from_partial_info(data_directory, snipshots = True)

    Console.print_info(f"Creating output file at \"{output_filepath}\".")

    with open(output_filepath, "w" if os.path.exists(output_filepath) else "x") as output_file:
        output_file.write(f"EAGLE directory: {data_directory}\n\n")

    Console.print_info("Doing snapshots.")

    bad_snapshot_numbers, missing_snapshot_file_indexes, corrupted_snapshot_file_indexes = check_particle_data(selected_snapshot_numbers, files.snapshots, SNAPSHOT_FIELDS_TO_CHECK, SNAPSHOT_FIELDS_TO_CHECK__DM, number_of_processes)

    # Write the snapshot results
    with open(output_filepath, "a") as output_file:
        output_file.write("Snapshot files:\n\n")
        for snapshot_number in bad_snapshot_numbers:
            if len(missing_snapshot_file_indexes[snapshot_number]) > 0 or len(corrupted_snapshot_file_indexes[snapshot_number]) > 0:
                output_file.write(f"        Snapshot {snapshot_number}:\n")
                if len(missing_snapshot_file_indexes[snapshot_number]) > 0:
                    output_file.write("            Missing snapshot file indexes:\n")
                    for i in missing_snapshot_file_indexes[snapshot_number]:
                        output_file.write(f"                {i}\n")
                if len(corrupted_snapshot_file_indexes[snapshot_number]) > 0:
                    output_file.write("            File indexes with corruption or missing data:\n")
                    for i in corrupted_snapshot_file_indexes[snapshot_number]:
                        output_file.write(f"                {i}\n")
                output_file.write("\n")
    #TODO: store a list of filepaths relitive to the root that need to be re-coppied

    Console.print_info("Doing snipshots.")

    bad_snipshot_numbers, missing_snipshot_file_indexes, corrupted_snipshot_file_indexes = check_particle_data(selected_snipshot_numbers, files.snipshots, SNIPSHOT_FIELDS_TO_CHECK, SNAPSHOT_FIELDS_TO_CHECK__DM, number_of_processes, snipshots = True)

    # Write the snipshot results
    with open(output_filepath, "a") as output_file:
        output_file.write("Snipshot files:\n\n")
        for snipshot_number in bad_snipshot_numbers:
            if len(missing_snipshot_file_indexes[snipshot_number]) > 0 or len(corrupted_snipshot_file_indexes[snipshot_number]) > 0:
                output_file.write(f"    Snipshot {snipshot_number}:\n")
                if len(missing_snipshot_file_indexes[snipshot_number]) > 0:
                    output_file.write("        Missing snipshot file indexes:\n")
                    for i in missing_snipshot_file_indexes[snipshot_number]:
                        output_file.write(f"            {i}\n")
                if len(corrupted_snipshot_file_indexes[snipshot_number]) > 0:
                    output_file.write("        File indexes with corruption or missing data:\n")
                    for i in corrupted_snipshot_file_indexes[snipshot_number]:
                        output_file.write(f"            {i}\n")
                output_file.write("\n")
    #TODO: store a list of filepaths relitive to the root that need to be re-coppied

    Console.print_info("Doing snapshot catalogues.")

    (
        bad_catalogue_numbers,
        (missing_catalogue_properties_file_indexes, corrupted_catalogue_properties_file_indexes),
        (missing_catalogue_membership_file_indexes, corrupted_catalogue_membership_file_indexes)
    ) = check_catalogue_data(
        selected_snapshot_catalogue_numbers,
        files.catalogues,
        CATALOGUE_PROPERTIES_GROUPS_TO_CHECK,
        CATALOGUE_PROPERTIES_FIELDS_TO_CHECK,
        SNAPSHOT_CATALOGUE_MEMBERSHIP_FIELDS_TO_CHECK,
        CATALOGUE_MEMBERSHIP_FIELDS_TO_CHECK__DM,
        number_of_processes
    )

    # Write the snapshot catalogue results
    with open(output_filepath, "a") as output_file:
        output_file.write("Snapshot catalogue files:\n\n")
        for snapshot_number in bad_catalogue_numbers:
            if len(missing_catalogue_properties_file_indexes[snapshot_number]) > 0 or len(corrupted_catalogue_properties_file_indexes[snapshot_number]) > 0 or len(missing_catalogue_membership_file_indexes[snapshot_number]) > 0 or len(corrupted_catalogue_membership_file_indexes[snapshot_number]) > 0:
                output_file.write(f"    Catalogue {snapshot_number}:\n")
                if len(missing_catalogue_properties_file_indexes[snapshot_number]) > 0 or len(corrupted_catalogue_properties_file_indexes[snapshot_number]) > 0:
                    output_file.write("        Properties:\n")
                    if len(missing_catalogue_properties_file_indexes[snapshot_number]) > 0:
                        output_file.write("            Missing catalogue file indexes:\n")
                        for i in missing_catalogue_properties_file_indexes[snapshot_number]:
                            output_file.write(f"                {i}\n")
                    if len(corrupted_catalogue_properties_file_indexes[snapshot_number]) > 0:
                        output_file.write("            File indexes with corruption or missing data:\n")
                        for i in corrupted_catalogue_properties_file_indexes[snapshot_number]:
                            output_file.write(f"                {i}\n")
                if len(missing_catalogue_membership_file_indexes[snapshot_number]) > 0 or len(corrupted_catalogue_membership_file_indexes[snapshot_number]) > 0:
                    output_file.write("        Membership:\n")
                    if len(missing_catalogue_membership_file_indexes[snapshot_number]) > 0:
                        output_file.write("            Missing catalogue file indexes:\n")
                        for i in missing_catalogue_membership_file_indexes[snapshot_number]:
                            output_file.write(f"                {i}\n")
                    if len(corrupted_catalogue_membership_file_indexes[snapshot_number]) > 0:
                        output_file.write("            File indexes with corruption or missing data:\n")
                        for i in corrupted_catalogue_membership_file_indexes[snapshot_number]:
                            output_file.write(f"                {i}\n")
                output_file.write("\n")
    #TODO: store a list of filepaths relitive to the root that need to be re-coppied

    Console.print_info("Doing snipshot catalogues.")

    (
        bad_catalogue_numbers,
        (missing_catalogue_properties_file_indexes, corrupted_catalogue_properties_file_indexes),
        (missing_catalogue_membership_file_indexes, corrupted_catalogue_membership_file_indexes)
    ) = check_catalogue_data(
        selected_snipshot_catalogue_numbers,
        files.snipshot_catalogues,
        CATALOGUE_PROPERTIES_GROUPS_TO_CHECK,
        CATALOGUE_PROPERTIES_FIELDS_TO_CHECK,
        SNIPSHOT_CATALOGUE_MEMBERSHIP_FIELDS_TO_CHECK,
        CATALOGUE_MEMBERSHIP_FIELDS_TO_CHECK__DM,
        number_of_processes,
        snipshots = True
    )

    # Write the snipshot catalogue results
    with open(output_filepath, "a") as output_file:
        output_file.write("Snipshot catalogue files:\n\n")
        for snipshot_number in bad_catalogue_numbers:
            if len(missing_catalogue_properties_file_indexes[snipshot_number]) > 0 or len(corrupted_catalogue_properties_file_indexes[snipshot_number]) > 0 or len(missing_catalogue_membership_file_indexes[snipshot_number]) > 0 or len(corrupted_catalogue_membership_file_indexes[snipshot_number]) > 0:
                output_file.write(f"    Snipshot catalogue {snipshot_number}:\n")
                if len(missing_catalogue_properties_file_indexes[snipshot_number]) > 0 or len(corrupted_catalogue_properties_file_indexes[snipshot_number]) > 0:
                    output_file.write("        Properties:\n")
                    if len(missing_catalogue_properties_file_indexes[snipshot_number]) > 0:
                        output_file.write("            Missing catalogue file indexes:\n")
                        for i in missing_catalogue_properties_file_indexes[snipshot_number]:
                            output_file.write(f"                {i}\n")
                    if len(corrupted_catalogue_properties_file_indexes[snipshot_number]) > 0:
                        output_file.write("            File indexes with corruption or missing data:\n")
                        for i in corrupted_catalogue_properties_file_indexes[snipshot_number]:
                            output_file.write(f"                {i}\n")
                if len(missing_catalogue_membership_file_indexes[snipshot_number]) > 0 or len(corrupted_catalogue_membership_file_indexes[snipshot_number]) > 0:
                    output_file.write("        Membership:\n")
                    if len(missing_catalogue_membership_file_indexes[snipshot_number]) > 0:
                        output_file.write("            Missing catalogue file indexes:\n")
                        for i in missing_catalogue_membership_file_indexes[snipshot_number]:
                            output_file.write(f"                {i}\n")
                    if len(corrupted_catalogue_membership_file_indexes[snipshot_number]) > 0:
                        output_file.write("            File indexes with corruption or missing data:\n")
                        for i in corrupted_catalogue_membership_file_indexes[snipshot_number]:
                            output_file.write(f"                {i}\n")
                output_file.write("\n")
    #TODO: store a list of filepaths relitive to the root that need to be re-coppied

    Console.print_info("DONE.")



def check_particle_data_inner(snapshot_number: str, snapshot_file_info: dict[int, str], expected_number_of_snapshot_files: int, fields_to_check: Iterable[str], dark_matter_fields_to_check: Iterable[str], missing_snapshot_file_indexes: dict[str, list[int]], corrupted_snapshot_file_indexes: dict[str, list[int]], snipshots = False) -> None:

    missing_snapshot_file_indexes[snapshot_number] = []
    corrupted_snapshot_file_indexes[snapshot_number] = []

    if not snipshots:
        Console.print_info(f"    Doing snapshot {snapshot_number}.")
    else:
        Console.print_info(f"    Doing snipshot {snapshot_number}.")
    snap_file_indexes = snapshot_file_info.keys()
    # Check that the target snapshot has the right number of files
    if len(snap_file_indexes) < expected_number_of_snapshot_files:
        Console.print_error(f"        Too few files! Got {len(snap_file_indexes)} but expected {expected_number_of_snapshot_files}.")
        for i in range(expected_number_of_snapshot_files):
            # Record any files that aren't present
            if i not in snap_file_indexes:
                Console.print_verbose_error(f"            Missing file index {i}.")
                missing_snapshot_file_indexes[snapshot_number].append(i)
    # Of the files that are present, check that they have the right data
    for snapshot_file_index, snapshot_file_path in snapshot_file_info.items():
        file_valid = True
        try:
            file = h5.File(snapshot_file_path)
        except:
            Console.print_error(f"        File index {snapshot_file_index} is unreadable.")
            file_valid = False
        if file_valid:
            with file:
                file_has_supplementary_groups = ("Header" in file.keys()) and ("Units" in file.keys()) and ("Parameters" in file.keys()) and ("Constants" in file.keys())
                file_has_gas = ParticleType.gas.common_hdf5_name in file.keys()
                file_has_dark_matter = ParticleType.dark_matter.common_hdf5_name in file.keys()
                file_has_stars = ParticleType.star.common_hdf5_name in file.keys()
                file_has_black_holes = ParticleType.black_hole.common_hdf5_name in file.keys()
                # All files must have both DM and gas particles
                if not (file_has_supplementary_groups and file_has_gas and file_has_dark_matter):
                    Console.print_error(f"        File index {snapshot_file_index} missing DM and/or gas particle group(s).")
                    if not file_has_supplementary_groups:
                        Console.print_verbose_error("            Header/Units/Parameters/Constants")
                    if not file_has_gas:
                        Console.print_verbose_error("            PartType0")
                    if not file_has_dark_matter:
                        Console.print_verbose_error("            PartType1")
                    file_valid = False
                else:
                    for part_type in ParticleType.get_all():
                        if (part_type == ParticleType.star and not file_has_stars) or (part_type == ParticleType.black_hole and not file_has_black_holes):
                            # Missing non-required fields can be skipped
                            continue
                        dataset = file[part_type.common_hdf5_name]
                        try:
                            _ = list(dataset.keys())
                        except RuntimeError as e:
                            Console.print_verbose_error(f"        Unable to read group {part_type.common_hdf5_name}.")
                            file_valid = False
                        if file_valid:
                            for field in dark_matter_fields_to_check if part_type == ParticleType.dark_matter else fields_to_check:
                                if field not in dataset:
                                    Console.print_verbose_error(f"        Unable to find dataset {field} of group {part_type.common_hdf5_name}.")
                                    file_valid = False
                                    # Can stop checking fields if even one required dataset is missing
                                    break
                                else:
                                    try:
                                        _ = str(dataset[field])
                                    except:
                                        # Field exists but is corrupted
                                        Console.print_verbose_error(f"        Unable to read dataset {field} of group {part_type.common_hdf5_name}.")
                                        file_valid = False
                                        # Can stop checking fields if even one required dataset is corrupted
                                        break
                        if not file_valid:
                            # Can stop checking fields if even one is corrupted
                            break
                    if not file_valid:
                        Console.print_error(f"        File index {snapshot_file_index} missing one or more required data field(s).")
        if not file_valid:
            corrupted_snapshot_file_indexes[snapshot_number].append(snapshot_file_index)

def check_particle_data(selected_snapshot_numbers: list[str]|None, snapshot_info: SimulationSnapshotFiles_EAGLE|SimulationSnipshotFiles_EAGLE, fields_to_check: Iterable[str], dark_matter_fields_to_check: Iterable[str], number_of_worker_processes: int = 1, snipshots = False) -> tuple[tuple[str, ...], dict[str, list[int]], dict[str, list[int]]]:

    if selected_snapshot_numbers is not None and len(selected_snapshot_numbers) == 0:
        return tuple(), {}, {}

    # Find the maximum number of files in a snapshot/snapshot - all snapshots should have the same number of files at each redshift
    expected_number_of_snapshot_files = 0
    for snapshot_file_data in snapshot_info:
        expected_number_of_snapshot_files = max(expected_number_of_snapshot_files, len(snapshot_file_data))

    missing_snapshot_file_indexes: dict[str, list[int]]
    corrupted_snapshot_file_indexes: dict[str, list[int]]

    ordered_snap_nums: Sequence[str] = snapshot_info.get_numbers()

    if selected_snapshot_numbers is None or len(selected_snapshot_numbers) == 1 and selected_snapshot_numbers[0].strip() == "-":
        Console.print_debug("No restrictions - checking all files.")
    else:
        if selected_snapshot_numbers[-1].strip() == "-":
            selected_snapshot_numbers = selected_snapshot_numbers[:-1]
            remaining_valid_selected_snapshot_numbers = [i for i in selected_snapshot_numbers if i in ordered_snap_nums]
            if len(remaining_valid_selected_snapshot_numbers) == 0:
                selected_snapshot_numbers.extend(ordered_snap_nums)
            else:
                selected_snapshot_numbers.extend(ordered_snap_nums[ordered_snap_nums.index(remaining_valid_selected_snapshot_numbers[-1]):])
        ordered_snap_nums = [i for i in ordered_snap_nums if i in selected_snapshot_numbers]
        if len(ordered_snap_nums) > len(selected_snapshot_numbers):
            for i in selected_snapshot_numbers:
                if i not in ordered_snap_nums:
                    is_valid = True
                    try:
                        is_valid = int(i) >= 0
                    except:
                        is_valid = False
                    if is_valid:
                        Console.print_warning(f"{'Snapshot' if not snipshots else 'Snipshot'} {i} requested but is not present. This will be ignored!")
                    else:
                        Console.print_warning(f"\"{i}\" is not a valid {'snapshot' if not snipshots else 'snipshot'} number. This will be ignored!")
        Console.print_info(f"Doing only {'snapshot(s)' if not snipshots else 'snipshot(s)'}:", ", ".join(ordered_snap_nums))

    parallel_indexed_filepaths: dict[int, str]

    if number_of_worker_processes > 1:
        Console.print_info(f"Running in parallel using {number_of_worker_processes} processes.")

        manager = multiprocessing.Manager()
        missing_snapshot_file_indexes = manager.dict()
        corrupted_snapshot_file_indexes = manager.dict()

        pool = multiprocessing.Pool(number_of_worker_processes)

        for i in ordered_snap_nums:
            parallel_indexed_filepaths = { int(f.split(".")[-2]) : f for f in snapshot_info.get_by_number(i).filepaths }
            pool.apply(check_particle_data_inner, args = (i, parallel_indexed_filepaths, expected_number_of_snapshot_files, fields_to_check, dark_matter_fields_to_check, missing_snapshot_file_indexes, corrupted_snapshot_file_indexes), kwds = { "snipshots": snipshots })

        pool.join()

    else:
        Console.print_info("Running in serial.")

        missing_snapshot_file_indexes = {}
        corrupted_snapshot_file_indexes = {}
        for i in ordered_snap_nums:
            parallel_indexed_filepaths = { int(f.split(".")[-2]) : f for f in snapshot_info.get_by_number(i).filepaths }
            check_particle_data_inner(i, parallel_indexed_filepaths, expected_number_of_snapshot_files, fields_to_check, dark_matter_fields_to_check, missing_snapshot_file_indexes, corrupted_snapshot_file_indexes, snipshots = snipshots)

    return tuple([i for i in ordered_snap_nums if (i in missing_snapshot_file_indexes or i in corrupted_snapshot_file_indexes)]), missing_snapshot_file_indexes, corrupted_snapshot_file_indexes

def check_catalogue_data_inner(snapshot_number: str, catalogue_properties_file_info: dict[int, str], catalogue_membership_file_info: dict[int, str], expected_number_of_properties_files: int, expected_number_of_membership_files: int, properties_groups_to_check: Iterable[str], properties_fields_to_check: dict[str, Iterable[str]], membership_fields_to_check: Iterable[str], membership_dark_matter_fields_to_check: Iterable[str], missing_properties_file_indexes: dict[str, list[int]], missing_membership_file_indexes: dict[str, list[int]], corrupted_properties_file_indexes: dict[str, list[int]], corrupted_membership_file_indexes: dict[str, list[int]], snipshots = False) -> None:

    missing_properties_file_indexes[snapshot_number] = []
    missing_membership_file_indexes[snapshot_number] = []
    corrupted_properties_file_indexes[snapshot_number] = []
    corrupted_membership_file_indexes[snapshot_number] = []

    if not snipshots:
        Console.print_info(f"    Doing snapshot catalogue {snapshot_number}.")
    else:
        Console.print_info(f"    Doing snipshot catalogue {snapshot_number}.")

    cat_properties_file_indexes = catalogue_properties_file_info.keys()
    # Check that the target catalogue has the right number of files
    if len(cat_properties_file_indexes) < expected_number_of_properties_files:
        Console.print_error(f"        Too few properties files! Got {len(cat_properties_file_indexes)} but expected {expected_number_of_properties_files}.")
        for i in range(expected_number_of_properties_files):
            # Record any files that aren't present
            if i not in cat_properties_file_indexes:
                Console.print_verbose_error(f"            Missing file index {i}.")
                missing_properties_file_indexes[snapshot_number].append(i)

    cat_membership_file_indexes = catalogue_membership_file_info.keys()
    # Check that the target catalogue has the right number of files
    if len(cat_membership_file_indexes) < expected_number_of_membership_files:
        Console.print_error(f"        Too few membership files! Got {len(cat_membership_file_indexes)} but expected {expected_number_of_membership_files}.")
        for i in range(expected_number_of_membership_files):
            # Record any files that aren't present
            if i not in cat_membership_file_indexes:
                Console.print_verbose_error(f"            Missing file index {i}.")
                missing_membership_file_indexes[snapshot_number].append(i)

    # Of the files that are present, check that they have the right data
    for catalogue_file_index, catalogue_file_path in catalogue_properties_file_info.items():
        file_valid = True
        try:
            file = h5.File(catalogue_file_path)
        except:
            Console.print_error(f"        Properties file index {catalogue_file_index} is unreadable.")
            file_valid = False
        if file_valid:
            with file:
                file_has_supplementary_groups = ("Header" in file.keys()) and ("Units" in file.keys()) and ("Parameters" in file.keys()) and ("Constants" in file.keys())
                if not file_has_supplementary_groups:
                    Console.print_error(f"        Properties file index {catalogue_file_index} missing required metadata group(s).")
                    if not file_has_supplementary_groups:
                        Console.print_verbose_error("            Header/Units/Parameters/Constants")
                    file_valid = False
                if file_valid:
                    has_haloes = file["Header"].attrs["TotNgroups"] > 0
                    if has_haloes:
                        missing_groups = { group_name : True for group_name in properties_groups_to_check }
                        for group_name in properties_groups_to_check:
                            if group_name in file:
                                try:
                                    group = file[group_name]
                                    _ = list(group.keys())
                                    missing_groups[group_name] = False
                                except (RuntimeError, KeyError) as e: # KeyError can occour when the group key exists but the group itsself is corrupt
                                    pass#missing_groups[group_name] = True
                        if any(missing_groups.values()):
                            Console.print_error(f"        Properties file index {catalogue_file_index} missing required data group(s).")
                            for key, value in missing_groups.items():
                                if value:
                                    Console.print_verbose_error(f"            {key}")
                            file_valid = False
                        else:
                            if file["FOF"].attrs["Ngroups"] > 0: # Data groups will be blank if there are no haloes in the parallel file!
                                for group_name, field_names in properties_fields_to_check.items():
                                    group = file[group_name]
                                    for field_name in field_names:
                                        if field_name not in group.keys():
                                            Console.print_verbose_error(f"        Unable to find dataset {field_name} of group {group_name}.")
                                            file_valid = False
                                            break
                                        try:
                                            dataset = group[field_name]
                                            _ = list(dataset.attrs.keys())
                                        except (RuntimeError, KeyError) as e: # KeyError can occour when the group key exists but the group itsself is corrupt
                                            Console.print_verbose_error(f"        Unable to read dataset {field_name} of group {group_name}.")
                                            file_valid = False
                                            break
                                    if not file_valid:
                                        break
                                if not file_valid:
                                    Console.print_error(f"        Properties file index {catalogue_file_index} missing one or more required data field(s).")
        if not file_valid:
            corrupted_properties_file_indexes[snapshot_number].append(catalogue_file_index)

    # Of the files that are present, check that they have the right data
    for catalogue_file_index, catalogue_file_path in catalogue_membership_file_info.items():
        file_valid = True
        try:
            file = h5.File(catalogue_file_path)
        except:
            Console.print_error(f"        Membership file index {catalogue_file_index} is unreadable.")
            file_valid = False
        if file_valid:
            with file:
                file_has_supplementary_groups = ("Header" in file.keys()) and ("Units" in file.keys()) and ("Parameters" in file.keys()) and ("Constants" in file.keys())
                if not file_has_supplementary_groups:
                    Console.print_error(f"        Membership file index {catalogue_file_index} missing required metadata group(s).")
                    if not file_has_supplementary_groups:
                        Console.print_verbose_error("            Header/Units/Parameters/Constants")
                    file_valid = False
                if file_valid:
                    has_haloes = sum(file["Header"].attrs["NumPart_ThisFile"]) > 0
                    if has_haloes:
                        file_has_gas = ParticleType.gas.common_hdf5_name in file.keys()
                        file_has_dark_matter = ParticleType.dark_matter.common_hdf5_name in file.keys()
                        file_has_stars = ParticleType.star.common_hdf5_name in file.keys()
                        file_has_black_holes = ParticleType.black_hole.common_hdf5_name in file.keys()
                        # All files must have both DM and gas particles if any haloes exist
                        if not (file_has_gas and file_has_dark_matter):
                            Console.print_error(f"        Membership file index {catalogue_file_index} missing DM and/or gas particle group(s).")
                            if not file_has_gas:
                                Console.print_verbose_error("            PartType0")
                            if not file_has_dark_matter:
                                Console.print_verbose_error("            PartType1")
                            file_valid = False
                        else:
                            for part_type in ParticleType.get_all():
                                if (part_type == ParticleType.star and not file_has_stars) or (part_type == ParticleType.black_hole and not file_has_black_holes):
                                    # Missing non-required fields can be skipped
                                    continue
                                dataset = file[part_type.common_hdf5_name]
                                try:
                                    _ = list(dataset.keys())
                                except RuntimeError as e:
                                    Console.print_verbose_error(f"        Unable to read group {part_type.common_hdf5_name}.")
                                    file_valid = False
                                if file_valid:
                                    for field in membership_dark_matter_fields_to_check if part_type == ParticleType.dark_matter else membership_fields_to_check:
                                        if field not in dataset:
                                            Console.print_verbose_error(f"        Unable to find dataset {field} of group {part_type.common_hdf5_name}.")
                                            file_valid = False
                                            # Can stop checking fields if even one required dataset is missing
                                            break
                                        else:
                                            try:
                                                _ = str(dataset[field])
                                            except:
                                                # Field exists but is corrupted
                                                Console.print_verbose_error(f"        Unable to read dataset {field} of group {part_type.common_hdf5_name}.")
                                                file_valid = False
                                                # Can stop checking fields if even one required dataset is corrupted
                                                break
                                if not file_valid:
                                    # Can stop checking fields if even one is corrupted
                                    break
                            if not file_valid:
                                Console.print_error(f"        Membership file index {catalogue_file_index} missing one or more required data field(s).")
        if not file_valid:
            corrupted_membership_file_indexes[snapshot_number].append(catalogue_file_index)

def check_catalogue_data(selected_snapshot_numbers: list[str]|None, catalogue_info: SimulationSnapshotCatalogueFiles_EAGLE|SimulationSnipshotCatalogueFiles_EAGLE, properties_groups_to_check: Iterable[str], properties_fields_to_check: Mapping[str, Iterable[str]], membership_fields_to_check: Iterable[str], membership_dark_matter_fields_to_check: Iterable[str], number_of_worker_processes: int = 1, snipshots = False) -> tuple[tuple[str, ...], tuple[dict[str, list[int]], dict[str, list[int]]], tuple[dict[str, list[int]], dict[str, list[int]]]]:

    if selected_snapshot_numbers is not None and len(selected_snapshot_numbers) == 0:
        return tuple(), ({}, {}), ({}, {})

    missing_catalogue_properties_file_indexes: dict[str, list[int]]
    missing_catalogue_membership_file_indexes: dict[str, list[int]]
    corrupted_catalogue_properties_file_indexes: dict[str, list[int]]
    corrupted_catalogue_membership_file_indexes: dict[str, list[int]]

    ordered_snap_nums: Sequence[str] = catalogue_info.get_numbers()

    if selected_snapshot_numbers is None or len(selected_snapshot_numbers) == 1 and selected_snapshot_numbers[0].strip() == "-":
        Console.print_debug("No restrictions - checking all files.")
    else:
        if selected_snapshot_numbers[-1].strip() == "-":
            selected_snapshot_numbers = selected_snapshot_numbers[:-1]
            remaining_valid_selected_snapshot_numbers = [i for i in selected_snapshot_numbers if i in ordered_snap_nums]
            if len(remaining_valid_selected_snapshot_numbers) == 0:
                selected_snapshot_numbers.extend(ordered_snap_nums)
            else:
                selected_snapshot_numbers.extend(ordered_snap_nums[ordered_snap_nums.index(remaining_valid_selected_snapshot_numbers[-1]):])
        ordered_snap_nums = [i for i in ordered_snap_nums if i in selected_snapshot_numbers]
        if len(ordered_snap_nums) > len(selected_snapshot_numbers):
            for i in selected_snapshot_numbers:
                if i not in ordered_snap_nums:
                    is_valid = True
                    try:
                        is_valid = int(i) >= 0
                    except:
                        is_valid = False
                    if is_valid:
                        Console.print_warning(f"{'Catalogue' if not snipshots else 'Snipshot catalogue'} {i} requested but is not present. This will be ignored!")
                    else:
                        Console.print_warning(f"\"{i}\" is not a valid {'snapshot' if not snipshots else 'snipshot'} catalogue number. This will be ignored!")
        Console.print_info(f"Doing only {'snapshot' if not snipshots else 'snipshot'} catalogue(s):", ", ".join(ordered_snap_nums))

    parallel_indexed_filepaths: dict[int, str]

    if number_of_worker_processes > 1:
        Console.print_info(f"Running in parallel using {number_of_worker_processes} processes.")

        manager = multiprocessing.Manager()
        missing_catalogue_properties_file_indexes = manager.dict()
        missing_catalogue_membership_file_indexes = manager.dict()
        corrupted_catalogue_properties_file_indexes = manager.dict()
        corrupted_catalogue_membership_file_indexes = manager.dict()

        pool = multiprocessing.Pool(number_of_worker_processes)
    else:
        Console.print_info("Running in serial.")

        missing_catalogue_properties_file_indexes = {}
        missing_catalogue_membership_file_indexes = {}
        corrupted_catalogue_properties_file_indexes = {}
        corrupted_catalogue_membership_file_indexes = {}

    for i in ordered_snap_nums:

        n_properties_files = None
        n_membership_files = None
        for filepath in catalogue_info.get_by_number(i).properties_filepaths:
            try:
                value = None
                with h5.File(filepath) as file:
                    if "Header" not in file:
                        raise RuntimeError()
                    if "NumFilesPerSnapshot" not in file["Header"].attrs:
                        raise RuntimeError()
                    value = file["Header"].attrs["NumFilesPerSnapshot"]
            except:
                continue
            if value is not None:
                n_properties_files = value
                break
        if n_properties_files is None:
            raise RuntimeError()
        for filepath in catalogue_info.get_by_number(i).membership_filepaths:
            try:
                value = None
                with h5.File(filepath) as file:
                    if "Header" not in file:
                        raise RuntimeError()
                    if "NumFilesPerSnapshot" not in file["Header"].attrs:
                        raise RuntimeError()
                    value = file["Header"].attrs["NumFilesPerSnapshot"]
            except:
                continue
            if value is not None:
                n_membership_files = value
                break
        if n_membership_files is None:
            raise RuntimeError()
        parallel_indexed_properties_filepaths = { int(f.split(".")[-2]) : f for f in catalogue_info.get_by_number(i).properties_filepaths }
        parallel_indexed_membership_filepaths = { int(f.split(".")[-2]) : f for f in catalogue_info.get_by_number(i).membership_filepaths }

        if number_of_worker_processes > 1:
            pool.apply(check_catalogue_data_inner, args = (i, parallel_indexed_properties_filepaths, parallel_indexed_membership_filepaths, n_properties_files, n_membership_files, properties_groups_to_check, properties_fields_to_check, membership_fields_to_check, membership_dark_matter_fields_to_check, missing_catalogue_properties_file_indexes, missing_catalogue_membership_file_indexes, corrupted_catalogue_properties_file_indexes, corrupted_catalogue_membership_file_indexes), kwds = { "snipshots": snipshots })
        else:
            check_catalogue_data_inner(i, parallel_indexed_properties_filepaths, parallel_indexed_membership_filepaths, n_properties_files, n_membership_files, properties_groups_to_check, properties_fields_to_check, membership_fields_to_check, membership_dark_matter_fields_to_check, missing_catalogue_properties_file_indexes, missing_catalogue_membership_file_indexes, corrupted_catalogue_properties_file_indexes, corrupted_catalogue_membership_file_indexes, snipshots = snipshots)
    if number_of_worker_processes > 1:
        pool.join()

#    if number_of_worker_processes > 1:
#        Console.print_info(f"Running in parallel using {number_of_worker_processes} processes.")
#
#        manager = multiprocessing.Manager()
#        missing_catalogue_properties_file_indexes = manager.dict()
#        missing_catalogue_membership_file_indexes = manager.dict()
#        corrupted_catalogue_properties_file_indexes = manager.dict()
#        corrupted_catalogue_membership_file_indexes = manager.dict()
#
#        pool = multiprocessing.Pool(number_of_worker_processes)
#
#        for i in ordered_snap_nums:
##            n_properties_files = 
##            n_membership_files = 
##            parallel_indexed_properties_filepaths = { int(f.split(".")[-2]) : f for f in catalogue_info.get_by_number(i).properties_filepaths }
##            parallel_indexed_membership_filepaths = { int(f.split(".")[-2]) : f for f in catalogue_info.get_by_number(i).membership_filepaths }
#            pool.apply(check_catalogue_data_inner, args = (i, parallel_indexed_properties_filepaths, parallel_indexed_membership_filepaths, n_properties_files, n_membership_files, properties_groups_to_check, properties_fields_to_check, membership_fields_to_check, membership_dark_matter_fields_to_check, missing_catalogue_properties_file_indexes, missing_catalogue_membership_file_indexes, corrupted_catalogue_properties_file_indexes, corrupted_catalogue_membership_file_indexes), kwds = { "snipshots": snipshots })
#
#        pool.join()
#
#    else:
#        Console.print_info("Running in serial.")
#
#        missing_catalogue_properties_file_indexes = {}
#        missing_catalogue_membership_file_indexes = {}
#        corrupted_catalogue_properties_file_indexes = {}
#        corrupted_catalogue_membership_file_indexes = {}
#        for i in ordered_snap_nums:
#            check_catalogue_data_inner(i, parallel_indexed_properties_filepaths, parallel_indexed_membership_filepaths, n_properties_files, n_membership_files, properties_groups_to_check, properties_fields_to_check, membership_fields_to_check, membership_dark_matter_fields_to_check, missing_catalogue_properties_file_indexes, missing_catalogue_membership_file_indexes, corrupted_catalogue_properties_file_indexes, corrupted_catalogue_membership_file_indexes, snipshots = snipshots)

    return (
        tuple([
            i
            for i
            in ordered_snap_nums
            if (i in missing_catalogue_properties_file_indexes or i in missing_catalogue_membership_file_indexes or i in corrupted_catalogue_properties_file_indexes or i in corrupted_catalogue_membership_file_indexes)
        ]),
        (missing_catalogue_properties_file_indexes, corrupted_catalogue_properties_file_indexes),
        (missing_catalogue_membership_file_indexes, corrupted_catalogue_membership_file_indexes)
    )
