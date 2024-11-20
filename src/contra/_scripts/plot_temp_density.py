# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .. import ParticleType
from .._L_star import get_L_star_halo_mass_of_z
from ..io import SnapshotBase, SnapshotEAGLE, SnapshotSWIFT, FileTreeScraper_EAGLE
from ..io._Output_Objects import OutputReader, ParticleTypeDataset, ContraData, DistributedOutputReader
from ..plotting._hexbin import plot_hexbin, create_hexbin_log10_count, create_hexbin_mean, create_hexbin_log10_mean, create_hexbin_weighted_mean, create_hexbin_log10_weighted_mean, create_hexbin_median

import datetime
import os
from typing import cast, Union, List, Tuple, Dict
import asyncio

import numpy as np
from matplotlib import pyplot as plt
from unyt import unyt_quantity, unyt_array
from QuasarCode import Settings, Console
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import ScriptWrapper
from QuasarCode.MPI import MPI_Config
import tol_colors

# 0.0134
Z_SOLAR = 0.012663729 # Z_sun (from EAGLE L50N752 -> Constants -> Z_Solar)
PRIMORDIAL_H_ABUNDANCE = 0.752 # Mass fraction of H (from EAGLE L50N752 -> Parameters -> ChemicalElements -> InitAbundance_Hydrogen)
N_CONTOURING_BINS = 50
N_PLOTTING_HEXES = 500

def main():
    ScriptWrapper(
        command = "contra-plot-tracked-environment",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 11, 20),
        description = "",
        usage_paramiter_examples = ["-z 0 -i contra-output.hdf5 --colour-count --colour-metallicity --colour-last-halo-mass --colour-metal-weighted-last-halo-mass --mean --min-colour-metalicity-plotted -5.0 --min-colour-metalicity-plotted 1.0 --stack-row"],
        parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.RequiredParam[float](
                name = "redshift",
                short_name = "z",
                sets_param = "target_redshift",
                conversion_function = float,
                description = "Redshift at which to plot data.\nIf a match is not found, use the file with the closest redshift with a higher value."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "data",
                short_name = "i",
                sets_param = "input_filepath",
                default_value = "./contra-output.hdf5",
                description = "Input contra data file. Defaults to \"./contra-output.hdf5\"."
            ),
            ScriptWrapper.Flag(
                name = "metals-only",
                description = "Ignore tracked particles with no metal content."
            ),
            ScriptWrapper.Flag(
                name = "use-number-density",
                description = "Plot the hydrogen number density instead of overdensity."
            ),
            ScriptWrapper.Flag(
                name = "colour-count",
                sets_param = "plot_hist",
                description = "Colour by the number of particles in heach bin.\nThis will be automatically set if no other colour options are specified."
            ),
            ScriptWrapper.Flag(
                name = "colour-metallicity",
                sets_param = "plot_metallicity",
                description = "Colour by the metal mass fraction."
            ),
            ScriptWrapper.Flag(
                name = "colour-last-halo-mass",
                sets_param = "plot_last_halo_mass",
                description = "Colour by the mass of the last known halo."
            ),
            ScriptWrapper.Flag(
                name = "colour-metal-weighted-last-halo-mass",
                sets_param = "plot_metal_weighted_last_halo_mass",
                description = "Colour by mean metal mass weighted log10 last known halo mass.\nRequires the use of the --mean option.",
                requirements = ["mean"]
            ),
            ScriptWrapper.Flag(
                name = "colour-metal-weighted-last-halo-redshift",
                sets_param = "plot_metal_weighted_last_halo_redshift",
                description = "Colour by mean metal mass weighted last known halo redshift.\nRequires the use of the --mean option.",
                requirements = ["mean"]
            ),
            ScriptWrapper.OptionalParam[str](
                name = "snapshots",
                sets_param = "snapshot_directory",
                description = "Where to search for snapshots.\nDefaults to the snapshot location specified in the Contra output file."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "output-file",
                short_name = "o",
                sets_param = "output_filepath",
                description = "File to save plot to.\nWhen not specified, interactive plot will be shown."
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-density",
                description = "Minimum density (in Msun/Mpc^3) to display.",
                conversion_function = float,
                conflicts = ["min-overdensity", "use-number-density"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-density",
                description = "Maximum density (in Msun/Mpc^3) to display.",
                conversion_function = float,
                conflicts = ["max-overdensity", "use-number-density"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-overdensity",
                sets_param = "min_log10_overdensity",
                description = "Minimum (log10) overdensity to display.",
                conversion_function = float,
                conflicts = ["min-density", "use-number-density"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-overdensity",
                sets_param = "max_log10_overdensity",
                description = "Maximum (log10) overdensity to display.",
                conversion_function = float,
                conflicts = ["max-density", "use-number-density"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-numberdensity",
                sets_param = "min_log10_numberdensity",
                description = "Minimum (log10) hydrogen number density to display.",
                conversion_function = float,
                requirements = ["use-number-density"],
                conflicts = ["min-density"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-numberdensity",
                sets_param = "max_log10_numberdensity",
                description = "Maximum (log10) hydrogen number density to display.",
                conversion_function = float,
                requirements = ["use-number-density"],
                conflicts = ["max-density"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-temp",
                sets_param = "min_log10_temp",
                description = "Minimum temperature (in log10 K) to display.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-temp",
                sets_param = "max_log10_temp",
                description = "Maximum temperature (in log10 K) to display.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-colour-log-count",
                description = "Minimum number of histogram counts, below which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-log-count",
                description = "Maximum number of histogram counts, above which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-colour-metalicity",
                description = "Minimum metalicity (metal-mass fraction), below which the colour will be uniform.",
                conversion_function = float,
                conflicts = ["min-colour-metalicity-solar", "min-colour-metalicity-solar"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-metalicity",
                description = "Maximum metalicity (metal-mass fraction), above which the colour will be uniform.",
                conversion_function = float,
                conflicts = ["max-colour-metalicity-solar", "max-colour-metalicity-solar"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-colour-metalicity-solar",
                description = "Minimum solar metalicity, below which the colour will be uniform.",
                conversion_function = float,
                conflicts = ["min-colour-metalicity", "min-colour-metalicity-plotted"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-metalicity-solar",
                description = "Maximum solar metalicity, above which the colour will be uniform.",
                conversion_function = float,
                conflicts = ["max-colour-metalicity", "max-colour-metalicity-plotted"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-colour-metalicity-plotted",
                description = "Minimum metalicity (log10 1 + (Z/Z_sun)), below which the colour will be uniform.",
                conversion_function = float,
                conflicts = ["min-colour-metalicity", "min-colour-metalicity-solar"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-metalicity-plotted",
                description = "Maximum solar metalicity (log10 1 + (Z/Z_sun)), above which the colour will be uniform.",
                conversion_function = float,
                conflicts = ["max-colour-metalicity", "max-colour-metalicity-solar"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-colour-halo-mass",
                description = "Minimum halo mass (in log10 Msun), below which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-halo-mass",
                description = "Maximum halo mass (in log10 Msun), above which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-colour-halo-redshift",
                description = "Minimum redshift, below which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-halo-redshift",
                description = "Maximum redshift, above which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.Flag(
                name = "mean",
                sets_param = "use_mean",
                description = "Use mean to determine colour values instead of the median."
            ),
            ScriptWrapper.Flag(
                name = "stack-row",
                description = "Stack plots as a single row.",
                conflicts = ["stack-column"]
            ),
            ScriptWrapper.Flag(
                name = "stack-column",
                description = "Stack plots as a single column.",
                conflicts = ["stack-row"]
            ),
#            ScriptWrapper.Flag(
#                name = "show-igm",
#                description = "Draw the z=0 IGM boundary from Wiersma et al. 2010."
#            ),
            ScriptWrapper.Flag(
                name = "ignore-halo-particles",
                description = "Exclude particles currently in a halo."
            ),
            ScriptWrapper.Flag(
                name = "show-contours",
                description = "Draw contours showing the mass distribution of included particles."
            ),
            ScriptWrapper.Flag(
                name = "contours-use-all-particles",
                description = "Extend the number of particles used to generate the contours to include particles that have no associated halo.\nThis will make use of all gas particles at the target redshift.",
                requirements = ["show-contours"]
            ),
            ScriptWrapper.Flag(
                name = "contours-use-masses",
                description = "Use contours based on the total mass density in each bin as opposed to the total number density.",
                requirements = ["show-contours"]
            ),
            ScriptWrapper.OptionalParam[list[float]](
                name = "contour-percentiles",
                default_value = [10, 25, 50, 75, 90],
                description = "Percentiles at which to plot contours, as a comma seperated list (in ascending order).\nDefaults to: 10%, 25%, 50%, 75% and 90%.",
                conversion_function = ScriptWrapper.make_list_converter(",", float),
                conflicts = ["contour-values"]
            ),
            ScriptWrapper.OptionalParam[list[float]](
                name = "contour-values",
                sets_param = "contour_log10_values",
                description = "log10 bin density values at which to plot contours, as a comma seperated list (in ascending order).\nNote, these values will be specific to either number density or mass density contours!",
                conversion_function = ScriptWrapper.make_list_converter(",", float),
                conflicts = ["contour-percentiles"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "solar-metal-mass-fraction",
                sets_param = "configured_Z_solar",
                default_value = Z_SOLAR,
                description = f"Value of Z_sun used to calculate solar metallicity from absolute metal mass fraction.\nDefaults to {Z_SOLAR}.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[str](
                name = "colourmap",
                description = "Alternitive matplotlib colourmap name."
            )
        )
    ).run_with_async(__main)

async def __main(
    target_redshift: float,
    input_filepath: str,
    metals_only: bool,
    use_number_density: bool,
    plot_hist: bool,
    plot_metallicity: bool,
    plot_last_halo_mass: bool,
    plot_metal_weighted_last_halo_mass: bool,
    plot_metal_weighted_last_halo_redshift: bool,
    snapshot_directory: str | None,
    output_filepath: str | None,
    min_density: float | None,
    max_density: float | None,
    min_log10_overdensity: float | None,
    max_log10_overdensity: float | None,
    min_log10_numberdensity: float | None,
    max_log10_numberdensity: float | None,
    min_log10_temp: float | None,
    max_log10_temp: float | None,
    min_colour_log_count: float | None,
    max_colour_log_count: float | None,
    min_colour_metalicity: float | None,
    max_colour_metalicity: float | None,
    min_colour_metalicity_solar: float | None,
    max_colour_metalicity_solar: float | None,
    min_colour_metalicity_plotted: float | None,
    max_colour_metalicity_plotted: float | None,
    min_colour_halo_mass: float | None,
    max_colour_halo_mass: float | None,
    min_colour_halo_redshift: float | None,
    max_colour_halo_redshift: float | None,
    use_mean: bool,
    stack_row: bool,
    stack_column: bool,
#    show_igm: bool,
    ignore_halo_particles: bool,
    show_contours: bool,
    contours_use_all_particles: bool,
    contours_use_masses: bool,
    contour_percentiles: list[float],
    contour_log10_values: list[float]|None,
    configured_Z_solar: float,
    colourmap: str|None
) -> None:

    # If not other plot types are specified, plot count bins
    if not (plot_metallicity or plot_last_halo_mass or plot_metal_weighted_last_halo_mass or plot_metal_weighted_last_halo_redshift):
        plot_hist = True

    split_files: int
    try:
        sections = input_filepath.rsplit(".", maxsplit = 1)
        int(sections[-1])
        input_filepath = sections[0]
        split_files = True
    except:
        split_files = not os.path.exists(input_filepath) and os.path.exists(f"{input_filepath}.0")
    if not split_files and not os.path.exists(input_filepath):
        Console.print_error(f"Unable to find file or distributed files matching the file " + ("name" if not split_files else "pattern") + f": {input_filepath}")
        Console.print_info("Terminating...")
        return

    # Read Contra header data
    with OutputReader(input_filepath if not split_files else f"{input_filepath}.0") as contra_reader:
        contra_header = contra_reader.read_header()
    if not contra_header.has_gas:
        Console.print_error("Contra data has no results for gas particles.")
        Console.print_info("Terminating...")
        return
    # This will get re-loaded, so free up the memory
    del contra_header

    Console.print_info("Locating search and simulation data sets.")

    if split_files:
        contra_data = DistributedOutputReader(input_filepath).read(include_stats = False, alternate_simulation_data_directory = snapshot_directory)
    else:
        contra_data = ContraData.load(input_filepath, include_stats = False, alternate_simulation_data_directory = snapshot_directory)

    Console.print_verbose_info(f"Successfully loaded data from {input_filepath}" + ("" if not split_files else ".*"))

    target_file_number: str = contra_data.find_file_number_from_redshift(target_redshift)
    snap: SnapshotBase = contra_data.get_snapshot(target_file_number)

    Console.print_info(f"Identified particle data at z={snap.redshift}.")
    if target_redshift >= 1.0 and target_redshift - snap.redshift > 0.5 or target_redshift < 1.0 and target_redshift - snap.redshift > 0.1:
        Console.print_warning(f"Attempted to find data at z={target_redshift} but only managed to retrive data for z={snap.redshift}.")

    assume_primordial_hydrogen_fraction: bool = snap.snipshot and use_number_density
    if assume_primordial_hydrogen_fraction:
        Console.print_warning(f"Elemental abundance data not avalible in snipshot data.\nAssuming primordial abundance of {PRIMORDIAL_H_ABUNDANCE}.")



    # Load data

    Console.print_info("Reading data.")

    gas_dataset: ParticleTypeDataset = cast(ParticleTypeDataset, contra_data.data[target_file_number].gas)

    # Create mask for:
    #     tracked particles (i.e. must have been in a halo at some point)
    #     non-zero metallicity
    #     (optional) particles currently in a halo

    located_particle_mask: np.ndarray = gas_dataset.halo_ids != -1

    #nonzero_halo_mass_mask: np.ndarray = gas_dataset.halo_masses > 0.0

    particle_metalicities: np.ndarray = snap.get_metalicities(ParticleType.gas).value

    data_mask: np.ndarray = located_particle_mask
    if metals_only:
        data_mask &= particle_metalicities > 0.0
    if ignore_halo_particles:
        data_mask &= gas_dataset.redshifts > snap.redshift
    n_particles: int = data_mask.sum()

    assert (~(gas_dataset.halo_masses[data_mask] > 0)).sum() == 0

    log10_last_halo_masses: np.ndarray = np.log10(gas_dataset.halo_masses[data_mask])
    last_halo_redshifts: np.ndarray = gas_dataset.redshifts[data_mask]

    if contours_use_all_particles:
        all_particle_masses = snap.get_masses(ParticleType.gas).to("Msun").value
        particle_masses = all_particle_masses[data_mask]
    else:
        particle_masses = snap.get_masses(ParticleType.gas).to("Msun").value[data_mask]

    particle_metalicities = particle_metalicities[data_mask]
    if use_mean:
        particle_metal_masses: np.ndarray = particle_masses * particle_metalicities

    log10_particle_temperatures: np.ndarray
    if contours_use_all_particles:
        all_log10_particle_temperatures = np.log10(snap.get_temperatures(ParticleType.gas).to("K").value)
        log10_particle_temperatures = all_log10_particle_temperatures[data_mask]
    else:
        log10_particle_temperatures = np.log10(snap.get_temperatures(ParticleType.gas).to("K").value[data_mask])

    particle_densities: np.ndarray
    if use_number_density:
        if contours_use_all_particles:
            all_particle_densities = snap.get_number_densities(ParticleType.gas, "Hydrogen", default_abundance = PRIMORDIAL_H_ABUNDANCE).to("cm**(-3)").value
            particle_densities = all_particle_densities[data_mask]
        else:
            particle_densities = snap.get_number_densities(ParticleType.gas, "Hydrogen").to("cm**(-3)").value[data_mask]
    else:
        if contours_use_all_particles:
            all_particle_densities = snap.get_densities(ParticleType.gas).to("Msun/Mpc**3").value
            particle_densities = all_particle_densities[data_mask]
        else:
            particle_densities = snap.get_densities(ParticleType.gas).to("Msun/Mpc**3").value[data_mask]

    particle_indexes: np.ndarray = np.arange(n_particles)



    Console.print_info("Converting units.")

    # Convert to solar
    log10_one_plus_particle_metalicities_solar = np.log10(1 + (particle_metalicities / configured_Z_solar))

    if min_colour_metalicity is not None:
        min_colour_metalicity_solar = min_colour_metalicity / configured_Z_solar
    if max_colour_metalicity is not None:
        max_colour_metalicity_solar = max_colour_metalicity / configured_Z_solar

    if min_colour_metalicity_solar is not None:
        min_colour_metalicity_plotted = np.log10(1 + min_colour_metalicity_solar)
    if max_colour_metalicity_solar is not None:
        max_colour_metalicity_plotted = np.log10(1 + max_colour_metalicity_solar)



    if not use_number_density:
        # Convert to overdensity

        mean_baryon_density: float = snap.calculate_comoving_critical_gas_density().to("Msun/Mpc**3").value

        log10_particle_overdensities: np.ndarray = np.log10(particle_densities / mean_baryon_density)

        if contours_use_all_particles:
            all_particle_densities = np.log10(all_particle_densities / mean_baryon_density)

        if min_density is not None:
            min_log10_overdensity = min_density / mean_baryon_density
        if max_density is not None:
            max_log10_overdensity = max_density / mean_baryon_density

    else:
        log10_particle_numberdensities: np.ndarray = np.log10(particle_densities)

        if contours_use_all_particles:
            all_particle_densities = np.log10(all_particle_densities)



    # Define the colour calculation functions

    Console.print_info("Defining hexbin colour calculations.")

    reduce_colour__count = None
    reduce_colour__metalicity = None
    reduce_colour__last_halo_mass = None
    reduce_colour__metal_weighted_last_halo_mass = None
    reduce_colour__metal_weighted_last_halo_redshift = None

    if plot_hist:
        reduce_colour__count = create_hexbin_log10_count()

    if plot_metallicity:
        if use_mean:
#            reduce_colour__metalicity = create_hexbin_weighted_mean(np.log10(particle_metalicities) - np.log10(configured_Z_solar), weights = particle_metal_masses)
            reduce_colour__metalicity = create_hexbin_log10_weighted_mean(particle_metalicities, weights = particle_masses, offset = -np.log10(configured_Z_solar))
            #TODO: is this mass or metal mass weighted and does the label need changing?!
        else:
            reduce_colour__metalicity = create_hexbin_median(log10_one_plus_particle_metalicities_solar)

    if plot_last_halo_mass:
        if use_mean:
            reduce_colour__last_halo_mass = create_hexbin_mean(log10_last_halo_masses)
        else:
            reduce_colour__last_halo_mass = create_hexbin_median(log10_last_halo_masses)

    if plot_metal_weighted_last_halo_mass:
        reduce_colour__metal_weighted_last_halo_mass = create_hexbin_weighted_mean(log10_last_halo_masses, weights = particle_metal_masses)

    if plot_metal_weighted_last_halo_redshift:
        reduce_colour__metal_weighted_last_halo_redshift = create_hexbin_weighted_mean(last_halo_redshifts, weights = particle_metal_masses)
        


    if show_contours:
        # Calculate contouring arguments

        Console.print_info("Calculating contours.")
        if contours_use_all_particles:
            Console.print_info("    Contours use all particles.")
        else:
            Console.print_info("    Contours use only tracked particles.")
        if contours_use_masses:
            Console.print_info("    Contours trace total particle mass.")
        else:
            Console.print_info("    Contours trace particle counts.")

        contour_densities: np.ndarray
        contour_temperatures: np.ndarray
        contour_weights: np.ndarray|None = None
        if contours_use_all_particles:
            contour_densities = all_particle_densities
            contour_temperatures = all_log10_particle_temperatures
            if contours_use_masses:
                contour_weights = all_particle_masses
        else:
            contour_densities = log10_particle_numberdensities if use_number_density else log10_particle_overdensities
            contour_temperatures = log10_particle_temperatures
            if contours_use_masses:
                contour_weights = particle_masses

        
        h, xedges, yedges = np.histogram2d(contour_densities, contour_temperatures, N_CONTOURING_BINS, weights = contour_weights)

        # To get the approximate hex value:
        #     hex_value = hist_value * hex_area / hist_area
        #               = hist_value * hist_to_hex_scale
        #hist_to_hex_scale = (N_PLOTTING_HEXES**2 / (2 * np.sqrt(3))) / ((xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))
        hex_width = (contour_densities.max() - contour_densities.min()) / N_PLOTTING_HEXES
        hist_to_hex_scale = (hex_width)**2 / ((xedges[1] - xedges[0]) * (yedges[1] - yedges[0]) * 2 * np.sqrt(3))

        if contour_log10_values is None:
            #total_contour_value = np.sum(contour_weights)
            #h /= total_contour_value
            check_values = h.reshape(((len(xedges) - 1) * (len(yedges) - 1),))
            contour_values = np.percentile(check_values[check_values != 0], contour_percentiles)
        else:
            # contour_log10_values is an input parameter scaled to be the approximate value of the hexes the contour overlays
            contour_values = 10**np.array(contour_log10_values, dtype = np.float64) / hist_to_hex_scale

        # Reported values are the log of the equivilant hexbin value and NOT the value of the 2D histogram used to create the contour!
        Console.print_info(f"Plotting contours at log10 sum({'M' if contours_use_masses else 'N'}) of: {', '.join([str(v) for v in np.log10(contour_values * hist_to_hex_scale)])}")
        Console.print_info(f"{', '.join([str(v) for v in np.log10(contour_values)])}")

        contour_args = (
            np.array(xedges[:-1] + ((xedges[1] - xedges[0])/2), dtype = np.float64),
            np.array(yedges[:-1] + ((yedges[1] - yedges[0])/2), dtype = np.float64),
            np.array(h.T, dtype = np.float64)
        )

        contour_kwargs = {
            "levels"     : np.array(contour_values, dtype = np.float64),
            "colors"     : "k",
            "alpha"      : 0.5,
            "linewidths" : 1,#0.7,
            "linestyles" : ["solid", "dotted"]
        }



    # Plot

    Console.print_info("Plotting.")

    plt.rcParams['font.size'] = 12

    n_subplots = int(plot_hist) + int(plot_metallicity) + int(plot_last_halo_mass) + int(plot_metal_weighted_last_halo_mass) + int(plot_metal_weighted_last_halo_redshift)
    seperate_plots = (n_subplots == 1) or not (stack_row or stack_column)

    current_subplot_index = 0
    if not seperate_plots:
        #fig, axes = plt.subplots(nrows = n_subplots if stack_column else 1, ncols = n_subplots if stack_row else 1, sharex = "row", sharey = "col", layout = "tight", figsize = (6 * (n_subplots if stack_row else 1), 5.25 * (n_subplots if stack_column else 1)))
        fig, axes = plt.subplots(
            nrows = n_subplots if stack_column else 1,
            ncols = n_subplots if stack_row else 1,
            sharex = "all",
            sharey = "all",
            layout = "tight",
            gridspec_kw = {"wspace": 0, "hspace": 0},
#            figsize = (6 * (n_subplots if stack_row else 1), 5.25 * (n_subplots if stack_column else 1))
            figsize = (6 * (n_subplots if stack_row else 1), 6 * (n_subplots if stack_column else 1))
        )

    plot_num: int = 0
    for plot_name, label, colour_reduction_function, min_colour, max_colour in zip(
        ("histogram",                             "metallicity",                                                                                                                   "last-halo-mass",                                                                                               "metal-weighted-last-halo-mass",                                                                                "metal-weighted-last-halo-redshift"),
#        ("${\\rm log_{10}}$ Number of Particles", ("Metal-mass Weighted Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ 1 + Z ($\\rm Z_{\\rm \\odot}$)", ("Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ $M_{\\rm 200}$ of last halo ($\\rm M_{\\rm \\odot}$)", "Metal-mass Weighted Mean ${\\rm log_{10}}$ $M_{\\rm 200}$ of last halo ($\\rm M_{\\rm \\odot}$)"),
        ("${\\rm log_{10}}$ Number of Particles", "${\\rm log_{10}}$ " + ("Metal-mass Weighted Mean" if use_mean else "Median") + " Z - ${\\rm log_{10}}$ $\\rm Z_{\\rm \\odot}$", ("Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ $M_{\\rm 200}$ of last halo ($\\rm M_{\\rm \\odot}$)", "Metal-mass Weighted Mean ${\\rm log_{10}}$ $M_{\\rm 200}$ of Last Halo ($\\rm M_{\\rm \\odot}$)",              "Metal-mass Weighted Mean $z$ of Last Halo Membership"),
        (reduce_colour__count,                    reduce_colour__metalicity,                                                                                                       reduce_colour__last_halo_mass,                                                                                  reduce_colour__metal_weighted_last_halo_mass,                                                                   reduce_colour__metal_weighted_last_halo_redshift),
        (min_colour_log_count,                    min_colour_metalicity_plotted,                                                                                                   min_colour_halo_mass,                                                                                           min_colour_halo_mass,                                                                                           min_colour_halo_redshift if min_colour_halo_redshift is not None else None),
        (max_colour_log_count,                    max_colour_metalicity_plotted,                                                                                                   max_colour_halo_mass,                                                                                           max_colour_halo_mass,                                                                                           max_colour_halo_redshift if max_colour_halo_redshift is not None else None)
    ):
        if not ((plot_name == "histogram" and plot_hist) or (plot_name == "metallicity" and plot_metallicity) or (plot_name == "last-halo-mass" and plot_last_halo_mass) or (plot_name == "metal-weighted-last-halo-mass" and plot_metal_weighted_last_halo_mass) or (plot_name == "metal-weighted-last-halo-redshift" and plot_metal_weighted_last_halo_redshift)):
            continue

        plot_num += 1
        Console.print_info(f"    #{plot_num}")

        if seperate_plots:
            fig = plt.figure(layout = "tight", figsize = (6, 5.25))
            axes = [fig.gca()]

        if colourmap is None:
            colourmap = tol_colors.LinearSegmentedColormap.from_list("custom-map", ["#125A56", "#FD9A44", "#A01813"])

        coloured_object = plot_hexbin(
            log10_particle_overdensities if not use_number_density else log10_particle_numberdensities,
            log10_particle_temperatures,
            colour_reduction_function,
            vmin = min_colour, vmax = max_colour,
            cmap = colourmap,
            gridsize = N_PLOTTING_HEXES,
            axis = axes[current_subplot_index]
        )
        fig.colorbar(
            coloured_object,
            ax = axes[current_subplot_index],
            location = "top" if stack_row else "right",
            label = label,
            extend =      "both"    if min_colour is not None and max_colour is not None
                     else "min"     if min_colour is not None
                     else "max"     if                            max_colour is not None
                     else "neither"
        )

        if show_contours:
            axes[current_subplot_index].contour(*contour_args, **contour_kwargs)

        xlims: tuple[float, float]
        if not use_number_density:
            xlims = axes[current_subplot_index].set_xlim((min_log10_overdensity, max_log10_overdensity))
        else:
            xlims = axes[current_subplot_index].set_xlim((min_log10_numberdensity, max_log10_numberdensity))
        ylims = axes[current_subplot_index].set_ylim((min_log10_temp, max_log10_temp))

#        if show_igm:#TODO: alter this region to be more appropriate for EAGLE/SWIFT data!!!!!!!!!!!!
#            axes[current_subplot_index].plot(
#                [xlims[0], 5.75, 5.75, 2.0, 2.0     ],
#                [7.0,      7.0,  4.5,  4.5, ylims[0]],
#                color = "black"
#            )
#            axes[current_subplot_index].set_xlim(xlims)
#            axes[current_subplot_index].set_ylim(ylims)
#            axes[current_subplot_index].text(
#                2.0, 6.0,
#                "IGM",
#                bbox = { "facecolor" : "lightgrey", "edgecolor" : "none" }
#            )

        if seperate_plots:
            axes[current_subplot_index].set_ylabel("${\\rm log_{10}}$ Temperature (${\\rm K}$)")
            if not use_number_density:
                axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ Overdensity = $\\rho$/<$\\rm \\rho$>")
            else:
                axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ $n_{\\rm H}$ ($\\rm cm^{-3}$)")
            if output_filepath is not None:
                Console.print_info("Saving image...", end = "")
                fig.savefig(output_filepath if n_subplots == 1 else (split_filename:=output_filepath.rsplit(".", maxsplit = 1))[0] + plot_name + "." + split_filename[1], dpi = 400)
                Console.print("done")
            else:
                Console.print_info("Rendering interactive window.")
                plt.show()
            plt.clf()
            plt.close()
        else:
            if stack_column or current_subplot_index == 0: # Y-axis --> Left-most plot of row or all
                axes[current_subplot_index].set_ylabel("${\\rm log_{10}}$ Temperature (${\\rm K}$)")
            if stack_row or current_subplot_index == n_subplots: # X-axis --> Bottom-most plot of column or all
                if not use_number_density:
                    axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ Overdensity = $\\rho$/<$\\rm \\rho$>")
                else:
                    axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ $n_{\\rm H}$ ($\\rm cm^{-3}$)")
            current_subplot_index += 1

    if not seperate_plots:
        if output_filepath is not None:
            Console.print_info("Saving image...", end = "")
            fig.savefig(output_filepath, dpi = 400)
            Console.print("done")
        else:
            Console.print_info("Rendering interactive window.")
            plt.show()
        plt.clf()
        plt.close()



    return



#    plt.figure(layout = "tight", figsize = (6, 5.25))
#
#    plt.hexbin(log10_particle_overdensities, log10_particle_temperatures, C = particle_indexes, reduce_C_function = reduce_colour__metalicity, gridsize = 500)
#    plt.colorbar(
#        label = ("Metal-mass Weighted Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ 1 + Z ($\\rm Z_{\\rm \\odot}$)",
#        extend =      "both"    if min_colour_metalicity_solar is not None and max_colour_metalicity_solar is not None
#                 else "min"     if min_colour_metalicity_solar is not None
#                 else "max"     if                                             max_colour_metalicity_solar is not None
#                 else "neither"
#    )
#
#    plt.xlabel("${\\rm log_{10}}$ Overdensity = $\\rho$/<$\\rm \\rho$>")
#    plt.ylabel("${\\rm log_{10}}$ Temperature (${\\rm K}$)")
#    plt.xlim((min_log10_overdensity, max_log10_overdensity))
#    plt.ylim((min_log10_temp, max_log10_temp))
#
#    if output_filepath is not None:
#        plt.savefig(output_filepath, dpi = 100)
#    else:
#        plt.show()




    plt.hexbin(log10_particle_overdensities, log10_particle_temperatures, C = particle_indexes, reduce_C_function = reduce_colour__last_halo_mass,
               gridsize = 500, cmap = "viridis", vmin = min_colour_halo_mass, vmax = max_colour_halo_mass)
    plt.colorbar(
#        label = ("Metal-mass Weighted Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ Last Halo Mass ($\\rm M_{\\rm \\odot}$)",
        label = ("Metal-mass Weighted Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ $M_{\\rm 200}$ of last halo ($\\rm M_{\\rm \\odot}$)",
        extend =      "both"    if min_colour_halo_mass is not None and max_colour_halo_mass is not None
                 else "min"     if min_colour_halo_mass is not None
                 else "max"     if                                      max_colour_halo_mass is not None
                 else "neither"
    )

    xlims = plt.xlim()
    ylims = plt.ylim()
    plt.plot(
        [xlims[0], 5.75, 5.75, 2.0, 2.0     ],
        [7.0,      7.0,  4.5,  4.5, ylims[0]],
        color = "black"
    )
    plt.xlim(xlims)
    plt.ylim(ylims)

    plt.text(
        2.0, 6.0,
        "IGM",
        bbox = { "facecolor" : "lightgrey", "edgecolor" : "none" }
    )

#    plt.xlabel("${\\rm log_{10}}$ Overdensity = $\\rho$/<$\\rm \\rho$>")
    plt.xlabel("${\\rm log_{10}}$ $\\rho$/<$\\rm \\rho$>")
    plt.ylabel("${\\rm log_{10}}$ Temperature (${\\rm K}$)")
    plt.xlim((min_log10_overdensity, max_log10_overdensity))
    plt.ylim((min_log10_temp, max_log10_temp))

    if output_filepath is not None:
        plt.savefig(output_filepath, dpi = 400)
    else:
        plt.show()
