# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from ..plotting._hexbin import create_hexbin_sum, plot_hexbin, create_hexbin_count, create_hexbin_log10_count, create_hexbin_mean, create_hexbin_log10_mean, create_hexbin_weighted_mean, create_hexbin_log10_weighted_mean, create_hexbin_median, test_function__create_hexbin_log10_weighted_mean

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
import h5py as h5

from astro_sph_tools import ParticleType
from astro_sph_tools.io.data_structures import SnapshotBase
from astro_sph_tools.io.EAGLE import FileTreeScraper_EAGLE, SnapshotEAGLE

# 0.0134
Z_SOLAR = 0.012663729 # Z_sun (from EAGLE L50N752 -> Constants -> Z_Solar)
PRIMORDIAL_H_ABUNDANCE = 0.752 # Mass fraction of H (from EAGLE L50N752 -> Parameters -> ChemicalElements -> InitAbundance_Hydrogen)
N_CONTOURING_BINS = 200
N_PLOTTING_HEXES = 200

def main():
    ScriptWrapper(
        command = "contra-plot-environment",
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
            ScriptWrapper.OptionalParam[str](
                name = "data",
                short_name = "i",
                sets_param = "input_directory",
                default_value = "./",
                description = "Input simulation data directory. Defaults to the current working directory."
            ),
            ScriptWrapper.Flag(
                name = "snipshot",
                sets_param = "use_snipshots",
                description = "Use particle data from snipshots.\nWARNING: Some options may not be supported or may make assumptions where data is not avalible (e.g. elemental abundances)."
            ),
            ScriptWrapper.OptionalParam[list[str]](
                name = "ignore-files",
                sets_param = "skip_file_numbers",
                conversion_function = ScriptWrapper.make_list_converter(","),
                default_value = [],
                description = "Snapshot/snipshot numbers to be ignored. This can be used in the case of corrupted files.\nUse a comma seperated list."
            ),
            ScriptWrapper.Flag(
                name = "metals-only",
                description = "Ignore particles with no metal content."
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
                name = "colour-enrichment-redshift",
                sets_param = "plot_z_Z",
                description = "Colour by the averaged redshift at which metal enrichment occoured."
            ),
            ScriptWrapper.Flag(
                name = "colour-halo-distance",
                sets_param = "plot_distance_to_halo",
                description = "Colour by the co-moving distance to the nearest halo.",
                requirements = ["nearest-halo-data"]
            ),
            ScriptWrapper.Flag(
                name = "colour-halo-distance-fraction",
                sets_param = "plot_fractional_distance_to_halo",
                description = "Colour by the distance to the nearest halo as a fraction of the halo's R_200.",
                requirements = ["nearest-halo-data"]
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



            ScriptWrapper.Flag(
                name = "zoom",
                description = "Show only a limited region of the rendered plot.",
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-density-zoom",
                description = "Minimum density (in Msun/Mpc^3) to display when zoomed in.",
                conversion_function = float,
                conflicts = ["min-overdensity-zoom", "use-number-density"],
                requirements = ["zoom"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-density-zoom",
                description = "Maximum density (in Msun/Mpc^3) to display when zoomed in.",
                conversion_function = float,
                conflicts = ["max-overdensity-zoom", "use-number-density"],
                requirements = ["zoom"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-overdensity-zoom",
                sets_param = "min_log10_overdensity_zoom",
                description = "Minimum (log10) overdensity to display when zoomed in.",
                conversion_function = float,
                conflicts = ["min-density-zoom", "use-number-density"],
                requirements = ["zoom"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-overdensity-zoom",
                sets_param = "max_log10_overdensity_zoom",
                description = "Maximum (log10) overdensity to display when zoomed in.",
                conversion_function = float,
                conflicts = ["max-density-zoom", "use-number-density"],
                requirements = ["zoom"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-numberdensity-zoom",
                sets_param = "min_log10_numberdensity_zoom",
                description = "Minimum (log10) hydrogen number density to display when zoomed in.",
                conversion_function = float,
                requirements = ["use-number-density", "zoom"],
                conflicts = ["min-density-zoom"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-numberdensity-zoom",
                sets_param = "max_log10_numberdensity_zoom",
                description = "Maximum (log10) hydrogen number density to display when zoomed in.",
                conversion_function = float,
                requirements = ["use-number-density", "zoom"],
                conflicts = ["max-density-zoom"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-temp-zoom",
                sets_param = "min_log10_temp_zoom",
                description = "Minimum temperature (in log10 K) to display when zoomed in.",
                conversion_function = float,
                requirements = ["zoom"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-temp-zoom",
                sets_param = "max_log10_temp_zoom",
                description = "Maximum temperature (in log10 K) to display when zoomed in.",
                conversion_function = float,
                requirements = ["zoom"]
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
                name = "min-colour-redshift",
                description = "Minimum redshift, below which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-redshift",
                description = "Maximum redshift, above which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-colour-halo-distance",
                description = "Minimum co-moving distance, below which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-halo-distance",
                description = "Maximum co-moving distance, above which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-colour-halo-distance-fraction",
                description = "Minimum fractional distance, below which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-colour-halo-distance-fraction",
                description = "Maximum fractional distance, above which the colour will be uniform.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                name = "number-of-hexes",
                description = f"Number of hexes to use. Defaults to {N_PLOTTING_HEXES}",
                conversion_function = int,
                default_value = N_PLOTTING_HEXES
            ),
#            ScriptWrapper.Flag(
#                name = "mean",
#                sets_param = "use_mean",
#                description = "Use mean to determine colour values instead of the median."
#            ),
            ScriptWrapper.Flag(
                name = "median",
                sets_param = "use_median",
                description = "Use median to determine colour values instead of the mean."
            ),
            ScriptWrapper.Flag(
                name = "metal-mass-weights",
                sets_param = "weight_using_metal_mass",
                description = "Use the metal mass of particles as oposed to the total mass when weighting particles in mean calculations.\nOnly valid for appropriate datasets.",
                conflicts = ["median"]
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
            ScriptWrapper.Flag(
                name = "show-contours",
                description = "Draw contours showing the distribution of included particles (shows binned number of particles by default)."
            ),
            ScriptWrapper.Flag(
                name = "contours-use-all-particles",
                description = "Extend the number of particles used to generate the contours to include all particles.\nThis will make use of all gas particles at the target redshift.\nThis will only have an effect when limiting to particles with nonzero metallicity.",
                requirements = ["show-contours"]
            ),
            ScriptWrapper.Flag(
                name = "contours-use-masses",
                description = "Use contours based on the total mass in each bin as opposed to the number of particles.",
                requirements = ["show-contours"],
                conflicts = ["contours-use-metal-masses"]
            ),
            ScriptWrapper.Flag(
                name = "contours-use-metal-masses",
                description = "Use contours based on the total metal mass in each bin as opposed to the number of particles.",
                requirements = ["show-contours"],
                conflicts = ["contours-use-masses"]
            ),
#            ScriptWrapper.OptionalParam[list[float]](
#                name = "contour-percentiles",
#                default_value = [10, 25, 50, 75, 90],
#                description = "Percentiles at which to plot contours, as a comma seperated list (in ascending order).\nDefaults to: 10%, 25%, 50%, 75% and 90%.",
#                conversion_function = ScriptWrapper.make_list_converter(",", float),
#                conflicts = ["contour-values"]
#            ),
#            ScriptWrapper.OptionalParam[list[float]](
#                name = "contour-values",
#                sets_param = "contour_log10_values",
#                description = "log10 bin density values at which to plot contours, as a comma seperated list (in ascending order).\nNote, these values will be specific to either number density or mass density contours!",
#                conversion_function = ScriptWrapper.make_list_converter(",", float),
#                conflicts = ["contour-percentiles"]
#            ),
            ScriptWrapper.OptionalParam[list[float]](
                name = "contour-percentiles",
                default_value = [10, 25, 50, 75, 90],
                description = "Percentiles at which to plot contours in log10 space, as a comma seperated list (in ascending order).\nDefaults to: 10%, 25%, 50%, 75% and 90%.\nHas no effect if specifying --contour-values.",
                conversion_function = ScriptWrapper.make_list_converter(",", float),
                conflicts = ["contour-values"]
            ),
            ScriptWrapper.OptionalParam[list[float]](
                name = "contour-values",
                sets_param = "contour_log10_values",
                description = "log10 bin density values at which to plot contours, as a comma seperated list (in ascending order).\nNote, these values will be specific to either particle number or mass contours!\nOverrides --contour-percentiles.",
                conversion_function = ScriptWrapper.make_list_converter(",", float),
                conflicts = ["contour-percentiles"]
            ),
            ScriptWrapper.Flag(
                name = "alpha-number",
                description = "Apply alpha (opacity) values to hexes based on the number of particles in the hex.",
                conflicts = ["alpha-masses", "alpha-metal-masses"]
            ),
            ScriptWrapper.Flag(
                name = "alpha-masses",
                description = "Apply alpha (opacity) values to hexes based on the particle mass in the hex.",
                conflicts = ["alpha-number", "alpha-metal-masses"]
            ),
            ScriptWrapper.Flag(
                name = "alpha-metal-masses",
                description = "Apply alpha (opacity) values to hexes based on the particle metal mass in the hex.",
                conflicts = ["alpha-number", "alpha-masses"]
            ),
            ScriptWrapper.Flag(
                name = "alpha-log10",
                description = "Apply log10(1+x) to the alpha value of each hex and rescale."
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
            ),
            ScriptWrapper.OptionalParam[str](
                name = "nearest-halo-data",
                description = "File containing nearest halo data."
            ),
            ScriptWrapper.OptionalParam[float](
                name = "minimum-nearest-halo-mass",
                description = "Minimum halo mass to use when reading halo distance data.",
                conversion_function = float,
                requirements = ["nearest-halo-data"]
            )
        )
    ).run_with_async(__main)

async def __main(
    target_redshift: float,
    is_EAGLE: bool,
    is_SWIFT: bool,
    input_directory: str,
    use_snipshots: bool,
    skip_file_numbers: list[str],
    metals_only: bool,
    use_number_density: bool,
    plot_hist: bool,
    plot_metallicity: bool,
    plot_z_Z: bool,
    plot_distance_to_halo: bool,
    plot_fractional_distance_to_halo: bool,
    output_filepath: str | None,
    min_density: float | None,
    max_density: float | None,
    min_log10_overdensity: float | None,
    max_log10_overdensity: float | None,
    min_log10_numberdensity: float | None,
    max_log10_numberdensity: float | None,
    min_log10_temp: float | None,
    max_log10_temp: float | None,
    min_density_zoom: float | None,
    max_density_zoom: float | None,





    zoom: bool,#TODO: use normal limits (+a bit extra) to mask data that actually needs plotting
    min_log10_overdensity_zoom: float|None,
    max_log10_overdensity_zoom: float|None,
    min_log10_numberdensity_zoom: float|None,
    max_log10_numberdensity_zoom: float|None,
    min_log10_temp_zoom: float|None,
    max_log10_temp_zoom: float|None,





    min_colour_log_count: float | None,
    max_colour_log_count: float | None,
    min_colour_metalicity: float | None,
    max_colour_metalicity: float | None,
    min_colour_metalicity_solar: float | None,
    max_colour_metalicity_solar: float | None,
    min_colour_metalicity_plotted: float | None,
    max_colour_metalicity_plotted: float | None,
    min_colour_redshift: float | None,
    max_colour_redshift: float | None,
    min_colour_halo_distance: float | None,
    max_colour_halo_distance: float | None,
    min_colour_halo_distance_fraction: float | None,
    max_colour_halo_distance_fraction: float | None,
#    use_mean: bool,
    use_median: bool,
    weight_using_metal_mass: bool,
    number_of_hexes: int,
    stack_row: bool,
    stack_column: bool,
    show_contours: bool,
    contours_use_all_particles: bool,
    contours_use_masses: bool,
    contours_use_metal_masses: bool,
    contour_percentiles: list[float],
    contour_log10_values: list[float]|None,
    alpha_number: bool,
    alpha_masses: bool,
    alpha_metal_masses: bool,
    alpha_log10: bool,
    configured_Z_solar: float,
    colourmap: str|None,
    nearest_halo_data: str|None,
    minimum_nearest_halo_mass: float|None
) -> None:
    
    use_mean = not use_median

    # Validate sim type
    if not (is_EAGLE or is_SWIFT):
        Console.print_error("Must specify either EAGLE or SWIFT simulation type.")
        return
    else:
        if is_EAGLE:
            Console.print_verbose_info("Snapshot type: EAGLE")
        elif is_SWIFT:
            Console.print_verbose_info("Snapshot type: SWIFT")
            raise NotImplementedError()

    # If not other plot types are specified, plot count bins
    if not (plot_metallicity or plot_z_Z or plot_distance_to_halo or plot_fractional_distance_to_halo):
        plot_hist = True

    # If not other plot types are specified, plot count bins
    if weight_using_metal_mass and not (plot_z_Z or plot_distance_to_halo or plot_fractional_distance_to_halo):
        Console.print_warning("--metal-mass-weights is set but no plots that were requested support configurable weights!")



    # Find the requested data file(s)
    sim_data = FileTreeScraper_EAGLE(input_directory, skip_file_numbers, skip_file_numbers)
    if use_snipshots:
        selected_data_file_type_info = sim_data.snipshots
        if len(selected_data_file_type_info) == 0:
            Console.print_warning(f"No snipshots avalible for this simulation dataset.")
            Console.print_info("Terminating...")
            return
    else:
        selected_data_file_type_info = sim_data.snapshots
        if len(selected_data_file_type_info) == 0:
            Console.print_warning(f"No snapshots avalible for this simulation dataset.")
            Console.print_info("Terminating...")
            return
    target_file_number: str = selected_data_file_type_info.find_file_number_from_redshift(target_redshift)
    snap: SnapshotBase = sim_data.snapshots.get_by_number(target_file_number).load()

    Console.print_info(f"Identified particle data at z={snap.redshift}.")
    if target_redshift >= 1.0 and target_redshift - snap.redshift > 0.5 or target_redshift < 1.0 and target_redshift - snap.redshift > 0.1:
        Console.print_warning(f"Attempted to find data at z={target_redshift} but only managed to retrive data for z={snap.redshift}.")

    assume_primordial_hydrogen_fraction: bool = snap.snipshot and use_number_density
    if assume_primordial_hydrogen_fraction:
        Console.print_warning(f"Elemental abundance data not avalible in snipshot data.\nAssuming primordial abundance of {PRIMORDIAL_H_ABUNDANCE}.")

    if plot_distance_to_halo or plot_fractional_distance_to_halo:
        particle_distance_to_nearest_halo: np.ndarray
        particle_distance_fraction_to_nearest_halo: np.ndarray
        if nearest_halo_data is not None:
            if not os.path.exists(nearest_halo_data):
                raise FileNotFoundError(f"Unable to find file at \"{nearest_halo_data}\".")
            data_path = f"redshift_{snap.redshift}/minimum_halo_mass_limited/{minimum_nearest_halo_mass:.2f}" if minimum_nearest_halo_mass is not None else f"redshift_{snap.redshift}"
            with h5.File(nearest_halo_data, "r") as file:
                particle_distance_to_nearest_halo = file[f"{data_path}/halo_comoving_distance"][:]
                if plot_fractional_distance_to_halo:
                    particle_distance_fraction_to_nearest_halo = particle_distance_to_nearest_halo / file[f"{data_path}/halo_comoving_radius"][:]
                if not plot_distance_to_halo:
                    del particle_distance_to_nearest_halo



    # Load data

    Console.print_info("Reading data.")

    log10_particle_temperatures: np.ndarray = np.log10(snap.get_temperatures(ParticleType.gas).to("K").value)

    particle_densities: np.ndarray
    if use_number_density:
        particle_densities = snap.get_number_densities(ParticleType.gas, "Hydrogen", default_abundance = PRIMORDIAL_H_ABUNDANCE).to("cm**(-3)").value
    else:
        particle_densities = snap.get_densities(ParticleType.gas).to("Msun/Mpc**3").value

    if plot_metallicity or metals_only or use_mean or contours_use_metal_masses:
        particle_metalicities: np.ndarray = snap.get_metallicities(ParticleType.gas).value

    if use_mean or contours_use_metal_masses:
        particle_masses = snap.get_masses(ParticleType.gas).to("Msun").value
        if contours_use_metal_masses or plot_z_Z or plot_distance_to_halo or plot_fractional_distance_to_halo:
            particle_metal_masses: np.ndarray = particle_masses * particle_metalicities

    if plot_z_Z:
        particle_z_Z = snap.get_mean_enrichment_redshift(ParticleType.gas).value



    Console.print_info("Converting units.")

    # Convert to solar

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

        if min_density is not None:
            min_log10_overdensity = min_density / mean_baryon_density
        if max_density is not None:
            max_log10_overdensity = max_density / mean_baryon_density

        if min_density_zoom is not None:
            min_log10_overdensity_zoom = min_density_zoom / mean_baryon_density
        if max_density_zoom is not None:
            max_log10_overdensity_zoom = max_density_zoom / mean_baryon_density

    else:
        log10_particle_numberdensities: np.ndarray = np.log10(particle_densities)



    # Create a mask for the data
    #TODO: always remove particles with 0 mass?

    data_mask: np.ndarray|slice = np.full(shape = log10_particle_temperatures.shape, fill_value = True, dtype = np.bool_)
    data_mask_set: bool = False
    if metals_only:
        Console.print("1")
        data_mask = data_mask & (particle_metalicities > 0.0)
        data_mask_set = True
    if min_log10_temp is not None or max_log10_temp:
        Console.print("2")
        data_range = (max_log10_temp if max_log10_temp is not None else log10_particle_temperatures.max()) - (min_log10_temp if min_log10_temp is not None else log10_particle_temperatures.min())
        if min_log10_temp is not None:
            data_mask = data_mask & (log10_particle_temperatures >= min_log10_temp - (data_range * 0.05))
            data_mask_set = True
        if max_log10_temp is not None:
            data_mask = data_mask & (log10_particle_temperatures <= max_log10_temp + (data_range * 0.05))
            data_mask_set = True
    if use_number_density:
        if min_log10_numberdensity is not None or max_log10_numberdensity:
            Console.print("3")
            data_range = (max_log10_numberdensity if max_log10_numberdensity is not None else log10_particle_numberdensities.max()) - (min_log10_numberdensity if min_log10_numberdensity is not None else log10_particle_numberdensities.min())
            if min_log10_numberdensity is not None:
                data_mask = data_mask & (log10_particle_numberdensities >= min_log10_numberdensity - (data_range * 0.05))
                data_mask_set = True
            if max_log10_numberdensity is not None:
                data_mask = data_mask & (log10_particle_numberdensities <= max_log10_numberdensity + (data_range * 0.05))
                data_mask_set = True
    else:
        if min_log10_overdensity is not None or max_log10_overdensity:
            Console.print("4")
            data_range = (max_log10_overdensity if max_log10_overdensity is not None else log10_particle_overdensities.max()) - (min_log10_overdensity if min_log10_overdensity is not None else log10_particle_overdensities.min())
            if min_log10_overdensity is not None:
                data_mask = data_mask & (log10_particle_overdensities >= min_log10_overdensity - (data_range * 0.05))
                data_mask_set = True
            if max_log10_overdensity is not None:
                data_mask = data_mask & (log10_particle_overdensities <= max_log10_overdensity + (data_range * 0.05))
                data_mask_set = True
    if not data_mask_set or data_mask.sum() == data_mask.shape[0]:
        Console.print("5")
        data_mask = slice(None, None, None)



    # Define the colour calculation functions

    Console.print_info("Defining hexbin colour calculations.")

    reduce_colour__count = None
    reduce_colour__metalicity = None
    reduce_colour__z_Z = None
    reduce_colour__halo_distance = None
    reduce_colour__halo_distance_fractional = None

    if plot_hist:
        reduce_colour__count = create_hexbin_log10_count()
        #reduce_colour__count = create_hexbin_count()

    if plot_metallicity:
        if use_mean:
#            reduce_colour__metalicity = create_hexbin_weighted_mean(np.log10(particle_metalicities[data_mask]) - np.log10(configured_Z_solar), weights = particle_masses[data_mask])
            reduce_colour__metalicity = create_hexbin_log10_weighted_mean(particle_metalicities[data_mask], weights = particle_masses[data_mask], offset = -np.log10(configured_Z_solar))
            #Console.print_debug(f"Z_sun = {configured_Z_solar}")
            #reduce_colour__metalicity = test_function__create_hexbin_log10_weighted_mean(particle_metalicities[data_mask], weights = particle_masses[data_mask], offset = configured_Z_solar)
        else:
            reduce_colour__metalicity = create_hexbin_median(np.log10(1 + (particle_metalicities[data_mask] / configured_Z_solar)))

    if plot_z_Z:
        if use_mean:
            reduce_colour__z_Z = create_hexbin_weighted_mean(particle_z_Z[data_mask], weights = particle_metal_masses[data_mask] if weight_using_metal_mass else particle_masses[data_mask])
        else:
            reduce_colour__z_Z = create_hexbin_median(particle_z_Z[data_mask])

    if plot_distance_to_halo:
        if use_mean:
            reduce_colour__halo_distance = create_hexbin_log10_weighted_mean(particle_distance_to_nearest_halo[data_mask], weights = particle_metal_masses[data_mask] if weight_using_metal_mass else particle_masses[data_mask])
        else:
            reduce_colour__halo_distance = create_hexbin_median(particle_distance_to_nearest_halo[data_mask])

    if plot_fractional_distance_to_halo:
        if use_mean:
            reduce_colour__halo_distance_fractional = create_hexbin_log10_weighted_mean(particle_distance_fraction_to_nearest_halo[data_mask], weights = particle_metal_masses[data_mask] if weight_using_metal_mass else particle_masses[data_mask])
        else:
            reduce_colour__halo_distance_fractional = create_hexbin_median(particle_distance_fraction_to_nearest_halo[data_mask])
        


    if show_contours:
        # Calculate contouring arguments

        Console.print_info("Calculating contours.")
        if contours_use_all_particles or not metals_only:
            Console.print_info("    Contours use all particles.")
        else:
            Console.print_info("    Contours use only particles with nonzero metallicity.")
        if contours_use_masses:
            Console.print_info("    Contours trace total particle mass.")
        elif contours_use_metal_masses:
            Console.print_info("    Contours trace total particle metal mass.")
        else:
            Console.print_info("    Contours trace particle counts.")

        contour_densities: np.ndarray
        contour_temperatures: np.ndarray
        contour_weights: np.ndarray|None = None
        if use_number_density:
            contour_densities = log10_particle_numberdensities
        else:
            contour_densities = log10_particle_overdensities
        contour_temperatures = log10_particle_temperatures
        if contours_use_masses:
            contour_weights = particle_masses
        elif contours_use_metal_masses:
            contour_weights = particle_metal_masses
        if not contours_use_all_particles and metals_only:
            contour_densities    =    contour_densities[data_mask]
            contour_temperatures = contour_temperatures[data_mask]
            contour_weights      =      contour_weights[data_mask]

        # Calculate the value in each bin
        h, xedges, yedges = np.histogram2d(contour_densities, contour_temperatures, N_CONTOURING_BINS, weights = contour_weights)

        # Get information about the dimensions of the bins and scale for density
        bin_width: float = (contour_densities.max() - contour_densities.min()) / N_CONTOURING_BINS
        bin_height: float = (contour_temperatures.max() - contour_temperatures.min()) / N_CONTOURING_BINS
        bin_density_conversion: float = (bin_width * bin_height * snap.get_total_mass(ParticleType.dark_matter).to("Msun").value)**-1 # 1 / area
        h_density = h * bin_density_conversion

        if contour_log10_values is None:
            check_values = h_density.reshape(((len(xedges) - 1) * (len(yedges) - 1),))
            contour_values = np.percentile(check_values[check_values != 0], contour_percentiles)
        else:
            # contour_log10_values is an input parameter scaled to be the approximate value of the hexes the contour overlays
            contour_values = 10**np.array(contour_log10_values, dtype = np.float64)



#        h, xedges, yedges = np.histogram2d(contour_densities, contour_temperatures, N_CONTOURING_BINS, weights = contour_weights)
#
#        # To get the approximate hex value:
#        #     hex_value = hist_value * hex_area / hist_area
#        #               = hist_value * hist_to_hex_scale
#        #hist_to_hex_scale = (number_of_hexes**2 / (2 * np.sqrt(3))) / ((xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))
#        hex_width = (contour_densities.max() - contour_densities.min()) / number_of_hexes
#        hist_to_hex_scale = (hex_width)**2 / ((xedges[1] - xedges[0]) * (yedges[1] - yedges[0]) * 2 * np.sqrt(3))
#
#        if contour_log10_values is None:
#            #total_contour_value = np.sum(contour_weights)
#            #h /= total_contour_value
#            check_values = h.reshape(((len(xedges) - 1) * (len(yedges) - 1),))
#            contour_values = np.percentile(check_values[check_values != 0], contour_percentiles)
#        else:
#            # contour_log10_values is an input parameter scaled to be the approximate value of the hexes the contour overlays
#            contour_values = 10**np.array(contour_log10_values, dtype = np.float64) / hist_to_hex_scale



        # Reported values are the log of the equivilant hexbin value and NOT the value of the 2D histogram used to create the contour!
        Console.print_info(f"Plotting contours at log10{{ sum({'M' if contours_use_masses else 'N'}) [{'Msun ' if contours_use_masses else ''}{'log10(cm^3)' if use_number_density else 'dex^-1'} log10(K)^-1] }} of: {', '.join([str(v) for v in np.log10(contour_values)])}")
        #Console.print_info(f"Plotting contours at log10 sum({'M' if contours_use_masses else 'N'}) of: {', '.join([str(v) for v in np.log10(contour_values * hist_to_hex_scale)])}")
        #Console.print_info(f"{', '.join([str(v) for v in np.log10(contour_values)])}")

        contour_args = (
            np.array(xedges[:-1] + ((xedges[1] - xedges[0])/2), dtype = np.float64),
            np.array(yedges[:-1] + ((yedges[1] - yedges[0])/2), dtype = np.float64),
            np.array(h_density.T, dtype = np.float64)
        )

        contour_kwargs = {
            "levels"     : np.array(contour_values, dtype = np.float64),
            "colors"     : "k",
            "alpha"      : 0.5,
            "linewidths" : 1,#0.7,
            "linestyles" : ["solid", "dotted"]
        }



    # Calculate alpha values

    if alpha_number or alpha_masses or alpha_metal_masses:
        Console.print_info("Calculating alpha values.")

        hex_alpha: np.ndarray = plot_hexbin(
            log10_particle_overdensities[data_mask] if not use_number_density else log10_particle_numberdensities[data_mask],
            log10_particle_temperatures[data_mask],
            colour_function = create_hexbin_count() if alpha_number else create_hexbin_sum(particle_masses[data_mask] if alpha_masses else particle_metal_masses[data_mask]),
            gridsize = number_of_hexes
        ).get_array()
        plt.clf()
        plt.close()

    #    import pickle
    #    with open("hex-masses.pickle", "wb") as file:
    #        pickle.dump(hex_alpha, file)
    #    exit()

        if alpha_log10:
            Console.print_info("Using logarithmic alphaing.")
            hex_alpha = np.log10(hex_alpha + 1)
        hex_alpha = hex_alpha / hex_alpha.max()

    else:
        hex_alpha = None



    # Plot

    Console.print_info("Plotting.")

    plt.rcParams['font.size'] = 12

    n_subplots = int(plot_hist) + int(plot_metallicity) + int(plot_z_Z) + int(plot_distance_to_halo) + int(plot_fractional_distance_to_halo)
    seperate_plots = (n_subplots == 1) or not (stack_row or stack_column)

    current_subplot_index = 0
    if not seperate_plots:
        fig, axes = plt.subplots(
            nrows = n_subplots if stack_column else 1,
            ncols = n_subplots if stack_row else 1,
            sharex = "all",
            sharey = "all",
            layout = "tight",
            gridspec_kw = {"wspace": 0, "hspace": 0},
            figsize = (6 * (n_subplots if stack_row else 1), 6 * (n_subplots if stack_column else 1))
        )

    plot_num: int = 0
    for plot_name, label, colour_reduction_function, min_colour, max_colour in zip(
        ("histogram",                             "metallicity",                                                                                                             "mean-enrichment-redshift",                                               "distance-to-halo",                                                                          "scaled-distance-to-halo"                                                ),
        ("${\\rm log_{10}}$ Number of Particles", "${\\rm log_{10}}$ " + ("Mass Weighted Mean" if use_mean else "Median") + " Z - ${\\rm log_{10}}$ $\\rm Z_{\\rm \\odot}$", ("Metal Mass Weighted Mean" if use_mean else "Median") + " $z_{\\rm Z}$", ("${\\rm log_{10}}$ Metal Mass Weighted Mean" if use_mean else "Median") + " Distance to Nearest Halo (cMpc)", ("${\\rm log_{10}}$ Metal Mass Weighted Mean" if use_mean else "Median") + " Distance to Nearest Halo ($\\rm R_{200}$)" ),
        (reduce_colour__count,                    reduce_colour__metalicity,                                                                                                 reduce_colour__z_Z,                                                       reduce_colour__halo_distance,                                                                reduce_colour__halo_distance_fractional                                  ),
        (min_colour_log_count,                    min_colour_metalicity_plotted,                                                                                             min_colour_redshift,                                                      min_colour_halo_distance,                                                                    min_colour_halo_distance_fraction                                        ),
        (max_colour_log_count,                    max_colour_metalicity_plotted,                                                                                             max_colour_redshift,                                                      max_colour_halo_distance,                                                                    max_colour_halo_distance_fraction                                        )
    ):
        if not ((plot_name == "histogram" and plot_hist) or (plot_name == "metallicity" and plot_metallicity) or (plot_name == "mean-enrichment-redshift" and plot_z_Z) or (plot_name == "distance-to-halo" and plot_distance_to_halo) or (plot_name == "scaled-distance-to-halo" and plot_fractional_distance_to_halo)):
            continue

        plot_num += 1
        Console.print_info(f"    #{plot_num}")

        if seperate_plots:
            fig = plt.figure(layout = "tight", figsize = (6, 5.25))
            axes = [fig.gca()]

        if colourmap is None:
            colourmap = tol_colors.LinearSegmentedColormap.from_list("custom-map", ["#125A56", "#FD9A44", "#A01813"])
            #colourmap = tol_colors.LinearSegmentedColormap.from_list("custom-map", ["#00BEC1", "#FD9A44", "#A01813"])
            #colourmap = tol_colors.LinearSegmentedColormap.from_list("custom-map", ["#5DA899", "#94CBEC", "#DCCD7D", "#C26A77"])
            #colourmap = tol_colors.LinearSegmentedColormap.from_list("custom-map", ["#009e73", "#0072b2", "#56b4e9", "#f0e442", "#e69f00", "#d55e00"])

        coloured_object = plot_hexbin(
            log10_particle_overdensities[data_mask] if not use_number_density else log10_particle_numberdensities[data_mask],
            log10_particle_temperatures[data_mask],
            colour_reduction_function,
            vmin = min_colour, vmax = max_colour,
            cmap = colourmap,
            alpha_values = hex_alpha,
            gridsize = number_of_hexes,
            axis = axes[current_subplot_index]
        )
        del colour_reduction_function # Memory optimisation as this function is holding a reference to the data arrays which are no longer needed after the call is completed
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
            xlims = axes[current_subplot_index].set_xlim(
                (
                    min_log10_overdensity_zoom if zoom and min_log10_overdensity_zoom is not None else min_log10_overdensity,
                    max_log10_overdensity_zoom if zoom and max_log10_overdensity_zoom is not None else max_log10_overdensity
                )
            )
        else:
            xlims = axes[current_subplot_index].set_xlim(
                (
                    min_log10_numberdensity_zoom if zoom and min_log10_numberdensity_zoom is not None else min_log10_numberdensity,
                    max_log10_numberdensity_zoom if zoom and max_log10_numberdensity_zoom is not None else max_log10_numberdensity
                )
            )
        ylims = axes[current_subplot_index].set_ylim(
            (
                min_log10_temp_zoom if zoom and min_log10_temp_zoom is not None else min_log10_temp,
                max_log10_temp_zoom if zoom and max_log10_temp_zoom is not None else max_log10_temp
            )
        )

        if seperate_plots:
            axes[current_subplot_index].set_ylabel("${\\rm log_{10}}$ Temperature (${\\rm K}$)")
            if not use_number_density:
                axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ $\\rho$/<$\\rm \\rho$>")
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
                    axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ $\\rho$/<$\\rm \\rho$>")
                else:
                    axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ $n_{\\rm H}$ ($\\rm cm^{-3}$)")
            current_subplot_index += 1

    if not seperate_plots:
        if output_filepath is not None:
            Console.print_info("Saving image...", end = "")
            fig.savefig(output_filepath, dpi = 400)#TODO: look into metadata parameter - consider QC addition to automatically set some basic metadata in the correct format
            Console.print("done")
        else:
            Console.print_info("Rendering interactive window.")
            plt.show()
        plt.clf()
        plt.close()
