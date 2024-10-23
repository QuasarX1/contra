# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .. import ParticleType
from .._L_star import get_L_star_halo_mass_of_z
from ..io import SnapshotBase, SnapshotEAGLE, SnapshotSWIFT, FileTreeScraper_EAGLE
from ..io._Output_Objects import OutputReader, ParticleTypeDataset, ContraData

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

Z_SOLAR = 0.0134 # Z_sun

def main():
    ScriptWrapper(
        command = "contra-plot-environment",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 7, 9),
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
                conflicts = ["min-overdensity"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-density",
                description = "Maximum density (in Msun/Mpc^3) to display.",
                conversion_function = float,
                conflicts = ["max-overdensity"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "min-overdensity",
                sets_param = "min_log10_overdensity",
                description = "Minimum (log10) overdensity to display.",
                conversion_function = float,
                conflicts = ["min-density"]
            ),
            ScriptWrapper.OptionalParam[float](
                name = "max-overdensity",
                sets_param = "max_log10_overdensity",
                description = "Maximum (log10) overdensity to display.",
                conversion_function = float,
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
            ScriptWrapper.Flag(
                name = "show-igm",
                description = "Draw the z=0 IGM boundary from Wiersma et al. 2010."
            ),
            ScriptWrapper.OptionalParam[float](
                name = "solar-metal-mass-fraction",
                sets_param = "configured_Z_solar",
                default_value = Z_SOLAR,
                description = f"Value of Z_sun used to calculate solar metallicity from absolute metal mass fraction.\nDefaults to {Z_SOLAR}.",
                conversion_function = float
            )
        )
    ).run_with_async(__main)

async def __main(
            target_redshift: float,
            input_filepath: str,
            plot_hist: bool,
            plot_metallicity: bool,
            plot_last_halo_mass: bool,
            plot_metal_weighted_last_halo_mass: bool,
            snapshot_directory: str | None,
            output_filepath: str | None,
            min_density: float | None,
            max_density: float | None,
            min_log10_overdensity: float | None,
            max_log10_overdensity: float | None,
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
            use_mean: bool,
            stack_row: bool,
            stack_column: bool,
            show_igm: bool,
            configured_Z_solar: float
          ) -> None:

    # If not other plot types are specified, plot count bins
    if not (plot_metallicity or plot_last_halo_mass or plot_metal_weighted_last_halo_mass):
        plot_hist = True

    # Read Contra header data
    with OutputReader(input_filepath) as contra_reader:
        contra_header = contra_reader.read_header()
    if not contra_header.has_gas:
        Console.print_error("Contra data has no results for gas particles.")
        Console.print_info("Terminating...")
        return
    # This will get re-loaded, so free up the memory
    del contra_header
    
    contra_data = ContraData.load(input_filepath, include_stats = False, alternate_simulation_data_directory = snapshot_directory)

    target_file_number: str = contra_data.find_file_number_from_redshift(target_redshift)
    snap: SnapshotBase = contra_data.get_snapshot(target_file_number)



    # Load data

    gas_dataset: ParticleTypeDataset = cast(ParticleTypeDataset, contra_data.data[target_file_number].gas)

    located_particle_mask: np.ndarray = gas_dataset.halo_ids != -1

    nonzero_halo_mass_mask: np.ndarray = gas_dataset.halo_masses > 0.0

    particle_metalicities: np.ndarray = snap.get_metalicities(ParticleType.gas).value
    metals_present_mask: np.ndarray = particle_metalicities > 0.0

    data_mask: np.ndarray = located_particle_mask & metals_present_mask# & nonzero_halo_mass_mask
    n_particles: int = data_mask.sum()

    log10_last_halo_masses: np.ndarray = np.log10(gas_dataset.halo_masses[data_mask])
    if use_mean:
        particle_masses = snap.get_masses(ParticleType.gas).to("Msun").value[data_mask]
        particle_metal_masses: np.ndarray = (snap.get_masses(ParticleType.gas).to("Msun").value * particle_metalicities)[data_mask]
    particle_metalicities = particle_metalicities[data_mask]
    particle_densities: np.ndarray = snap.get_densities(ParticleType.gas).to("Msun/Mpc**3").value[data_mask]
    log10_particle_temperatures: np.ndarray = np.log10(snap.get_temperatures(ParticleType.gas).to("K").value[data_mask])

    particle_indexes: np.ndarray = np.arange(n_particles)



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



    # Convert to overdensity

    mean_baryon_density: float = snap.calculate_comoving_critical_gas_density().to("Msun/Mpc**3").value

    log10_particle_overdensities: np.ndarray = np.log10(particle_densities / mean_baryon_density)

    if min_density is not None:
        min_log10_overdensity = min_density / mean_baryon_density
    if max_density is not None:
        max_log10_overdensity = max_density / mean_baryon_density



    # Define the colour calculation functions

    def reduce_colour__count(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return -np.inf
        counts = np.log10(len(indices))
        if min_colour_log_count is not None and counts < min_colour_log_count:
            return min_colour_log_count
        elif max_colour_log_count is not None and counts > max_colour_log_count:
            return max_colour_log_count
        else:
            return counts

    def reduce_colour__metalicity(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return -np.inf
        if use_mean:
            average_Z = np.average(np.log10(particle_metalicities[indices]) - np.log10(configured_Z_solar), weights = particle_metal_masses[indices])
#            average_Z = np.average(log10_one_plus_particle_metalicities_solar[indices], weights = particle_metal_masses[indices])
#            average_Z = np.average(log10_one_plus_particle_metalicities_solar[indices], weights = particle_masses[indices])
        else:
            average_Z = np.median(log10_one_plus_particle_metalicities_solar[indices])
        if min_colour_metalicity_plotted is not None and average_Z < min_colour_metalicity_plotted:
            return min_colour_metalicity_plotted
        elif max_colour_metalicity_plotted is not None and average_Z > max_colour_metalicity_plotted:
            return max_colour_metalicity_plotted
        else:
            return average_Z

    def reduce_colour__last_halo_mass(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return -np.inf
        if use_mean:
            average_m200 = np.average(log10_last_halo_masses[indices])
        else:
            average_m200 = np.median(log10_last_halo_masses[indices])
        if min_colour_halo_mass is not None and average_m200 < min_colour_halo_mass:
            return min_colour_halo_mass
        elif max_colour_halo_mass is not None and average_m200 > max_colour_halo_mass:
            return max_colour_halo_mass
        else:
            return average_m200

    def reduce_colour__metal_weighted_last_halo_mass(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return -np.inf
        average_m200 = np.average(log10_last_halo_masses[indices], weights = particle_metal_masses[indices])
        if min_colour_halo_mass is not None and average_m200 < min_colour_halo_mass:
            return min_colour_halo_mass
        elif max_colour_halo_mass is not None and average_m200 > max_colour_halo_mass:
            return max_colour_halo_mass
        else:
            return average_m200
        


    #TODO: calculate contours



    # Plot

    plt.rcParams['font.size'] = 12

    n_subplots = int(plot_hist) + int(plot_metallicity) + int(plot_last_halo_mass) + int(plot_metal_weighted_last_halo_mass)
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

    for plot_name, label, colour_reduction_function, min_colour, max_colour in zip(
        ("histogram",                             "metallicity",                                                                                                "last-halo-mass",                                                                                               "metal-weighted-last-halo-mass"                                                                  ),
#        ("${\\rm log_{10}}$ Number of Particles", ("Metal-mass Weighted Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ 1 + Z ($\\rm Z_{\\rm \\odot}$)", ("Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ $M_{\\rm 200}$ of last halo ($\\rm M_{\\rm \\odot}$)", "Metal-mass Weighted Mean ${\\rm log_{10}}$ $M_{\\rm 200}$ of last halo ($\\rm M_{\\rm \\odot}$)"),
        ("${\\rm log_{10}}$ Number of Particles", "${\\rm log_{10}}$ " + ("Metal-mass Weighted Mean" if use_mean else "Median") + " Z - ${\\rm log_{10}}$ $\\rm Z_{\\rm \\odot}$", ("Mean" if use_mean else "Median") + " ${\\rm log_{10}}$ $M_{\\rm 200}$ of last halo ($\\rm M_{\\rm \\odot}$)", "Metal-mass Weighted Mean ${\\rm log_{10}}$ $M_{\\rm 200}$ of last halo ($\\rm M_{\\rm \\odot}$)"),
        (reduce_colour__count,                    reduce_colour__metalicity,                                                                                    reduce_colour__last_halo_mass,                                                                                  reduce_colour__metal_weighted_last_halo_mass                                                     ),
        (min_colour_log_count,                    min_colour_metalicity_plotted,                                                                                min_colour_halo_mass,                                                                                           min_colour_halo_mass                                                                             ),
        (max_colour_log_count,                    max_colour_metalicity_plotted,                                                                                max_colour_halo_mass,                                                                                           max_colour_halo_mass                                                                             )
    ):
        if not ((plot_name == "histogram" and plot_hist) or (plot_name == "metallicity" and plot_metallicity) or (plot_name == "last-halo-mass" and plot_last_halo_mass) or (plot_name == "metal-weighted-last-halo-mass" and plot_metal_weighted_last_halo_mass)):
            continue

        if seperate_plots:
            fig = plt.figure(layout = "tight", figsize = (6, 5.25))
            axes = [fig.gca()]

#        colourmap = "viridis"
        colourmap = tol_colors.LinearSegmentedColormap.from_list("custom-map", ["#125A56", "#FD9A44", "#A01813"])
#        "#125A56;#FD9A44;#A01813"
        coloured_object = axes[current_subplot_index].hexbin(
            log10_particle_overdensities, log10_particle_temperatures,
            C = particle_indexes, reduce_C_function = colour_reduction_function,
            gridsize = 500, cmap = colourmap, vmin = min_colour, vmax = max_colour
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

        #TODO: plot contours

        xlims = axes[current_subplot_index].set_xlim((min_log10_overdensity, max_log10_overdensity))
        ylims = axes[current_subplot_index].set_ylim((min_log10_temp, max_log10_temp))

        if show_igm:
            axes[current_subplot_index].plot(
                [xlims[0], 5.75, 5.75, 2.0, 2.0     ],
                [7.0,      7.0,  4.5,  4.5, ylims[0]],
                color = "black"
            )
            axes[current_subplot_index].set_xlim(xlims)
            axes[current_subplot_index].set_ylim(ylims)
            axes[current_subplot_index].text(
                2.0, 6.0,
                "IGM",
                bbox = { "facecolor" : "lightgrey", "edgecolor" : "none" }
            )

        if seperate_plots:
            axes[current_subplot_index].set_ylabel("${\\rm log_{10}}$ Temperature (${\\rm K}$)")
            axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ Overdensity = $\\rho$/<$\\rm \\rho$>")
            if output_filepath is not None:
                fig.savefig(output_filepath if n_subplots == 1 else (split_filename:=output_filepath.rsplit(".", maxsplit = 1))[0] + plot_name + "." + split_filename[1], dpi = 400)
            else:
                plt.show()
            plt.clf()
            plt.close()
        else:
            if stack_column or current_subplot_index == 0: # Y-axis --> Left-most plot of row or all
                axes[current_subplot_index].set_ylabel("${\\rm log_{10}}$ Temperature (${\\rm K}$)")
            if stack_row or current_subplot_index == n_subplots: # X-axis --> Bottom-most plot of column or all
                axes[current_subplot_index].set_xlabel("${\\rm log_{10}}$ Overdensity = $\\rho$/<$\\rm \\rho$>")
            current_subplot_index += 1

    if not seperate_plots:
        if output_filepath is not None:
            fig.savefig(output_filepath, dpi = 400)
        else:
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
