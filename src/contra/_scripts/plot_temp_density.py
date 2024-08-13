# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .. import VERSION, ParticleType, ArrayReorder, ArrayMapping
from .._L_star import get_L_star_halo_mass_of_z
from ..io import SnapshotBase, SnapshotEAGLE, SnapshotSWIFT, \
                OutputReader, HeaderDataset, ParticleTypeDataset, \
                SnapshotStatsDataset, ContraData, \
                ParticleFilterFile, ParticleFilter, LOSFilter

import datetime
import os
from typing import Union, List, Tuple, Dict
import asyncio
from matplotlib import pyplot as plt

import numpy as np
from unyt import unyt_quantity, unyt_array
from QuasarCode import Settings, Console
from QuasarCode.Data import VersionInfomation
from QuasarCode.Tools import ScriptWrapper

Z_SOLAR = 0.0134 # Z_sun

def main():
    ScriptWrapper(
        command = "contra-plot-environment",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 7, 9),
        description = "",
        parameters = ScriptWrapper.ParamSpec(
            ScriptWrapper.OptionalParam[str](
                name = "data",
                short_name = "i",
                sets_param = "input_filepath",
                default_value = "./contra-output.hdf5",
                description = "Input contra data file. Defaults to \"./contra-output.hdf5\"."
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
                description = "Use metal-mass-weighted mean to determine colour values instead of the median."
            ),
            ScriptWrapper.Flag(
                name = "stack-row",
                description = "Stack both plots as a row.",
                conflicts = ["stack-column"]
            ),
            ScriptWrapper.Flag(
                name = "stack-column",
                description = "Stack both plots as a column.",
                conflicts = ["stack-row"]
            )
        )
    ).run_with_async(__main)

async def __main(
            input_filepath: str,
            snapshot_directory: str | None,#TODO:
            output_filepath: str | None,
            min_density: float | None,
            max_density: float | None,
            min_log10_overdensity: float | None,
            max_log10_overdensity: float | None,
            min_log10_temp: float | None,
            max_log10_temp: float | None,
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
            stack_column: bool
          ) -> None:
    
    if stack_row or stack_column:
        raise NotImplementedError("Subplot format not yet impleented.")#TODO:

    # Read Contra header data
    with OutputReader(input_filepath) as contra_reader:
        contra_header = contra_reader.read_header()

    if not contra_header.has_gas:
        Console.print_error("Contra data has no results for gas particles.")
        Console.print_info("Terminating...")
        return

    # This will get re-loaded, so free up the memory
    del contra_header
    
    contra_data = ContraData.load(input_filepath, include_stats = False)

    if snapshot_directory is not None:
        # Ensure that the path is an absolute path
        snapshot_directory = os.path.abspath(snapshot_directory)
        # Find the correct path for the target snapshot
        snap_dir, *file_elements = contra_data.header.target_snapshot.rsplit(os.path.sep, maxsplit = 1 if contra_data.header.simulation_type == "SWIFT" else 2)
        target_snap = os.path.join(snapshot_directory, *file_elements)
        snap = contra_data._get_snapshot(target_snap)
    else:
        snap = contra_data.get_target_snapshot()



    # Load data

    located_particle_mask: np.ndarray = contra_data.gas.halo_ids != -1

    particle_metalicities: np.ndarray = snap.get_metalicities(ParticleType.gas).value
    metals_present_mask: np.ndarray = particle_metalicities > 0.0

    data_mask: np.ndarray = located_particle_mask & metals_present_mask
    n_particles: int = data_mask.sum()

    log10_last_halo_masses: np.ndarray = np.log10(contra_data.gas.halo_masses[data_mask])
    if use_mean:
        particle_metal_masses: np.ndarray = (snap.get_masses(ParticleType.gas).to("Msun").value * particle_metalicities)[data_mask]
    particle_metalicities: np.ndarray = particle_metalicities[data_mask]
    particle_densities: np.ndarray = snap.get_densities(ParticleType.gas).to("Msun/Mpc**3").value[data_mask]
    log10_particle_temperatures: np.ndarray = np.log10(snap.get_temperatures(ParticleType.gas).to("K").value[data_mask])

    particle_indexes: np.ndarray = np.arange(n_particles)



    # Convert to solar
    log10_one_plus_particle_metalicities_solar = np.log10(1 + (particle_metalicities / Z_SOLAR))

    if min_colour_metalicity is not None:
        min_colour_metalicity_solar = min_colour_metalicity / Z_SOLAR
    if max_colour_metalicity is not None:
        max_colour_metalicity_solar = max_colour_metalicity / Z_SOLAR

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

    def reduce_colour__metalicity(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        if use_mean:
            average_Z = np.average(log10_one_plus_particle_metalicities_solar[indices], weights = particle_metal_masses[indices])
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
            return 0.0
        if use_mean:
            average_m200 = np.average(log10_last_halo_masses[indices], weights = particle_metal_masses[indices])
        else:
            average_m200 = np.median(log10_last_halo_masses[indices])
        if min_colour_halo_mass is not None and average_m200 < min_colour_halo_mass:
            return min_colour_halo_mass
        elif max_colour_halo_mass is not None and average_m200 > max_colour_halo_mass:
            return max_colour_halo_mass
        else:
            return average_m200



    # Plot

    plt.rcParams['font.size'] = 12



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



    plt.figure(layout = "tight", figsize = (6, 5.25))

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
