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

def main():
    ScriptWrapper(
        command = "contra-plot-igm-occupancy",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 7, 11),
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
            )
        )
    ).run_with_async(__main)

async def __main(
            input_filepath: str,
            snapshot_directory: str | None,
            output_filepath: str | None
          ) -> None:

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

    halo_ids = contra_data.gas.halo_ids
    located_particle_mask = halo_ids != -1

    last_halo_redshifts = contra_data.gas.redshifts
    ejected_before_target_snap_mask = last_halo_redshifts > snap.redshift

    particle_metalicities = snap.get_metalicities(ParticleType.gas).value
    metals_present_mask = particle_metalicities > 0

    halo_masses = contra_data.gas.halo_masses
    nonzero_halo_mass_mask = halo_masses > 0.0

    data_mask = located_particle_mask & ejected_before_target_snap_mask & nonzero_halo_mass_mask & metals_present_mask

    # Determine particle order by last halo mass
    # Uses un-logged values to avoid loss of precision (proberbly not nessessary)
    halo_masses = halo_masses[data_mask]
    sorted_order = np.argsort(halo_masses)

    # Contra quantities
    halo_masses = halo_masses[sorted_order]
    log10_halo_masses = np.log10(halo_masses)

    # Raw particle quantities
    particle_metalicities = particle_metalicities[data_mask][sorted_order]
    particle_masses = snap.get_masses(ParticleType.gas).to("Msun").value[data_mask][sorted_order]
    particle_volumes = snap.get_volumes(ParticleType.gas).to("Mpc**3").value[data_mask][sorted_order]

    # Derrived particle quantities
    particle_metal_masses = particle_masses * particle_metalicities
#    particle_metal_densities = particle_metal_masses / particle_volumes

    # Cumulitive particle quantities
    cumulitive_particle_metal_masses = np.cumsum(particle_metal_masses)
    cumulitive_particle_volumes = np.cumsum(particle_volumes)
#    cumulitive_particle_metal_densities = np.cumsum(particle_metal_densities)

    plt.figure(layout = "tight", figsize = (6, 5.25))
    plt.rcParams['font.size'] = 12
    plt.plot(log10_halo_masses, cumulitive_particle_metal_masses / cumulitive_particle_metal_masses[-1], label = "Metal Mass", linewidth = 2)
    plt.plot(log10_halo_masses, cumulitive_particle_volumes / cumulitive_particle_volumes[-1], label = "Volume Filling Fraction", linewidth = 2)
#    plt.plot(log10_halo_masses, cumulitive_particle_metal_densities / cumulitive_particle_metal_densities[-1], label = "Particle Metal Density", linewidth = 2)
    xlims = plt.xlim()
    ylims = (0.0, 1.01)
    mass_midpoint = log10_halo_masses[cumulitive_particle_metal_masses / cumulitive_particle_metal_masses[-1] >= 0.5][0]
    volume_midpoint = log10_halo_masses[cumulitive_particle_volumes / cumulitive_particle_volumes[-1] >= 0.5][0]
    plt.plot([xlims[0], max(mass_midpoint, volume_midpoint)], [0.5, 0.5], color = "red", alpha = 0.3, linestyle = "--", linewidth = 2)
    plt.plot([mass_midpoint, mass_midpoint], [0.5, ylims[0]], color = "red", alpha = 0.3, linestyle = "--", linewidth = 2)
    plt.plot([volume_midpoint, volume_midpoint], [0.5, ylims[0]], color = "red", alpha = 0.3, linestyle = "--", linewidth = 2)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel("$\\rm log_{\\rm 10}$ $M_{\\rm 200}$ of last halo ($\\rm M_\\odot$)")
    plt.ylabel("Cumulative Fraction (for particles with Z > 0)")
    plt.legend(loc = "upper left", prop={ "size" : 10 })
    if output_filepath is not None:
        plt.savefig(output_filepath, dpi = 400)
    else:
        plt.show()
    return
