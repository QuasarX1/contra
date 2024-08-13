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
        command = "contra-plot-igm-occupancy-hist",
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
            ),
            ScriptWrapper.OptionalParam[int](
                name = "n-bins",
                short_name = "n",
                sets_param = "number_of_bins",
                description = "Number of bins to use. Defaults to 25.",
                conversion_function = int,
                default_value = 25
            )
        )
    ).run_with_async(__main)

async def __main(
            input_filepath: str,
            snapshot_directory: str | None,
            output_filepath: str | None,
            number_of_bins: int
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

    plt.figure(layout = "tight", figsize = (6, 5.25))
    plt.rcParams['font.size'] = 12

    bins_edges = np.linspace(log10_halo_masses[0], log10_halo_masses[-1], number_of_bins + 1)
    bin_centres = (bins_edges[1:] + bins_edges[:-1]) / 2

    plt.plot(bin_centres, np.histogram(density = True, a = log10_halo_masses, weights = particle_metal_masses, bins = bins_edges)[0], label = "Metal Mass", linewidth = 2)
    plt.plot(bin_centres, np.histogram(density = True, a = log10_halo_masses, weights = particle_volumes, bins = bins_edges)[0], label = "Particle Volume", linewidth = 2)

    plt.xlabel("$\\rm log_{10}$ $M_{\\rm 200}$ of last halo ($\\rm M_\\odot$)")
    plt.ylabel("Probability Density")
    plt.legend()
    if output_filepath is not None:
        plt.savefig(output_filepath, dpi = 100)
    else:
        plt.show()
    return
