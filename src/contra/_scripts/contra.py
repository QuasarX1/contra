# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None

def main():
    print("""\
CONTRA:
    Reverse search for EAGLE and SWIFT particle data to identify last occupied structure.

    This software searches the snapshots of hydro simulations (supported EAGLE and SWIFT)
        to find the last halo membership for all particles in a given snapshot.

    Usage:
        Specify a target snapshot and particle type(s) (gas, star, black hole or dark matter)
            and optionally which higher redshift snapshots to search. This will produce a
            supplementary file in HDF5 format containing the most recent identified halo for
            each target particle, along with some basic information about that halo.

        From within the simulation output directory (assuming catalogue files in same folder):
        contra-run 0012 --swift

        Allow inference of some settings:
        contra-run -t /path/to/sim/snapshot_0012 \\
                   -c /path/to/sim/catalogue/SOAP_halo_membership__0012 \\
                    --swift \\
                   -g -s -bh -dm \\
                   -o /path/to/output/dir/outputfile.hdf5

        More explicit ( e.g. /path/to/sim/snap_0012.0.hdf5          )
                      (      /path/to/sim/catalogue/haloes_0012.hdf5):
        contra-run --snapshots /path/to/sim \\               # Directory containing snapshots
                   --snapshot-basename snap_ \\              # Prefix to snapshot number
                   --snapshot-parallel \\                    # Snapshots written in parallel
                   --snapshot-extension .hdf5 \\             # File extension of snapshots
                   --catalogue /path/to/sim/catalogue \\     # SOAP catalogue directory
                   --catalogue-basename haloes_ \\           # SOAP prefix to snapshot number
                   --catalogue-parallel \\                   # Catalogue written in parallel
                   --catalogue-extension .hdf5 \\            # File extension of catalogue
                   -t 0012 \\                                # Target snapshot number 0012
                   --swift \\                                # Simulation is SWIFTsim
                   -g -s -bh -dm \\                          # Use all particle types
                   -o /path/to/output/dir/outputfile.hdf5 \\ # Output file
                   --allow-overwrite                        # Allow overwrite of outputs
""")
