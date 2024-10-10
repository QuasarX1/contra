# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
import datetime
from QuasarCode import Settings, Console
from QuasarCode.MPI import MPI_Config
from QuasarCode.Tools import ScriptWrapper



def main():
    ScriptWrapper(
        command = "contra-test-mpi",
        authors = [ScriptWrapper.AuthorInfomation()],
        version = ScriptWrapper.VersionInfomation.from_string("1.0.0"),
        edit_date = datetime.date(2024, 10, 10),
        description = "Run this script using `mpirun -np 2 contra-test-mpi` to verify MPI is installed and working correctly.\nIf errors are encountered, try using `mpiexec` and try altering the number of ranks to be different from 2.",
        parameters = ScriptWrapper.ParamSpec()
    ).run_with_async(__main)



async def __main() -> None:

    if not Settings.mpi_avalible:
        Console.print_error("Unable to load mpi4py module.\nEither it is not installed or it failed to load.")
        return

    if MPI_Config.comm_size == 1:
        Console.print_error("Comm size of 1. Test using more than 1 rank!\nIf multiple ranks were launched, they were not linked together.\nMake sure to use `mpirun` or `mpiexec` with `-np` to specify the number of ranks.")
        return

    if MPI_Config.is_root:
        Console.print_info(f"Running with MPI on {MPI_Config.comm_size} ranks.\nRoot MPI rank is {MPI_Config.root}.")
    MPI_Config.comm.barrier()
    if not MPI_Config.is_root:
        Console.print_info("This rank also working.")
