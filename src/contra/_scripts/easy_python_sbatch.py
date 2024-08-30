import sys
import os

def main() -> None:
    if len(sys.argv[1:]) == 0 or "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print("""\
Easy SBATCH:
    Use this to easily run sbatch in exclusive mode.

    Usage:
        easy-sbatch -p <partition> -N 1 -t <time> -o "slurm-%j.out" <command> [args]
    
        Evaluates to:
        sbatch --exclusive -p <partition> -N 1 -t <time> -o "slurm-%j.out" <command> [args]\
""")
    else:
        _ = os.system("sbatch --exclusive " + " ".join(sys.argv[1:]))
