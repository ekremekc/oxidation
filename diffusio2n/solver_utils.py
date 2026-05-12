import datetime
from mpi4py import MPI


def start_time():
    return datetime.datetime.now()


def execution_time(start_time, string="\nTotal execution time: "):
    if MPI.COMM_WORLD.rank == 0:
        print(string, datetime.datetime.now() - start_time, "\n")


def info(str):
    """Only prints information message for once. Useful for logging,

    Args:
        str ('str'): log entry
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(str)
