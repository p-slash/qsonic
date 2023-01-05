import logging

import fitsio
import numpy as np


def mpi_parse(parser, comm, mpi_rank, options=None):
    """ Parse arguments on the master node, then broadcast.

    Arguments
    ---------
    parser: argparse.ArgumentParser
    comm: MPI comm object
        MPI comm object for bcast
    mpi_rank: int
        Rank of the MPI process.
    options: list. Default is None, which parses sys.argv
    """
    if mpi_rank == 0:
        try:
            args = parser.parse_args(options)
        except SystemExit:
            args = -1
    else:
        args = -1

    args = comm.bcast(args)
    if args == -1:
        exit(0)

    return args


def logging_mpi(msg, mpi_rank, fnc="info"):
    """ Logs only on `mpi_rank=0`.

    Arguments
    ---------
    msg: str
        Message to log.
    mpi_rank: int
        Rank of the MPI process.
    fnc: logging attr
        Channel to log, usually info or error.
    """
    if mpi_rank == 0:
        getattr(logging, fnc)(msg)


def balance_load(split_catalog, mpi_size, mpi_rank):
    """ Load balancing function.

    Arguments
    ---------
    split_catalog: list of named ndarray
        List of catalog. Each element is a ndarray with the same healpix.
    mpi_size: int
        Number of MPI tasks running.
    mpi_rank: int
        Rank of the MPI process.

    Returns
    ---------
    local_queue: list of named ndarray
        Spectra that current rank is reponsible for in `split_catalog` format.
    """
    number_of_spectra = np.zeros(mpi_size, dtype=int)
    local_queue = []
    split_catalog.sort(key=lambda x: x.size, reverse=True)  # Descending order
    for cat in split_catalog:
        min_idx = np.argmin(number_of_spectra)
        number_of_spectra[min_idx] += cat.size

        if min_idx == mpi_rank:
            local_queue.append(cat)

    return local_queue


class MPISaver(object):
    """ A simple object to write to a FITS file on master node.

    Parameters
    ----------
    fname: str
        Filename. Does not create if empty string.
    mpi_rank: int
        Rank of the MPI process. Creates FITS if 0.
    """

    def __init__(self, fname, mpi_rank):
        if mpi_rank == 0 and fname:
            self.fts = fitsio.FITS(fname, 'rw', clobber=True)
        else:
            self.fts = None

    def close(self):
        """ Close FITS file."""
        if self.fts is not None:
            self.fts.close()

    def write(self, data, names, extname):
        """ Write to FITS file.

        Arguments
        ---------
        data: list of numpy arrays
            Data to write to extention.
        names: list of str
            Column names for data.
        extname: str
            Extention name.
        """
        if self.fts is not None:
            self.fts.write(data, names=names, extname=extname)
