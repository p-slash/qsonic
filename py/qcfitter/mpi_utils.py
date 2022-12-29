import logging

import fitsio
import numpy as np


def balance_load(split_catalog, mpi_size):
    """Load balancing function.

    Arguments
    ---------
    split_catalog: list of named ndarray
    list of catalog. Each element is a ndarray with the same healpix

    mpi_size: int
    number of mpi tasks running

    Returns
    ---------
    local_queues: list of list of named ndarray
    spectra that each rank is reponsible for. same format as split_catalog
    """
    number_of_spectra = np.zeros(mpi_size, dtype=int)
    local_queues = [[]] * mpi_size
    split_catalog.sort(key=lambda x: x.size, reverse=True)  # Descending order
    for cat in split_catalog:
        min_idx = np.argmin(number_of_spectra)
        number_of_spectra[min_idx] += cat.size

        local_queues[min_idx].append(cat)

    return local_queues


def logging_mpi(msg, mpi_rank, fnc="info"):
    if mpi_rank == 0:
        getattr(logging, fnc)(msg)


class MPISaver(object):
    """ A simple object to write to a FITS file on master node.

    Parameters
    ----------
    fname: str
        Filename.
    mpi_rank: int
        MPI rank. Creates FITS if 0.
    is_dummy: bool
        Does not create file if true.
    """

    def __init__(self, fname, mpi_rank, is_dummy):
        if mpi_rank == 0 and not is_dummy:
            self.fts = fitsio.FITS(fname, 'rw', clobber=True)
        else:
            self.fts = None

    def close(self):
        if self.fts is not None:
            self.fts.close()

    def write(self, data, names, extname):
        if self.fts is not None:
            self.fts.write(data, names=names, extname=extname)
