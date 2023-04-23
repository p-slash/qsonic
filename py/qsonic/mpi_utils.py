import logging
import warnings

import fitsio
import numpy as np

from qsonic import QsonicException


def mpi_parse(parser, comm, mpi_rank, options=None):
    """ Parse arguments on the master node, then broadcast.

    Arguments
    ---------
    parser: argparse.ArgumentParser
        Parser to be used.
    comm: MPI.COMM_WORLD
        MPI comm object for bcast
    mpi_rank: int
        Rank of the MPI process.
    options: None or list, default: None
        Options to parse. None parses ``sys.argv``
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
    """ Logs only on ``mpi_rank=0``.

    Arguments
    ---------
    msg: str
        Message to log.
    mpi_rank: int
        Rank of the MPI process.
    fnc: logging method
        Channel to log, usually info or error.
    """
    if mpi_rank == 0:
        getattr(logging, fnc)(msg)


def warn_mpi(msg, mpi_rank):
    """ Warns of RuntimeWarning only on ``mpi_rank=0``.

    Arguments
    ---------
    msg: str
        Message to log.
    mpi_rank: int
        Rank of the MPI process.
    """

    if mpi_rank == 0:
        warnings.warn(msg, RuntimeWarning)


class _INVALID_VALUE(object):
    """ Sentinel for invalid values. """


def mpi_fnc_bcast(fnc, comm=None, mpi_rank=0, err_msg="", *args, **kwargs):
    """ Wrapper function to run function then broadcast. Return value of
    ``fnc`` must be broadcastable.

    Example::

        # Reading lines of integers
        # from file `filename_for_integers` using np.loadtxt
        import numpy as np

        ints_in_file = mpi_fnc_bcast(
                np.loadtxt,
                comm, mpi_rank, "Error while reading file.",
                filename_for_integers, dtype=int)

    Arguments
    ---------
    fnc: callable
        Function to evaluate.
    comm: MPI.COMM_WORLD or None, default: None
        MPI comm object for bcast
    mpi_rank: int, default: 0
        Rank of the MPI process.
    err_msg: str, default: ''
        Error message to use raising QsonicException.
    *args:
        Arguments such as ``fnc(3, 4)``.
    **kwargs:
        Keyword arguments such as ``fnc(dtype=int)``

    Returns
    -------
    result: fnc
        Function ``fnc(*args, **kwargs)``.

    Raises
    ------
    QsonicException
        If any error occur while doing ``fnc``.
    """
    result = _INVALID_VALUE
    if (mpi_rank == 0 or comm is None):
        try:
            result = fnc(*args, **kwargs)
        except Exception as e:
            logging.exception(e)
            logging.error(f"Error in {fnc.__module__}.{fnc.__name__}.")
            result = _INVALID_VALUE

    if comm is not None:
        result = comm.bcast(result)

    if result is _INVALID_VALUE:
        raise QsonicException(err_msg)

    return result


def balance_load(split_catalog, mpi_size, mpi_rank):
    """ Load balancing function.

    Arguments
    ---------
    split_catalog: list(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        List of catalog. Each element is a ndarray with the same healpix.
    mpi_size: int
        Number of MPI tasks running.
    mpi_rank: int
        Rank of the MPI process.

    Returns
    ---------
    local_queue: list(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        Spectra that current rank is reponsible for in ``split_catalog`` format
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


class MPISaver():
    """ A simple class to write to a FITS file on master node.

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

    def write(self, data, names=None, extname=None, header=None):
        """ Write to FITS file.

        Arguments
        ---------
        data: list(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
            Data to write to extention.
        names: list(str)or None, default: None
            Column names for data.
        extname: str or None, default: None
            Extention name.
        header: dict or None, default: None
            Header dictionary to save.
        """
        if self.fts is not None:
            self.fts.write(data, names=names, extname=extname, header=header)
