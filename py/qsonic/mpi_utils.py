import logging

import fitsio
import numpy as np

from qsonic import QsonicException


def _logic_true(args):
    return True


def mpi_parse(
        parser, comm, mpi_rank, options=None, args_logic_fnc=_logic_true
):
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
    args_logic_fnc: Callable[[args], bool], default: True
        Logic function to check for args.
    """
    if mpi_rank == 0:
        try:
            args = parser.parse_args(options)
            if not args_logic_fnc(args):
                args = -1
        except SystemExit:
            args = -1
    else:
        args = -1

    args = comm.bcast(args)
    if args == -1:
        exit(0)

    return args


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


def balance_load(split_catalog, mpi_size):
    """Load balancing function. The return value can be scattered.

    Arguments
    ---------
    split_catalog: list(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        List of catalog. Each element is a ndarray with the same healpix.
    mpi_size: int
        Number of MPI tasks running.

    Returns
    ---------
    local_queue: list(list(:external+numpy:py:class:`ndarray <numpy.ndarray>`))
        Spectra that ranks are reponsible for in ``split_catalog`` format.
    """
    number_of_spectra = np.zeros(mpi_size, dtype=int)
    local_queues = [[] for _ in range(mpi_size)]
    split_catalog.sort(key=lambda x: x.size, reverse=True)  # Descending order
    for cat in split_catalog:
        min_idx = np.argmin(number_of_spectra)
        number_of_spectra[min_idx] += cat.size
        local_queues[min_idx].append(cat)

    return local_queues


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
            Data to write to extension.
        names: list(str)or None, default: None
            Column names for data.
        extname: str or None, default: None
            Extention name.
        header: dict or None, default: None
            Header dictionary to save.
        """
        if self.fts is not None:
            self.fts.write(data, names=names, extname=extname, header=header)
