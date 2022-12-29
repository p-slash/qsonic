import fitsio
from healpy import ang2pix
import numpy as np
from numpy.lib.recfunctions import rename_fields, append_fields

from qcfitter.mpi_utils import logging_mpi, balance_load

_accepted_extnames = set(['QSO_CAT', 'ZCATALOG', 'METADATA'])
_accepted_columns = [
    'TARGETID', 'Z', 'TARGET_RA', 'RA', 'TARGET_DEC', 'DEC',
    'SURVEY', 'HPXPIXEL', 'VMIN_CIV_450', 'VMAX_CIV_450',
    'VMIN_CIV_2000', 'VMAX_CIV_2000'
]


def read_local_qso_catalog(
        filename, comm, mpi_rank, mpi_size,
        n_side=64, keep_surveys=None, zmin=2.1, zmax=6.0):
    """Returns quasar catalog object for mpi_rank. It is sorted in the
    following order: HPXPIXEL, SURVEY (if applicable), TARGETID

    Arguments
    ----------
    filename: str
        Filename to catalog.
    comm: MPI comm object
        MPI comm object for bcast
    mpi_rank: int
        Rank of MPI process
    mpi_size: int
        Size of MPI processes
    n_side: int (default: 64)
        Healpix nside.
    keep_surveys: list (default: all)
        List of surveys to subselect.
    zmin: float (default: 2.1)
        Minimum quasar redshift
    zmax: float (default: 6.0)
        Maximum quasar redshift

    Returns
    ----------
    local_queue: list of ndarray
        list of sorted catalogs.
        BAL info included if available (req. for BAL masking)
    """
    catalog = None

    if mpi_rank == 0:
        catalog = _read_catalog_on_master(
            filename, n_side, keep_surveys, zmin, zmax)

    catalog = comm.bcast(catalog, root=0)
    if catalog is None:
        logging_mpi("Error while reading catalog.", mpi_rank, "error")
        exit(0)

    return _get_local_queue(catalog, mpi_rank, mpi_size)


def _get_local_queue(catalog, mpi_rank, mpi_size):
    # We decide forest filename list
    # Group into unique pixels
    unique_pix, s = np.unique(catalog['HPXPIXEL'], return_index=True)
    split_catalog = np.split(catalog, s[1:])
    logging_mpi(
        f"There are {unique_pix.size} healpixels."
        " Don't use more MPI processes.", 0)

    # Roughly equal number of spectra
    logging_mpi("Load balancing.", 0)
    # Returns a list of catalog (ndarray)
    return balance_load(split_catalog, mpi_size, mpi_rank)


def _read(filename):
    logging_mpi(f'Reading catalogue from {filename}', 0)
    fitsfile = fitsio.FITS(filename)
    extnames = [hdu.get_extname() for hdu in fitsfile]
    cat_hdu = _accepted_extnames.intersection(extnames)
    if not cat_hdu:
        cat_hdu = 1
        logging_mpi(
            "Catalog HDU not found by hduname. Using extension 1.",
            0, "warning")
    else:
        cat_hdu = cat_hdu.pop()

    colnames = fitsfile[cat_hdu].get_colnames()
    keep_columns = [col for col in colnames if col in _accepted_columns]
    catalog = np.array(fitsfile[cat_hdu].read(columns=keep_columns))
    fitsfile.close()

    return catalog, keep_columns


def _add_healpix(catalog, n_side, keep_columns):
    if 'HPXPIXEL' not in keep_columns:
        pixnum = ang2pix(n_side, catalog['RA'], catalog['DEC'],
                         lonlat=True, nest=True)
        catalog = append_fields(catalog, 'HPXPIXEL', pixnum, dtypes=int)

    return catalog


def _read_catalog_on_master(filename, n_side, keep_surveys, zmin, zmax):
    """Returns quasar catalog object. It is sorted in the following order:
    HPXPIXEL, SURVEY (if applicable), TARGETID

    Arguments
    ----------
    filename: str
        Filename to catalog.
    n_side: int
        Healpix nside.
    keep_surveys: list
        List of surveys to subselect.
    zmin: float
        Minimum quasar redshift
    zmax: float
        Maximum quasar redshift

    Returns
    ----------
    catalog: ndarray
        Sorted catalog. BAL info included if available (req. for BAL masking)
    """
    catalog, keep_columns = _read(filename)
    logging_mpi(f"There are {catalog.size} quasars in the catalog.", 0)

    if catalog.size != np.unique(catalog['TARGETID']).size:
        logging_mpi("There are duplicate TARGETIDs in catalog!", 0, "error")
        return None

    # Redshift cuts
    w = (catalog['Z'] >= zmin) & (catalog['Z'] <= zmax)
    logging_mpi(f"There are {w.sum()} quasars in the redshift range.", 0)
    catalog = catalog[w]

    if 'TARGET_RA' in keep_columns and 'RA' not in keep_columns:
        catalog = rename_fields(
            catalog, {'TARGET_RA': 'RA', 'TARGET_DEC': 'DEC'})

    sort_order = ['HPXPIXEL', 'TARGETID']
    # Filter all the objects in the catalogue not belonging to the specified
    # surveys.
    if 'SURVEY' in keep_columns and keep_surveys is not None:
        w = np.isin(catalog["SURVEY"], keep_surveys)
        catalog = catalog[w]
        if len(keep_surveys) > 1:
            sort_order.insert(1, 'SURVEY')
        logging_mpi(
            f"There are {w.sum()} quasars in given surveys {keep_surveys}.", 0)

    if catalog.size == 0:
        logging_mpi("Empty quasar catalogue.", 0, "error")
        return None

    catalog = _add_healpix(catalog, n_side, keep_columns)
    catalog.sort(order=sort_order)

    return catalog
