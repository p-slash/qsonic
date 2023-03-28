""" This module contains functions to read, validate and make necessary changes
to DESI quasar catalogs. Can be imported without a need for MPI. """

import warnings

import fitsio
from healpy import ang2pix
import numpy as np
from numpy.lib.recfunctions import rename_fields, append_fields

from qsonic.mpi_utils import logging_mpi, balance_load, mpi_fnc_bcast

_accepted_extnames = set(['QSO_CAT', 'ZCATALOG', 'METADATA'])
"""set: Accepted extentions for quasar catalog."""
_required_columns = [
    set(['TARGETID']), set(['Z']), set(['TARGET_RA', 'RA']),
    set(['TARGET_DEC', 'DEC'])
]
"""list(set): Required columns for all cases."""
_required_data_columns = [
    set(['SURVEY']),
    set(['COADD_LASTNIGHT', 'LAST_NIGHT', 'LASTNIGHT'])
]
"""list(set): Required columns for real data analysis."""
_optional_columns = [
    'HPXPIXEL', 'VMIN_CIV_450', 'VMAX_CIV_450', 'VMIN_CIV_2000',
    'VMAX_CIV_2000'
]
"""list(str): Optional columns."""
_all_columns = [
    col for reqset in (
        _required_columns + _required_data_columns
    ) for col in reqset
] + _optional_columns


def read_quasar_catalog(
        filename, is_mock=False, keep_surveys=None,
        zmin=0, zmax=100.0):
    """ Returns a quasar catalog object (ndarray).

    It is sorted in the following order: HPXPIXEL, SURVEY (if applicable),
    TARGETID. BAL info included if available. It is required for BAL masking.
    If 'HPXPIXEL' column is not present, n_side is assumed 16 for mocks, 64 for
    data.

    Arguments
    ----------
    filename: str
        Filename to catalog.
    is_mock: bool, default: False
        If the catalog is for mocks.
    keep_surveys: None or list(str), default: None
        List of surveys to subselect. None keeps all.
    zmin: float, default: 0
        Minimum quasar redshift
    zmax: float, default: 100
        Maximum quasar redshift

    Returns
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Sorted catalog.
    """
    n_side = 16 if is_mock else 64
    catalog = _read(filename)
    catalog = _validate_adjust_column_names(catalog, is_mock)
    catalog = _prime_catalog(catalog, n_side, keep_surveys, zmin, zmax)

    return catalog


def mpi_read_quasar_catalog(
        filename, comm=None, mpi_rank=0, is_mock=False,
        keep_surveys=None, zmin=0, zmax=100
):
    """ Returns the same quasar catalog object on all MPI ranks.

    It is sorted in the following order: HPXPIXEL, SURVEY (if applicable),
    TARGETID. BAL info included if available. It is required for BAL masking.

    Can be used without MPI by passing ``comm=None`` and ``mpi_rank=0``.

    Arguments
    ----------
    filename: str
        Filename to catalog.
    comm: MPI comm object or None, default: None
        MPI comm object for bcast
    mpi_rank: int, default: 0
        Rank of the MPI process
    is_mock: bool, default: False
        If the catalog is for mocks.
    keep_surveys: None or list(str), default: None
        List of surveys to subselect. None keeps all.
    zmin: float, default: 0
        Minimum quasar redshift
    zmax: float, default: 100
        Maximum quasar redshift

    Returns
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Sorted catalog on all MPI ranks.

    Raises
    ------
    QsonicException
        If error occurs while reading the catalog.
    """
    catalog = mpi_fnc_bcast(
        read_quasar_catalog,
        comm, mpi_rank, "Error while reading catalog.",
        filename, is_mock, keep_surveys, zmin, zmax)

    return catalog


def mpi_get_local_queue(catalog, mpi_rank, mpi_size):
    """ Take a 'HPXPIXEL' sorted `catalog` and assign a list of catalogs to
    mpi_rank

    Arguments
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        'HPXPIXEL' sorted catalog.
    mpi_rank: int
        Rank of the MPI process
    mpi_size: int
        Size of MPI processes

    Returns
    ----------
    local_queue: list(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        List of sorted catalogs.
    """
    # We decide forest filename list
    # Group into unique pixels
    unique_pix, s = np.unique(catalog['HPXPIXEL'], return_index=True)
    split_catalog = np.split(catalog, s[1:])
    logging_mpi(
        f"There are {unique_pix.size} healpixels."
        " Don't use more MPI processes.", mpi_rank)

    # Roughly equal number of spectra
    logging_mpi("Load balancing.", mpi_rank)
    # Returns a list of catalog (ndarray)
    return balance_load(split_catalog, mpi_size, mpi_rank)


def _check_required_columns(required_cols, colnames):
    """Asserts all required columns are present.

    Arguments
    ----------
    required_cols: list(set)
        Required columns as list of sets.
    colnames: list(str)
        Present column names

    Raises
    ------
    Exception
        If none of a required set is in colnames.
    """
    for reqset in required_cols:
        if reqset.intersection(colnames):
            continue
        raise Exception(
            "One of these columns must be present in the catalog: "
            f"{', '.join(reqset)}!")


def _validate_adjust_column_names(catalog, is_mock):
    """ Validate `catalog` for required columns in `_required_columns`.
    'SURVEY' is also required for data. 'TARGET_{RA, DEC}' is transformed to
    'RA' and 'DEC' in return.

    Arguments
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog.
    is_mock: bool
        If the catalog is for mocks, does not perform 'SURVEY' check.

    Returns
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog. No checks are performed.
    """
    colnames = catalog.dtype.names
    # Check if required columns are present
    _check_required_columns(_required_columns, colnames)
    if not is_mock:
        _check_required_columns(_required_data_columns, colnames)

    logging_mpi(f"There are {catalog.size} quasars in the catalog.", 0)

    if catalog.size != np.unique(catalog['TARGETID']).size:
        raise Exception("There are duplicate TARGETIDs in catalog!")

    # Adjust column names
    colname_map = {}
    for x in ['RA', 'DEC']:
        if (f'TARGET_{x}' in colnames) and (x not in colnames):
            colname_map[f'TARGET_{x}'] = x
    if ('LASTNIGHT' not in colnames) and ('COADD_LASTNIGHT' in colnames):
        colname_map['COADD_LASTNIGHT'] = 'LASTNIGHT'
    elif ('LASTNIGHT' not in colnames) and ('LAST_NIGHT' in colnames):
        colname_map['LAST_NIGHT'] = 'LASTNIGHT'

    if colname_map:
        catalog = rename_fields(catalog, colname_map)

    return catalog


def _read(filename):
    """ Reads FITS file catalog with all columns present that are in
    `_all_columns`

    Arguments
    ----------
    filename: str

    Returns
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog. No checks are performed.
    """
    logging_mpi(f'Reading catalogue from {filename}', 0)
    fitsfile = fitsio.FITS(filename)
    extnames = [hdu.get_extname() for hdu in fitsfile]
    cat_hdu = _accepted_extnames.intersection(extnames)
    if not cat_hdu:
        cat_hdu = 1
        warnings.warn(
            "Catalog HDU not found by hduname. Using extension 1.",
            RuntimeWarning)
    else:
        cat_hdu = cat_hdu.pop()

    colnames = fitsfile[cat_hdu].get_colnames()
    keep_columns = [col for col in colnames if col in _all_columns]
    catalog = np.array(fitsfile[cat_hdu].read(columns=keep_columns))
    fitsfile.close()

    return catalog


def _add_healpix(catalog, n_side, keep_columns):
    """ Add 'HPXPIXEL' column to catalog if not present.

    Arguments
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog.
    n_side: int
        Healpix nside.
    keep_columns: list(str)
        List of surveys to subselect.

    Returns
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        'HPXPIXEL' calculated and added catalog.
    """
    if 'HPXPIXEL' not in keep_columns:
        pixnum = ang2pix(
            n_side, catalog['RA'], catalog['DEC'], lonlat=True, nest=True)
        catalog = append_fields(catalog, 'HPXPIXEL', pixnum, dtypes=int)

    return catalog


def _prime_catalog(catalog, n_side, keep_surveys, zmin, zmax):
    """ Returns quasar catalog object. It is sorted in the following order:
    HPXPIXEL, SURVEY (if applicable), TARGETID

    Arguments
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog.
    n_side: int
        Healpix nside.
    keep_surveys: list(str)
        List of surveys to subselect.
    zmin: float
        Minimum quasar redshift
    zmax: float
        Maximum quasar redshift

    Returns
    ----------
    catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Sorted catalog. BAL info included if available (req. for BAL masking)
    """
    colnames = catalog.dtype.names

    # Redshift cuts
    w = (catalog['Z'] >= zmin) & (catalog['Z'] <= zmax)
    logging_mpi(f"There are {w.sum()} quasars in the redshift range.", 0)
    catalog = catalog[w]

    sort_order = ['HPXPIXEL', 'TARGETID']
    # Filter all the objects in the catalogue not belonging to the specified
    # surveys.
    if 'SURVEY' in colnames and keep_surveys is not None:
        w = np.isin(catalog["SURVEY"], keep_surveys)
        catalog = catalog[w]
        if len(keep_surveys) > 1:
            sort_order.insert(1, 'SURVEY')
        logging_mpi(
            f"There are {w.sum()} quasars in given surveys {keep_surveys}.", 0)

    if catalog.size == 0:
        raise Exception("Empty quasar catalogue.")

    catalog = _add_healpix(catalog, n_side, colnames)
    catalog.sort(order=sort_order)

    return catalog
