import logging

import fitsio
from healpy import ang2pix
import numpy as np
from numpy.lib.recfunctions import rename_fields, append_fields

from qcfitter.mpi_utils import logging_mpi

_accepted_extnames = set(['QSO_CAT', 'ZCATALOG', 'METADATA'])
_extra_columns_tokeep = [
    'SURVEY', 'HPXPIXEL', 'VMIN_CIV_450',
    'VMAX_CIV_450', 'VMIN_CIV_2000', 'VMAX_CIV_2000'
]

def read_qso_catalog(filename, comm, n_side=64, keep_surveys=None, zmin=2.1, zmax=6.0):
    """Returns quasar catalog object. It is sorted in the following order:
    HPXPIXEL, SURVEY (if applicable), TARGETID

    Parameters
    ----------
    filename: str
        Filename to catalog.
    comm: MPI comm object
        MPI comm object for bcast
    n_side: int (default: 64)
        Healpix nside.
    keep_surveys: list (default: all)
        List of surveys to subselect.
    zmin: float (default: 2.1)
        Minimum quasar redshift
    zmax: float (default: 6.0)
        Maximum quasar redshift
    """
    catalog = None

    if comm.Get_rank() == 0:
        catalog = _readCatalogue(filename, n_side, keep_surveys, zmin, zmax)

    catalog = comm.bcast(catalog, root=0)

def _readCatalogue(filename, n_side, keep_surveys, zmin, zmax):
    logging_mpi(f'Reading catalogue from {filename}', 0)
    fitsfile = fitsio.FITS(filename)
    extnames = [hdu.get_extname() for hdu in fitsfile]
    cat_hdu = _accepted_extnames.intersection(extnames)
    if not cat_hdu:
        cat_hdu = 1
        logging_mpi("Catalog HDU not found by hduname. Using extension 1.", 0, "warning")
    else:
        cat_hdu = cat_hdu.pop()

    colnames = fitsfile[cat_hdu].get_colnames()
    catalog = np.array(fitsfile[cat_hdu].read())

    logging_mpi(f"There are {catalog.size} quasars in the catalog.", 0)

    # Redshift cuts
    w = (catalog['Z'] >= zmin) & (catalog['Z'] <= zmax)
    logging_mpi(f"There are {w.sum()} quasars in the redshift range.", 0)
    catalog = catalog[w]

    if 'TARGET_RA' in colnames:
        catalog = rename_fields(catalog, {'TARGET_RA':'RA', 'TARGET_DEC':'DEC'} )
    keep_columns = ['RA', 'DEC', 'Z', 'TARGETID']
    keep_columns.extend([_col for _col in _extra_columns_tokeep if _col in colnames])

    catalog = catalog[keep_columns]

    sort_order = ['HPXPIXEL', 'TARGETID']
    # Filter all the objects in the catalogue not belonging to the specified
    # surveys.
    if 'SURVEY' in keep_columns and keep_surveys is not None:
        w = np.isin(catalog["SURVEY"], keep_surveys)
        catalog = catalog[w]
        if len(keep_surveys) > 1:
            sort_order.insert(1, 'SURVEY')
        logging_mpi(f"There are {w.sum()} quasars in given surveys {keep_surveys}.", 0)

    if catalog.size == 0:
        raise Exception("Empty quasar catalogue.")

    if not 'HPXPIXEL' in keep_columns:
        pixnum = ang2pix(n_side, catalog['RA'], catalog['DEC'], lonlat=True, nest=True)
        catalog = append_fields(catalog, 'HPXPIXEL', pixnum, dtypes=int)

    catalog.sort(order=sort_order)

    return catalog



        