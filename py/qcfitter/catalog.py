import logging

import fitsio
from healpy import ang2pix
import numpy as np
from numpy.lib.recfunctions import rename_fields, append_fields

from qcfitter.mpi_utils import logging_mpi

class Catalog(object):
    """Quasar catalog object.

    Parameters
    ----------
    filename: str
        Filename to catalog.
    n_side: int (default: 64)
        Healpix nside.
    keep_surveys: list (default: all)
        List of surveys to subselect.
    zmin: float (default: 2.1)
        Minimum quasar redshift
    zmax: float (default: 6.0)
        Maximum quasar redshift

    Methods
    ----------
    readCatalogue()

    """
    _accepted_extnames = set(['QSO_CAT', 'ZCATALOG', 'METADATA'])

    def __init__(self, filename, n_side=64, keep_surveys=None,
        zmin=2.1, zmax=6.0, mpi_rank=0):
        self.n_side = n_side
        self.keep_surveys = keep_surveys
        self.zmin = zmin
        self.zmax = zmax
        self.catalog = None
        self.mpi_rank = mpi_rank

        self._readCatalogue(filename)

    def _readCatalogue(self, filename):
        logging_mpi(f'Reading catalogue from {filename}', self.mpi_rank)
        fitsfile = fitsio.FITS(filename)
        extnames = [hdu.get_extname() for hdu in fitsfile]
        cat_hdu = Catalog._accepted_extnames.intersection(extnames)
        if not cat_hdu:
            cat_hdu = 1
            logging_mpi("Catalog HDU not found by hduname. Using extension 1.", self.mpi_rank, "warning")
        else:
            cat_hdu = cat_hdu.pop()

        colnames = fitsfile[cat_hdu].get_colnames()
        self.catalog = np.array(fitsfile[cat_hdu].read())

        logging_mpi(f"There are {self.catalog.size} quasars in the catalog.", self.mpi_rank)

        # Redshift cuts
        w = (self.catalog['Z'] >= self.zmin) & (self.catalog['Z'] <= self.zmax)
        logging_mpi(f"There are {w.sum()} quasars in the redshift range.", self.mpi_rank)
        self.catalog = self.catalog[w]

        if 'TARGET_RA' in colnames:
            self.catalog = rename_fields(self.catalog, {'TARGET_RA':'RA', 'TARGET_DEC':'DEC'} )
        keep_columns = ['RA', 'DEC', 'Z', 'TARGETID']
        for _col in ['SURVEY', 'HPXPIXEL']:
            if _col in colnames:
                keep_columns.append(_col)

        self.catalog = self.catalog[keep_columns]

        # Filter all the objects in the catalogue not belonging to the specified
        # surveys.
        if 'SURVEY' in keep_columns and self.keep_surveys is not None:
            w = np.isin(self.catalog["SURVEY"], self.keep_surveys)
            self.catalog = self.catalog[w]
            logging_mpi(f"There are {w.sum()} quasars in given surveys {self.keep_surveys}.", self.mpi_rank)

        if self.catalog.size == 0:
            raise Exception("Empty quasar catalogue.")

        if not 'HPXPIXEL' in keep_columns:
            pixnum = ang2pix(self.n_side, self.catalog['RA'], self.catalog['DEC'], lonlat=True, nest=True)
            self.catalog = append_fields(self.catalog, 'HPXPIXEL', pixnum, dtypes=int)
        self.catalog.sort(order='HPXPIXEL')

        

        