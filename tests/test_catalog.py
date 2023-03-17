import pytest
import os

import fitsio
import numpy as np
from numpy.lib.recfunctions import drop_fields
import numpy.testing as npt

import qsonic.catalog


class TestCatalog(object):
    def test_read_validate_adjust_column_names(self, tmp_path):
        # Test typo in SURVEY and DUMMY column
        cat_dtype = np.dtype([
            ('TARGETID', '>i8'), ('Z', '>f8'), ('TARGET_RA', '>f8'),
            ('DEC', '>f8'), ('HPXPIXEL', '>i8'), ('SURVEX', '<U4'),
            ('DUMMY', 'f4')])
        input_catalog = np.array([
            (111, 1.1, 13.1, 13.1, 1000, b'main', 1.6),
            (222, 2.1, 23.1, 23.1, 2000, b'main', 2.6)],
            dtype=cat_dtype)

        fname = tmp_path / "test_read_validate_adjust_column_names.fits"
        with fitsio.FITS(fname, 'rw', clobber=True) as fts:
            fts.write(
                input_catalog,
                extname=list(qsonic.catalog._accepted_extnames)[0])

        catalog = qsonic.catalog._read(fname)
        os.remove(fname)

        colnames = catalog.dtype.names
        assert ('DUMMY' not in colnames)
        assert ('SURVEY' not in colnames)
        assert ('RA' not in colnames)
        assert ('DEC' in colnames)
        assert ('TARGET_RA' in colnames)
        npt.assert_equal(input_catalog['TARGETID'], catalog['TARGETID'])
        npt.assert_almost_equal(input_catalog['Z'], catalog['Z'])

        catalog1 = qsonic.catalog._validate_adjust_column_names(
            catalog.copy(), is_mock=True)
        colnames = catalog1.dtype.names
        assert ('SURVEY' not in colnames)
        assert ('RA' in colnames)
        assert ('DEC' in colnames)
        assert ('TARGET_RA' not in colnames)

        expected_msg = (
            "One of these columns must be present in the catalog: SURVEY!")
        with pytest.raises(Exception, match=expected_msg):
            qsonic.catalog._validate_adjust_column_names(
                catalog, is_mock=False)

    def test_prime_catalog(self):
        cat_dtype = np.dtype([
            ('TARGETID', '>i8'), ('Z', '>f8'), ('RA', '>f8'), ('DEC', '>f8'),
            ('HPXPIXEL', '>i8'), ('SURVEY', '<U4')])
        input_catalog = np.array([
            (999, 1.1, 93.1, 93.1, 1000, b'main'),
            (666, 3.1, 63.1, 63.1, 2000, b'main'),
            (668, 3.1, 63.1, 63.1, 2000, b'main'),
            (667, 3.1, 63.1, 63.1, 2000, b'main'),
            (777, 3.0, 73.1, 73.1, 1000, b'sv1'),
            (778, 3.0, 73.1, 73.1, 1000, b'main'),
            (779, 3.0, 73.1, 73.1, 1000, b'sv1'),
            (111, 8.1, 13.1, 13.1, 1000, b'main'),
            (444, 4.1, 43.1, 43.1, 1000, b'main')],
            dtype=cat_dtype)

        nside = 64
        keep_surveys = ['sv1', 'main']
        zmin, zmax = 2.1, 7.1
        # Sort order HPXPIXEL, SURVEY, TARGETID
        expected_catalog = np.array([
            (444, 4.1, 43.1, 43.1, 1000, b'main'),
            (778, 3.0, 73.1, 73.1, 1000, b'main'),
            (777, 3.0, 73.1, 73.1, 1000, b'sv1'),
            (779, 3.0, 73.1, 73.1, 1000, b'sv1'),
            (666, 3.1, 63.1, 63.1, 2000, b'main'),
            (667, 3.1, 63.1, 63.1, 2000, b'main'),
            (668, 3.1, 63.1, 63.1, 2000, b'main')],
            dtype=cat_dtype)
        catalog1 = qsonic.catalog._prime_catalog(
            input_catalog.copy(), nside, keep_surveys, zmin, zmax)
        npt.assert_array_equal(catalog1, expected_catalog)

        # Sort order HPXPIXEL, TARGETID
        input_catalog = drop_fields(input_catalog, 'SURVEY')
        expected_catalog = np.array([
            (444, 4.1, 43.1, 43.1, 1000),
            (777, 3.0, 73.1, 73.1, 1000),
            (778, 3.0, 73.1, 73.1, 1000),
            (779, 3.0, 73.1, 73.1, 1000),
            (666, 3.1, 63.1, 63.1, 2000),
            (667, 3.1, 63.1, 63.1, 2000),
            (668, 3.1, 63.1, 63.1, 2000)],
            dtype=input_catalog.dtype)
        catalog1 = qsonic.catalog._prime_catalog(
            input_catalog.copy(), nside, keep_surveys, zmin, zmax)
        npt.assert_array_equal(catalog1, expected_catalog)

    def test_mpi_get_local_queue(self):
        cat_dtype = np.dtype([
            ('TARGETID', '>i8'), ('Z', '>f8'), ('RA', '>f8'), ('DEC', '>f8'),
            ('HPXPIXEL', '>i8'), ('SURVEY', '<U4')])

        # Must be sorted in HPXPIXEL
        input_catalog = np.array([
            (444, 4.1, 43.1, 43.1, 1000, b'main'),
            (778, 3.0, 73.1, 73.1, 1001, b'main'),
            (777, 3.0, 73.1, 73.1, 1001, b'sv1'),
            (779, 3.0, 73.1, 73.1, 1003, b'sv1'),
            (666, 3.1, 63.1, 63.1, 2000, b'main'),
            (667, 3.1, 63.1, 63.1, 2000, b'main'),
            (668, 3.1, 63.1, 63.1, 2000, b'main')],
            dtype=cat_dtype)

        mpi_size = 2
        q0 = qsonic.catalog.mpi_get_local_queue(input_catalog, 0, mpi_size)
        q1 = qsonic.catalog.mpi_get_local_queue(input_catalog, 1, mpi_size)
        assert (len(q0) == 2)
        assert (len(q1) == 2)
        npt.assert_equal(q0[0]['TARGETID'], [666, 667, 668])
        npt.assert_equal(q1[0]['TARGETID'], [778, 777])
