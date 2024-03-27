import argparse
import pytest
import shutil

import fitsio
import numpy as np
import numpy.testing as npt

import qsonic.io


class TestIOParsers(object):
    def test_add_io_parser(self):
        parser = argparse.ArgumentParser()
        qsonic.io.add_io_parser(parser)
        text = "--input-dir indir --catalog incat -o outdir"
        args = parser.parse_args(text.split(' '))
        assert (args.input_dir == "indir")
        assert (args.catalog == "incat")
        assert (args.outdir == "outdir")

        with pytest.raises(SystemExit):
            text = "--catalog incat -o outdir"
            parser.parse_args(text.split(' '))

        with pytest.raises(SystemExit):
            text = "--input-dir indir --catalog incat --arms T"
            parser.parse_args(text.split(' '))

        with pytest.raises(SystemExit):
            text = "--input-dir indir --catalog incat --skip 1.2"
            parser.parse_args(text.split(' '))


class TestIOReading(object):
    def test_read_onehealpix_file_data_coadd(self, my_setup_fits):
        cat_by_survey, input_dir, xarms, indata = my_setup_fits

        spectra_list = qsonic.io.read_onehealpix_file_data_coadd(
            cat_by_survey, input_dir, xarms, skip_resomat=True)

        for key, inval in indata.items():
            # assert (key in outdata)
            if key == 'reso':
                assert np.all((s.reso is {} for s in spectra_list))
                assert (not inval)
                continue

            for arm in xarms:
                assert np.all(arm in s.wave for s in spectra_list)
                assert (arm in inval)
                # npt.assert_allclose(outdata[key][arm], inval[arm])

    def test_save_deltas(self):
        qsonic.io.save_deltas([], "", None)

        expected_msg = "save_by_hpx and mpi_rank can't both be None."
        with pytest.raises(Exception, match=expected_msg):
            qsonic.io.save_deltas([], "outdir", None)

    def test_read_spectra_onehealpix(self, my_setup_fits):
        cat_by_survey, input_dir, xarms, data = my_setup_fits

        slist = qsonic.io.read_onehealpix_file_data_coadd(
            cat_by_survey, input_dir, xarms, True)

        assert (len(slist) == cat_by_survey.size)
        for jj, spec in enumerate(slist):
            for arm in xarms:
                npt.assert_allclose(spec.wave[arm], data['wave'][arm])
                npt.assert_allclose(spec.flux[arm], data['flux'][arm][jj])
                npt.assert_allclose(spec.ivar[arm], data['ivar'][arm][jj])

        # Test extra targetid in catalog
        ens_ = 3
        cat_by_survey2 = np.concatenate((cat_by_survey, cat_by_survey[-ens_:]))
        cat_by_survey2[-ens_:]['TARGETID'] += 20
        npt.assert_array_equal(cat_by_survey, cat_by_survey2[:-ens_])
        with pytest.warns(RuntimeWarning):
            slist = qsonic.io.read_onehealpix_file_data_coadd(
                cat_by_survey2, input_dir, xarms, True)

        assert (len(slist) == cat_by_survey.size)
        for jj, spec in enumerate(slist):
            for arm in xarms:
                npt.assert_allclose(spec.wave[arm], data['wave'][arm])
                npt.assert_allclose(spec.flux[arm], data['flux'][arm][jj])
                npt.assert_allclose(spec.ivar[arm], data['ivar'][arm][jj])

    def test_read_onehealpix_file_data_uncoadd(self, my_setup_fits_spectra):
        cat_by_survey, input_dir, xarms, data = my_setup_fits_spectra
        _, w = np.unique(cat_by_survey['TARGETID'], return_index=True)

        slist = qsonic.io.read_onehealpix_file_data_uncoadd(
            cat_by_survey[w][['TARGETID', 'Z', 'HPXPIXEL', 'SURVEY']],
            input_dir, xarms, True)

        new_catalog = np.hstack([_.catrow for _ in slist])
        for key in new_catalog.dtype.names:
            assert all(new_catalog[key] == cat_by_survey[key])

        assert (len(slist) == cat_by_survey.size)
        for jj, spec in enumerate(slist):
            assert (spec.targetid == cat_by_survey['TARGETID'][jj])
            for arm in xarms:
                npt.assert_allclose(spec.wave[arm], data['wave'][arm])
                npt.assert_allclose(spec.flux[arm], data['flux'][arm][jj])
                npt.assert_allclose(spec.ivar[arm], data['ivar'][arm][jj])


@pytest.fixture()
def my_setup_fits(tmp_path, setup_data):
    cat_by_survey, _, data = setup_data(5)
    pixnum = 8258
    cat_by_survey['HPXPIXEL'] = pixnum
    xarms = data['wave'].keys()

    d = tmp_path / "main"
    d.mkdir()
    d = d / "dark"
    d.mkdir()
    d = d / f"{pixnum//100}"
    d.mkdir()
    d = d / f"{pixnum}"
    d.mkdir()
    fname = d / "coadd-main-dark-8258.fits"

    with fitsio.FITS(fname, 'rw', clobber=True) as fts:
        fts.write(cat_by_survey, extname="FIBERMAP")
        for arm in xarms:
            shape = data['flux'][arm].shape
            fts.write(data['wave'][arm], extname=f"{arm}_WAVELENGTH")
            fts.write(data['flux'][arm], extname=f"{arm}_FLUX")
            fts.write(data['ivar'][arm], extname=f"{arm}_IVAR")
            fts.write(np.zeros(shape, dtype='i4'), extname=f"{arm}_MASK")

    # return sorted data truth
    # Sort the generated catalog first.
    sort_idx = np.argsort(cat_by_survey, order="TARGETID")
    cat_by_survey = cat_by_survey[sort_idx]
    for key in ['flux', 'ivar', 'mask']:
        for arm in xarms:
            data[key][arm] = data[key][arm][sort_idx]

    yield cat_by_survey, tmp_path, xarms, data

    shutil.rmtree(tmp_path / "main")


@pytest.fixture()
def my_setup_fits_spectra(tmp_path, setup_data):
    cat_by_survey, _, data = setup_data(5)
    pixnum = 8259
    cat_by_survey['HPXPIXEL'] = pixnum
    cat_by_survey['TARGETID'][:3] = cat_by_survey['TARGETID'][0]
    cat_by_survey['TARGETID'][3:] = cat_by_survey['TARGETID'][3]
    cat_by_survey['EXPID'][:3] = np.arange(3)
    cat_by_survey['EXPID'][3:] = np.arange(2)
    xarms = data['wave'].keys()

    d = tmp_path / "main"
    d.mkdir()
    d = d / "dark"
    d.mkdir()
    d = d / f"{pixnum//100}"
    d.mkdir()
    d = d / f"{pixnum}"
    d.mkdir()
    fname = d / "spectra-main-dark-8259.fits"

    with fitsio.FITS(fname, 'rw', clobber=True) as fts:
        fts.write(cat_by_survey, extname="FIBERMAP")
        for arm in xarms:
            shape = data['flux'][arm].shape
            fts.write(data['wave'][arm], extname=f"{arm}_WAVELENGTH")
            fts.write(data['flux'][arm], extname=f"{arm}_FLUX")
            fts.write(data['ivar'][arm], extname=f"{arm}_IVAR")
            fts.write(np.zeros(shape, dtype='i4'), extname=f"{arm}_MASK")

    # return sorted data truth
    # Sort the generated catalog first.
    sort_idx = np.argsort(cat_by_survey, order="TARGETID")
    cat_by_survey = cat_by_survey[sort_idx]
    for key in ['flux', 'ivar', 'mask']:
        for arm in xarms:
            data[key][arm] = data[key][arm][sort_idx]

    yield cat_by_survey, tmp_path, xarms, data

    shutil.rmtree(tmp_path / "main")


if __name__ == '__main__':
    pytest.main()
