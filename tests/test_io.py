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
    def test_read_onehealpix_file_data(self, my_setup_fits):
        cat_by_survey, input_dir, xarms, indata = my_setup_fits

        outdata, _ = qsonic.io.read_onehealpix_file_data(
            cat_by_survey, input_dir, xarms, skip_resomat=True)

        for key, inval in indata.items():
            assert (key in outdata)
            if key == 'reso':
                assert (not outdata[key])
                assert (not inval)
                continue

            for arm in xarms:
                assert (arm in outdata[key])
                assert (arm in inval)
                npt.assert_allclose(outdata[key][arm], inval[arm])

    def test_save_deltas(self):
        qsonic.io.save_deltas([], "", None)

        expected_msg = "save_by_hpx and mpi_rank can't both be None."
        with pytest.raises(Exception, match=expected_msg):
            qsonic.io.save_deltas([], "outdir", None)

    def test_read_spectra_onehealpix(self, my_setup_fits):
        cat_by_survey, input_dir, xarms, data = my_setup_fits

        slist = qsonic.io.read_spectra_onehealpix(
            cat_by_survey, input_dir, xarms, False, True)

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
            slist = qsonic.io.read_spectra_onehealpix(
                cat_by_survey2, input_dir, xarms, False, True)

        assert (len(slist) == cat_by_survey.size)
        for jj, spec in enumerate(slist):
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


if __name__ == '__main__':
    pytest.main()
