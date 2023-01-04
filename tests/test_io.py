import argparse
import pytest

import fitsio
import numpy as np
import numpy.testing as npt

import qcfitter.io


class TestIOParsers(object):
    def test_add_io_parser(self):
        parser = argparse.ArgumentParser()
        qcfitter.io.add_io_parser(parser)
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
        cat_by_survey, input_dir, pixnum, arms_to_keep, b_wave = my_setup_fits
        nsize = b_wave.size

        data, _ = qcfitter.io.read_onehealpix_file_data(
            cat_by_survey, input_dir, pixnum, arms_to_keep, skip_resomat=True)

        assert (len(data['wave']) == 1)
        assert (len(data['flux']) == 1)
        assert (data['wave']['B'].size == nsize)
        assert (data['flux']['B'].shape == (2, nsize))
        npt.assert_allclose(data['wave']['B'], b_wave)
        npt.assert_allclose(data['flux']['B'], 1)
        npt.assert_allclose(data['ivar']['B'], 2)

    def test_save_deltas(self):
        qcfitter.io.save_deltas([], "", None)

        expected_msg = "save_by_hpx and mpi_rank can't both be None."
        with pytest.raises(Exception, match=expected_msg):
            qcfitter.io.save_deltas([], "outdir", None)

    # Including this test fails spectrum tests?
    # def test_read_spectra(self, my_setup_fits):
    #     cat_by_survey, input_dir, _, arms_to_keep, b_wave = my_setup_fits

    #     slist = qcfitter.io.read_spectra(
    #         cat_by_survey, input_dir, arms_to_keep, False, True)

    #     assert (len(slist) == 2)
    #     for spec in slist:
    #         npt.assert_allclose(spec.wave['B'], b_wave)
    #         npt.assert_allclose(spec.flux['B'], 1)
    #         npt.assert_allclose(spec.ivar['B'], 2)


@pytest.fixture
def my_setup_fits(tmp_path):
    cat_dtype = np.dtype([
        ('TARGETID', '>i8'), ('Z', '>f8'), ('RA', '>f8'), ('DEC', '>f8'),
        ('HPXPIXEL', '>i8'), ('SURVEY', '<U4')])
    pixnum = 8258
    arms_to_keep = ['B']
    cat_by_survey = np.array([
        (39627939372861215, 2.328, 229.861, 6.1925, pixnum, b'main'),
        (39627939372861216, 2.328, 229.861, 6.1925, pixnum, b'main')],
        dtype=cat_dtype)

    d = tmp_path / "main"
    d.mkdir()
    d = d / "dark"
    d.mkdir()
    d = d / f"{pixnum//100}"
    d.mkdir()
    d = d / f"{pixnum}"
    d.mkdir()
    fname = d / "coadd-main-dark-8258.fits"
    nsize = 1000

    b_wave = 3600. + 0.8 * np.arange(nsize)
    with fitsio.FITS(fname, 'rw', clobber=True) as fts:
        fts.write(cat_by_survey, extname="FIBERMAP")
        fts.write(b_wave, extname="B_WAVELENGTH")
        fts.write(np.ones((2, nsize)), extname="B_FLUX")
        fts.write(2 * np.ones((2, nsize)), extname="B_IVAR")
        fts.write(np.zeros((2, nsize), dtype='i4'), extname="B_MASK")

    return cat_by_survey, tmp_path, pixnum, arms_to_keep, b_wave


if __name__ == '__main__':
    pytest.main()
