import argparse
import copy
import pytest

import fitsio
import numpy as np
import numpy.testing as npt

import qsonic.spectrum
from qsonic.mathtools import FastLinear1DInterp


class TestSpecParsers(object):
    def test_add_wave_region_parser(self):
        parser = argparse.ArgumentParser()

        qsonic.spectrum.add_wave_region_parser(parser)
        assert parser.parse_args([])

        args = parser.parse_args(["--forest-w1", "1050"])
        npt.assert_almost_equal(args.forest_w1, 1050.)


class TestSpectrum(object):
    def test_generate_spectra_list_from_data(self, setup_data):
        cat_by_survey, _, data = setup_data(2)
        spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)

        assert (len(spectra_list) == 2)
        for idx, (catrow, spec) in enumerate(zip(cat_by_survey, spectra_list)):
            assert (spec.targetid == catrow['TARGETID'])
            npt.assert_almost_equal(spec.z_qso, catrow['Z'])
            npt.assert_almost_equal(spec.ra, catrow['RA'])
            npt.assert_almost_equal(spec.dec, catrow['DEC'])
            npt.assert_almost_equal(spec.dwave, 0.8)
            npt.assert_almost_equal(spec.cont_params['x'], [1., 0.])

            npt.assert_allclose(spec.flux['B'], data['flux']['B'][idx])
            npt.assert_allclose(spec.flux['R'], data['flux']['R'][idx])
            assert (not spec.forestflux)
            assert (not spec.reso)

    def test_set_forest_region(self, setup_data):
        cat_by_survey, _, data = setup_data(1)
        spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)

        spec = spectra_list[0]
        spec.set_forest_region(3600., 6000., 1050., 1180.)

        npt.assert_almost_equal(spec.rsnr, 2.1)
        npt.assert_almost_equal(spec.mean_snr['B'], 2.1)
        npt.assert_almost_equal(spec.cont_params['x'], [2.1, 0.])
        npt.assert_allclose(spec.forestflux['B'], 2.1)
        npt.assert_allclose(spec.forestivar['B'], spec.forestivar_sm['B'])
        assert ('R' not in spec.forestflux.keys())
        assert (not spec.forestreso)

        spec.flux['B'][10:15] = 1e2
        spec.set_forest_region(3600., 6000., 1050., 1180.)
        npt.assert_equal(spec.forestflux['B'][10:15], 0)

    def test_drop_short_arms(self, setup_data):
        cat_by_survey, _, data = setup_data(1)
        spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)

        spec = spectra_list[0]
        spec.set_forest_region(3600., 6000., 1050., 1210.)

        assert ('R' in spec.forestflux.keys())
        assert (not spec.cont_params['cont'])

        spec.drop_short_arms(1050., 1180., 0.5)
        assert ('R' not in spec.forestflux.keys())

    def test_get_real_size(self, setup_data):
        cat_by_survey, npix, data = setup_data(1)
        spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)

        spec = spectra_list[0]
        npt.assert_equal(spec.get_real_size(), 0)

        spec.set_forest_region(3600., 6000., 1000., 2000.)
        npt.assert_equal(spec.get_real_size(), 2 * npix)

        spec.set_forest_region(3600., 4400., 1000., 2000.)
        npt.assert_equal(spec.get_real_size(), 1.5 * npix)

    def test_is_long(self, setup_data):
        cat_by_survey, _, data = setup_data(1)
        spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)

        dforest_wave = 1180. - 1050.
        skip_ratio = 0.5

        spec = spectra_list[0]
        assert (not spec.is_long(dforest_wave, skip_ratio))

        spec.set_forest_region(3600., 6000., 1000., 2000.)
        assert (spec.is_long(dforest_wave, skip_ratio))

        spec._forestivar = {}
        spec.set_forest_region(3600., 6000., 1120., 1130.)
        assert (not spec.is_long(dforest_wave, skip_ratio))

    def test_coadd_arms_forest(self, setup_data):
        cat_by_survey, _, data = setup_data(1)
        spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)

        # var_lss zero
        varlss_interp = FastLinear1DInterp(0, 1, np.zeros(3))
        spec = copy.deepcopy(spectra_list[0])
        spec.set_forest_region(3600., 6000., 1050., 1300.)
        spec.cont_params['valid'] = True
        spec.cont_params['cont'] = {
            arm: np.ones_like(farm) for arm, farm in spec.forestflux.items()
        }
        spec.set_forest_weight(varlss_interp)
        spec.coadd_arms_forest(varlss_interp)

        assert ('brz' in spec.forestflux.keys())
        npt.assert_almost_equal(spec.forestflux['brz'], 2.1)
        npt.assert_almost_equal(spec.cont_params['cont']['brz'], 1.)
        w = (spec.forestwave['brz'] < data['wave']['R'][0])\
            | (spec.forestwave['brz'] > data['wave']['B'][-1])
        npt.assert_almost_equal(spec.forestivar['brz'][w], 1)
        npt.assert_almost_equal(spec.forestivar['brz'][~w], 2)

        # var_lss non-zero
        varlss_interp = FastLinear1DInterp(0, 1, 0.5 * np.ones(3))
        spec = copy.deepcopy(spectra_list[0])
        spec.set_forest_region(3600., 6000., 1050., 1300.)
        spec.cont_params['valid'] = True
        spec.cont_params['cont'] = {
            arm: np.ones_like(farm) for arm, farm in spec.forestflux.items()
        }
        spec.set_forest_weight(varlss_interp)
        spec.coadd_arms_forest(varlss_interp)

        assert ('brz' in spec.forestflux.keys())
        npt.assert_almost_equal(spec.forestflux['brz'], 2.1)
        npt.assert_almost_equal(spec.cont_params['cont']['brz'], 1.)

        wbonly = int(
            (data['wave']['R'][0] - spec.forestwave['brz'][0]) / spec.dwave
            + 0.1)
        npt.assert_almost_equal(spec.forestivar['brz'][:wbonly], 1)

        wronly = int(
            (data['wave']['B'][-1] - spec.forestwave['brz'][0]) / spec.dwave
            + 0.1) + 1
        npt.assert_almost_equal(spec.forestivar['brz'][wronly:], 1)

        npt.assert_almost_equal(spec.forestivar['brz'][wbonly:wronly], 2)

    def test_simple_coadd(self, setup_data):
        cat_by_survey, _, data = setup_data(1)
        spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)

        spec = spectra_list[0]
        spec.simple_coadd()

        assert (len(spec.wave.keys()) == 1)
        assert ('brz' in spec.wave.keys())
        assert ('brz' in spec.flux.keys())
        npt.assert_equal(spec.wave['brz'].size, spec.flux['brz'].size)
        npt.assert_equal(spec.wave['brz'].size, spec.ivar['brz'].size)
        npt.assert_almost_equal(
            spec.wave['brz'][[0, -1]],
            [data['wave']['B'][0], data['wave']['R'][-1]])

        npt.assert_almost_equal(spec.flux['brz'], 2.1)

        wbonly = int(
            (data['wave']['R'][0] - spec.wave['brz'][0]) / spec.dwave + 0.1
        )
        npt.assert_almost_equal(spec.ivar['brz'][:wbonly], 1)

        wronly = int(
            (data['wave']['B'][-1] - spec.wave['brz'][0]) / spec.dwave + 0.1
        ) + 1
        npt.assert_almost_equal(spec.ivar['brz'][wronly:], 1)

        npt.assert_almost_equal(spec.ivar['brz'][wbonly:wronly], 2)


class TestDelta(object):
    def test_delta_init(self, setup_delta_data):
        fname = setup_delta_data
        with fitsio.FITS(fname) as fts:
            delta = qsonic.spectrum.Delta(fts[1])

        wave = delta.wave.copy()
        npt.assert_almost_equal(delta.delta, 1)
        npt.assert_almost_equal(delta.ivar, 1)
        npt.assert_almost_equal(delta.weight, 0.5)
        npt.assert_almost_equal(delta.cont, 1)
        npt.assert_almost_equal(delta.reso, 1)

        with fitsio.FITS(fname) as fts:
            delta2 = qsonic.spectrum.Delta(fts[1])

        delta.delta *= 0.4
        delta2.delta *= 0.6
        delta2.ivar[:5] = 0
        delta2.weight[:5] = 0
        delta.coadd(delta2)
        npt.assert_almost_equal(delta.ivar[:5], 1)
        npt.assert_almost_equal(delta.delta[:5], 0.4)
        npt.assert_almost_equal(delta.wave, wave)
        npt.assert_almost_equal(delta.delta[5:], 0.5)
        npt.assert_almost_equal(delta.cont[5:], 1)
        npt.assert_almost_equal(delta.ivar[5:], 2)
        npt.assert_almost_equal(delta.reso[5:], 1)
        npt.assert_almost_equal(delta.weight[5:], 2. / 3.)

        with fitsio.FITS(fname, 'rw', clobber=True) as fts:
            delta.write(fts)

        with fitsio.FITS(fname) as fts:
            delta2 = qsonic.spectrum.Delta(fts[1])

        npt.assert_almost_equal(delta2.delta, delta.delta)
        npt.assert_almost_equal(delta2.wave, wave)


@pytest.fixture
def setup_delta_data(tmp_path):
    fname = tmp_path / "delta-0.fits"

    hdr_dict = {
        'TARGETID': 39627939372861215,
        'Z': 2.328,
        'MEANSNR': 1
    }

    sigma = 1
    sigma_lss = 1
    ndata = 100
    ivar = np.ones(ndata) / sigma**2
    weight = np.ones(ndata) / (sigma**2 + sigma_lss**2)

    wave_arm = 3800. + 0.8 * np.arange(ndata)
    delta = np.ones(ndata)
    cont_est = np.ones(ndata)
    reso = np.ones((11, ndata))
    with fitsio.FITS(fname, 'rw', clobber=True) as fts:
        cols = [wave_arm, delta, ivar, weight, cont_est, reso.T.astype('f8')]

        fts.write(
            cols,
            names=['LAMBDA', 'DELTA', 'IVAR', 'WEIGHT', 'CONT', 'RESOMAT'],
            header=hdr_dict,
            extname=f"{hdr_dict['TARGETID']}")

    yield fname


if __name__ == '__main__':
    pytest.main()
