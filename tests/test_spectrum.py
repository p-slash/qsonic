import argparse
import copy
import pytest

import numpy as np
import numpy.testing as npt

import qcfitter.spectrum
from qcfitter.mathtools import Fast1DInterpolator


class TestSpecParsers(object):
    def setup_method(self):
        self.parser = argparse.ArgumentParser()

    def test_add_wave_region_parser(self):
        qcfitter.spectrum.add_wave_region_parser(self.parser)
        assert self.parser.parse_args([])

        args = self.parser.parse_args(["--forest-w1", "1050"])
        npt.assert_almost_equal(args.forest_w1, 1050.)


class TestSpectrum(object):
    def test_generate_spectra_list_from_data(self, setup_data):
        cat_by_survey, _, data, spectra_list = setup_data
        assert (len(spectra_list) == 1)
        spec = spectra_list[0]

        assert (spec.targetid == cat_by_survey['TARGETID'])
        npt.assert_almost_equal(spec.z_qso, cat_by_survey['Z'])
        npt.assert_almost_equal(spec.ra, cat_by_survey['RA'])
        npt.assert_almost_equal(spec.dec, cat_by_survey['DEC'])
        npt.assert_almost_equal(spec.dwave, 0.8)
        npt.assert_almost_equal(spec.cont_params['x'], [1., 0.])

        npt.assert_allclose(spec.flux['B'], data['flux']['B'][0])
        npt.assert_allclose(spec.flux['R'], data['flux']['R'][0])
        assert (not spec.forestflux)
        assert (not spec.reso)

    def test_set_forest_region(self, setup_data):
        _, _, _, spectra_list = setup_data
        spec = copy.deepcopy(spectra_list[0])
        spec.set_forest_region(3600., 6000., 1050., 1180.)

        npt.assert_almost_equal(spec.rsnr, 2.1)
        npt.assert_almost_equal(spec.mean_snr(), 2.1)
        npt.assert_almost_equal(spec.cont_params['x'], [2.1, 0.])
        npt.assert_allclose(spec.forestflux['B'], 2.1)
        npt.assert_allclose(spec.forestivar['B'], spec.forestivar_sm['B'])
        assert ('R' not in spec.forestflux.keys())
        assert (not spec.forestreso)

    def test_drop_short_arms(self, setup_data):
        _, _, _, spectra_list = setup_data
        spec = copy.deepcopy(spectra_list[0])
        spec.set_forest_region(3600., 6000., 1050., 1210.)

        assert ('R' in spec.forestflux.keys())
        assert (not spec.cont_params['cont'])

        spec.drop_short_arms(1050., 1180., 0.5)
        assert ('R' not in spec.forestflux.keys())

    def test_get_real_size(self, setup_data):
        _, npix, _, spectra_list = setup_data
        spec = copy.deepcopy(spectra_list[0])
        npt.assert_equal(spec.get_real_size(), 0)

        spec.set_forest_region(3600., 6000., 1000., 2000.)
        npt.assert_equal(spec.get_real_size(), 2 * npix)

        spec.set_forest_region(3600., 4400., 1000., 2000.)
        npt.assert_equal(spec.get_real_size(), 1.5 * npix)

    def test_is_long(self, setup_data):
        _, _, _, spectra_list = setup_data
        dforest_wave = 1180. - 1050.
        skip_ratio = 0.5

        spec = copy.deepcopy(spectra_list[0])
        assert (not spec.is_long(dforest_wave, skip_ratio))

        spec.set_forest_region(3600., 6000., 1000., 2000.)
        assert (spec.is_long(dforest_wave, skip_ratio))

        spec._forestivar = {}
        spec.set_forest_region(3600., 6000., 1120., 1130.)
        assert (not spec.is_long(dforest_wave, skip_ratio))

    def test_coadd_arms_forest(self, setup_data):
        _, _, data, spectra_list = setup_data
        # var_lss zero
        varlss_interp = Fast1DInterpolator(0, 1, np.zeros(3))
        spec = copy.deepcopy(spectra_list[0])
        spec.set_forest_region(3600., 6000., 1050., 1300.)
        spec.cont_params['valid'] = True
        spec.cont_params['cont'] = {
            arm: np.ones_like(farm) for arm, farm in spec.forestflux.items()
        }
        spec.coadd_arms_forest(varlss_interp)

        assert ('brz' in spec.forestflux.keys())
        npt.assert_almost_equal(spec.forestflux['brz'], 2.1)
        npt.assert_almost_equal(spec.cont_params['cont']['brz'], 1.)
        w = (spec.forestwave['brz'] < data['wave']['R'][0])\
            | (spec.forestwave['brz'] > data['wave']['B'][-1])
        npt.assert_almost_equal(spec.forestivar['brz'][w], 1)
        npt.assert_almost_equal(spec.forestivar['brz'][~w], 2)

        # var_lss non-zero
        varlss_interp = Fast1DInterpolator(0, 1, 0.5 * np.ones(3))
        spec = copy.deepcopy(spectra_list[0])
        spec.set_forest_region(3600., 6000., 1050., 1300.)
        spec.cont_params['valid'] = True
        spec.cont_params['cont'] = {
            arm: np.ones_like(farm) for arm, farm in spec.forestflux.items()
        }
        spec.coadd_arms_forest(varlss_interp)

        assert ('brz' in spec.forestflux.keys())
        npt.assert_almost_equal(spec.forestflux['brz'], 2.1)
        npt.assert_almost_equal(spec.cont_params['cont']['brz'], 1.)
        w = (spec.forestwave['brz'] < data['wave']['R'][0])\
            | (spec.forestwave['brz'] > data['wave']['B'][-1])
        npt.assert_almost_equal(spec.forestivar['brz'][w], 1)
        npt.assert_almost_equal(spec.forestivar['brz'][~w], 2)


@pytest.fixture
def setup_data():
    cat_dtype = np.dtype([
        ('TARGETID', '>i8'), ('Z', '>f8'), ('RA', '>f8'), ('DEC', '>f8'),
        ('HPXPIXEL', '>i8'), ('SURVEY', '<U4')])
    cat_by_survey = np.array([
        (39627939372861215, 2.328, 229.861, 6.1925, 8258, b'main')],
        dtype=cat_dtype)

    npix = 1000
    data = {
        'wave': {
            'B': 3600. + 0.8 * np.arange(npix),
            'R': 4000. + 0.8 * np.arange(npix)},
        'flux': {
            'B': 2.1 * np.ones(npix).reshape(1, npix),
            'R': 2.1 * np.ones(npix).reshape(1, npix)},
        'ivar': {
            'B': np.ones(npix).reshape(1, npix),
            'R': np.ones(npix).reshape(1, npix)},
        'mask': {
            'B': np.zeros(npix, dtype='i4').reshape(1, npix),
            'R': np.zeros(npix, dtype='i4').reshape(1, npix)},
        'reso': {}
    }

    spectra_list = qcfitter.spectrum.generate_spectra_list_from_data(
        cat_by_survey, data)

    return cat_by_survey, npix, data, spectra_list


if __name__ == '__main__':
    pytest.main()
