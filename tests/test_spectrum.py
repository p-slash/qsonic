import argparse
import copy
import pytest

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
        w = (spec.forestwave['brz'] < data['wave']['R'][0])\
            | (spec.forestwave['brz'] > data['wave']['B'][-1])
        npt.assert_almost_equal(spec.forestivar['brz'][w], 1)
        npt.assert_almost_equal(spec.forestivar['brz'][~w], 2)


if __name__ == '__main__':
    pytest.main()
