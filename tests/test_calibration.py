import pytest

import fitsio
import numpy as np
import numpy.testing as npt

import qsonic.spectrum
import qsonic.calibration


@pytest.fixture()
def my_setup_attributes(tmp_path):
    fname = tmp_path / "attributes.fits"

    with fitsio.FITS(fname, 'rw', clobber=True) as fts:
        wavelength = np.linspace(3600, 9800, 775)
        stacked_flux = 5. * np.ones(wavelength.size)
        fts.write(
            [wavelength, stacked_flux],
            names=['lambda', 'stacked_flux'],
            extname='STACKED_FLUX')

        wavelength = np.linspace(3600, 9800, 50)
        ones_array = np.ones(wavelength.size)
        fts.write(
            [wavelength, ones_array, ones_array, ones_array * 2, ones_array],
            names=['lambda', 'var_lss', 'e_var_lss', 'eta', 'e_eta'],
            extname='VAR_FUNC')

    yield fname


def test_calibrations(setup_data, my_setup_attributes):
    fname = my_setup_attributes
    ncal = qsonic.calibration.NoiseCalibrator(fname)
    ncal_plus = qsonic.calibration.NoiseCalibrator(fname, add_varlss=True)
    fcal = qsonic.calibration.FluxCalibrator(fname)

    cat_by_survey, _, data = setup_data(1)
    spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
        cat_by_survey, data)
    spec = spectra_list[0]
    spec.set_forest_region(3600., 6000., 1050., 1300.)
    f0 = spec.flux['B'][0]

    ncal.apply(spectra_list)
    npt.assert_allclose(spec.forestivar['B'], 0.5)
    npt.assert_allclose(spec.forestivar['R'], 0.5)

    with pytest.raises(Exception):
        ncal_plus.apply(spectra_list)

    spec.cont_params['cont'] = {
        arm: np.ones_like(flux) for arm, flux in spec.forestflux.items()
    }
    ncal_plus.apply(spectra_list)
    npt.assert_allclose(spec.forestivar['B'], 0.2)
    npt.assert_allclose(spec.forestivar['R'], 0.2)

    fcal.apply(spectra_list)
    npt.assert_allclose(spec.forestflux['B'], f0 / 5)
    npt.assert_allclose(spec.forestflux['R'], f0 / 5)
