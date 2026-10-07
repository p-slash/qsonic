import os
import pytest

import fitsio
import numpy as np
from numpy.lib.recfunctions import merge_arrays
import numpy.testing as npt

import qsonic.masks
from qsonic.spectrum import Spectrum


def test_skymask(tmp_path, setup_data):
    # Create skymask
    fname_skymask = tmp_path / "test_skymask.txt"
    with open(fname_skymask, 'w') as file_sky:
        file_sky.write("Ca\t 3700\t 3750\t OBS\n")
        file_sky.write("RF\t 1142\t 1150\t RF\n")
        file_sky.write("Ca\t 4150\t 4200\t OBS\n")

    skymask = qsonic.masks.SkyMask(fname_skymask)
    os.remove(fname_skymask)

    # Create spectrum
    cat_by_survey, _, data = setup_data(1)
    spec = Spectrum(
        cat_by_survey[0], data['wave'], data['flux'],
        data['ivar'], data['mask'], data['reso'], 0
    )
    spec.set_forest_region(3600., 6000., 1000., 2000.)
    z_qso = cat_by_survey['Z']

    skymask.apply(spec)

    for arm, wave_arm in spec.forestwave.items():
        w = (wave_arm >= 3700.) & (wave_arm < 3750.)
        w |= (wave_arm >= 4150.) & (wave_arm < 4200.)
        w |= (wave_arm >= 1142 * (1 + z_qso)) & (wave_arm < 1150 * (1 + z_qso))
        npt.assert_equal(spec.forestivar[arm][w], 0)
        npt.assert_equal(spec.forestivar[arm][~w], 1)


def test_balmask_setup_filters_and_attaches_catalog(tmp_path, setup_data):
    cat_by_survey, _, data = setup_data(3)
    spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
        cat_by_survey, data)

    bal_dtype = [
        ('TARGETID', 'i8'),
        ('VMIN_CIV_450', 'f8', 3),
        ('VMAX_CIV_450', 'f8', 3),
        ('VMIN_CIV_2000', 'f8', 2),
        ('VMAX_CIV_2000', 'f8', 2)
    ]
    catalog = np.array([
        (0, [20., 30., 40.], [25., 35., 45.], [200., 300.], [300., 400.]),
        (0, [25., 35., 45.], [400., 500., 550.], [400., 500.], [500., 600.]),
        (0, [10., 15., 20.], [100., 150., 200.], [100., 150.], [150., 200.]),
        (0, [25., 35., 45.], [250., 350., 450.], [250., 350.], [350., 450.])
    ], dtype=bal_dtype)
    catalog['TARGETID'][:3] = cat_by_survey['TARGETID']
    catalog['TARGETID'][3] = 333
    fnamebal = tmp_path / "bal_catalog.fits"
    with fitsio.FITS(fnamebal, 'rw', clobber=True) as fts:
        fts.write(catalog, extname='ZCATALOG')

    balmask = qsonic.masks.BALMask([cat_by_survey], fname=fnamebal)
    for spec in spectra_list:
        balmask.apply(spec)
    os.remove(fnamebal)

    bal_dtype = [
        ('VMIN_CIV_450', 'f8', 3),
        ('VMAX_CIV_450', 'f8', 3),
        ('VMIN_CIV_2000', 'f8', 2),
        ('VMAX_CIV_2000', 'f8', 2)
    ]
    catalog = np.array([
        ([20., 30., 40.], [25., 35., 45.], [200., 300.], [300., 400.]),
        ([25., 35., 45.], [400., 500., 550.], [400., 500.], [500., 600.]),
        ([10., 15., 20.], [100., 150., 200.], [100., 150.], [150., 200.]),
    ], dtype=bal_dtype)

    cat_by_survey = merge_arrays(
        (cat_by_survey, catalog), flatten=True, usemask=False)
    spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
        cat_by_survey, data)
    balmask = qsonic.masks.BALMask([cat_by_survey])
    for spec in spectra_list:
        balmask.apply(spec)


if __name__ == '__main__':
    pytest.main()
