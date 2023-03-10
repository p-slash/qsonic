import os
import pytest

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


if __name__ == '__main__':
    pytest.main()
