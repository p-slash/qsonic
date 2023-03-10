import pytest

import numpy as np
from scipy.interpolate import interp1d
import numpy.testing as npt

import qsonic.mathtools


class TestMathtools(object):
    def test_Fast1DInterpolator(self):
        xin, dxp = np.linspace(320., 550., 300, retstep=True)
        fp = np.power(xin / 100. - 4., 3)

        fast_interp = qsonic.mathtools.Fast1DInterpolator(xin[0], dxp, fp)
        xarr = np.linspace(310., 560., 100)
        yarr = fast_interp(xarr)

        ytrue = interp1d(xin, fp, fill_value='extrapolate')(xarr)

        npt.assert_allclose(yarr, ytrue)

    def test_mypoly1d(self):
        coefs = np.array([5.5, 1.5, 0.7])
        xarr = np.linspace(-2, 2, 100)
        yarr = qsonic.mathtools.mypoly1d(coefs, xarr)

        ytrue = np.poly1d(coefs[::-1])(xarr)

        npt.assert_allclose(yarr, ytrue)

    def test_fft_gaussian_smooth(self):
        xarr = np.ones(2**10)
        xarr_sm = qsonic.mathtools.fft_gaussian_smooth(xarr)
        npt.assert_allclose(xarr, xarr_sm)

    def test_get_smooth_ivar(self):
        ivar = np.ones(2**10)
        idces = np.array([16, 17, 18, 234, 235, 512, 667, 898, 910, 956])
        ivar[idces] = 0

        ivar_sm = qsonic.mathtools.get_smooth_ivar(ivar)
        npt.assert_allclose(ivar_sm[idces], 0)
        npt.assert_allclose(ivar_sm, ivar)


if __name__ == '__main__':
    pytest.main()
