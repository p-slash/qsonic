import pytest

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import numpy.testing as npt

import qsonic.mathtools


class TestMathtools(object):
    def test_FastLinear1DInterp(self):
        xin, dxp = np.linspace(320., 550., 5, retstep=True)
        fp = np.power(xin / 100. - 4., 3)

        fast_interp = qsonic.mathtools.FastLinear1DInterp(xin[0], dxp, fp)
        xarr = np.linspace(310., 560., 100)
        yarr = fast_interp(xarr)

        ytrue = interp1d(xin, fp, fill_value='extrapolate')(xarr)

        npt.assert_allclose(yarr, ytrue)

        xin, dxp = np.linspace(320., 550., 300, retstep=True)
        fp = np.power(xin / 100. - 4., 3)

        fast_interp = qsonic.mathtools.FastLinear1DInterp(xin[0], dxp, fp)
        xarr = np.linspace(310., 560., 100)
        yarr = fast_interp(xarr)

        ytrue = interp1d(xin, fp, fill_value='extrapolate')(xarr)

        npt.assert_allclose(yarr, ytrue)

    def test_FastCubic1DInterp(self):
        xin, dxp = np.linspace(320., 550., 300, retstep=True)
        fp = np.power(xin / 100. - 4., 3)

        fast_interp = qsonic.mathtools.FastCubic1DInterp(
            xin[0], dxp, fp, bc_type='natural')
        xarr = np.linspace(310., 560., 100)
        yarr = fast_interp(xarr)

        ytrue = CubicSpline(xin, fp, bc_type='natural', extrapolate=True)(xarr)

        npt.assert_allclose(yarr, ytrue)

        fast_interp = qsonic.mathtools.FastCubic1DInterp(
            xin[0], dxp, fp, bc_type='not-a-knot')
        xarr = np.linspace(310., 560., 100)
        yarr = fast_interp(xarr)

        ytrue = CubicSpline(
            xin, fp, bc_type='not-a-knot', extrapolate=True)(xarr)

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

    def test_get_median_outlier_mask(self):
        rng = np.random.default_rng(10)
        size = 100
        sigmas = 1 + rng.poisson(5, size=size).astype(float)
        ivar = sigmas**-2
        flux = rng.normal(0, sigmas)

        w = qsonic.mathtools.get_median_outlier_mask(flux, ivar)
        assert not any(w)

        flux[20] = 1e3
        w = qsonic.mathtools.get_median_outlier_mask(flux, ivar)
        assert w[20]
        assert not any(w[:20]) and not any(w[21:])

        ivar[20:40] = 0

        w = qsonic.mathtools.get_median_outlier_mask(flux, ivar)
        assert all(w[20:40])
        assert not any(w[:20])
        assert not any(w[40:])

        # test different spectral shapes
        flux = rng.poisson(sigmas) - sigmas + (np.arange(size) / 25)**2
        ivar = sigmas**-2
        w = qsonic.mathtools.get_median_outlier_mask(flux, ivar)
        assert not any(w)

        flux = 100 + ((1 + np.arange(size)) / 5)**2
        ivar = flux**-2
        flux = rng.poisson(flux)
        assert not any(w)

    def test_get_smooth_ivar(self):
        ivar = np.ones(2**10)
        idces = np.array([16, 17, 18, 234, 235, 512, 667, 898, 910, 956])
        ivar[idces] = 0

        ivar_sm = qsonic.mathtools.get_smooth_ivar(ivar)
        npt.assert_allclose(ivar_sm[idces], 0)
        npt.assert_allclose(ivar_sm, ivar)

    def test_SubsampleCov_theory(self):
        n, m = 300000, 5
        nsample = 10000
        subsampler = qsonic.mathtools.SubsampleCov((2, m), nsample)

        randoms = np.random.default_rng(0).normal(loc=1, size=(n, m))

        true_mean = np.mean(randoms, axis=0)
        true_var = np.var(randoms, axis=0)
        var_on_mean = true_var / randoms.shape[0]
        rtol = 3.5 / np.sqrt(nsample - 1)

        for r in randoms:
            xvec = np.empty((2, m))
            xvec[0] = r
            xvec[1] = 2 * r
            subsampler.add_measurement(xvec, 1)

        mean, var = subsampler.get_mean_n_var(bias_correct=False)
        npt.assert_allclose(mean[0], true_mean)
        npt.assert_allclose(var[0], var_on_mean, rtol=rtol)
        npt.assert_allclose(var[1], 4 * var_on_mean, rtol=rtol)

        mean, cov = subsampler.get_mean_n_cov(bias_correct=False)
        npt.assert_allclose(mean[0], true_mean)
        npt.assert_allclose(mean[1], 2 * true_mean)
        npt.assert_allclose(cov[0].diagonal(), var_on_mean, rtol=rtol)
        npt.assert_allclose(cov[1].diagonal(), 4 * var_on_mean, rtol=rtol)

    def test_SubsampleCov_shape(self):
        subsampler = qsonic.mathtools.SubsampleCov((3, 10), 20)
        randoms = np.random.default_rng().normal(size=(100, 3, 10))
        randoms[:, 1, :] += 1
        randoms[:, 2, :] += 2

        true_mean = np.mean(randoms, axis=0)
        # true_var = np.std(randoms, axis=0)**2
        # var_on_mean = true_var / randoms.shape[0]

        for r in randoms:
            subsampler.add_measurement(r, 1)

        mean, var = subsampler.get_mean_n_var()
        npt.assert_allclose(mean, true_mean)
        npt.assert_equal(mean.shape, var.shape)

        mean, cov = subsampler.get_mean_n_cov()
        npt.assert_allclose(mean, true_mean)
        assert (len(cov) == 3)
        for jj in range(3):
            npt.assert_equal(cov[jj].shape, (10, 10))

        mean, cov = subsampler.get_mean_n_cov(indices=[0, 1], blockdim=5)
        npt.assert_allclose(mean, true_mean)
        assert (len(cov) == 3)
        for jj in range(2):
            npt.assert_equal(cov[jj].shape, (2, 5, 5))
        assert cov[2] is None

    def test_SubsampleCov_reset(self):
        subsampler = qsonic.mathtools.SubsampleCov((3, 10), 20)
        subsampler.reset()

        npt.assert_equal(subsampler.all_measurements.shape, (20, 3, 10))
        npt.assert_equal(subsampler.all_weights.shape, (20, 1, 10))


if __name__ == '__main__':
    pytest.main()
