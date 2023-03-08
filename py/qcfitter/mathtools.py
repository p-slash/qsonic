import numpy as np
from numba import njit


class Fast1DInterpolator():
    """ Fast interpolator class for equally spaced data. Out of domain points
    are linearly extrapolated without producing any warnings or errors.

    Example::

        one_interp = Fast1DInterpolator(0., 1., np.ones(3))
        one_interp(5) # = 1

    Parameters
    ----------
    xp0: float
        Initial x point for interpolation data.
    dxp0: float
        Spacing of x points.
    fp: numpy array
        Function calculated at interpolation points
    ep: numpy array (optional)
        Error on fp points. Not used! Bookkeeping purposes only.
    copy: bool (default: False)
        Copy input data, specifically fp
    """

    def __init__(self, xp0, dxp, fp, copy=False, ep=None):
        self.xp0 = float(xp0)
        self.dxp = float(dxp)
        if copy:
            self.fp = fp.copy()
        else:
            self.fp = fp
        self.ep = ep

    def __call__(self, x):
        return _fast_eval_interp1d(x, self.xp0, self.dxp, self.fp)


@njit("f8[:](f8[:], f8, f8, f8[:])")
def _fast_eval_interp1d(x, xp0, dxp, fp):
    xx = (x - xp0) / dxp
    idx = np.clip(xx, 0, fp.size - 1 - 1e-8).astype(np.int_)

    d_idx = xx - idx
    y1, y2 = fp[idx], fp[idx + 1]

    return y1 * (1 - d_idx) + y2 * d_idx
# ===================================================


@njit("f8[:](f8[:], f8[:])")
def mypoly1d(coef, x):
    """ My simple power series polynomial calculator.

    Arguments
    ---------
    coef: ndarray of floats
        Coefficient array in increasing power starting with the constant.
    x: ndarray of floats
        Array to calculate polynomial.

    Returns
    ---------
    results: ndarray of floats
        Polynomial calculated at x.
    """
    results = np.zeros_like(x)
    for i, a in enumerate(coef):
        results += a * x**i
    return results


def fft_gaussian_smooth(x, sigma_pix=20, pad_size=25, mode='edge'):
    """ My Gaussian smoother using FFTs. Input array is padded with edge
    values at the boundary by default.

    Arguments
    ---------
    x: 1D array of floats
        Array to smooth.
    sigma_pix: int or float (default: 20)
        Smoothing Gaussian sigma
    pad_size: int (default: 25)
        Number of pixels to pad the array `x` at the boundary.
    mode: string
        Padding method. See `np.pad` for options.

    Returns
    ---------
    y: 1D array of floats
        Smoothed `x` values. Same size as `x`
    """
    # Pad the input array to get rid of annoying edge effects
    # Pad values are set to the edge value
    arrsize = x.size + 2 * pad_size
    padded_arr = np.pad(x, pad_size, mode=mode)

    kvals = np.fft.rfftfreq(arrsize)
    smerror_k = np.fft.rfft(padded_arr) * np.exp(-(kvals * sigma_pix)**2 / 2.)

    y = np.fft.irfft(smerror_k, n=arrsize)[pad_size:-pad_size]

    return y


def get_smooth_ivar(ivar, sigma_pix=20, pad_size=25, esigma=3.5):
    """ Smoothing `ivar` values to reduce signal-noise coupling. Smoothing
    is done on `error=1/sqrt(ivar)`, while replacing `ivar=0` and outliers in
    `error`values with the median. These replaced values are put back in in the
    final result.

    Arguments
    ---------
    ivar: 1D array of floats
        Inverse variance array.
    sigma_pix: int or float (default: 20)
        Smoothing Gaussian sigma.
    esigma: float (default: 3.5)
        Sigma to identify outliers via MAD.

    Returns
    ---------
    ivar2: 1D array of floats
        Smoothed `ivar` values. Outliers and masked values are put back in.
    """
    error = np.empty_like(ivar)
    w1 = ivar > 0
    error[w1] = 1 / np.sqrt(ivar[w1])
    median_err = np.median(error[w1])
    error[~w1] = median_err

    # Isolate high noise pixels
    mad = np.median(np.abs(error[w1] - median_err)) * 1.4826
    w2 = (error - median_err) > esigma * mad
    err_org = error[w2].copy()

    # Replace them with the median
    error[w2] = median_err
    error = fft_gaussian_smooth(error, sigma_pix, pad_size)

    # Restore values of bad pixels
    error[w2] = err_org
    ivar2 = 1 / error**2
    ivar2[~w1] = 0

    return ivar2
