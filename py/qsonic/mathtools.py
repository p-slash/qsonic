"""Mathematical utility objects and functions."""
import numpy as np
from numba import njit


@njit("f8[:](f8[:], f8, f8, f8[:])")
def _fast_eval_interp1d(x, xp0, dxp, fp):
    xx = (x - xp0) / dxp
    idx = np.clip(xx, 0, fp.size - 1 - 1e-8).astype(np.int_)

    d_idx = xx - idx
    y1, y2 = fp[idx], fp[idx + 1]

    return y1 * (1 - d_idx) + y2 * d_idx


@njit("f8[:](f8[:], f8[:])")
def mypoly1d(coef, x):
    """ My simple power series polynomial calculator.

    Arguments
    ---------
    coef: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Coefficient array in increasing power starting with the constant.
    x: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Array to calculate polynomial.

    Returns
    ---------
    results: :external+numpy:py:class:`ndarray <numpy.ndarray>`
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
    x: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        1D array to smooth.
    sigma_pix: float, default: 20
        Smoothing Gaussian sigma
    pad_size: int, default: 25
        Number of pixels to pad the array x at the boundary.
    mode: str
        Padding method. See :external+numpy:func:`numpy.pad` for options.

    Returns
    ---------
    y: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Smoothed x values. Same size as x.
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
    """ Smoothing ``ivar`` values to reduce signal-noise coupling.

    Smoothing is done on ``error=1/sqrt(ivar)``, while replacing ``ivar=0`` and
    outliers in ``error`` values with the median. These replaced values are put
    back in in the final result.

    Arguments
    ---------
    ivar: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Inverse variance array.
    sigma_pix: float, default: 20
        Smoothing Gaussian sigma.
    pad_size: int, default: 25
        Number of pixels to pad the array at the boundary.
    esigma: float, default: 3.5
        Sigma to identify outliers via MAD.

    Returns
    ---------
    ivar2: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Smoothed ivar values. Outliers and masked values are put back in.
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


class Fast1DInterpolator():
    """Fast interpolator class for equally spaced data. Out of domain points
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
    fp: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Function calculated at interpolation points.
    ep: :external+numpy:py:class:`ndarray <numpy.ndarray>`, optional
        Error on fp points. Not used! Bookkeeping purposes only.
    copy: bool, default: False
        Copy input data, specifically fp.
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


class SubsampleCov():
    """Utility class to store all subsamples and calculate delete-one Jackknife
    covariance matrix.

    Usage::

        subsampler = SubsampleCov(ndata, nsamples, is_weighted=True)

        for measurement, weights in SomeDataAfterFunction:
            # You can do this more than nsamples times
            subsampler.add_measurement(measurement, weights)

        mean, covariance = subsampler.get_mean_n_cov()

    .. warning::

            You cannot call :meth:`add_measurement` after calling
            :meth:`get_mean` or :meth:`get_mean_n_cov`.

    Parameters
    ----------
    ndata: int
        Size of the data vector.
    nsamples: int
        Number of samples.
    is_weighted: bool, default: False
        Whether the samples are weighted.

    Attributes
    ----------
    ndata: int
        Size of the data vector.
    nsamples: int
        Number of samples. You can more measurements then this.
    is_weighted: bool, default: False
        Whether the samples are weighted.
    _isample: int
        Sample counter. Wraps around nsamples
    _is_normalized: bool
        If the weights are normalized. Keeps track if :func:`_normalize` is
        called.
    all_measurements: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        2D array of zeros of shape ``(nsamples, ndata)``.
    all_weights: :external+numpy:py:class:`ndarray <numpy.ndarray>` or None
        2D array of zeros of shape ``(nsamples, ndata)`` if
        ``is_weighted=True``.
    """

    def __init__(self, ndata, nsamples, is_weighted=False):
        self.ndata = ndata
        self.nsamples = nsamples
        self._isample = 0
        self.is_weighted = is_weighted

        self.all_measurements = np.zeros((nsamples, ndata))
        if self.is_weighted:
            self.all_weights = np.zeros((nsamples, ndata))
            self._is_normalized = False
        else:
            self.all_weights = np.ones(self.nsamples) / self.nsamples
            self._is_normalized = True

    def add_measurement(self, xvec, wvec=None):
        """ Adds a measurement to the sample.

        You can call this function more then ``nsamples`` times. If ``wvec`` is
        passed, the provided measurement (``xvec``) should be weighted, but
        unnormalized. After a mean or covariance is obtained, you cannot add
        more measurements.

        Arguments
        ---------
        xvec: :class:`ndarray <numpy.ndarray>`
            1D data (measurement) vector.
        wvec: :class:`ndarray <numpy.ndarray>` or None, default: None
            1D weight vector.

        Raises
        ---------
        RuntimeError
            If the object is initialized with ``is_weighted=True``,
            but no weights are provided (``wved=None``).
        RuntimeError
            If the object is initialized with ``is_weighted=False``,
            but unexpected weights are provided.
        RuntimeError
            If the object is initialized with ``is_weighted=True`` and later
            normalized by calling ``_normalize, get_mean, get_mean_n_cov``.
        """
        if (wvec is None) and self.is_weighted:
            raise RuntimeError("SubsampleCov requires weights.")
        if (wvec is not None) and (not self.is_weighted):
            raise RuntimeError("SubsampleCov unexpected weights.")
        if (self._is_normalized) and self.is_weighted:
            raise RuntimeError(
                "SubsampleCov has already been normalized. "
                "You cannot add more measurements.")

        self.all_measurements[self._isample] += xvec

        if (wvec is not None) and self.is_weighted:
            self.all_weights[self._isample] += wvec

        self._isample = (self._isample + 1) % self.nsamples

    def _normalize(self):
        if not self.is_weighted:
            return

        self.all_measurements /= self.all_weights + np.finfo(float).eps
        self.all_weights /= np.sum(
            self.all_weights, axis=0) + np.finfo(float).eps

        self._is_normalized = True

    def get_mean(self):
        """ Get the mean of all subsamples.

        .. warning::

            You cannot call :meth:`add_measurement` after calling this.

        Returns
        -------
        mean_xvec: :class:`ndarray <numpy.ndarray>`
            Mean.
        """
        if not self._is_normalized:
            self._normalize()

        mean_xvec = np.sum(self.all_measurements * self.all_weights, axis=0)

        return mean_xvec

    def get_mean_n_cov(self, bias_correct=False):
        """ Get the mean and covariance using delete-one Jackknife.

        .. warning::

            You cannot call :meth:`add_measurement` after calling this.

        Returns
        -------
        mean_xvec: :class:`ndarray <numpy.ndarray>`
            Mean.
        cov: :class:`ndarray <numpy.ndarray>`
            Covariance. 2D array
        """
        if not self._is_normalized:
            self._normalize()

        mean_xvec = self.getMean()

        # remove one measurement, then renormalize
        jack_i = (
            mean_xvec - self.all_measurements * self.all_weights
        ) / (1 - self.all_weights)
        mean_jack = np.mean(jack_i, axis=0)

        if bias_correct:
            bias_jack = (self.nsamples - 1) * (mean_jack - mean_xvec)
            mean_xvec -= bias_jack

        xdiff = jack_i - mean_jack

        cov = np.dot(xdiff.T, xdiff) * (self.nsamples - 1) / self.nsamples

        return mean_xvec, cov
