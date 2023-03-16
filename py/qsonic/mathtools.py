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
    """Utility class to store all subsamples with weights and calculate
    the delete-one Jackknife covariance matrix.

    Usage::

        subsampler = SubsampleCov(ndata, nsamples)

        for measurement, weights in SomeDataAfterFunction:
            # You can do this more than nsamples times
            subsampler.add_measurement(measurement, weights)

        mean, covariance = subsampler.get_mean_n_cov()

    .. warning::

            You cannot call :meth:`add_measurement` after calling
            :meth:`get_mean`, :meth:`get_mean_n_cov`,
            or :meth:`get_mean_n_var`.

    Parameters
    ----------
    ndata: int
        Size of the data vector.
    nsamples: int
        Number of samples.
    istart: int, default: 0
        Start index for the subsampling array

    Attributes
    ----------
    ndata: int
        Size of the data vector.
    nsamples: int
        Number of samples. You can more measurements then this.
    _isample: int
        Sample counter. Wraps around nsamples
    _is_normalized: bool
        If the weights are normalized. Keeps track if :func:`_normalize` is
        called.
    all_measurements: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        2D array of zeros of shape ``(nsamples, ndata)``.
    all_weights: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        2D array of zeros of shape ``(nsamples, ndata)``.
    mean: :class:`ndarray <numpy.ndarray>` or None
        Mean. 1D array of size ``ndata``
    covariance: :class:`ndarray <numpy.ndarray>` or None
        Covariance. 2D array of shape ``(ndata, ndata)``
    variance: :class:`ndarray <numpy.ndarray>` or None
        Variance. 1D array of size ``ndata``
    """

    def __init__(self, ndata, nsamples, istart=0):
        self.ndata = ndata
        self.nsamples = nsamples
        self._isample = istart % nsamples

        self.all_measurements = np.zeros((nsamples, ndata))
        self.all_weights = np.zeros((nsamples, ndata))
        self._is_normalized = False

        self.mean = None
        self.covariance = None
        self.variance = None

    def add_measurement(self, xvec, wvec):
        """ Adds a measurement to the sample.

        You can call this function more then ``nsamples`` times. The provided
        measurement should be weighted, but unnormalized, i.e. ``xvec=wi*xi``.
        After a mean or covariance is obtained, you cannot add
        more measurements.

        Arguments
        ---------
        xvec: :class:`ndarray <numpy.ndarray>`
            1D data (measurement) vector.
        wvec: :class:`ndarray <numpy.ndarray>`
            1D weight vector.

        Raises
        ---------
        RuntimeError
            If the object is normalized by calling
            ``_normalize, get_mean, get_mean_n_cov`` and
            ``get_mean_n_var``.
        """
        if self._is_normalized:
            raise RuntimeError(
                "SubsampleCov has already been normalized. "
                "You cannot add more measurements.")

        self.all_measurements[self._isample] += xvec
        self.all_weights[self._isample] += wvec
        self._isample = (self._isample + 1) % self.nsamples

    def allreduce(self, comm, inplace):
        """Sums statistics from all MPI process.

        .. note::

            Call this with ``inplace=MPI.IN_PLACE``.

        Arguments
        ---------
        comm: MPI.COMM_WORLD
            MPI comm object for Allreduce
        inplace: BufSpec
            MPI.IN_PLACE
        """

        comm.Allreduce(inplace, self.all_measurements)
        comm.Allreduce(inplace, self.all_weights)

    def _normalize(self):
        self.all_measurements /= self.all_weights + np.finfo(float).eps
        self.all_weights /= np.sum(
            self.all_weights, axis=0) + np.finfo(float).eps

        self._is_normalized = True

    def get_mean(self):
        """ Get the mean of all subsamples.

        .. warning::

            You cannot call :meth:`add_measurement` after calling this unless
            you :meth:`reset`.

        Returns
        -------
        mean: :class:`ndarray <numpy.ndarray>`
            Mean.
        """
        if not self._is_normalized:
            self._normalize()

        self.mean = np.sum(self.all_measurements * self.all_weights, axis=0)

        return self.mean

    def _get_xdiff(self, mean_xvec, bias_correct=False):
        # remove one measurement, then renormalize
        jack_i = (
            mean_xvec - self.all_measurements * self.all_weights
        ) / (1 - self.all_weights)
        mean_jack = np.mean(jack_i, axis=0)

        if bias_correct:
            bias_jack = (self.nsamples - 1) * (mean_jack - mean_xvec)
            mean_xvec -= bias_jack

        xdiff = jack_i - mean_jack

        return mean_xvec, xdiff

    def get_mean_n_cov(self, bias_correct=False):
        """ Get the mean and covariance of the mean using delete-one Jackknife.

        Also sets :attr:`mean` and :attr:`covariance`.

        .. warning::

            You cannot call :meth:`add_measurement` after calling this unless
            you :meth:`reset`.

        Arguments
        ---------
        bias_correct: bool, default: False
            Jackknife bias correction term for the mean.

        Returns
        -------
        mean: :class:`ndarray <numpy.ndarray>`
            Mean.
        cov: :class:`ndarray <numpy.ndarray>`
            Covariance of the mean. 2D array
        """
        mean_xvec = self.get_mean()
        self.mean, xdiff = self._get_xdiff(mean_xvec, bias_correct)

        self.covariance = (
            np.dot(xdiff.T, xdiff) * (self.nsamples - 1) / self.nsamples
        )

        return self.mean, self.covariance

    def get_mean_n_var(self, bias_correct=False):
        """ Get the mean and variance of themean (i.e. diagonal of the
        covariance) using delete-one Jackknife.

        .. warning::

            You cannot call :meth:`add_measurement` after calling this unless
            you :meth:`reset`.

        Arguments
        ---------
        bias_correct: bool, default: False
            Jackknife bias correction term for the mean.

        Returns
        -------
        mean_xvec: :class:`ndarray <numpy.ndarray>`
            Mean.
        var_xvec: :class:`ndarray <numpy.ndarray>`
            Variance of the mean. 1D array
        """
        mean_xvec = self.get_mean()
        self.mean, xdiff = self._get_xdiff(mean_xvec, bias_correct)

        self.variance = (
            np.sum(xdiff**2, axis=1) * (self.nsamples - 1) / self.nsamples
        )

        return self.mean, self.variance

    def reset(self, istart=0):
        self._isample = istart % self.nsamples

        self.all_measurements = 0
        self.all_weights = 0
        self._is_normalized = False

        self.mean = None
        self.covariance = None
        self.variance = None
