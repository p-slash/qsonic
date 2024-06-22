"""Mathematical utility objects and functions."""
import numpy as np
from numba import njit
from scipy.ndimage import median_filter

from qsonic import QsonicException


def _zero_function(x):
    return 0


def _one_function(x):
    return 1


@njit("f8[:](f8[:], f8, f8, f8[:])")
def _fast_eval_interp1d_lin(x, xp0, dxp, fp):
    """JIT fast linear interpolation."""
    xx = (x - xp0) / dxp
    idx = np.clip(xx, 0, fp.size - 1 - 1e-8).astype(np.int_)

    d_idx = xx - idx
    y1, y2 = fp[idx], fp[idx + 1]

    return y1 * (1 - d_idx) + y2 * d_idx


@njit("f8[:](f8[:], f8, f8, f8[:], f8[:])")
def _fast_eval_interp1d_cubic(x, xp0, dxp, fp, y2p):
    """JIT fast cubic spline."""
    xx = (x - xp0) / dxp
    idx = np.clip(xx, 0, fp.size - 1 - 1e-8).astype(np.int_)

    d_idx = xx - idx
    a, b = 1 - d_idx, d_idx
    y1, y2 = fp[idx], fp[idx + 1]
    ypp1, ypp2 = y2p[idx], y2p[idx + 1]

    r1 = a * y1 + b * y2
    r2 = ((a + 1) * ypp1 + (b + 1) * ypp2) * (-a * b * dxp**2 / 6.)

    return r1 + r2


@njit("f8[:](f8[:], f8)")
def _spline_cubic_natural(fp, dxp):
    """ Constructs the second derivative array with natural boundary
    conditions.

    As coded in Numerical Recipes in C Chaper 3.3, but specialized for equally
    spaced input data.

    Arguments
    ---------
    fp: ndarray
        1D data points array to interpolate.
    dxp: float
        Spacing of x points.

    Returns
    -------
    y2p: ndarray
        Second derivative array. Same size as ``fp``.
    """
    y2p = np.empty(fp.size)
    u = np.empty(fp.size - 1)
    u[0] = 0
    y2p[0] = y2p[-1] = 0

    sig = 0.5

    for ii in range(1, fp.size - 1):
        p = sig * y2p[ii - 1] + 2.
        y2p[ii] = (sig - 1) / p
        u[ii] = (fp[ii + 1] + fp[ii - 1] - 2 * fp[ii]) / dxp
        u[ii] = (3 * u[ii] / dxp - sig * u[ii - 1]) / p

    for ii in range(fp.size - 2, 0, -1):
        y2p[ii] = y2p[ii] * y2p[ii + 1] + u[ii]

    return y2p


@njit("f8[:](f8[:], f8)")
def _spline_cubic_notaknot(fp, dxp):
    """Constructs the second derivative array with not-a-knot boundary
    conditions:

    .. math::

        y''_0 - 2 y''_1 + y''_2 = 0

    Arguments
    ---------
    fp: ndarray
        1D data points array to interpolate.
    dxp: float
        Spacing of x points.

    Returns
    -------
    y2p: ndarray
        Second derivative array. Same size as ``fp``.
    """
    N = fp.size
    y2p = np.empty(N)
    # Solve the submatrix 1:-1 using precalculated u and p values
    u = np.array([
        6.000000000000000000e+00, 4.000000000000000000e+00,
        3.750000000000000000e+00, 3.733333333333333393e+00,
        3.732142857142857206e+00, 3.732057416267942518e+00,
        3.732051282051282115e+00, 3.732050841635176752e+00,
        3.732050810014727382e+00, 3.732050807744481613e+00,
        3.732050807581485330e+00, 3.732050807569782691e+00,
        3.732050807568942474e+00, 3.732050807568882078e+00,
        3.732050807568877637e+00, 3.732050807568877193e+00])
    p = np.array([
        1.666666666666666574e-01, 2.500000000000000000e-01,
        2.666666666666666630e-01, 2.678571428571428492e-01,
        2.679425837320574266e-01, 2.679487179487179405e-01,
        2.679491583648230812e-01, 2.679491899852724512e-01,
        2.679491922555185535e-01, 2.679491924185148921e-01,
        2.679491924302174755e-01, 2.679491924310576922e-01,
        2.679491924311180329e-01, 2.679491924311223627e-01,
        2.679491924311226958e-01])

    for i in range(1, N - 1):
        y2p[i] = fp[i - 1] - 2 * fp[i] + fp[i + 1]

    for i in range(1, min(N - 3, p.size + 1)):
        y2p[i + 1] -= p[i - 1] * y2p[i]
    for i in range(p.size + 1, N - 3):
        y2p[i + 1] -= p[-1] * y2p[i]

    y2p[-2] /= u[0]
    for i in range(N - 4, u.size - 1, -1):
        y2p[i + 1] = (y2p[i + 1] - y2p[i + 2]) / u[-1]
    for i in range(min(N - 4, u.size - 1), 0, -1):
        y2p[i + 1] = (y2p[i + 1] - y2p[i + 2]) / u[i]

    y2p[1] /= u[0]
    y2p[0] = 2 * y2p[1] - y2p[2]
    y2p[-1] = 2 * y2p[-2] - y2p[-3]
    y2p *= 6.0 / dxp**2

    return y2p


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


@njit("f8[:, :, :](f8[:], f8[:], f8[:, :, :])")
def block_covariance_of_square(mean, var, cov):
    """ Return the block covariance of x^2, i.e.
    :math:`<x_i^2 x_j^2> - <x_i^2><x_j^2>`. Compatible with ``blockdim``
    argument of :meth:`SubsampleCov.get_mean_n_cov`.

    .. math::

        Cov[x_i^2, x_j^2] &= Var[x_i] Var[x_j] + Cov(x_i, x_j)^2 \\\\
        &+ 2 Cov(x_i, x_j) E[x_i] E[x_j] \\\\
        &+  Var[x_i] E[x_j]^2 + Var[x_j] E[x_i]^2

    Arguments
    ---------
    mean: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        1D array mean values.
    var: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        1D array variance values.
    cov: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        3D array covariance. Shape is ``(nblock, ndata, ndata)``.

    Returns
    -------
    new_cov: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        3D array propagated covariance.
    """
    nblock = cov.shape[0]
    ndata = cov.shape[1]
    new_cov = np.empty_like(cov)

    for jj in range(nblock):
        i1 = jj * ndata
        i2 = i1 + ndata
        v = var[i1:i2]
        m = mean[i1:i2]

        np.outer(v, v, out=new_cov[jj])
        new_cov[jj] += cov[jj]**2
        new_cov[jj] += 2 * cov[jj] * np.outer(m, m)

        C1 = np.outer(v, m**2)
        new_cov[jj] += C1 + C1.T

    return new_cov


def fft_gaussian_smooth(x, sigma_pix=20, mode='edge'):
    """ My Gaussian smoother using FFTs. Input array is padded with edge
    values at the boundary by default. Pad size is ``3*sigma_pix``.

    Arguments
    ---------
    x: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        1D array to smooth.
    sigma_pix: float, default: 20
        Smoothing Gaussian sigma in terms of number of pixels.
    mode: str
        Padding method. See :external+numpy:func:`numpy.pad` for options.

    Returns
    ---------
    y: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Smoothed x values. Same size as x.
    """
    # Pad the input array to get rid of annoying edge effects
    # Pad values are set to the edge value
    pad_size = max(1, int(3 * sigma_pix))
    arrsize = x.size + 2 * pad_size
    padded_arr = np.pad(x, pad_size, mode=mode)

    kvals = np.fft.rfftfreq(arrsize)
    smerror_k = np.fft.rfft(padded_arr) * np.exp(-(kvals * sigma_pix)**2 / 2.)

    y = np.fft.irfft(smerror_k, n=arrsize)[pad_size:-pad_size]

    return y


def get_median_outlier_mask(flux, ivar, esigma=3.5, nwindow=11):
    """Calculates median[i] of flux and sigma = 1.4826 * MAD[i] based on moving
    median filter of size ``nwindow``. Returns a boolean array to mask pixels
    if ``abs(f[i] - median[i]) > esigma * max(n[i], sigma[i])``.

    Arguments
    ---------
    flux: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Flux array.
    ivar: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Inverse variance array.
    esigma: float, default: 3.5
        Sigma to identify outliers via MAD.
    nwindow: int, default: 11
        Window size of the moving median filter.

    Returns
    ---------
    mask: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Bool array. Mask if true.
    """
    w = ivar != 0
    mask = np.zeros(flux.size, dtype=bool)
    mask[~w] = True

    if not any(w):
        return mask

    abs_diff = np.abs(flux[w] - median_filter(flux[w], nwindow))
    isig_mad = np.fmin(
        1.0, 1.0 / (1.4826 * median_filter(abs_diff, nwindow) + 1e-12))
    isig_cut = np.fmin(np.sqrt(ivar[w]), isig_mad) / esigma
    mask[w] = abs_diff * isig_cut > 1.0
    return mask


def get_smooth_ivar(ivar, sigma_pix=20, esigma=3.5):
    """ Smoothing ``ivar`` values to reduce signal-noise coupling.

    Smoothing is done on ``error=1/sqrt(ivar)``, while replacing ``ivar=0`` and
    outliers in ``error`` values with the median. These replaced values are put
    back in in the final result.

    Arguments
    ---------
    ivar: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Inverse variance array.
    sigma_pix: float, default: 20
        Smoothing Gaussian sigma in terms of number of pixels.
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
    error = fft_gaussian_smooth(error, sigma_pix)

    # Restore values of bad pixels
    error[w2] = err_org
    ivar2 = 1 / error**2
    ivar2[~w1] = 0

    return ivar2


class FastLinear1DInterp():
    """Fast interpolator class for equally spaced data. Out of domain points
    are linearly extrapolated without producing any warnings or errors.

    Uses :func:`_fast_eval_interp1d_lin`.

    Example::

        one_interp = FastLinear1DInterp(0., 1., np.ones(3))
        one_interp(5) # = 1

    Parameters
    ----------
    xp0: float
        Initial x point for interpolation data.
    dxp: float
        Spacing of x points.
    fp: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Function calculated at interpolation points.
    copy: bool, default: False
        Copy input data, specifically fp.
    ep: :external+numpy:py:class:`ndarray <numpy.ndarray>`, optional
        Error on fp points. Not used! Bookkeeping purposes only.
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
        return _fast_eval_interp1d_lin(x, self.xp0, self.dxp, self.fp)

    def reset(self, fp, copy=False, ep=None):
        if copy:
            self.fp = fp.copy()
        else:
            self.fp = fp

        self.ep = ep


class FastCubic1DInterp():
    """ Fast cubic spline for equally spaced data. Out of domain points
    are extrapolated without producing any warnings or errors.

    This class supports natural, not-a-knot boundary conditions. Uses
    :func:`_spline_cubic_natural`, :func:`_spline_cubic_notaknot`, and
    :func:`_fast_eval_interp1d_cubic`.

    Parameters
    ----------
    xp0: float
        Initial x point for interpolation data.
    dxp: float
        Spacing of x points.
    fp: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Function calculated at interpolation points.
    copy: bool, default: False
        Copy input data, specifically fp.
    ep: :external+numpy:py:class:`ndarray <numpy.ndarray>`, optional
        Error on fp points. Not used! Bookkeeping purposes only.
    bc_type: str, default: 'not-a-knot'
        Boundary condition type. Other option is 'natural'. See
        :external+scipy:py:class:`scipy.interpolate.CubicSpline`.
    """

    def __init__(
            self, xp0, dxp, fp, copy=False, ep=None, bc_type='not-a-knot'
    ):
        self.xp0 = float(xp0)
        self.dxp = float(dxp)
        if copy:
            self.fp = fp.copy()
        else:
            self.fp = fp
        self._bc_type = bc_type

        if self._bc_type == 'not-a-knot':
            self._y2p = _spline_cubic_notaknot(fp, dxp)
        elif self._bc_type == 'natural':
            self._y2p = _spline_cubic_natural(fp, dxp)
        else:
            raise Exception("Unknown bc_type in FastCubic1DInterp.")

        self.ep = ep

    def __call__(self, x):
        return _fast_eval_interp1d_cubic(
            x, self.xp0, self.dxp, self.fp, self._y2p)

    def reset(self, fp, copy=False, ep=None):
        if copy:
            self.fp = fp.copy()
        else:
            self.fp = fp

        self.ep = ep

        if self._bc_type == 'not-a-knot':
            self._y2p = _spline_cubic_notaknot(fp, self.dxp)
        elif self._bc_type == 'natural':
            self._y2p = _spline_cubic_natural(fp, self.dxp)


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
    ndata: int or tuple(int)
        Size or shape of the data vector. If tuple, it should be
        ``(nset, size1d)``. For example, 3 quantities share the same
        weights, data vector shape should pass ``ndata=(3, size1d)``.
    nsamples: int
        Number of samples. You can add more measurements then this.
    istart: int, default: 0
        Start index for the subsampling array

    Attributes
    ----------
    _istart: int
        Sampler initial index.
    _isample: int
        Sample counter. Wraps around nsamples
    _is_normalized: bool
        If the weights are normalized. Keeps track if :func:`_normalize` is
        called.
    all_measurements: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        3D array of shape ``(nsamples, nset, ndata)``.
    all_weights: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        3D array of shape ``(nsamples, nset, ndata)``.
    mean: :class:`ndarray <numpy.ndarray>` or None
        Mean. 2D array of shape ``(nset, ndata)``
    variance: :class:`ndarray <numpy.ndarray>` or None
        Variance. 2D array of shape ``(nset, ndata)``
    covariance: list(:class:`ndarray <numpy.ndarray>`) or None
        Covariance. 2D arrays of shape ``(ndata, ndata)`` or 3D arrays of
        shape ``(nblock, blockdim, blockdim)``.
    """

    def __init__(self, ndata, nsamples, istart=0):
        self.nsamples = nsamples
        self._istart = istart % nsamples
        self._isample = self._istart

        if isinstance(ndata, int):
            newshape = (nsamples, 1, ndata)
            self.ndata = ndata
        elif isinstance(ndata, tuple):
            newshape = (nsamples, ndata[0], ndata[1])
            self.ndata = ndata[1]
        else:
            raise QsonicException("ndata must be int or tuple of ints.")

        self.all_measurements = np.zeros(newshape)
        self.all_weights = np.zeros((nsamples, 1, self.ndata))
        self._is_normalized = False

        self.mean = None
        self.covariance = None
        self.variance = None
        self.xdiff = None

    def add_measurement(self, xvec, wvec):
        """ Adds a measurement to the sample.

        You can call this function more then ``nsamples`` times. The provided
        measurement should be weighted, but unnormalized, i.e. ``xvec=wi*xi``.
        After a mean or covariance is obtained, you cannot add
        more measurements.

        Arguments
        ---------
        xvec: :class:`ndarray <numpy.ndarray>`
            Data (measurement) vector.
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

        self.all_measurements[self._isample, :, :] += xvec
        self.all_weights[self._isample, 0, :] += wvec
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
        self.all_measurements = np.divide(
            self.all_measurements, self.all_weights,
            out=self.all_measurements, where=self.all_weights != 0)
        norm = self.all_weights.sum(0)
        self.all_weights = np.divide(
            self.all_weights, norm, out=self.all_weights, where=norm != 0)

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

        self.mean = np.einsum(
            'ijk,ik->jk', self.all_measurements, self.all_weights[:, 0, :])

        return self.mean

    def _get_xdiff(self, mean_xvec, bias_correct=False):
        # remove one measurement, then renormalize
        jack_i = (
            mean_xvec - self.all_measurements * self.all_weights
        ) / (1 - self.all_weights + np.finfo(float).eps)
        mean_jack = np.mean(jack_i, axis=0)

        if bias_correct:
            bias_jack = (self.nsamples - 1) * (mean_jack - mean_xvec)
            mean_xvec -= bias_jack

        xdiff = jack_i - mean_jack

        return mean_xvec, xdiff

    def _get_block_covariance(self, x, blockdim):
        nblock = self.ndata // blockdim
        cov = np.empty((nblock, blockdim, blockdim), dtype=np.float_)
        for kk in range(nblock):
            y = x[:, kk * blockdim:(kk + 1) * blockdim]
            cov[kk] = (y.T @ y) * (self.nsamples - 1) / self.nsamples

        return cov

    def get_mean_n_cov(self, indices=None, blockdim=None, bias_correct=False):
        """ Get the mean and covariance of the mean using delete-one Jackknife.

        Also sets :attr:`mean` and :attr:`covariance`.

        .. warning::

            You cannot call :meth:`add_measurement` after calling this unless
            you :meth:`reset`.

        Arguments
        ---------
        indices: list(int), default: None
            Data set indices to estimate the covariance.
        blockdim: int, default: None
            Calculate covariance by this block size instead of the full space.
        bias_correct: bool, default: False
            Jackknife bias correction term for the mean.

        Returns
        -------
        mean: :class:`ndarray <numpy.ndarray>`
            Mean.
        cov: list(:class:`ndarray <numpy.ndarray>`)
            Covariances of the mean.
        """
        if self.mean is None:
            mean_xvec = self.get_mean()
            self.mean, self.xdiff = self._get_xdiff(mean_xvec, bias_correct)

        if indices is None:
            indices = range(self.all_measurements.shape[1])

        if blockdim is not None:
            assert (self.ndata % blockdim == 0)

        self.covariance = [None] * self.all_measurements.shape[1]
        for jj in indices:
            x = self.xdiff[:, jj, :]

            if blockdim is None:
                cov = (x.T @ x) * (self.nsamples - 1) / self.nsamples
            else:
                cov = self._get_block_covariance(x, blockdim)
            self.covariance[jj] = cov

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
        if self.mean is None:
            mean_xvec = self.get_mean()
            self.mean, self.xdiff = self._get_xdiff(mean_xvec, bias_correct)

        self.variance = (np.einsum('ijk,ijk->jk', self.xdiff, self.xdiff)
                         * (self.nsamples - 1) / self.nsamples)

        return self.mean, self.variance

    def reset(self):
        self._isample = self._istart

        self.all_measurements.fill(0)
        self.all_weights.fill(0)
        self._is_normalized = False

        self.mean = None
        self.covariance = None
        self.variance = None
        self.xdiff = None
