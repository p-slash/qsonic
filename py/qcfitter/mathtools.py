import numpy as np
# from numba import njit

# @njit
def _fast_interp1d(x, xp0, dxp, fp):
    xx = (x - xp0)/dxp
    idx = (np.clip(xx, 0, fp.size-2)).astype(int)
    # idx = xx.astype(np.int32)
    # idx[idx<0]=0
    # idx[idx>fp.size-2]=fp.size-2

    d_idx = xx - idx
    y1, y2 = fp[idx], fp[idx+1]

    return y1*(1-d_idx) + y2*d_idx

class Fast1DInterpolator(object):
    """Fast interpolator for equally spaced data.

    Parameters
    ----------
    xp0: float
        Initial x point for interpolation data.
    dxp0: float
        Spacing of x points.
    fp: numpy array
        Function calculated at interpolation points
    copy: bool (default: False)
        Copy input data, specifically fp

    Attributes
    ----------
    arms: list
        List of characters to id spectrograph like 'B', 'R' and 'Z'. Static variable!
    _f1, _f2: dict of int
        Forest indices. Set up using set_forest method. Then use property functions
        to access forest wave, flux, ivar instead.
    cont_params: dict
        Initial estimates are constructed.

    Methods
    ----------
    removePixels(idx_to_remove)

    """
    def __init__(self, xp0, dxp, fp, copy=False):
        self.xp0 = float(xp0)
        self.dxp = float(dxp)
        if copy:
            self.fp = fp.copy()
        else:
            self.fp = fp

    def __call__(self, x):
        return _fast_interp1d(x, self.xp0, self.dxp, self.fp)

# ===================================================

def get_smooth_ivar(ivar, sigma_pix=20, pad_size=25, esigma=3.5):
    error = np.empty_like(ivar)
    w1 = ivar > 0
    error[w1] = 1/np.sqrt(ivar)
    median_err = np.median(error[w1])
    error[~w1] = median_err

    # Isolate high noise pixels
    mad = np.median(np.abs(error - median_err))*1.4826
    w2 = (error - median_err) > esigma*mad
    err_org = error[w2].copy()

    # Replace them with the median
    error[w2] = median_err

    # Pad the input array to get rid of annoying edge effects
    # Pad values are set to the edge value
    arrsize    = self.size+2*pad_size
    padded_arr = np.pad(error, pad_size, mode='edge')

    kvals     = np.fft.rfftfreq(arrsize)
    smerror_k = np.fft.rfft(padded_arr)*np.exp(-(kvals*sigma_pix)**2/2.)

    error = np.fft.irfft(smerror_k, n=arrsize)[pad_size:-pad_size]

    # Restore values of bad pixels
    error[w2] = err_org
    ivar2 = 1/error**2
    ivar2[~w1] = 0

    return ivar2

