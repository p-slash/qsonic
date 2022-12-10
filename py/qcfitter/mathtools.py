import numpy as np

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

    Methods
    ----------
    __call__(x)

    """
    def __init__(self, xp0, dxp, fp, copy=False):
        self.xp0 = float(xp0)
        self.dxp = float(dxp)
        if copy:
            self.fp = fp.copy()
        else:
            self.fp = fp

    def __call__(self, x):
        xx = (x - self.xp0)/self.dxp
        idx = (np.clip(xx, 0, self.fp.size-2)).astype(int)

        d_idx = xx - idx
        y1, y2 = self.fp[idx], self.fp[idx+1]

        return y1*(1-d_idx) + y2*d_idx

# ===================================================

def get_smooth_ivar(ivar, sigma_pix=20, pad_size=25, esigma=3.5):
    error = np.empty_like(ivar)
    w1 = ivar > 0
    error[w1] = 1/np.sqrt(ivar[w1])
    median_err = np.median(error[w1])
    error[~w1] = median_err

    # Isolate high noise pixels
    mad = np.median(np.abs(error[w1] - median_err))*1.4826
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

