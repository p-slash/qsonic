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
