import numpy as np

class Spectrum(object):
    """Represents one spectrum.

    Parameters
    ----------
    z_qso: float
        Quasar redshift.
    targetid: int
        Unique TARGETID identifier.
    arms: list
        List of characters to id spectrograph like 'B', 'R' and 'Z'.
    wave: dict of numpy array
        Dictionary of arrays specifying the wavelength grid.
    flux: dict
        Dictionary of arrays specifying the flux.
    ivar: dict
        Dictionary of arrays specifying the inverse variance.
    mask: dict
        Dictionary of arrays specifying the bitmask.
    reso: dict
        Dictionary of 2D arrays specifying the resolution matrix.

    Methods
    ----------
    removePixels(idx_to_remove)

    """
    def __init__(self, z_qso, targetid, wave, flux, ivar, mask, reso idx):
        self.z_qso = z_qso
        self.targetid = targetid
        self.arms = wave.keys()
        self.wave = wave.copy()
        self.flux = {}
        self.ivar = {}
        self.mask = {}
        self.reso = {}

        for arm in self.arms:
            self.flux[arm] = flux[arm][idx]
            self.ivar[arm] = ivar[arm][idx]
            self.mask[arm] = mask[arm][idx]
            self.reso[arm] = reso[arm][idx]


    # def removePixels(self, idx_to_remove):



