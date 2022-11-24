import numpy as np
from scipy.optimize import minimize

def _continuum_chi2(x, wave_rf, norm_flux, ivar):
    chi2 = 0

    for arm in wave_rf.keys():
        wv = wave_rf[arm]
        if wv.size == 0:
            continue

        Slope = np.log(wv)
        Slope = (Slope-Slope[0])/(Slope[-1]-Slope[0])

        chi2 += np.sum(
            ivar[arm]*(norm_flux[arm]-(x[0]+x[1]*Slope))**2
        )

    return chi2

class ContinuumFitter(object):
    def __init__(self, w1rf, w2rf, dwrf):
        self.nbins = int((w2rf-w1rf)/dwrf)+1
        self.rfwave = w1rf + np.arange(self.nbins)*dwrf

        self.mean_cont = np.ones(self.nbins)

    def get_normalized_flux(self, spectrum):
        norm_flux = {}

        for arm in spectrum.arms:
            wv = spectrum.forestwave[arm]/(1+spectrum.z_qso)
            cont_est = np.interp(
                wv,
                self.rfwave,
                self.mean_cont
            )
            # Multiply with resolution
            norm_flux[arm] = spectrum.forestflux[arm]/cont_est

        return norm_flux

    def fit_continuum(self, spectrum):
        wave_rf = spectrum.forestwave
        norm_flux = self.get_normalized_flux(spectrum)

        result = minimize(_continuum_chi2, spectrum.cont_params['x'],
            args=(wave_rf, norm_flux, spectrum.forestivar),
            bounds=(np.array([0, -20]), np.array(500, 20)),
            method=None,
            jac=None
        )

        spectrum.cont_params['valid'] = result.success
        if result.success:
            spectrum.cont_params['x'] = result.x


        

